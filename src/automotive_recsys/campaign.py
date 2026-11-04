"""Unattended study that can resume after VM loss using verified Drive state."""
import gc
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import polars as pl

from .audit import write_json
from .campaign_io import CampaignJournal, DurableRun, StopRequested, utcnow
from .drive_store import DriveStore
from .evaluate_run import evaluate_checkpoint, evaluate_rankings, save_evaluation
from .experiment_plan import tuning_jobs, depth_jobs, freeze_choices, final_jobs, plan_hash
from .popularity import rank_popularity
from .train_gpu import run_training


def run_campaign(root, credential_file, root_id, plan, identity="colab", guard=None):
    root = Path(root); data = root / "data"; output = root / "results/experiment"
    store = DriveStore(credential_file, root_id)
    journal = CampaignJournal(output, store, plan, identity, guard)
    records = {}
    journal.log("worker_started", {"worker": identity, "plan_hash": plan_hash(plan)})

    def load_completed(job):
        record = store.get_json(f"runs/{job['run_id']}/completed.json")
        if not record:
            candidate = store.get_json(f"runs/{job['run_id']}/restore.json")
            if candidate and candidate.get("status") == "completed": record = candidate
        if record:
            if record["summary"]["campaign_plan_hash"] != plan_hash(plan): raise ValueError("Completed run belongs to a different plan.")
            return record

    def execute(job, selection=None):
        existing = load_completed(job)
        if existing:
            journal.state["completed_runs"] += 1
            records[job["run_id"]] = existing
            return existing
        run_id = job["run_id"]; run_dir = output / "runs" / run_id
        durable = DurableRun(journal, run_id, run_dir)
        resume = durable.restore()
        journal.log("run_started", {"run_id": run_id, "purpose": job["purpose"]})
        settings = job["settings"]
        result = run_training(data, run_dir, **settings, epochs=job["epochs"], seed=job["seed"],
                              eval_every=plan["eval_every"], patience=job["patience"], resume=resume,
                              event_hook=durable.hook, save_recommendations=job["purpose"].startswith("final"),
                              run_metadata={"run_id": run_id, "purpose": job["purpose"], **plan["metadata"]})
        result["campaign_plan_hash"] = plan_hash(plan); result["purpose"] = job["purpose"]
        if selection:
            journal.progress("test_evaluation", {"run_id": run_id, **settings, "seed": job["seed"], "epoch": result["completed_epochs"], "max_epochs": job["epochs"]}, force=True)
            evaluated = evaluate_checkpoint(data, run_dir / "best.pt", run_dir / "test", selection,
                                             save_predictions=job["purpose"] == "final_main")
            result["test_metrics"] = evaluated["summary"]["metrics"]
            result["test_evaluated"] = True
            del evaluated
        result["best_valid_ndcg_at_10"] = float(result["best_valid_ndcg_at_10"])
        write_json(run_dir / "summary.json", result)
        record = durable.completed(result)
        records[run_id] = record
        journal.state["completed_runs"] += 1
        journal.log("run_completed", {"run_id": run_id, "best_epoch": result["best_epoch"]})
        journal.sync_small()
        # Drive owns the durable best model and outputs; remove local completed
        # full optimizer snapshots to keep Colab disk use bounded.
        for name in ["last.pt", "best.pt"]:
            (run_dir / name).unlink(missing_ok=True)
        gc.collect()
        return record

    def baseline(split):
        marker = store.get_json(f"baseline/{split}/completed.json")
        if marker: return
        journal.progress("popularity_" + split, {}, force=True)
        items = pl.read_parquet(data / "item_map.parquet").sort("iid")
        truth = pl.read_parquet(data / f"{split}.parquet")
        seen = pl.read_parquet(data / f"seen_for_{split}.parquet")
        uid = truth["uid"].unique().sort().to_numpy()
        counts = items["n_train"].to_numpy()
        iid = rank_popularity(counts, uid, seen, min(100, items.height))
        result = evaluate_rankings(data, split, uid, iid, counts[iid].astype(np.float32), {"model": "popularity", "seed": None})
        folder = output / "baseline" / split
        save_evaluation(result, folder)
        for file in folder.iterdir(): store.put(file, f"baseline/{split}/{file.name}")
        store.put_json({"complete": True, "summary": result["summary"]}, f"baseline/{split}/completed.json", journal.staging)

    try:
        baseline("valid")
        tuning = tuning_jobs(plan)
        for job in tuning: execute(job)
        base_choices, selected_records = {}, {}
        for model in ["bpr_mf", "lightgcn", "xsimgcl"]:
            eligible = [job for job in tuning if job["settings"]["model"] == model]
            chosen = max(eligible, key=lambda job: records[job["run_id"]]["summary"]["best_valid_ndcg_at_10"])
            base_choices[model] = chosen["settings"]
            selected_records[model] = chosen["run_id"]
        extra = depth_jobs(plan, base_choices["lightgcn"])
        for job in extra: execute(job)
        light_records = [(selected_records["lightgcn"], base_choices["lightgcn"])] + [(job["run_id"], job["settings"]) for job in extra]
        light_id, light = max(light_records, key=lambda item: records[item[0]]["summary"]["best_valid_ndcg_at_10"])
        selected_records["lightgcn_depth"] = light_id
        selection = freeze_choices(plan, base_choices, light, selected_records)
        selection["frozen_at"] = utcnow()
        existing = store.get_json("frozen_selection.json")
        if existing:
            # frozen_at is provenance, not a source of selection nondeterminism.
            for key in ["choices", "selection_records", "plan_hash"]:
                if existing[key] != selection[key]: raise ValueError("A resumed study would change an already frozen selection.")
            selection = existing
        else: store.put_json(selection, "frozen_selection.json", journal.staging)
        write_json(output / "frozen_selection.json", selection)
        baseline("test")
        for job in final_jobs(plan, selection): execute(job, selection)
        journal.progress("reporting", {}, force=True)
        from .experiment_report import build_report
        build_report(output, store, plan, records, progress=lambda: journal.progress("reporting"))
        journal.state["status"] = "completed"
        journal.log("campaign_completed")
        journal.progress("completed", {}, force=True); journal.sync_small()
        return {"status": "completed", "completed_runs": len(records)}
    except StopRequested:
        journal.state["status"] = "paused"
        journal.log("campaign_paused")
        journal.progress("paused", force=True, check_control=False)
        journal.sync_small()
        return {"status": "paused"}
    except Exception as error:
        journal.state["status"] = "failed"
        journal.log("worker_failed", {"type": type(error).__name__, "stage": journal.state.get("stage"),
                    "locations": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                  for frame in traceback.extract_tb(error.__traceback__)[-6:]]})
        journal.progress("failed", force=True, check_control=False)
        journal.sync_small()
        raise
