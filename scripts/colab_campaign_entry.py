"""Remote kernel entry: verified Drive inputs, unattended study, own VM release."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

PRIVATE = Path("/content/.automotive-private")
CONFIG = json.loads((PRIVATE / "worker.json").read_text())
DRIVE_CREDENTIAL = PRIVATE / "drive.json"
COLAB_CREDENTIAL = PRIVATE / "colab.json"
for credential in [DRIVE_CREDENTIAL, COLAB_CREDENTIAL]: os.chmod(credential, 0o600)
sys.path.insert(0, "/content/automotive-bootstrap")
from drive_store import DriveStore, checksums
from colab_cli.client import Client, Prod
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

store = DriveStore(DRIVE_CREDENTIAL, CONFIG["root_id"])
client = Client(Prod(), AuthorizedSession(Credentials.from_authorized_user_file(str(COLAB_CREDENTIAL))))
root = Path("/content/automotive-recsys")
root.mkdir(exist_ok=True)
status = "failed"
cost_rows = []
last_cost = 0


def budget_guard(journal):
    global last_cost
    if time.monotonic() - last_cost < 300: return
    from automotive_recsys.campaign_io import StopRequested, utcnow
    usage = client.get_consumption_user_info()
    row = {"time": utcnow(), "balance_cu": usage.paid_compute_units_balance,
           "account_rate_cu_hour": usage.consumption_rate_hourly, "worker": CONFIG["worker"]}
    cost_rows.append(row)
    store.put_json(cost_rows, f"workers/{CONFIG['worker']}/cost_observations.json", root / "transfer")
    journal.state["account_balance_cu"] = usage.paid_compute_units_balance
    journal.state["account_cu_since_start"] = CONFIG["initial_balance_cu"] - usage.paid_compute_units_balance
    last_cost = time.monotonic()
    control = store.get_json("control.json", {})
    operational_budget = float(control.get("cu_budget", CONFIG["plan"]["cu_budget"]))
    if not math.isfinite(operational_budget) or operational_budget <= 0: raise ValueError("Invalid operational CU budget")
    if usage.paid_compute_units_balance <= CONFIG["plan"]["minimum_balance"] or CONFIG["initial_balance_cu"] - usage.paid_compute_units_balance >= operational_budget:
        journal.state["pause_reason"] = "compute_unit_budget"
        raise StopRequested("Compute budget reached; verified checkpoints retained.")


try:
    environment_manifest=Path("/content/automotive-bootstrap/kernel_environment.json")
    if environment_manifest.exists(): store.put(environment_manifest,f"workers/{CONFIG['worker']}/kernel_environment.json")
    store.put_json({"worker": CONFIG["worker"], "started_at": time.time(), "controller_revision": CONFIG.get("controller_revision"),
                    "entry_sha256": CONFIG.get("entry_sha256")}, f"workers/{CONFIG['worker']}/started.json", root / "transfer")
    upload = Path("/content/automotive-input"); upload.mkdir(exist_ok=True)
    manifest_path = store.download(CONFIG["manifest"]["id"], upload / "manifest.json", CONFIG["manifest"]["sha256"])
    manifest = json.loads(manifest_path.read_text())
    archive = upload / "input.tar"
    with archive.open("wb") as merged:
        for index, part in enumerate(CONFIG["parts"], 1):
            source = store.download(part["id"], upload / part["name"], part["sha256"])
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""): merged.write(block)
            source.unlink()
            print(f"Verified input part {index}/{len(CONFIG['parts'])}", flush=True)
    assert checksums(archive)["sha256"] == manifest["archive_sha256"]
    with tarfile.open(archive) as bundle: bundle.extractall(root, filter="data")
    archive.unlink()
    for name, expected in manifest["files"].items():
        assert checksums(root / name)["sha256"] == expected, "Input SHA mismatch"
    sys.path.insert(0, str(root / "src"))
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    assert checksums(Path("/content/automotive-bootstrap/operational_io.py"))["sha256"] == CONFIG["operational_io_sha256"]
    from operational_io import install
    operational_io = install()
    store.put_json(operational_io | {"source_sha256": CONFIG["operational_io_sha256"]}, f"workers/{CONFIG['worker']}/operational_io.json", root / "transfer")
    if CONFIG['plan'].get('kind')=='vehicle_weight_training':
        from automotive_recsys.campaign_io import CampaignJournal
        from automotive_recsys.vehicle_training import prepare_weights,verify_cuda_weighted_objective
        feature_journal=CampaignJournal(root/'results/feature_journal',store,CONFIG['plan'],CONFIG['worker'],budget_guard)
        audit=prepare_weights(root,lambda step,total:feature_journal.progress('preparing_vehicle_weights',{'run_id':'feature_preparation','model':'bpr_mf','seed':42,'step':step,'steps':total}))
        expected_weights=CONFIG['plan']['metadata'].get('expected_weights_sha256')
        if expected_weights:assert audit['weights_sha256']==expected_weights,'Vehicle treatment differs from frozen BPR lineage'
        store.put_json(audit,'vehicle_weight_audit.json',root/'transfer')
        weights=store.put(root/'features/train_vehicle_weights.npy','features/train_vehicle_weights.npy')
        roundtrip=store.download(weights['id'],root/'features/roundtrip.npy',weights['sha256']);roundtrip.unlink()
        if CONFIG['plan'].get('preflights'):
            from automotive_recsys.vehicle_training import run_weighted_preflights
            run_weighted_preflights(root,store,CONFIG['plan'],CONFIG['worker'],budget_guard)
        else:
            kernel_check=verify_cuda_weighted_objective(root)
            store.put_json(kernel_check | {'weights_sha256':weights['sha256'],'external_weights_roundtrip':True},'preflight/weighted_objective.json',root/'transfer')
    # Before the study, prove that a real optimizer/RNG checkpoint survives a
    # Drive round trip into a separate output directory and continues training.
    if not CONFIG['plan'].get('preflights') and not store.get_json("preflight/verified.json"):
        from automotive_recsys.campaign_io import CampaignJournal, DurableRun
        from automotive_recsys.train_gpu import run_training
        journal = CampaignJournal(root / "results/preflight_journal", store, CONFIG["plan"], CONFIG["worker"], budget_guard)
        run_id = "preflight_bpr_mf_seed42"
        preflight_settings=dict(CONFIG['plan'].get('preflight_settings',{}))
        preflight_model=preflight_settings.pop('model','bpr_mf')
        if preflight_settings.get('interaction_weights'):
            preflight_settings['interaction_weights']=root/preflight_settings['interaction_weights']
        first = root / "results/preflight_source"
        durable = DurableRun(journal, run_id, first)
        previous = durable.restore()
        if previous is None:
            run_training(root / "data", first, preflight_model, **preflight_settings, epochs=1, eval_every=1,
                         verify_resume=True, save_recommendations=False, event_hook=durable.hook,
                         run_metadata={"purpose": "external checkpoint recovery preflight"})
        recovered = root / "results/preflight_recovered"
        durable = DurableRun(journal, run_id, recovered)
        checkpoint = durable.restore()
        assert checkpoint is not None, "No committed external checkpoint"
        verified = run_training(root / "data", recovered, preflight_model, **preflight_settings, epochs=2, eval_every=1,
                                resume=checkpoint, verify_resume=True, save_recommendations=False,
                                event_hook=durable.hook, run_metadata={"purpose": "external checkpoint recovery preflight"})
        assert verified["completed_epochs"] == 2 and verified["resume_check"]["passed"]
        durable.completed(verified)
        from automotive_recsys.experiment_plan import plan_hash
        store.put_json({"passed": True, "plan_hash": plan_hash(CONFIG['plan']),
                        "external_restore_then_epoch": 2, "optimizer_rng_replay": verified["resume_check"],
                        "training_settings":verified['settings'],
                        "worker": CONFIG["worker"]}, "preflight/verified.json", root / "transfer")
        for folder in [first, recovered]:
            for name in ["last.pt", "best.pt"]: (folder / name).unlink(missing_ok=True)
        print("Verified external Drive restore and epoch-2 continuation.", flush=True)
    if CONFIG['plan'].get('kind')=='vehicle_weight_training':
        from automotive_recsys.experiment_plan import plan_hash
        receipt=store.get_json('preflight/verified.json')
        assert receipt['passed'] and receipt['plan_hash']==plan_hash(CONFIG['plan'])
        if CONFIG['plan'].get('preflights'):
            assert set(receipt['models'])=={p['model'] for p in CONFIG['plan']['preflights']}
            for result in receipt['models'].values():
                assert result['passed'] and result['training_settings']['interaction_weighting']['weights_sha256']==audit['weights_sha256']
        else:
            assert receipt['training_settings']['interaction_weighting']['weights_sha256']==audit['weights_sha256']
    from automotive_recsys.campaign import run_campaign
    result = run_campaign(root, DRIVE_CREDENTIAL, CONFIG["root_id"], CONFIG["plan"], CONFIG["worker"], guard=budget_guard)
    status = result["status"]
    print(json.dumps(result), flush=True)
except Exception as error:
    # Avoid printing HTTP exception URLs or credential-related diagnostics.
    print("Worker ended with " + type(error).__name__ + "; inspect private progress artifacts.", flush=True)
    import traceback
    print(json.dumps({'failure_locations':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(error.__traceback__)[-8:]]}),flush=True)
    try:
        from runtime_policy import transient_network_error
        retryable = transient_network_error(error)
    except ImportError: retryable = False
    try:
        store = DriveStore(DRIVE_CREDENTIAL, CONFIG["root_id"])
        progress = store.get_json("progress.json", {})
        progress.update(status="failed", worker=CONFIG["worker"], error_type=type(error).__name__, retryable_failure=retryable)
        store.put_json(progress, "progress.json", root / "transfer")
    except Exception: pass
finally:
    try:
        usage = client.get_consumption_user_info()
        cost_rows.append({"time": time.time(), "balance_cu": usage.paid_compute_units_balance,
                          "account_rate_cu_hour": usage.consumption_rate_hourly, "worker": CONFIG["worker"], "status": status})
        store.put_json(cost_rows, f"workers/{CONFIG['worker']}/cost_observations.json", root / "transfer")
        store.put_json({"status": status, "worker": CONFIG["worker"], "release_requested": True}, f"workers/{CONFIG['worker']}/exit.json", root / "transfer")
    finally:
        DRIVE_CREDENTIAL.unlink(missing_ok=True)
        COLAB_CREDENTIAL.unlink(missing_ok=True)
        # This endpoint came from the controller's own newly allocated session.
        client.unassign(CONFIG["endpoint"])
