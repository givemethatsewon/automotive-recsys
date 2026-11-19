"""Resumable evaluation stages using the campaign's verified Drive I/O.

Reuse CampaignJournal and DurableRun's content-addressed output uploader.
No optimizer checkpoints are invented for a no-training experiment.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import shutil
import tempfile

from .campaign_io import CampaignJournal, DurableRun


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


class DurableStages:
    def __init__(
        self,
        store,
        root,
        run_id,
        output,
        context,
        stage_names,
        run_index=0,
        planned_runs=3,
        heartbeat_seconds=30,
    ):
        self.store, self.output = store, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.context = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()
        self.stages = list(stage_names)
        self.run_index = run_index
        self.planned_runs = planned_runs
        previous = store.get_json("progress.json", {})
        self.journal = CampaignJournal(
            root,
            store,
            {"campaign_id": "vehicle-v2-durable", "planned_run_count": planned_runs},
            run_id,
        )
        self.journal.state["completed_runs"] = int(previous.get("completed_runs", 0))
        self.journal.state["durable_position"] = previous.get(
            "durable_position", [0, 0, 0]
        )
        self.run = DurableRun(self.journal, run_id, self.output)
        self.pointer = self.run.pointer
        if self.pointer and self.pointer.get("context_sha256") != self.context:
            raise ValueError("Durable stage context/source/input hash differs")
        self.done = set(self.pointer.get("completed_stages", []))
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.error = None
        self.interval = heartbeat_seconds
        self.thread = None
        self.last_save = time.monotonic()

    def restore(self):
        if not self.pointer:
            return False
        temporary = Path(
            tempfile.mkdtemp(prefix="stage-restore-", dir=self.output.parent)
        )
        try:
            for artifact in self.pointer["artifacts"]:
                path = (temporary / artifact["relative"]).resolve()
                if not path.is_relative_to(temporary.resolve()):
                    raise ValueError("Unsafe restore path")
                self.store.download(artifact["id"], path, artifact["sha256"])
            # Publish exactly the committed generation; uncommitted local files
            # must never be mistaken for completed validation/test stages.
            shutil.rmtree(self.output)
            temporary.replace(self.output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        if self.pointer.get("status") == "completed":
            # The restore pointer may have committed just before publication of
            # completed.json was interrupted. Repair that idempotent marker.
            self.store.put_json(
                self.pointer, self.run.prefix + "/completed.json", self.journal.staging
            )
            self.journal.state["completed_runs"] = max(
                self.journal.state["completed_runs"], self.run_index + 1
            )
        self.journal.log(
            "stage_restored",
            {
                "run_id": self.run.run_id,
                "completed_stages": sorted(self.done),
                "position": self.pointer["position"],
            },
        )
        return True

    def start(self):
        self.progress("starting", force=True)

        def loop():
            while not self.stop.wait(self.interval):
                try:
                    with self.lock:
                        self.journal.log(
                            "worker_heartbeat",
                            {
                                "run_id": self.run.run_id,
                                "current": self.journal.state["current"],
                            },
                        )
                        self.journal.progress(force=True)
                        self.journal.sync_small()
                except Exception as e:
                    self.error = e
                    return

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def progress(self, stage, step=None, steps=None, force=False):
        if self.error:
            raise RuntimeError(
                "Drive heartbeat failed; refusing unprotected evaluation"
            ) from self.error
        with self.lock:
            value = {
                "run_id": self.run.run_id,
                "model": self.run.run_id,
                "seed": 42,
                "stage": stage,
            }
            if step is not None:
                value["step"] = int(step)
            if steps is not None:
                value["steps"] = int(steps)
            self.journal.progress(stage, value, force=force)

    def commit(self, stage, cursor=0, done=False, complete=False):
        if stage not in self.stages:
            raise ValueError("Undeclared stage")
        with self.lock:
            self.progress("saving_" + stage, step=cursor)
            artifacts = self.run.upload_outputs(immutable=True)
            completed = self.done | ({stage} if done else set())
            position = [self.run_index, self.stages.index(stage), int(cursor)]
            generation = {
                "stage": stage,
                "cursor": int(cursor),
                "position": position,
                "artifacts": artifacts,
            }
            kept = (self.pointer.get("generations", []) + [generation])[-2:]
            pointer = {
                "status": "completed" if complete else "running",
                "context_sha256": self.context,
                "stage": stage,
                "cursor": int(cursor),
                "run_id": self.run.run_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed_stages": sorted(completed),
                "position": position,
                "generations": kept,
                "artifacts": artifacts,
            }
            # Commit pointer only after the existing uploader verified all blobs.
            self.store.put_json(
                pointer, self.run.prefix + "/restore.json", self.journal.staging
            )
            if complete:
                self.store.put_json(
                    pointer, self.run.prefix + "/completed.json", self.journal.staging
                )
            self.pointer, self.done = pointer, completed
            self.journal.state["durable_position"] = position
            self.journal.state["durable_checkpoint"] = (
                f"{self.run.run_id} stage {stage} step {cursor}"
            )
            if complete:
                self.journal.state["completed_runs"] = max(
                    self.journal.state["completed_runs"], self.run_index + 1
                )
            self.journal.log(
                "stage_verified_on_drive",
                {
                    "run_id": self.run.run_id,
                    "stage": stage,
                    "step": cursor,
                    "complete": complete,
                },
            )
            self.journal.progress(stage, force=True)
            self.journal.sync_small()
            self.last_save = time.monotonic()
        return pointer

    def close(self, error=None):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            self.journal.state["status"] = (
                "failed"
                if error
                else (
                    "completed"
                    if self.journal.state["completed_runs"] >= self.planned_runs
                    else "running"
                )
            )
            if error:
                self.journal.log(
                    "worker_failed", {"run_id": self.run.run_id, "error": str(error)}
                )
            self.journal.progress(force=True, check_control=False)
            self.journal.sync_small()
