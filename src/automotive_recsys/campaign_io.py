"""Human-readable progress and verified, generation-based Drive checkpoints."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .audit import write_json


def utcnow(): return datetime.now(timezone.utc).isoformat()


class StopRequested(RuntimeError): pass


class CampaignJournal:
    def __init__(self, root, store, plan, identity, guard=None):
        self.root, self.store, self.plan = Path(root), store, plan
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = self.root / "transfer_staging"
        self.staging.mkdir(exist_ok=True)
        previous = store.get_json("progress.json", {})
        self.state = {"campaign_id": plan["campaign_id"], "status": "running", "started_at": utcnow(),
                      "worker": identity, "completed_runs": 0, "planned_runs": plan["planned_run_count"], "current": {}}
        if previous.get("started_at"): self.state["started_at"] = previous["started_at"]
        existing_events = store.find("events.jsonl")
        if existing_events:
            store.download(existing_events["id"], self.root / "events.jsonl", existing_events.get("appProperties", {}).get("sha256"))
        self.last_push, self.last_control = 0, 0
        self.guard = guard

    def log(self, event, value=None):
        record = {"time": utcnow(), "event": event, **(value or {})}
        with (self.root / "events.jsonl").open("a") as stream: stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def progress(self, stage=None, current=None, force=False, check_control=True):
        if stage: self.state["stage"] = stage
        if current is not None:
            if not current or (current.get("run_id") and current["run_id"] != self.state["current"].get("run_id")):
                self.state["current"] = current
            else: self.state["current"] = self.state["current"] | current
        self.state["updated_at"] = utcnow()
        write_json(self.root / "progress.json", self.state)
        c = self.state["current"]
        text = [f"캠페인: {self.state['campaign_id']}", f"상태: {self.state['status']} / 단계: {self.state.get('stage','preparing')}",
                f"완료 작업: {self.state['completed_runs']} / {self.state['planned_runs']}",
                f"현재: {c.get('run_id','-')} | {c.get('model','-')} | seed {c.get('seed','-')}",
                f"epoch: {c.get('epoch',0)} / {c.get('max_epochs','-')} | step: {c.get('step','-')} / {c.get('steps','-')}",
                f"best valid NDCG@10: {c.get('best_valid_ndcg','-')}", f"최근 Drive checkpoint: {self.state.get('durable_checkpoint','아직 없음')}",
                f"갱신: {self.state['updated_at']}", "작업 완료 수는 실제 완료 수이며 epoch 상한/ETA는 학습 종료 보장이 아닙니다."]
        (self.root / "progress.txt").write_text("\n".join(text) + "\n")
        if force or time.monotonic() - self.last_push >= 30:
            self.store.put(self.root / "progress.json", "progress.json")
            self.store.put(self.root / "progress.txt", "progress.txt")
            self.last_push = time.monotonic()
        if check_control and time.monotonic() - self.last_control >= 60:
            if self.guard: self.guard(self)
            control = self.store.get_json("control.json", {})
            self.last_control = time.monotonic()
            if control.get("stop_requested"): raise StopRequested("Stop requested via Drive control.json")

    def sync_small(self):
        for name in ["events.jsonl", "progress.json", "progress.txt"]:
            if (self.root / name).exists(): self.store.put(self.root / name, name)
        if (self.root / "events.jsonl").exists():
            self.store.put(self.root / "events.jsonl", f"workers/{self.state['worker']}/events.jsonl")


class DurableRun:
    def __init__(self, journal, run_id, output):
        self.journal, self.store, self.run_id, self.output = journal, journal.store, run_id, Path(output)
        self.prefix = f"runs/{run_id}"
        self.last_sync = 0
        self.pointer = self.store.get_json(self.prefix + "/restore.json", {})

    def restore(self):
        if not self.pointer or self.pointer.get("status") == "completed": return None
        for item in self.pointer.get("artifacts", []):
            destination = (self.output / item["relative"]).resolve()
            if not destination.is_relative_to(self.output.resolve()): raise ValueError("Invalid artifact path in restore manifest.")
            self.store.download(item["id"], destination, item["sha256"])
        latest = self.pointer["generations"][-1]
        self.store.download(latest["last"]["id"], self.output / "last.pt", latest["last"]["sha256"])
        self.store.download(latest["best"]["id"], self.output / "best.pt", latest["best"]["sha256"])
        self.journal.log("checkpoint_restored", {"run_id": self.run_id, "epoch": latest["epoch"]})
        return self.output / "last.pt"

    def upload_outputs(self, immutable=False):
        from .drive_store import checksums
        artifacts = []
        for path in sorted(self.output.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".parquet", ".npz"}:
                relative = path.relative_to(self.output).as_posix()
                destination = ("snapshots/" + checksums(path)["sha256"] + "/" if immutable else "") + relative
                metadata = self.store.put(path, self.prefix + "/" + destination)
                artifacts.append({"relative": relative, "id": metadata["id"], "sha256": metadata["sha256"]})
        return artifacts

    def sync_checkpoint(self, epoch, best_epoch, force=False):
        from .drive_store import checksums
        if not force and time.monotonic() - self.last_sync < 120: return
        if not (self.output / "best.pt").exists(): return
        self.journal.progress("saving_checkpoint", {**self.journal.state["current"], "epoch": epoch}, force=True)
        best_hash = checksums(self.output / "best.pt")["sha256"]
        last_hash = checksums(self.output / "last.pt")["sha256"]
        best = self.store.put(self.output / "best.pt", self.prefix + f"/checkpoints/best-e{best_epoch:04d}-{best_hash}.pt")
        last = self.store.put(self.output / "last.pt", self.prefix + f"/checkpoints/state-e{epoch:04d}-{last_hash}.pt")
        generations = [g for g in self.pointer.get("generations", []) if g["epoch"] != epoch]
        artifacts = self.upload_outputs(immutable=True)
        generations.append({"epoch": epoch, "last": {k:last[k] for k in ["id","sha256"]}, "best": {k:best[k] for k in ["id","sha256"]}, "artifacts": artifacts})
        old, kept = generations[:-2], generations[-2:]
        pointer = {"status": "running", "updated_at": utcnow(), "run_id": self.run_id, "generations": kept, "artifacts": artifacts}
        self.store.put_json(pointer, self.prefix + "/restore.json", self.journal.staging)
        self.pointer = pointer
        retained = {part["id"] for g in kept for part in [g["last"], g["best"]]}
        for generation in old:
            for part in [generation["last"], generation["best"]]:
                if part["id"] not in retained:
                    self.store.remove_verified_checkpoint(part["id"], part["sha256"])
                    retained.add(part["id"])
        retained_artifacts = {part["id"] for g in kept for part in g.get("artifacts", [])}
        for generation in old:
            for part in generation.get("artifacts", []):
                if part["id"] not in retained_artifacts:
                    self.store.remove_verified_artifact(part["id"], part["sha256"])
                    retained_artifacts.add(part["id"])
        self.last_sync = time.monotonic()
        self.journal.state["durable_checkpoint"] = f"{self.run_id} epoch {epoch} ({utcnow()})"
        self.journal.log("checkpoint_verified_on_drive", {"run_id": self.run_id, "epoch": epoch})
        self.journal.progress(force=True)
        self.journal.sync_small()

    def hook(self, event, value):
        current = value | {"run_id": self.run_id}
        self.journal.progress(event, current)
        if event == "epoch_finished":
            self.journal.log(event, current)
            self.sync_checkpoint(value["epoch"], value["best_epoch"], force=bool(value["epoch_metrics"].get("is_best")))
        elif event == "run_finished": self.sync_checkpoint(value["epoch"], value["best_epoch"], force=True)

    def completed(self, result):
        from .drive_store import checksums
        best_epoch = result["best_epoch"]
        best_hash = checksums(self.output / "best.pt")["sha256"]
        best = self.store.put(self.output / "best.pt", self.prefix + f"/checkpoints/best-e{best_epoch:04d}-{best_hash}.pt")
        record = {"status": "completed", "updated_at": utcnow(), "run_id": self.run_id,
                  "best": {k:best[k] for k in ["id","sha256"]}, "artifacts": self.upload_outputs(), "summary": result}
        self.store.put_json(record, self.prefix + "/restore.json", self.journal.staging)
        self.store.put_json(record, self.prefix + "/completed.json", self.journal.staging)
        seen = {best["id"]}
        for generation in self.pointer.get("generations", []):
            for part in [generation["last"], generation["best"]]:
                if part["id"] not in seen:
                    self.store.remove_verified_checkpoint(part["id"], part["sha256"]); seen.add(part["id"])
            for part in generation.get("artifacts", []):
                if part["id"] not in seen:
                    self.store.remove_verified_artifact(part["id"], part["sha256"]); seen.add(part["id"])
        self.pointer = record
        return record
