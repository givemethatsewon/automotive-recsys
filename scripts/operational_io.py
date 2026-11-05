"""Versioned I/O policy adapter; frozen training math and RNG remain intact."""
import json
from pathlib import Path
import time


def install(checkpoint_seconds=600):
    import automotive_recsys.drive_store as storage
    import automotive_recsys.campaign_io as journal
    from drive_store import DriveStore
    storage.DriveStore = DriveStore
    original = journal.DurableRun
    metadata = {"checkpoint_seconds": checkpoint_seconds, "drive_chunk_bytes": 64 << 20,
                "forced_saves": "first/best/final unchanged", "model_math_changed": False}

    class DurableRun(original):
        def sync_checkpoint(self, epoch, best_epoch, force=False):
            if not force and time.monotonic() - self.last_sync < checkpoint_seconds: return
            return super().sync_checkpoint(epoch, best_epoch, force=force)

        def hook(self, event, value):
            if event == "run_started":
                path = Path(self.output) / "manifest.json"
                if path.exists():
                    document = json.loads(path.read_text())
                    document["operational_io"] = metadata
                    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
            return super().hook(event, value)

    journal.DurableRun = DurableRun
    return metadata
