"""Tiny real Drive protocol check; no local model training or user data scan."""
import json
from pathlib import Path

from experiment import DRIVE_CREDENTIAL, load_config
from automotive_recsys.campaign_io import DurableRun
from automotive_recsys.drive_store import DriveStore


def main():
    config = load_config(); local = Path(config["local_dir"]) / "storage_preflight"
    local.mkdir(exist_ok=True)
    store = DriveStore(DRIVE_CREDENTIAL, config["root_id"])
    class Journal:
        staging = local / "staging"
        state = {"current": {}}
        def progress(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def sync_small(self): pass
    journal = Journal(); journal.store = store
    source = local / "source"; source.mkdir(exist_ok=True)
    run = DurableRun(journal, "storage_protocol_preflight", source)
    (source / "best.pt").write_bytes(b"synthetic storage fixture; no trained model")
    for epoch in [1, 2, 3]:
        (source / "last.pt").write_bytes(f"synthetic-state-{epoch}".encode())
        (source / "history.json").write_text(json.dumps({"epoch": epoch}))
        run.sync_checkpoint(epoch, 1, force=True)
    recovered = local / "recovered"
    pointer = DurableRun(journal, "storage_protocol_preflight", recovered)
    checkpoint = pointer.restore()
    assert checkpoint.read_bytes() == b"synthetic-state-3"
    assert json.loads((recovered / "history.json").read_text())["epoch"] == 3
    assert [entry["epoch"] for entry in pointer.pointer["generations"]] == [2,3]
    result = {"passed": True, "upload_md5_verified": True, "download_sha256_verified": True,
              "generations_retained": [2,3], "restored_to_separate_directory": True}
    store.put_json(result, "preflight/storage_protocol.json", journal.staging)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
