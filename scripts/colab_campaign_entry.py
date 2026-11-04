"""Remote kernel entry: verified Drive inputs, unattended study, own VM release."""
import hashlib
import json
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
    if usage.paid_compute_units_balance <= CONFIG["plan"]["minimum_balance"] or CONFIG["initial_balance_cu"] - usage.paid_compute_units_balance >= CONFIG["plan"]["cu_budget"]:
        journal.state["pause_reason"] = "compute_unit_budget"
        raise StopRequested("Compute budget reached; verified checkpoints retained.")


try:
    store.put_json({"worker": CONFIG["worker"], "started_at": time.time()}, f"workers/{CONFIG['worker']}/started.json", root / "transfer")
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
    from automotive_recsys.campaign import run_campaign
    result = run_campaign(root, DRIVE_CREDENTIAL, CONFIG["root_id"], CONFIG["plan"], CONFIG["worker"], guard=budget_guard)
    status = result["status"]
    print(json.dumps(result), flush=True)
except Exception as error:
    # Avoid printing HTTP exception URLs or credential-related diagnostics.
    print("Worker ended with " + type(error).__name__ + "; inspect private progress artifacts.", flush=True)
    try:
        progress = store.get_json("progress.json", {})
        progress.update(status="failed", worker=CONFIG["worker"], error_type=type(error).__name__)
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
