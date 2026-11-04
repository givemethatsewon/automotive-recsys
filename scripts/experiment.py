"""Prepare, inspect, or pause a durable Colab experiment (no local training)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from automotive_recsys.drive_store import DriveStore
from automotive_recsys.experiment_plan import default_plan

PRIVATE = Path.home() / ".config/automotive-recsys"
DRIVE_CREDENTIAL = PRIVATE / "drive/gcloud/application_default_credentials.json"
CAMPAIGNS = ROOT / "results/campaigns"


def load_config(path=None):
    return json.loads(Path(path or CAMPAIGNS / "current.json").read_text())


def prepare():
    if not DRIVE_CREDENTIAL.exists():
        raise SystemExit("Drive authentication required: .venv/bin/python scripts/colab_login.py --drive")
    os.chmod(DRIVE_CREDENTIAL, 0o600)
    store = DriveStore(DRIVE_CREDENTIAL)
    quota = store.quota()["storageQuota"]
    if quota.get("limit") and int(quota["limit"]) - int(quota.get("usage", 0)) < 12 * 2**30:
        raise SystemExit("Drive needs at least 12 GiB free before starting this checkpoint retention plan.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_id = "automotive-" + stamp
    folder = store.create_root(campaign_id)
    destination = CAMPAIGNS / campaign_id; destination.mkdir(parents=True)
    staging = destination / "staging"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    from colab_bundle import build_bundle
    bundle, manifest = build_bundle()
    source_hash = hashlib.sha256(json.dumps({k:v for k,v in manifest["files"].items() if k.startswith("src/")}, sort_keys=True).encode()).hexdigest()
    plan = default_plan(campaign_id, {"git_revision": revision, "working_diff_sha256": hashlib.sha256(diff).hexdigest(), "source_manifest_sha256": source_hash})
    store.put_json(plan, "experiment_plan.json", staging)
    uploaded = []
    for index, part in enumerate(manifest["parts"], 1):
        print(f"Drive input upload {index}/{len(manifest['parts'])}", flush=True)
        item = store.put(bundle / part["name"], "inputs/" + part["name"])
        uploaded.append({"id": item["id"], **part})
    manifest_item = store.put(bundle / "bundle-manifest.json", "inputs/bundle-manifest.json")
    config = {"campaign_id": campaign_id, "root_id": folder["id"], "drive_url": folder["webViewLink"],
              "manifest": {"id": manifest_item["id"], "sha256": manifest_item["sha256"]},
              "parts": uploaded, "plan": plan, "local_dir": str(destination), "gpu": "L4"}
    for path in [destination / "config.json", CAMPAIGNS / "current.json"]:
        path.write_text(json.dumps(config, indent=2) + "\n")
    store.put_json({"stop_requested": False}, "control.json", staging)
    store.put_json({"campaign_id": campaign_id, "status": "prepared", "completed_runs": 0,
                    "planned_runs": plan["planned_run_count"], "updated_at": datetime.now(timezone.utc).isoformat()}, "progress.json", staging)
    print(json.dumps({"campaign_id": campaign_id, "drive_url": folder["webViewLink"], "config": str(destination / "config.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "status", "pause", "resume"])
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.action == "prepare": return prepare()
    config = load_config(args.config); store = DriveStore(DRIVE_CREDENTIAL, config["root_id"])
    if args.action in {"pause", "resume"}:
        store.put_json({"stop_requested": args.action == "pause"}, "control.json", Path(config["local_dir"]) / "staging")
        if args.action == "resume":
            state = store.get_json("progress.json", {})
            if state.get("status") == "completed": raise SystemExit("The campaign is already completed.")
            state.update(status="resuming", stage="waiting_for_controller")
            store.put_json(state, "progress.json", Path(config["local_dir"]) / "staging")
        print("Pause requested; the last verified checkpoint is retained." if args.action == "pause" else "Pause cleared. Start the controller again to resume.")
    state = store.get_json("progress.json", {})
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print(config["drive_url"])


if __name__ == "__main__": main()
