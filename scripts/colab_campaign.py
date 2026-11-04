"""Small detached controller: provision, resume VM loss, collect reports, release GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment import ROOT, PRIVATE, DRIVE_CREDENTIAL, load_config
from automotive_recsys.drive_store import DriveStore
from colab_cli.client import Client, Prod
from colab_cli.state import StateStore
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

COLAB_CREDENTIAL = PRIVATE / "colab/gcloud/application_default_credentials.json"
STATE_FILE = PRIVATE / "colab/sessions.json"


def control(config):
    directory = Path(config["local_dir"])
    lock = (directory / "controller.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("A controller already owns this campaign.")
    (directory / "controller.pid").write_text(str(os.getpid()) + "\n")
    store = DriveStore(DRIVE_CREDENTIAL, config["root_id"])
    client = Client(Prod(), AuthorizedSession(Credentials.from_authorized_user_file(str(COLAB_CREDENTIAL))))
    states = StateStore(str(STATE_FILE))
    worker_path = directory / "controller_state.json"
    state = json.loads(worker_path.read_text()) if worker_path.exists() else {"workers": []}
    if "initial_balance_cu" not in state:
        state["initial_balance_cu"] = client.get_consumption_user_info().paid_compute_units_balance

    def save(): worker_path.write_text(json.dumps(state, indent=2) + "\n")
    def log(message): print(datetime.now(timezone.utc).isoformat() + " " + message, flush=True)
    def call(*arguments, timeout=600):
        subprocess.run([str(ROOT / "scripts/colab.sh"), *map(str, arguments)], cwd=ROOT, timeout=timeout, check=True)
    def stop(worker):
        assigned = {entry.endpoint for entry in client.list_assignments()}
        if worker["endpoint"] in assigned: client.unassign(worker["endpoint"])
        states.remove(worker["worker"])
    def snapshot_cost():
        usage = client.get_consumption_user_info()
        value = {"time": datetime.now(timezone.utc).isoformat(), "initial_balance_cu": state["initial_balance_cu"],
                 "current_balance_cu": usage.paid_compute_units_balance,
                 "account_delta_cu": state["initial_balance_cu"] - usage.paid_compute_units_balance,
                 "account_rate_cu_hour": usage.consumption_rate_hourly,
                 "note": "Account-level observations include any other concurrent Colab usage; no fabricated per-run CU attribution."}
        store.put_json(value, "account_cost.json", directory / "staging")
        (directory / "account_cost.json").write_text(json.dumps(value, indent=2) + "\n")
        return value
    def provision():
        if len(state["workers"]) >= 6: raise RuntimeError("Automatic recovery limit reached; checkpoints remain on Drive.")
        worker = f"auto-{config['campaign_id'].removeprefix('automotive-').lower()}-{len(state['workers'])+1}"
        before = {entry.endpoint for entry in client.list_assignments()}
        error = None
        try: call("new", "-s", worker, "--gpu", config["gpu"], timeout=240)
        except (subprocess.SubprocessError, OSError) as caught: error = caught
        # Reconcile the server before any allocation retry.
        after = {entry.endpoint for entry in client.list_assignments()}
        session = states.get(worker)
        if not session or session.endpoint not in after:
            created = after - before
            if len(created) == 1:
                client.unassign(created.pop())
                log("Released an assignment from this allocation whose local handoff was incomplete.")
            raise RuntimeError("Allocation handoff failed; no duplicate allocation requested.") from error
        item = {"worker": worker, "endpoint": session.endpoint, "started_at": datetime.now(timezone.utc).isoformat()}
        state["workers"].append(item); save()
        try:
            call("exec", "-s", worker, "--timeout", 120, "-f", ROOT / "scripts/colab_campaign_init.py")
            call("install", "-s", worker, "numpy>=2.2,<3", "polars>=1.30,<2", "matplotlib>=3.10,<4", "google-api-python-client==2.201.0", "google-colab-cli==0.7.5", timeout=900)
            call("upload", "-s", worker, ROOT / "src/automotive_recsys/drive_store.py", "/content/automotive-bootstrap/drive_store.py")
            # Credentials travel separately over the authenticated runtime channel.
            # They never enter a source/data bundle or the Drive output folder.
            call("upload", "-s", worker, DRIVE_CREDENTIAL, "/content/.automotive-private/drive.json")
            call("upload", "-s", worker, COLAB_CREDENTIAL, "/content/.automotive-private/colab.json")
            remote_config = config | item | {"initial_balance_cu": state["initial_balance_cu"]}
            handoff = directory / "staging/worker.json"; handoff.parent.mkdir(exist_ok=True)
            handoff.write_text(json.dumps(remote_config)); os.chmod(handoff, 0o600)
            call("upload", "-s", worker, handoff, "/content/.automotive-private/worker.json")
            stream = (directory / f"{worker}.log").open("a")
            process = subprocess.Popen([str(ROOT / "scripts/colab.sh"), "exec", "-s", worker, "--timeout", "259200", "-f", str(ROOT / "scripts/colab_campaign_entry.py")], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            item["exec_pid"] = process.pid; save()
            log("Started " + worker + "; heavy work runs on its CUDA device.")
            return item
        except Exception:
            stop(item)
            raise

    save(); log("Controller ready; polling Drive once per 120 seconds.")
    network_failures = 0
    while True:
        try:
            progress = store.get_json("progress.json", {})
            current = state["workers"][-1] if state["workers"] else None
            active = {entry.endpoint for entry in client.list_assignments()}
            if progress.get("status") == "completed":
                if current: stop(current)
                for folder in ["report", "report/figures"]:
                    for item in store.children(store.folder(folder)):
                        if item["mimeType"] != "application/vnd.google-apps.folder":
                            store.download(item["id"], directory / folder / item["name"], item.get("appProperties", {}).get("sha256"))
                cost = snapshot_cost()
                log(f"Completed; report downloaded; account CU delta {cost['account_delta_cu']:.2f}.")
                subprocess.run(["osascript", "-e", 'display notification "3-seed 결과와 보고서가 Drive 및 프로젝트 폴더에 저장됐어." with title "빅데이터 실험 완료"'], check=False, capture_output=True)
                return
            if progress.get("status") in {"paused", "failed"}:
                if current: stop(current)
                snapshot_cost(); log("Worker " + progress["status"] + "; verified Drive results retained."); return
            cost = snapshot_cost()
            if cost["current_balance_cu"] <= config["plan"]["minimum_balance"] or cost["account_delta_cu"] >= config["plan"]["cu_budget"]:
                store.put_json({"stop_requested": True, "reason": "compute_unit_budget"}, "control.json", directory / "staging")
                log("Compute limit reached; worker will pause at its next control check.")
                if not current or current["endpoint"] not in active: return
            elif not current or current["endpoint"] not in active:
                # A new VM restores only committed Drive generations and skips completed jobs.
                if current: states.remove(current["worker"])
                provision()
            else:
                # Losing an exec websocket is not evidence that the kernel stopped.
                # An assigned worker is never duplicated or killed just for a stale log.
                (directory / "latest_progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
                started = store.get_json(f"workers/{current['worker']}/started.json")
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(current["started_at"])).total_seconds()
                if not started and age > 900:
                    stop(current)
                    store.put_json(progress | {"status": "failed", "error_type": "WorkerDidNotStart"}, "progress.json", directory / "staging")
                    log("Released a worker that never acknowledged entry; inspect its log before resuming.")
                    return
            network_failures = 0
        except Exception as error:
            network_failures += 1
            log("Control check error: " + type(error).__name__ + "; no secret diagnostics logged.")
            if network_failures >= 5:
                log("Controller stopped after repeated errors. A running worker still saves and self-releases independently.")
                return
        time.sleep(120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    if args.background:
        directory = Path(config["local_dir"])
        with (directory / "controller.log").open("a") as stream:
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(directory / "config.json")]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(directory / "controller.log"), "drive_url": config["drive_url"]}, indent=2))
    else: control(config)


if __name__ == "__main__": main()
