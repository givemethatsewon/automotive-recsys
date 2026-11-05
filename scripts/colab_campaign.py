"""Small detached controller: provision, resume VM loss, collect reports, release GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import socket
import sys
import time

from experiment import ROOT, PRIVATE, DRIVE_CREDENTIAL, load_config
from automotive_recsys.drive_store import DriveStore
from automotive_recsys.controller_policy import progress_token, unsuccessful_workers, transient_network_error, effective_budget, owner_matches
from colab_cli.client import Client, Prod
from colab_cli.state import StateStore
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

COLAB_CREDENTIAL = PRIVATE / "colab/gcloud/application_default_credentials.json"
STATE_FILE = PRIVATE / "colab/sessions.json"
os.environ.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "120")


class RecoveryLimitReached(RuntimeError): pass


def control(config):
    directory = Path(config["local_dir"])
    lock = (directory / "controller.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("A controller already owns this campaign.")
    (directory / "controller.pid").write_text(str(os.getpid()) + "\n")
    store = DriveStore(DRIVE_CREDENTIAL, config["root_id"])
    if not owner_matches(config, store.get_json("controller_owner.json")):
        print("This campaign belongs to another controller host; no VM action taken.", flush=True)
        return
    client = Client(Prod(), AuthorizedSession(Credentials.from_authorized_user_file(str(COLAB_CREDENTIAL))))
    states = StateStore(str(STATE_FILE))
    worker_path = directory / "controller_state.json"
    state = json.loads(worker_path.read_text()) if worker_path.exists() else {"workers": []}
    if "initial_balance_cu" not in state:
        state["initial_balance_cu"] = client.get_consumption_user_info().paid_compute_units_balance

    def save(): worker_path.write_text(json.dumps(state, indent=2) + "\n")
    def log(message): print(datetime.now(timezone.utc).isoformat() + " " + message, flush=True)
    def record_controller(status, reason=None):
        value = {"status": status, "reason": reason, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "recovery_limit": "6 consecutive workers without durable progress", "workers_started": len(state["workers"]),
                 "host": socket.gethostname(), "controller_id": config.get("controller_id"),
                 "last_progress_token": state.get("last_progress_token")}
        (directory / "controller_status.json").write_text(json.dumps(value, indent=2) + "\n")
        try: store.put_json(value, "controller_status.json", directory / "staging")
        except Exception: pass
    def mark_interrupted(reason):
        record_controller("interrupted", reason)
        try:
            assigned = {entry.endpoint for entry in client.list_assignments()}
            if state["workers"] and state["workers"][-1]["endpoint"] in assigned: return
            progress = store.get_json("progress.json", {})
            if progress.get("status") == "completed": return
            progress.update(status="interrupted", controller_reason=reason, detected_at=datetime.now(timezone.utc).isoformat())
            store.put_json(progress, "progress.json", directory / "staging")
            text_path = directory / "staging/interrupted-progress.txt"
            text_path.write_text("상태: 중단\n이유: " + reason + "\n완료 작업: " + str(progress.get("completed_runs", 0)) + "/32\n마지막 저장: " + (progress.get("durable_checkpoint") or "확인 필요") + "\n")
            store.put(text_path, "progress.txt")
        except Exception: pass
    def call(*arguments, timeout=600):
        subprocess.run([str(ROOT / "scripts/colab.sh"), *map(str, arguments)], cwd=ROOT, timeout=timeout, check=True)
    def stop(worker):
        assigned = {entry.endpoint for entry in client.list_assignments()}
        if worker["endpoint"] in assigned: client.unassign(worker["endpoint"])
        states.remove(worker["worker"])
    def snapshot_cost():
        usage = client.get_consumption_user_info()
        control = store.get_json("control.json", {})
        value = {"time": datetime.now(timezone.utc).isoformat(), "initial_balance_cu": state["initial_balance_cu"],
                 "current_balance_cu": usage.paid_compute_units_balance,
                 "account_delta_cu": state["initial_balance_cu"] - usage.paid_compute_units_balance,
                 "account_rate_cu_hour": usage.consumption_rate_hourly,
                 "effective_budget_cu": effective_budget(config["plan"], control),
                 "stop_requested": bool(control.get("stop_requested")), "stop_reason": control.get("reason"),
                 "note": "Account-level observations include any other concurrent Colab usage; no fabricated per-run CU attribution."}
        store.put_json(value, "account_cost.json", directory / "staging")
        (directory / "account_cost.json").write_text(json.dumps(value, indent=2) + "\n")
        return value
    def provision():
        if unsuccessful_workers(state["workers"], state.get("last_progress_token")) >= 6:
            raise RecoveryLimitReached("Six consecutive runtimes made no durable progress.")
        worker = f"auto-{config['campaign_id'].removeprefix('automotive-').lower()}-{len(state['workers'])+1}"
        before = {entry.endpoint for entry in client.list_assignments()}
        error = None
        cached = states.get(worker)
        if cached and cached.endpoint in before:
            log("Recovered an existing named allocation before issuing another request.")
        else:
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
        item = {"worker": worker, "endpoint": session.endpoint, "started_at": datetime.now(timezone.utc).isoformat(),
                "start_progress_token": state.get("last_progress_token")}
        state["workers"].append(item); save()
        try:
            call("exec", "-s", worker, "--timeout", 120, "-f", ROOT / "scripts/colab_campaign_init.py")
            call("install", "-s", worker, "numpy>=2.2,<3", "polars>=1.30,<2", "matplotlib>=3.10,<4", "google-api-python-client==2.201.0", "google-colab-cli==0.7.5", timeout=900)
            call("upload", "-s", worker, ROOT / "src/automotive_recsys/drive_store.py", "/content/automotive-bootstrap/drive_store.py")
            call("upload", "-s", worker, ROOT / "scripts/operational_io.py", "/content/automotive-bootstrap/operational_io.py")
            call("upload", "-s", worker, ROOT / "src/automotive_recsys/controller_policy.py", "/content/automotive-bootstrap/runtime_policy.py")
            # Credentials travel separately over the authenticated runtime channel.
            # They never enter a source/data bundle or the Drive output folder.
            call("upload", "-s", worker, DRIVE_CREDENTIAL, "/content/.automotive-private/drive.json")
            call("upload", "-s", worker, COLAB_CREDENTIAL, "/content/.automotive-private/colab.json")
            remote_config = config | item | {"initial_balance_cu": state["initial_balance_cu"],
                "entry_sha256": hashlib.sha256((ROOT / "scripts/colab_campaign_entry.py").read_bytes()).hexdigest(),
                "operational_io_sha256": hashlib.sha256((ROOT / "scripts/operational_io.py").read_bytes()).hexdigest()}
            handoff = directory / "staging/worker.json"; handoff.parent.mkdir(exist_ok=True)
            handoff.write_text(json.dumps(remote_config)); os.chmod(handoff, 0o600)
            call("upload", "-s", worker, handoff, "/content/.automotive-private/worker.json")
            stream = (directory / f"{worker}.log").open("a")
            process = subprocess.Popen([str(ROOT / "scripts/colab.sh"), "exec", "-s", worker, "--timeout", "259200", "-f", str(ROOT / "scripts/colab_campaign_entry.py")], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            item["exec_pid"] = process.pid; save()
            log("Started " + worker + "; heavy work runs on its CUDA device.")
            record_controller("running")
            return item
        except Exception:
            stop(item)
            raise

    save(); record_controller("starting"); log("Controller ready; polling Drive once per 120 seconds.")
    network_failures = 0
    while True:
        try:
            if not owner_matches(config, store.get_json("controller_owner.json")):
                log("Controller ownership changed; exiting without touching the live runtime.")
                return
            progress = store.get_json("progress.json", {})
            state["last_progress_token"] = progress_token(progress, state.get("last_progress_token")); save()
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
                if shutil.which("osascript"):
                    subprocess.run(["osascript", "-e", 'display notification "3-seed 결과와 보고서가 Drive 및 프로젝트 폴더에 저장됐어." with title "빅데이터 실험 완료"'], check=False, capture_output=True)
                record_controller("completed"); return
            if progress.get("status") in {"paused", "failed"}:
                control = store.get_json("control.json", {})
                cost = snapshot_cost()
                if control.get("reason") == "operational_upgrade" or (progress.get("retryable_failure") and not control.get("stop_requested")):
                    if current: stop(current)
                    control.update(stop_requested=False, reason="recovering_retryable_worker")
                    store.put_json(control, "control.json", directory / "staging")
                    progress.update(status="resuming", stage="resuming_after_upgrade")
                    store.put_json(progress, "progress.json", directory / "staging")
                    continue
                if progress.get("pause_reason") == "compute_unit_budget" and not control.get("stop_requested") and cost["account_delta_cu"] < cost["effective_budget_cu"] and cost["current_balance_cu"] > config["plan"]["minimum_balance"]:
                    progress.update(status="resuming", stage="resuming_with_operational_budget")
                    store.put_json(progress, "progress.json", directory / "staging")
                    if current: stop(current)
                    continue
                if current: stop(current)
                snapshot_cost(); log("Worker " + progress["status"] + "; verified Drive results retained.")
                record_controller(progress["status"]); return
            cost = snapshot_cost()
            if cost["stop_requested"] and cost["stop_reason"] != "operational_upgrade" and (not current or current["endpoint"] not in active):
                progress.update(status="paused", stage="user_pause_without_active_runtime")
                store.put_json(progress, "progress.json", directory / "staging")
                record_controller("paused"); return
            if cost["current_balance_cu"] <= config["plan"]["minimum_balance"] or cost["account_delta_cu"] >= cost["effective_budget_cu"]:
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
                record_controller("running")
                started = store.get_json(f"workers/{current['worker']}/started.json")
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(current["started_at"])).total_seconds()
                if not started and age > 900:
                    stop(current)
                    control = store.get_json("control.json", {})
                    if control.get("reason") == "operational_upgrade":
                        control.update(stop_requested=False, reason="recovering_bootstrap")
                        store.put_json(control, "control.json", directory / "staging")
                    store.put_json(progress | {"status": "resuming", "stage": "recovering_bootstrap", "error_type": "WorkerDidNotStart"}, "progress.json", directory / "staging")
                    log("Released a worker that never acknowledged entry; applying the bounded recovery policy.")
                    continue
            network_failures = 0
        except RecoveryLimitReached:
            mark_interrupted("six_consecutive_workers_without_durable_progress")
            log("Recovery halted after six workers without durable progress; saved results retained.")
            return
        except Exception as error:
            network_failures += 1
            log("Control check error: " + type(error).__name__ + "; no secret diagnostics logged.")
            if transient_network_error(error):
                record_controller("waiting_for_network", type(error).__name__)
                time.sleep(min(120 * network_failures, 600))
                continue
            if network_failures >= 5:
                mark_interrupted(type(error).__name__)
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
    else:
        try: control(config)
        except Exception as error:
            path = Path(config["local_dir"]) / "controller_failure.json"
            path.write_text(json.dumps({"type": type(error).__name__, "time": datetime.now(timezone.utc).isoformat()}) + "\n")
            print("Controller exited with " + type(error).__name__ + "; no secret diagnostics logged.", flush=True)
            raise SystemExit(1)


if __name__ == "__main__": main()
