"""Bounded Ubuntu controller using the campaign Drive journal and recovery policy.

An active notebook launcher keeps execution alive; Drive receipts, not websocket
or CLI BUSY state, determine progress. Only owned endpoints are released.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from automotive_recsys.drive_store import DriveStore
from automotive_recsys.durable_stages import atomic_json
from automotive_recsys.controller_policy import unsuccessful_workers, owner_matches

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path.home() / ".config/automotive-recsys"
REMOTE = "/content/automotive-vehicle-durable"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cli(*args, timeout=300, check=True):
    process = subprocess.Popen(
        [str(ROOT / "scripts/colab.sh"), *map(str, args)],
        cwd=ROOT,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if check and code:
        raise subprocess.CalledProcessError(code, args)
    return code


def controller(directory):
    from colab_cli.client import Client, Prod
    from colab_cli.state import StateStore, SessionState
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials

    run = directory.resolve()
    cfg = json.loads((run / "config.json").read_text())
    lock = (run / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = DriveStore(
        PRIVATE / "drive/gcloud/application_default_credentials.json", cfg["drive_root"]
    )
    assert owner_matches(cfg, store.get_json("controller_owner.json")), (
        "Another controller owns this experiment"
    )
    client = Client(
        Prod(),
        AuthorizedSession(
            Credentials.from_authorized_user_file(
                str(PRIVATE / "colab/gcloud/application_default_credentials.json")
            )
        ),
    )
    states = StateStore(str(PRIVATE / "colab/sessions.json"))
    path = run / "controller_state.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"workers": [], "token": [0, 0, 0]}
    )
    if "initial_balance" not in state:
        state["initial_balance"] = (
            client.get_consumption_user_info().paid_compute_units_balance
        )

    def save():
        atomic_json(path, state)

    def record(status, **extra):
        value = {
            "status": status,
            "time": datetime.now(timezone.utc).isoformat(),
            "workers": len(state["workers"]),
            "durable_position": state["token"],
            **extra,
        }
        atomic_json(run / "controller_status.json", value)
        store.put_json(value, "controller_status.json", run / "staging")

    def release(worker):
        if worker and worker["endpoint"] in {
            x.endpoint for x in client.list_assignments()
        }:
            client.unassign(worker["endpoint"])
        if worker:
            states.remove(worker["name"])

    def budget():
        usage = client.get_consumption_user_info()
        value = {
            "balance": usage.paid_compute_units_balance,
            "rate": usage.consumption_rate_hourly,
            "initial_balance": state["initial_balance"],
            "delta": state["initial_balance"] - usage.paid_compute_units_balance,
        }
        atomic_json(run / "usage.json", value)
        store.put_json(value, "usage.json", run / "staging")
        if value["balance"] < 30 or value["delta"] >= cfg.get("cu_budget", 8):
            raise RuntimeError("Bounded experiment CU limit reached")

    def prepare(worker):
        name = worker["name"]
        init = run / "init.py"
        init.write_text(
            f"from pathlib import Path\nimport torch,json\nroot=Path({REMOTE!r});root.mkdir(exist_ok=True)\n(root/'.private').mkdir(mode=0o700,exist_ok=True)\nassert torch.cuda.is_available()\nprint(json.dumps({{'gpu':torch.cuda.get_device_name(0)}}))\n"
        )
        cli("exec", "-s", name, "--timeout", "120", "-f", init, timeout=160)
        for part in cfg["upload_parts"]:
            cli("upload", "-s", name, run / part["name"], REMOTE + "/" + part["name"])
        cli(
            "upload",
            "-s",
            name,
            PRIVATE / "drive/gcloud/application_default_credentials.json",
            REMOTE + "/.private/drive.json",
        )
        remote_cfg = cfg | {"worker": name}
        atomic_json(run / "run_config.json", remote_cfg)
        cli("upload", "-s", name, run / "run_config.json", REMOTE + "/run_config.json")
        cli(
            "exec",
            "-s",
            name,
            "--timeout",
            "1800",
            "-f",
            run / "bootstrap.py",
            timeout=1900,
        )
        cli("download", "-s", name, REMOTE + "/prepared.json", run / "prepared.json")
        receipt = json.loads((run / "prepared.json").read_text())
        assert (
            receipt["ready"]
            and receipt["context"] == cfg["context"]
            and receipt["kernel_unchanged"]
        )
        worker["ready"] = True
        save()

    def provision():
        if unsuccessful_workers(state["workers"], state["token"]) >= 6:
            raise RuntimeError("Six allocations without new durable progress")
        name = cfg["session_prefix"] + "-" + str(len(state["workers"]) + 1)
        before = {x.endpoint for x in client.list_assignments()}
        cached = states.get(name)
        error = None
        if not cached or cached.endpoint not in before:
            try:
                cli("new", "-s", name, "--gpu", "L4", timeout=300)
            except (subprocess.SubprocessError, OSError) as e:
                error = e
        assignments = client.list_assignments()
        after = {x.endpoint for x in assignments}
        cached = states.get(name)
        if not cached or cached.endpoint not in after:
            created = [x for x in assignments if x.endpoint not in before]
            if len(created) != 1:
                raise RuntimeError(
                    "Allocation could not be reconciled; no duplicate request issued"
                ) from error
            a = created[0]
            proxy = a.runtime_proxy_info
            assert a.accelerator.value == "L4"
            states.add(
                SessionState(
                    name=name,
                    endpoint=a.endpoint,
                    token=proxy.token,
                    url=proxy.url,
                    token_expires_at=proxy.expires_at(),
                    variant="GPU",
                    accelerator="L4",
                    machine_shape="STANDARD",
                )
            )
            cached = states.get(name)
        worker = {
            "name": name,
            "endpoint": cached.endpoint,
            "ready": False,
            "start_progress_token": list(state["token"]),
        }
        state["workers"].append(worker)
        save()
        return worker

    def launch(worker):
        # As in colab_campaign_launcher: actual kernel execution remains active.
        entry = run / "launcher.py"
        entry.write_text(f"""from pathlib import Path
import subprocess,time,json,os
root=Path({REMOTE!r});receipt=root/'worker_started.json'
if receipt.exists():
 pid=json.loads(receipt.read_text())['pid']
 if not (root/'worker_done.json').exists():os.kill(pid,0)
else:
 with (root/'worker.log').open('w') as log:
  p=subprocess.Popen([str(root/'venv/bin/python'),'-u',str(root/'scripts/vehicle_durable_worker.py')],env=os.environ|{{'PYTHONPATH':str(root/'src')}},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 pid=p.pid;receipt.write_text(json.dumps({{'pid':pid}}))
while not (root/'worker_done.json').exists():
 os.kill(pid,0);print('NOTEBOOK_EXECUTION_ACTIVE',flush=True);time.sleep(15)
print((root/'worker_done.json').read_text(),flush=True)
""")
        log = (run / (worker["name"] + "-launcher.log")).open("a")
        process = subprocess.Popen(
            [
                str(ROOT / "scripts/colab.sh"),
                "exec",
                "-s",
                worker["name"],
                "--timeout",
                "21600",
                "-f",
                str(entry),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        worker["launcher_pid"] = process.pid
        worker["launched_at"] = time.time()
        save()

    def recover_results():
        for name in cfg["runs"]:
            pointer = store.get_json(f"runs/{name}/completed.json")
            assert pointer and pointer["status"] == "completed"
            for a in pointer["artifacts"]:
                store.download(
                    a["id"], run / "output" / name / a["relative"], a["sha256"]
                )
        for name in [
            "events.jsonl",
            "progress.json",
            "progress.txt",
            "verification_passed.json",
            "completion.json",
        ]:
            meta = store.find(name)
            if meta:
                store.download(
                    meta["id"], run / name, meta.get("appProperties", {}).get("sha256")
                )

    record("starting")
    save()
    current = state["workers"][-1] if state["workers"] else None
    last_cost = 0
    last_token_time = time.time()
    try:
        while True:
            if time.time() - last_cost > 60:
                budget()
                last_cost = time.time()
            completion = store.get_json("completion.json")
            if completion and completion.get("status") == "completed":
                recover_results()
                release(current)
                record("completed", results_recovered=True)
                break
            progress = store.get_json("progress.json", {})
            token = progress.get("durable_position", [0, 0, 0])
            if token > state["token"]:
                state["token"] = token
                last_token_time = time.time()
                save()
            assigned = {x.endpoint for x in client.list_assignments()}
            current = state["workers"][-1] if state["workers"] else None
            if not current or current["endpoint"] not in assigned:
                if current:
                    states.remove(current["name"])
                    record("recovering", reason="owned_session_missing")
                current = provision()
            if not current.get("ready"):
                for attempt in range(3):
                    try:
                        prepare(current)
                        break
                    except (subprocess.SubprocessError, OSError):
                        receipt = run / "prepared.json"
                        receipt.unlink(missing_ok=True)
                        cli(
                            "download",
                            "-s",
                            current["name"],
                            REMOTE + "/prepared.json",
                            receipt,
                            check=False,
                            timeout=60,
                        )
                        if receipt.exists():
                            ready = json.loads(receipt.read_text())
                            if (
                                ready.get("ready")
                                and ready.get("context") == cfg["context"]
                            ):
                                current["ready"] = True
                                save()
                                break
                        if attempt == 2:
                            raise
                        (ROOT / ".fresh-kernel-once").touch()
            heartbeat = store.get_json("worker_heartbeat.json", {})
            if heartbeat.get("worker") != current["name"]:
                if (
                    not current.get("launcher_pid")
                    or time.time() - current.get("launched_at", 0) > 180
                ):
                    # Do not retry a launch if a remote PID receipt already exists.
                    local = run / "remote_worker_started.json"
                    local.unlink(missing_ok=True)
                    cli(
                        "download",
                        "-s",
                        current["name"],
                        REMOTE + "/worker_started.json",
                        local,
                        check=False,
                        timeout=60,
                    )
                    if local.exists():
                        if time.time() - current.get("launched_at", time.time()) > 600:
                            raise RuntimeError(
                                "Worker started but no durable heartbeat"
                            )
                    else:
                        if current.get("launch_attempts", 0) >= 3:
                            raise RuntimeError(
                                "Worker did not start after three reconciled attempts"
                            )
                        current["launch_attempts"] = (
                            current.get("launch_attempts", 0) + 1
                        )
                        if current["launch_attempts"] > 1:
                            (ROOT / ".fresh-kernel-once").touch()
                        launch(current)
            else:
                age = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(heartbeat["time"])
                ).total_seconds()
                if age > 600:
                    release(current)
                    record("recovering", reason="heartbeat_stale")
                    time.sleep(5)
                    continue
                failure = store.get_json("worker_failure.json", {})
                if failure.get("worker") == current["name"]:
                    raise RuntimeError(failure.get("error", "Worker failed"))
            if time.time() - last_token_time > 3600:
                raise RuntimeError("No newly verified stage/batch for one hour")
            record(
                "running",
                worker=current["name"],
                phase=heartbeat.get("phase", "preparing"),
            )
            time.sleep(30)
    except BaseException as e:
        try:
            record("failed", error=str(e))
        finally:
            release(current)
        raise
    finally:
        for worker in state["workers"]:
            try:
                meta = store.find(f"workers/{worker['name']}/worker.log")
                if meta:
                    store.download(
                        meta["id"],
                        run / "worker-logs" / (worker["name"] + ".log"),
                        meta.get("appProperties", {}).get("sha256"),
                    )
            except Exception:
                pass
        try:
            cli("usage", check=False)
            cli("sessions", check=False)
        finally:
            lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    try:
        controller(p.parse_args().directory)
    except Exception as error:
        # Controlled failure is already recorded. Do not create an infinite
        # systemd restart loop; unexpected process death remains restartable.
        print(type(error).__name__ + ": " + str(error), flush=True)
