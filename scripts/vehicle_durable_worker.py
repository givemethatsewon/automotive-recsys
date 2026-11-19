"""Isolated Colab worker; retain verified progress externally after each stage."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from automotive_recsys.drive_store import DriveStore
from automotive_recsys.durable_stages import atomic_json

ROOT = Path("/content/automotive-vehicle-durable")


def main():
    cfg = json.loads((ROOT / "run_config.json").read_text())
    credential = ROOT / ".private/drive.json"
    env = os.environ | {
        "PYTHONPATH": str(ROOT / "src"),
        "NUMBA_NUM_THREADS": "4",
        "POLARS_MAX_THREADS": "4",
        "AUTOMOTIVE_DRIVE_ROOT": cfg["drive_root"],
        "AUTOMOTIVE_DRIVE_CREDENTIAL": str(credential),
    }
    store = DriveStore(credential, cfg["drive_root"])
    stop = threading.Event()
    phase = {"name": "preflight"}
    worker = cfg["worker"]
    staging = ROOT / "transfer-parent"
    store.put_json(
        {"worker": worker, "time": datetime.now(timezone.utc).isoformat()},
        "worker_started.json",
        staging,
    )

    def heartbeat():
        separate = DriveStore(credential, cfg["drive_root"])
        while not stop.is_set():
            try:
                separate.put_json(
                    {
                        "worker": worker,
                        "phase": phase["name"],
                        "time": datetime.now(timezone.utc).isoformat(),
                    },
                    "worker_heartbeat.json",
                    ROOT / "transfer-heartbeat",
                )
                log = ROOT / "worker.log"
                if log.exists():
                    snapshot = ROOT / "transfer-heartbeat/worker-log.txt"
                    snapshot.write_bytes(log.read_bytes())
                    separate.put(snapshot, f"workers/{worker}/worker.log")
            except Exception as e:
                print("Heartbeat transfer failed: " + type(e).__name__, flush=True)
            stop.wait(30)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/vehicle_durable_preflight.py"),
                str(ROOT),
            ],
            env=env,
            check=True,
        )
        verification = store.get_json("verification_passed.json", {})
        assert (
            verification.get("passed") and verification.get("context") == cfg["context"]
        ), "Preflight gate not verified for current source/input context"
        phase["name"] = "awaiting_host_release"
        while not store.get_json("control.json", {}).get("release_real"):
            stop.wait(10)
        for index, run in enumerate(cfg["runs"]):
            phase["name"] = run
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/vehicle_v2_experiment.py"),
                    "--root",
                    str(ROOT),
                    "--run",
                    run,
                    "--output",
                    str(ROOT / "output" / run),
                    "--run-index",
                    str(index),
                    "--planned-runs",
                    str(len(cfg["runs"])),
                ],
                env=env,
                check=True,
            )
        records = [store.get_json(f"runs/{run}/completed.json") for run in cfg["runs"]]
        assert all(r and r["status"] == "completed" for r in records)
        store.put_json(
            {
                "status": "completed",
                "runs": cfg["runs"],
                "time": datetime.now(timezone.utc).isoformat(),
            },
            "completion.json",
            staging,
        )
        atomic_json(ROOT / "worker_done.json", {"returncode": 0})
    except BaseException as e:
        atomic_json(ROOT / "worker_done.json", {"returncode": 1, "error": str(e)})
        store.put_json(
            {
                "worker": worker,
                "status": "failed",
                "error": str(e),
                "time": datetime.now(timezone.utc).isoformat(),
            },
            "worker_failure.json",
            staging,
        )
        raise
    finally:
        stop.set()
        thread.join(timeout=10)
        try:
            log = ROOT / "worker.log"
            if log.exists():
                store.put(log, f"workers/{worker}/worker.log")
        finally:
            credential.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
