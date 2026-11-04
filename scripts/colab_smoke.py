"""End-to-end CLI check on an existing named VM; always stop the owned VM."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from colab_bundle import build_bundle, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="automotive-setup")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "scripts/colab.sh"
    destination = root / "results/colab"
    destination.mkdir(parents=True, exist_ok=True)

    def call(*arguments):
        subprocess.run([str(wrapper), *map(str, arguments)], cwd=root, check=True)

    try:
        staging, manifest = build_bundle()
        call("exec", "-s", args.session, "--timeout", "120", "-f", root / "scripts/colab_remote_init.py")
        call("install", "-s", args.session, "numpy>=2.2,<3", "polars>=1.30,<2")
        call("upload", "-s", args.session, staging / "bundle-manifest.json", "/content/automotive-recsys-upload/bundle-manifest.json")
        for index, part in enumerate(manifest["parts"], 1):
            print(f"Uploading data part {index}/{len(manifest['parts'])}", flush=True)
            call("upload", "-s", args.session, staging / part["name"], "/content/automotive-recsys-upload/" + part["name"])
        call("exec", "-s", args.session, "--timeout", "600", "-f", root / "scripts/colab_remote_setup.py")
        archive = destination / "smoke-results.tar.gz"
        call("download", "-s", args.session, "/content/automotive-recsys-smoke-results.tar.gz", archive)
        with tarfile.open(archive) as results:
            results.extractall(destination, filter="data")
        summary = json.loads((destination / "colab_smoke/summary.json").read_text())
        assert summary["passed"] and summary["popularity_matches_local"]
        assert sha256(destination / "colab_smoke/gpu_smoke.pt") == summary["checkpoint_sha256"]
        call("log", "-s", args.session, "--output", destination / "setup-log.ipynb")
        print("Remote CUDA, dataset transfer, baseline reproduction and artifact recovery passed.", flush=True)
    finally:
        # Only the explicitly named task-owned VM is touched.
        call("stop", "-s", args.session)
        call("usage")


if __name__ == "__main__":
    main()
