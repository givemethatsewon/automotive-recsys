"""Upload/run/recover the GPU review on an existing VM, then release that VM."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from colab_bundle import build_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="automotive-perf")
    parser.add_argument("--pilot", action="store_true", help="Run full-epoch integration instead of kernel benchmarks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "results/colab"

    def call(*arguments):
        subprocess.run([str(root / "scripts/colab.sh"), *map(str, arguments)], cwd=root, check=True)

    try:
        staging, manifest = build_bundle()
        call("exec", "-s", args.session, "--timeout", 120, "-f", root / "scripts/colab_remote_init.py")
        call("install", "-s", args.session, "numpy>=2.2,<3", "polars>=1.30,<2")
        call("upload", "-s", args.session, staging / "bundle-manifest.json", "/content/automotive-recsys-upload/bundle-manifest.json")
        for index, part in enumerate(manifest["parts"], 1):
            print(f"Uploading part {index}/{len(manifest['parts'])}", flush=True)
            call("upload", "-s", args.session, staging / part["name"], "/content/automotive-recsys-upload/" + part["name"])
        call("exec", "-s", args.session, "--timeout", 600, "-f", root / "scripts/colab_remote_setup.py")
        entry = "colab_remote_trainpilot.py" if args.pilot else "colab_remote_perf.py"
        tag = "pilot" if args.pilot else "perf"
        call("exec", "-s", args.session, "--timeout", 1800, "-f", root / "scripts" / entry)
        archive = output / f"{tag}-results.tar.gz"
        call("download", "-s", args.session, f"/content/automotive-recsys-{tag}-results.tar.gz", archive)
        with tarfile.open(archive) as bundle:
            bundle.extractall(output, filter="data")
        report = json.loads((output / ("train_pilot" if args.pilot else "gpu_performance") / "summary.json").read_text())
        assert report["passed"]
        if not args.pilot:
            assert report["full_catalog_cpu_gpu_top10_match"]
        call("log", "-s", args.session, "--output", output / f"{tag}-log.ipynb")
        print("GPU review results recovered successfully.", flush=True)
    finally:
        call("stop", "-s", args.session)
        call("usage")


if __name__ == "__main__":
    main()
