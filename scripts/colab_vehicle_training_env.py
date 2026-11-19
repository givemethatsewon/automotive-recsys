"""Training task environment; pin worker torch without changing kernel packages."""

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

packages = ["ipykernel", "jupyter-client", "tornado", "google-auth", "numpy", "torch"]
before = {p: importlib.metadata.version(p) for p in packages}
root = Path("/content/automotive-recsys")
root.mkdir(exist_ok=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "venv",
        "--without-pip",
        "--system-site-packages",
        str(root / ".venv"),
    ],
    check=True,
)
python = root / ".venv/bin/python"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "--python",
        str(python),
        "install",
        "-q",
        "numpy>=2.2,<3",
        "polars>=1.30,<2",
        "numba",
        "pandas",
        "pyarrow",
        "matplotlib>=3.10,<4",
        "google-api-python-client==2.201.0",
        "google-colab-cli==0.7.5",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "--python",
        str(python),
        "install",
        "-q",
        "torch==2.11.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
    ],
    check=True,
)
after = {p: importlib.metadata.version(p) for p in packages}
assert before == after, "Managed notebook packages changed"
subprocess.run(
    [
        str(python),
        "-c",
        "import torch;assert torch.cuda.is_available();assert str(torch.__version__)=='2.11.0+cu130'",
    ],
    check=True,
)
Path("/content/automotive-bootstrap/kernel_environment.json").write_text(
    json.dumps(
        {
            "before": before,
            "after": after,
            "unchanged": True,
            "cuda_verified": True,
            "worker_torch": "2.11.0+cu130",
        }
    )
)
print("ISOLATED_VEHICLE_TRAINING_ENV_READY", flush=True)
