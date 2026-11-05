"""Prepare an isolated worker environment without modifying Colab's kernel."""
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import venv

packages = ["ipykernel", "jupyter-client", "tornado", "google-auth", "numpy", "torch"]
before = {name: importlib.metadata.version(name) for name in packages}
root = Path("/content/automotive-recsys"); root.mkdir(exist_ok=True)
environment = root / ".venv"
venv.EnvBuilder(system_site_packages=True, with_pip=False).create(environment)
python = environment / "bin/python"
uv = shutil.which("uv")
uv_environment = os.environ.copy()
if uv:
    command = [uv]
else:
    tools = root / ".uv-tool"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(tools), "uv==0.12.5"], check=True)
    uv_environment["PYTHONPATH"] = str(tools) + os.pathsep + uv_environment.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "uv"]
subprocess.run(command + ["pip", "install", "--python", str(python), "--quiet", "numpy>=2.2,<3", "polars>=1.30,<2",
                "matplotlib>=3.10,<4", "google-api-python-client==2.201.0", "google-colab-cli==0.7.5"], env=uv_environment, check=True)
after = {name: importlib.metadata.version(name) for name in packages}
assert before == after, "The managed notebook environment changed"
subprocess.run([str(python), "-c", "import sys,torch; assert sys.prefix != sys.base_prefix; assert torch.cuda.is_available(); print('Isolated CUDA worker ready')"], check=True)
Path("/content/automotive-bootstrap/kernel_environment.json").write_text(json.dumps({"before":before,"after":after,"unchanged":True,"cuda_verified":True,"worker_python":str(python)},indent=2)+"\n")
print(json.dumps({"kernel_environment_unchanged":True,"worker_python":str(python)}),flush=True)
