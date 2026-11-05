"""Keep the real notebook execution active while the isolated worker trains."""
from pathlib import Path
import subprocess

python=Path("/content/automotive-recsys/.venv/bin/python")
script=Path("/content/automotive-bootstrap/campaign-worker.py")
process=subprocess.Popen([str(python),"-u",str(script)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
for line in process.stdout: print(line,end="",flush=True)
if process.wait(): raise RuntimeError("Isolated worker exited; inspect the private progress record.")
