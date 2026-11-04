"""Remote-only setup of private, task-owned directories; refuse CPU fallback."""
import os
from pathlib import Path
import torch

assert torch.cuda.is_available(), "CUDA GPU required"
private = Path("/content/.automotive-private")
private.mkdir(mode=0o700, exist_ok=True); os.chmod(private, 0o700)
Path("/content/automotive-bootstrap").mkdir(exist_ok=True)
print("CUDA ready: " + torch.cuda.get_device_name(0), flush=True)
