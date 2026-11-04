from pathlib import Path
import json
import torch

Path("/content/automotive-recsys-upload").mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), "CUDA GPU is required for this session."
print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}))
