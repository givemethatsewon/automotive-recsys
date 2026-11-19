"""Freeze inputs/source, verify real Drive recovery, then stage a new run."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from automotive_recsys.drive_store import DriveStore
from automotive_recsys.durable_stages import DurableStages, atomic_json

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    args = p.parse_args()
    run = args.directory.resolve()
    run.mkdir(parents=True, exist_ok=True)
    inputs = args.input_root.resolve()
    campaign = ROOT / "results/campaigns/automotive-20261104T204146Z"
    prior = json.loads((campaign / "config.json").read_text())
    runs = ["final_" + m + "_seed42" for m in ["bpr_mf", "lightgcn", "xsimgcl"]]
    source = [f for folder in ["src", "scripts"] for f in (ROOT / folder).rglob("*.py")]
    source += [ROOT / "scripts/colab.sh"]
    files = [(f, str(f.relative_to(ROOT))) for f in source]
    for folder in ["inputs", "v2", "provider"]:
        for f in (inputs / folder).iterdir():
            if f.suffix in [".py", ".parquet"]:
                files.append((f, folder + "/" + f.name))
    records = {
        name: json.loads((campaign / f"review/{name}.json").read_text())
        for name in runs
    }
    manifest = {
        "files": {
            rel: {"sha256": sha(f), "bytes": f.stat().st_size} for f, rel in files
        },
        "source_sha256": {str(f.relative_to(ROOT)): sha(f) for f in source},
        "data_parts": prior["parts"],
        "baseline": [],
    }
    for name, meta in records[runs[0]]["summary"]["inputs"].items():
        manifest["files"]["data/" + name] = meta
    for name, record in records.items():
        for a in record["artifacts"]:
            if a["relative"].endswith(("recommendations.npz", "metrics.json")):
                rel = f"baseline/{name}/" + a["relative"]
                manifest["baseline"].append(a | {"relative": rel})
                manifest["files"][rel] = {"sha256": a["sha256"]}
        manifest["baseline"].append(
            record["best"] | {"relative": f"checkpoints/{name}/best.pt"}
        )
    context = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(run / "manifest.json", manifest)
    with tarfile.open(run / "upload.tar", "w") as t:
        for f, rel in files:
            t.add(f, arcname=rel)
        t.add(run / "manifest.json", arcname="manifest.json")
    parts = []
    with (run / "upload.tar").open("rb") as stream:
        index = 0
        while block := stream.read(16 << 20):
            path = run / f"upload.part{index:03d}"
            path.write_bytes(block)
            parts.append({"name": path.name, "sha256": sha(path)})
            index += 1
    (run / "upload.tar").unlink()
    credential = (
        Path.home()
        / ".config/automotive-recsys/drive/gcloud/application_default_credentials.json"
    )
    store = DriveStore(credential)
    if (run / "config.json").exists():
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["context"] == context, "Source changed: use a new run directory"
        store = DriveStore(credential, cfg["drive_root"])
    else:
        root = store.create_root("automotive-vehicle-v2-durable-20261117")
        cfg = {
            "context": context,
            "drive_root": root["id"],
            "drive_url": root.get("webViewLink"),
            "runs": runs,
            "upload_parts": parts,
            "session_prefix": "automotive-v2-durable-1117",
            "controller_id": "ubuntu-v2-durable-1117",
            "cu_budget": 8,
        }
        atomic_json(run / "config.json", cfg)
        store.put_json(
            {"controller_id": cfg["controller_id"]},
            "controller_owner.json",
            run / "staging",
        )
        store.put_json(
            {
                "context": context,
                "runs": runs,
                "source_sha256": manifest["source_sha256"],
                "protocol": "no fitting; train-only A1+A2; valid-only beta; same-beta primary pool500",
            },
            "plan.json",
            run / "staging",
        )
    # Actual provider writes + SHA-verified downloads, with an interrupted commit.
    teststore = DriveStore(credential, store.folder("preflight/host"))
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        stage = DurableStages(
            teststore,
            base / "journal",
            "host_roundtrip",
            base / "out",
            {"context": context},
            ["first", "second"],
            planned_runs=1,
        )
        if stage.pointer:
            stage.restore()
        atomic_json(stage.output / "value.json", {"value": 1})
        stage.commit("first", done=True)
        old = teststore.put_json

        def fail(value, path, staging):
            if path.endswith("/restore.json"):
                raise IOError("Intentional preflight pointer failure")
            return old(value, path, staging)

        teststore.put_json = fail
        atomic_json(stage.output / "value.json", {"value": 2})
        try:
            stage.commit("second")
        except IOError:
            pass
        else:
            raise AssertionError("Injected failure did not occur")
        teststore.put_json = old
        restored = DurableStages(
            teststore,
            base / "new-journal",
            "host_roundtrip",
            base / "restored",
            {"context": context},
            ["first", "second"],
            planned_runs=1,
        )
        assert restored.restore()
        assert json.loads((restored.output / "value.json").read_text()) == {"value": 1}
        artifact = restored.pointer["artifacts"][0]
        try:
            teststore.download(artifact["id"], base / "bad-hash.json", "0" * 64)
        except IOError:
            pass
        else:
            raise AssertionError("Corruption check accepted wrong SHA")
    proof = {
        "passed": True,
        "context": context,
        "real_drive_upload_download": True,
        "sha256_mismatch_rejected": True,
        "failed_pointer_commit_kept_previous_generation": True,
        "fresh_directory_restore": True,
    }
    atomic_json(run / "host_verified.json", proof)
    store.put_json(proof, "host_verification.json", run / "staging")
    bootstrap = """from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile,importlib.metadata
root=Path('/content/automotive-vehicle-durable')
cfg=json.loads((root/'run_config.json').read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
receipt=root/'prepared.json'
if not receipt.exists() or json.loads(receipt.read_text()).get('context')!=cfg['context']:
 with (root/'upload.tar').open('wb') as stream:
  for part in cfg['upload_parts']:
   p=root/part['name'];assert sha(p)==part['sha256'];stream.write(p.read_bytes())
 with tarfile.open(root/'upload.tar') as t:t.extractall(root,filter='data')
 before={p:importlib.metadata.version(p) for p in ['torch','numpy','ipykernel','jupyter-client','google-auth']}
 subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(root/'venv')],check=True)
 subprocess.run([sys.executable,'-m','pip','--python',str(root/'venv/bin/python'),'install','-q','polars>=1.30,<2','numba','pandas','pyarrow','google-api-python-client==2.201.0'],check=True)
 subprocess.run([sys.executable,'-m','pip','--python',str(root/'venv/bin/python'),'install','-q','torch==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],check=True)
 after={p:importlib.metadata.version(p) for p in before};assert before==after,'Managed kernel packages changed'
 subprocess.run([str(root/'venv/bin/python'),'-c',"import torch;assert torch.cuda.is_available();assert str(torch.__version__).startswith('2.11.0')"],check=True)
 sys.path.insert(0,str(root/'src'))
 from automotive_recsys.drive_store import DriveStore
 credential=root/'.private/drive.json';os.chmod(credential,0o600)
 store=DriveStore(credential)
 manifest=json.loads((root/'manifest.json').read_text())
 with (root/'base.tar').open('wb') as stream:
  for part in manifest['data_parts']:
   p=root/part['name']
   if not p.exists() or sha(p)!=part['sha256']:store.download(part['id'],p,part['sha256'])
   with p.open('rb') as source:
    for block in iter(lambda:source.read(8<<20),b''):stream.write(block)
 with tarfile.open(root/'base.tar') as t:t.extractall(root,members=[m for m in t.getmembers() if m.name.startswith('data/')],filter='data')
 for a in manifest['baseline']:
  p=root/a['relative']
  if not p.exists() or sha(p)!=a['sha256']:store.download(a['id'],p,a['sha256'])
 receipt.write_text(json.dumps({'ready':True,'context':cfg['context'],'kernel_unchanged':before==after}))
print('DURABLE_INPUTS_READY',flush=True)
"""
    (run / "bootstrap.py").write_text(bootstrap)
    print(
        json.dumps(
            {
                "host_preflight": "passed",
                "config": str(run / "config.json"),
                "drive_url": cfg["drive_url"],
                "context": context,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
