"""Runs only on the Colab VM: verify handoff, CUDA, evaluation, artifact recovery."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    upload = Path("/content/automotive-recsys-upload")
    root = Path("/content/automotive-recsys")
    root.mkdir(exist_ok=True)
    manifest = json.loads((upload / "bundle-manifest.json").read_text())
    archive = upload / "input.tar"
    with archive.open("wb") as output:
        for part in manifest["parts"]:
            source = upload / part["name"]
            assert source.stat().st_size == part["bytes"] and digest(source) == part["sha256"]
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    output.write(block)
    assert digest(archive) == manifest["archive_sha256"]
    with tarfile.open(archive) as source:
        source.extractall(root, filter="data")
    for name, expected in manifest["files"].items():
        assert digest(root / name) == expected, name
    print(f"Verified {len(manifest['files'])} uploaded files by SHA-256.", flush=True)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))

    import numpy as np
    import polars as pl
    import torch
    from automotive_recsys.audit import audit, write_json
    from automotive_recsys.popularity import run_popularity
    from benchmark_local import timed_step

    assert torch.cuda.is_available(), "Colab CUDA GPU was not attached."
    output = root / "results" / "colab_smoke"
    output.mkdir(parents=True, exist_ok=True)
    checks = audit(root / "data")
    write_json(output / "audit.json", checks)
    assert checks["passed"], "Data contract checks failed."
    baseline = run_popularity(root / "data", output / "popularity_valid", "valid", 10, 100)
    reference = json.loads((root / "reference/popularity_valid_metrics.json").read_text())
    for group in ["all", "1-5", "6+"]:
        for metric in ["ndcg", "recall", "hit_rate", "aplt", "arp"]:
            assert math.isclose(baseline["metrics"][group][metric]["mean"], reference["metrics"][group][metric]["mean"], rel_tol=1e-10, abs_tol=1e-12)
    environment = os.environ | {"PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"], env=environment, check=True)

    torch.manual_seed(42)
    train = pl.read_parquet(root / "data/train.parquet", columns=["uid", "iid"])
    n_users = checks["training_population"]["users"]
    n_nodes = n_users + checks["training_population"]["items"]
    user = torch.from_numpy(train["uid"].to_numpy().astype(np.int64))
    item = torch.from_numpy(train["iid"].to_numpy().astype(np.int64)) + n_users
    rows, cols = torch.cat([user, item]), torch.cat([item, user])
    degree = torch.bincount(rows, minlength=n_nodes).float()
    weights = torch.rsqrt(degree[rows] * degree[cols])
    graph = torch.sparse_coo_tensor(torch.stack([rows, cols]), weights, (n_nodes, n_nodes)).coalesce().to("cuda")
    uid = user[:8192].to("cuda")
    positive = item[:8192].to("cuda")
    negative = torch.randint(n_users, n_nodes, (8192,), device="cuda")
    del user, item, rows, cols, degree, weights, train
    gc.collect()
    embedding = torch.nn.Parameter(torch.randn(n_nodes, 64, device="cuda") * 0.1)
    optimizer = torch.optim.Adam([embedding], lr=1e-3)
    timed_step(embedding, optimizer, uid, positive, negative, graph, 3)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = timed_step(embedding, optimizer, uid, positive, negative, graph, 3)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    assert bool(torch.isfinite(loss))
    checkpoint = {"purpose": "CUDA plumbing smoke only; not a trained recommender", "tensor": embedding[:4].detach().cpu()}
    torch.save(checkpoint, output / "gpu_smoke.pt")
    restored = torch.load(output / "gpu_smoke.pt", map_location="cpu", weights_only=True)
    torch.testing.assert_close(checkpoint["tensor"], restored["tensor"])
    summary = {
        "passed": True, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": sys.version.split()[0], "data_checks": len(checks["checks"]),
        "uploaded_files_verified": len(manifest["files"]), "popularity_matches_local": True,
        "valid_users": baseline["evaluated_users"], "ndcg_at_10": baseline["metrics"]["all"]["ndcg"]["mean"],
        "full_train_graph_lightgcn_3_layer_update_seconds": seconds,
        "cuda_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "note": "Two diagnostic GPU updates only; no model convergence or final test evaluation.",
        "checkpoint_sha256": digest(output / "gpu_smoke.pt"),
    }
    write_json(output / "summary.json", summary)
    with tarfile.open("/content/automotive-recsys-smoke-results.tar.gz", "w:gz") as bundle:
        bundle.add(output, arcname="colab_smoke")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
