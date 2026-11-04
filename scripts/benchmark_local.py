"""Bounded kernel timings on the actual graph; this does not train an experiment.

Run: uv run --extra training python scripts/benchmark_local.py
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def timed_step(embedding, optimizer, uids, positives, negatives, graph=None, layers=3):
    optimizer.zero_grad(set_to_none=True)
    if graph is None:
        final = embedding
    else:
        propagated = embedding
        final = embedding
        for _ in range(layers):
            propagated = torch.sparse.mm(graph, propagated)
            final = final + propagated
        final = final / (layers + 1)
    u, pos, neg = final[uids], final[positives], final[negatives]
    loss = F.softplus((u * neg).sum(1) - (u * pos).sum(1)).mean()
    loss = loss + 1e-4 * (embedding[uids].square().sum() + embedding[positives].square().sum() + embedding[negatives].square().sum()) / (2 * len(uids))
    loss.backward()
    optimizer.step()
    return loss.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/hardware_benchmark.json"))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    train = pl.read_parquet(args.data_dir / "train.parquet", columns=["uid", "iid"])
    n_users = int(train["uid"].max()) + 1
    n_items = int(train["iid"].max()) + 1
    n_nodes = n_users + n_items
    n_edges = train.height
    selected = np.random.default_rng(42).integers(0, n_edges, args.batch_size)
    uid = torch.from_numpy(train["uid"].to_numpy()[selected].astype(np.int64))
    positive = torch.from_numpy(train["iid"].to_numpy()[selected].astype(np.int64)) + n_users
    negative = torch.randint(n_users, n_nodes, (args.batch_size,))
    result = {
        "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
        "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1",
        "config": vars(args) | {"data_dir": str(args.data_dir), "output": str(args.output)},
        "nodes": n_nodes, "train_interactions": n_edges,
        "embedding_float32_MiB": n_nodes * args.dim * 4 / 2**20,
        "steps_per_epoch": math.ceil(n_edges / args.batch_size),
        "note": "Kernel timings with fixed minibatch, not convergence or a completed training epoch. Negative rejection, shuffling and evaluation excluded.",
        "timings": {},
    }

    if result["mps_available"]:
        try:
            graph = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.ones(2), (2, 2)).to("mps")
            feature = torch.randn(2, 4, device="mps", requires_grad=True)
            torch.sparse.mm(graph, feature).sum().backward()
            sync("mps")
            result["mps_sparse_graph"] = {"supported": True}
            del graph, feature
        except (RuntimeError, NotImplementedError) as error:
            result["mps_sparse_graph"] = {"supported": False, "error": str(error).splitlines()[0][:400]}

    devices = ["cpu"] + (["mps"] if result["mps_available"] else [])
    for device in devices:
        embedding = torch.nn.Parameter(torch.randn(n_nodes, args.dim, device=device) * 0.1)
        optimizer = torch.optim.Adam([embedding], lr=1e-3)
        tensors = [value.to(device) for value in [uid, positive, negative]]
        timed_step(embedding, optimizer, *tensors)
        sync(device)
        durations = []
        for _ in range(args.steps):
            started = time.perf_counter()
            loss = timed_step(embedding, optimizer, *tensors)
            sync(device)
            durations.append(time.perf_counter() - started)
        result["timings"][f"bpr_mf_{device}"] = {"step_seconds": durations, "median_step_seconds": statistics.median(durations)}
        print(f"BPR-MF {device}: {statistics.median(durations):.4f} s/update", flush=True)
        del embedding, optimizer, tensors, loss
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # Actual full train-only undirected graph, normalized as D^-1/2 A D^-1/2.
    user = torch.from_numpy(train["uid"].to_numpy().astype(np.int64))
    item = torch.from_numpy(train["iid"].to_numpy().astype(np.int64)) + n_users
    rows, columns = torch.cat([user, item]), torch.cat([item, user])
    degrees = torch.bincount(rows, minlength=n_nodes).float()
    weights = torch.rsqrt(degrees[rows] * degrees[columns])
    graph_coo = torch.sparse_coo_tensor(torch.stack([rows, columns]), weights, (n_nodes, n_nodes)).coalesce()
    del user, item, rows, columns, degrees, weights, train
    gc.collect()
    graph_devices = ["cpu"] + (["mps"] if result.get("mps_sparse_graph", {}).get("supported") else [])
    for device in graph_devices:
        try:
            graph = graph_coo.to_sparse_csr() if device == "cpu" else graph_coo.to(device)
            if device == "mps":
                reference = torch.sparse.mm(graph_coo, torch.ones(n_nodes, 1))
                observed = torch.sparse.mm(graph, torch.ones(n_nodes, 1, device=device)).cpu()
                torch.testing.assert_close(observed, reference, rtol=1e-4, atol=1e-4)
                result["mps_full_graph_matches_cpu"] = True
                del reference, observed
            embedding = torch.nn.Parameter(torch.randn(n_nodes, args.dim, device=device) * 0.1)
            optimizer = torch.optim.Adam([embedding], lr=1e-3)
            tensors = [value.to(device) for value in [uid, positive, negative]]
            # One warm-up and one measured update keep this diagnostic bounded.
            timed_step(embedding, optimizer, *tensors, graph, args.layers)
            sync(device)
            started = time.perf_counter()
            timed_step(embedding, optimizer, *tensors, graph, args.layers)
            sync(device)
            duration = time.perf_counter() - started
            result["timings"][f"lightgcn_{device}"] = {"layers": args.layers, "step_seconds": duration, "measured_updates": 1}
            print(f"LightGCN {device} ({args.layers} layers, full graph): {duration:.4f} s/update", flush=True)
            del embedding, optimizer, tensors, graph
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
        except (RuntimeError, NotImplementedError, AssertionError) as error:
            result[f"lightgcn_{device}_error"] = str(error).splitlines()[0][:400]
            print(f"LightGCN {device} failed: {result[f'lightgcn_{device}_error']}", flush=True)
    for timing in result["timings"].values():
        step = timing.get("median_step_seconds", timing["step_seconds"])
        timing["extrapolated_epoch_minutes_excluding_evaluation"] = step * result["steps_per_epoch"] / 60
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
