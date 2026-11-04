"""Actual-data CUDA performance/correctness check; not model selection or fitting."""
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile
import time

ROOT = Path("/content/automotive-recsys")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl
import torch
from automotive_recsys.audit import write_json
from automotive_recsys.gpu_kernels import (
    deterministic_topk, full_catalog_topk, key_membership, normalized_graph, objective,
    sample_unseen, training_keys,
)


def verify_cuda_math():
    torch.manual_seed(42)
    cpu_graph = normalized_graph([0, 0, 1], [0, 1, 1], 2, 2, layout="coo")
    initial = torch.randn(4, 8) * 0.1
    users, pos, neg = torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([3, 2])
    noise = [torch.rand(4, 8) for _ in range(3)]
    results = []
    for model in ["bpr_mf", "lightgcn", "xsimgcl"]:
        reference = initial.clone().requires_grad_()
        expected = objective(reference, cpu_graph, users, pos, neg, model=model, noise=noise)
        expected.backward()
        for layout, symmetric in [("coo", False), ("csr", False), ("csr", True)]:
            parameter = initial.cuda().requires_grad_()
            graph = cpu_graph.cuda()
            if layout == "csr":
                graph = graph.to_sparse_csr()
            actual = objective(parameter, graph, users.cuda(), pos.cuda(), neg.cuda(), model=model, noise=[value.cuda() for value in noise], symmetric_backward=symmetric)
            actual.backward()
            torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(parameter.grad.cpu(), reference.grad, rtol=2e-3, atol=2e-6)
            results.append({"model": model, "layout": layout, "symmetric_backward": symmetric, "gradient_max_abs_difference": float((parameter.grad.cpu() - reference.grad).abs().max())})
    left, right = torch.nn.Parameter(initial.cuda()), torch.nn.Parameter(initial.cuda())
    eager = torch.optim.Adam([left], lr=1e-3, fused=False)
    fused = torch.optim.Adam([right], lr=1e-3, fused=True)
    for _ in range(3):
        for weight, optimizer in [(left, eager), (right, fused)]:
            optimizer.zero_grad(set_to_none=True)
            weight.square().sum().backward()
            optimizer.step()
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    tied_scores = torch.randint(0, 5, (16, 64), device="cuda").float()
    selected = deterministic_topk(tied_scores, 10).cpu().numpy()
    host_scores = tied_scores.cpu().numpy()
    expected_ids = np.array([sorted(range(64), key=lambda iid: (-row[iid], iid))[:10] for row in host_scores])
    np.testing.assert_array_equal(selected, expected_ids)
    return results


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = ROOT / "results/gpu_performance"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "dtype": "float32", "tf32": False, "amp": False, "dim": 64,
        "layers": 3, "batch_size": 8192, "warmup_updates": 2, "measured_updates": 7,
        "cuda_math_checks": verify_cuda_math(), "timings": [],
        "recipe_note": "Shared raw embedding squared L2 for all models; XSimGCL's published repository uses a different regularizer. This is a controlled recipe variant, not a verbatim paper reproduction.",
    }
    train = pl.read_parquet(ROOT / "data/train.parquet", columns=["uid", "iid"])
    n_users = int(train["uid"].max()) + 1
    n_items = int(train["iid"].max()) + 1
    n_nodes = n_users + n_items
    report["train_interactions"] = train.height
    report["updates_per_epoch"] = math.ceil(train.height / report["batch_size"])
    graphs = {"coo": normalized_graph(train["uid"].to_numpy(), train["iid"].to_numpy(), n_users, n_items, "cuda", "coo")}
    graphs["csr"] = graphs["coo"].to_sparse_csr()
    gpu_users = torch.as_tensor(train["uid"].to_numpy().astype(np.int64), device="cuda")
    gpu_items = torch.as_tensor(train["iid"].to_numpy().astype(np.int64), device="cuda")
    keys = training_keys(gpu_users, gpu_items, n_items, "cuda")
    torch.manual_seed(42)
    batch_indices = torch.randint(len(gpu_users), (report["batch_size"],), device="cuda")
    users = gpu_users[batch_indices]
    positives = gpu_items[batch_indices] + n_users
    probe = sample_unseen(users, keys, n_items)
    assert not bool(key_membership(keys, users * n_items + probe).any())
    report["negative_sampler_checked"] = True
    del probe
    initial = torch.randn(n_nodes, 64, device="cuda") * 0.1

    # Full-size parity: identical parameters, positive/negative batch and XSimGCL
    # noise for native CSR autograd versus fixed symmetric backward.
    report["full_graph_backward_checks"] = []
    fixed_negatives = sample_unseen(users, keys, n_items) + n_users
    fixed_noise = [torch.rand_like(initial) for _ in range(3)]
    for model in ["lightgcn", "xsimgcl"]:
        reference = initial.clone().requires_grad_()
        expected_loss = objective(reference, graphs["csr"], users, positives, fixed_negatives, model=model, noise=fixed_noise)
        expected_loss.backward()
        expected_gradient = reference.grad.detach().clone()
        del reference
        candidate = initial.clone().requires_grad_()
        actual_loss = objective(candidate, graphs["csr"], users, positives, fixed_negatives, model=model, noise=fixed_noise, symmetric_backward=True)
        actual_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(candidate.grad, expected_gradient, rtol=2e-3, atol=2e-7)
        record = {"model": model, "gradient_max_abs_difference": float((candidate.grad - expected_gradient).abs().max()), "loss_absolute_difference": float((actual_loss.detach() - expected_loss.detach()).abs())}
        report["full_graph_backward_checks"].append(record)
        print("FULL_GRAPH_PARITY " + json.dumps(record), flush=True)
        del expected_gradient, candidate, actual_loss, expected_loss
        gc.collect()
        torch.cuda.empty_cache()
    del fixed_noise, fixed_negatives

    for model in ["bpr_mf", "lightgcn", "xsimgcl"]:
        variants = [("coo", False, False), ("coo", True, False)]
        if model != "bpr_mf":
            variants.extend([("csr", True, False), ("csr", True, True)])
        for layout, fused, symmetric in variants:
            torch.manual_seed(42)
            parameter = torch.nn.Parameter(initial.clone())
            optimizer = torch.optim.Adam([parameter], lr=1e-3, fused=fused)

            def step():
                negatives = sample_unseen(users, keys, n_items) + n_users
                optimizer.zero_grad(set_to_none=True)
                loss = objective(parameter, graphs[layout], users, positives, negatives, model=model, symmetric_backward=symmetric)
                loss.backward()
                optimizer.step()
                return loss.detach()

            for _ in range(report["warmup_updates"]):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            events = []
            started = time.perf_counter()
            for _ in range(report["measured_updates"]):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                loss = step()
                end.record()
                events.append((begin, end))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert bool(torch.isfinite(loss))
            seconds = elapsed / report["measured_updates"]
            record = {
                "model": model, "layout": layout, "fused_adam": fused,
                "symmetric_backward": symmetric,
                "wall_seconds_per_update": seconds,
                "cuda_event_median_seconds": statistics.median(a.elapsed_time(b) / 1000 for a, b in events),
                "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                "extrapolated_training_seconds_per_epoch": seconds * report["updates_per_epoch"],
            }
            report["timings"].append(record)
            print(json.dumps(record), flush=True)
            del parameter, optimizer, events, loss
            gc.collect()
            torch.cuda.empty_cache()

    # Verify all-item scoring and history masking independently before timing it.
    valid = pl.read_parquet(ROOT / "data/valid.parquet", columns=["uid", "iid"])
    valid_users = valid["uid"].unique().sort().to_numpy()
    history_uid, history_iid = train["uid"].to_numpy(), train["iid"].to_numpy()
    selected = valid_users[:8]
    observed = full_catalog_topk(initial, selected, n_users, history_uid, history_iid, k=10, batch_size=8)
    cpu = initial.cpu()
    scores = cpu[torch.as_tensor(selected.astype(np.int64))] @ cpu[n_users:].T
    for row, uid in enumerate(selected):
        seen = history_iid[history_uid == uid]
        scores[row, torch.as_tensor(seen.astype(np.int64))] = -torch.inf
    expected = torch.topk(scores, 10, dim=1).indices.numpy()
    np.testing.assert_array_equal(observed, expected)
    report["full_catalog_cpu_gpu_top10_match"] = True
    del cpu, scores, expected
    report["validation_timings"] = []
    for batch_size in [128, 512, 1024]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictions = full_catalog_topk(initial, valid_users, n_users, history_uid, history_iid, k=100, batch_size=batch_size)
        torch.cuda.synchronize()
        record = {"user_batch_size": batch_size, "users": len(valid_users), "candidate_items": n_items,
                  "seconds": time.perf_counter() - started, "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20}
        report["validation_timings"].append(record)
        print("VALIDATION " + json.dumps(record), flush=True)
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

    # Obtain a real CUDA trace, separately from the timed measurements.
    best_light = min((row for row in report["timings"] if row["model"] == "lightgcn"), key=lambda row: row["wall_seconds_per_update"])
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([parameter], lr=1e-3, fused=best_light["fused_adam"])
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as profiler:
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                negative = sample_unseen(users, keys, n_items) + n_users
                objective(parameter, graphs[best_light["layout"]], users, positives, negative,
                          symmetric_backward=best_light["symmetric_backward"]).backward()
                optimizer.step()
                profiler.step()
            torch.cuda.synchronize()
        events = [event for event in profiler.events() if "CUDA" in str(event.device_type)]
        report["profiler_cuda_events"] = len(events)
        report["profiler_verified_cuda"] = len(events) > 0
        profiler.export_chrome_trace(str(output / "cuda-trace.json"))
        averages = profiler.key_averages()
        names = sorted(averages, key=lambda event: getattr(event, "self_device_time_total", 0), reverse=True)[:12]
        report["profile_top_ops"] = [{"name": event.key, "self_device_microseconds": getattr(event, "self_device_time_total", 0), "calls": event.count} for event in names]
    except (RuntimeError, AssertionError) as error:
        report["profiler_verified_cuda"] = False
        report["profiler_error"] = str(error).splitlines()[0][:400]
    validation_seconds = min(row["seconds"] for row in report["validation_timings"])
    report["chosen_kernels"] = {
        model: min((row for row in report["timings"] if row["model"] == model), key=lambda row: row["wall_seconds_per_update"])
        for model in ["bpr_mf", "lightgcn", "xsimgcl"]
    }
    report["time_scenario"] = {
        "epochs": 50, "seeds": 3, "validation_every_epochs": 5,
        "final_training_runs": 9,
        "final_runs_hours": sum(3 * (50 * row["extrapolated_training_seconds_per_epoch"] + 10 * validation_seconds) for row in report["chosen_kernels"].values()) / 3600,
        "small_search_hours": sum(trials * (10 * report["chosen_kernels"][model]["extrapolated_training_seconds_per_epoch"] + 2 * validation_seconds) for model, trials in [("bpr_mf", 2), ("lightgcn", 6), ("xsimgcl", 4)]) / 3600,
        "limits": "Short fixed-minibatch extrapolation, not measured full epochs or convergence. Validation timing includes full-catalog scoring/masking/top-k/copy but excludes embedding propagation, metric aggregation and checkpoint I/O. Search counts and epoch counts are a proposed budget, not empirical convergence requirements.",
    }
    report["passed"] = True
    write_json(output / "summary.json", report)
    with tarfile.open("/content/automotive-recsys-perf-results.tar.gz", "w:gz") as bundle:
        bundle.add(output, arcname="gpu_performance")
    print("TIME_SCENARIO " + json.dumps(report["time_scenario"]), flush=True)
    print("Performance and CUDA correctness checks complete.", flush=True)


if __name__ == "__main__":
    main()
