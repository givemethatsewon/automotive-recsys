"""CUDA-only training/validation runner; validation never uses test answers."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import time
import platform
import importlib.metadata
import subprocess

import numpy as np
import polars as pl
import torch

from .audit import fingerprint, write_json
from .evaluate_run import evaluate_embedding, save_evaluation
from .gpu_kernels import encode, full_catalog_topk, normalized_graph, objective, sample_unseen, training_keys


def run_training(data_dir, output, model, epochs=50, seed=42, dim=64, layers=3,
                 batch_size=8192, learning_rate=1e-3, regularization=1e-4,
                 cl_weight=0.1, eps=0.1, temperature=0.2, eval_every=5,
                 patience=5, validation_batch=1024, resume=None,
                 verify_resume=False, save_recommendations=True, variant=None,
                 event_hook=None, run_metadata=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run on Colab; local CPU/MPS fallback is disabled.")
    if epochs < 1 or batch_size < 1 or eval_every < 1 or dim < 1:
        raise ValueError("epochs, batch size, eval interval and dim must be positive.")
    if model not in {"bpr_mf", "lightgcn", "xsimgcl"}:
        raise ValueError("Unknown model.")
    if learning_rate <= 0 or temperature <= 0 or min(regularization, cl_weight, eps, patience) < 0:
        raise ValueError("Invalid optimizer, contrastive loss or patience setting.")
    output, data_dir = Path(output), Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train = pl.read_parquet(data_dir / "train.parquet", columns=["uid", "iid"])
    truth = pl.read_parquet(data_dir / "valid.parquet", columns=["uid", "iid"])
    users = pl.read_parquet(data_dir / "user_map.parquet").sort("uid")
    items = pl.read_parquet(data_dir / "item_map.parquet").sort("iid")
    n_users, n_items = users.height, items.height
    if int(users["n_train"].max()) >= n_items:
        raise ValueError("A user has no available negative item.")
    history_uid, history_iid = train["uid"].to_numpy(), train["iid"].to_numpy()
    gpu_uid = torch.as_tensor(history_uid.astype(np.int64), device="cuda")
    gpu_iid = torch.as_tensor(history_iid.astype(np.int64), device="cuda")
    keys = training_keys(gpu_uid, gpu_iid, n_items, "cuda")
    graph = None if model == "bpr_mf" else normalized_graph(history_uid, history_iid, n_users, n_items, "cuda", "csr")
    weight = torch.nn.Parameter(torch.randn(n_users + n_items, dim, device="cuda") * 0.1)
    optimizer = torch.optim.Adam([weight], lr=learning_rate, fused=True)
    # Keep data sampling independent of XSimGCL noise, so all models see the
    # same order and sampled negatives for a given seed and epoch.
    data_rng = torch.Generator(device="cuda").manual_seed(seed + 1000)
    settings = dict(model=model, variant=variant or model, dim=dim, layers=layers, seed=seed, batch_size=batch_size,
                    learning_rate=learning_rate, regularization=regularization,
                    cl_weight=cl_weight, eps=eps, temperature=temperature, validation_batch=validation_batch)
    inputs = fingerprint(data_dir)
    history, best_score, best_epoch, bad_evaluations, first_epoch = [], -math.inf, 0, 0, 1
    valid_uid = truth["uid"].unique().sort().to_numpy()
    answer_keys = np.sort(truth["uid"].to_numpy().astype(np.int64) * n_items + truth["iid"].to_numpy())
    answer_counts = np.bincount(truth["uid"].to_numpy(), minlength=n_users)[valid_uid]
    group = users["group"].to_numpy()[valid_uid]

    def emit(event, **value):
        if event_hook: event_hook(event, value | {"model": model, "seed": seed, "variant": settings["variant"], "max_epochs": epochs})

    manifest = {"settings": settings, "inputs": inputs, "metadata": run_metadata or {},
                "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "precision": "float32", "tf32": False, "amp": False, "optimizer": "fused Adam", "graph": "fixed symmetric CSR",
                "max_epochs": epochs, "eval_every": eval_every, "patience_evaluations": patience,
                "started_at": datetime.now(timezone.utc).isoformat(), "training_split": "train", "selection_split": "valid"}
    manifest["packages"] = {}
    for package in ["numpy", "polars", "torch", "matplotlib", "google-api-python-client", "google-colab-cli"]:
        try: manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    try: manifest["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError): manifest["nvidia_smi"] = None
    write_json(output / "manifest.json", manifest)

    def save_checkpoint(path, epoch, resumable=True):
        snapshot = {"settings": settings, "inputs": inputs, "epoch": epoch,
                    "embedding": weight.detach(), "format_version": 2,
                    "best_score": best_score, "best_epoch": best_epoch,
                    "bad_evaluations": bad_evaluations, "history": history}
        if resumable:
            snapshot.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), data_rng=data_rng.get_state())
        temporary = path.with_suffix(".tmp")
        torch.save(snapshot, temporary)
        temporary.replace(path)

    def restore(path):
        snapshot = torch.load(path, map_location="cuda", weights_only=True)
        if snapshot["settings"] != settings or snapshot["inputs"] != inputs:
            raise ValueError("Checkpoint configuration or input hashes differ.")
        with torch.no_grad():
            weight.copy_(snapshot["embedding"])
        optimizer.load_state_dict(snapshot["optimizer"])
        torch.set_rng_state(snapshot["torch_rng"].cpu())
        torch.cuda.set_rng_state(snapshot["cuda_rng"].cpu())
        data_rng.set_state(snapshot["data_rng"].cpu())
        return snapshot

    if resume:
        snapshot = restore(Path(resume))
        first_epoch = snapshot["epoch"] + 1
        history, best_score, best_epoch = snapshot["history"], snapshot["best_score"], snapshot["best_epoch"]
        bad_evaluations = snapshot["bad_evaluations"]
        if (output / "history.json").exists():
            persisted = json.loads((output / "history.json").read_text())["history"]
            persisted = [row for row in persisted if row["epoch"] <= snapshot["epoch"]]
            if persisted and persisted[-1]["epoch"] == snapshot["epoch"]: history = persisted
        del snapshot

    def step(ids):
        uid = gpu_uid[ids]
        pos = gpu_iid[ids] + n_users
        negative = sample_unseen(uid, keys, n_items, generator=data_rng) + n_users
        optimizer.zero_grad(set_to_none=True)
        loss, parts = objective(weight, graph, uid, pos, negative, model=model, layers=layers,
                         regularization=regularization, cl_weight=cl_weight, eps=eps,
                         temperature=temperature, symmetric_backward=model != "bpr_mf", return_components=True)
        loss.backward()
        optimizer.step()
        return loss.detach(), parts

    emit("run_started", epoch=first_epoch - 1, best_epoch=best_epoch, best_valid_ndcg=best_score if math.isfinite(best_score) else None)
    for epoch in range(first_epoch, epochs + 1):
        if patience > 0 and bad_evaluations >= patience: break
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        emit("epoch_started", epoch=epoch, step=0, steps=train.height)
        order = torch.randperm(train.height, device="cuda", generator=data_rng)
        total_loss = torch.zeros((), device="cuda")
        totals = {name: torch.zeros((), device="cuda") for name in ["bpr", "cl_raw", "regularization"]}
        last_progress = time.monotonic()
        for start in range(0, train.height, batch_size):
            indices = order[start:start + batch_size]
            loss, parts = step(indices)
            total_loss += loss * len(indices)
            for name in totals: totals[name] += parts[name] * len(indices)
            if time.monotonic() - last_progress > 15:
                emit("training", epoch=epoch, step=min(start + batch_size, train.height), steps=train.height,
                     mean_loss=float(total_loss / min(start + batch_size, train.height)))
                last_progress = time.monotonic()
        torch.cuda.synchronize()
        row = {"epoch": epoch, "training_seconds": time.perf_counter() - epoch_started,
               "mean_loss": float(total_loss / train.height), "cl_weight": cl_weight,
               **{"train_" + name: float(value / train.height) for name, value in totals.items()}}
        row["cl_raw_evaluated"] = model == "xsimgcl" and cl_weight != 0
        if not row["cl_raw_evaluated"]: row["train_cl_raw"] = None
        if not math.isfinite(row["mean_loss"]):
            raise RuntimeError("Non-finite training loss.")
        history.append(row)
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            emit("validating", epoch=epoch)
            validation_started = time.perf_counter()
            evaluated = evaluate_embedding(data_dir, None, weight, graph, settings)
            metrics = evaluated["summary"]["metrics"]
            score = metrics["all"]["ndcg"]["mean"]
            row["validation_seconds"] = time.perf_counter() - validation_started
            row["valid_ndcg_at_10"] = score
            for name, value in metrics["all"].items():
                if isinstance(value, dict) and "mean" in value: row["valid_" + name] = value["mean"]
            row["valid_catalog_coverage"] = metrics["all"]["catalog_coverage"]
            write_json(output / "latest_validation.json", {"split": "valid", "epoch": epoch, "metrics": metrics})
            if score > best_score:
                best_score, best_epoch, bad_evaluations = score, epoch, 0
                write_json(output / "best_validation.json", {"split": "valid", "epoch": epoch, "metrics": metrics})
                save_evaluation(evaluated, output / "valid", save_recommendations)
                save_checkpoint(output / "best.pt", epoch, resumable=False)
                row["is_best"] = True
            else:
                bad_evaluations += 1
            del evaluated
        save_checkpoint(output / "last.pt", epoch)
        row["wall_seconds_with_validation_and_checkpoint"] = time.perf_counter() - epoch_started
        write_json(output / "history.json", {"settings": settings, "history": history})
        (output / "epochs.jsonl").write_text("".join(json.dumps(value) + "\n" for value in history))
        columns = sorted({key for value in history for key in value})
        with (output / "epochs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(history)
        emit("epoch_finished", epoch=epoch, step=train.height, steps=train.height, best_epoch=best_epoch, best_valid_ndcg=best_score, **{"epoch_metrics": row})
        print(json.dumps({"model": model, **row}), flush=True)
        if patience > 0 and bad_evaluations >= patience:
            break

    if not history:
        raise ValueError("No remaining epoch: --epochs must exceed the checkpoint epoch.")
    resume_check = None
    if verify_resume:
        # Verify optimizer moments AND both RNG states by replaying one update.
        restore(output / "last.pt")
        probe = torch.arange(min(batch_size, train.height), device="cuda")
        step(probe)
        expected = weight.detach().clone()
        restore(output / "last.pt")
        step(probe)
        torch.testing.assert_close(weight, expected, rtol=1e-5, atol=2e-6)
        resume_check = {"passed": True, "max_parameter_abs_difference": float((weight.detach() - expected).abs().max())}
        del expected
        restore(output / "last.pt")
    summary = {
        "model": model, "settings": settings, "completed_epochs": history[-1]["epoch"],
        "best_valid_ndcg_at_10": best_score, "best_epoch": best_epoch,
        "history": history, "resume_check": resume_check, "inputs": inputs,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "test_evaluated": False, "training_data": "train only", "selection_data": "valid only",
        "recipe_note": "Common squared raw-embedding L2 across all three models; XSimGCL is a controlled recipe variant.",
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }
    write_json(output / "summary.json", summary)
    emit("run_finished", epoch=summary["completed_epochs"], best_epoch=best_epoch, best_valid_ndcg=best_score)
    del weight, optimizer, graph, keys, gpu_uid, gpu_iid
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["bpr_mf", "lightgcn", "xsimgcl"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--cl-weight", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--validation-batch", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--verify-resume", action="store_true")
    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == "__main__":
    main()
