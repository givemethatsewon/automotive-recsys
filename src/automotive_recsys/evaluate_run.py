"""Shared complete validation/test artifacts, with frozen train popularity."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import polars as pl

from .audit import fingerprint, write_json
from .evaluation import contains, metrics_at_k, summarize


def evaluate_rankings(data_dir, split, uid, iid, score, metadata=None):
    data_dir = Path(data_dir)
    truth = pl.read_parquet(data_dir / f"{split}.parquet", columns=["uid", "iid"])
    seen = pl.read_parquet(data_dir / f"seen_for_{split}.parquet").sort("uid", "iid")
    users = pl.read_parquet(data_dir / "user_map.parquet").sort("uid")
    items = pl.read_parquet(data_dir / "item_map.parquet").sort("iid")
    expected = truth["uid"].unique().sort().to_numpy()
    if not np.array_equal(uid, expected): raise ValueError("Evaluation user cohort differs from the complete eligible split.")
    if iid.shape != score.shape or iid.shape[0] != len(uid) or iid.shape[1] < 10: raise ValueError("Invalid ranked ID/score shapes.")
    if not np.isfinite(score).all(): raise ValueError("Recommendation scores contain non-finite values.")
    n_items = items.height
    if np.any(iid < 0) or np.any(iid >= n_items): raise ValueError("Invalid item ID in ranked output.")
    keys = np.sort(truth["uid"].to_numpy().astype(np.int64) * n_items + truth["iid"].to_numpy())
    seen_keys = np.sort(seen["uid"].to_numpy().astype(np.int64) * n_items + seen["iid"].to_numpy())
    recommended_keys = uid[:, None].astype(np.int64) * n_items + iid
    if contains(seen_keys, recommended_keys).any() or contains(seen_keys, keys).any(): raise ValueError("History masking or answer eligibility failed.")
    if np.any(np.diff(np.sort(iid, axis=1), axis=1) == 0): raise ValueError("Repeated item in ranked output.")
    counts = np.bincount(truth["uid"].to_numpy(), minlength=users.height)[uid]
    unpopular = ~items["popular"].to_numpy()
    values = metrics_at_k(uid, iid[:, :10], keys, counts, unpopular, items["n_train"].to_numpy())
    values["candidate_recall100"] = contains(keys, recommended_keys).sum(1) / counts
    tail_truth = truth.filter(pl.col("iid").is_in(items.filter(~pl.col("popular"))["iid"].implode()))
    tail_keys = np.sort(tail_truth["uid"].to_numpy().astype(np.int64) * n_items + tail_truth["iid"].to_numpy())
    tail_counts = np.bincount(tail_truth["uid"].to_numpy(), minlength=users.height)[uid]
    tail_recall = np.full(len(uid), np.nan)
    has_tail = tail_counts > 0
    tail_recall[has_tail] = contains(tail_keys, recommended_keys[:, :10])[has_tail].sum(1) / tail_counts[has_tail]
    group = users["group"].to_numpy()[uid]
    summary = {}
    for name in ["all", "1-5", "6+"]:
        mask = np.ones(len(uid), bool) if name == "all" else group == name
        if mask.any():
            entry = summarize({key: value[mask] for key, value in values.items()}, iid[mask, :10], n_items)
            entry["ground_truth_interactions"] = int(counts[mask].sum())
            tail_mask = mask & has_tail
            entry["tail_recall10"] = {"mean": float(tail_recall[tail_mask].mean()) if tail_mask.any() else None, "eligible_users": int(tail_mask.sum())}
            summary[name] = entry
    exposure = np.bincount(iid[:, :10].ravel(), minlength=n_items)
    exposure_table = items.select("iid", "n_train", "popular").with_columns(pl.Series("recommendation_count_top10", exposure), pl.Series("recommendation_share_top10", exposure / exposure.sum()))
    frame = pl.DataFrame({"uid": uid, "group": group, "n_train": users["n_train"].to_numpy()[uid], "n_gt": counts, "n_tail_gt": tail_counts, **values, "tail_recall10": tail_recall}).with_columns(pl.col("tail_recall10").fill_nan(None))
    report = {"split": split, "users": len(uid), "ground_truth_interactions": truth.height, "candidate_items": n_items,
              "top_n": iid.shape[1], "metrics": summary, "metadata": metadata or {},
              "protocol": "all train items minus seen_for_split; score descending, IID ascending ties; macro user means",
              "created_at": datetime.now(timezone.utc).isoformat()}
    return {"summary": report, "per_user": frame, "exposure": exposure_table, "uid": uid, "iid": iid, "score": score}


def save_evaluation(result, output, save_predictions=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result["summary"])
    result["per_user"].write_parquet(output / "per_user.parquet")
    result["exposure"].write_parquet(output / "item_exposure.parquet")
    if save_predictions:
        np.savez_compressed(output / "recommendations.npz", uid=result["uid"], iid=result["iid"], score=result["score"])


def evaluate_embedding(data_dir, output, embedding, graph, settings, split="valid", save_predictions=True):
    from .gpu_kernels import encode, full_catalog_topk
    import torch
    data_dir = Path(data_dir)
    truth = pl.read_parquet(data_dir / f"{split}.parquet", columns=["uid"])
    seen = pl.read_parquet(data_dir / f"seen_for_{split}.parquet").sort("uid", "iid")
    n_users = pl.scan_parquet(data_dir / "user_map.parquet").select(pl.len()).collect().item()
    uid = truth["uid"].unique().sort().to_numpy()
    with torch.inference_mode():
        representation, _ = encode(embedding, graph, settings["model"], layers=settings["layers"], perturbed=False, symmetric_backward=settings["model"] != "bpr_mf")
        iid, score = full_catalog_topk(representation, uid, n_users, seen["uid"].to_numpy(), seen["iid"].to_numpy(), k=min(100, embedding.shape[0] - n_users), batch_size=settings.get("validation_batch", 1024), return_scores=True)
    result = evaluate_rankings(data_dir, split, uid, iid, score, settings)
    if output: save_evaluation(result, output, save_predictions)
    return result


def evaluate_checkpoint(data_dir, checkpoint, output, selection, save_predictions=True):
    import torch
    from .gpu_kernels import normalized_graph
    if not torch.cuda.is_available(): raise RuntimeError("Final evaluation requires Colab CUDA; no local fallback.")
    snapshot = torch.load(checkpoint, map_location="cuda", weights_only=True)
    settings = snapshot["settings"]
    chosen = selection["choices"][settings.get("variant", settings["model"])]
    if selection.get("test_used_for_selection") is not False or selection.get("selection_data") != "valid only":
        raise ValueError("Test evaluation requires an explicitly frozen validation-only selection.")
    for key in ["model", "dim", "layers", "learning_rate", "regularization", "cl_weight", "eps", "temperature"]:
        if settings[key] != chosen[key]: raise ValueError("Test evaluation setting differs from frozen validation selection.")
    if fingerprint(Path(data_dir)) != snapshot["inputs"]: raise ValueError("Evaluation input hashes differ.")
    train = pl.read_parquet(Path(data_dir) / "train.parquet", columns=["uid", "iid"])
    n_users = pl.scan_parquet(Path(data_dir) / "user_map.parquet").select(pl.len()).collect().item()
    n_items = snapshot["embedding"].shape[0] - n_users
    graph = None if settings["model"] == "bpr_mf" else normalized_graph(train["uid"].to_numpy(), train["iid"].to_numpy(), n_users, n_items, "cuda", "csr")
    return evaluate_embedding(data_dir, output, snapshot["embedding"], graph, settings | {"checkpoint_epoch": snapshot["epoch"]}, "test", save_predictions)
