"""Exact popularity ranking over the whole train catalog, after history masking."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from .audit import fingerprint, write_json
from .evaluation import contains, metrics_at_k, summarize


def rank_popularity(train_counts: np.ndarray, user_ids: np.ndarray, seen: pl.DataFrame, top_n: int) -> np.ndarray:
    n_items = len(train_counts)
    if not 1 <= top_n <= n_items:
        raise ValueError("top_n must be between 1 and the number of items.")
    order = np.lexsort((np.arange(n_items), -train_counts))
    inverse = np.empty(n_items, dtype=np.int32)
    inverse[order] = np.arange(n_items)
    history = seen.sort("uid", "iid")
    max_uid = max(int(user_ids.max()), int(history["uid"].max()))
    degrees = np.bincount(history["uid"].to_numpy(), minlength=max_uid + 1)
    offsets = np.r_[0, np.cumsum(degrees)]
    seen_items = history["iid"].to_numpy()
    result = np.empty((len(user_ids), top_n), dtype=np.int32)
    for row, uid in enumerate(user_ids):
        previous = seen_items[offsets[uid]:offsets[uid + 1]]
        # At most len(previous) candidates can be removed. This prefix is exact.
        prefix_len = min(n_items, top_n + len(previous))
        available = np.ones(prefix_len, dtype=bool)
        ranks = inverse[previous]
        available[ranks[ranks < prefix_len]] = False
        selected = order[:prefix_len][available][:top_n]
        if len(selected) != top_n:
            raise ValueError(f"User {uid} has fewer than {top_n} unseen items.")
        result[row] = selected
    return result


def run_popularity(data_dir: Path, output: Path, split: str, k: int, save_top_n: int, limit_users=None) -> dict:
    started = time.perf_counter()
    if split not in {"valid", "test"}:
        raise ValueError("split must be valid or test")
    if k < 1 or save_top_n < k:
        raise ValueError("save_top_n must be >= k >= 1")
    if limit_users is not None and limit_users < 1:
        raise ValueError("limit_users must be positive")
    output.mkdir(parents=True, exist_ok=True)
    users = pl.read_parquet(data_dir / "user_map.parquet").sort("uid")
    items = pl.read_parquet(data_dir / "item_map.parquet").sort("iid")
    truth = pl.read_parquet(data_dir / f"{split}.parquet", columns=["uid", "iid"])
    seen = pl.read_parquet(data_dir / f"seen_for_{split}.parquet")
    counts = items["n_train"].to_numpy()
    unpopular = ~items["popular"].to_numpy()
    n_items = len(counts)
    user_ids = truth["uid"].unique().sort().to_numpy()
    all_users = len(user_ids)
    if limit_users:
        # Diagnostic only: prefix sampling is explicitly not a benchmark score.
        user_ids = user_ids[:limit_users]
    recommendations = rank_popularity(counts, user_ids, seen, save_top_n)
    pairs = recommendations[:, :k]
    seen_keys = np.sort(seen["uid"].to_numpy().astype(np.int64) * n_items + seen["iid"].to_numpy())
    if contains(seen_keys, user_ids[:, None].astype(np.int64) * n_items + recommendations).any():
        raise ValueError("A recommendation includes a previously seen item.")
    answer_keys = np.sort(truth["uid"].to_numpy().astype(np.int64) * n_items + truth["iid"].to_numpy())
    if contains(seen_keys, answer_keys).any():
        raise ValueError("An answer is masked by seen history.")
    answer_counts = np.bincount(truth["uid"].to_numpy(), minlength=users.height)[user_ids]
    values = metrics_at_k(user_ids, pairs, answer_keys, answer_counts, unpopular, counts)
    groups = users["group"].to_numpy()[user_ids]
    summaries = {"all": summarize(values, pairs, n_items)}
    for group in ["1-5", "6+"]:
        mask = groups == group
        if mask.any():
            summaries[group] = summarize({key: value[mask] for key, value in values.items()}, pairs[mask], n_items)
    stats = json.loads((data_dir / "stats.json").read_text())
    summary = {
        "model": "popularity",
        "split": split,
        "k": k,
        "save_top_n": save_top_n,
        "full_eligible_population": len(user_ids) == all_users,
        "eligible_users": all_users,
        "evaluated_users": len(user_ids),
        "evaluated_interactions": int(answer_counts.sum()),
        "retained_interaction_pct": 100 * truth.height / stats["original"][split]["rows"],
        "candidate_items": n_items,
        "candidate_protocol": "all train items minus seen_for_split; no sampled negatives",
        "training_data": "train only; no train+valid retraining",
        "history_mask": f"seen_for_{split}.parquet",
        "tie_break": "lower iid first",
        "aggregation": "macro mean over eligible users with at least one unmasked answer",
        "ci_note": "Normal approximation for per-user mean; not seed uncertainty or a paired model comparison.",
        "mismatch_at_k": None,
        "mismatch_note": "Vehicle compatibility features have not been supplied.",
        "metrics": summaries,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": fingerprint(data_dir),
    }
    np.savez_compressed(output / "recommendations.npz", uid=user_ids, iid=recommendations)
    pl.DataFrame({"uid": user_ids, "group": groups, "answer_count": answer_counts, **values}).write_parquet(output / "per_user.parquet")
    write_json(output / "metrics.json", summary)
    return summary
