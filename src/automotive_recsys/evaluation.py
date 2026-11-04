"""Shared binary relevance metrics, averaged equally over eligible users."""

import numpy as np


def contains(sorted_keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if len(sorted_keys) == 0:
        return np.zeros(queries.shape, dtype=bool)
    positions = np.searchsorted(sorted_keys, queries)
    return (positions < len(sorted_keys)) & (sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] == queries)


def user_metrics(recommendations, relevant, unpopular, train_counts):
    """Relevant is a boolean matrix; answer_counts is supplied separately by caller."""
    k = recommendations.shape[1]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    return {
        "dcg": relevant @ discounts,
        "hits": relevant.sum(axis=1),
        "aplt": unpopular[recommendations].mean(axis=1),
        "arp": train_counts[recommendations].mean(axis=1),
    }


def metrics_at_k(user_ids, recommendations, answer_keys, answer_counts, unpopular, train_counts):
    n_items = len(train_counts)
    if recommendations.ndim != 2 or len(user_ids) != len(recommendations) or recommendations.shape[1] == 0:
        raise ValueError("Expected one nonempty ranked list for each user.")
    if np.any(recommendations < 0) or np.any(recommendations >= n_items):
        raise ValueError("Invalid item ID in recommendations.")
    if np.any(np.diff(np.sort(recommendations, axis=1), axis=1) == 0):
        raise ValueError("Recommendations must contain unique items per user.")
    if np.any(answer_counts <= 0):
        raise ValueError("Every evaluated user needs at least one relevant answer.")
    queries = user_ids[:, None].astype(np.int64) * n_items + recommendations
    relevant = contains(answer_keys, queries)
    values = user_metrics(recommendations, relevant, unpopular, train_counts)
    k = recommendations.shape[1]
    cumulative_discounts = np.r_[0.0, np.cumsum(1 / np.log2(np.arange(2, k + 2)))]
    return {
        "ndcg": values["dcg"] / cumulative_discounts[np.minimum(answer_counts, k)],
        "recall": values["hits"] / answer_counts,
        "hit_rate": (values["hits"] > 0).astype(float),
        "aplt": values["aplt"],
        "arp": values["arp"],
    }


def summarize(user_values, recommendations, n_items):
    n = len(recommendations)
    if n == 0:
        raise ValueError("Cannot summarize zero users.")
    summary = {"users": n}
    for name, values in user_values.items():
        mean = float(np.mean(values))
        error = float(1.96 * np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None
        low = mean - error if error is not None else None
        high = mean + error if error is not None else None
        if name != "arp" and error is not None:
            low, high = max(0.0, low), min(1.0, high)
        summary[name] = {"mean": mean, "user_mean_normal_ci95": [low, high]}
    summary["catalog_coverage"] = float(np.unique(recommendations).size / n_items)
    return summary
