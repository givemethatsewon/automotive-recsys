"""Read-only checks of the handed-off data contract; no source data is modified."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fingerprint(data_dir: Path) -> dict:
    """Hash actual inputs so a rerun can detect a changed handoff."""
    result = {}
    for path in sorted(data_dir.iterdir()):
        if path.suffix not in {".parquet", ".json", ".txt"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        result[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def audit(data_dir: Path) -> dict:
    stats = json.loads((data_dir / "stats.json").read_text())
    users = pl.read_parquet(data_dir / "user_map.parquet").sort("uid")
    items = pl.read_parquet(data_dir / "item_map.parquet").sort("iid")
    checks = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for name, frame, key, original in [
        ("user_map", users, "uid", "user_id"),
        ("item_map", items, "iid", "parent_asin"),
    ]:
        check(f"{name}: no nulls", sum(frame.null_count().row(0)) == 0)
        check(f"{name}: original IDs unique", frame[original].n_unique() == frame.height)
        check(f"{name}: contiguous IDs", np.array_equal(frame[key].to_numpy(), np.arange(frame.height)))

    frames = {}
    profiles = {}
    for split in ["train", "valid", "test"]:
        frame = pl.read_parquet(data_dir / f"{split}.parquet")
        frames[split] = frame
        low, high = frame["timestamp"].min(), frame["timestamp"].max()
        profile = {
            "rows": frame.height,
            "users": frame["uid"].n_unique(),
            "items": frame["iid"].n_unique(),
            "schema": {key: str(value) for key, value in frame.schema.items()},
            "null_counts": dict(zip(frame.columns, frame.null_count().row(0))),
            "duplicate_pairs": frame.height - frame.select("uid", "iid").unique().height,
            "timestamp_min_ms": low,
            "timestamp_max_ms": high,
            "timestamp_min_utc": datetime.fromtimestamp(low / 1000, timezone.utc).isoformat(),
            "timestamp_max_utc": datetime.fromtimestamp(high / 1000, timezone.utc).isoformat(),
            "retained_interaction_pct": 100 * frame.height / stats["original"][split]["rows"],
        }
        profiles[split] = profile
        check(f"{split}: counts match stats.json", all(
            profile[key] == stats["processed"][split][key] for key in ["rows", "users", "items"]
        ))
        check(f"{split}: schema", frame.schema == {"uid": pl.Int32, "iid": pl.Int32, "timestamp": pl.Int64})
        check(f"{split}: no nulls", sum(profile["null_counts"].values()) == 0)
        check(f"{split}: no duplicate user-item pairs", profile["duplicate_pairs"] == 0, profile["duplicate_pairs"])
        check(f"{split}: IDs in train vocabulary", (
            frame["uid"].min() >= 0 and frame["uid"].max() < users.height
            and frame["iid"].min() >= 0 and frame["iid"].max() < items.height
        ))
        check(f"{split}: sorted by user and time", frame.equals(frame.sort("uid", "timestamp")))

    train = frames["train"]
    for frame, key, name in [(users, "uid", "user_map"), (items, "iid", "item_map")]:
        counts = train.group_by(key).len().sort(key)["len"].to_numpy()
        check(f"{name}: n_train matches interactions", np.array_equal(counts, frame["n_train"].to_numpy()))
    expected_group = np.where(users["n_train"].to_numpy() <= 5, "1-5", "6+")
    check("user groups match training counts", np.array_equal(users["group"].to_numpy(), expected_group))

    counts = items["n_train"].to_numpy()
    sorted_counts = np.sort(counts)[::-1]
    boundary = np.searchsorted(np.cumsum(sorted_counts), 0.8 * counts.sum())
    threshold = int(sorted_counts[boundary])
    popular = counts >= threshold
    check("popularity threshold matches training-only 80% rule", threshold == stats["popular_min_reviews"], threshold)
    check("popular flags match threshold including ties", np.array_equal(popular, items["popular"].to_numpy()))

    check("train strictly precedes validation", profiles["train"]["timestamp_max_ms"] < profiles["valid"]["timestamp_min_ms"])
    check("validation strictly precedes test", profiles["valid"]["timestamp_max_ms"] < profiles["test"]["timestamp_min_ms"])
    for split, expected in [
        ("valid", train.select("uid", "iid")),
        ("test", pl.concat([train, frames["valid"]]).select("uid", "iid").unique()),
    ]:
        seen = pl.read_parquet(data_dir / f"seen_for_{split}.parquet")
        check(f"seen_for_{split}: no nulls", sum(seen.null_count().row(0)) == 0)
        check(f"seen_for_{split}: unique pairs", seen.unique().height == seen.height)
        missing = expected.join(seen, on=["uid", "iid"], how="anti").height
        extra = seen.join(expected, on=["uid", "iid"], how="anti").height
        overlap = frames[split].join(seen, on=["uid", "iid"], how="semi").height
        check(f"seen_for_{split}: matches expected history", missing == 0 and extra == 0, {"missing": missing, "extra": extra})
        check(f"{split}: no answer masked by seen history", overlap == 0, overlap)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(data_dir.resolve()),
        "inputs": fingerprint(data_dir),
        "profiles": profiles,
        "training_population": {"users": users.height, "items": items.height},
        "train_degree_min": {"user": users["n_train"].min(), "item": items["n_train"].min()},
        "popular_min_reviews": threshold,
        "popular_items": int(popular.sum()),
        "popular_review_share": float(counts[popular].sum() / counts.sum()),
        "vehicle_features_present": False,
        "vehicle_features_note": "Supplied files contain interactions and ID maps only; no vehicle/title/text tables.",
        "timestamp_note": "Use supplied split files; README calendar dates are approximate, not new midnight cutoffs.",
        "checks": checks,
        "passed": all(row["passed"] for row in checks),
        "limitations": [
            "Source 5-core generation cannot be independently reconstructed from this handoff.",
            "Original cold-start rows already removed; retained fractions use stats.json source totals.",
            "Per-split training degrees need not remain >=5 after time splitting; do not reapply 5-core.",
            "Vehicle/text extraction quality and time leakage cannot be checked until those tables arrive.",
        ],
    }
