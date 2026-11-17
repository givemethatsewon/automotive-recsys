"""Compact adapter for the supplied v2 fitment rules; no fitting or I/O.

State convention follows vehicle_rerank: -1 unknown, 0 mismatch, 1 match.
The provider's model canonicalization/compatibility functions are supplied by
the caller and archived unchanged alongside each experiment.
"""
import numpy as np


def relative_ndcg_change(delta, baseline):
    """A zero baseline has no defined relative change; retain absolute delta."""
    return float(delta / baseline) if baseline > 0 else None


def compact_v2(user_rows, item_rows, n_users, n_items, canon, model_ok):
    # Input tuples: user id/make/model/segment/year; item additionally year_to/open_end.
    users, items = list(user_rows), list(item_rows)
    canonical = [canon(r[1], r[2]) for r in users + items]
    makes = {v: i for i, v in enumerate(sorted({m for m, _ in canonical}))}
    models = {None: 0}
    for _, model in canonical:
        if model not in models: models[model] = len(models)
    compatible = np.array([[model_ok(a, b) for b in models] for a in models], dtype=np.bool_)

    def segment(value):
        if not isinstance(value, str) or not value: return 3
        return (1 if 'car' in value.split('|') else 0) | (2 if 'powersports' in value.split('|') else 0)

    def year(value):
        return -1 if value is None or not np.isfinite(value) else int(value)

    def pack(rows, entries, n, item=False):
        result = []
        for r, (make, model) in zip(rows, entries):
            lo = year(r[4])
            hi = year(r[5]) if item else lo
            if item and hi < 0: hi = 9999 if r[6] else lo
            result.append((int(r[0]), makes[make], models[model], segment(r[3]), lo, hi))
        result = sorted(set(result))
        ids = np.array([r[0] for r in result], dtype=np.int64)
        ptr = np.r_[0, np.cumsum(np.bincount(ids, minlength=n))].astype(np.int64)
        values = np.array([r[1:] for r in result], dtype=np.int64).reshape(-1, 5)
        return ptr, values

    up, uv = pack(users, canonical[:len(users)], n_users)
    ip, pf = pack(items, canonical[len(users):], n_items, True)
    return up, uv, ip, pf, compatible


def match_v2(uid, iid, user_ptr, users, item_ptr, items, universal, compatible, use_year):
    """Provider semantics: unknown model passes; universal matching can be +1."""
    state = np.full(iid.shape, -1, dtype=np.int8)
    for row in range(len(uid)):
        u = uid[row]
        for col in range(iid.shape[1]):
            i = iid[row, col]
            if user_ptr[u] == user_ptr[u+1] or item_ptr[i] == item_ptr[i+1]: continue
            state[row, col] = -1 if universal[i] else 0
            matched = False
            for a in range(user_ptr[u], user_ptr[u+1]):
                for b in range(item_ptr[i], item_ptr[i+1]):
                    uv, pf = users[a], items[b]
                    if uv[0] != pf[0] or not (uv[2] & pf[2]) or not compatible[uv[1], pf[1]]: continue
                    if use_year and uv[3] >= 0 and pf[3] >= 0 and not pf[3] <= uv[3] <= pf[4]: continue
                    matched = True
                    break
                if matched: break
            if matched: state[row, col] = 1
    return state
