"""Exploratory vehicle matching. Unknown fitment never implies a mismatch.

Rows in the compact arrays are (make_id, model_id), with -1 for unknown.
This pilot deliberately does not score year compatibility.
"""
import numpy as np


def match_candidates(uid, iid, user_ptr, user_rows, item_ptr, item_rows, universal, level):
    """Return -1 unknown, 0 inferred mismatch, 1 match; optionally JIT on Colab.

    Multiple vehicles use any-match semantics. At model level a shared make
    with a missing model is unresolved, not a known mismatch. Universal items
    are always unknown, including when their title also names a vehicle.
    """
    state = np.full(iid.shape, -1, dtype=np.int8)
    for row in range(len(uid)):
        u = uid[row]
        for col in range(iid.shape[1]):
            item = iid[row, col]
            if universal[item] or user_ptr[u] == user_ptr[u + 1] or item_ptr[item] == item_ptr[item + 1]:
                continue
            known = False
            unresolved = False
            matched = False
            for a in range(user_ptr[u], user_ptr[u + 1]):
                umake, umodel = user_rows[a, 0], user_rows[a, 1]
                for b in range(item_ptr[item], item_ptr[item + 1]):
                    pmake, pmodel = item_rows[b, 0], item_rows[b, 1]
                    if umake < 0 or pmake < 0:
                        unresolved = True
                        continue
                    known = True
                    if umake != pmake:
                        continue
                    if level == 0:
                        matched = True
                    elif umodel < 0 or pmodel < 0:
                        unresolved = True
                    elif umodel == pmodel:
                        matched = True
            if matched:
                state[row, col] = 1
            elif known and not unresolved:
                state[row, col] = 0
    return state


def stable_rerank(iid, score, match, beta):
    """Small CPU reference. Production pilot uses float64 CUDA sorting.

    Beta is in units of the user's top-100 score range. A zero score range
    uses 1.0. Normalization is only used to set the additive bonus; sorting
    retains the original float64 score to preserve beta=0 ranks exactly.
    """
    score = score.astype(np.float64)
    scale = score.max(1) - score.min(1)
    scale[scale == 0] = 1.0
    adjusted = score + beta * scale[:, None] * (match == 1)
    order = np.lexsort((iid, -adjusted), axis=1)
    return np.take_along_axis(iid, order, axis=1), order
