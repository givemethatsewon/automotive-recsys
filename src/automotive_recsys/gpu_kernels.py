"""GPU learning kernels shared by performance checks and the future trainer.

No training loop, tuning or convergence claim is made by this module.
Graph/embedding dtype stays float32; approximate matmul/AMP is not enabled here.
"""

import torch
from torch.nn import functional as F


def normalized_graph(uid, iid, n_users, n_items, device="cpu", layout="csr"):
    """Train-only symmetric D^-1/2 A D^-1/2, prepared once per run."""
    user = torch.as_tensor(uid, dtype=torch.long, device="cpu")
    item = torch.as_tensor(iid, dtype=torch.long, device="cpu") + n_users
    rows, cols = torch.cat([user, item]), torch.cat([item, user])
    n_nodes = n_users + n_items
    degree = torch.bincount(rows, minlength=n_nodes).float()
    values = torch.rsqrt(degree[rows] * degree[cols])
    graph = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n_nodes, n_nodes)).coalesce()
    graph = graph.to(device)
    return graph.to_sparse_csr() if layout == "csr" else graph


class _FixedSymmetricMM(torch.autograd.Function):
    """A^T=A: reuse fixed adjacency in backward, avoiding CSR transpose sorting.

    Only for the fixed, symmetric graph produced by normalized_graph. This is
    not suitable for trainable edge weights or directed/nonsymmetric graphs.
    """

    @staticmethod
    def forward(ctx, features, adjacency):
        if adjacency.requires_grad:
            raise ValueError("The symmetric backward requires fixed edge weights.")
        ctx.save_for_backward(adjacency)
        return torch.sparse.mm(adjacency, features)

    @staticmethod
    def backward(ctx, output_gradient):
        adjacency, = ctx.saved_tensors
        return torch.sparse.mm(adjacency, output_gradient), None


def encode(embedding, graph, model, layers=3, eps=0.1, contrast_layer=1, perturbed=False, noise=None,
           symmetric_backward=False):
    if model == "bpr_mf":
        return embedding, None
    if model not in {"lightgcn", "xsimgcl"} or layers < 1:
        raise ValueError("Expected lightgcn/xsimgcl with at least one layer.")
    if model == "xsimgcl" and not 1 <= contrast_layer <= layers:
        raise ValueError("contrast_layer must be in [1, layers].")
    current = embedding
    total = embedding if model == "lightgcn" else None
    contrast = None
    for layer in range(layers):
        current = _FixedSymmetricMM.apply(current, graph) if symmetric_backward else torch.sparse.mm(graph, current)
        if model == "xsimgcl" and perturbed:
            random = noise[layer] if noise is not None else torch.rand_like(current)
            current = current + eps * current.sign() * F.normalize(random, dim=-1)
        # Streaming sum saves the extra [nodes,layers,dim] stack allocation.
        total = current if total is None else total + current
        if layer + 1 == contrast_layer:
            contrast = current
    divisor = layers + 1 if model == "lightgcn" else layers
    return total / divisor, contrast


def info_nce(left, right, temperature=0.2):
    left, right = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
    logits = left @ right.T / temperature
    targets = torch.arange(left.shape[0], device=left.device)
    # Fused stable log-softmax + NLL; equivalent to the diagonal InfoNCE objective.
    return F.cross_entropy(logits, targets)


def objective(embedding, graph, users, positives, negatives, model="lightgcn", layers=3,
              regularization=1e-4, cl_weight=0.1, temperature=0.2, eps=0.1,
              contrast_layer=1, noise=None, symmetric_backward=False):
    final, contrast = encode(embedding, graph, model, layers, eps, contrast_layer,
                             perturbed=model == "xsimgcl", noise=noise, symmetric_backward=symmetric_backward)
    user, positive, negative = final[users], final[positives], final[negatives]
    loss = F.softplus((user * negative).sum(1) - (user * positive).sum(1)).mean()
    # Shared raw-embedding L2 protocol across models; record this recipe choice.
    penalty = sum(embedding[index].square().sum() for index in [users, positives, negatives])
    loss = loss + regularization * penalty / (2 * len(users))
    if model == "xsimgcl":
        uids, iids = torch.unique(users), torch.unique(positives)
        loss = loss + cl_weight * (
            info_nce(final[uids], contrast[uids], temperature)
            + info_nce(final[iids], contrast[iids], temperature)
        )
    return loss


def training_keys(uid, iid, n_items, device):
    return torch.sort(torch.as_tensor(uid, dtype=torch.long, device=device) * n_items
                      + torch.as_tensor(iid, dtype=torch.long, device=device)).values


def key_membership(sorted_keys, queries):
    positions = torch.searchsorted(sorted_keys, queries)
    return (positions < sorted_keys.numel()) & (sorted_keys[positions.clamp_max(sorted_keys.numel() - 1)] == queries)


def sample_unseen(users, sorted_train_keys, n_items, generator=None):
    """GPU rejection sampling; known train positives are never negatives.

    Eval answers are not consulted. The caller must reject users with no unseen
    item in the catalog. Every user in the supplied dataset has unseen items.
    """
    negatives = torch.randint(n_items, users.shape, device=users.device, generator=generator)
    for _ in range(100):
        rejected = key_membership(sorted_train_keys, users * n_items + negatives)
        if not bool(rejected.any()):
            return negatives
        negatives[rejected] = torch.randint(n_items, (int(rejected.sum()),), device=users.device, generator=generator)
    raise RuntimeError("Negative sampler exhausted rejection limit; check user degrees.")


@torch.inference_mode()
def deterministic_topk(scores, k):
    """Exact descending score, ascending item-ID tie break, including kth ties."""
    count = min(k + 1, scores.shape[1])
    values, indices = torch.topk(scores, count, dim=1, sorted=True)
    selected = indices[:, :k].clone()
    if count > k:
        boundary_ties = values[:, k - 1] == values[:, k]
        # Usually no row needs this slow path. It is necessary for identical
        # graph embeddings (e.g. items with identical graph neighborhoods).
        for row in boundary_ties.nonzero().flatten().tolist():
            threshold = values[row, k - 1]
            stronger = indices[row, :k][values[row, :k] > threshold]
            tied = (scores[row] == threshold).nonzero().flatten()
            selected[row] = torch.cat([stronger, tied[:k - len(stronger)]])
    ids = selected.sort(dim=1).values
    ranked = torch.argsort(scores.gather(1, ids), dim=1, descending=True, stable=True)
    return ids.gather(1, ranked)


@torch.inference_mode()
def full_catalog_topk(embedding, user_ids, n_users, seen_uid, seen_iid, k=100, batch_size=512):
    """Exact all-item scoring/top-k on GPU, bounded by a user batch.

    Embeddings remain on GPU. Small history indices and final top-k IDs cross
    the host boundary; no users-by-entire-catalog result is stored on the host.
    Input seen rows must be sorted by uid. This is the supplied handoff order.
    """
    import numpy as np

    user_ids = np.asarray(user_ids, dtype=np.int64)
    seen_uid, seen_iid = np.asarray(seen_uid), np.asarray(seen_iid)
    counts = np.bincount(seen_uid, minlength=n_users)
    pointers = np.r_[0, np.cumsum(counts)]
    if np.any(embedding.shape[0] - n_users - counts[user_ids] < k):
        raise ValueError("An evaluated user has fewer than k unseen candidates.")
    output = np.empty((len(user_ids), k), dtype=np.int32)
    item_embeddings = embedding[n_users:].T
    for start in range(0, len(user_ids), batch_size):
        batch = user_ids[start:start + batch_size]
        ids = torch.as_tensor(batch, dtype=torch.long, device=embedding.device)
        scores = embedding[ids] @ item_embeddings
        lengths = counts[batch]
        local_rows = np.repeat(np.arange(len(batch)), lengths)
        history = np.concatenate([seen_iid[pointers[uid]:pointers[uid + 1]] for uid in batch])
        if len(history):
            scores[torch.as_tensor(local_rows, device=scores.device),
                   torch.as_tensor(history.astype(np.int64), device=scores.device)] = -torch.inf
        selected = deterministic_topk(scores, k)
        output[start:start + len(batch)] = selected.cpu().numpy().astype(np.int32)
    return output
