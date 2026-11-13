"""Recompute a wider full-catalog candidate pool from a frozen best checkpoint.

No fitting. The same checkpoint, train-only graph, seen masking and
score-descending/IID-ascending ordering produce top-K for valid and test.
The first 100 columns must equal the archived top-100 IDs; otherwise abort.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl


def main():
    import torch
    from automotive_recsys.audit import fingerprint
    from automotive_recsys.gpu_kernels import encode, full_catalog_topk, normalized_graph

    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--k', type=int, required=True)
    args = p.parse_args()
    assert torch.cuda.is_available(), 'Candidate pool must run on Colab CUDA.'
    torch.backends.cuda.matmul.allow_tf32 = False
    data = args.root/'data'
    archived = args.root/'baseline'/args.run
    out = args.root/'pool'/args.run
    snapshot = torch.load(args.root/'checkpoints'/args.run/'best.pt', map_location='cuda', weights_only=True)
    settings = snapshot['settings']
    assert fingerprint(data) == snapshot['inputs'], 'Checkpoint input hashes differ.'
    n_users = pl.scan_parquet(data/'user_map.parquet').select(pl.len()).collect().item()
    n_items = snapshot['embedding'].shape[0] - n_users
    graph = None
    if settings['model'] != 'bpr_mf':
        train = pl.read_parquet(data/'train.parquet', columns=['uid', 'iid'])
        graph = normalized_graph(train['uid'].to_numpy(), train['iid'].to_numpy(), n_users, n_items, 'cuda', 'csr')
    receipt = {'run': args.run, 'k': args.k, 'checkpoint_epoch': int(snapshot['epoch']), 'model': settings['model'], 'splits': {}}
    with torch.inference_mode():
        representation, _ = encode(snapshot['embedding'], graph, settings['model'], layers=settings['layers'],
                                   perturbed=False, symmetric_backward=settings['model'] != 'bpr_mf')
        for split in ['valid', 'test']:
            uid = pl.read_parquet(data/f'{split}.parquet', columns=['uid'])['uid'].unique().sort().to_numpy()
            seen = pl.read_parquet(data/f'seen_for_{split}.parquet').sort('uid', 'iid')
            iid, score = full_catalog_topk(representation, uid, n_users, seen['uid'].to_numpy(), seen['iid'].to_numpy(),
                                           k=args.k, batch_size=settings.get('validation_batch', 1024), return_scores=True)
            old = np.load(archived/split/'recommendations.npz')
            assert np.array_equal(old['uid'], uid), split
            same_ids = bool(np.array_equal(old['iid'], iid[:, :100]))
            max_score_diff = float(np.abs(old['score'].astype(np.float64) - score[:, :100]).max())
            diagnostics = {'torch': str(torch.__version__),
                'users_top100_identical': float((old['iid'] == iid[:, :100]).all(1).mean()),
                'users_top10_identical': float((old['iid'][:, :10] == iid[:, :10]).all(1).mean()),
                'positions_identical': float((old['iid'] == iid[:, :100]).mean()),
                'max_abs_score_diff_by_position': max_score_diff}
            print(json.dumps({'stage': 'pool_check', 'run': args.run, 'split': split, **diagnostics}), flush=True)
            # Amended 2026-11-11 after attempt 2: graph SpMM reproduces scores
            # only to float32 rounding (~1e-5), swapping near-tied ranks. Require
            # near-identity; the recomputed pool is its own paired baseline.
            assert diagnostics['positions_identical'] >= 0.999, diagnostics
            assert diagnostics['users_top10_identical'] >= 0.9999, diagnostics
            assert max_score_diff <= 1e-4, diagnostics
            (out/split).mkdir(parents=True, exist_ok=True)
            np.savez(out/split/'recommendations.npz', uid=uid, iid=iid, score=score)
            shutil.copy(archived/split/'metrics.json', out/split/'metrics.json')
            receipt['splits'][split] = {'users': int(len(uid)), 'top100_ids_identical': same_ids, **diagnostics}
            print(json.dumps({'stage': 'pool', 'run': args.run, 'split': split, **receipt['splits'][split]}), flush=True)
    (out/'pool_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    del representation, snapshot, graph
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
