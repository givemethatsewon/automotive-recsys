"""Run a validation-selected vehicle reranking pilot on an existing Colab GPU.

No fitting, new candidates, test-based selection, or post-train user profiles.
Input manifests name immutable baseline artifacts and read-only added data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import polars as pl

from automotive_recsys.evaluation import contains
from automotive_recsys.vehicle_rerank import match_candidates, stable_rerank


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def compact(frame, key, n, makes, models):
    frame = frame.select(key, 'make', 'model').unique().sort(key, 'make', 'model')
    ids = frame[key].to_numpy().astype(np.int64)
    ptr = np.r_[0, np.cumsum(np.bincount(ids, minlength=n))].astype(np.int64)
    rows = np.array([[makes.get(m, -1), models.get((m, v), -1) if v else -1]
                     for m, v in frame.select('make', 'model').iter_rows()], dtype=np.int64)
    return ptr, rows.reshape(-1, 2)


def paired_ci(delta, repetitions=2000):
    # Resampling unchanged zero differences is equivalent to a multinomial on
    # unique difference values and is much cheaper for this sparse task.
    values, counts = np.unique(delta, return_counts=True)
    rng = np.random.default_rng(20261111)
    boot = np.empty(repetitions)
    for a in range(0, repetitions, 100):
        weights = rng.multinomial(len(delta), counts / len(delta), size=min(100, repetitions-a))
        boot[a:a+len(weights)] = weights @ values / len(delta)
    return np.quantile(boot, [.025, .975]).tolist()


def main():
    import torch
    from numba import njit
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), 'This pilot must run on Colab CUDA.'
    match = njit(match_candidates)
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', default='bpr_mf', help='label of the frozen run whose rankings are reranked')
    p.add_argument('--baseline', type=Path, help='directory holding <split>/recommendations.npz and metrics.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--candidates', type=int, default=100, help='candidate pool size per user (100 = archived top100)')
    p.add_argument('--primary', default='both-model', help='pre-specified source-level family reported as primary')
    args = p.parse_args()
    root = args.root
    data, inputs = root/'data', root/'inputs'
    baseline_dir = args.baseline or root/'baseline'
    out = args.output or root/'output'
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads((root/'manifest.json').read_text())
    for rel, expected in manifest['files'].items():
        assert sha(root/rel) == expected['sha256'], rel
    users = pl.read_parquet(data/'user_map.parquet').sort('uid')
    items = pl.read_parquet(data/'item_map.parquet').sort('iid')
    nu, ni = users.height, items.height
    valid_boundary = int(pl.scan_parquet(data/'valid.parquet').select(pl.col('timestamp').min()).collect().item())
    a1 = pl.read_parquet(inputs/'user_vehicles.parquet')
    a2 = pl.read_parquet(inputs/'user_vehicles_items.parquet')
    pf = pl.read_parquet(inputs/'product_fitment.parquet')
    audit = {'valid_boundary_ms':valid_boundary, 'profile_period':'train only for both valid and test',
             'raw_rows':{'text':a1.height,'items':a2.height,'fitment':pf.height},
             'schemas':{k:{c:str(t) for c,t in f.schema.items()} for k,f in [('text',a1),('items',a2),('fitment',pf)]}}
    for name, f in [('text',a1),('items',a2)]:
        assert 'last_seen' in f.columns, 'Missing timestamp provenance'
        maximum = f['last_seen'].max()
        assert maximum < valid_boundary, 'User vehicle includes post-train reviews'
        assert f['last_seen'].null_count() == 0
        audit[name+'_last_seen_max_ms'] = int(maximum)
    a1 = a1.join(users.select('user_id','uid'), on='user_id', how='inner')
    a2 = a2.filter(pl.col('n_items') >= 2).join(users.select('user_id','uid'), on='user_id', how='inner')
    pf = pf.join(items.select('parent_asin','iid'), on='parent_asin', how='inner')
    audit['joined_rows'] = {'text':a1.height,'items_min2':a2.height,'fitment':pf.height}
    def norm(f):
        return f.with_columns(pl.col('make','model').fill_null('').str.strip_chars().str.to_lowercase())
    a1, a2, pf = norm(a1), norm(a2), norm(pf)
    sources = {'text':a1, 'items':a2, 'both':pl.concat([a1.select('uid','make','model'),a2.select('uid','make','model')]).unique()}
    all_rows = pl.concat([f.select('make','model') for f in [a1,a2,pf]])
    makes = {m:i for i,m in enumerate(sorted(set(all_rows['make'].to_list())-{''}))}
    models = {v:i for i,v in enumerate(sorted(set(all_rows.select('make','model').iter_rows())))}
    item_ptr, item_rows = compact(pf,'iid',ni,makes,models)
    universal = np.zeros(ni, bool)
    universal[pf.filter(pl.col('universal'))['iid'].unique().to_numpy()] = True
    profiles = {name:compact(frame,'uid',nu,makes,models) for name,frame in sources.items()}
    audit['profile_users'] = {k:int(np.count_nonzero(np.diff(v[0]))) for k,v in profiles.items()}
    audit['fitment_items'] = int(np.count_nonzero(np.diff(item_ptr)))
    audit['universal_items'] = int(universal.sum())
    audit['notes'] = ['Year ignored in make/model comparisons; no new alias mapping.',
        'Product title observation time is not supplied; historical point-in-time correctness unverified.',
        'A2 uses train-history item-derived features, not independent vehicle ownership labels.',
        'Mismatch is a proxy derived from the same rules used for the bonus; coverage is reported.']
    write(out/'data_audit.json',audit)
    print(json.dumps({'stage':'audit_complete',**{k:v for k,v in audit.items() if k not in ['schemas','notes']}}),flush=True)
    grid = [0,.02,.05,.1,.2,.5,1,2]
    design = {'model':args.model,'seed':42,'fit':'reuse trained seed42 best checkpoint rankings; no retraining',
       'candidates':(f'same stored full-catalog top{args.candidates} per user; reranking cannot retrieve outside these {args.candidates}'
                     if args.candidates==100 else f'full-catalog top{args.candidates} recomputed from the same best checkpoint; first 100 verified identical to archived top100'),
       'candidate_pool':args.candidates,
       'sources':['text','items','both'],'levels':['make','model'],'beta_grid':grid,
       'score':f'float64 original score + beta * per-user top{args.candidates} score range * match_indicator; zero range replaced by 1',
       'tie':'score descending then iid ascending','selection':'highest all-user valid NDCG per source/level, smallest beta on ties',
       'primary_family':dict(zip(['source','level'],args.primary.split('-'))),'test_used_for_selection':False,
       'profiles':'train only; A2 n_items >= 2; any vehicle can match; universal and unknown get zero bonus',
       'mismatch':'known inferred mismatches / judged top10 pairs; unknown excluded; pair counts and coverage given',
       'bootstrap':'2000 paired user resamples; fixed training seed; not uncertainty across model seeds',
       'created_at':datetime.now(timezone.utc).isoformat()}
    write(out/'design.json',design)
    discounts = torch.tensor(1/np.log2(np.arange(2,12)),device='cuda',dtype=torch.float64)
    unpopular = torch.tensor(~items['popular'].to_numpy(),device='cuda')
    ntrain = torch.tensor(items['n_train'].to_numpy().astype(np.float64),device='cuda')
    selections, rows, results = {}, [], []
    baseline_per_user = {}
    for split in ['valid','test']:
        if split == 'test':
            # This durable receipt is written before reading any test answers.
            write(out/'frozen_selection.json', {'selection_data':'valid only','test_used_for_selection':False,
                  'choices':selections,'design_sha256':sha(out/'design.json'),
                  'created_at':datetime.now(timezone.utc).isoformat()})
        pred = np.load(baseline_dir/split/'recommendations.npz')
        uid, iid, score = pred['uid'],pred['iid'],pred['score']
        truth = pl.read_parquet(data/f'{split}.parquet',columns=['uid','iid'])
        assert np.array_equal(uid,truth['uid'].unique().sort().to_numpy())
        assert iid.shape == score.shape and iid.shape[1] == args.candidates
        assert np.isfinite(score).all() and not np.any(np.diff(np.sort(iid,axis=1),axis=1)==0)
        seen = pl.read_parquet(data/f'seen_for_{split}.parquet',columns=['uid','iid'])
        keys = np.sort(truth['uid'].to_numpy().astype(np.int64)*ni+truth['iid'].to_numpy())
        seen_keys = np.sort(seen['uid'].to_numpy().astype(np.int64)*ni+seen['iid'].to_numpy())
        candidate_keys = uid[:,None].astype(np.int64)*ni+iid
        assert not contains(seen_keys,candidate_keys).any(), 'Seen item in baseline candidates'
        assert not contains(seen_keys,keys).any(), 'Ineligible truth'
        relevance = torch.tensor(contains(keys,candidate_keys),device='cuda')
        counts = np.bincount(truth['uid'].to_numpy(),minlength=nu)[uid]
        idcg = torch.tensor(np.r_[0,np.cumsum(1/np.log2(np.arange(2,12)))][np.minimum(counts,10)],device='cuda')
        count_gpu = torch.tensor(counts,device='cuda',dtype=torch.float64)
        gi = torch.tensor(iid.astype(np.int64),device='cuda')
        gs = torch.tensor(score.astype(np.float64),device='cuda')
        scale = gs.max(1).values-gs.min(1).values
        scale[scale==0]=1
        item_order = torch.argsort(gi,dim=1,stable=True)
        def evaluate(state, beta):
            adjusted = gs + float(beta)*scale[:,None]*(state==1)
            by_id = torch.gather(adjusted,1,item_order)
            order = torch.gather(item_order,1,torch.argsort(by_id,dim=1,descending=True,stable=True))
            top = order[:,:10]
            ranked = torch.gather(gi,1,top)
            hits = torch.gather(relevance,1,top)
            classified = torch.gather(state,1,top)
            known, wrong = classified>=0, classified==0
            values = {'ndcg':(hits*discounts).sum(1)/idcg,'recall':hits.sum(1)/count_gpu,
                      'hit_rate':hits.any(1).double(),'aplt':unpopular[ranked].double().mean(1),
                      'arp':ntrain[ranked].mean(1)}
            return values, ranked, order, adjusted, classified, known, wrong
        def summarize(values, ranked, classified, mask=None):
            if mask is None:mask=torch.ones(len(uid),device='cuda',dtype=torch.bool)
            cls=classified[mask]; judged=cls>=0; wrong=cls==0
            users_count=int(mask.sum())
            denom=int(judged.sum()); num=int(wrong.sum())
            return {'users':users_count,**{k:float(v[mask].mean()) for k,v in values.items()},
               'catalog_coverage':float(torch.unique(ranked[mask]).numel()/ni),
               'judged_pairs':denom,'inferred_mismatch_pairs':num,
               'judged_pair_coverage':denom/(users_count*10),
               'inferred_mismatch_rate':num/denom if denom else None,
               'match_share':float((cls==1).double().mean())}
        zero = torch.full(iid.shape,-1,device='cuda',dtype=torch.int8)
        base_values,base_ranked,*_ = evaluate(zero,0)
        assert np.array_equal(base_ranked.cpu().numpy(),iid[:,:10]), 'beta0 ranking differs'
        baseline_per_user[split]={k:v.cpu().numpy() for k,v in base_values.items()}
        base_json=json.loads((baseline_dir/split/'metrics.json').read_text())
        archived_diff={m:float(v.mean())-base_json['metrics']['all'][m]['mean'] for m,v in base_values.items()}
        for metric,diff in archived_diff.items():
            if args.candidates==100:
                assert abs(diff) < 1e-10, metric
            else:
                # Recomputed graph rankings differ from the archive only by
                # float32 SpMM rounding; the paired baseline is the recomputed pool.
                reference=abs(base_json['metrics']['all'][metric]['mean'])
                assert abs(diff) <= 1e-3*max(reference,1e-12), (metric,diff)
        write(out/f'{split}_baseline.json',{'users':len(uid),'ground_truth_interactions':truth.height,
             **{k:float(v.mean()) for k,v in base_values.items()},
             f'candidate_recall{args.candidates}':float((relevance.sum(1)/count_gpu).mean()),'matches_archived_metrics':all(abs(d)<1e-10 for d in archived_diff.values()),'archived_metric_diff':archived_diff})
        del zero
        for source,(up,uv) in profiles.items():
            for level,level_name in enumerate(['make','model']):
                begin=time.perf_counter()
                state_np=match(uid,iid,up,uv,item_ptr,item_rows,universal,level)
                state=torch.tensor(state_np,device='cuda')
                family=f'{source}_{level_name}'
                betas=grid if split=='valid' else [0,selections[family]['beta']]
                betas=sorted(set(betas))
                candidate_metrics=[]
                for beta in betas:
                    values,ranked,order,adjusted,classified,known,wrong=evaluate(state,beta)
                    row={'split':split,'source':source,'level':level_name,'beta':beta,
                         **summarize(values,ranked,classified)}
                    if split=='valid':
                        rows.append(row); candidate_metrics.append(row)
                    else:
                        row['selected_on_valid']=beta==selections[family]['beta']
                        results.append(row)
                        if row['selected_on_valid']:
                            array={k:v.cpu().numpy() for k,v in values.items()}
                            group=users['group'].to_numpy()[uid]
                            covered=np.diff(up)[uid]>0
                            peruser=pl.DataFrame({'uid':uid,'group':group,'has_vehicle':covered,
                                'judged_top10':known.sum(1).cpu().numpy(),'inferred_mismatch_top10':wrong.sum(1).cpu().numpy(),
                                **array,**{'delta_'+k:v-baseline_per_user[split][k] for k,v in array.items()}})
                            peruser.write_parquet(out/f'{family}_per_user.parquet')
                            top=ranked.cpu().numpy()
                            top_scores=torch.gather(adjusted,1,order[:,:10]).cpu().numpy()
                            np.savez_compressed(out/f'{family}_top10.npz',uid=uid,iid=top,score=top_scores)
                            delta=array['ndcg']-baseline_per_user[split]['ndcg']
                            row['ndcg_delta']=float(delta.mean())
                            row['ndcg_relative_change']=float(delta.mean()/baseline_per_user[split]['ndcg'].mean())
                            row['ndcg_paired_ci95']=paired_ci(delta)
                            row['profile_coverage']=float(covered.mean())
                            row['changed_top10_users']=int(np.any(top!=iid[:,:10],axis=1).sum())
                            row['groups']={}
                            for label,mask in [('1-5',group=='1-5'),('6+',group=='6+'),('has_vehicle',covered),('unknown_vehicle',~covered)]:
                                row['groups'][label]=summarize(values,ranked,classified,torch.tensor(mask,device='cuda'))
                                row['groups'][label]['ndcg_delta']=float(delta[mask].mean())
                            # Verify GPU ordering against the independent CPU reference on a fixed sample.
                            sample=np.linspace(0,len(uid)-1,101,dtype=int)
                            reference,_=stable_rerank(iid[sample],score[sample],state_np[sample],beta)
                            assert np.array_equal(top[sample],reference[:,:10]), 'CPU/CUDA ranking mismatch'
                if split=='valid':
                    best=max(candidate_metrics,key=lambda r:(r['ndcg'],-r['beta']))
                    selections[family]={'source':source,'level':level_name,'beta':best['beta'],'valid_ndcg':best['ndcg']}
                    pl.DataFrame(rows).write_csv(out/'valid_sweep.csv')
                print(json.dumps({'stage':split,'family':family,'selected_beta':selections[family]['beta'],
                      'seconds':time.perf_counter()-begin}),flush=True)
                del state
        del pred,relevance,gi,gs,item_order
        torch.cuda.empty_cache()
    write(out/'test_results.json',results)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]
    pl.DataFrame(flat).write_csv(out/'test_results.csv')
    psource,plevel=args.primary.split('-')
    primary=next(r for r in results if r['source']==psource and r['level']==plevel and r['selected_on_valid'])
    summary={'status':'completed','created_at':datetime.now(timezone.utc).isoformat(),
        'design':design,'primary_result':primary,'choices':selections,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'torch':str(torch.__version__),
        'packages':{k:importlib.metadata.version(k) for k in ['numpy','polars','numba','torch']},
        'elapsed_seconds':time.perf_counter()-started,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'cuda_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'verification':{'input_hashes':True,'beta0_matches_baseline':True,'cpu_cuda_order':True,
                        'full_eligible_cohorts':True,'seen_masking':True,'train_only_profile_timestamps':True},
        'limitations':audit['notes']+['Single training seed; validation sweep may overfit valid.',
            f'Reranking uses the checkpoint top{args.candidates}; candidate retrieval is unchanged.',
            'Not retraining with vehicle features; cannot claim encoder/training improvements.']}
    write(out/'summary.json',summary)
    print(json.dumps({'stage':'completed','primary':primary,'elapsed_seconds':summary['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
