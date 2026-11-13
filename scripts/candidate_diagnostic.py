"""Valid-only frozen-checkpoint candidate bottleneck audit on Colab CUDA.

Compare nested 100/500/1000 pools, with the top100 score range fixed across
pools. No test answers, fitting, or test-based family/parameter selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
from vehicle_pilot import compact, sha, write
from automotive_recsys.evaluation import contains
from automotive_recsys.vehicle_rerank import match_candidates


def main():
    import torch
    from numba import njit
    from automotive_recsys.audit import fingerprint
    from automotive_recsys.gpu_kernels import encode, full_catalog_topk, normalized_graph
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.root; out=a.output; out.mkdir(exist_ok=True,parents=True)
    started=time.perf_counter()
    assert torch.cuda.is_available(), 'Colab CUDA required'
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    data=root/'data'; inputs=root/'inputs'
    users=pl.read_parquet(data/'user_map.parquet').sort('uid')
    items=pl.read_parquet(data/'item_map.parquet').sort('iid')
    nu,ni=users.height,items.height
    boundary=pl.scan_parquet(data/'valid.parquet').select(pl.col('timestamp').min()).collect().item()
    frames=[]
    for filename in ['user_vehicles.parquet','user_vehicles_items.parquet']:
        f=pl.read_parquet(inputs/filename)
        assert f['last_seen'].null_count()==0 and f['last_seen'].max()<boundary
        if 'n_items' in f.columns:f=f.filter(pl.col('n_items')>=2)
        frames.append(f.join(users.select('user_id','uid'),on='user_id',how='inner').select('uid','make','model'))
    def norm(f):return f.with_columns(pl.col('make','model').fill_null('').str.strip_chars().str.to_lowercase())
    profiles=norm(pl.concat(frames)).unique()
    pf=norm(pl.read_parquet(inputs/'product_fitment.parquet').join(items.select('parent_asin','iid'),on='parent_asin',how='inner'))
    allrows=pl.concat([profiles.select('make','model'),pf.select('make','model')])
    makes={m:i for i,m in enumerate(sorted(set(allrows['make'].to_list())-{''}))}
    models={v:i for i,v in enumerate(sorted(set(allrows.select('make','model').iter_rows())))}
    up,ur=compact(profiles,'uid',nu,makes,models); ip,ir=compact(pf,'iid',ni,makes,models)
    universal=np.zeros(ni,bool);universal[pf.filter(pl.col('universal'))['iid'].unique().to_numpy()]=True
    match=njit(match_candidates)
    ck=root/'checkpoints'/a.run/'best.pt'
    snapshot=torch.load(ck,map_location='cuda',weights_only=True)
    assert fingerprint(data)==snapshot['inputs']
    settings=snapshot['settings']; graph=None
    if settings['model']!='bpr_mf':
        train=pl.read_parquet(data/'train.parquet',columns=['uid','iid'])
        graph=normalized_graph(train['uid'].to_numpy(),train['iid'].to_numpy(),nu,ni,'cuda','csr')
        del train
    truth=pl.read_parquet(data/'valid.parquet',columns=['uid','iid'])
    uid=truth['uid'].unique().sort().to_numpy()
    seen=pl.read_parquet(data/'seen_for_valid.parquet',columns=['uid','iid']).sort('uid','iid')
    with torch.inference_mode():
        rep,_=encode(snapshot['embedding'],graph,settings['model'],layers=settings['layers'],perturbed=False,symmetric_backward=settings['model']!='bpr_mf')
        iid,score=full_catalog_topk(rep,uid,nu,seen['uid'].to_numpy(),seen['iid'].to_numpy(),k=1000,batch_size=settings.get('validation_batch',1024),return_scores=True)
    del rep,snapshot,graph;torch.cuda.empty_cache()
    old=np.load(root/'baseline'/a.run/'valid/recommendations.npz')
    assert np.array_equal(uid,old['uid'])
    integrity={'positions_identical':float((iid[:,:100]==old['iid']).mean()),'users_top10_identical':float((iid[:,:10]==old['iid'][:,:10]).all(1).mean()),'score_max_abs_diff':float(abs(score[:,:100]-old['score']).max())}
    assert integrity['positions_identical']>=.999 and integrity['users_top10_identical']>=.9999 and integrity['score_max_abs_diff']<=1e-4,integrity
    del old
    keys=np.sort(truth['uid'].to_numpy().astype('int64')*ni+truth['iid'].to_numpy())
    seenkeys=np.sort(seen['uid'].to_numpy().astype('int64')*ni+seen['iid'].to_numpy())
    counts=np.bincount(truth['uid'].to_numpy(),minlength=nu)[uid]
    has=np.diff(up)[uid]>0; groups=users['group'].to_numpy()[uid]
    tail=~items['popular'].to_numpy()
    discounts=np.r_[0,np.cumsum(1/np.log2(np.arange(2,12)))]
    idcg=discounts[np.minimum(counts,10)]
    betas=[0,.02,.05,.1,.2,.5,1,2]; pools=[100,500,1000]
    # Per-user counters stay private; only aggregate summaries enter docs/Git.
    arrays={k:{n:np.empty(len(uid),dtype=np.int32) for n in ['tail','match','tail_match','judged','truth','tail_truth','match_truth','tail_match_truth']} for k in pools}
    sums={(k,b):np.zeros(3) for k in pools for b in betas}
    gd=torch.tensor(1/np.log2(np.arange(2,12)),device='cuda',dtype=torch.float64)
    for lo in range(0,len(uid),1024):
        hi=min(lo+1024,len(uid)); ids=iid[lo:hi]; scores=score[lo:hi]
        assert np.isfinite(scores).all()
        assert not np.any(np.diff(np.sort(ids,axis=1),axis=1)==0)
        query=uid[lo:hi,None].astype('int64')*ni+ids
        assert not contains(seenkeys,query).any()
        relevant=contains(keys,query); is_tail=tail[ids]
        state=match(uid[lo:hi],ids,up,ur,ip,ir,universal,1)
        gscore=torch.tensor(scores.astype('float64'),device='cuda')
        gmatch=torch.tensor(state==1,device='cuda')
        grel=torch.tensor(relevant,device='cuda');gtail=torch.tensor(is_tail,device='cuda')
        # A common score range prevents candidate size from changing beta units.
        scale=gscore[:,:100].max(1).values-gscore[:,:100].min(1).values;scale[scale==0]=1
        gidcg=torch.tensor(idcg[lo:hi],device='cuda');gcounts=torch.tensor(counts[lo:hi],device='cuda')
        for k in pools:
            t=is_tail[:,:k]; m=state[:,:k]==1; r=relevant[:,:k]
            for name,value in [('tail',t),('match',m),('tail_match',t&m),('judged',state[:,:k]>=0),('truth',r),('tail_truth',t&r),('match_truth',m&r),('tail_match_truth',t&m&r)]:arrays[k][name][lo:hi]=value.sum(1)
            idorder=torch.tensor(np.argsort(ids[:,:k],axis=1),device='cuda')
            for b in betas:
                adjusted=gscore[:,:k]+b*scale[:,None]*gmatch[:,:k]
                order=torch.gather(idorder,1,torch.argsort(torch.gather(adjusted,1,idorder),dim=1,descending=True,stable=True))[:,:10]
                hits=torch.gather(grel[:,:k],1,order)
                values=torch.stack([(hits*gd).sum(1)/gidcg,hits.sum(1)/gcounts,torch.gather(gtail[:,:k],1,order).double().mean(1)],dim=1)
                sums[k,b]+=values.sum(0).cpu().numpy()
        if lo%51200==0:print(json.dumps({'stage':'audit','run':a.run,'users_done':hi,'users':len(uid)}),flush=True)
    summary=[]; private=[]
    for k in pools:
        vals=arrays[k]
        private.append(pl.DataFrame({'uid':uid,'pool':k,'group':groups,'has_vehicle':has,**vals}))
        for label,mask in [('all',np.ones(len(uid),bool)),('has_vehicle',has),('unknown_vehicle',~has),('1-5',groups=='1-5'),('6+',groups=='6+')]:
            row={'pool':k,'group':label,'users':int(mask.sum())}
            for name,v in vals.items():
                row[name+'_mean']=float(v[mask].mean());row[name+'_zero_share']=float((v[mask]==0).mean());row[name+'_p50']=float(np.quantile(v[mask],.5));row[name+'_p90']=float(np.quantile(v[mask],.9))
            row['candidate_recall']=float((vals['truth'][mask]/counts[mask]).mean())
            row['oracle_ndcg10']=float((discounts[np.minimum(vals['truth'][mask],10)]/idcg[mask]).mean())
            summary.append(row)
    pl.concat(private).write_parquet(out/'candidate_counts_per_user.parquet')
    pl.DataFrame(summary).write_csv(out/'candidate_availability.csv')
    sweep=[{'pool':k,'beta':b,**dict(zip(['ndcg','recall','aplt'],(sums[k,b]/len(uid)).tolist()))} for k in pools for b in betas]
    pl.DataFrame(sweep).write_csv(out/'valid_fixed_scale_sweep.csv')
    baseline=[r for r in sweep if r['beta']==0]
    assert max(r['ndcg'] for r in baseline)-min(r['ndcg'] for r in baseline)<1e-12
    # Each larger pool must contain all the smaller pool's candidates and hits.
    for small,large in zip(pools,pools[1:]):
        for name in arrays[small]:assert (arrays[large][name]>=arrays[small][name]).all()
    receipt={'status':'completed','run':a.run,'split':'valid only','test_answers_read':False,'fit':False,'family':'A1+A2 model match; train-only profiles; year ignored',
        'pools':pools,'beta_grid':betas,'scale':'same top100 score range for all pools','integrity':integrity,'users':len(uid),'truth_interactions':truth.height,
        'checkpoint_sha256':sha(ck),'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'selected_on_valid':{str(k):max([r for r in sweep if r['pool']==k],key=lambda r:(r['ndcg'],-r['beta'])) for k in pools},
        'elapsed_seconds':time.perf_counter()-started,'cuda_peak_MiB':torch.cuda.max_memory_allocated()/2**20,
        'verification':{'nested_pool_monotonicity':True,'beta0_pool_invariance':True,'train_only_profile_timestamps':True,'seen_exclusion':True,'candidate_uniqueness':True},
        'limitations':['Unknown and universal get no vehicle bonus; inferred matching is not verified fitment.','Valid-only exploration; no new independent test claim.','Prior pool500 used pool-specific score ranges and may not isolate pool size.']}
    write(out/'summary.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
