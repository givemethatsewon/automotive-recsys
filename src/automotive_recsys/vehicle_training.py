"""Train-only vehicle confidence weights; unchanged embeddings and inference."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

from .audit import write_json


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def confidence_weights(states, mismatch_weight=.5):
    states=np.asarray(states)
    if not 0 < mismatch_weight <= 1 or not np.isin(states,[-1,0,1]).all():
        raise ValueError('Invalid confidence weight or fit states')
    return np.where(states==0,mismatch_weight,1).astype(np.float32)


def load_weights(path, data_dir, rows):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta['training_split']!='train' or meta['train_sha256']!=sha(Path(data_dir)/'train.parquet') or meta['weights_sha256']!=sha(path):
        raise ValueError('Vehicle weight provenance or train row order differs')
    weights=np.load(path,allow_pickle=False)
    if weights.shape!=(rows,) or weights.dtype!=np.float32 or not np.isfinite(weights).all() or not ((weights>0)&(weights<=1)).all():
        raise ValueError('Invalid train interaction weights')
    return weights, {k:meta[k] for k in ['weights_sha256','train_sha256','mismatch_weight','recipe']}


def prepare_weights(root, progress=None):
    import torch
    from numba import njit
    from .vehicle_v2 import compact_v2,match_v2
    if not torch.cuda.is_available():raise RuntimeError('Full feature preparation belongs on Colab CUDA')
    root=Path(root);data=root/'data';vehicles=root/'vehicles/v2';out=root/'features';out.mkdir(exist_ok=True)
    target=out/'train_vehicle_weights.npy'
    source={str(p.relative_to(root)):sha(p) for p in sorted(vehicles.glob('*.parquet'))}
    source['provider/vehicle_fit.py']=sha(root/'provider/vehicle_fit.py')
    if target.exists() and target.with_suffix('.json').exists():
        meta=json.loads(target.with_suffix('.json').read_text())
        if meta['source_sha256']==source:
            load_weights(target,data,pl.scan_parquet(data/'train.parquet').select(pl.len()).collect().item());return meta
    boundary=pl.scan_parquet(data/'valid.parquet').select(pl.col('timestamp').min()).collect().item()
    for name in ['user_vehicles.parquet','user_vehicles_items.parquet']:
        profile=pl.read_parquet(vehicles/name,columns=['last_seen'])
        assert profile['last_seen'].null_count()==0 and profile['last_seen'].max()<boundary,'Non-train vehicle profile'
    processed=root/'processed'
    if not processed.exists():processed.symlink_to(data,target_is_directory=True)
    spec=importlib.util.spec_from_file_location('weight_provider',root/'provider/vehicle_fit.py');provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    provider.VEHICLES_VERSION='v2'
    users=pl.read_parquet(data/'user_map.parquet').sort('uid');items=pl.read_parquet(data/'item_map.parquet').sort('iid')
    profiles=provider.load_vehicles('both',until='valid',min_items=2,project_dir=root)
    joined=pl.from_pandas(profiles).join(users.select('user_id','uid'),on='user_id',how='inner')
    fit=pl.read_parquet(vehicles/'product_fitment.parquet').join(items.select('parent_asin','iid'),on='parent_asin',how='inner')
    arrays=compact_v2(joined.select('uid','make','model','segment','year').iter_rows(),fit.select('iid','make','model','segment','year_from','year_to','open_end').iter_rows(),users.height,items.height,provider._canon,provider._model_ok)
    universal=np.zeros(items.height,bool);universal[fit.filter(pl.col('universal'))['iid'].unique().to_numpy()]=True
    train=pl.read_parquet(data/'train.parquet',columns=['uid','iid']);uid=train['uid'].to_numpy();iid=train['iid'].to_numpy()
    up,uv,ip,ir,compatible=arrays;match=njit(match_v2,nogil=True);states=np.empty(train.height,np.int8)
    for lo in range(0,train.height,100000):
        hi=min(lo+100000,train.height)
        states[lo:hi]=match(uid[lo:hi],iid[lo:hi,None],up,uv,ip,ir,universal,compatible,False)[:,0]
        if progress:progress(hi,train.height)
    # Independent provider reference includes matches, mismatches and unknowns.
    sample=np.unique(np.concatenate([np.linspace(0,train.height-1,128,dtype=int)]+[np.flatnonzero(states==v)[:32] for v in [-1,0,1]]))
    import pandas as pd
    reference=provider.VehicleFit(project_dir=root,user_vehicles=profiles,use_year=False).fit_status(pd.DataFrame({'uid':uid[sample],'iid':iid[sample]}))['fit'].to_numpy()
    np.testing.assert_array_equal(states[sample],np.where(reference==0,-1,np.where(reference==-1,0,1)))
    counts={str(v):int((states==v).sum()) for v in [-1,0,1]}
    if not counts['0']:raise ValueError('No mismatches: confidence-weight experiment has no treatment')
    np.save(target,confidence_weights(states),allow_pickle=False)
    meta={'recipe':'v2 A1+A2 n_items>=2; provider no-year; unknown/match=1, mismatch=.5; globally normalized BPR only',
          'training_split':'train','train_sha256':sha(data/'train.parquet'),'weights_sha256':sha(target),'source_sha256':source,
          'mismatch_weight':.5,'rows':train.height,'states':counts,'normalizer':float(confidence_weights(states).mean(dtype=np.float64)),
          'profile_cutoff_exclusive':int(boundary),'provider_reference_pairs':len(sample),'test_profiles_used':False,
          'limitations':['fitment is inferred, not true dislike','A2 inferred from the same train items; self-consistency can reduce mismatch coverage','product metadata and v2 rules have exploratory test-development history']}
    write_json(target.with_suffix('.json'),meta);return meta


def verify_cuda_weighted_objective(root):
    import torch
    from torch.nn import functional as F
    from .gpu_kernels import objective,sample_unseen,training_keys
    root=Path(root);data=root/'data';train=pl.read_parquet(data/'train.parquet',columns=['uid','iid']);nu=pl.scan_parquet(data/'user_map.parquet').select(pl.len()).collect().item();ni=pl.scan_parquet(data/'item_map.parquet').select(pl.len()).collect().item()
    weights,_=load_weights(root/'features/train_vehicle_weights.npy',data,train.height)
    torch.manual_seed(42);w=(torch.randn(nu+ni,64,device='cuda')*.1).requires_grad_();ref=w.detach().clone().requires_grad_()
    u=torch.as_tensor(train['uid'][:8192].to_numpy().astype('int64'),device='cuda');p=torch.as_tensor(train['iid'][:8192].to_numpy().astype('int64'),device='cuda')+nu
    keys=training_keys(train['uid'].to_numpy(),train['iid'].to_numpy(),ni,'cuda');n=sample_unseen(u,keys,ni,generator=torch.Generator(device='cuda').manual_seed(1042))+nu
    # Use actual weights, plus ensure both values exercise the probe batch.
    b=torch.as_tensor(weights[:len(u)],device='cuda').clone();b[::2]=.5;b[1::2]=1;norm=float(weights.mean(dtype=np.float64))
    loss=objective(w,None,u,p,n,model='bpr_mf',regularization=1e-5,interaction_weights=b,weight_normalizer=norm)
    expected=(F.softplus((ref[u]*ref[n]).sum(1)-(ref[u]*ref[p]).sum(1))*b).mean()/norm+1e-5*sum(ref[x].square().sum() for x in [u,p,n])/(2*len(u))
    loss.backward();expected.backward();torch.testing.assert_close(loss,expected);torch.testing.assert_close(w.grad,ref.grad,rtol=1e-5,atol=1e-8)
    return {'passed':True,'batch':len(u),'loss_abs_difference':float((loss-expected).abs()),'gradient_max_abs_difference':float((w.grad-ref.grad).abs().max()),'cuda':True}


def finish_report(output, store, plan, records, journal):
    import csv
    from .experiment_report import paired_bootstrap
    report=Path(output)/'report';report.mkdir(exist_ok=True,parents=True);rows=[];vectors={}
    for job in plan['jobs']:
        rec=records[job['run_id']];summary=rec['summary'];metrics=summary['test_metrics']['all']
        rows.append({'run':job['run_id'],'seed':job['seed'],'variant':job['settings']['variant'],'best_epoch':summary['best_epoch'],
                     'ndcg':metrics['ndcg']['mean'],'recall':metrics['recall']['mean'],'aplt':metrics['aplt']['mean'],'arp':metrics['arp']['mean'],
                     'tail_recall10':metrics['tail_recall10']['mean']})
        artifact=next(a for a in rec['artifacts'] if a['relative']=='test/per_user.parquet')
        local=store.download(artifact['id'],report/(job['run_id']+'.parquet'),artifact['sha256'])
        frame=pl.read_parquet(local).sort('uid');vectors[job['settings']['variant'],job['seed']]=(frame['uid'].to_numpy(),frame['ndcg'].to_numpy());local.unlink()
    paired=[]
    for seed in plan['seeds']:
        ua,a=vectors['vehicle_control',seed];ub,b=vectors['vehicle_mismatch_half',seed];assert np.array_equal(ua,ub)
        ci=paired_bootstrap(b-a,progress=lambda:journal.progress('paired_bootstrap'))
        paired.append({'seed':seed,'ndcg_delta':float((b-a).mean()),'ci95':ci})
    means=[]
    for variant in ['vehicle_control','vehicle_mismatch_half']:
        values=[r for r in rows if r['variant']==variant]
        means.append({'variant':variant,**{f'{metric}_{stat}':float(getattr(np,stat)([r[metric] for r in values],**({'ddof':1} if stat=='std' else {}))) for metric in ['ndcg','recall','aplt','arp'] for stat in ['mean','std']}})
    for name,values in [('all_seed_results.csv',rows),('seed_summary.csv',means)]:
        with (report/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    write_json(report/'paired_by_seed.json',paired)
    (report/'RESULTS.md').write_text('# Vehicle-weighted training\n\nSix predeclared runs completed. See all_seed_results.csv, seed_summary.csv and paired_by_seed.json. Seed SD and per-seed paired user intervals are different uncertainty estimates. Inferred mismatch is not verified dislike.\n')
    for p in report.iterdir():store.put(p,'report/'+p.name)
    journal.log('vehicle_weight_report_completed',{'runs':len(rows)})
