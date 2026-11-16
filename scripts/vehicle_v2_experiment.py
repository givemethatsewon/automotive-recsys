"""Paired v1/v2 cleaning comparison on frozen CUDA checkpoints, seed42.

Four predeclared cases separate data changes, provider matching, and years.
All user profiles remain train-only. Pool/beta choices use validation only.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import polars as pl

from automotive_recsys.audit import fingerprint
from automotive_recsys.evaluation import contains, metrics_at_k
from automotive_recsys.vehicle_rerank import match_candidates, stable_rerank
from automotive_recsys.vehicle_v2 import compact_v2, match_v2
from vehicle_pilot import compact, paired_ci, sha, write

CASES = ['v1_exact', 'v2_exact', 'v2_provider_no_year', 'v2_provider_year']
POOLS = [100, 500, 1000]
BETAS = [0, .02, .05, .1, .2, .5, 1, 2]


def main():
    import torch
    from numba import njit
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available(), 'Colab CUDA required'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter();data=root/'data'
    manifest=json.loads((root/'manifest.json').read_text())
    for rel, meta in manifest['files'].items(): assert sha(root/rel)==meta['sha256'],rel
    users=pl.read_parquet(data/'user_map.parquet').sort('uid')
    items=pl.read_parquet(data/'item_map.parquet').sort('iid')
    nu,ni=users.height,items.height
    boundary=pl.scan_parquet(data/'valid.parquet').select(pl.col('timestamp').min()).collect().item()
    module_path=root/'provider/vehicle_fit.py'
    spec=importlib.util.spec_from_file_location('provider_vehicle_fit',module_path)
    provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    (root/'processed').symlink_to(data,target_is_directory=True) if not (root/'processed').exists() else None
    (root/'vehicles').mkdir(exist_ok=True)
    (root/'vehicles/v2').symlink_to(root/'v2',target_is_directory=True) if not (root/'vehicles/v2').exists() else None
    datasets={};audit={}
    exact=njit(match_candidates);native=njit(match_v2)
    for version,folder in [('v1',root/'inputs'),('v2',root/'v2')]:
        frames=[];details={}
        for name in ['user_vehicles.parquet','user_vehicles_items.parquet']:
            f=pl.read_parquet(folder/name)
            assert f['last_seen'].null_count()==0 and f['last_seen'].max()<boundary
            details[name]={'rows':f.height,'last_seen_max':int(f['last_seen'].max()),'schema':{k:str(v) for k,v in f.schema.items()}}
            if 'n_items' in f.columns:f=f.filter(pl.col('n_items')>=2)
            frames.append(f.join(users.select('user_id','uid'),on='user_id',how='inner'))
        profiles=pl.concat([f.select('uid','make','model') for f in frames]).unique()
        fit=pl.read_parquet(folder/'product_fitment.parquet').join(items.select('parent_asin','iid'),on='parent_asin',how='inner')
        def norm(f):return f.with_columns(pl.col('make','model').fill_null('').str.strip_chars().str.to_lowercase())
        prof,pf=norm(profiles),norm(fit)
        rows=pl.concat([prof.select('make','model'),pf.select('make','model')])
        makes={v:i for i,v in enumerate(sorted(set(rows['make'].to_list())-{''}))}
        models={v:i for i,v in enumerate(sorted(set(rows.select('make','model').iter_rows())))}
        up,uv=compact(prof,'uid',nu,makes,models);ip,ir=compact(pf,'iid',ni,makes,models)
        universal=np.zeros(ni,bool);universal[pf.filter(pl.col('universal'))['iid'].unique().to_numpy()]=True
        datasets[version]={'exact':(up,uv,ip,ir,universal),'covered':np.diff(up)>0}
        if version=='v2':
            # Use the exact A1+A2 merge/filter policy supplied by the provider.
            uv_ref=provider.load_vehicles('both',until='valid',min_items=2,project_dir=root)
            joined=pl.from_pandas(uv_ref).join(users.select('user_id','uid'),on='user_id',how='inner')
            arrays=compact_v2(joined.select('uid','make','model','segment','year').iter_rows(),
                fit.select('iid','make','model','segment','year_from','year_to','open_end').iter_rows(),nu,ni,provider._canon,provider._model_ok)
            datasets[version]['provider']=arrays
            datasets[version]['provider_covered']=np.diff(arrays[0])>0
            datasets[version]['provider_reference_profiles']=uv_ref
        audit[version]={'inputs':details,'profile_users':int((np.diff(up)>0).sum()),'fitment_items':fit['iid'].n_unique(),'universal_items':int(universal.sum())}
    write(out/'data_audit.json',{'valid_boundary':int(boundary),'train_only':True,'datasets':audit})
    design={'created_at':datetime.now(timezone.utc).isoformat(),'run':a.run,'seed':42,'cases':CASES,'pools':POOLS,'betas':BETAS,
        'training':False,'profiles':'A1+A2 n_items>=2; train-only for both splits','primary_pool':500,
        'primary_comparison':'v2_exact minus v1_exact, shared v1-valid-selected beta at fixed pool500; independently retuned results secondary',
        'rule_comparisons':['v2_provider_no_year minus v2_exact','v2_provider_year minus v2_provider_no_year'],
        'score':'original float64 score + beta * fixed top100 score range * match',
        'selection':'max full-user valid NDCG; smallest beta on ties; each pool reported; test never selects',
        'test_independence':'exploratory follow-up: test used in prior research and provider rule development',
        'input_hashes':manifest['files'],'provider_code_sha256':sha(module_path)}
    write(out/'design.json',design)
    print(json.dumps({'stage':'input_audit_complete','run':a.run,'datasets':audit}),flush=True)
    subprocess.run([sys.executable,str(root/'scripts/candidate_pool.py'),'--root',str(root),'--run',a.run,'--k','1000'],check=True)
    pool=root/'pool'/a.run;shutil.copy(pool/'pool_receipt.json',out/'pool_receipt.json')
    choices={};sweep=[];test_rows=[];test_arrays={};fixed_arrays={}
    discounts=torch.tensor(1/np.log2(np.arange(2,12)),device='cuda',dtype=torch.float64)
    tail=~items['popular'].to_numpy();train_counts=items['n_train'].to_numpy()
    for split in ['valid','test']:
        if split=='test':write(out/'frozen_selection.json',{'selection_data':'valid only','test_used_for_selection':False,'choices':choices,'design_sha256':sha(out/'design.json')})
        pred=np.load(pool/split/'recommendations.npz');uid,iid,score=pred['uid'],pred['iid'],pred['score']
        truth=pl.read_parquet(data/f'{split}.parquet',columns=['uid','iid'])
        assert np.array_equal(uid,truth['uid'].unique().sort().to_numpy())
        counts=np.bincount(truth['uid'].to_numpy(),minlength=nu)[uid]
        keys=np.sort(truth['uid'].to_numpy().astype('int64')*ni+truth['iid'].to_numpy())
        seen=pl.read_parquet(data/f'seen_for_{split}.parquet',columns=['uid','iid'])
        seenkeys=np.sort(seen['uid'].to_numpy().astype('int64')*ni+seen['iid'].to_numpy())
        assert not contains(seenkeys,uid[:,None].astype('int64')*ni+iid).any()
        base=metrics_at_k(uid,iid[:,:10],keys,counts,tail,train_counts)
        idcg=np.r_[0,np.cumsum(1/np.log2(np.arange(2,12)))][np.minimum(counts,10)]
        for case in CASES:
            version='v1' if case=='v1_exact' else 'v2';ds=datasets[version]
            if 'provider' in case:
                up,uv,ip,ir,compat=ds['provider'];universal=ds['exact'][-1]
                state=native(uid,iid,up,uv,ip,ir,universal,compat,case.endswith('_year') and not case.endswith('no_year'))
                covered=ds['provider_covered'][uid]
                # Independent supplied pandas implementation checks actual pairs.
                import pandas as pd
                sample=np.linspace(0,len(uid)-1,128,dtype=int)
                pairs=pd.DataFrame({'uid':np.repeat(uid[sample],10),'iid':iid[sample,:10].ravel()})
                vf=provider.VehicleFit(project_dir=root,user_vehicles=ds['provider_reference_profiles'],use_year=case=='v2_provider_year')
                reference=vf.fit_status(pairs)['fit'].to_numpy().reshape(-1,10)
                translated=np.where(reference==0,-1,np.where(reference==-1,0,1))
                assert np.array_equal(translated,state[sample,:10]), 'Provider reference mismatch'
                del vf
            else:
                up,uv,ip,ir,universal=ds['exact'];state=exact(uid,iid,up,uv,ip,ir,universal,1);covered=ds['covered'][uid]
            configs=[(k,b) for k in POOLS for b in (BETAS if split=='valid' else sorted({0,choices[f'{case}_{k}']['beta'],choices[f'v1_exact_{k}']['beta']}))]
            sums={c:np.zeros(3) for c in configs}
            saved={c:np.empty((len(uid),10),dtype=np.int32) for c in configs} if split=='test' else {}
            saved_state={c:np.empty((len(uid),10),dtype=np.int8) for c in configs} if split=='test' else {}
            saved_score={c:np.empty((len(uid),10),dtype=np.float64) for c in configs} if split=='test' else {}
            for lo in range(0,len(uid),1024):
                hi=min(lo+1024,len(uid));ids=iid[lo:hi];s=torch.tensor(score[lo:hi],device='cuda',dtype=torch.float64)
                match=torch.tensor(state[lo:hi]==1,device='cuda');relevant=torch.tensor(contains(keys,uid[lo:hi,None].astype('int64')*ni+ids),device='cuda')
                gt=torch.tensor(tail[ids],device='cuda');scale=s[:,0]-s[:,99];scale[scale==0]=1
                dcg=torch.tensor(idcg[lo:hi],device='cuda');ct=torch.tensor(counts[lo:hi],device='cuda')
                idorders={k:torch.tensor(np.argsort(ids[:,:k],axis=1),device='cuda') for k in POOLS}
                for k,b in configs:
                    idorder=idorders[k]
                    adjusted=s[:,:k]+b*scale[:,None]*match[:,:k]
                    order=idorder.gather(1,adjusted.gather(1,idorder).argsort(dim=1,descending=True,stable=True))[:,:10]
                    hits=relevant[:,:k].gather(1,order)
                    vals=torch.stack([(hits*discounts).sum(1)/dcg,hits.sum(1)/ct,gt[:,:k].gather(1,order).double().mean(1)],1)
                    sums[k,b]+=vals.sum(0).cpu().numpy()
                    if split=='test':
                        cpu_order=order.cpu().numpy()
                        saved[k,b][lo:hi]=np.take_along_axis(ids[:,:k],cpu_order,1)
                        saved_state[k,b][lo:hi]=np.take_along_axis(state[lo:hi,:k],cpu_order,1)
                        saved_score[k,b][lo:hi]=adjusted.gather(1,order).cpu().numpy()
            for (k,b),value in sums.items():
                row={'case':case,'split':split,'pool':k,'beta':b,**dict(zip(['ndcg','recall','aplt'],(value/len(uid)).tolist()))}
                if split=='valid':sweep.append(row)
                else:
                    top=saved[k,b];v=metrics_at_k(uid,top,keys,counts,tail,train_counts)
                    delta=v['ndcg']-base['ndcg'];row.update(selected_on_valid=b==choices[f'{case}_{k}']['beta'],fixed_v1_beta=b==choices[f'v1_exact_{k}']['beta'],ndcg_delta=float(delta.mean()),ndcg_relative_change=float(delta.mean()/base['ndcg'].mean()),ndcg_ci95=paired_ci(delta),changed_top10_share=float(np.any(top!=iid[:,:10],axis=1).mean()),profile_coverage=float(covered.mean()))
                    classified=saved_state[k,b]
                    judged=classified>=0
                    row.update(judged_pair_coverage=float(judged.mean()),inferred_mismatch_rate=float((classified==0).sum()/judged.sum()) if judged.any() else None)
                    groups=users['group'].to_numpy()[uid];row['groups']={}
                    for label,mask in [('all',np.ones(len(uid),bool)),('has_vehicle',covered),('unknown_vehicle',~covered),('1-5',groups=='1-5'),('6+',groups=='6+')]:
                        row['groups'][label]={'users':int(mask.sum()),**{m:float(val[mask].mean()) if mask.any() else None for m,val in v.items()},'ndcg_delta':float(delta[mask].mean()) if mask.any() else None}
                    assert not (~covered).any() or np.all(delta[~covered]==0), 'Unknown-profile ranking changed'
                    if b==choices[f'v1_exact_{k}']['beta']:fixed_arrays[case,k]=v['ndcg']
                    if b==choices[f'{case}_{k}']['beta']:
                        test_arrays[case,k]=v['ndcg']
                        pl.DataFrame({'uid':uid,'group':groups,'has_vehicle':covered,**v,'ndcg_delta':delta}).write_parquet(out/f'{case}_pool{k}_per_user.parquet')
                        np.savez_compressed(out/f'{case}_pool{k}_top10.npz',uid=uid,iid=top,score=saved_score[k,b])
                    sample=np.linspace(0,len(uid)-1,31,dtype=int)
                    # Fixed-scale CPU reference (stable_rerank's default is pool-specific).
                    sc=score[sample,:k].astype('float64');scale=score[sample,0].astype('float64')-score[sample,99].astype('float64');scale[scale==0]=1
                    order=np.lexsort((iid[sample,:k],-(sc+b*scale[:,None]*(state[sample,:k]==1))),axis=1)[:,:10]
                    assert np.array_equal(top[sample],np.take_along_axis(iid[sample,:k],order,1))
                    test_rows.append(row)
            if split=='valid':
                for k in POOLS:choices[f'{case}_{k}']=max([r for r in sweep if r['case']==case and r['pool']==k],key=lambda r:(r['ndcg'],-r['beta']))
                pl.DataFrame(sweep).write_csv(out/'valid_sweep.csv')
            print(json.dumps({'stage':split,'case':case,'run':a.run,'users':len(uid),'choices':{str(k):choices[f'{case}_{k}']['beta'] for k in POOLS}}),flush=True)
            del state
        if split=='test':
            comparisons=[]
            for k in POOLS:
                for left,right in [('v1_exact','v2_exact'),('v2_exact','v2_provider_no_year'),('v2_provider_no_year','v2_provider_year')]:
                    d=test_arrays[right,k]-test_arrays[left,k]
                    comparisons.append({'pool':k,'left':left,'right':right,'ndcg_delta':float(d.mean()),'paired_ci95':paired_ci(d)})
            for k in POOLS:
                d=fixed_arrays['v2_exact',k]-fixed_arrays['v1_exact',k]
                comparisons.append({'pool':k,'left':'v1_exact','right':'v2_exact','beta_policy':'same v1-valid-selected beta','ndcg_delta':float(d.mean()),'paired_ci95':paired_ci(d)})
            write(out/'paired_comparisons.json',comparisons)
    write(out/'test_results.json',test_rows)
    pl.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in test_rows]).write_csv(out/'test_results.csv')
    summary={'status':'completed','run':a.run,'seed':42,'training':False,'choices':choices,'elapsed_seconds':time.perf_counter()-started,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'provider_reference_check':True,'input_hashes_verified':True,'train_only_profiles':True,'primary_pool':500,'paired_comparisons':comparisons}
    write(out/'summary.json',summary)
    print(json.dumps({'stage':'completed',**summary}),flush=True)


if __name__=='__main__':main()
