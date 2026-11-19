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


def verify_cuda_weighted_objective(root, settings=None):
    import torch
    from torch.nn import functional as F
    from .gpu_kernels import objective,sample_unseen,training_keys,normalized_graph
    settings=settings or {'model':'bpr_mf','dim':64,'layers':3,'regularization':1e-5,'cl_weight':.1,'eps':.1,'temperature':.2}
    model=settings['model']
    root=Path(root);data=root/'data';train=pl.read_parquet(data/'train.parquet',columns=['uid','iid']);nu=pl.scan_parquet(data/'user_map.parquet').select(pl.len()).collect().item();ni=pl.scan_parquet(data/'item_map.parquet').select(pl.len()).collect().item()
    weights,_=load_weights(root/'features/train_vehicle_weights.npy',data,train.height)
    torch.manual_seed(42);w=(torch.randn(nu+ni,settings['dim'],device='cuda')*.1).requires_grad_();ref=w.detach().clone().requires_grad_()
    u=torch.as_tensor(train['uid'][:8192].to_numpy().astype('int64'),device='cuda');p=torch.as_tensor(train['iid'][:8192].to_numpy().astype('int64'),device='cuda')+nu
    keys=training_keys(train['uid'].to_numpy(),train['iid'].to_numpy(),ni,'cuda');n=sample_unseen(u,keys,ni,generator=torch.Generator(device='cuda').manual_seed(1042))+nu
    # Use actual weights, plus ensure both values exercise the probe batch.
    b=torch.as_tensor(weights[:len(u)],device='cuda').clone();b[::2]=.5;b[1::2]=1;norm=float(weights.mean(dtype=np.float64))
    graph=None if model=='bpr_mf' else normalized_graph(train['uid'].to_numpy(),train['iid'].to_numpy(),nu,ni,'cuda','csr')
    params={k:settings[k] for k in ['model','layers','regularization','cl_weight','eps','temperature']}
    noise=[torch.rand_like(w) for _ in range(settings['layers'])] if model=='xsimgcl' else None
    loss=objective(w,graph,u,p,n,**params,noise=noise,symmetric_backward=model!='bpr_mf',interaction_weights=b,weight_normalizer=norm)
    if model=='bpr_mf':
        expected=(F.softplus((ref[u]*ref[n]).sum(1)-(ref[u]*ref[p]).sum(1))*b).mean()/norm+settings['regularization']*sum(ref[x].square().sum() for x in [u,p,n])/(2*len(u))
    else:
        expected=objective(ref,graph,u,p,n,**params,noise=noise,symmetric_backward=False,interaction_weights=b,weight_normalizer=norm)
    loss.backward();expected.backward();torch.testing.assert_close(loss,expected);torch.testing.assert_close(w.grad,ref.grad,rtol=1e-5,atol=1e-8)
    return {'passed':True,'model':model,'batch':len(u),'loss_abs_difference':float((loss-expected).detach().abs()),'gradient_max_abs_difference':float((w.grad-ref.grad).abs().max()),'cuda':True,
            'reference':'independent BPR formula' if model=='bpr_mf' else 'native CSR autograd; same fixed full graph, parameters and XSimGCL noise'}


def weight_jobs(settings_by_model, seeds=(42,43,44)):
    """Keep paired conditions within each model/seed; no cross-model pooling."""
    if not settings_by_model or set(settings_by_model)-{'bpr_mf','lightgcn','xsimgcl'}:raise ValueError('Unknown or empty model set')
    jobs=[];comparisons=[]
    for model in settings_by_model:
        prefix='' if model=='bpr_mf' else model+'_'
        comparisons.append({'model':model,'control':prefix+'vehicle_control','weighted':prefix+'vehicle_mismatch_half'})
    for seed in seeds:
        for pair in comparisons:
            base={k:v for k,v in settings_by_model[pair['model']].items() if k not in ['seed','variant']}
            for key in ['control','weighted']:
                variant=pair[key];settings=base|{'variant':variant}
                if key=='weighted':settings['interaction_weights']='features/train_vehicle_weights.npy'
                jobs.append({'run_id':f'{variant}_seed{seed}','purpose':'final_main','settings':settings,'seed':seed,'epochs':100,'patience':8})
    return jobs,comparisons


def run_weighted_preflights(root, store, plan, identity, guard=None):
    """Real per-model epochs and external optimizer/RNG recovery gate."""
    from .campaign_io import CampaignJournal,DurableRun
    from .experiment_plan import plan_hash
    from .train_gpu import run_training
    root=Path(root);context=plan_hash(plan);receipts={}
    weights_sha=json.loads((root/'features/train_vehicle_weights.json').read_text())['weights_sha256']
    for config in plan['preflights']:
        model=config['model'];marker=f'preflight/{model}/verified.json';receipt=store.get_json(marker)
        if receipt:
            assert receipt['passed'] and receipt['plan_hash']==context and receipt['training_settings']['interaction_weighting']['weights_sha256']==weights_sha
            receipts[model]=receipt;continue
        journal=CampaignJournal(root/f'results/preflight_{model}/journal',store,plan,identity,guard)
        journal.progress('weighted_kernel_verification',{'run_id':f'preflight_{model}_seed42','model':model,'seed':42},force=True)
        kernel=verify_cuda_weighted_objective(root,config)
        store.put_json(kernel|{'weights_sha256':weights_sha},f'preflight/{model}/weighted_objective.json',journal.staging)
        params=dict(config);params.pop('model');params['interaction_weights']=root/params['interaction_weights']
        run_id=f'preflight_{model}_seed42';first=root/f'results/preflight_{model}/source'
        completed=store.get_json(f'runs/{run_id}/completed.json')
        if not completed:
            candidate=store.get_json(f'runs/{run_id}/restore.json',{})
            if candidate.get('status')=='completed':completed=candidate
        if completed:
            verified=completed['summary'];assert verified['preflight_plan_hash']==context
        else:
            durable=DurableRun(journal,run_id,first);previous=durable.restore()
            if previous is None:
                run_training(root/'data',first,model,**params,epochs=1,eval_every=1,verify_resume=True,save_recommendations=False,event_hook=durable.hook,run_metadata={'purpose':'weighted external recovery preflight','plan_hash':context})
            recovered=root/f'results/preflight_{model}/recovered';durable=DurableRun(journal,run_id,recovered);checkpoint=durable.restore();assert checkpoint
            target=max(2,int(durable.pointer['generations'][-1]['epoch'])+1)
            verified=run_training(root/'data',recovered,model,**params,epochs=target,eval_every=1,resume=checkpoint,verify_resume=True,save_recommendations=False,event_hook=durable.hook,run_metadata={'purpose':'external restore then next epoch','plan_hash':context})
            verified['preflight_plan_hash']=context;durable.completed(verified)
        assert verified['completed_epochs']>=2 and verified['resume_check']['passed']
        receipt={'passed':True,'plan_hash':context,'model':model,'external_restore_then_epoch':verified['completed_epochs'],'optimizer_rng_replay':verified['resume_check'],'training_settings':verified['settings'],'worker':identity}
        store.put_json(receipt,marker,journal.staging);receipts[model]=receipt
        for folder in [first,root/f'results/preflight_{model}/recovered']:
            for name in ['last.pt','best.pt']:(folder/name).unlink(missing_ok=True)
        print(json.dumps({'weighted_model_preflight_verified':model,**receipt}),flush=True)
    result={'passed':True,'plan_hash':context,'models':receipts,'weights_sha256':weights_sha}
    store.put_json(result,'preflight/verified.json',root/'transfer');return result


def finish_report(output, store, plan, records, journal):
    import csv
    from .experiment_report import paired_bootstrap
    report=Path(output)/'report';report.mkdir(exist_ok=True,parents=True);rows=[];vectors={}
    for job in plan['jobs']:
        rec=records[job['run_id']];summary=rec['summary'];metrics=summary['test_metrics']['all']
        rows.append({'run':job['run_id'],'model':job['settings'].get('model','bpr_mf'),'seed':job['seed'],'variant':job['settings']['variant'],'best_epoch':summary['best_epoch'],
                     'ndcg':metrics['ndcg']['mean'],'recall':metrics['recall']['mean'],'aplt':metrics['aplt']['mean'],'arp':metrics['arp']['mean'],
                     'tail_recall10':metrics['tail_recall10']['mean']})
        artifact=next(a for a in rec['artifacts'] if a['relative']=='test/per_user.parquet')
        local=store.download(artifact['id'],report/(job['run_id']+'.parquet'),artifact['sha256'])
        frame=pl.read_parquet(local).sort('uid');vectors[job['settings']['variant'],job['seed']]=(frame['uid'].to_numpy(),frame['ndcg'].to_numpy());local.unlink()
    paired=[]
    comparisons=plan.get('comparisons',[{'model':'bpr_mf','control':'vehicle_control','weighted':'vehicle_mismatch_half'}])
    for pair in comparisons:
        for seed in plan['seeds']:
            ua,a=vectors[pair['control'],seed];ub,b=vectors[pair['weighted'],seed];assert np.array_equal(ua,ub)
            ci=paired_bootstrap(b-a,progress=lambda:journal.progress('paired_bootstrap'))
            paired.append({'model':pair['model'],'control':pair['control'],'weighted':pair['weighted'],'seed':seed,'ndcg_delta':float((b-a).mean()),'ci95':ci})
    means=[]
    for variant in dict.fromkeys(j['settings']['variant'] for j in plan['jobs']):
        values=[r for r in rows if r['variant']==variant]
        means.append({'variant':variant,**{f'{metric}_{stat}':float(getattr(np,stat)([r[metric] for r in values],**({'ddof':1} if stat=='std' else {}))) for metric in ['ndcg','recall','aplt','arp'] for stat in ['mean','std']}})
    for name,values in [('all_seed_results.csv',rows),('seed_summary.csv',means)]:
        with (report/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    write_json(report/'paired_by_seed.json',paired)
    (report/'RESULTS.md').write_text(f'# Vehicle-weighted training\n\n{len(rows)} predeclared runs completed. See all_seed_results.csv, seed_summary.csv and paired_by_seed.json. Models and variants are reported separately. Seed SD and per-seed paired user intervals are different uncertainty estimates. Inferred mismatch is not verified dislike.\n')
    for p in report.iterdir():store.put(p,'report/'+p.name)
    journal.log('vehicle_weight_report_completed',{'runs':len(rows)})
