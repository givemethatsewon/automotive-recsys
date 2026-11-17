"""Actual CUDA/Drive crash-and-restore test before any real v2 evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import polars as pl

from automotive_recsys.audit import fingerprint
from automotive_recsys.drive_store import DriveStore
from automotive_recsys.durable_stages import atomic_json
from automotive_recsys.gpu_kernels import full_catalog_topk


def fixture(root,provider):
    import torch
    data=root/'data';data.mkdir(parents=True,exist_ok=True)
    nusers,nitems=8,1300
    train=pl.DataFrame({'uid':np.arange(nitems,dtype='int32')%nusers,'iid':np.arange(nitems,dtype='int32'),'timestamp':np.arange(nitems,dtype='int64')+1000}).sort('uid','iid')
    answers=[]
    for u in range(nusers):answers.append([i for i in range(nitems) if i%nusers!=u][:4])
    valid=pl.DataFrame({'uid':np.repeat(np.arange(nusers,dtype='int32'),2),'iid':np.array([r[:2] for r in answers],dtype='int32').ravel(),'timestamp':np.repeat(10000,nusers*2).astype('int64')})
    test=valid.with_columns(pl.Series('iid',np.array([r[2:] for r in answers],dtype='int32').ravel()),pl.lit(20000).cast(pl.Int64).alias('timestamp'))
    users=pl.DataFrame({'uid':np.arange(nusers,dtype='int32'),'user_id':[f'fixture-{i}' for i in range(nusers)],'n_train':np.bincount(train['uid'].to_numpy()),'group':['6+']*nusers})
    items=pl.DataFrame({'iid':np.arange(nitems,dtype='int32'),'parent_asin':[f'fixture-item-{i}' for i in range(nitems)],'n_train':np.ones(nitems,dtype='int64'),'popular':np.arange(nitems)<1040})
    frames={'train':train,'valid':valid,'test':test,'user_map':users,'item_map':items,'seen_for_valid':train.select('uid','iid'),'seen_for_test':pl.concat([train.select('uid','iid'),valid.select('uid','iid')]).unique().sort('uid','iid')}
    for name,f in frames.items():f.write_parquet(data/(name+'.parquet'))
    pf=pl.DataFrame({'parent_asin':items['parent_asin'],'make':['Honda' if i%3 else 'Ford' for i in range(nitems)],'model':['Civic' if i%3 else 'F-150' for i in range(nitems)],'segment':['car']*nitems,'universal':np.arange(nitems)%11==0,'year_from':np.full(nitems,2009.),'year_to':np.full(nitems,2011.),'open_end':[False]*nitems})
    for version in ['inputs','v2']:
        folder=root/version;folder.mkdir(exist_ok=True)
        count=5 if version=='inputs' else 7
        uv=pl.DataFrame({'user_id':[f'fixture-{i}' for i in range(count)],'make':['Honda' if i%3 else 'Ford' for i in range(count)],'model':['Civic' if i%3 else 'F-150' for i in range(count)],'segment':['car']*count,'year':[2010.]*count,'last_seen':[9000]*count})
        uv.write_parquet(folder/'user_vehicles.parquet')
        uv.with_columns(pl.lit(2).alias('n_items'),pl.lit(None).cast(pl.Float64).alias('year')).write_parquet(folder/'user_vehicles_items.parquet')
        pf.write_parquet(folder/'product_fitment.parquet')
    (root/'provider').mkdir(exist_ok=True);shutil.copy(provider,root/'provider/vehicle_fit.py')
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    weights=torch.randn(nusers+nitems,8,device='cuda')*.1
    run='fixture_bpr';ck=root/'checkpoints'/run;ck.mkdir(parents=True,exist_ok=True)
    settings={'model':'bpr_mf','seed':42,'layers':3,'dim':8,'validation_batch':4}
    torch.save({'embedding':weights,'inputs':fingerprint(data),'settings':settings,'epoch':1},ck/'best.pt')
    for split in ['valid','test']:
        seen=frames['seen_for_'+split].sort('uid','iid');uid=np.arange(nusers,dtype='int32')
        iid,score=full_catalog_topk(weights,uid,nusers,seen['uid'].to_numpy(),seen['iid'].to_numpy(),k=100,batch_size=4,return_scores=True)
        out=root/'baseline'/run/split;out.mkdir(parents=True,exist_ok=True)
        np.savez(out/'recommendations.npz',uid=uid,iid=iid,score=score)
        atomic_json(out/'metrics.json',{'fixture':True})
    files={}
    for sub in ['data','inputs','v2','provider']:
        for p in (root/sub).iterdir():files[str(p.relative_to(root))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    atomic_json(root/'manifest.json',{'files':files,'source_sha256':{'test':'CUDA fixture'}})


def main():
    import torch
    assert torch.cuda.is_available()
    parent=Path(sys.argv[1]);root=parent/'preflight-fixture';root.mkdir(exist_ok=True)
    if not (root/'scripts').exists():(root/'scripts').symlink_to(parent/'scripts',target_is_directory=True)
    real=DriveStore(os.environ['AUTOMOTIVE_DRIVE_CREDENTIAL'],os.environ['AUTOMOTIVE_DRIVE_ROOT'])
    context=json.loads((parent/'run_config.json').read_text())['context']
    verified=real.get_json('verification_passed.json')
    if verified and verified.get('passed') and verified.get('context')==context:return
    fixture(root,parent/'provider/vehicle_fit.py')
    # A separate Drive folder isolates fixture progress from the real experiment.
    fixture_id=real.folder('preflight/'+context[:16])
    env=os.environ|{'AUTOMOTIVE_DRIVE_ROOT':fixture_id}
    script=parent/'scripts/vehicle_v2_experiment.py'
    command=[sys.executable,'-u',str(script),'--root',str(root),'--run','fixture_bpr','--output',str(root/'out'),'--planned-runs','1','--eval-user-batch','4']
    first=subprocess.run(command+['--crash-after-stage','valid_v1_exact:partial'],env=env)
    assert first.returncode==86,first.returncode
    store=DriveStore(os.environ['AUTOMOTIVE_DRIVE_CREDENTIAL'],fixture_id)
    pointer=store.get_json('runs/fixture_bpr/restore.json')
    assert pointer['stage']=='valid_v1_exact' and pointer['cursor']==4 and 'valid_v1_exact' not in pointer['completed_stages']
    # Remove every evaluation output. Only a verified Drive restore can skip it.
    shutil.rmtree(root/'out');shutil.rmtree(root/'journal-fixture_bpr')
    second=subprocess.run(command+['--crash-after-stage','test_v1_exact:partial'],env=env)
    assert second.returncode==86
    committed=store.get_json('runs/fixture_bpr/restore.json')
    assert {'valid_v1_exact','valid_v2_exact'} <= set(committed['completed_stages'])
    assert committed['stage']=='test_v1_exact' and committed['cursor']==4
    shutil.rmtree(root/'out');shutil.rmtree(root/'journal-fixture_bpr')
    final_run=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(final_run.stdout,flush=True);assert final_run.returncode==0
    assert 'skipped_verified_stage' in final_run.stdout
    resumed=json.loads((root/'out/test_results.json').read_text())
    baseline_env={k:v for k,v in env.items() if not k.startswith('AUTOMOTIVE_DRIVE_')}
    reference=root/'uninterrupted'
    subprocess.run([sys.executable,'-u',str(script),'--root',str(root),'--run','fixture_bpr','--output',str(reference),'--planned-runs','1','--eval-user-batch','4'],env=baseline_env,check=True)
    expected=json.loads((reference/'test_results.json').read_text())
    assert resumed==expected,'Crash/resume results differ from uninterrupted CUDA reference'
    final=store.get_json('runs/fixture_bpr/completed.json');assert final['status']=='completed'
    # Actual SHA-verified round trip of every committed fixture artifact.
    for item in final['artifacts']:store.download(item['id'],root/'roundtrip'/item['relative'],item['sha256'])
    receipt={'passed':True,'context':context,'cuda':True,'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
        'abrupt_process_exit':86,'forced_crashes':2,'partial_batch_resume':True,'test_recommendation_arrays_resume':True,'restored_from_external_drive_after_output_deletion':True,
        'completed_stage_skipped_after_restore':True,'results_equal_uninterrupted':True,
        'all_committed_artifacts_downloaded_with_sha256':True,'test_rows':len(resumed)}
    atomic_json(parent/'preflight_verified.json',receipt)
    real.put_json(receipt,'verification_passed.json',parent/'transfer-staging')
    print(json.dumps({'stage':'DURABLE_PREFLIGHT_VERIFIED',**receipt}),flush=True)


if __name__=='__main__':main()
