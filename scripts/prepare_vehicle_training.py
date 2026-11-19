"""Freeze paired confidence-weight studies; never resume completed studies."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from colab_bundle import build_bundle,sha256
from automotive_recsys.audit import write_json
from automotive_recsys.drive_store import DriveStore
from automotive_recsys.experiment_plan import plan_hash
from automotive_recsys.vehicle_training import weight_jobs

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=['bpr_mf','lightgcn','xsimgcl'],default=['bpr_mf'])
    parser.add_argument('--cu-budget',type=float,default=12)
    args=parser.parse_args();destination=args.directory.resolve();destination.mkdir(parents=True,exist_ok=True)
    config_path=destination/'config.json'
    if config_path.exists():raise SystemExit('Existing study: inspect its frozen config rather than overwriting')
    if len(set(args.models))!=len(args.models) or not 0<args.cu_budget<1000:raise ValueError('Invalid models/budget')
    bases={model:{k:v for k,v in json.loads((ROOT/f'results/campaigns/automotive-20261104T204146Z/review/final_{model}_seed42.json').read_text())['summary']['settings'].items() if k not in ['seed','variant']} for model in args.models}
    inputs=ROOT/'results/vehicle-v2-20261117'
    extra=[(inputs/'v2'/name,'vehicles/v2/'+name) for name in ['user_vehicles.parquet','user_vehicles_items.parquet','product_fitment.parquet']]
    extra += [(inputs/'provider/vehicle_fit.py','provider/vehicle_fit.py')]
    bundle,manifest=build_bundle(extra,destination/'bundle')
    for model in args.models:
        reference=json.loads((ROOT/f'results/campaigns/automotive-20261104T204146Z/review/final_{model}_seed42.json').read_text())['summary']['inputs']
        for name,meta in reference.items():
            if manifest['files']['data/'+name]!=meta['sha256']:raise ValueError('Original training input differs: '+name)
    source_hash=hashlib.sha256(json.dumps({k:v for k,v in manifest['files'].items() if k.startswith(('src/','scripts/'))},sort_keys=True).encode()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');campaign_id='vehicle-weight-'+stamp
    jobs,comparisons=weight_jobs(bases)
    plan={'campaign_id':campaign_id,'kind':'vehicle_weight_training','seeds':[42,43,44],'planned_run_count':len(jobs),'jobs':jobs,'comparisons':comparisons,
          'cu_budget':args.cu_budget,'minimum_balance':30,'eval_every':5,'candidate_top_n':100,
          'preflight_settings':bases[args.models[0]]|{'variant':'vehicle_weight_preflight','interaction_weights':'features/train_vehicle_weights.npy'},
          'metadata':{'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_manifest_sha256':source_hash,
          'working_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
          'design':'Matched from-scratch '+','.join(args.models)+'; paired seeds/train order/negatives/model-noise/graph/L2/CL. v2 provider no-year A1+A2(n_items>=2). Unknown/match retain1; mismatch .5. Normalize BPR only by fixed train mean; never relabel negatives. No tuning of mismatch factor.',
          'selection':'train only; valid full-catalog checkpoint/early stopping; all jobs frozen before test; no train+valid refit',
          'limitations':'Inferred vehicle inconsistency is not verified dislike; A2 circularity and exploratory metadata/test-development history remain.'}}
    if any(m!='bpr_mf' for m in args.models):
        plan['preflights']=[base|{'variant':'preflight_weighted_'+model,'interaction_weights':'features/train_vehicle_weights.npy'} for model,base in bases.items()]
        prior=json.loads((ROOT/'docs/results/vehicle-weight-training-2026-11-17/vehicle_weight_audit.json').read_text())
        for path,digest in prior['source_sha256'].items():
            if manifest['files'][path]!=digest:raise ValueError('Vehicle inputs differ from BPR treatment')
        plan['metadata']['expected_weights_sha256']=prior['weights_sha256']
    credential=Path.home()/'.config/automotive-recsys/drive/gcloud/application_default_credentials.json';os.chmod(credential,0o600)
    store=DriveStore(credential);quota=store.quota()['storageQuota']
    if quota.get('limit') and int(quota['limit'])-int(quota.get('usage',0))<12*2**30:raise RuntimeError('Insufficient Drive checkpoint space')
    folder=store.create_root(campaign_id);staging=destination/'staging';uploaded=[]
    for part in manifest['parts']:
        item=store.put(bundle/part['name'],'inputs/'+part['name']);uploaded.append(part|{'id':item['id']})
        print('Verified Drive input part '+part['name'],flush=True)
    source=store.put(bundle/'bundle-manifest.json','inputs/bundle-manifest.json')
    owner='ubuntu-'+campaign_id.lower()
    cfg={'campaign_id':campaign_id,'root_id':folder['id'],'drive_url':folder['webViewLink'],'manifest':{'id':source['id'],'sha256':source['sha256']},
         'parts':uploaded,'plan':plan,'local_dir':str(destination),'gpu':'L4','controller_id':owner,
         'environment_script':'scripts/colab_vehicle_training_env.py','fresh_kernel_before_launch':True}
    write_json(config_path,cfg);write_json(destination/'plan.json',plan)
    store.put_json(plan,'experiment_plan.json',staging);store.put_json({'controller_id':owner},'controller_owner.json',staging)
    store.put_json({'stop_requested':False,'cu_budget':args.cu_budget},'control.json',staging)
    store.put_json({'campaign_id':campaign_id,'status':'prepared','completed_runs':0,'planned_runs':len(jobs)},'progress.json',staging)
    print(json.dumps({'config':str(config_path),'drive_url':cfg['drive_url'],'plan_hash':plan_hash(plan)}),flush=True)


if __name__=='__main__':main()
