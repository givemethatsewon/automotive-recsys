"""Freeze a separate six-run confidence-weight study; never resume old studies."""
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

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();destination=args.directory.resolve();destination.mkdir(parents=True,exist_ok=True)
    config_path=destination/'config.json'
    if config_path.exists():raise SystemExit('Existing study: inspect its frozen config rather than overwriting')
    record=json.loads((ROOT/'results/campaigns/automotive-20261104T204146Z/review/final_bpr_mf_seed42.json').read_text())
    base={k:v for k,v in record['summary']['settings'].items() if k not in ['seed','variant']}
    inputs=ROOT/'results/vehicle-v2-20261117'
    extra=[(inputs/'v2'/name,'vehicles/v2/'+name) for name in ['user_vehicles.parquet','user_vehicles_items.parquet','product_fitment.parquet']]
    extra += [(inputs/'provider/vehicle_fit.py','provider/vehicle_fit.py')]
    bundle,manifest=build_bundle(extra,destination/'bundle')
    source_hash=hashlib.sha256(json.dumps({k:v for k,v in manifest['files'].items() if k.startswith(('src/','scripts/'))},sort_keys=True).encode()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');campaign_id='vehicle-weight-'+stamp
    jobs=[]
    for seed in [42,43,44]:
        for variant in ['vehicle_control','vehicle_mismatch_half']:
            settings=base|{'variant':variant}
            if variant=='vehicle_mismatch_half':settings['interaction_weights']='features/train_vehicle_weights.npy'
            jobs.append({'run_id':f'{variant}_seed{seed}','purpose':'final_main','settings':settings,'seed':seed,'epochs':100,'patience':8})
    plan={'campaign_id':campaign_id,'kind':'vehicle_weight_training','seeds':[42,43,44],'planned_run_count':6,'jobs':jobs,
          'cu_budget':12,'minimum_balance':30,'eval_every':5,'candidate_top_n':100,
          'preflight_settings':base|{'variant':'vehicle_weight_preflight','interaction_weights':'features/train_vehicle_weights.npy'},
          'metadata':{'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_manifest_sha256':source_hash,
          'working_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
          'design':'Matched from-scratch BPR-MF; same seeds/train order/negatives/L2. v2 provider no-year A1+A2(n_items>=2). Unknown/match retain1; mismatch .5. Normalize BPR by fixed train mean; never relabel negatives. No tuning of mismatch factor.',
          'selection':'train only; valid full-catalog checkpoint/early stopping; fixed six jobs before test; no train+valid refit',
          'limitations':'Inferred vehicle inconsistency is not verified dislike; A2 circularity and exploratory metadata/test-development history remain.'}}
    credential=Path.home()/'.config/automotive-recsys/drive/gcloud/application_default_credentials.json';os.chmod(credential,0o600)
    store=DriveStore(credential);quota=store.quota()['storageQuota']
    if quota.get('limit') and int(quota['limit'])-int(quota.get('usage',0))<12*2**30:raise RuntimeError('Insufficient Drive checkpoint space')
    folder=store.create_root(campaign_id);staging=destination/'staging';uploaded=[]
    for part in manifest['parts']:
        item=store.put(bundle/part['name'],'inputs/'+part['name']);uploaded.append(part|{'id':item['id']})
        print('Verified Drive input part '+part['name'],flush=True)
    source=store.put(bundle/'bundle-manifest.json','inputs/bundle-manifest.json')
    owner='ubuntu-vehicle-weight-1117'
    cfg={'campaign_id':campaign_id,'root_id':folder['id'],'drive_url':folder['webViewLink'],'manifest':{'id':source['id'],'sha256':source['sha256']},
         'parts':uploaded,'plan':plan,'local_dir':str(destination),'gpu':'L4','controller_id':owner,
         'environment_script':'scripts/colab_vehicle_training_env.py','fresh_kernel_before_launch':True}
    write_json(config_path,cfg);write_json(destination/'plan.json',plan)
    store.put_json(plan,'experiment_plan.json',staging);store.put_json({'controller_id':owner},'controller_owner.json',staging)
    store.put_json({'stop_requested':False,'cu_budget':12},'control.json',staging)
    store.put_json({'campaign_id':campaign_id,'status':'prepared','completed_runs':0,'planned_runs':6},'progress.json',staging)
    print(json.dumps({'config':str(config_path),'drive_url':cfg['drive_url'],'plan_hash':plan_hash(plan)}),flush=True)


if __name__=='__main__':main()
