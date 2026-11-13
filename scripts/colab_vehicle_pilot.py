"""Stage/run/recover a bounded vehicle pilot with the project Colab CLI.

The only credential transfer is a private ADC file outside source/data bundles;
the remote bootstrap removes it immediately after read-only Drive downloads.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--session',required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--runs',nargs='+',default=['final_bpr_mf_seed42'],
        help='frozen campaign runs whose stored top100 rankings are reranked, in order')
    p.add_argument('--primary',default='both-model',choices=[f'{a}-{b}' for a in ['text','items','both'] for b in ['make','model']],
        help='pre-specified primary family, recorded before any test answer is read')
    p.add_argument('--pool',type=int,default=100,
        help='candidate pool size; >100 recomputes top-K from each run best checkpoint (no fitting)')
    p.add_argument('--torch-pin',default='',help='torch version installed in the isolated venv, e.g. 2.11.0 (training-time version)')
    p.add_argument('--candidate-diagnostic', action='store_true', help='Valid-only candidate availability and fixed-scale reranking; no test evaluation')
    args=p.parse_args()
    run=args.directory.resolve()
    run.mkdir(parents=True,exist_ok=True)
    remote='/content/automotive-vehicle-pilot'
    campaign=ROOT/'results/campaigns/automotive-20261104T204146Z'
    config=json.loads((campaign/'config.json').read_text())
    completed={run_id:json.loads((campaign/f'review/{run_id}.json').read_text()) for run_id in args.runs}
    assert len({json.dumps(c['summary']['inputs'],sort_keys=True) for c in completed.values()})==1, 'Runs use different inputs'
    files=[(f,'inputs/'+f.name) for f in sorted((run/'inputs').glob('*.parquet'))]
    assert {f.name for f,_ in files}=={'user_vehicles.parquet','user_vehicles_items.parquet','product_fitment.parquet'}
    files += [(f,f.relative_to(ROOT).as_posix()) for f in sorted((ROOT/'src').rglob('*.py'))]
    files += [(ROOT/f'scripts/{name}',f'scripts/{name}') for name in ['vehicle_pilot.py','candidate_pool.py']]
    if args.candidate_diagnostic:
        assert args.pool >= 1000
        files.append((ROOT/'scripts/candidate_diagnostic.py', 'scripts/candidate_diagnostic.py'))
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),
              'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff'],cwd=ROOT)).hexdigest(),
              'source_sha256':{rel:digest(f) for f,rel in files if rel.startswith(('src/','scripts/'))},
              'files':{rel:{'sha256':digest(f),'bytes':f.stat().st_size} for f,rel in files},
              'added_data_folder':'https://drive.google.com/drive/folders/1mYYDiornIreYYatRbICYNKmlPFhjPYRn',
              'baseline_runs':args.runs,'primary_family':args.primary,'candidate_pool':args.pool,'torch_pin':args.torch_pin,'baseline_campaign':config['campaign_id'],
              'candidate_diagnostic':args.candidate_diagnostic,
              'baseline':[],'data_parts':config['parts']}
    for name,meta in completed[args.runs[0]]['summary']['inputs'].items():manifest['files']['data/'+name]=meta
    for run_id,record in completed.items():
        for artifact in record['artifacts']:
            if artifact['relative'].endswith(('recommendations.npz','metrics.json')):
                rel=f'baseline/{run_id}/'+artifact['relative']
                manifest['baseline'].append({**artifact,'relative':rel})
                manifest['files'][rel]={'sha256':artifact['sha256']}
        if args.pool!=100:
            # Downloaded with the baseline artifacts; checked by sha only, not
            # listed in files because it is not a model input of the rerank.
            manifest['baseline'].append({**record['best'],'relative':f'checkpoints/{run_id}/best.pt'})
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=run/'upload.tar'
    with tarfile.open(archive,'w') as tar:
        for f,rel in files:tar.add(f,arcname=rel,recursive=False)
        tar.add(run/'manifest.json',arcname='manifest.json')
    parts=[]
    with archive.open('rb') as f:
        for index in range(100):
            block=f.read(16<<20)
            if not block:break
            path=run/f'upload.part{index:03d}'
            path.write_bytes(block)
            parts.append({'name':path.name,'sha256':digest(path),'bytes':len(block)})
    archive.unlink()
    def call(*a,check=True,capture=False):
        # exec stages are idempotent and the worker launch has its own PID
        # receipt. A websocket failure does not justify a duplicate worker.
        for attempt in range(3 if a[0]=='exec' else 1):
            r=subprocess.run([str(ROOT/'scripts/colab.sh'),*map(str,a)],cwd=ROOT,
                check=False,text=True,stdout=subprocess.PIPE if capture else None)
            if r.returncode==0:break
            if a[0]=='exec' and attempt<2:time.sleep(3*(attempt+1))
        if check:r.check_returncode()
        return r.stdout if capture else r
    before=call('usage',capture=True)
    (run/'usage_before.txt').write_text(before)
    print(before,flush=True)
    # New session is task-owned, never resume the completed campaign.
    owned=True
    try:
        call('new','-s',args.session,'--gpu','L4')
        init=run/'initialize.py'
        init.write_text("from pathlib import Path\nimport os,torch,json\n"
            +f"Path('{remote}').mkdir(exist_ok=True)\n"
            +"Path('/content/automotive-pilot-private').mkdir(mode=0o700,exist_ok=True)\n"
            +"assert torch.cuda.is_available()\nprint(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}))\n")
        call('exec','-s',args.session,'--timeout','120','-f',init)
        for part in parts:
            call('upload','-s',args.session,run/part['name'],remote+'/'+part['name'])
        credential=Path.home()/'.config/automotive-recsys/drive/gcloud/application_default_credentials.json'
        call('upload','-s',args.session,credential,'/content/automotive-pilot-private/drive.json')
        bootstrap=run/'bootstrap.py'
        bootstrap.write_text('''from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile,time
root=Path('/content/automotive-vehicle-pilot')
parts=PARTS
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
with (root/'upload.tar').open('wb') as out:
 for part in parts:
  p=root/part['name'];assert digest(p)==part['sha256']
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):out.write(b)
with tarfile.open(root/'upload.tar') as tar:tar.extractall(root,filter='data')
sys.path.insert(0,str(root/'src'))
subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(root/'venv')],check=True)
subprocess.run([sys.executable,'-m','pip','--python',str(root/'venv/bin/python'),'install','-q','polars>=1.30,<2','numba'],check=True)
if TORCH_PIN:
 # Recomputing rankings must use the training-time kernels; the archived
 # campaign ran on torch 2.11.0+cu130 and graph SpMM differs across versions.
 subprocess.run([sys.executable,'-m','pip','--python',str(root/'venv/bin/python'),'install','-q','torch=='+TORCH_PIN,'--index-url','https://download.pytorch.org/whl/cu130'],check=True)
 out=subprocess.run([str(root/'venv/bin/python'),'-c','import torch;assert torch.cuda.is_available();print(torch.__version__)'],check=True,text=True,capture_output=True).stdout.strip()
 assert out.startswith(TORCH_PIN),out
from automotive_recsys.drive_store import DriveStore
credential=Path('/content/automotive-pilot-private/drive.json')
os.chmod(credential,0o600)
try:
 store=DriveStore(credential)
 manifest=json.loads((root/'manifest.json').read_text())
 with (root/'base.tar').open('wb') as out:
  for part in manifest['data_parts']:
   path=store.download(part['id'],root/part['name'],part['sha256'])
   with path.open('rb') as f:
    for b in iter(lambda:f.read(8<<20),b''):out.write(b)
   path.unlink()
 with tarfile.open(root/'base.tar') as tar:
  members=[m for m in tar.getmembers() if m.name.startswith('data/')]
  tar.extractall(root,members=members,filter='data')
 for artifact in manifest['baseline']:
  store.download(artifact['id'],root/artifact['relative'],artifact['sha256'])
finally:
 credential.unlink(missing_ok=True)
(root/'prepared.json').write_text(json.dumps({'ready':True,'credential_removed':not credential.exists()}))
print('VEHICLE_PILOT_INPUTS_READY',flush=True)
'''.replace('parts=PARTS','parts='+repr(parts)).replace('TORCH_PIN',repr(args.torch_pin)))
        call('exec','-s',args.session,'--timeout','1200','-f',bootstrap)
        call('download','-s',args.session,remote+'/prepared.json',run/'prepared.json')
        prepared=json.loads((run/'prepared.json').read_text())
        assert prepared['ready'] and prepared['credential_removed']
        entry=run/'entry.py'
        worker=run/'worker.py'
        worker.write_text('''import json,os,subprocess,sys,tarfile
from pathlib import Path
root=Path('/content/automotive-vehicle-pilot')
output=root/'output';output.mkdir(exist_ok=True)
env=os.environ|{'PYTHONPATH':str(root/'src'),'NUMBA_NUM_THREADS':'4','POLARS_MAX_THREADS':'4'}
codes={}
def stream(command,log):
 process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in process.stdout:
  print(line,end='',flush=True);log.write(line);log.flush()
 return process.wait()
for run_id in RUNS:
 out=output/run_id;out.mkdir(exist_ok=True)
 with (out/'execution.log').open('w') as log:
  base=root/'baseline'/run_id
  codes[run_id]=0
  if DIAGNOSTIC:
   codes[run_id]=stream([sys.executable,'-u',str(root/'scripts/candidate_diagnostic.py'),'--root',str(root),'--run',run_id,'--output',str(out)],log)
  elif POOL!=100:
   codes[run_id]=stream([sys.executable,'-u',str(root/'scripts/candidate_pool.py'),'--root',str(root),'--run',run_id,'--k',str(POOL)],log)
   base=root/'pool'/run_id
   if (base/'pool_receipt.json').exists():
    import shutil;shutil.copy(base/'pool_receipt.json',out/'pool_receipt.json')
  if codes[run_id]==0 and not DIAGNOSTIC:
   codes[run_id]=stream([sys.executable,'-u',str(root/'scripts/vehicle_pilot.py'),'--root',str(root),
    '--model',run_id,'--baseline',str(base),'--output',str(out),'--primary',PRIMARY,'--candidates',str(POOL)],log)
  if POOL!=100:
   import shutil;shutil.rmtree(root/'pool'/run_id,ignore_errors=True);shutil.rmtree(root/'checkpoints'/run_id,ignore_errors=True)
 if codes[run_id]:break
code=max(codes.values())
(output/'exit.json').write_text(json.dumps({'returncode':code,'runs':codes}))
with tarfile.open('/content/automotive-vehicle-pilot-results.tar.gz','w:gz') as tar:tar.add(output,arcname='output')
import hashlib
archive=Path('/content/automotive-vehicle-pilot-results.tar.gz')
parts=[]
with archive.open('rb') as stream:
 for index in range(1000):
  block=stream.read(4<<20)
  if not block:break
  part=archive.with_name(f'vehicle-results.part{index:03d}');part.write_bytes(block)
  parts.append({'name':part.name,'sha256':hashlib.sha256(block).hexdigest(),'bytes':len(block)})
with archive.open('rb') as stream:archive_hash=hashlib.file_digest(stream,'sha256').hexdigest()
archive.with_name('vehicle-results-manifest.json').write_text(json.dumps({'sha256':archive_hash,'parts':parts}))
(root/'worker_done.json').write_text(json.dumps({'returncode':code}))
'''.replace('RUNS',repr(args.runs)).replace('PRIMARY',repr(args.primary)).replace('POOL',repr(args.pool)).replace('DIAGNOSTIC',repr(args.candidate_diagnostic)))
        call('upload','-s',args.session,worker,remote+'/worker.py')
        entry.write_text('''import json,os,subprocess
from pathlib import Path
root=Path('/content/automotive-vehicle-pilot')
assert json.loads((root/'prepared.json').read_text())['ready']
receipt=root/'worker_started.json'
if receipt.exists():
 pid=json.loads(receipt.read_text())['pid']
 assert (root/'worker_done.json').exists() or Path(f'/proc/{pid}').exists(),'Worker disappeared'
 print('VEHICLE_PILOT_ALREADY_STARTED',flush=True)
else:
 with (root/'worker.log').open('w') as log:
  process=subprocess.Popen([str(root/'venv/bin/python'),str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt.write_text(json.dumps({'pid':process.pid}))
 print('VEHICLE_PILOT_STARTED',flush=True)
''')
        call('exec','-s',args.session,'--timeout','120','-f',entry)
        call('download','-s',args.session,remote+'/worker_started.json',run/'worker_started.json')
        assert json.loads((run/'worker_started.json').read_text())['pid']>0
        deadline=time.monotonic()+1800*len(args.runs)
        while time.monotonic()<deadline:
            receipt=run/'worker_done.json'
            call('download','-s',args.session,remote+'/worker_done.json',receipt,check=False)
            if receipt.exists():break
            time.sleep(20)
        else:raise TimeoutError('Pilot did not finish within the time limit')
        call('download','-s',args.session,'/content/vehicle-results-manifest.json',run/'result-parts.json')
        receipt_parts=json.loads((run/'result-parts.json').read_text())
        def recover(part):
            path=run/part['name']
            call('download','-s',args.session,'/content/'+part['name'],path)
            assert path.stat().st_size==part['bytes'] and digest(path)==part['sha256']
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(recover,receipt_parts['parts']))
        with (run/'results.tar.gz').open('wb') as stream:
            for part in receipt_parts['parts']:stream.write((run/part['name']).read_bytes())
        assert digest(run/'results.tar.gz')==receipt_parts['sha256']
        with tarfile.open(run/'results.tar.gz') as tar:tar.extractall(run,filter='data')
        assert json.loads(receipt.read_text())['returncode']==0, 'See output/<run>/execution.log'
        for run_id in args.runs:
            assert json.loads((run/'output'/run_id/'summary.json').read_text())['status']=='completed', run_id
        call('log','-s',args.session,'-o',run/'session.ipynb')
        print('VEHICLE_PILOT_RECOVERED',flush=True)
    finally:
        if owned:
            # A failed entry still produces an archive before raising. Recover
            # it opportunistically, then always release this named session.
            if not (run/'results.tar.gz').exists():
                call('download','-s',args.session,'/content/automotive-vehicle-pilot-results.tar.gz',run/'failed-results.tar.gz',check=False)
            call('stop','-s',args.session,check=False)
            after=call('usage',capture=True)
            sessions=call('sessions',capture=True)
            (run/'usage_after.txt').write_text(after)
            (run/'sessions_after.txt').write_text(sessions)
            print(after,sessions,flush=True)


if __name__=='__main__':main()
