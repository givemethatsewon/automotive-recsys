"""One bounded Ubuntu controller: run, recover, report, release task VM.

No LLM watcher or scheduler is created. The Colab wrapper owns final cleanup.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/vehicle-v2-20261117'


def status(value):
    value['updated_at']=datetime.now(timezone.utc).isoformat()
    tmp=RUN/'controller_status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(RUN/'controller_status.json')


def main():
    RUN.mkdir(exist_ok=True,parents=True)
    status({'status':'running','session':'automotive-v2-1117','phase':'setup/rerank','training':False})
    command=[sys.executable,str(ROOT/'scripts/colab_vehicle_pilot.py'),'--session','automotive-v2-1117',
        '--directory',str(RUN),'--runs','final_bpr_mf_seed42','final_lightgcn_seed42','final_xsimgcl_seed42',
        '--pool','1000','--torch-pin','2.11.0','--v2-comparison']
    if (RUN/'allocation_recovery.json').exists():
        recovery=json.loads((RUN/'allocation_recovery.json').read_text())
        assert recovery['assignments_before']==0 and recovery['task']=='automotive-v2-1117'
        command += ['--adopt-owned-endpoint',recovery['endpoint']]
    try:
        code=subprocess.run(command,cwd=ROOT).returncode
        if code:raise RuntimeError(f'Colab wrapper exit {code}; inspect recovered output and controller.log')
        rows=[]
        for model in ['bpr_mf','lightgcn','xsimgcl']:
            folder=RUN/'output'/f'final_{model}_seed42'
            s=json.loads((folder/'summary.json').read_text());assert s['status']=='completed'
            results=json.loads((folder/'test_results.json').read_text())
            for r in results:
                if r['beta']==s['choices'][f"{r['case']}_{r['pool']}"]['beta']:
                    rows.append({'model':model,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))}})
        with (RUN/'selected_test_results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
        text=['# Vehicle v1/v2 comparison — seed42 exploratory results','','Validation selected beta; training unchanged; train-only profiles.','','|Model|Case|Pool|Beta|Test NDCG|Change vs baseline (%)|','|---|---|---:|---:|---:|---:|']
        for r in rows:
            if r['pool']==500:text.append(f"|{r['model']}|{r['case']}|500|{r['beta']}|{r['ndcg']:.9f}|{100*r['ndcg_relative_change']:+.3f}|")
        text+=['','Paired cleaning/rule/year differences and CI are in each output/<run>/paired_comparisons.json.',
               'This is not retraining and not an independent new holdout: test informed earlier work and extraction rule development.',
               'usage_after.txt and sessions_after.txt record the final account/resource observation.']
        (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n')
        status({'status':'completed','completed_models':3,'selected_rows':len(rows),'training':False,'results':'RESULTS.md','resource_receipts':['usage_after.txt','sessions_after.txt']})
    except Exception as e:
        status({'status':'failed','error':str(e),'training':False,'inspect':'controller.log and output/<run>/execution.log'})
        raise


if __name__=='__main__':main()
