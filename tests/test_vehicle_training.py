import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from automotive_recsys.gpu_kernels import objective
from automotive_recsys.vehicle_training import confidence_weights,load_weights,sha


class VehicleTrainingTests(unittest.TestCase):
    def test_unknown_and_match_are_retained_and_only_mismatch_downweighted(self):
        np.testing.assert_array_equal(confidence_weights([-1,0,1]),[1,.5,1])
        with self.assertRaises(ValueError):confidence_weights([2])
        with self.assertRaises(ValueError):confidence_weights([0],0)

    def test_weighted_objective_and_gradient_match_independent_reference(self):
        torch.manual_seed(42)
        a=torch.randn(7,3,dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
        u,p,n=torch.tensor([0,1,0]),torch.tensor([3,4,5]),torch.tensor([5,6,3]);weights=torch.tensor([.5,1,.5]);norm=.9
        actual=objective(a,None,u,p,n,model='bpr_mf',regularization=.01,interaction_weights=weights,weight_normalizer=norm)
        expected=(F.softplus((b[u]*b[n]).sum(1)-(b[u]*b[p]).sum(1))*weights).mean()/norm+.01*sum(b[x].square().sum() for x in [u,p,n])/(2*len(u))
        actual.backward();expected.backward();torch.testing.assert_close(actual,expected);torch.testing.assert_close(a.grad,b.grad)

    def test_all_one_weights_preserve_baseline_loss_and_gradient(self):
        torch.manual_seed(43);a=torch.randn(7,3,requires_grad=True);b=a.detach().clone().requires_grad_()
        u,p,n=torch.tensor([0,1,0]),torch.tensor([3,4,5]),torch.tensor([5,6,3])
        x=objective(a,None,u,p,n,model='bpr_mf');y=objective(b,None,u,p,n,model='bpr_mf',interaction_weights=torch.ones(3))
        x.backward();y.backward();torch.testing.assert_close(x,y,rtol=0,atol=0);torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_train_row_order_hash_and_weight_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train.parquet').write_bytes(b'ordered train fixture');path=root/'weights.npy';np.save(path,np.array([1,.5],dtype=np.float32))
            meta={'training_split':'train','train_sha256':sha(root/'train.parquet'),'weights_sha256':sha(path),'mismatch_weight':.5,'recipe':'fixture'}
            path.with_suffix('.json').write_text(json.dumps(meta));load_weights(path,root,2)
            (root/'train.parquet').write_bytes(b'different ordering')
            with self.assertRaises(ValueError):load_weights(path,root,2)
            (root/'train.parquet').write_bytes(b'ordered train fixture');np.save(path,np.array([1,1],dtype=np.float32))
            with self.assertRaises(ValueError):load_weights(path,root,2)

    def test_six_run_report_keeps_seed_sd_and_paired_users_separate(self):
        import shutil
        import polars as pl
        from automotive_recsys.vehicle_training import finish_report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=[];records={}
            for seed in [42,43,44]:
                for variant,bonus in [('vehicle_control',0),('vehicle_mismatch_half',.01)]:
                    name=f'{variant}_{seed}';path=root/(name+'.parquet')
                    value=(seed-41)*.1+bonus;pl.DataFrame({'uid':[0,1],'ndcg':[value,value]}).write_parquet(path)
                    jobs.append({'run_id':name,'seed':seed,'settings':{'variant':variant}})
                    metrics={k:{'mean':value} for k in ['ndcg','recall','aplt','arp','tail_recall10']}
                    records[name]={'summary':{'best_epoch':1,'test_metrics':{'all':metrics}},'artifacts':[{'relative':'test/per_user.parquet','id':str(path),'sha256':sha(path)}]}
            class Store:
                def download(self,source,dest,expected):
                    assert sha(source)==expected;Path(dest).parent.mkdir(exist_ok=True,parents=True);shutil.copy(source,dest);return Path(dest)
                def put(self,*args):pass
            class Journal:
                def progress(self,*args,**kwargs):pass
                def log(self,*args,**kwargs):pass
            finish_report(root/'out',Store(),{'jobs':jobs,'seeds':[42,43,44]},records,Journal())
            summary=pl.read_csv(root/'out/report/seed_summary.csv')
            self.assertAlmostEqual(summary['ndcg_std'][0],.1)
            paired=json.loads((root/'out/report/paired_by_seed.json').read_text());self.assertEqual(len(paired),3)
            self.assertAlmostEqual(paired[0]['ndcg_delta'],.01)


if __name__=='__main__':unittest.main()
