import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from automotive_recsys.gpu_kernels import objective
from automotive_recsys.vehicle_training import confidence_weights,load_weights,sha,weight_jobs


class VehicleTrainingTests(unittest.TestCase):
    def test_graph_jobs_are_twelve_unique_paired_conditions_with_frozen_recipes(self):
        from automotive_recsys.experiment_plan import base_settings
        bases={m:base_settings(m) for m in ['lightgcn','xsimgcl']}
        jobs,pairs=weight_jobs(bases)
        self.assertEqual(len(jobs),12);self.assertEqual(len({j['run_id'] for j in jobs}),12)
        self.assertEqual({p['model'] for p in pairs},{'lightgcn','xsimgcl'})
        for a,b in zip(jobs[::2],jobs[1::2]):
            self.assertEqual(a['seed'],b['seed'])
            unweighted={k:v for k,v in a['settings'].items() if k!='variant'}
            weighted={k:v for k,v in b['settings'].items() if k not in ['variant','interaction_weights']}
            self.assertEqual(unweighted,weighted)
            self.assertEqual(b['settings']['interaction_weights'],'features/train_vehicle_weights.npy')
        self.assertNotIn('interaction_weights',bases['lightgcn'])
        with self.assertRaises(ValueError):weight_jobs({})

    def test_weighted_graph_loss_and_gradient_match_dense_reference_with_cl_unchanged(self):
        from automotive_recsys.gpu_kernels import normalized_graph
        graph=normalized_graph([0,0,1],[0,1,2],2,3).to_dense().double()
        u,p,n=torch.tensor([0,1]),torch.tensor([2,4]),torch.tensor([4,2]);weights=torch.tensor([.5,1.],dtype=torch.float64)
        for model in ['lightgcn','xsimgcl']:
            torch.manual_seed(44);a=torch.randn(5,3,dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
            noise=[torch.rand_like(a) for _ in range(3)]
            actual=objective(a,graph.to_sparse_csr(),u,p,n,model=model,noise=noise,symmetric_backward=True,regularization=.01,cl_weight=.1,interaction_weights=weights,weight_normalizer=.9)
            current=b;states=[b] if model=='lightgcn' else [];contrast=None
            for layer in range(3):
                current=graph@current
                if model=='xsimgcl':current=current+.1*current.sign()*F.normalize(noise[layer],dim=-1)
                states.append(current)
                if layer==0:contrast=current
            final=torch.stack(states).mean(0)
            expected=(F.softplus((final[u]*final[n]).sum(1)-(final[u]*final[p]).sum(1))*weights).mean()/.9+.01*sum(b[x].square().sum() for x in [u,p,n])/(2*len(u))
            if model=='xsimgcl':
                for ids in [torch.unique(u),torch.unique(p)]:
                    logits=F.normalize(final[ids],dim=-1)@F.normalize(contrast[ids],dim=-1).T/.2
                    expected=expected-.1*F.log_softmax(logits,dim=1).diagonal().mean()
            actual.backward();expected.backward();torch.testing.assert_close(actual,expected);torch.testing.assert_close(a.grad,b.grad)

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
        from automotive_recsys.experiment_plan import base_settings
        for models in [['bpr_mf'],['lightgcn','xsimgcl']]:
            with self.subTest(models=models),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);records={};jobs,pairs=weight_jobs({m:base_settings(m) for m in models})
                for job in jobs:
                    name=job['run_id'];path=root/(name+'.parquet');model=job['settings']['model']
                    bonus=.001 if 'interaction_weights' in job['settings'] else 0
                    value=models.index(model)*.2+(job['seed']-41)*.01+bonus
                    pl.DataFrame({'uid':[0,1],'ndcg':[value,value]}).write_parquet(path)
                    metrics={k:{'mean':value} for k in ['ndcg','recall','aplt','arp','tail_recall10']}
                    records[name]={'summary':{'best_epoch':1,'test_metrics':{'all':metrics}},'artifacts':[{'relative':'test/per_user.parquet','id':str(path),'sha256':sha(path)}]}
                class Store:
                    def download(self,source,dest,expected):
                        assert sha(source)==expected;Path(dest).parent.mkdir(exist_ok=True,parents=True);shutil.copy(source,dest);return Path(dest)
                    def put(self,*args):pass
                class Journal:
                    def progress(self,*args,**kwargs):pass
                    def log(self,*args,**kwargs):pass
                plan={'jobs':jobs,'seeds':[42,43,44]}
                if models!=['bpr_mf']:plan['comparisons']=pairs
                finish_report(root/'out',Store(),plan,records,Journal())
                summary=pl.read_csv(root/'out/report/seed_summary.csv');self.assertEqual(summary.height,2*len(models))
                for sd in summary['ndcg_std']:self.assertAlmostEqual(sd,.01)
                for i,model in enumerate(models):
                    actual=summary.filter(pl.col('variant')==pairs[i]['control'])['ndcg_mean'][0]
                    self.assertAlmostEqual(actual,i*.2+.02)
                paired=json.loads((root/'out/report/paired_by_seed.json').read_text());self.assertEqual(len(paired),3*len(models))
                for p in paired:self.assertAlmostEqual(p['ndcg_delta'],.001)


if __name__=='__main__':unittest.main()
