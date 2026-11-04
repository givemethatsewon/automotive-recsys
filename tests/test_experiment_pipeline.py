import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
import torch

from automotive_recsys.campaign_io import DurableRun
from automotive_recsys.evaluate_run import evaluate_rankings
from automotive_recsys.experiment_plan import default_plan, tuning_jobs, depth_jobs, freeze_choices, final_jobs
from automotive_recsys.experiment_report import paired_bootstrap, build_report
from automotive_recsys.gpu_kernels import full_catalog_topk, normalized_graph, objective


class FakeStore:
    def __init__(self): self.files = {}; self.deleted = []
    def get_json(self, path, default=None):
        return json.loads(self.files[path]["content"]) if path in self.files else default
    def find(self, path): return self.files.get(path)
    def put(self, source, path):
        value = Path(source).read_bytes(); checksum = hashlib.sha256(value).hexdigest()
        result = {"id": path, "sha256": checksum, "content": value}
        self.files[path] = result
        return result
    def put_json(self, value, path, staging):
        content = json.dumps(value).encode()
        self.files[path] = {"id": path, "content": content, "sha256": hashlib.sha256(content).hexdigest()}
    def download(self, file_id, destination, expected_sha256=None):
        value = self.files[file_id]
        assert expected_sha256 == value["sha256"]
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(value["content"])
    def remove_verified_checkpoint(self, file_id, expected_sha256):
        assert self.files[file_id]["sha256"] == expected_sha256
        del self.files[file_id]; self.deleted.append(file_id)
    remove_verified_artifact = remove_verified_checkpoint


class FakeJournal:
    def __init__(self, root, store): self.store=store; self.staging=root; self.state={"current":{}}
    def progress(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def sync_small(self): pass


class ExperimentTests(unittest.TestCase):
    def test_plan_has_three_seeds_and_frozen_ablation_choices(self):
        plan = default_plan("test")
        tuning = tuning_jobs(plan)
        base = {name:next(job["settings"] for job in tuning if job["settings"]["model"]==name) for name in ["bpr_mf","lightgcn","xsimgcl"]}
        selection = freeze_choices(plan, base, base["lightgcn"] | {"layers":2}, {})
        jobs = final_jobs(plan, selection)
        self.assertEqual(len(tuning)+len(depth_jobs(plan,base["lightgcn"]))+len(jobs),plan["planned_run_count"])
        self.assertEqual(len(jobs),18)
        self.assertFalse(selection["test_used_for_selection"])
        self.assertEqual(selection["choices"]["xsimgcl_cl0"]["cl_weight"],0)

    def test_checkpoint_generation_retention_and_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); output=root/"run"; output.mkdir()
            store=FakeStore(); journal=FakeJournal(root,store)
            run=DurableRun(journal,"fixture",output)
            (output/"best.pt").write_bytes(b"best1")
            for epoch in [1,2,3]:
                (output/"last.pt").write_bytes(f"epoch{epoch}".encode())
                (output/"history.json").write_text(json.dumps({"epoch":epoch}))
                run.sync_checkpoint(epoch,1,force=True)
            self.assertEqual([g["epoch"] for g in run.pointer["generations"]],[2,3])
            self.assertTrue(any("state-e0001-" in path for path in store.deleted))
            (output/"last.pt").unlink()
            restored=DurableRun(journal,"fixture",output).restore()
            self.assertEqual(restored.read_bytes(),b"epoch3")
            self.assertEqual((output/"best.pt").read_bytes(),b"best1")

    def test_failed_new_pointer_does_not_mutate_previous_recovery_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); output=root/"run"; output.mkdir()
            store=FakeStore(); journal=FakeJournal(root,store)
            run=DurableRun(journal,"failure",output)
            for name,value in [("best.pt",b"best"),("last.pt",b"epoch1"),("history.json",b'{"epoch":1}')]:
                (output/name).write_bytes(value)
            run.sync_checkpoint(1,1,force=True)
            original_put=store.put_json
            def fail(*args,**kwargs): raise IOError("Simulated pointer commit failure")
            store.put_json=fail
            (output/"last.pt").write_bytes(b"epoch2")
            (output/"history.json").write_bytes(b'{"epoch":2}')
            with self.assertRaises(IOError): run.sync_checkpoint(2,1,force=True)
            store.put_json=original_put
            restored=DurableRun(journal,"failure",output).restore()
            self.assertEqual(restored.read_bytes(),b"epoch1")
            self.assertEqual(json.loads((output/"history.json").read_text())["epoch"],1)

    def test_paired_bootstrap_constant_difference(self):
        result=paired_bootstrap(np.full(100,0.125),repetitions=40)
        self.assertEqual(result["ci95_low"],0.125)
        self.assertEqual(result["ci95_high"],0.125)

    def test_report_keeps_deterministic_baseline_separate_from_three_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root)
            uid=np.arange(2,dtype=np.int32)
            iid=np.array([np.r_[22,np.arange(50,149)],np.r_[23,np.arange(50,149)]],dtype=np.int32)
            evaluated=evaluate_rankings(root,"test",uid,iid,np.tile(np.arange(100,0,-1,dtype=np.float32),(2,1)))
            store=FakeStore(); plan=default_plan("report_fixture")
            store.put_json({"summary":evaluated["summary"]},"baseline/test/completed.json",root)
            parquet=root/"per_user.parquet"; evaluated["per_user"].write_parquet(parquet)
            store.put(parquet,"baseline/test/per_user.parquet")
            base={job["settings"]["model"]:job["settings"] for job in tuning_jobs(plan)}
            selection=freeze_choices(plan,base,base["lightgcn"],{})
            records={}
            for job in final_jobs(plan,selection):
                run_id=job["run_id"]
                artifact=store.put(parquet,run_id+"/test/per_user.parquet")
                records[run_id]={"summary":{"settings":job["settings"]|{"seed":job["seed"]},"model":job["settings"]["model"],
                    "purpose":job["purpose"],"test_metrics":evaluated["summary"]["metrics"],"best_epoch":1,"completed_epochs":1,
                    "elapsed_seconds":2.0,"peak_allocated_MiB":42,"history":[{"epoch":1,"training_seconds":1.0,"valid_ndcg_at_10":1.0}]},
                    "artifacts":[{"relative":"test/per_user.parquet","id":artifact["id"],"sha256":artifact["sha256"]}]}
            build_report(root,store,plan,records)
            comparison=pl.read_csv(root/"report/model_comparison.csv").filter(pl.col("group")=="all")
            self.assertEqual(comparison.filter(pl.col("variant")=="popularity")["n_runs"][0],1)
            self.assertEqual(comparison.filter(pl.col("variant")=="bpr_mf")["n_runs"][0],3)
            self.assertIsNone(comparison.filter(pl.col("variant")=="popularity")["ndcg_seed_sd"][0])
            self.assertAlmostEqual(comparison["ndcg_mean"].mean(),1.0)
            self.assertTrue((root/"report/RESULTS.html").exists())

    def fixture(self, path):
        users=pl.DataFrame({"uid":np.arange(2,dtype=np.int32),"user_id":["u0","u1"],"n_train":[6,6],"group":["6+","6+"]})
        items=pl.DataFrame({"iid":np.arange(220,dtype=np.int32),"parent_asin":[str(i) for i in range(220)],"n_train":np.ones(220,dtype=np.int64),"popular":np.arange(220)<20})
        train=pl.DataFrame({"uid":np.repeat(np.arange(2,dtype=np.int32),6),"iid":np.arange(12,dtype=np.int32)})
        valid=pl.DataFrame({"uid":np.arange(2,dtype=np.int32),"iid":np.array([12,21],dtype=np.int32)})
        test=pl.DataFrame({"uid":np.arange(2,dtype=np.int32),"iid":np.array([22,23],dtype=np.int32)})
        for name,frame in [("user_map",users),("item_map",items),("train",train),("valid",valid),("test",test),("seen_for_valid",train),("seen_for_test",pl.concat([train,valid]))]: frame.write_parquet(path/(name+".parquet"))

    def test_evaluation_artifacts_and_test_history_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root)
            uid=np.arange(2,dtype=np.int32)
            iid=np.array([np.r_[12,np.arange(50,149)],np.r_[40,21,np.arange(50,148)]],dtype=np.int32)
            score=np.tile(np.arange(100,0,-1,dtype=np.float32),(2,1))
            result=evaluate_rankings(root,"valid",uid,iid,score)
            self.assertAlmostEqual(result["per_user"]["ndcg"][1],1/np.log2(3))
            self.assertEqual(result["per_user"]["tail_recall10"][0],None)
            self.assertEqual(result["exposure"].height,220)
            with self.assertRaises(ValueError): evaluate_rankings(root,"test",uid,iid,score)
            with self.assertRaises(ValueError): evaluate_rankings(root,"valid",uid[::-1],iid,score)

    def test_recorded_objective_components_keep_gradients(self):
        torch.manual_seed(4)
        graph=normalized_graph([0,0,1],[0,1,1],2,2,layout="coo")
        source=torch.randn(4,8,requires_grad=True)
        other=source.detach().clone().requires_grad_()
        ids=[torch.tensor([0,1]),torch.tensor([2,3]),torch.tensor([3,2])]
        noise=[torch.rand(4,8) for _ in range(3)]
        a=objective(source,graph,*ids,model="xsimgcl",noise=noise)
        b,parts=objective(other,graph,*ids,model="xsimgcl",noise=noise,return_components=True)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(b.detach(),parts["bpr"]+parts["regularization"]+0.1*parts["cl_raw"])
        a.backward();b.backward();torch.testing.assert_close(source.grad,other.grad)

    def test_returned_scores_correspond_to_ranked_ids(self):
        weights=torch.randn(2+220,4)
        uid=np.arange(2)
        iid,score=full_catalog_topk(weights,uid,2,np.array([0,1]),np.array([0,1]),k=10,return_scores=True)
        expected=(weights[:2]@weights[2:].T).numpy()
        np.testing.assert_allclose(score,np.take_along_axis(expected,iid,axis=1),rtol=1e-6)


if __name__=="__main__": unittest.main()
