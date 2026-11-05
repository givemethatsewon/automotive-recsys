import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from automotive_recsys import campaign_io, drive_store
from test_experiment_pipeline import FakeStore, FakeJournal


class OperationalIOTests(unittest.TestCase):
    def test_normal_save_is_spaced_while_best_save_and_restore_still_work(self):
        original_run, original_store = campaign_io.DurableRun, drive_store.DriveStore
        spec = importlib.util.spec_from_file_location("operational_io_test", Path(__file__).resolve().parents[1]/"scripts/operational_io.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        try:
            with patch.dict(sys.modules, {"drive_store": SimpleNamespace(DriveStore=original_store)}):
                metadata = module.install(600)
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp); output=root/"source"; output.mkdir()
                store=FakeStore(); journal=FakeJournal(root,store)
                run=campaign_io.DurableRun(journal,"operational_fixture",output)
                (output/"best.pt").write_bytes(b"best")
                (output/"last.pt").write_bytes(b"state")
                (output/"manifest.json").write_text(json.dumps({"settings":{"model":"xsimgcl"}}))
                run.last_sync=time.monotonic()-200
                run.sync_checkpoint(2,1)
                self.assertFalse(run.pointer)
                run.sync_checkpoint(2,1,force=True)
                self.assertEqual(run.pointer["generations"][-1]["epoch"],2)
                run.hook("run_started",{})
                document=json.loads((output/"manifest.json").read_text())
                self.assertEqual(document["settings"],{"model":"xsimgcl"})
                self.assertEqual(document["operational_io"],metadata)
                restored=campaign_io.DurableRun(journal,"operational_fixture",root/"restored").restore()
                self.assertEqual(restored.read_bytes(),b"state")
        finally:
            campaign_io.DurableRun=original_run; drive_store.DriveStore=original_store
