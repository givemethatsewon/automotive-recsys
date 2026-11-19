import json
from pathlib import Path
import tempfile
import unittest
from automotive_recsys.durable_stages import DurableStages
from test_experiment_pipeline import FakeStore


class CampaignStore(FakeStore):
    def put(self, source, path):
        result = super().put(source, path)
        result["appProperties"] = {"sha256": result["sha256"]}
        return result


class DurableStagesTests(unittest.TestCase):
    def make(self, root, store, context=None):
        return DurableStages(
            store,
            root / "journal",
            "fixture",
            root / "out",
            context or {"input": "fixed"},
            ["valid", "test"],
        )

    def test_external_restore_and_old_pointer_survives_failed_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CampaignStore()
            s = self.make(root, store)
            (s.output / "valid.json").write_text('{"value":1}')
            s.commit("valid", cursor=32, done=True)
            original = store.put_json

            def fail(value, path, staging):
                if path.endswith("/restore.json"):
                    raise IOError("Injected commit failure")
                return original(value, path, staging)

            store.put_json = fail
            (s.output / "valid.json").write_text('{"value":2}')
            with self.assertRaises(IOError):
                s.commit("test", cursor=64)
            store.put_json = original
            recovered = self.make(root / "fresh", store)
            self.assertTrue(recovered.restore())
            self.assertEqual(
                json.loads((recovered.output / "valid.json").read_text()), {"value": 1}
            )
            self.assertEqual(recovered.done, {"valid"})
            self.assertEqual(recovered.pointer["position"], [0, 0, 32])

    def test_changed_input_and_corrupt_blob_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CampaignStore()
            s = self.make(root, store)
            (s.output / "state.json").write_text("{}")
            s.commit("valid", done=True)
            with self.assertRaises(ValueError):
                self.make(root / "wrong", store, {"input": "changed"})
            artifact = s.pointer["artifacts"][0]
            store.files[artifact["id"]]["sha256"] = "invalid"
            with self.assertRaises(AssertionError):
                self.make(root / "corrupt", store).restore()

    def test_completion_marker_interruption_is_repaired_after_verified_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CampaignStore()
            s = self.make(root, store)
            (s.output / "summary.json").write_text('{"status":"completed"}')
            original = store.put_json

            def fail(value, path, staging):
                if path.endswith("/completed.json"):
                    raise IOError("Interrupted completion marker")
                return original(value, path, staging)

            store.put_json = fail
            with self.assertRaises(IOError):
                s.commit("test", done=True, complete=True)
            store.put_json = original
            recovered = self.make(root / "fresh", store)
            self.assertTrue(recovered.restore())
            self.assertEqual(
                store.get_json("runs/fixture/completed.json")["status"], "completed"
            )
            self.assertEqual(recovered.journal.state["completed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
