import unittest

from automotive_recsys.controller_policy import progress_token, unsuccessful_workers, transient_network_error


class ControllerPolicyTests(unittest.TestCase):
    def test_productive_renewals_do_not_exhaust_recovery_limit(self):
        workers = [{"start_progress_token": [8, "x", epoch]} for epoch in range(1, 15)]
        self.assertEqual(unsuccessful_workers(workers, [8, "x", 15]), 0)
        self.assertEqual(unsuccessful_workers([{"start_progress_token": [8, "x", 15]}] * 6, [8, "x", 15]), 6)

    def test_worker_startup_and_repeated_upload_do_not_fake_progress(self):
        previous = [8, "tune_xsimgcl_08", 5]
        self.assertEqual(progress_token({"completed_runs": 0}, previous), previous)
        self.assertEqual(progress_token({"completed_runs": 8}, previous), previous)
        self.assertEqual(progress_token({"completed_runs": 8, "durable_checkpoint": "tune_xsimgcl_08 epoch 5 (new timestamp)"}, previous), previous)
        self.assertEqual(progress_token({"completed_runs": 8, "durable_checkpoint": "tune_xsimgcl_08 epoch 6 (time)"}, previous), [8, "tune_xsimgcl_08", 6])

    def test_transient_network_and_permanent_errors_are_distinguished(self):
        from requests.exceptions import ConnectionError, HTTPError
        from requests import Response
        self.assertTrue(transient_network_error(ConnectionError("offline")))
        response = Response(); response.status_code = 403
        self.assertFalse(transient_network_error(HTTPError("denied", response=response)))
        self.assertFalse(transient_network_error(ValueError("invalid settings")))
