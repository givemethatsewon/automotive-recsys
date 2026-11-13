import unittest
import numpy as np
from automotive_recsys.vehicle_rerank import match_candidates, stable_rerank


class VehicleRerankTests(unittest.TestCase):
    def setUp(self):
        self.uid = np.array([0, 1, 2])
        self.iid = np.tile(np.arange(5), (3, 1))
        self.up = np.array([0, 2, 3, 3])
        self.uv = np.array([[0, 0], [1, 1], [0, -1]])
        self.ip = np.array([0, 1, 2, 3, 4, 4])
        self.pf = np.array([[0, 0], [1, 1], [0, 2], [0, 0]])
        self.universal = np.array([False, False, False, True, False])

    def test_multi_vehicle_any_match_and_universal_unknown(self):
        state = match_candidates(self.uid, self.iid, self.up, self.uv, self.ip, self.pf, self.universal, 1)
        np.testing.assert_array_equal(state, [[1, 1, 0, -1, -1], [-1, 0, -1, -1, -1], [-1]*5])

    def test_make_only_is_valid_at_make_level(self):
        state = match_candidates(self.uid, self.iid, self.up, self.uv, self.ip, self.pf, self.universal, 0)
        np.testing.assert_array_equal(state[1], [1, 0, 1, -1, -1])

    def test_unknown_does_not_get_bonus_and_ties_use_item_id(self):
        iid = np.array([[4, 2, 3, 1]])
        score = np.array([[1., 1., .9, .8]])
        state = np.array([[-1, 0, 1, -1]])
        ranked, _ = stable_rerank(iid, score, state, 1)
        np.testing.assert_array_equal(ranked, [[3, 2, 4, 1]])
        baseline, _ = stable_rerank(iid, score, state, 0)
        np.testing.assert_array_equal(baseline, [[2, 4, 3, 1]])

    def test_zero_score_range(self):
        ranked, _ = stable_rerank(np.array([[2, 1]]), np.zeros((1, 2)), np.array([[1, -1]]), .1)
        np.testing.assert_array_equal(ranked, [[2, 1]])


if __name__ == '__main__':
    unittest.main()
