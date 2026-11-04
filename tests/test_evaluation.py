import unittest

import numpy as np
import polars as pl

from automotive_recsys.evaluation import contains, metrics_at_k
from automotive_recsys.popularity import rank_popularity


class EvaluationTests(unittest.TestCase):
    def test_multirelevance_ndcg_uses_min_k_answer_count(self):
        # User 0 has 3 answers, and two appear at positions 1 and 3.
        # User 1 has one answer at rank 2, so IDCG must be 1.
        result = metrics_at_k(
            np.array([0, 1]), np.array([[2, 1, 0], [0, 3, 2]]),
            np.array([0, 2, 4, 8]), np.array([3, 1]),
            np.array([False, False, True, True, True]), np.array([10, 8, 3, 2, 1]),
        )
        self.assertAlmostEqual(result["ndcg"][0], 1.5 / (1 + 1 / np.log2(3) + 0.5))
        self.assertAlmostEqual(result["ndcg"][1], 1 / np.log2(3))
        np.testing.assert_allclose(result["recall"], [2 / 3, 1])
        np.testing.assert_allclose(result["aplt"], [1 / 3, 2 / 3])

    def test_masking_and_deterministic_popularity_ties(self):
        seen = pl.DataFrame({"uid": [0, 0, 1], "iid": [0, 2, 1]})
        result = rank_popularity(np.array([9, 9, 4, 3, 2]), np.array([0, 1]), seen, 3)
        np.testing.assert_array_equal(result, [[1, 3, 4], [0, 2, 3]])

    def test_duplicate_recommendations_rejected(self):
        with self.assertRaises(ValueError):
            metrics_at_k(np.array([0]), np.array([[1, 1]]), np.array([1]), np.array([1]), np.array([True, True]), np.array([2, 1]))

    def test_empty_membership(self):
        np.testing.assert_array_equal(contains(np.array([], dtype=np.int64), np.array([1, 2])), [False, False])

    def test_all_candidates_seen_rejected(self):
        with self.assertRaises(ValueError):
            rank_popularity(np.array([3, 2]), np.array([0]), pl.DataFrame({"uid": [0, 0], "iid": [0, 1]}), 1)


if __name__ == "__main__":
    unittest.main()
