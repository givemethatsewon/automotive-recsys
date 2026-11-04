"""Tiny mathematical checks only; no local full-data training."""
import unittest

import torch
from torch.nn import functional as F

from automotive_recsys.gpu_kernels import encode, full_catalog_topk, info_nce, normalized_graph, sample_unseen, training_keys


class KernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.graph = normalized_graph([0, 0, 1], [0, 1, 1], 2, 2, layout="coo")

    def test_normalization_matches_hand_computed_graph(self):
        root = 2 ** -0.5
        expected = torch.tensor([[0, 0, root, 0.5], [0, 0, 0, root], [root, 0, 0, 0], [0.5, root, 0, 0]])
        torch.testing.assert_close(self.graph.to_dense(), expected)

    def test_streaming_sum_output_and_gradient_match_layer_stack(self):
        for model in ["lightgcn", "xsimgcl"]:
            for layout in ["coo", "csr"]:
                with self.subTest(model=model, layout=layout):
                    graph = self.graph if layout == "coo" else self.graph.to_sparse_csr()
                    weight = torch.randn(4, 3, requires_grad=True)
                    reference = weight.detach().clone().requires_grad_()
                    noise = [torch.rand(4, 3) for _ in range(3)]
                    observed, contrast = encode(weight, graph, model, perturbed=model == "xsimgcl", noise=noise,
                                                symmetric_backward=True)
                    current = reference
                    states = [reference] if model == "lightgcn" else []
                    for layer in range(3):
                        current = self.graph.to_dense() @ current
                        if model == "xsimgcl":
                            current = current + 0.1 * current.sign() * F.normalize(noise[layer], dim=-1)
                        states.append(current)
                    expected = torch.stack(states).mean(0)
                    torch.testing.assert_close(observed, expected)
                    observed.square().sum().backward()
                    expected.square().sum().backward()
                    torch.testing.assert_close(weight.grad, reference.grad)

    def test_symmetric_shortcut_rejects_trainable_edge_weights(self):
        graph = self.graph.detach().requires_grad_()
        with self.assertRaises(ValueError):
            encode(torch.randn(4, 3, requires_grad=True), graph, "lightgcn", symmetric_backward=True)

    def test_cross_entropy_infonce_matches_diagonal_logsoftmax_gradients(self):
        a, b = torch.randn(5, 3, requires_grad=True), torch.randn(5, 3, requires_grad=True)
        aa, bb = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
        actual = info_nce(a, b)
        scores = F.normalize(aa, dim=-1) @ F.normalize(bb, dim=-1).T / 0.2
        expected = -F.log_softmax(scores, dim=1).diagonal().mean()
        torch.testing.assert_close(actual, expected)
        actual.backward(); expected.backward()
        torch.testing.assert_close(a.grad, aa.grad)
        torch.testing.assert_close(b.grad, bb.grad)

    def test_negative_sampler_never_returns_training_positive(self):
        # Each of the two users has only one available negative item.
        keys = training_keys([0, 0, 1, 1], [0, 1, 1, 2], 3, "cpu")
        users = torch.tensor([0, 1] * 100)
        negatives = sample_unseen(users, keys, 3)
        torch.testing.assert_close(negatives, torch.tensor([2, 0] * 100))

    def test_full_catalog_topk_handles_boundary_ties_and_seen_items(self):
        import numpy as np
        # All scores tie, including at the cutoff; choose the lowest unseen IID.
        observed = full_catalog_topk(torch.zeros(7, 3), np.array([0, 1]), 2,
                                     np.array([0, 0, 1]), np.array([0, 1, 3]), k=3, batch_size=2)
        np.testing.assert_array_equal(observed, [[2, 3, 4], [0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
