"""Targeted tests for the proof in docs/gpu-ready-design.md: the heterogeneous layer-1
Product self bias cannot change the loss or rankings.

Exact claims are checked in float64 on the vectorized model, which matches the accepted
model in float32 (tests/test_vectorized_execution.py).
"""

import unittest

import torch

from scripts.gpu_ready_parity import as_index, batch, model_pair, synthetic
from scripts.vectorized_models import VectorHeteroGNN
from scripts.differentiable_models import build_hetero_graph

CPU = torch.device('cpu')
BIAS = 'self_biases_l1.Product'
CASES = [(seed, condition) for seed in (42, 2026, 3407) for condition in ('all_ratings', 'rating_ge_4')]


def model64(seed, condition):
    context, cats, brands, _ = synthetic(seed, condition)
    model = VectorHeteroGNN(build_hetero_graph(context, cats, brands), seed, CPU).double()
    return model, as_index(batch(context, seed), model.graph.users, model.graph.products)


def shift(seed):
    return torch.randn(64, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)) * 10


def param(model, name):
    return dict(model.named_parameters())[name]


class LossInvarianceTests(unittest.TestCase):
    def test_loss_is_unchanged_by_any_bias_shift(self):
        for seed, condition in CASES:
            model, idx = model64(seed, condition)
            with torch.no_grad():
                before = model.compute_batch_loss(*idx)
                param(model, BIAS).add_(shift(seed))
                after = model.compute_batch_loss(*idx)
            for key in before:
                torch.testing.assert_close(after[key], before[key], atol=1e-12, rtol=1e-12,
                                           msg=f'{key} {seed} {condition}')

    def test_control_user_bias_does_change_the_loss(self):
        model, idx = model64(42, 'all_ratings')
        with torch.no_grad():
            before = model.compute_batch_loss(*idx)['total_loss']
            param(model, 'self_biases_l1.User').add_(shift(42))
            after = model.compute_batch_loss(*idx)['total_loss']
        self.assertGreater(abs(float(after - before)), 1e-3)


class GradientTests(unittest.TestCase):
    def test_computed_gradient_is_rounding_level_and_unique(self):
        structural = {'relation_weights_l1.belongs_to', 'relation_weights_l1.category_child_of',
                      'relation_weights_l1.category_parent_of', 'relation_weights_l1.has_brand',
                      'self_weights_l1.CategoryPath', 'self_weights_l1.Brand',
                      'self_biases_l1.CategoryPath', 'self_biases_l1.Brand'}
        for seed, condition in CASES:
            model, idx = model64(seed, condition)
            model.compute_batch_loss(*idx)['total_loss'].backward()
            grads = {k: v.grad for k, v in model.named_parameters()}
            self.assertLess(grads[BIAS].abs().max().item(), 1e-14)
            self.assertEqual({k for k, g in grads.items() if g is None}, structural)
            others = {k: g.abs().max().item() for k, g in grads.items()
                      if g is not None and k != BIAS}
            self.assertGreater(min(others.values()), 1e-10, (seed, condition, others))


class RankingInvarianceTests(unittest.TestCase):
    def test_scores_shift_by_a_per_user_constant_and_rankings_are_identical(self):
        for seed, condition in CASES:
            model, _ = model64(seed, condition)
            delta = shift(seed)
            with torch.no_grad():
                u, p = model.representations()
                before = u @ p.T
                param(model, BIAS).add_(delta)
                u2, p2 = model.representations()
                after = u2 @ p2.T
            torch.testing.assert_close(u2, u, atol=0, rtol=0)
            torch.testing.assert_close(after - before, (u @ delta).unsqueeze(1).expand_as(before),
                                       atol=1e-9, rtol=1e-9)
            order = lambda s: torch.sort(-s, dim=1, stable=True).indices[:, :20]
            self.assertTrue(torch.equal(order(after), order(before)), (seed, condition))


class AdamDriftTests(unittest.TestCase):
    def test_float32_noise_driven_drift_stays_within_lr_per_step(self):
        context, cats, brands, _ = synthetic(3407, 'all_ratings')
        _, model = model_pair('Heterogeneous KG+GNN', context, cats, brands, 3407, CPU)
        idx = as_index(batch(context, 3407), model.graph.users, model.graph.products)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        start = param(model, BIAS).detach().clone()
        for step in range(1, 6):
            opt.zero_grad(set_to_none=True)
            model.compute_batch_loss(*idx)['total_loss'].backward()
            opt.step()
            drift = (param(model, BIAS).detach() - start).abs().max().item()
            self.assertLessEqual(drift, 0.001 * step * 1.01)


if __name__ == '__main__':
    unittest.main()
