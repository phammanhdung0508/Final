"""EX01 parity: vectorized models and evaluation against the accepted CPU references.

The accepted models and evaluator are the oracle. Forward values, losses, gradients and
five Adam steps must agree within float32 tolerance; the validation report must be
identical when both sides rank the same scores.
"""

import copy
import unittest

import torch

from scripts.gpu_ready_parity import RANKING_INVARIANT, as_index, batch, model_pair, synthetic
from scripts.recommendation_core import Evaluator
from scripts.recommendation_evaluation import ScoringFailureError, TrainingContext
from scripts.vectorized_evaluation import evaluate_model, evaluate_validation

CPU = torch.device('cpu')
ATOL, RTOL = 1e-5, 1e-4
RANKING_INVARIANT = RANKING_INVARIANT['Heterogeneous KG+GNN']


def light_pair(context, seed):
    return model_pair('LightGCN', context, None, None, seed, CPU)


def hetero_pair(context, categories, brands, seed):
    return model_pair('Heterogeneous KG+GNN', context, categories, brands, seed, CPU)


class ParityCase(unittest.TestCase):
    def assertClose(self, actual, expected, msg=''):
        torch.testing.assert_close(actual.detach(), expected.detach(), atol=ATOL, rtol=RTOL, msg=msg)

    def assertGradientsClose(self, ref, vec):
        ref_params = dict(ref.named_parameters())
        for name, p in vec.named_parameters():
            r = ref_params[name]
            if r.grad is None:
                self.assertTrue(p.grad is None or not p.grad.any(), name)
            else:
                self.assertClose(p.grad, r.grad, name)

    def run_adam(self, model, step):
        opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0.0)
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            step(model)['total_loss'].backward()
            opt.step()
        return {k: v.detach().clone() for k, v in model.state_dict().items()
                if k in dict(model.named_parameters())}


class LightGCNParityTests(ParityCase):
    def test_state_names_and_initialization_match(self):
        for seed in (42, 2026, 3407):
            ref, vec = light_pair(synthetic(seed)[0], seed)
            self.assertEqual([n for n, _ in ref.named_parameters()],
                             [n for n, _ in vec.named_parameters()])
            for name, p in vec.named_parameters():
                self.assertTrue(torch.equal(p, dict(ref.named_parameters())[name]))

    def test_forward_loss_gradients_and_adam_match(self):
        for seed in (42, 2026, 3407):
            for condition in ('all_ratings', 'rating_ge_4'):
                context = synthetic(seed, condition)[0]
                ref, vec = light_pair(context, seed)
                ref_u, ref_i = ref.forward()
                vec_u, vec_i = vec.forward()
                self.assertClose(vec_u, ref_u)
                self.assertClose(vec_i, ref_i)
                triples = batch(context, seed)
                users, pos, neg = map(list, zip(*triples))
                idx = as_index(triples, ref.graph.users, ref.graph.warm_items)
                ref_loss = ref.compute_batch_loss(users, pos, neg)
                vec_loss = vec.compute_batch_loss(*idx)
                for key in ref_loss:
                    self.assertClose(vec_loss[key], ref_loss[key], key)
                ref_loss['total_loss'].backward()
                vec_loss['total_loss'].backward()
                self.assertGradientsClose(ref, vec)
                ref_after = self.run_adam(ref, lambda m: m.compute_batch_loss(users, pos, neg))
                vec_after = self.run_adam(vec, lambda m: m.compute_batch_loss(*idx))
                for name, value in vec_after.items():
                    torch.testing.assert_close(value, ref_after[name], atol=1e-4, rtol=0)


class HeteroParityTests(ParityCase):
    def test_state_dict_is_interchangeable_and_initialization_matches(self):
        context, cats, brands, _ = synthetic(42)
        ref, vec = hetero_pair(context, cats, brands, 42)
        ref_state = ref.state_dict()
        vec_state = vec.state_dict()
        self.assertEqual(set(ref_state), set(vec_state))
        for key in ref_state:
            self.assertTrue(torch.equal(ref_state[key], vec_state[key]), key)
        ref.load_state_dict(vec_state, strict=True)

    def test_unmasked_and_masked_forward_match(self):
        for seed in (42, 2026, 3407):
            context, cats, brands, _ = synthetic(seed)
            ref, vec = hetero_pair(context, cats, brands, seed)
            ref_h, vec_h = ref.forward(None), vec.forward(None)
            for t in ref_h:
                self.assertClose(vec_h[t], ref_h[t], t)
            triples = batch(context, seed)
            mask = {(u, p) for u, p, _ in triples}
            n = len(ref.graph.products)
            u_map = ref.graph.node_to_idx['User']
            p_map = ref.graph.node_to_idx['Product']
            keys = torch.unique(torch.tensor([u_map[u] * n + p_map[p] for u, p in mask]))
            ref_m, vec_m = ref.forward(mask), vec.forward(keys)
            for t in ref_m:
                self.assertClose(vec_m[t], ref_m[t], t)
            self.assertFalse(torch.allclose(ref_m['User'], ref_h['User']))

    def test_loss_gradients_and_adam_match(self):
        for seed in (42, 2026, 3407):
            for condition in ('all_ratings', 'rating_ge_4'):
                context, cats, brands, _ = synthetic(seed, condition)
                ref, vec = hetero_pair(context, cats, brands, seed)
                triples = batch(context, seed)
                users, pos, neg = map(list, zip(*triples))
                idx = as_index(triples, ref.graph.users, ref.graph.products)
                ref_loss = ref.compute_batch_loss(users, pos, neg)
                vec_loss = vec.compute_batch_loss(*idx)
                for key in ref_loss:
                    self.assertClose(vec_loss[key], ref_loss[key], key)
                ref_loss['total_loss'].backward()
                vec_loss['total_loss'].backward()
                self.assertGradientsClose(ref, vec)
                ref_after = self.run_adam(ref, lambda m: m.compute_batch_loss(users, pos, neg))
                vec_after = self.run_adam(vec, lambda m: m.compute_batch_loss(*idx))
                for name, value in vec_after.items():
                    if name != RANKING_INVARIANT:
                        torch.testing.assert_close(value, ref_after[name], atol=1e-4, rtol=0, msg=name)

    def test_product_output_bias_has_zero_gradient_and_cannot_change_rankings(self):
        context, cats, brands, _ = synthetic(3407, 'all_ratings')
        ref, _ = hetero_pair(context, cats, brands, 3407)
        users, pos, neg = map(list, zip(*batch(context, 3407)))
        ref.compute_batch_loss(users, pos, neg)['total_loss'].backward()
        bias = dict(ref.named_parameters())[RANKING_INVARIANT]
        self.assertLess(bias.grad.abs().max().item(), 1e-7)
        shift = torch.randn(64, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            h = ref.forward(None)
            before = h['User'] @ h['Product'].T
            bias.add_(shift)
            after = ref.forward(None)['Product']
        row_shift = (h['User'] @ shift).unsqueeze(1)
        torch.testing.assert_close(h['User'] @ after.T, before + row_shift, atol=1e-5, rtol=1e-5)

    def test_invalid_batches_fail_closed(self):
        context, cats, brands, _ = synthetic(42)
        _, vec = hetero_pair(context, cats, brands, 42)
        users, pos, neg = as_index(batch(context, 42), vec.graph.users, vec.graph.products)
        with self.assertRaises(ValueError):
            vec.compute_batch_loss(users, neg, neg)
        with self.assertRaises(ValueError):
            vec.compute_batch_loss(users, pos, pos)
        with self.assertRaises(ValueError):
            vec.compute_batch_loss(users[:0], pos[:0], neg[:0])


class TableScorer:
    """Accepted-evaluator scorer over the same score table the vectorized path ranks."""

    def __init__(self, users, items, user_table, item_table):
        self.u = {u: i for i, u in enumerate(users)}
        self.i = {p: i for i, p in enumerate(items)}
        self.scores = (user_table @ item_table.T).tolist()

    def score_candidates(self, user_id, candidates):
        row = self.scores[self.u[user_id]]
        return {p: row[self.i[p]] for p in candidates}, 'personalized', None


class EvaluationParityTests(unittest.TestCase):
    def check_identical(self, context, users, items, user_table, item_table, targets, chunk):
        expected = Evaluator(context).evaluate(
            copy.deepcopy(targets), TableScorer(users, items, user_table, item_table),
            split='validation')
        actual = evaluate_validation(context, users, items, user_table, item_table,
                                     targets, chunk_users=chunk)
        self.assertEqual(actual, expected)

    def test_report_identical_for_same_scores(self):
        for seed in (42, 2026, 3407):
            for condition in ('all_ratings', 'rating_ge_4'):
                context, _, _, targets = synthetic(seed, condition)
                users, items = sorted(context.users), context.sorted_warm_items
                gen = torch.Generator().manual_seed(seed)
                ut = torch.randn(len(users), 64, generator=gen)
                it = torch.randn(len(items), 64, generator=gen)
                for chunk in (1, 5, 1024):
                    self.check_identical(context, users, items, ut, it, targets, chunk)

    def test_exact_ties_rank_by_parent_asin(self):
        context, _, _, targets = synthetic(7, 'all_ratings')
        users, items = sorted(context.users), context.sorted_warm_items
        ut = torch.ones(len(users), 64)
        it = torch.zeros(len(items), 64)
        it[::3] = 1.0
        self.check_identical(context, users, items, ut, it, targets, 4)

    def test_shortage_counts_match(self):
        records = [('u0', f'p{i}', 5.0, 0) for i in range(12)] + [('u1', 'p0', 5.0, 0)]
        context = TrainingContext(records, condition='all_ratings')
        users, items = sorted(context.users), context.sorted_warm_items
        gen = torch.Generator().manual_seed(1)
        ut, it = torch.randn(2, 64, generator=gen), torch.randn(len(items), 64, generator=gen)
        targets = [('u0', 'p13', 5.0), ('u1', 'p3', 5.0)]
        self.check_identical(context, users, items, ut, it, targets, 2)

    def test_model_reports_match_accepted_models(self):
        context, cats, brands, targets = synthetic(2026)
        for ref, vec in (light_pair(context, 2026), hetero_pair(context, cats, brands, 2026)):
            with torch.no_grad():
                expected = Evaluator(context).evaluate(copy.deepcopy(targets), ref, split='validation')
            actual = evaluate_model(vec, targets, chunk_users=3)
            self.assertEqual([r['top_20'] for r in actual['per_user_results']],
                             [r['top_20'] for r in expected['per_user_results']])
            self.assertEqual(actual['audit'], expected['audit'])
            for key, value in expected['overall_metrics'].items():
                self.assertAlmostEqual(actual['overall_metrics'][key], value, places=12)

    def test_non_finite_candidate_score_raises(self):
        context, _, _, targets = synthetic(42, 'all_ratings')
        users, items = sorted(context.users), context.sorted_warm_items
        ut = torch.randn(len(users), 64)
        it = torch.randn(len(items), 64)
        it[0, 0] = float('nan')
        with self.assertRaises(ScoringFailureError):
            evaluate_validation(context, users, items, ut, it, targets)

    def test_item_order_must_be_sorted_warm_items(self):
        context, _, _, targets = synthetic(42)
        users, items = sorted(context.users), list(reversed(context.sorted_warm_items))
        with self.assertRaises(ValueError):
            evaluate_validation(context, users, items, torch.zeros(len(users), 64),
                                torch.zeros(len(items), 64), targets)


if __name__ == '__main__':
    unittest.main()
