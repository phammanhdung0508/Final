"""Separate-trainer acceptance checks requested by the user on 2026-10-24.

1. Same inputs, same sampled triples and same initialization: every optimizer step of
   `VectorTrainer.step` matches the accepted path (accepted model, `HCPUAdam`,
   `CPUBackend.update`).
2. The vectorized negative sampler draws from the accepted eligible pool, uniformly,
   and skips exactly the positives the accepted sampler skips.
"""

import random
import unittest

import torch
from scipy.stats import chi2_contingency, chisquare

from scripts.gpu_ready_parity import RANKING_INVARIANT, model_pair, synthetic
from scripts.gpu_runtime import environment_fingerprint
from scripts.learned_models import LightGCNGraph
from scripts.recommendation_core import NegativeSampler
from scripts.recommendation_evaluation import TrainingContext
from scripts.training_harness import ADAM_OPTIONS, CPUBackend, HCPUAdam
from scripts.vector_trainer import VectorTrainer
from scripts.vectorized_models import VectorLightGCN

CPU = torch.device('cpu')
FINGERPRINT = environment_fingerprint(CPU)
METHODS = ('LightGCN', 'Heterogeneous KG+GNN')
P_FLOOR = 1e-3


def items_of(graph):
    return graph.warm_items if hasattr(graph, 'warm_items') else graph.products


def trainer_for(model, seed, batch_size=8):
    return VectorTrainer(model, [], seed=seed, batch_size=batch_size, validation_frequency=100,
                         fingerprint=FINGERPRINT)


class StepEquivalenceTests(unittest.TestCase):
    def test_every_step_matches_the_accepted_model_and_optimizer(self):
        backend = CPUBackend()
        for method in METHODS:
            for seed in (42, 2026, 3407):
                for condition in ('all_ratings', 'rating_ge_4'):
                    context, cats, brands, _ = synthetic(seed, condition)
                    ref, vec = model_pair(method, context, cats, brands, seed, CPU)
                    names = [n for n, _ in ref.named_parameters()]
                    self.assertEqual(names, [n for n, _ in vec.named_parameters()])
                    for n, p in vec.named_parameters():
                        self.assertTrue(torch.equal(p, dict(ref.named_parameters())[n]), n)
                    ref_opt = HCPUAdam(list(ref.parameters()), **ADAM_OPTIONS)
                    trainer = trainer_for(vec, seed)
                    users, items = ref.graph.users, items_of(ref.graph)
                    skip = RANKING_INVARIANT.get(method)
                    steps = 0
                    for _ in range(2):
                        u, p, n, _ = trainer.sample_epoch()
                        for start in range(0, len(u), trainer.batch_size):
                            sl = slice(start, start + trainer.batch_size)
                            triples = [(users[a], items[b], items[c]) for a, b, c in
                                       zip(u[sl].tolist(), p[sl].tolist(), n[sl].tolist())]
                            expected = backend.update(ref, ref_opt, triples)
                            actual = trainer.step(u[sl], p[sl], n[sl])
                            steps += 1
                            label = (method, seed, condition, steps)
                            for key in expected:
                                self.assertAlmostEqual(actual[key], expected[key],
                                                       delta=1e-5 * abs(expected[key]) + 1e-7,
                                                       msg=(key, label))
                            self.assert_state_close(ref, ref_opt, vec, trainer.optimizer, skip, label)
                    self.assertGreaterEqual(steps, 4)

    def assert_state_close(self, ref, ref_opt, vec, vec_opt, skip, label):
        ref_params = dict(ref.named_parameters())
        ref_state = ref_opt.state_dict()['state']
        vec_state = vec_opt.state_dict()['state']
        self.assertEqual(set(ref_state), set(vec_state), label)
        for i, (name, p) in enumerate(vec.named_parameters()):
            if name == skip:
                continue
            torch.testing.assert_close(p.detach(), ref_params[name].detach(), atol=1e-4, rtol=0,
                                       msg=f'{name} {label}')
            if i in ref_state:
                self.assertTrue(torch.equal(vec_state[i]['step'], ref_state[i]['step']), name)
                for key in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(vec_state[i][key], ref_state[i][key],
                                               atol=1e-6, rtol=1e-3, msg=f'{name}.{key} {label}')


def sampler_fixture(condition='rating_ge_4'):
    """Users with different pool sizes; u_low rated p03 below 4, so in rating_ge_4 that
    globally warm item must never be its negative. u_full rated every warm item."""
    records = [(f'u_bg{k}', f'p{k:02d}', 5.0, 0) for k in range(12)]
    records += [('u_small', f'p{k:02d}', 5.0, 0) for k in range(9)]
    records += [('u_low', 'p00', 5.0, 0), ('u_low', 'p03', 1.0, 0)]
    records += [('u_full', f'p{k:02d}', 5.0, 0) for k in range(12)]
    return TrainingContext(records, condition=condition)


class SamplerDistributionTests(unittest.TestCase):
    def setUp(self):
        self.contexts = {c: sampler_fixture(c) for c in ('all_ratings', 'rating_ge_4')}

    def draws(self, context, user, n, seed=5):
        graph = LightGCNGraph(context)
        model = VectorLightGCN(graph, torch.zeros(len(graph.users), 64),
                               torch.zeros(len(graph.warm_items), 64), CPU)
        trainer = trainer_for(model, seed)
        row = graph.users.index(user)
        drawn = trainer._negatives(torch.full((n,), row, dtype=torch.long)).tolist()
        return [graph.warm_items[i] for i in drawn]

    def test_support_equals_the_accepted_eligible_pool(self):
        for condition, context in self.contexts.items():
            accepted = NegativeSampler(context)
            for user in ('u_small', 'u_low', 'u_bg0'):
                drawn = self.draws(context, user, 3000)
                self.assertEqual(set(drawn), set(accepted.get_negative_pool(user)), (condition, user))
        self.assertNotIn('p03', self.draws(self.contexts['rating_ge_4'], 'u_low', 3000))

    def test_draws_are_uniform_over_the_pool(self):
        for condition, context in self.contexts.items():
            pool_of = NegativeSampler(context).get_negative_pool
            for user in ('u_small', 'u_low', 'u_bg0'):
                pool = pool_of(user)
                drawn = self.draws(context, user, 400 * len(pool))
                counts = [drawn.count(item) for item in pool]
                self.assertGreater(chisquare(counts).pvalue, P_FLOOR, (condition, user, counts))

    def test_distribution_matches_the_accepted_sampler(self):
        context = self.contexts['rating_ge_4']
        accepted = NegativeSampler(context)
        rng = random.Random(11)
        for user in ('u_small', 'u_low', 'u_bg0'):
            pool = accepted.get_negative_pool(user)
            n = 300 * len(pool)
            new = self.draws(context, user, n, seed=7)
            old = [accepted.sample_negative(user, rng) for _ in range(n)]
            table = [[new.count(i) for i in pool], [old.count(i) for i in pool]]
            if len(pool) > 1:
                self.assertGreater(chi2_contingency(table).pvalue, P_FLOOR, user)

    def test_per_user_distribution_does_not_depend_on_other_users_in_the_batch(self):
        context = self.contexts['all_ratings']
        graph = LightGCNGraph(context)
        model = VectorLightGCN(graph, torch.zeros(len(graph.users), 64),
                               torch.zeros(len(graph.warm_items), 64), CPU)
        trainer = trainer_for(model, 3)
        a, b = graph.users.index('u_small'), graph.users.index('u_bg0')
        users = torch.tensor([a, b] * 4000)
        drawn = trainer._negatives(users)
        pool = [graph.warm_items.index(i) for i in NegativeSampler(context).get_negative_pool('u_small')]
        counts = [int((drawn[users == a] == i).sum()) for i in pool]
        self.assertEqual(int((~torch.isin(drawn[users == a], torch.tensor(pool))).sum()), 0)
        self.assertGreater(chisquare(counts).pvalue, P_FLOOR)

    def test_empty_pool_skips_match_the_accepted_sampler(self):
        for condition, context in self.contexts.items():
            graph = LightGCNGraph(context)
            model = VectorLightGCN(graph, torch.zeros(len(graph.users), 64),
                                   torch.zeros(len(graph.warm_items), 64), CPU)
            users, _, negatives, skips = trainer_for(model, 1).sample_epoch()
            pairs = sorted((u, p) for u in context.users for p in context.positive_history.get(u, ()))
            triples, accepted_skips = NegativeSampler(context).prepare_batch(pairs, random.Random(0))
            self.assertEqual(skips, accepted_skips, condition)
            self.assertEqual(len(users), len(triples), condition)
            self.assertNotIn(graph.users.index('u_full'), users.tolist())


if __name__ == '__main__':
    unittest.main()
