"""EX01 trainer: epoch semantics, RNG guard and bit-for-bit epoch-boundary restart (CPU)."""

import unittest
from unittest.mock import patch

import torch

from scripts.gpu_runtime import RuntimeRefusal, environment_fingerprint
from scripts.vector_trainer import VectorTrainer
from scripts.vectorized_models import VectorHeteroGNN, VectorLightGCN
from scripts.differentiable_models import build_hetero_graph
from scripts.learned_models import LightGCNGraph
from scripts.recommendation_evaluation import TrainingContext
from scripts.gpu_ready_parity import synthetic

CPU = torch.device('cpu')
FINGERPRINT = environment_fingerprint(CPU)


def models(seed, condition='rating_ge_4'):
    context, cats, brands, targets = synthetic(seed, condition)
    graph = LightGCNGraph(context)
    gen = torch.Generator().manual_seed(seed)
    light = VectorLightGCN(graph, torch.normal(0.0, 0.1, (len(graph.users), 64), generator=gen),
                           torch.normal(0.0, 0.1, (len(graph.warm_items), 64), generator=gen), CPU)
    hetero = VectorHeteroGNN(build_hetero_graph(context, cats, brands), seed, CPU)
    return targets, {'LightGCN': light, 'Heterogeneous KG+GNN': hetero}


def trainer(model, targets, seed, frequency=1):
    return VectorTrainer(model, targets, seed=seed, batch_size=8, validation_frequency=frequency,
                         fingerprint=FINGERPRINT)


def fresh(method, seed):
    targets, built = models(seed)
    return built[method], targets


def assert_same(case, a, b, path='state'):
    case.assertIs(type(a), type(b), path)
    if isinstance(a, torch.Tensor):
        case.assertTrue(torch.equal(a, b) and a.dtype == b.dtype, path)
    elif isinstance(a, dict):
        case.assertEqual(set(a), set(b), path)
        for k in a:
            assert_same(case, a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(case, x, y, f'{path}[{i}]')
    else:
        case.assertEqual(a, b, path)


class RestartTests(unittest.TestCase):
    def test_restart_is_bit_for_bit_identical_to_uninterrupted(self):
        for method in ('LightGCN', 'Heterogeneous KG+GNN'):
            for seed in (42, 2026, 3407):
                model, targets = fresh(method, seed)
                straight = trainer(model, targets, seed)
                straight_logs = [straight.train_epoch() for _ in range(4)]

                model, targets = fresh(method, seed)
                first = trainer(model, targets, seed)
                first_logs = [first.train_epoch() for _ in range(2)]
                saved = first.capture_state()

                model, targets = fresh(method, seed)
                resumed = trainer(model, targets, seed)
                resumed.load_state(saved)
                resumed_logs = [resumed.train_epoch() for _ in range(2)]

                self.assertEqual(first_logs + resumed_logs, straight_logs, method)
                a, b = straight.capture_state(), resumed.capture_state()
                assert_same(self, a, b, method)
                self.assertEqual(straight.validation_report, resumed.validation_report)

    def test_training_changes_parameters_and_records_selection(self):
        model, targets = fresh('Heterogeneous KG+GNN', 42)
        t = trainer(model, targets, 42)
        before = t.capture_state()['parameters']
        log = t.train_epoch()
        self.assertGreater(log['optimizer_steps'], 0)
        self.assertIsInstance(log['score'], float)
        after = t.capture_state()['parameters']
        self.assertFalse(torch.equal(before['user_embeddings'], after['user_embeddings']))
        self.assertEqual(t.selection.state['selected_epoch'], 1)
        self.assertEqual(t.best_model['epoch'], 1)

    def test_unscheduled_epochs_skip_validation(self):
        model, targets = fresh('LightGCN', 42)
        t = trainer(model, targets, 42, frequency=2)
        self.assertIsNone(t.train_epoch()['score'])
        self.assertIsInstance(t.train_epoch()['score'], float)

    def test_negatives_are_eligible_and_empty_pools_skip(self):
        records = [('u0', f'p{i}', 5.0, 0) for i in range(4)] + [('u1', 'p0', 5.0, 0)]
        context = TrainingContext(records, condition='all_ratings')
        graph = LightGCNGraph(context)
        model = VectorLightGCN(graph, torch.zeros(2, 64), torch.zeros(4, 64), CPU)
        t = VectorTrainer(model, [('u1', 'p2', 5.0)], seed=1, batch_size=8,
                          validation_frequency=1, fingerprint=FINGERPRINT)
        users = torch.tensor([1] * 50)
        negatives = t._negatives(users)
        self.assertTrue(set(negatives.tolist()) <= {1, 2, 3})
        self.assertEqual(t.train_epoch()['empty_pool_skips'], 4)


class DeterministicModeTests(unittest.TestCase):
    def test_both_models_train_and_validate_under_deterministic_algorithms(self):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for method in ('LightGCN', 'Heterogeneous KG+GNN'):
                model, targets = fresh(method, 2026)
                self.assertIsInstance(trainer(model, targets, 2026).train_epoch()['score'], float)
        finally:
            torch.use_deterministic_algorithms(previous)


class RefusalTests(unittest.TestCase):
    def test_hidden_global_randomness_fails_the_epoch(self):
        model, targets = fresh('LightGCN', 42)
        t = trainer(model, targets, 42)
        original = model.compute_batch_loss

        def leaky(*args):
            torch.rand(1)
            return original(*args)
        with patch.object(model, 'compute_batch_loss', side_effect=leaky):
            with self.assertRaisesRegex(RuntimeRefusal, 'RNG state changed'):
                t.train_epoch()
        with self.assertRaises(RuntimeRefusal):
            t.capture_state()

    def test_resume_refuses_changed_environment_or_settings(self):
        model, targets = fresh('LightGCN', 42)
        t = trainer(model, targets, 42)
        t.train_epoch()
        saved = t.capture_state()
        for key, value in (('fingerprint', {**FINGERPRINT, 'gpu_name': 'other'}),
                           ('seed', 43), ('batch_size', 9), ('schema_version', 'x')):
            model, targets = fresh('LightGCN', 42)
            fresh_trainer = trainer(model, targets, 42)
            with self.assertRaises(RuntimeRefusal, msg=key):
                fresh_trainer.load_state({**saved, key: value})

    def test_resume_refuses_a_used_trainer(self):
        model, targets = fresh('LightGCN', 42)
        t = trainer(model, targets, 42)
        saved = t.capture_state()
        t.train_epoch()
        with self.assertRaises(RuntimeRefusal):
            t.load_state(saved)


if __name__ == '__main__':
    unittest.main()
