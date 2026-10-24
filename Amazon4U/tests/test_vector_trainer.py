"""EX01 trainer: epoch semantics, eligible negatives and the global RNG guard (CPU).

Restart through the checkpoint path is tested in tests/test_checkpoint_cuda.py.
"""

import unittest
from unittest.mock import patch

import torch

from scripts.differentiable_models import build_hetero_graph
from scripts.gpu_ready_parity import synthetic
from scripts.gpu_runtime import RuntimeRefusal, environment_fingerprint
from scripts.learned_models import LightGCNGraph
from scripts.recommendation_evaluation import TrainingContext
from scripts.vector_trainer import VectorTrainer, fixture_rng
from scripts.vectorized_models import VectorHeteroGNN, VectorLightGCN

CPU = torch.device('cpu')
FINGERPRINT = environment_fingerprint(CPU)


def fresh(method, seed, condition='rating_ge_4'):
    context, cats, brands, targets = synthetic(seed, condition)
    if method == 'LightGCN':
        graph = LightGCNGraph(context)
        gen = torch.Generator().manual_seed(seed)
        model = VectorLightGCN(graph, torch.normal(0.0, 0.1, (len(graph.users), 64), generator=gen),
                               torch.normal(0.0, 0.1, (len(graph.warm_items), 64), generator=gen), CPU)
    else:
        model = VectorHeteroGNN(build_hetero_graph(context, cats, brands), seed, CPU)
    return model, targets


def trainer(model, targets, seed, frequency=1):
    return VectorTrainer(model, targets, rng=fixture_rng(seed), batch_size=8,
                         validation_frequency=frequency, fingerprint=FINGERPRINT)


class EpochTests(unittest.TestCase):
    def test_training_changes_parameters_and_records_selection(self):
        model, targets = fresh('Heterogeneous KG+GNN', 42)
        t = trainer(model, targets, 42)
        before = t._parameters()
        log = t.train_epoch()
        self.assertGreater(log['optimizer_steps'], 0)
        self.assertIsInstance(log['score'], float)
        self.assertFalse(torch.equal(before['user_embeddings'], t._parameters()['user_embeddings']))
        self.assertEqual(t.selection.state['selected_epoch'], 1)
        self.assertEqual(t.best_model['epoch'], 1)

    def test_unscheduled_epochs_skip_validation(self):
        model, targets = fresh('LightGCN', 42)
        t = trainer(model, targets, 42, frequency=2)
        self.assertIsNone(t.train_epoch()['score'])
        self.assertIsInstance(t.train_epoch()['score'], float)

    def test_same_generators_give_the_same_trajectory(self):
        for method in ('LightGCN', 'Heterogeneous KG+GNN'):
            logs = []
            for _ in range(2):
                model, targets = fresh(method, 2026)
                t = trainer(model, targets, 2026)
                logs.append([t.train_epoch() for _ in range(2)])
            self.assertEqual(logs[0], logs[1], method)

    def test_negatives_are_eligible_and_empty_pools_skip(self):
        records = [('u0', f'p{i}', 5.0, 0) for i in range(4)] + [('u1', 'p0', 5.0, 0)]
        context = TrainingContext(records, condition='all_ratings')
        graph = LightGCNGraph(context)
        model = VectorLightGCN(graph, torch.zeros(2, 64), torch.zeros(4, 64), CPU)
        t = VectorTrainer(model, [('u1', 'p2', 5.0)], rng=fixture_rng(1), batch_size=8,
                          validation_frequency=1, fingerprint=FINGERPRINT)
        self.assertTrue({t._negative(1) for _ in range(50)} <= {1, 2, 3})
        self.assertIsNone(t._negative(0))
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
    def test_hidden_global_randomness_fails_the_epoch_and_poisons(self):
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
            t.train_epoch()

    def test_restore_refuses_a_used_trainer(self):
        model, targets = fresh('LightGCN', 42)
        t = trainer(model, targets, 42)
        t.train_epoch()
        with self.assertRaises(RuntimeRefusal):
            t.restore_core({})


if __name__ == '__main__':
    unittest.main()
