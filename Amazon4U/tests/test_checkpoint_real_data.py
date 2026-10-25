"""Checkpoint readiness C1: real-data expectations through D2/D3, on a synthetic dataset."""

import copy
import os
import tempfile
import unittest
from pathlib import Path

import torch

from test_profiling_data import _csv, _kg_root
from test_profiling_runner import _synthetic
from scripts.checkpoint_identity import CheckpointError
from scripts.checkpoint_real_data import HETERO, LIGHTGCN, build
from scripts.checkpoint_storage_cuda import publish_cuda_checkpoint, retrieve_cuda_checkpoint
from scripts.gpu_runtime import environment_fingerprint, pin_global_generators
from scripts.profiling_data import MANIFEST_NAME, ProfilingData, _sha256, stage_dataset
from scripts.training_harness import D2_KEYS, owned_digest

CPU = torch.device('cpu')
FP = environment_fingerprint(CPU)
BINDING = {'bundle_sha256': 'b' * 64, 'run_spec_sha256': 'c' * 64}
DEVICE = {'fingerprint': FP, 'determinism': None, 'run_binding': BINDING}
CASES = [(m, c) for m in (LIGHTGCN, HETERO) for c in ('all_ratings', 'rating_ge_4')]


class RealDataCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        train, valid = _synthetic(seed=5)
        splits = tmp / 'splits'
        splits.mkdir()
        _csv(splits / 'Musical_Instruments.train.csv', train)
        _csv(splits / 'Musical_Instruments.valid.csv', valid)
        stage_dataset(_kg_root(tmp, train=train), splits, tmp / 'dataset')
        cls.data = ProfilingData(tmp / 'dataset', _sha256(tmp / 'dataset' / MANIFEST_NAME))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        pin_global_generators()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        os.chmod(scratch.name, 0o700)
        self.root = os.open(scratch.name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.addCleanup(os.close, self.root)

    def fresh(self, method, condition):
        return build(self.data, method, condition, batch_size=8, fingerprint=FP)

    def publish(self, name, trainer, expectations, identity):
        d2 = {k: expectations[k] for k in D2_KEYS}
        return publish_cuda_checkpoint(self.root, name, trainer.cuda_payload(expectations, identity, BINDING),
                                       expected_device=DEVICE, **d2, io_policy=expectations['io_policy'])

    def retrieve(self, name, expectations):
        d2 = {k: expectations[k] for k in D2_KEYS}
        return retrieve_cuda_checkpoint(self.root, name, expected_device=DEVICE, **d2,
                                        io_policy=expectations['io_policy'])

    def test_resume_after_epoch_one_equals_two_uninterrupted_epochs(self):
        for method, condition in CASES:
            with self.subTest(method=method, condition=condition):
                straight, expectations, identity = self.fresh(method, condition)
                logs = [straight.train_epoch() for _ in range(2)]
                first, _, _ = self.fresh(method, condition)
                part = [first.train_epoch()]
                name = self.publish(f'{condition}-{len(method)}.a4ucp', first, expectations, identity)
                resumed, again, _ = self.fresh(method, condition)
                resumed.restore_cuda(self.retrieve(name, again))
                part.append(resumed.train_epoch())
                self.assertEqual(part, logs)
                self.assertEqual(owned_digest(resumed.cuda_payload(expectations, identity, BINDING)),
                                 owned_digest(straight.cuda_payload(expectations, identity, BINDING)))

    def test_identity_follows_the_dataset_and_condition(self):
        _, all_ratings, _ = self.fresh(LIGHTGCN, 'all_ratings')
        _, ge4, _ = self.fresh(LIGHTGCN, 'rating_ge_4')
        a, b = all_ratings['expected_identity'], ge4['expected_identity']
        self.assertEqual((a['category'], a['feedback_condition']), ('Musical_Instruments', 'all_ratings'))
        self.assertEqual(len(a['training_records']), len(self.data.train))
        self.assertLess(len(b['warm_items']), len(a['warm_items']))
        self.assertNotEqual(a['data_fingerprint'], b['data_fingerprint'])

    def test_a_checkpoint_is_refused_under_another_condition_or_method(self):
        trainer, expectations, identity = self.fresh(LIGHTGCN, 'all_ratings')
        trainer.train_epoch()
        name = self.publish('one.a4ucp', trainer, expectations, identity)
        for method, condition in ((LIGHTGCN, 'rating_ge_4'), (HETERO, 'all_ratings')):
            _, other, _ = self.fresh(method, condition)
            with self.subTest(method=method, condition=condition), self.assertRaises(CheckpointError):
                self.retrieve(name, other)

    def test_a_changed_training_record_is_refused(self):
        trainer, expectations, identity = self.fresh(HETERO, 'rating_ge_4')
        changed = copy.deepcopy(expectations)
        record = changed['expected_identity']['training_records'][0]
        record['rating'] = 5.0 if record['rating'] != 5.0 else 1.0
        with self.assertRaises(CheckpointError):
            self.publish('bad.a4ucp', trainer, changed, identity)


if __name__ == '__main__':
    unittest.main()
