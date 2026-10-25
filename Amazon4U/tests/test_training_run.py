"""Comparative-training runner (protocol v2.1.0) on a synthetic profiling dataset, on CPU."""

import copy
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import torch

from test_profiling_data import _csv, _kg_root
from test_profiling_runner import _synthetic
from scripts.profiling_data import MANIFEST_NAME, _sha256, stage_dataset
from scripts.training_run import HETERO, LIGHTGCN, RESULT_NAME, run, spec, validate_result

CPU = torch.device('cpu')


def _comparable(result):
    """What must match between runs: everything except wall time, memory and attempt files."""
    epochs = [{k: v for k, v in e.items() if k not in ('seconds', 'peak_host_bytes', 'peak_gpu_allocated_bytes')}
              for e in result['epochs']]
    return epochs, result['selection'], result['final_checkpoint']['sha256']


class TrainingRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        train, valid = _synthetic(seed=11)
        splits = tmp / 'splits'
        splits.mkdir()
        _csv(splits / 'Musical_Instruments.train.csv', train)
        _csv(splits / 'Musical_Instruments.valid.csv', valid)
        cls.data = tmp / 'dataset'
        stage_dataset(_kg_root(tmp, train=train), splits, cls.data)
        cls.sha = _sha256(cls.data / MANIFEST_NAME)
        cls.n = itertools.count()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def out(self):
        return Path(self._tmp.name) / f'run{next(self.n)}'

    def train(self, out, method=LIGHTGCN, **kw):
        args = dict(method=method, condition='all_ratings', learning_rate=0.002, seed=42, stage='tuning')
        args.update(kw)
        return run(self.data, self.sha, CPU, out, stream=io.StringIO(), **args)

    def test_a_run_trains_to_the_stopping_rule_and_keeps_a_final_checkpoint(self):
        for method in (LIGHTGCN, HETERO):
            with self.subTest(method):
                out = self.out()
                result = self.train(out, method)
                self.assertEqual(result['status'], 'run_complete')
                last = result['epochs'][-1]['epoch']
                self.assertTrue(last == 100 or last - result['selected_epoch'] == 10)
                self.assertIn('NDCG@10', result['selected_validation']['overall_metrics'])
                self.assertTrue((out / 'checkpoints' / 'final.a4ucp').exists())
                self.assertEqual(result['test_execution'], 'not_run_test_split_not_loaded')
                self.assertEqual(validate_result(json.loads((out / RESULT_NAME).read_text()), out),
                                 json.loads((out / RESULT_NAME).read_text()))

    def test_an_interrupted_run_resumes_to_the_same_result(self):
        straight = self.train(self.out(), HETERO)
        out = self.out()
        first = self.train(out, HETERO, max_epochs=3)
        self.assertEqual(first['status'], 'runner_timeout')
        self.assertTrue((out / 'checkpoints' / 'epoch-003.a4ucp').exists())
        second = self.train(out, HETERO, max_epochs=8)
        self.assertEqual(second['status'], 'runner_timeout')
        resumed = self.train(out, HETERO)
        self.assertEqual(resumed['status'], 'run_complete')
        self.assertEqual(len(resumed['attempts']), 3)
        self.assertEqual(_comparable(resumed), _comparable(straight))

    def test_learning_rates_change_the_trajectory(self):
        slow = self.train(self.out(), learning_rate=0.0005)
        fast = self.train(self.out(), learning_rate=0.002)
        self.assertNotEqual(slow['epochs'][0]['mean_total_loss'], fast['epochs'][1]['mean_total_loss'])
        self.assertNotEqual(slow['run_binding'], fast['run_binding'])

    def test_tampered_results_are_refused(self):
        out = self.out()
        good = self.train(out)
        edits = {
            'selected epoch': lambda r: r.update(selected_epoch=r['selected_epoch'] + 1),
            'a score': lambda r: r['epochs'][0].update(score=0.999),
            'best score': lambda r: r.update(**{'best_validation_NDCG@10': 1.0}),
            'stopped early': lambda r: r['epochs'].pop(),
            'checkpoint': lambda r: r['final_checkpoint'].update(sha256='0' * 64),
            'spec': lambda r: r['spec'].update(learning_rate=0.001),
            'test touched': lambda r: r.update(test_execution='run'),
        }
        for name, edit in edits.items():
            bad = copy.deepcopy(good)
            edit(bad)
            with self.subTest(name), self.assertRaises(ValueError):
                validate_result(bad, out)

    def test_unapproved_settings_are_refused(self):
        for kw in ({'learning_rate': 0.003}, {'seed': 7}, {'stage': 'final', 'seed': 42},
                   {'method': 'BPR-MF'}, {'condition': 'test'}):
            args = dict(method=LIGHTGCN, condition='all_ratings', learning_rate=0.001, seed=42, stage='tuning')
            args.update(kw)
            with self.subTest(kw), self.assertRaises(ValueError):
                spec(**args, manifest_sha256=self.sha)


if __name__ == '__main__':
    unittest.main()
