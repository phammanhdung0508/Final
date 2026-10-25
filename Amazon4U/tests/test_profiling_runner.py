"""Profiling P3: the runner on a synthetic profiling dataset, on CPU."""

import io
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from test_profiling_data import _csv, _kg_root
from scripts.profiling_data import MANIFEST_NAME, ProfilingData, _sha256, stage_dataset
from scripts.profiling_runner import (
    HETERO, JOURNAL_NAME, RESULT_NAME, JournalError, Limits, read_journal, run, validate_result,
)
from scripts.vectorized_evaluation import evaluate_model, evaluate_validation
from scripts.vectorized_models import VectorLightGCN
from scripts.learned_models import LightGCNGraph

CPU = torch.device('cpu')
CASES = (('LightGCN', 16, 'all_ratings'), ('LightGCN', 16, 'rating_ge_4'),
         (HETERO, 16, 'all_ratings'))
OPEN = Limits(preflight_seconds=600, preflight_gpu_bytes=1 << 40, preflight_host_bytes=1 << 40,
              runner_seconds=1200, threshold_seconds=1000)


def _synthetic(seed=3, n_users=30, n_items=20):
    rng = random.Random(seed)
    train, valid = [], []
    for u in range(n_users):
        items = rng.sample(range(1, n_items + 1), rng.randint(2, 7))
        for t, p in enumerate(items[:-1]):
            train.append((f'u{u:02d}', f'p{p}', float(rng.choice([1, 3, 4, 5, 5])), t))
        valid.append((f'u{u:02d}', f'p{items[-1]}', float(rng.choice([2, 4, 5])), 99))
    return train, valid


class ProfilingRunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        train, valid = _synthetic()
        splits = self.tmp / 'splits'
        splits.mkdir()
        _csv(splits / 'Musical_Instruments.train.csv', train)
        _csv(splits / 'Musical_Instruments.valid.csv', valid)
        self.data = self.tmp / 'dataset'
        stage_dataset(_kg_root(self.tmp, train=train), splits, self.data)
        self.sha = _sha256(self.data / MANIFEST_NAME)
        self.n = itertools.count()

    def run_plan(self, limits=OPEN, **kw):
        out = self.tmp / f'out{next(self.n)}'
        result = run(self.data, self.sha, CPU, out, limits=limits, cases=CASES,
                     stream=io.StringIO(), **kw)
        return result, out

    def test_complete_run_times_every_epoch_and_keeps_no_weights(self):
        result, out = self.run_plan()
        self.assertEqual(result['status'], 'profile_complete', result['stop_reason'])
        self.assertEqual(sorted(p.name for p in out.iterdir()), [JOURNAL_NAME, RESULT_NAME])
        self.assertEqual(len(result['preflight']), 3)
        self.assertEqual(len(result['cases']), 3)
        for case in result['cases']:
            self.assertEqual([e['label'] for e in case['epochs']],
                             ['warm_up', 'steady_state', 'steady_state'])
            for epoch in case['epochs']:
                v = epoch['validation']
                self.assertTrue(v['checks']['all_finite'])
                self.assertAlmostEqual(v['inference'] + v['ranking'] + v['metrics'], v['total'])
        projection = result['projection']
        self.assertEqual(projection['gate'], max(projection['mean'], projection['p90']))
        on_disk = json.loads((out / RESULT_NAME).read_text())
        self.assertEqual(validate_result(on_disk, out / JOURNAL_NAME), on_disk)
        self.assertNotIn('NDCG', (out / RESULT_NAME).read_text())

    def test_tampered_results_are_refused(self):
        _, out = self.run_plan()
        good = json.loads((out / RESULT_NAME).read_text())
        edits = {
            'epoch time': lambda r: r['cases'][0]['epochs'][1].update(training_steps=0.001),
            'preflight step time': lambda r: r['preflight'][0]['step_stats'].update(p90=0.0),
            'status': lambda r: r.update(status='preflight_exceeded'),
            'journal tail': lambda r: r.update(journal_tail_sha256='0' * 64),
            'weights': lambda r: r.update(weights_kept=True),
            'dropped case': lambda r: r['cases'].pop(),
        }
        for name, edit in edits.items():
            bad = json.loads(json.dumps(good))
            edit(bad)
            with self.subTest(name), self.assertRaises(ValueError):
                validate_result(bad, out / JOURNAL_NAME)

    def test_journal_detects_a_cut_off_line_and_a_changed_line(self):
        _, out = self.run_plan()
        path = out / JOURNAL_NAME
        lines = path.read_text().splitlines(keepends=True)
        path.write_text(''.join(lines[:-1]) + lines[-1][:10])
        records, _, cut_off = read_journal(path)
        self.assertTrue(cut_off)
        self.assertEqual(len(records), len(lines) - 1)
        lines[1] = lines[1].replace('"load"', '"loaded"')
        path.write_text(''.join(lines))
        with self.assertRaises(JournalError):
            read_journal(path)

    def test_projection_over_threshold_stops_before_any_case(self):
        result, _ = self.run_plan(limits=Limits(**{**OPEN.__dict__, 'threshold_seconds': 0.0}))
        self.assertEqual(result['status'], 'preflight_exceeded')
        self.assertEqual(len(result['preflight']), 1, 'lower bound stops after the first case')
        self.assertEqual(result['cases'], [])

    def test_preflight_memory_limit_stops_the_run(self):
        result, _ = self.run_plan(limits=Limits(**{**OPEN.__dict__, 'preflight_host_bytes': 1}))
        self.assertEqual(result['status'], 'preflight_limit_exceeded')
        self.assertIn('host memory', result['stop_reason'])
        self.assertEqual(result['cases'], [])

    def test_runner_deadline_stops_inside_a_phase_and_keeps_the_journal(self):
        ticks = itertools.count()
        result, out = self.run_plan(clock=lambda: float(next(ticks)),
                                    limits=Limits(**{**OPEN.__dict__, 'runner_seconds': 25}))
        self.assertEqual(result['status'], 'runner_timeout')
        records, _, cut_off = read_journal(out / JOURNAL_NAME)
        self.assertFalse(cut_off)
        self.assertEqual(records[-1]['event'], 'end')
        self.assertEqual(records[-1]['status'], 'runner_timeout')

    def test_a_failure_is_recorded_not_raised(self):
        (self.data / 'Musical_Instruments.valid.csv').write_text('changed')
        result, _ = self.run_plan()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('ProfilingDataError', result['stop_reason'])

    def test_evaluator_tick_hook_leaves_the_report_unchanged(self):
        loaded = ProfilingData(self.data, self.sha)
        graph = LightGCNGraph(loaded.context('rating_ge_4'))
        gen = torch.Generator().manual_seed(1)
        model = VectorLightGCN(graph, torch.randn(len(graph.users), 64, generator=gen),
                               torch.randn(len(graph.warm_items), 64, generator=gen), CPU)
        targets = loaded.validation_targets()
        seen = []
        with torch.no_grad():
            users, items = model.representations()
        ticked = evaluate_validation(graph.context, graph.users, graph.warm_items, users, items,
                                     targets, chunk_users=7, tick=seen.append)
        self.assertEqual(ticked, evaluate_model(model, targets, chunk_users=7))
        self.assertEqual(seen[:2], ['ranking', 'metrics'])
        self.assertEqual(len(seen) % 2, 0)


if __name__ == '__main__':
    unittest.main()
