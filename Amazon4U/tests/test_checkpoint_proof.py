"""Checkpoint readiness C4: the proof runner on a synthetic profiling dataset, on CPU."""

import copy
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from test_profiling_data import _csv, _kg_root
from test_profiling_runner import _synthetic
from scripts.checkpoint_proof import (
    HETERO, JOURNAL_NAME, RESULT_NAME, run, validate_result,
)
from scripts.profiling_data import MANIFEST_NAME, _sha256, stage_dataset

CASES = (('LightGCN', 16, 'all_ratings'), (HETERO, 16, 'rating_ge_4'))


class CheckpointProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        train, valid = _synthetic(seed=9)
        splits = tmp / 'splits'
        splits.mkdir()
        _csv(splits / 'Musical_Instruments.train.csv', train)
        _csv(splits / 'Musical_Instruments.valid.csv', valid)
        cls.data = tmp / 'dataset'
        stage_dataset(_kg_root(tmp, train=train), splits, cls.data)
        cls.sha = _sha256(cls.data / MANIFEST_NAME)
        cls.n = itertools.count()
        cls.result, cls.out = cls.run_proof(cls)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_proof(self, **kw):
        out = Path(self._tmp.name) / f'out{next(self.n)}'
        result = run(self.data, self.sha, 'cpu', out, cases=CASES, epochs=2, split=1,
                     stream=io.StringIO(), **kw)
        return result, out

    def test_every_case_resumes_exactly_and_keeps_nothing_else(self):
        result = self.result
        self.assertEqual(result['status'], 'proof_complete', result['stop_reason'])
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), [JOURNAL_NAME, RESULT_NAME])
        for case in result['cases']:
            self.assertTrue(all(case['checks'].values()), case['case'])
            self.assertGreater(case['cost']['file_bytes'], 0)
            self.assertGreater(case['cost']['retrieve_seconds'], 0)
        text = (self.out / RESULT_NAME).read_text()
        self.assertNotIn('NDCG', text)
        self.assertEqual(validate_result(json.loads(text), self.out / JOURNAL_NAME), json.loads(text))

    def test_tampered_results_are_refused(self):
        good = json.loads((self.out / RESULT_NAME).read_text())

        def child(r, role):
            return r['cases'][0]['children'][role]
        edits = {
            'state digest': lambda r: child(r, 'resume').update(state_digest='0' * 64),
            'publish time': lambda r: child(r, 'first')['phases'][-1].update(seconds=0.001),
            'verdict': lambda r: r['cases'][0].update(passed=False),
            'status': lambda r: r.update(status='proof_failed'),
            'dropped case': lambda r: r['cases'].pop(),
            'weights': lambda r: r.update(weights_kept=True),
            'journal tail': lambda r: r.update(journal_tail_sha256='0' * 64),
        }
        for name, edit in edits.items():
            bad = copy.deepcopy(good)
            edit(bad)
            with self.subTest(name), self.assertRaises(ValueError):
                validate_result(bad, self.out / JOURNAL_NAME)

    def test_a_short_deadline_runs_no_case_and_says_so(self):
        result, _ = self.run_proof(runner_seconds=10, min_case_seconds=60)
        self.assertEqual(result['status'], 'runner_timeout')
        self.assertEqual(result['cases'], [])

    def test_a_failed_child_fails_its_case(self):
        result, _ = self.run_proof(child_seconds=0.01)
        self.assertEqual(result['status'], 'proof_failed')
        self.assertIn('timeout', result['cases'][0]['child_errors']['straight'])
        self.assertEqual(result['cases'][0]['children']['first']['child_error'], 'not started')


if __name__ == '__main__':
    unittest.main()
