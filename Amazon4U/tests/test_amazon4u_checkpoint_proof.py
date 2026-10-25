"""Checkpoint readiness C4: proof bundle, kernel and launch against a fake Kaggle CLI."""

import copy
import io
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiautoresearch.adapters.amazon4u import Amazon4UError
from multiautoresearch.adapters.amazon4u_checkpoint_proof import build_bundle, check_result, collect
from multiautoresearch.adapters.amazon4u_profile import SCHEMA_APPROVAL, launch, load_bundle
from scripts.gpu_runtime import EXPECTED_DETERMINISM
from scripts.profiling_runner import Journal, read_journal
from test_amazon4u_gpu import FakeCli, pushed
from test_amazon4u_profile import DATASET_REF, ProfileCase, shared


def as_gpu(out, dest):
    """CPU proof outputs relabelled as a cuda:0 run under the fixed policy, journal re-sealed."""
    result = json.loads((out / 'result.json').read_text())
    result['environment'].update(device='cuda:0', used_device_count=1, visible_device_count=1)
    for case in result['cases']:
        for record in case['children'].values():
            record.update(environment=result['environment'], determinism=copy.deepcopy(EXPECTED_DETERMINISM))
    records, _, _ = read_journal(out / 'progress.jsonl')
    journal = Journal(dest / 'progress.jsonl', io.StringIO())
    for r in records:
        journal.write(r.pop('event'), **{k: v for k, v in r.items() if k not in ('seq', 'prev_sha256')})
    journal.close()
    result['journal_tail_sha256'] = journal.prev
    (dest / 'result.json').write_text(json.dumps(result))
    return result


class ProofLaunchTests(ProfileCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._proof = tempfile.TemporaryDirectory()
        root = Path(cls._proof.name)
        cls.bundle = root / 'bundle'
        cls.manifest = build_bundle(2, DATASET_REF, cls.sha, cls.bundle, version=2)
        cls.manifest_sha = load_bundle(cls.bundle)[1]
        cls.out = root / 'working'
        cls.out.mkdir()
        kernel = runpy.run_path(str(cls.bundle / 'kernel.py'))
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            kernel['main']('cpu', 'none', cls.out, shared()['root'] / 'input')

    @classmethod
    def tearDownClass(cls):
        cls._proof.cleanup()

    def approval(self, **changes):
        return super().approval(**{'run': 2, 'statement': 'Go ahead with the Kaggle proof run.', **changes})

    def kernels(self, **scenario):
        gpu = self.case / 'gpu'
        gpu.mkdir()
        as_gpu(self.out, gpu)
        slug = self.manifest['kernel_ref'].split('/')[1]
        log = self.case / 'out.log'
        log.write_text('A4U-PROFILE {}\n')
        d = self.case / 'kcli'
        d.mkdir()
        fake = FakeCli(d, **{
            'push_stdout': pushed(1, self.manifest['kernel_ref']),
            'statuses': ['KernelWorkerStatus.RUNNING', 'KernelWorkerStatus.COMPLETE'],
            'outputs': {'result.json': str(gpu / 'result.json'), 'progress.jsonl': str(gpu / 'progress.jsonl'),
                        f'{slug}.log': str(log)}, **scenario})
        patcher = patch.dict(os.environ, {'FAKE_KAGGLE': str(fake.scenario_path)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def test_kernel_runs_the_proof_on_the_mounted_dataset(self):
        result = json.loads((self.out / 'result.json').read_text())
        self.assertEqual(result['status'], 'proof_complete', result['stop_reason'])
        self.assertEqual(result['run_binding'], self.manifest['run_binding'])
        self.assertEqual(len(result['cases']), 4)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ['progress.jsonl', 'result.json'])
        self.assertIn('amazon4u-ckpt-proof-mi-run-2-', self.manifest['kernel_ref'])
        self.assertEqual(result['plan']['checkpoint_version'], 2)
        self.assertTrue(all(c['children']['first']['checkpoint_version'] == 2 for c in result['cases']))

    def test_launch_books_run_2_in_the_profiling_ledger_and_collects(self):
        reports = self.case / 'reports'
        reports.mkdir()
        (reports / 'run-1.reserved.json').write_text(json.dumps({'booked_gpu_seconds': 7200}))
        datasets, _ = self.datasets()
        record = launch(self.bundle, self.approval(), self.kernels().transport(), datasets,
                        reports=reports, sleep=lambda s: None, collect=collect)
        self.assertEqual(record['result_status'], 'proof_complete')
        self.assertEqual(record['gpu_seconds_booked_before'], 7200)
        self.assertIn('checkpoint proof result reviewed by the user', record['remaining_gates'])
        self.assertTrue((reports / 'run-2' / 'record.json').exists())

    def test_cpu_or_off_policy_results_are_refused(self):
        gpu = self.case / 'g'
        gpu.mkdir()
        good = as_gpu(self.out, gpu)
        check_result(good, gpu / 'progress.jsonl', self.manifest)
        for name, edit in {
            'cpu': lambda r: r['environment'].update(device='cpu'),
            'policy': lambda r: r['cases'][0]['children']['resume'].update(determinism=None),
            'binding': lambda r: r['run_binding'].update(bundle_sha256='0' * 64),
        }.items():
            bad = copy.deepcopy(good)
            edit(bad)
            with self.subTest(name), self.assertRaises((Amazon4UError, ValueError)):
                check_result(bad, gpu / 'progress.jsonl', self.manifest)


if __name__ == '__main__':
    unittest.main()
