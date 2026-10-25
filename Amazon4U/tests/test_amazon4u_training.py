"""Training campaign: plan, bundle, approvals, ledger and collection against a fake Kaggle CLI."""

import copy
import io
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiautoresearch.adapters import amazon4u_training as tr
from multiautoresearch.adapters.amazon4u import Amazon4UError
from multiautoresearch.adapters.amazon4u_profile import SCHEMA_APPROVAL, load_bundle
from scripts.gpu_runtime import EXPECTED_DETERMINISM
from scripts.profiling_runner import Journal, read_journal
from test_amazon4u_gpu import FakeCli, pushed
from test_amazon4u_profile import DATASET_REF, ProfileCase, shared

TUNING = {'method': tr.METHODS[0], 'condition': 'all_ratings', 'learning_rate': 0.001, 'seed': 42, 'stage': 'tuning'}


def as_gpu(out, dest):
    """CPU kernel outputs relabelled as a cuda:0 run under the fixed policy, journals re-sealed."""
    dest.mkdir()
    result = json.loads((out / 'result.json').read_text())
    result['environment'].update(device='cuda:0', used_device_count=1, visible_device_count=1)
    result['determinism'] = copy.deepcopy(EXPECTED_DETERMINISM)
    attempts = []
    for name in sorted(n for n in os.listdir(out) if n.startswith('progress-')):
        records, _, _ = read_journal(out / name)
        journal = Journal(dest / name, io.StringIO())
        for r in records:
            journal.write(r.pop('event'), **{k: v for k, v in r.items() if k not in ('seq', 'prev_sha256')})
        journal.close()
        attempts.append({**result['attempts'][len(attempts)], 'tail_sha256': journal.prev})
    result['attempts'] = attempts
    (dest / 'final.a4ucp').write_bytes((out / 'final.a4ucp').read_bytes())
    (dest / 'result.json').write_text(json.dumps(result))
    return result


class TrainingCampaignTests(ProfileCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._t = tempfile.TemporaryDirectory()
        root = Path(cls._t.name)
        cls.empty = root / 'no-reports'
        cls.bundle = root / 'bundle'
        cls.manifest = tr.build_bundle(1, TUNING, DATASET_REF, cls.sha, cls.bundle, reports=cls.empty)
        cls.manifest_sha = load_bundle(cls.bundle)[1]
        cls.out = root / 'working'
        cls.out.mkdir()
        kernel = runpy.run_path(str(cls.bundle / 'kernel.py'))
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            kernel['main']('cpu', 'none', cls.out, shared()['root'] / 'input')

    @classmethod
    def tearDownClass(cls):
        cls._t.cleanup()

    def kernels(self, result_dir, tag=''):
        slug = self.manifest['kernel_ref'].split('/')[1]
        log = self.case / 'out.log'
        log.write_text('A4U-PROFILE {}\n')
        d = self.case / f'kcli{tag}'
        d.mkdir()
        outputs = {n: str(result_dir / n) for n in os.listdir(result_dir)}
        fake = FakeCli(d, push_stdout=pushed(1, self.manifest['kernel_ref']),
                       statuses=['KernelWorkerStatus.RUNNING', 'KernelWorkerStatus.COMPLETE'],
                       outputs={**outputs, f'{slug}.log': str(log)})
        patcher = patch.dict(os.environ, {'FAKE_KAGGLE': str(fake.scenario_path)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def campaign_approval(self, **changes):
        path = self.case / 'campaign.json'
        path.write_text(json.dumps({
            'schema_version': tr.SCHEMA_CAMPAIGN_APPROVAL, 'campaign_plan_sha256': tr.plan_sha256(self.sha),
            'source_bundle_sha256': self.manifest['source_bundle_sha256'], 'dataset_ref': DATASET_REF,
            'dataset_manifest_sha256': self.sha, 'cap_gpu_seconds': tr.CAP_GPU_SECONDS,
            'approved_by': 'user', 'approved_on': '2026-10-26', 'statement': 'approve the campaign', **changes}))
        return path

    def run_approval(self):
        path = self.case / 'run.json'
        path.write_text(json.dumps({
            'schema_version': SCHEMA_APPROVAL, 'run': 1, 'kernel_ref': self.manifest['kernel_ref'],
            'bundle_manifest_sha256': self.manifest_sha, 'dataset_ref': DATASET_REF,
            'dataset_manifest_sha256': self.sha, 'approved_by': 'user', 'approved_on': '2026-10-26',
            'statement': 'approve run 1'}))
        return path

    def test_the_kernel_runs_the_planned_run_on_the_mounted_dataset(self):
        result = json.loads((self.out / 'result.json').read_text())
        self.assertEqual(result['status'], 'run_complete')
        self.assertEqual(result['spec'], self.manifest['spec'])
        self.assertEqual(result['run_binding'], self.manifest['run_binding'])
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ['final.a4ucp', 'progress-1.jsonl', 'result.json'])

    def test_plan_holds_twelve_tuning_runs_and_refuses_anything_else(self):
        plan = tr.campaign_plan(self.sha)
        self.assertEqual(len(plan['tuning_runs']), 12)
        self.assertEqual({r['learning_rate'] for r in plan['tuning_runs']}, set(tr.LEARNING_RATES))
        for bad in ({**TUNING, 'seed': 2026}, {**TUNING, 'learning_rate': 0.003},
                    {**TUNING, 'stage': 'final', 'seed': 2026}):
            with self.subTest(bad), self.assertRaises((Amazon4UError, ValueError)):
                tr.build_bundle(2, bad, DATASET_REF, self.sha, self.case / f'b{len(os.listdir(self.case))}',
                                reports=self.empty)

    def test_final_runs_unlock_with_the_selected_rate_only(self):
        reports = self.case / 'reports'
        reports.mkdir()
        for run, (lr, score) in enumerate(((0.0005, 0.2), (0.001, 0.3), (0.002, 0.3)), 1):
            (reports / f'run-{run}.reserved.json').write_text('{"booked_gpu_seconds": 2610}')
            (reports / f'run-{run}').mkdir()
            (reports / f'run-{run}' / 'record.json').write_text(json.dumps({
                'spec': {**TUNING, 'learning_rate': lr}, 'best_validation_NDCG@10': score,
                'actual_gpu_seconds': 1000}))
        self.assertEqual(tr.select_learning_rate(reports, TUNING['method'], 'all_ratings'), 0.001)
        self.assertIsNone(tr.select_learning_rate(reports, TUNING['method'], 'rating_ge_4'))
        final = {**TUNING, 'stage': 'final', 'seed': 2026}
        tr.build_bundle(4, final, DATASET_REF, self.sha, self.case / 'final-ok', reports=reports)
        with self.assertRaisesRegex(Amazon4UError, 'learning rate 0.001'):
            tr.build_bundle(5, {**final, 'learning_rate': 0.002}, DATASET_REF, self.sha, self.case / 'f2', reports=reports)
        with self.assertRaisesRegex(Amazon4UError, 'wait until all three'):
            tr.build_bundle(6, {**final, 'condition': 'rating_ge_4'}, DATASET_REF, self.sha, self.case / 'f3', reports=reports)
        self.assertEqual(tr.training_ledger(reports), 3000)

    def test_launch_with_either_approval_collects_and_records_actual_use(self):
        for approval in (self.run_approval, self.campaign_approval):
            with self.subTest(approval.__name__):
                gpu = self.case / f'gpu-{approval.__name__}'
                as_gpu(self.out, gpu)
                reports = self.case / f'reports-{approval.__name__}'
                datasets, _ = self.datasets()
                fake = self.kernels(gpu, approval.__name__)
                record = tr.launch(self.bundle, approval(), fake.transport(), datasets, reports=reports,
                                   sleep=lambda s: None)
                self.assertEqual(record['result_status'], 'run_complete')
                self.assertEqual(sorted(os.listdir(reports / 'run-1')),
                                 ['final.a4ucp', 'kernel-log.txt', 'progress-1.jsonl', 'record.json', 'result.json'])
                self.assertLess(tr.training_ledger(reports), tr.BOOKING_SECONDS[TUNING['method']])
                self.assertEqual(record['test_execution'], 'not_run')

    def test_a_campaign_approval_for_other_sources_or_cap_is_refused(self):
        approve = tr.approver(self.empty)
        for change in ({'source_bundle_sha256': '0' * 64}, {'cap_gpu_seconds': 999999},
                       {'campaign_plan_sha256': '1' * 64}, {'approved_by': 'agent'}):
            approval = json.loads(self.campaign_approval(**change).read_text())
            with self.subTest(change), self.assertRaises(Amazon4UError):
                approve(approval, self.manifest, self.manifest_sha)

    def test_cpu_or_tampered_results_are_not_collected(self):
        gpu = self.case / 'g'
        good = as_gpu(self.out, gpu)
        tr.check_result(good, gpu, self.manifest)
        for name, edit in {'cpu': lambda r: r['environment'].update(device='cpu'),
                           'policy': lambda r: r.update(determinism=None),
                           'score': lambda r: r.update(**{'best_validation_NDCG@10': 0.99})}.items():
            bad = copy.deepcopy(good)
            edit(bad)
            with self.subTest(name), self.assertRaises((Amazon4UError, ValueError)):
                tr.check_result(bad, gpu, self.manifest)


if __name__ == '__main__':
    unittest.main()
