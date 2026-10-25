"""Profiling P4: private dataset, bundle, approval, ledger and launch against a fake CLI."""

import copy
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiautoresearch.adapters.amazon4u import Amazon4UError
from multiautoresearch.adapters.amazon4u_profile import (
    SCHEMA_APPROVAL, KaggleDatasetCli, build_bundle, build_upload_dir, launch, load_bundle,
)
from multiautoresearch.adapters.amazon4u_gpu import _sha256 as _sha256_bytes
from scripts.gpu_runtime import EXPECTED_DETERMINISM
from scripts.profiling_data import MANIFEST_NAME, _sha256, stage_dataset
from test_amazon4u_gpu import FakeCli, pushed
from test_profiling_data import _csv, _kg_root
from test_profiling_runner import _synthetic

DATASET_REF = 'dungsunf/amazon4u-mi-profiling'

FAKE_DATASETS = r'''#!{python}
import json, os, sys
s = json.load(open(os.environ["FAKE_KAGGLE_DATASETS"]))
with open(s["calls"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
a = sys.argv[1:]
if a[:2] == ["datasets", "status"]:
    print(s.get("status", "ready"))
elif a[:2] == ["datasets", "metadata"]:
    d = a[a.index("-p") + 1]
    if s.get("metadata") is not None:
        with open(os.path.join(d, "dataset-metadata.json"), "w") as f:
            json.dump(s["metadata"], f)
elif a[:2] == ["datasets", "create"]:
    print("Your private Dataset is being created.")
else:
    sys.exit(2)
'''

_SHARED = {}


def shared():
    """One staged dataset, one bundle and one CPU kernel run, shared by every test here."""
    if _SHARED:
        return _SHARED
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    train, valid = _synthetic()
    splits = root / 'splits'
    splits.mkdir()
    _csv(splits / 'Musical_Instruments.train.csv', train)
    _csv(splits / 'Musical_Instruments.valid.csv', valid)
    _csv(splits / 'Musical_Instruments.test.csv', valid)
    stage_dataset(_kg_root(root, train=train), splits, root / 'staged')
    sha = _sha256(root / 'staged' / MANIFEST_NAME)
    upload = root / 'input' / 'amazon4u-mi-profiling'
    upload.parent.mkdir()
    build_upload_dir(root / 'staged', upload, slug='amazon4u-mi-profiling', manifest_sha256=sha)
    bundle = root / 'bundle'
    manifest = build_bundle(1, DATASET_REF, sha, bundle)
    out = root / 'working'
    out.mkdir()
    kernel = runpy.run_path(str(bundle / 'kernel.py'))
    with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
        kernel['main']('cpu', 'none', out, root / 'input')
    _SHARED.update(tmp=tmp, root=root, sha=sha, upload=upload, bundle=bundle, manifest=manifest,
                   manifest_sha=load_bundle(bundle)[1], out=out)
    return _SHARED


def as_gpu(result_path, journal_path, dest):
    """The CPU kernel outputs relabelled as a cuda:0 run, with the journal tail re-sealed."""
    from scripts.profiling_runner import Journal, read_journal
    result = json.loads(result_path.read_text())
    result['environment'].update(device='cuda:0', used_device_count=1, visible_device_count=1)
    result['determinism'] = copy.deepcopy(EXPECTED_DETERMINISM)
    records, _, _ = read_journal(journal_path)
    journal = Journal(dest / 'progress.jsonl', io.StringIO())
    for r in records:
        journal.write(r.pop('event'), **{k: v for k, v in r.items() if k not in ('seq', 'prev_sha256')})
    journal.close()
    result['journal_tail_sha256'] = journal.prev
    (dest / 'result.json').write_text(json.dumps(result))
    return dest / 'result.json', dest / 'progress.jsonl'


class ProfileCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = shared()
        cls.root, cls.sha, cls.bundle, cls.manifest, cls.manifest_sha, cls.out = (
            s['root'], s['sha'], s['bundle'], s['manifest'], s['manifest_sha'], s['out'])

    def setUp(self):
        self._case = tempfile.TemporaryDirectory()
        self.case = Path(self._case.name)
        self.addCleanup(self._case.cleanup)

    def approval(self, **changes):
        path = self.case / f'approval-{"-".join(sorted(changes))}.json'
        path.write_text(json.dumps({
            'schema_version': SCHEMA_APPROVAL, 'run': 1, 'kernel_ref': self.manifest['kernel_ref'],
            'bundle_manifest_sha256': self.manifest_sha, 'dataset_ref': DATASET_REF,
            'dataset_manifest_sha256': self.sha, 'approved_by': 'user',
            'approved_on': '2026-10-25', 'statement': 'approve profiling run 1', **changes}))
        return path

    def datasets(self, **scenario):
        d = self.case / 'dcli'
        d.mkdir(exist_ok=True)
        exe = d / 'kaggle-datasets'
        exe.write_text(FAKE_DATASETS.format(python=sys.executable))
        exe.chmod(0o755)
        calls = d / 'calls.jsonl'
        calls.touch()
        scen = d / 'scenario.json'
        scen.write_text(json.dumps({'calls': str(calls),
                                    'metadata': {'id': DATASET_REF, 'isPrivate': True},
                                    **scenario}))
        patcher = patch.dict(os.environ, {'FAKE_KAGGLE_DATASETS': str(scen)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return KaggleDatasetCli(exe), calls

    def kernels(self, **scenario):
        gpu = self.case / 'gpu'
        gpu.mkdir()
        result, journal = as_gpu(self.out / 'result.json', self.out / 'progress.jsonl', gpu)
        slug = self.manifest['kernel_ref'].split('/')[1]
        log = self.case / 'out.log'
        log.write_text('A4U-PROFILE {}\napi token=abc123\n')
        d = self.case / 'kcli'
        d.mkdir()
        fake = FakeCli(d, **{
            'push_stdout': pushed(1, self.manifest['kernel_ref']),
            'statuses': ['KernelWorkerStatus.RUNNING', 'KernelWorkerStatus.COMPLETE'],
            'outputs': {'result.json': str(result), 'progress.jsonl': str(journal),
                        f'{slug}.log': str(log)}, **scenario})
        patcher = patch.dict(os.environ, {'FAKE_KAGGLE': str(fake.scenario_path)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def launch(self, fake, datasets, approval=None, reports=None):
        return launch(self.bundle, approval or self.approval(), fake.transport(), datasets,
                      reports=reports or self.case / 'reports', sleep=lambda s: None)


class UploadAndKernelTests(ProfileCase):
    def test_upload_dir_is_private_flat_and_has_no_test_split(self):
        upload = shared()['upload']
        names = sorted(p.name for p in upload.iterdir())
        self.assertEqual(len(names), 12)
        self.assertFalse(any('test' in n for n in names))
        self.assertFalse(any(p.is_dir() for p in upload.iterdir()))
        meta = json.loads((upload / 'dataset-metadata.json').read_text())
        self.assertIs(meta['isPrivate'], True)
        self.assertIn('huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023', meta['description'])

    def test_create_never_passes_public_and_refuses_public_metadata(self):
        cli, calls = self.datasets()
        cli.create(shared()['upload'])
        call = json.loads(calls.read_text().splitlines()[0])
        self.assertEqual(call[:2], ['datasets', 'create'])
        self.assertNotIn('--public', call)
        self.assertNotIn('-u', call)
        public = self.case / 'public'
        public.mkdir()
        (public / 'dataset-metadata.json').write_text(json.dumps({'id': DATASET_REF, 'isPrivate': False}))
        with self.assertRaisesRegex(Amazon4UError, 'private'):
            cli.create(public)

    def test_kernel_ran_the_runner_on_the_mounted_dataset(self):
        result = json.loads((self.out / 'result.json').read_text())
        self.assertEqual(result['status'], 'profile_complete', result['stop_reason'])
        self.assertEqual(result['run_binding'], self.manifest['run_binding'])
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ['progress.jsonl', 'result.json'])

    def test_kernel_refuses_when_no_mounted_dataset_has_the_pinned_manifest(self):
        kernel = runpy.run_path(str(self.bundle / 'kernel.py'))
        (self.case / 'input').mkdir()
        with self.assertRaisesRegex(SystemExit, 'found 0'):
            kernel['main']('cpu', 'none', self.case / 'w', self.case / 'input')

    def test_metadata_names_only_the_dataset_and_is_offline(self):
        meta = json.loads((self.bundle / 'kernel-metadata.json').read_text())
        self.assertEqual((meta['dataset_sources'], meta['enable_internet'], meta['is_private']),
                         ([DATASET_REF], False, True))


class LaunchTests(ProfileCase):
    def test_launch_collects_and_records_one_run(self):
        datasets, _ = self.datasets()
        record = self.launch(self.kernels(), datasets)
        self.assertEqual(record['result_status'], 'profile_complete')
        run_dir = self.case / 'reports' / 'run-1'
        self.assertEqual(sorted(p.name for p in run_dir.iterdir()),
                         ['kernel-log.txt', 'progress.jsonl', 'record.json', 'result.json'])
        self.assertNotIn('abc123', (run_dir / 'kernel-log.txt').read_text())

    def test_dataset_not_verified_private_refuses_before_any_push(self):
        fake = self.kernels()
        for scenario in ({'metadata': {'id': DATASET_REF}},
                         {'metadata': {'id': DATASET_REF, 'isPrivate': False}},
                         {'metadata': None}, {'status': 'pending'}):
            with self.subTest(scenario=scenario):
                datasets, _ = self.datasets(**scenario)
                with self.assertRaises(Amazon4UError):
                    self.launch(fake, datasets)
                self.assertEqual(fake.calls('push'), [])
                self.assertFalse((self.case / 'reports' / 'run-1.reserved.json').exists())

    def test_approval_bound_to_bundle_and_dataset(self):
        fake = self.kernels()
        datasets, _ = self.datasets()
        for change in ({'dataset_manifest_sha256': '0' * 64},
                       {'dataset_ref': 'dungsunf/other-dataset'},
                       {'bundle_manifest_sha256': '0' * 64}, {'run': 2}, {'approved_by': 'agent'}):
            with self.subTest(change=change), self.assertRaises(Amazon4UError):
                self.launch(fake, datasets, approval=self.approval(**change))
        self.assertEqual(fake.calls('push'), [])

    def test_profiling_ledger_is_separate_and_capped(self):
        fake = self.kernels()
        datasets, _ = self.datasets()
        reports = self.case / 'reports'
        reports.mkdir()
        for n in (7, 8):
            (reports / f'run-{n}.reserved.json').write_text(json.dumps({'booked_gpu_seconds': 7200}))
        (reports / 'run-9.reserved.json').write_text(json.dumps({'booked_gpu_seconds': 3 * 3600}))
        with self.assertRaisesRegex(Amazon4UError, 'GPU budget'):
            self.launch(fake, datasets, reports=reports)
        self.assertEqual(fake.calls('push'), [])

    def test_tampered_or_mismatched_result_is_not_promoted(self):
        datasets, _ = self.datasets()
        fake = self.kernels()
        gpu = self.case / 'gpu' / 'result.json'
        result = json.loads(gpu.read_text())
        result['cases'][0]['epochs'][2]['training_steps'] = 0.0
        gpu.write_text(json.dumps(result))
        with self.assertRaisesRegex(Amazon4UError, 'collection'):
            self.launch(fake, datasets)
        self.assertFalse((self.case / 'reports' / 'run-1').exists())
        self.assertTrue((self.case / 'reports' / 'run-1.failure.json').exists())

    def test_result_from_another_bundle_device_or_policy_is_not_promoted(self):
        edits = {
            'run binding': lambda r: r['run_binding'].update(bundle_sha256='0' * 64),
            'determinism': lambda r: r.update(determinism=None),
            'cpu device': lambda r: r['environment'].update(device='cpu', used_device_count=0),
            'limits': lambda r: r['limits'].update(threshold_seconds=99999.0),
        }
        for name, edit in edits.items():
            with self.subTest(name):
                case = self.case / name.replace(' ', '-')
                case.mkdir()
                self.case, base = case, self.case
                try:
                    fake = self.kernels()
                    datasets, _ = self.datasets()
                    gpu = self.case / 'gpu' / 'result.json'
                    result = json.loads(gpu.read_text())
                    edit(result)
                    gpu.write_text(json.dumps(result))
                    with self.assertRaisesRegex(Amazon4UError, 'collection'):
                        self.launch(fake, datasets)
                    self.assertFalse((self.case / 'reports' / 'run-1').exists())
                finally:
                    self.case = base

    def test_bundle_whose_metadata_names_another_dataset_is_refused(self):
        bundle = self.case / 'bundle'
        bundle.mkdir()
        for name in ('kernel.py', 'bundle.json'):
            (bundle / name).write_bytes((self.bundle / name).read_bytes())
        meta = json.loads((self.bundle / 'kernel-metadata.json').read_text())
        meta['dataset_sources'] = [DATASET_REF, 'someone/else']
        raw = json.dumps(meta).encode()
        (bundle / 'kernel-metadata.json').write_bytes(raw)
        manifest = json.loads((bundle / 'bundle.json').read_text())
        manifest['metadata_sha256'] = _sha256_bytes(raw)
        (bundle / 'bundle.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(Amazon4UError, 'only the bundle dataset'):
            load_bundle(bundle)

    def test_a_reserved_run_is_never_relaunched(self):
        datasets, _ = self.datasets()
        fake = self.kernels()
        self.launch(fake, datasets)
        with self.assertRaises(FileExistsError):
            self.launch(fake, datasets)
        self.assertEqual(len(fake.calls('push')), 1)


if __name__ == '__main__':
    unittest.main()
