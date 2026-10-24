"""GR08: Kaggle CLI transport and GPU-READY orchestration against a fake CLI (no network).

The fake `kaggle` executable prints what Kaggle CLI 2.2.5 prints and records every call,
so the tests can assert which calls were (and were not) made.
"""

import copy
import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiautoresearch.adapters.amazon4u import Amazon4UError, SubmissionUnknownError
from multiautoresearch.adapters.amazon4u_gpu import (
    ROOT, SCHEMA_APPROVAL, build_bundle, collect_gpu_outputs, launch, load_bundle,
)
from multiautoresearch.adapters.amazon4u_kaggle import KaggleCliTransport
from scripts.gpu_ready_parity import bundle_digest, plan_digest
from scripts.gpu_runtime import EXPECTED_DETERMINISM

FAKE_CLI = r'''#!{python}
import json, os, shutil, sys, time
s = json.load(open(os.environ["FAKE_KAGGLE"]))
with open(s["calls"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
a = sys.argv[1:]
if a == ["--version"]:
    print(s.get("version", "Kaggle CLI 2.2.5"))
elif a[:2] == ["kernels", "push"]:
    time.sleep(s.get("push_sleep", 0))
    print(s["push_stdout"])
    sys.exit(s.get("push_rc", 0))
elif a[:2] == ["kernels", "status"]:
    n = sum('"status"' in line for line in open(s["calls"])) - 1
    print('%s has status "%s"' % (a[2], s["statuses"][min(n, len(s["statuses"]) - 1)]))
elif a[:2] == ["kernels", "output"]:
    d = a[a.index("-p") + 1]
    for name, src in s.get("outputs", {{}}).items():
        shutil.copyfile(src, os.path.join(d, name))
    for name, target in s.get("symlinks", {{}}).items():
        os.symlink(target, os.path.join(d, name))
else:
    sys.exit(2)
'''


def pushed(version, ref):
    return (f'Kernel version {version} successfully pushed.  Please check progress at '
            f'https://www.kaggle.com/code/{ref}')


class FakeCli:
    def __init__(self, tmp, **scenario):
        self.dir = Path(tmp)
        self.exe = self.dir / 'kaggle'
        self.exe.write_text(FAKE_CLI.format(python=sys.executable))
        self.exe.chmod(0o755)
        self.calls_path = self.dir / 'calls.jsonl'
        self.calls_path.touch()
        self.scenario_path = self.dir / 'scenario.json'
        self.scenario_path.write_text(json.dumps({'calls': str(self.calls_path), **scenario}))

    def calls(self, verb=None):
        rows = [json.loads(line) for line in self.calls_path.read_text().splitlines()]
        return [r for r in rows if verb is None or r[1:2] == [verb]]

    def transport(self, **kw):
        return KaggleCliTransport(self.exe, **kw)


def gpu_result(cpu_result):
    """The CPU kernel result relabelled as a cuda:0 run, for collection tests."""
    r = copy.deepcopy(cpu_result)
    r['device'], r['accelerator'] = 'cuda:0', 'gpu'
    r['environment'].update(device='cuda:0', used_device_count=1, visible_device_count=1)
    return r


_SHARED = {}


def shared():
    """One bundle and one CPU kernel run (about 20 s), shared by every test here."""
    if not _SHARED:
        tmp = tempfile.TemporaryDirectory()
        bundle = Path(tmp.name) / 'bundle'
        manifest = build_bundle(1, bundle)
        out = Path(tmp.name) / 'cpu-result.json'
        kernel = runpy.run_path(str(bundle / 'kernel.py'))
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            kernel['main']('cpu', 'none', out)
        _SHARED.update(tmp=tmp, bundle=bundle, manifest=manifest,
                       manifest_sha=load_bundle(bundle)[1], cpu_result=json.loads(out.read_text()))
    return _SHARED


class GpuReadyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = shared()
        cls.bundle, cls.manifest, cls.manifest_sha, cls.cpu_result = (
            s['bundle'], s['manifest'], s['manifest_sha'], s['cpu_result'])

    def setUp(self):
        self._case = tempfile.TemporaryDirectory()
        self.case = Path(self._case.name)

    def tearDown(self):
        self._case.cleanup()

    def write(self, name, obj):
        path = self.case / name
        path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)
        return path

    def approval(self, **changes):
        return self.write(f'approval-{"-".join(sorted(changes))}.json', {
            'schema_version': SCHEMA_APPROVAL, 'run': 1, 'kernel_ref': self.manifest['kernel_ref'],
            'bundle_manifest_sha256': self.manifest_sha, 'approved_by': 'user',
            'approved_on': '2026-10-24', 'statement': 'approve run 1', **changes})

    def cli(self, **scenario):
        ref = self.manifest['kernel_ref']
        slug = ref.split('/')[1]
        defaults = {
            'push_stdout': pushed(1, ref),
            'statuses': ['KernelWorkerStatus.QUEUED', 'KernelWorkerStatus.RUNNING',
                         'KernelWorkerStatus.COMPLETE'],
            'outputs': {'result.json': str(self.write('out.json', gpu_result(self.cpu_result))),
                        f'{slug}.log': str(self.write('out.log', 'epoch 1\napi token=abc123\nAuthorization: Bearer xyz789\n'))},
        }
        cli_dir = self.case / 'cli'
        cli_dir.mkdir()
        fake = FakeCli(cli_dir, **{**defaults, **scenario})
        patcher = patch.dict(os.environ, {'FAKE_KAGGLE': str(fake.scenario_path)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def launch(self, fake, approval=None, **kw):
        return launch(self.bundle, approval or self.approval(), fake.transport(**kw),
                      reports=self.case / 'reports', sleep=lambda s: None)


class TransportTests(GpuReadyCase):
    def test_push_reads_the_acknowledged_version(self):
        fake = self.cli()
        receipt = fake.transport().push(self.bundle, timeout=1800, accelerator='NvidiaTeslaT4',
                                        plan_digest='d')
        self.assertEqual((receipt['returncode'], receipt['acknowledged_version'], receipt['version_ref']),
                         (0, 1, self.manifest['kernel_ref'] + '/v1'))
        self.assertEqual(fake.calls('push')[0][2:], ['-p', str(self.bundle), '-t', '1800',
                                                     '--accelerator', 'NvidiaTeslaT4'])

    def test_push_error_line_is_a_failed_receipt(self):
        fake = self.cli(push_stdout='Kernel push error: Notebook not found')
        receipt = fake.transport().push(self.bundle, timeout=1800, accelerator='x', plan_digest='d')
        self.assertNotEqual(receipt['returncode'], 0)
        self.assertIsNone(receipt['acknowledged_version'])

    def test_unparsed_or_late_push_is_submission_unknown(self):
        for scenario in ({'push_stdout': 'Kernel version successfully pushed.'},
                         {'push_stdout': pushed(1, 'a/b') + '\n' + pushed(2, 'a/b')},
                         {'push_stdout': pushed(1, 'a/b'), 'push_sleep': 3}):
            with self.subTest(scenario=scenario):
                fake = self.cli(**scenario)
                with self.assertRaises(SubmissionUnknownError):
                    fake.transport(command_timeout=1).push(self.bundle, timeout=1800,
                                                           accelerator='x', plan_digest='d')
                shutil.rmtree(self.case / 'cli')

    def test_status_maps_exact_tokens_only(self):
        ref = self.manifest['kernel_ref']
        for printed, want in (('KernelWorkerStatus.COMPLETE', 'complete'),
                              ('KernelWorkerStatus.CANCEL_REQUESTED', None),
                              ('complete', None), ('KernelWorkerStatus.COMPLETED', None)):
            with self.subTest(printed=printed):
                t = self.cli(statuses=[printed]).transport()
                t.push(self.bundle, timeout=1800, accelerator='x', plan_digest='d')
                res = t.status(ref)
                self.assertEqual(res.get('status') if res['returncode'] == 0 else None, want)
                shutil.rmtree(self.case / 'cli')

    def test_status_and_download_refuse_a_kernel_this_transport_did_not_push(self):
        fake = self.cli()
        t = fake.transport()
        self.assertEqual(t.status(self.manifest['kernel_ref'])['returncode'], 1)
        with self.assertRaises(Amazon4UError):
            t.download_outputs(self.manifest['kernel_ref'], self.case)
        self.assertEqual(fake.calls('status') + fake.calls('output'), [])

    def test_cli_version_is_pinned(self):
        for printed in ('Kaggle CLI 2.2.4', 'Kaggle CLI 2.2.6'):
            with self.assertRaisesRegex(Amazon4UError, '2.2.5'):
                self.cli(version=printed).transport().require_cli_version()
            shutil.rmtree(self.case / 'cli')

    def test_status_and_output_name_the_acknowledged_version(self):
        fake = self.cli(push_stdout=pushed(3, self.manifest['kernel_ref']))
        t = fake.transport()
        t.push(self.bundle, timeout=1800, accelerator='x', plan_digest='d')
        self.assertEqual(t.status(self.manifest['kernel_ref'])['version'], 3)
        t.download_outputs(self.manifest['kernel_ref'], self.case)
        versioned = self.manifest['kernel_ref'] + '/3'
        self.assertEqual([c[2] for c in fake.calls('status') + fake.calls('output')],
                         [versioned, versioned])

    def test_log_reader_refuses_a_symlinked_log(self):
        slug = self.manifest['kernel_ref'].split('/')[1]
        target = self.write('secret.txt', 'not a log')
        t = self.cli(outputs={}, symlinks={f'{slug}.log': str(target)}).transport()
        t.push(self.bundle, timeout=1800, accelerator='x', plan_digest='d')
        t.download_outputs(self.manifest['kernel_ref'], self.case)
        self.assertEqual(t.open_log_reader(self.manifest['kernel_ref'])['returncode'], 1)


class BundleTests(GpuReadyCase):
    def test_metadata_is_private_gpu_offline_without_attached_sources(self):
        meta = json.loads((self.bundle / 'kernel-metadata.json').read_text())
        self.assertEqual((meta['is_private'], meta['enable_gpu'], meta['enable_internet']),
                         (True, True, False))
        self.assertEqual([meta[k] for k in ('dataset_sources', 'competition_sources',
                                            'kernel_sources', 'model_sources')], [[]] * 4)
        self.assertEqual(meta['id'], self.manifest['kernel_ref'])

    def test_binding_is_the_checkout_digest_and_the_plan_digest(self):
        plan = json.loads((ROOT / 'configs/gpu-ready-phase2-run-plan.json').read_text())
        self.assertEqual(self.manifest['run_binding'],
                         {'bundle_sha256': bundle_digest(ROOT), 'run_spec_sha256': plan_digest(plan)})

    def test_kernel_built_from_embedded_sources_reports_the_same_binding(self):
        self.assertEqual(self.cpu_result['status'], 'parity_complete')
        self.assertEqual(self.cpu_result['run_binding'], self.manifest['run_binding'])

    def test_refusals(self):
        with self.assertRaises(FileExistsError):
            build_bundle(1, self.bundle)
        with self.assertRaises(Amazon4UError):
            build_bundle(4, self.case / 'b4')
        edited = self.case / 'edited'
        shutil.copytree(self.bundle, edited)
        with open(edited / 'kernel.py', 'a') as f:
            f.write('\n')
        with self.assertRaisesRegex(Amazon4UError, 'kernel.py differs'):
            load_bundle(edited)


class LaunchTests(GpuReadyCase):
    def test_approved_run_is_collected_with_a_record(self):
        fake = self.cli()
        record = self.launch(fake)
        run_dir = self.case / 'reports/run-1'
        self.assertEqual(sorted(os.listdir(run_dir)), ['kernel-log.txt', 'record.json', 'result.json'])
        self.assertEqual(json.loads((run_dir / 'record.json').read_text()), record)
        self.assertEqual((record['schema_version'], record['terminal_status'], record['acceptance'],
                          record['deployment_decision']),
                         ('amazon4u-O-record-cuda-v1', 'complete', 'requires_user_review', 'NOT_AUTHORIZED'))
        log = (run_dir / 'kernel-log.txt').read_text()
        self.assertIn('api token=[REDACTED]', log)
        self.assertIn('Bearer [REDACTED]', log)
        self.assertNotIn('abc123', log)
        self.assertNotIn('xyz789', log)
        self.assertEqual(len(fake.calls('push')), 1)
        self.assertEqual(len(fake.calls('status')), 3)

    def test_no_cli_call_without_a_matching_approval(self):
        fake = self.cli()
        for approval in (self.case / 'missing.json', self.approval(bundle_manifest_sha256='0' * 64),
                         self.approval(approved_by='agent'), self.approval(run=2)):
            with self.subTest(approval=approval.read_text() if approval.exists() else None):
                with self.assertRaises((Amazon4UError, FileNotFoundError)):
                    self.launch(fake, approval)
        self.assertEqual(fake.calls(), [])

    def test_a_run_number_is_never_launched_twice(self):
        fake = self.cli(push_stdout='Kernel push error: quota')
        with self.assertRaisesRegex(Amazon4UError, 'receipt'):
            self.launch(fake)
        with self.assertRaises(FileExistsError):
            self.launch(fake)
        self.assertEqual(len(fake.calls('push')), 1)

    def test_gpu_ledger_refuses_a_run_that_would_exceed_the_budget(self):
        fake = self.cli()
        reports = self.case / 'reports'
        reports.mkdir()
        (reports / 'run-2.reserved.json').write_text(json.dumps({'booked_gpu_seconds': 1800}))
        (reports / 'run-3.reserved.json').write_text(json.dumps({}))
        self.launch(fake)
        reserved = json.loads((reports / 'run-1.reserved.json').read_text())
        self.assertEqual((reserved['gpu_seconds_booked_before'], reserved['booked_gpu_seconds']),
                         (3600, 1800))
        shutil.rmtree(reports)
        reports.mkdir()
        (reports / 'run-2.reserved.json').write_text(json.dumps({'booked_gpu_seconds': 4000}))
        before = len(fake.calls())
        with self.assertRaisesRegex(Amazon4UError, 'GPU budget'):
            self.launch(fake)
        self.assertEqual(len(fake.calls()), before)
        self.assertFalse((reports / 'run-1.reserved.json').exists())

    def test_submission_unknown_records_failure_and_never_repushes(self):
        fake = self.cli(push_stdout='Something unexpected')
        with self.assertRaisesRegex(Amazon4UError, 'submission_unknown'):
            self.launch(fake)
        failure = json.loads((self.case / 'reports/run-1.failure.json').read_text())
        self.assertEqual(failure['stage'], 'push')
        self.assertEqual((len(fake.calls('push')), fake.calls('status')), (1, []))

    def test_error_status_is_recorded_and_not_collected(self):
        fake = self.cli(statuses=['KernelWorkerStatus.ERROR'])
        with self.assertRaisesRegex(Amazon4UError, 'status'):
            self.launch(fake)
        self.assertFalse((self.case / 'reports/run-1').exists())
        self.assertEqual(json.loads((self.case / 'reports/run-1.failure.json').read_text())['stage'],
                         'status')


class CollectionTests(GpuReadyCase):
    def outputs(self, result, extra=None):
        d = self.case / 'download'
        d.mkdir()
        (d / 'result.json').write_text(json.dumps(result))
        for name, text in (extra or {}).items():
            (d / name).write_text(text)
        return d

    def collect(self, d, dest=None):
        return collect_gpu_outputs(d, dest or self.case / 'run-1', manifest=self.manifest,
                                   record_fields={}, log={'status': 'available', 'logs': ''})

    def test_refusals_leave_nothing_promoted(self):
        good = gpu_result(self.cpu_result)
        cases = {
            'cpu result': self.cpu_result,
            'v1 schema': {**good, 'schema_version': 'amazon4u-H-result-v1'},
            'other bundle': {**good, 'run_binding': {**good['run_binding'], 'bundle_sha256': '0' * 64}},
            'two gpus used': {**good, 'environment': {**good['environment'], 'used_device_count': 2}},
            'relaxed determinism': {**good, 'determinism': {**EXPECTED_DETERMINISM,
                                                            'cuda_matmul_allow_tf32': True}},
            'adjusted tolerances': {**good, 'tolerances_float32': {}},
            'verdict forged': {**good, 'all_checks_passed': not good['all_checks_passed']},
        }
        for name, result in cases.items():
            with self.subTest(name):
                d = self.outputs(result)
                with self.assertRaises((Amazon4UError, ValueError)):
                    self.collect(d)
                self.assertFalse((self.case / 'run-1').exists())
                shutil.rmtree(d)

    def test_unexpected_files_symlinks_and_existing_destination_are_refused(self):
        good = gpu_result(self.cpu_result)
        d = self.outputs(good, {'notes.txt': 'x'})
        with self.assertRaisesRegex(Amazon4UError, 'outputs must be'):
            self.collect(d)
        shutil.rmtree(d)
        d = self.outputs(good)
        os.unlink(d / 'result.json')
        os.symlink(self.write('real.json', good), d / 'result.json')
        with self.assertRaisesRegex(Amazon4UError, 'regular file'):
            self.collect(d)
        shutil.rmtree(d)
        d = self.outputs(good)
        (self.case / 'run-1').mkdir()
        with self.assertRaisesRegex(Amazon4UError, 'overwrite refused'):
            self.collect(d)


if __name__ == '__main__':
    unittest.main()
