"""Checkpoint readiness C4: real-data resume equality and checkpoint cost on one device.

Runs checkpoint version `--version`: 1 (run 2) or 2, identity v2 (run 3,
docs/checkpoint-identity-v2-plan.md). The version is part of the plan and the run binding.

For each case (method, batch size, condition) three new Python processes share only the
checkpoint bytes, as in GR06:

  straight  trains EPOCHS epochs and reports the final state;
  first     trains SPLIT epochs and publishes a cuda-v1 checkpoint through D3;
  resume    retrieves it, restores into a fresh trainer and finishes.

The case passes when the logs, the final payload digest and the validation report digest
are identical. Each process times its phases (load, expectations, epochs, payload,
publish or retrieve) and records its peak RSS and GPU memory after each. Validation
metric values are reduced to a SHA-256 and never reported. The checkpoint lives in a
private temporary directory outside the output directory and is deleted after the case;
no weights are kept.

Every finished child is a line in the profiling runner's hash-chained `progress.jsonl`.
`result.json` is written once at the end. Plan: docs/checkpoint-real-data-readiness-plan.md.

Usage: CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.checkpoint_proof \\
           --device cuda:0 --accelerator gpu --data <dataset dir> \\
           --manifest-sha256 <pinned manifest SHA-256> --out-dir <directory>
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from scripts.checkpoint_real_data import HETERO, LIGHTGCN, build
from scripts.gpu_ready_parity import bundle_digest
from scripts.profiling_runner import GIB, Journal, read_journal

SCHEMA = 'amazon4u-checkpoint-proof-result-v1'
VERSIONS = (1, 2)
JOURNAL_NAME = 'progress.jsonl'
RESULT_NAME = 'result.json'
FILE_NAME = 'proof.a4ucp'
SEED = 42
EPOCHS = 4
SPLIT = 2
CASES = (
    (LIGHTGCN, 2048, 'all_ratings'),
    (HETERO, 1024, 'all_ratings'),
    (LIGHTGCN, 2048, 'rating_ge_4'),
    (HETERO, 1024, 'rating_ge_4'),
)
ROLES = ('straight', 'first', 'resume')
TARGETS = {'publish_seconds': 60.0, 'retrieve_seconds': 60.0, 'peak_host_bytes': 8 * GIB}
RUNNER_SECONDS = 6600.0
CHILD_SECONDS = 2400.0
# A case is started only if this much runner time remains; otherwise it is not run.
MIN_CASE_SECONDS = 900.0
STATUSES = ('proof_complete', 'proof_failed', 'runner_timeout', 'failed')
ROOT = Path(__file__).resolve().parents[1]
REMAINING_GATES = [
    'checkpoint proof result reviewed by the user',
    'checkpoint cost added to the comparative-training projection',
    'frozen execution settings and tuning budget',
    'explicit comparative-training authorization',
]


def plan(manifest_sha256, cases=CASES, epochs=EPOCHS, split=SPLIT, version=1):
    spec = {'schema': SCHEMA, 'dataset_manifest_sha256': manifest_sha256, 'seed': SEED,
            'cases': [list(c) for c in cases], 'epochs': epochs, 'split': split}
    # Run 2's plan had no version field; version 1 keeps that exact plan and binding.
    return spec if version == 1 else {**spec, 'checkpoint_version': version}


def run_binding(manifest_sha256, cases=CASES, epochs=EPOCHS, split=SPLIT, root=ROOT, version=1):
    """The checkpoint run binding: the source bundle and this proof plan."""
    spec = json.dumps(plan(manifest_sha256, cases, epochs, split, version), sort_keys=True).encode()
    return {'bundle_sha256': bundle_digest(root), 'run_spec_sha256': hashlib.sha256(spec).hexdigest()}


def _peaks(device):
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    return {'peak_host_bytes': host, 'peak_gpu_allocated_bytes': gpu}


def child(role, *, data_dir, manifest_sha256, device, case, directory, binding, epochs, split, version=1):
    """One role in this process; returns its timings, logs and digests."""
    from scripts.checkpoint_storage_cuda import publish_cuda_checkpoint, retrieve_cuda_checkpoint
    from scripts.checkpoint_v2 import publish_checkpoint_v2, retrieve_checkpoint_v2
    from scripts.gpu_runtime import configure_determinism, environment_fingerprint, pin_global_generators
    from scripts.profiling_data import ProfilingData
    from scripts.training_harness import D2_KEYS, owned_digest

    method, batch_size, condition = case
    pin_global_generators()
    determinism = configure_determinism() if device.type == 'cuda' else None
    fingerprint = environment_fingerprint(device)
    phases = []

    def timed(name, fn):
        start = time.perf_counter()
        value = fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        phases.append({'name': name, 'seconds': time.perf_counter() - start, **_peaks(device)})
        return value

    data = timed('load', lambda: ProfilingData(data_dir, manifest_sha256))
    trainer, expectations, identity = timed('build_expectations', lambda: build(
        data, method, condition, seed=SEED, batch_size=batch_size, device=device,
        fingerprint=fingerprint, determinism=determinism, version=version))
    publish = publish_cuda_checkpoint if version == 1 else publish_checkpoint_v2
    retrieve = retrieve_cuda_checkpoint if version == 1 else retrieve_checkpoint_v2
    make_payload = trainer.cuda_payload if version == 1 else trainer.cuda_payload_v2
    del data
    d2 = {k: expectations[k] for k in D2_KEYS}
    expected_device = {'fingerprint': fingerprint, 'determinism': determinism, 'run_binding': binding}
    out = {'role': role, 'pid': os.getpid(), 'determinism': determinism, 'environment': fingerprint,
           'checkpoint_version': version}
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        if role == 'resume':
            owned = timed('retrieve', lambda: retrieve(
                root, FILE_NAME, expected_device=expected_device, **d2,
                io_policy=expectations['io_policy']))
            timed('restore', lambda: trainer.restore_cuda(owned))
            del owned
        todo = {'straight': epochs, 'first': split, 'resume': epochs - split}[role]
        logs = [timed('train_epoch', trainer.train_epoch) for _ in range(todo)]
        payload = timed('build_payload', lambda: make_payload(expectations, identity, binding))
        if role == 'first':
            timed('publish', lambda: publish(
                root, FILE_NAME, payload, expected_device=expected_device, **d2,
                io_policy=expectations['io_policy']))
            with open(os.path.join(directory, FILE_NAME), 'rb') as f:
                out['file_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
            out['file_bytes'] = os.path.getsize(os.path.join(directory, FILE_NAME))
    finally:
        os.close(root)
    report = json.dumps(trainer.validation_report, sort_keys=True, allow_nan=True)
    out.update(phases=phases, logs=logs, state_digest=owned_digest(payload),
               validation_report_sha256=hashlib.sha256(report.encode()).hexdigest(),
               on_device=all(p.device == device for p in trainer.model.parameters()))
    return out


def _run_child(role, case, directory, args, timeout):
    """Start one role in a new process; its JSON, or the failure as a record."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'out.json')
        command = [sys.executable, '-m', 'scripts.checkpoint_proof', 'child', '--role', role,
                   '--case', json.dumps(list(case)), '--dir', directory, '--out', out,
                   '--data', str(args['data_dir']), '--manifest-sha256', args['manifest_sha256'],
                   '--device', args['device'], '--binding', json.dumps(args['binding']),
                   '--epochs', str(args['epochs']), '--split', str(args['split']),
                   '--version', str(args['version'])]
        try:
            proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {'role': role, 'child_error': f'timeout after {timeout:.0f} s'}
        if proc.returncode != 0 or not os.path.exists(out):
            return {'role': role, 'child_error': f'exit {proc.returncode}: {proc.stderr[-2000:]}'}
        with open(out) as f:
            return json.load(f)


def _phase(record, name):
    found = [p for p in record.get('phases', ()) if p['name'] == name]
    return found[0] if found else None


def summarize(case, records):
    """A case's verdict and checkpoint costs, recomputed from its three child records."""
    errors = {r['role']: r['child_error'] for r in records.values() if 'child_error' in r}
    summary = {'case': list(case), 'children': records}
    if errors:
        return {**summary, 'child_errors': errors, 'passed': False}
    straight, first, resume = (records[r] for r in ROLES)
    checks = {
        'distinct_processes': len({straight['pid'], first['pid'], resume['pid']}) == 3,
        'logs_identical': straight['logs'] == first['logs'] + resume['logs'],
        'state_identical': straight['state_digest'] == resume['state_digest'],
        'validation_identical': straight['validation_report_sha256'] == resume['validation_report_sha256'],
        'parameters_on_device_after_restore': resume['on_device'],
    }
    publish, retrieve = _phase(first, 'publish'), _phase(resume, 'retrieve')
    cost = {
        'publish_seconds': publish['seconds'], 'retrieve_seconds': retrieve['seconds'],
        'file_bytes': first['file_bytes'],
        'publish_process_peak_host_bytes': publish['peak_host_bytes'],
        'retrieve_process_peak_host_bytes': retrieve['peak_host_bytes'],
        'expectations_seconds': [_phase(r, 'build_expectations')['seconds'] for r in (straight, first, resume)],
    }
    targets = {
        'publish_seconds': cost['publish_seconds'] <= TARGETS['publish_seconds'],
        'retrieve_seconds': cost['retrieve_seconds'] <= TARGETS['retrieve_seconds'],
        'peak_host_bytes': max(cost['publish_process_peak_host_bytes'],
                               cost['retrieve_process_peak_host_bytes']) <= TARGETS['peak_host_bytes'],
    }
    return {**summary, 'checks': checks, 'cost': cost, 'targets_met': targets,
            'passed': all(checks.values())}


def run(data_dir, manifest_sha256, device_name, out_dir, *, cases=CASES, epochs=EPOCHS, split=SPLIT,
        runner_seconds=RUNNER_SECONDS, child_seconds=CHILD_SECONDS, min_case_seconds=MIN_CASE_SECONDS,
        environment=None, clock=time.monotonic, stream=sys.stdout, version=1):
    """Run the proof into `out_dir` (it may exist; files are opened with `x`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(mode=0o700, exist_ok=True)
    if version not in VERSIONS:
        raise ValueError(f'unknown checkpoint version {version!r}')
    binding = run_binding(manifest_sha256, cases, epochs, split, version=version)
    journal = Journal(out_dir / JOURNAL_NAME, stream)
    journal.write('start', run_binding=binding, plan=plan(manifest_sha256, cases, epochs, split, version),
                  targets=TARGETS, runner_seconds=runner_seconds)
    start = clock()
    results, status, stop_reason = [], 'proof_complete', None
    args = {'data_dir': data_dir, 'manifest_sha256': manifest_sha256, 'device': device_name,
            'binding': binding, 'epochs': epochs, 'split': split, 'version': version}
    try:
        for case in cases:
            remaining = runner_seconds - (clock() - start)
            if remaining < min_case_seconds:
                status, stop_reason = 'runner_timeout', f'{remaining:.0f} s left; later cases not run'
                break
            records = {}
            with tempfile.TemporaryDirectory(prefix='a4u-proof-') as shared:
                os.chmod(shared, 0o700)
                for role in ROLES:
                    remaining = runner_seconds - (clock() - start)
                    records[role] = (_run_child(role, case, shared, args, min(child_seconds, remaining))
                                     if remaining > 0 else {'role': role, 'child_error': 'runner deadline'})
                    journal.write('child', case=list(case), role=role,
                                  record=_journal_view(records[role]))
                    if 'child_error' in records[role]:
                        break
            for role in ROLES:
                records.setdefault(role, {'role': role, 'child_error': 'not started'})
            summary = summarize(case, records)
            results.append(summary)
            journal.write('case', case=list(case), passed=summary['passed'],
                          cost=summary.get('cost'), targets_met=summary.get('targets_met'))
        if status == 'proof_complete' and not all(r['passed'] for r in results):
            status, stop_reason = 'proof_failed', 'a case did not resume exactly'
    except Exception as exc:
        status, stop_reason = 'failed', f'{type(exc).__name__}: {exc}'
    journal.write('end', status=status, stop_reason=stop_reason)
    journal.close()
    _, tail, _ = read_journal(out_dir / JOURNAL_NAME)
    result = {
        'schema_version': SCHEMA, 'status': status, 'stop_reason': stop_reason,
        'plan': plan(manifest_sha256, cases, epochs, split, version), 'targets': TARGETS, 'cases': results,
        'environment': environment, 'run_binding': binding, 'journal_tail_sha256': tail,
        'weights_kept': False, 'metric_values_reported': False, 'test_execution': 'not_authorized',
        'deployment_decision': 'NOT_AUTHORIZED', 'remaining_gates': list(REMAINING_GATES),
    }
    with open(out_dir / RESULT_NAME, 'x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    return validate_result(result, out_dir / JOURNAL_NAME)


def _journal_view(record):
    """A child record without its bulky or collection-checked fields."""
    return {k: v for k, v in record.items() if k not in ('environment', 'determinism', 'logs')}


RESULT_KEYS = {'schema_version', 'status', 'stop_reason', 'plan', 'targets', 'cases', 'environment',
               'run_binding', 'journal_tail_sha256', 'weights_kept', 'metric_values_reported',
               'test_execution', 'deployment_decision', 'remaining_gates'}


def validate_result(result, journal_path):
    """Refuse a result that differs from its journal or whose verdicts do not follow."""
    if type(result) is not dict or set(result) != RESULT_KEYS or result['schema_version'] != SCHEMA:
        raise ValueError('result keys or schema differ from the proof result')
    if result['status'] not in STATUSES:
        raise ValueError(f'unknown status {result["status"]!r}')
    if result['weights_kept'] is not False or result['metric_values_reported'] is not False:
        raise ValueError('a proof result keeps no weights and reports no metric values')
    records, tail, cut_off = read_journal(journal_path)
    if cut_off or tail != result['journal_tail_sha256']:
        raise ValueError('result does not match its journal')
    if records[0]['event'] != 'start' or records[0]['run_binding'] != result['run_binding']:
        raise ValueError('journal start differs from the result')
    if records[-1] != {**records[-1], 'event': 'end', 'status': result['status'],
                       'stop_reason': result['stop_reason']}:
        raise ValueError('journal end differs from the result')
    journaled = [r for r in records if r['event'] == 'case']
    if len(journaled) != len(result['cases']):
        raise ValueError('result cases differ from the journal')
    for entry, case in zip(journaled, result['cases']):
        again = summarize(tuple(case['case']), case['children'])
        if again != case or [entry['passed'], entry['cost'], entry['targets_met']] != [
                case['passed'], case.get('cost'), case.get('targets_met')]:
            raise ValueError(f'case {case["case"]} verdict does not follow from its records')
        for role in ROLES:
            views = [r['record'] for r in records if r['event'] == 'child'
                     and r['case'] == case['case'] and r['role'] == role]
            if views and views[0] != _journal_view(case['children'][role]):
                raise ValueError(f'case {case["case"]} {role} record differs from the journal')
    planned = [list(c) for c in result['plan']['cases']]
    if [c['case'] for c in result['cases']] != planned[:len(result['cases'])]:
        raise ValueError('cases ran out of plan order')
    if result['status'] == 'proof_complete' and (len(result['cases']) != len(planned)
                                                 or not all(c['passed'] for c in result['cases'])):
        raise ValueError('proof_complete needs every planned case run and passed')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command')
    c = sub.add_parser('child')
    for name in ('--role', '--case', '--dir', '--out', '--data', '--manifest-sha256', '--device', '--binding'):
        c.add_argument(name, required=True)
    c.add_argument('--epochs', type=int, required=True)
    c.add_argument('--split', type=int, required=True)
    c.add_argument('--version', type=int, choices=VERSIONS, default=1)
    parser.add_argument('--device')
    parser.add_argument('--accelerator')
    parser.add_argument('--data')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--out-dir')
    parser.add_argument('--version', type=int, choices=VERSIONS, default=1)
    args = parser.parse_args(argv)
    if args.command == 'child':
        record = child(args.role, data_dir=args.data, manifest_sha256=args.manifest_sha256,
                       device=torch.device(args.device), case=tuple(json.loads(args.case)),
                       directory=args.dir, binding=json.loads(args.binding),
                       epochs=args.epochs, split=args.split, version=args.version)
        with open(args.out, 'x') as f:
            json.dump(record, f, sort_keys=True)
        return 0
    from scripts.gpu_runtime import environment_fingerprint, require_device
    device = require_device(args.device, args.accelerator)
    result = run(args.data, args.manifest_sha256, args.device, args.out_dir,
                 environment=environment_fingerprint(device), version=args.version)
    print(json.dumps({'status': result['status'], 'stop_reason': result['stop_reason']}))
    return 0 if result['status'] == 'proof_complete' else 1


if __name__ == '__main__':
    sys.exit(main())
