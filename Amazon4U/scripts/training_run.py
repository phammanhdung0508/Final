"""One Musical_Instruments comparative-training run under amazon4u-protocol-v2.1.0.

A run is one (method, feedback condition, learning rate, seed). It trains the EX01
vectorized model with the Block 2 recipe until early stopping (10 validation checks
without a strictly better validation NDCG@10) or 100 epochs, validating with full
ranking every epoch. Settings: docs/musical-instruments-training-settings.md and
docs/learning-rate-tuning-amendment.md.

Checkpoints are identity v2, published through D3 into `<out>/checkpoints/`: one every
`CHECKPOINT_EVERY[method]` epochs and `final.a4ucp` at the end, which also holds the
best-model snapshot that the test stage will use. If the directory already holds
checkpoints, the run resumes from the latest one, exactly (proof runs 2 and 3), so a
crashed or timed-out attempt continues instead of restarting.

Each attempt writes its own hash-chained `progress-<n>.jsonl`, with one line per epoch
(losses, timings, peak memory, validation metrics and audit counts). `result.json` is
written once, when the run finishes. Only training and validation data are loaded; the
test split is not part of the dataset.

Usage: CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.training_run --device cuda:0 \\
           --accelerator gpu --data <dataset dir> --manifest-sha256 <sha> --out-dir <dir> \\
           --method LightGCN --condition all_ratings --learning-rate 0.001 --seed 42 --stage tuning
"""

import argparse
import hashlib
import json
import os
import re
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

from scripts.checkpoint_real_data import HETERO, LIGHTGCN, METHODS, build
from scripts.checkpoint_v2 import APPROVED_LEARNING_RATES, publish_checkpoint_v2, retrieve_checkpoint_v2
from scripts.gpu_ready_parity import bundle_digest
from scripts.profiling_data import CONDITIONS, ProfilingData
from scripts.profiling_runner import Journal, read_journal
from scripts.training_harness import D2_KEYS

SCHEMA = 'amazon4u-training-run-result-v1'
PROTOCOL_VERSION = 'amazon4u-protocol-v2.1.0'
RESULT_NAME = 'result.json'
CHECKPOINT_DIR = 'checkpoints'
FINAL_NAME = 'final.a4ucp'
BATCH_SIZE = {LIGHTGCN: 2048, HETERO: 1024}
CHECKPOINT_EVERY = {LIGHTGCN: 20, HETERO: 10}
SEEDS = {'tuning': (42,), 'final': (2026, 3407)}
STATUSES = ('run_complete', 'runner_timeout', 'failed')
_PERIODIC = re.compile(r'epoch-(\d{3})\.a4ucp')
_PROGRESS = re.compile(r'progress-(\d+)\.jsonl')


def spec(method, condition, learning_rate, seed, stage, manifest_sha256):
    """The run's identity; its hash is the checkpoint run binding's run-spec digest."""
    if method not in METHODS or condition not in CONDITIONS:
        raise ValueError(f'unknown method or condition {method!r} {condition!r}')
    if learning_rate not in APPROVED_LEARNING_RATES or type(learning_rate) is not float:
        raise ValueError(f'learning rate must be one of {APPROVED_LEARNING_RATES}')
    if stage not in SEEDS or seed not in SEEDS[stage]:
        raise ValueError(f'seed {seed!r} is not a {stage} seed')
    return {'schema': SCHEMA, 'protocol_version': PROTOCOL_VERSION, 'category': 'Musical_Instruments',
            'method': method, 'condition': condition, 'learning_rate': learning_rate, 'seed': seed,
            'stage': stage, 'batch_size': BATCH_SIZE[method], 'checkpoint_every': CHECKPOINT_EVERY[method],
            'validation_frequency': 1, 'dataset_manifest_sha256': manifest_sha256}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _peaks(device):
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    return {'peak_host_bytes': host, 'peak_gpu_allocated_bytes': gpu}


def _latest_checkpoint(directory):
    found = sorted((int(m.group(1)), name) for name in os.listdir(directory)
                   if (m := _PERIODIC.fullmatch(name)))
    return found[-1][1] if found else None


def _epoch_record(log, report, seconds, device):
    losses = [step['total_loss'] for step in log['losses']]
    record = {'epoch': log['epoch'], 'optimizer_steps': log['optimizer_steps'],
              'empty_pool_skips': log['empty_pool_skips'], 'mean_total_loss': statistics.fmean(losses),
              'score': log['score'], 'seconds': seconds, **_peaks(device)}
    if report is not None:
        record['validation'] = {'overall_metrics': report['overall_metrics'], 'audit': report['audit']}
    return record


def run(data_dir, manifest_sha256, device, out_dir, *, method, condition, learning_rate, seed, stage,
        determinism=None, runner_seconds=6600.0, clock=time.monotonic, stream=sys.stdout,
        max_epochs=None):
    """Train (or resume) the run in `out_dir`; return the result once it is complete.

    Stops cleanly with status `runner_timeout` at the deadline, after publishing a
    checkpoint, so the next attempt resumes. `max_epochs` only limits this attempt (tests).
    """
    from scripts.gpu_runtime import environment_fingerprint, pin_global_generators
    out_dir = Path(out_dir)
    out_dir.mkdir(mode=0o700, exist_ok=True)
    checkpoints = out_dir / CHECKPOINT_DIR
    checkpoints.mkdir(mode=0o700, exist_ok=True)
    os.chmod(checkpoints, 0o700)
    if (out_dir / RESULT_NAME).exists():
        raise FileExistsError('run already complete; result.json exists')
    run_spec = spec(method, condition, learning_rate, seed, stage, manifest_sha256)
    binding = {'bundle_sha256': bundle_digest(), 'run_spec_sha256': _digest(run_spec)}
    attempt = 1 + max([int(m.group(1)) for n in os.listdir(out_dir) if (m := _PROGRESS.fullmatch(n))],
                      default=0)
    journal = Journal(out_dir / f'progress-{attempt}.jsonl', stream)
    journal.write('start', spec=run_spec, run_binding=binding, attempt=attempt)
    start = clock()
    status, reason = 'run_complete', None
    pin_global_generators()
    fingerprint = environment_fingerprint(device)
    root = os.open(checkpoints, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        data = ProfilingData(data_dir, manifest_sha256)
        trainer, expectations, reference = build(
            data, method, condition, seed=seed, batch_size=BATCH_SIZE[method], device=device,
            fingerprint=fingerprint, determinism=determinism, version=2, learning_rate=learning_rate)
        del data
        d2 = {k: expectations[k] for k in D2_KEYS}
        expected_device = {'fingerprint': fingerprint, 'determinism': determinism, 'run_binding': binding}

        def publish(name):
            t0 = clock()
            publish_checkpoint_v2(root, name, trainer.cuda_payload_v2(expectations, reference, binding),
                                  expected_device=expected_device, **d2, io_policy=expectations['io_policy'])
            journal.write('checkpoint', name=name, epoch=trainer.progress['epoch'], seconds=clock() - t0)

        latest = _latest_checkpoint(checkpoints)
        if latest is not None:
            t0 = clock()
            trainer.restore_cuda(retrieve_checkpoint_v2(root, latest, expected_device=expected_device, **d2,
                                                        io_policy=expectations['io_policy']))
            journal.write('resume', name=latest, epoch=trainer.progress['epoch'], seconds=clock() - t0)
        done_here = 0
        while trainer.selection.state['status'] != 'stopped':
            if clock() - start > runner_seconds or (max_epochs is not None and done_here >= max_epochs):
                status, reason = 'runner_timeout', f'stopped after epoch {trainer.progress["epoch"]}; resumable'
                if trainer.progress['epoch'] and _latest_checkpoint(checkpoints) != f'epoch-{trainer.progress["epoch"]:03d}.a4ucp':
                    publish(f'epoch-{trainer.progress["epoch"]:03d}.a4ucp')
                break
            t0 = clock()
            log = trainer.train_epoch()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            report = trainer.validation_report if log['score'] is not None else None
            journal.write('epoch', record=_epoch_record(log, report, clock() - t0, device))
            done_here += 1
            epoch = trainer.progress['epoch']
            if epoch % CHECKPOINT_EVERY[method] == 0 and trainer.selection.state['status'] != 'stopped':
                publish(f'epoch-{epoch:03d}.a4ucp')
        if status == 'run_complete':
            publish(FINAL_NAME)
    except Exception as exc:
        status, reason = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        os.close(root)
    journal.write('end', status=status, stop_reason=reason)
    journal.close()
    if status != 'run_complete':
        return {'status': status, 'stop_reason': reason, 'attempt': attempt}
    return _write_result(out_dir, run_spec, binding, trainer.selection.state, fingerprint, determinism)


def _attempts(out_dir):
    names = sorted((int(m.group(1)), n) for n in os.listdir(out_dir) if (m := _PROGRESS.fullmatch(n)))
    return [(n, read_journal(out_dir / n)) for _, n in names]


def _write_result(out_dir, run_spec, binding, selection, environment, determinism):
    """The run's result, rebuilt from every attempt's journal and the final checkpoint."""
    epochs, attempts = {}, []
    for name, (records, tail, cut_off) in _attempts(out_dir):
        attempts.append({'journal': name, 'tail_sha256': tail, 'cut_off': cut_off,
                         'status': records[-1].get('status') if records[-1]['event'] == 'end' else None})
        for r in records:
            if r['event'] == 'epoch':
                epochs[r['record']['epoch']] = r['record']
    final = out_dir / CHECKPOINT_DIR / FINAL_NAME
    with open(final, 'rb') as f:
        final_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    ordered = [epochs[e] for e in sorted(epochs)]
    selected = selection['selected_epoch']
    result = {
        'schema_version': SCHEMA, 'status': 'run_complete', 'spec': run_spec, 'run_binding': binding,
        'selection': selection, 'selected_epoch': selected, 'best_validation_NDCG@10': selection['best_score'],
        'selected_validation': epochs[selected].get('validation') if selected in epochs else None,
        'epochs': ordered, 'epochs_run': len(ordered), 'attempts': attempts,
        'final_checkpoint': {'name': FINAL_NAME, 'sha256': final_sha, 'bytes': final.stat().st_size,
                             'holds_best_model': True},
        'environment': environment, 'determinism': determinism,
        'test_execution': 'not_run_test_split_not_loaded', 'deployment_decision': 'NOT_AUTHORIZED',
    }
    with open(out_dir / RESULT_NAME, 'x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    return validate_result(result, out_dir)


RESULT_KEYS = {'schema_version', 'status', 'spec', 'run_binding', 'selection', 'selected_epoch',
               'best_validation_NDCG@10', 'selected_validation', 'epochs', 'epochs_run', 'attempts',
               'final_checkpoint', 'environment', 'determinism', 'test_execution', 'deployment_decision'}


def validate_result(result, out_dir):
    """Refuse a result that differs from its journals, its checkpoint or the selection rule."""
    out_dir = Path(out_dir)
    if type(result) is not dict or set(result) != RESULT_KEYS or result['schema_version'] != SCHEMA:
        raise ValueError('result keys or schema differ from a training-run result')
    run_spec = result['spec']
    again = spec(run_spec['method'], run_spec['condition'], run_spec['learning_rate'], run_spec['seed'],
                 run_spec['stage'], run_spec['dataset_manifest_sha256'])
    if again != run_spec or result['run_binding']['run_spec_sha256'] != _digest(run_spec):
        raise ValueError('result spec or binding is not a valid run spec')
    epochs = {}
    attempts = []
    for name, (records, tail, cut_off) in _attempts(out_dir):
        if records[0]['event'] != 'start' or records[0]['spec'] != run_spec or records[0]['run_binding'] != result['run_binding']:
            raise ValueError(f'{name} belongs to another run')
        attempts.append({'journal': name, 'tail_sha256': tail, 'cut_off': cut_off,
                         'status': records[-1].get('status') if records[-1]['event'] == 'end' else None})
        for r in records:
            if r['event'] == 'epoch':
                epochs[r['record']['epoch']] = r['record']
    if attempts != result['attempts'] or attempts[-1]['status'] != 'run_complete':
        raise ValueError('result attempts differ from the journals')
    if result['epochs'] != [epochs[e] for e in sorted(epochs)] or result['epochs_run'] != len(epochs):
        raise ValueError('result epochs differ from the journals')
    if sorted(epochs) != list(range(1, len(epochs) + 1)):
        raise ValueError('epochs are not consecutive from 1')
    history = [{'epoch': e['epoch'], 'score': e['score']} for e in result['epochs']]
    selection = result['selection']
    if selection['history'] != history:
        raise ValueError('selection history differs from the epoch scores')
    best = max(history, key=lambda h: (h['score'], -h['epoch']))
    if (selection['best_score'], selection['selected_epoch']) != (best['score'], best['epoch']) \
            or result['selected_epoch'] != best['epoch'] or result['best_validation_NDCG@10'] != best['score']:
        raise ValueError('selected epoch does not follow the earliest-best rule')
    last = history[-1]['epoch']
    stopped_by_patience = last - best['epoch'] == 10
    if selection['status'] != 'stopped' or not (stopped_by_patience or last == 100):
        raise ValueError('run ended before patience or the 100-epoch maximum')
    if result['selected_validation'] != epochs[best['epoch']].get('validation'):
        raise ValueError('selected validation differs from the selected epoch')
    final = out_dir / CHECKPOINT_DIR / FINAL_NAME
    with open(final, 'rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != result['final_checkpoint']['sha256']:
            raise ValueError('final checkpoint differs from the result')
    if result['test_execution'] != 'not_run_test_split_not_loaded':
        raise ValueError('a training run never touches the test split')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    for name in ('--device', '--accelerator', '--data', '--manifest-sha256', '--out-dir', '--method',
                 '--condition', '--stage'):
        ap.add_argument(name, required=True)
    ap.add_argument('--learning-rate', type=float, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--runner-seconds', type=float, default=6600.0)
    args = ap.parse_args(argv)
    from scripts.gpu_runtime import configure_determinism, require_device
    device = require_device(args.device, args.accelerator)
    determinism = configure_determinism() if device.type == 'cuda' else None
    result = run(args.data, args.manifest_sha256, device, args.out_dir, method=args.method,
                 condition=args.condition, learning_rate=args.learning_rate, seed=args.seed,
                 stage=args.stage, determinism=determinism, runner_seconds=args.runner_seconds)
    print(json.dumps({'status': result['status'], 'stop_reason': result.get('stop_reason')}))
    return 0 if result['status'] == 'run_complete' else 1


if __name__ == '__main__':
    sys.exit(main())
