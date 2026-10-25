"""Profiling P3: the Musical_Instruments profiling runner.

One process runs the agreed plan (docs/musical-instruments-profiling-plan.md):

1. load and hash-check the profiling dataset (`profiling_data.ProfilingData`);
2. preflight (5.1): for each case, build, sample one epoch, 5 warm-up and 50 timed
   steps, one full validation pass. Project the whole run with the mean, median and
   p90 step times; the gate is the larger of the mean and p90 projections. After each
   case a lower-bound projection stops the run early;
3. the six cases, 3 epochs each, every epoch timed on its own (5.2): sampling,
   training steps, validation inference, ranking and metrics, and peak memory.

There is no checkpoint step (user, 2026-10-25). Weights are never written; validation
metric values are reduced to finiteness checks, counts and a SHA-256.

Every finished measurement is a line in a hash-chained, fsynced `progress.jsonl`, also
printed to stdout. Deadlines are checked between training steps and between validation
chunks. `result.json` is written once at the end, with whatever status ended the run.

Usage: CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.profiling_runner \\
           --device cuda:0 --accelerator gpu --data <dataset dir> \\
           --manifest-sha256 <pinned manifest SHA-256> --out-dir <directory>
"""

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from scripts.differentiable_models import build_hetero_graph
from scripts.gpu_ready_parity import bundle_digest
from scripts.gpu_runtime import (
    configure_determinism, environment_fingerprint, global_rng_states, require_device,
    require_rng_unchanged,
)
from scripts.learned_models import LightGCNGraph
from scripts.profiling_data import ProfilingData
from scripts.vector_trainer import VectorTrainer, fixture_rng
from scripts.vectorized_evaluation import METRIC_KEYS, evaluate_validation
from scripts.vectorized_models import VectorHeteroGNN, VectorLightGCN

SCHEMA = 'amazon4u-profile-result-v1'
JOURNAL_NAME = 'progress.jsonl'
RESULT_NAME = 'result.json'
JOURNAL_PREFIX = 'A4U-PROFILE '
SEED = 7
EPOCHS = 3
WARM_UP_STEPS = 5
TIMED_STEPS = 50
CHUNK_USERS = 1024
HETERO = 'Heterogeneous KG+GNN'
CASES = (
    ('LightGCN', 2048, 'all_ratings'),
    ('LightGCN', 4096, 'all_ratings'),
    ('LightGCN', 2048, 'rating_ge_4'),
    ('LightGCN', 4096, 'rating_ge_4'),
    (HETERO, 1024, 'all_ratings'),
    (HETERO, 1024, 'rating_ge_4'),
)
GIB = 1 << 30
GENESIS = '0' * 64
STATUSES = ('profile_complete', 'preflight_exceeded', 'preflight_limit_exceeded',
            'runner_timeout', 'failed')
REMAINING_GATES = [
    'profiling report reviewed by the user',
    'checkpoint speed-up task approved, done and re-measured',
    'frozen execution settings and tuning budget',
    'explicit comparative-training authorization',
]


@dataclass(frozen=True)
class Limits:
    preflight_seconds: float = 900.0
    preflight_gpu_bytes: int = 12 * GIB
    preflight_host_bytes: int = 12 * GIB
    runner_seconds: float = 6600.0
    threshold_seconds: float = 0.8 * 6600.0


AGREED_LIMITS = Limits()


class Stop(Exception):
    """A limit ended the run; `status` is the result status."""

    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


def peak_host_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def gpu_peaks(device):
    if device.type != 'cuda':
        return {'allocated': 0, 'reserved': 0}
    return {'allocated': torch.cuda.max_memory_allocated(device),
            'reserved': torch.cuda.max_memory_reserved(device)}


class Budget:
    """Runner deadline always; the preflight's own time and memory limits while it runs."""

    def __init__(self, limits, device, clock=time.monotonic):
        self.limits, self.device, self.clock = limits, device, clock
        self.start = clock()
        self.preflight_start = None

    def elapsed(self):
        return self.clock() - self.start

    def check(self):
        if self.elapsed() > self.limits.runner_seconds:
            raise Stop('runner_timeout', f'runner deadline {self.limits.runner_seconds} s passed')
        if self.preflight_start is None:
            return
        if self.clock() - self.preflight_start > self.limits.preflight_seconds:
            raise Stop('preflight_limit_exceeded',
                       f'preflight time over {self.limits.preflight_seconds} s')
        if gpu_peaks(self.device)['allocated'] > self.limits.preflight_gpu_bytes:
            raise Stop('preflight_limit_exceeded', 'preflight GPU memory over its limit')
        if peak_host_bytes() > self.limits.preflight_host_bytes:
            raise Stop('preflight_limit_exceeded', 'preflight host memory over its limit')


class Journal:
    """Append-only, hash-chained progress log; each line is fsynced and echoed to stdout."""

    def __init__(self, path, stream=sys.stdout):
        self.file = open(path, 'x', encoding='utf-8')
        self.stream = stream
        self.seq, self.prev = 0, GENESIS

    def write(self, event, **fields):
        line = json.dumps({'seq': self.seq, 'prev_sha256': self.prev, 'event': event, **fields},
                          sort_keys=True, allow_nan=False)
        self.file.write(line + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        print(JOURNAL_PREFIX + line, file=self.stream, flush=True)
        self.seq += 1
        self.prev = hashlib.sha256(line.encode()).hexdigest()
        return self.prev

    def close(self):
        self.file.close()


class JournalError(ValueError):
    """The progress journal's chain is broken."""


def read_journal(path):
    """Return (records, tail_sha256, cut_off). A cut-off last line is reported, not repaired."""
    data = Path(path).read_bytes().decode('utf-8')
    lines = data.split('\n')
    cut_off = lines[-1] != ''
    complete = lines[:-1]
    records, prev = [], GENESIS
    for seq, line in enumerate(complete):
        record = json.loads(line)
        if record.get('seq') != seq or record.get('prev_sha256') != prev:
            raise JournalError(f'journal chain broken at line {seq}')
        records.append(record)
        prev = hashlib.sha256(line.encode()).hexdigest()
    return records, prev, cut_off


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def step_stats(times):
    ordered = sorted(times)
    return {'count': len(times), 'mean': statistics.fmean(times),
            'median': statistics.median(times),
            'p90': ordered[math.ceil(0.9 * len(ordered)) - 1], 'max': ordered[-1]}


def build_case(data, case, targets, device, fingerprint, determinism):
    method, batch_size, condition = case
    context = data.context(condition)
    if method == 'LightGCN':
        graph = LightGCNGraph(context)
        gen = torch.Generator().manual_seed(SEED)
        u0 = torch.normal(0.0, 0.1, size=(len(graph.users), 64), generator=gen)
        i0 = torch.normal(0.0, 0.1, size=(len(graph.warm_items), 64), generator=gen)
        model = VectorLightGCN(graph, u0, i0, device)
    else:
        graph = build_hetero_graph(context, data.product_categories, data.product_brands)
        model = VectorHeteroGNN(graph, SEED, device)
    return VectorTrainer(model, targets, rng=fixture_rng(SEED), batch_size=batch_size,
                         validation_frequency=1, fingerprint=fingerprint,
                         determinism=determinism, chunk_users=CHUNK_USERS)


def _report_checks(report):
    """What profiling keeps of a validation report: checks, counts and a digest."""
    values = [v for v in report['overall_metrics'].values()]
    values += [r['metrics'][k] for r in report['per_user_results'] for k in r['metrics']]
    return {
        'metric_keys': sorted(report['overall_metrics']) == sorted(METRIC_KEYS),
        'all_finite': all(v is not None and math.isfinite(v) for v in values),
        'audit_counts': {k: v for k, v in report['audit'].items() if isinstance(v, int)},
        'candidate_shortages': {str(k): v for k, v in report['candidate_shortages'].items()},
        'subgroup_counts': {k: g['count'] for k, g in report['subgroups'].items()},
        'report_sha256': hashlib.sha256(
            json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }


def validate_once(trainer, budget, device):
    """One full validation pass: inference, ranking and metrics timed apart."""
    model = trainer.model
    graph = model.graph
    items = graph.products if hasattr(graph, 'products') else graph.warm_items
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        user_table, item_table = model.representations()
    _sync(device)
    inference = time.perf_counter() - start
    budget.check()
    stages = {'ranking': 0.0, 'metrics': 0.0}
    mark = [time.perf_counter()]

    def tick(stage):
        _sync(device)
        now = time.perf_counter()
        stages[stage] += now - mark[0]
        mark[0] = now
        budget.check()

    report = evaluate_validation(graph.context, graph.users, items, user_table, item_table,
                                 trainer.targets, chunk_users=CHUNK_USERS, tick=tick)
    total = time.perf_counter() - start
    del user_table, item_table
    return {'inference': inference, 'ranking': stages['ranking'],
            'metrics': total - inference - stages['ranking'], 'total': total,
            'checks': _report_checks(report)}


def _timed(fn, device):
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - start


def _batches(sample, batch_size):
    users, positives, negatives = sample[:3]
    for start in range(0, len(users), batch_size):
        sl = slice(start, start + batch_size)
        yield users[sl], positives[sl], negatives[sl]


def case_cost(case_record, step_time):
    return case_record['build'] + EPOCHS * (
        case_record['sampling'] + case_record['steps_per_epoch'] * step_time
        + case_record['validation'])


def gate_step_time(stats):
    return max(stats['mean'], stats['p90'])


def projections(measured, load, elapsed):
    """Whole-run projections per estimator, and the gate (larger of mean and p90)."""
    out = {}
    for name in ('mean', 'median', 'p90'):
        out[name] = elapsed + load + sum(case_cost(m, m['step_stats'][name]) for m in measured)
    out['gate'] = max(out['mean'], out['p90'])
    return out


def lower_bound(cases, measured, load, elapsed):
    """Measured cases at the gate estimate; each unmeasured case at the cheapest measured
    case of the same model, or zero if none is measured yet."""
    costs = [case_cost(m, gate_step_time(m['step_stats'])) for m in measured]
    by_method = {}
    for m, cost in zip(measured, costs):
        by_method[m['method']] = min(cost, by_method.get(m['method'], math.inf))
    rest = sum(by_method.get(method, 0.0) for method, *_ in cases[len(measured):])
    return elapsed + load + sum(costs) + rest


def preflight_case(data, case, targets, device, fingerprint, determinism, budget):
    trainer, build = _timed(
        lambda: build_case(data, case, targets, device, fingerprint, determinism), device)
    budget.check()
    sample, sampling = _timed(trainer.sample_epoch, device)
    budget.check()
    steps_per_epoch = math.ceil(len(sample[0]) / case[1])
    times = []
    for i, batch in enumerate(_batches(sample, case[1])):
        if i == WARM_UP_STEPS + TIMED_STEPS:
            break
        _, seconds = _timed(lambda: trainer.step(*batch), device)
        if i >= WARM_UP_STEPS:
            times.append(seconds)
        budget.check()
    if not times:
        times = [seconds]
    validation = validate_once(trainer, budget, device)
    return {'method': case[0], 'batch_size': case[1], 'condition': case[2],
            'build': build, 'sampling': sampling, 'steps_per_epoch': steps_per_epoch,
            'empty_pool_skips': sample[3], 'step_stats': step_stats(times),
            'warm_up_steps': min(WARM_UP_STEPS, steps_per_epoch),
            'validation': validation['total'], 'validation_checks': validation['checks'],
            'gpu_peak': gpu_peaks(device), 'host_peak_bytes': peak_host_bytes()}


def profile_epoch(trainer, case, epoch, device, budget):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    before = global_rng_states(device)
    sample, sampling = _timed(trainer.sample_epoch, device)
    budget.check()
    times = []
    for batch in _batches(sample, case[1]):
        _, seconds = _timed(lambda: trainer.step(*batch), device)
        times.append(seconds)
        budget.check()
    validation = validate_once(trainer, budget, device)
    require_rng_unchanged(before, global_rng_states(device), epoch)
    training = sum(times)
    return {'epoch': epoch, 'label': 'warm_up' if epoch == 1 else 'steady_state',
            'sampling': sampling, 'training_steps': training, 'steps': len(times),
            'step_stats': step_stats(times), 'empty_pool_skips': sample[3],
            'examples_per_second': len(sample[0]) / training if training > 0 else None,
            'validation': validation, 'gpu_peak': gpu_peaks(device),
            'host_peak_bytes': peak_host_bytes()}


def _release(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def run(data_dir, manifest_sha256, device, out_dir, *, limits=AGREED_LIMITS, cases=CASES,
        determinism=None, clock=time.monotonic, stream=sys.stdout):
    """Run the plan into `out_dir`; return the validated result.

    `out_dir` may already exist (on Kaggle it is `/kaggle/working`); the journal and the
    result are opened with `x`, so nothing is ever overwritten."""
    out_dir = Path(out_dir)
    out_dir.mkdir(mode=0o700, exist_ok=True)
    fingerprint = environment_fingerprint(device)
    run_binding = {'bundle_sha256': bundle_digest(), 'dataset_manifest_sha256': manifest_sha256}
    budget = Budget(limits, device, clock)
    journal = Journal(out_dir / JOURNAL_NAME, stream)
    journal.write('start', run_binding=run_binding, limits=asdict(limits),
                  cases=[list(c) for c in cases], seed=SEED, epochs=EPOCHS)
    state = {'load': None, 'preflight': [], 'projection': None, 'lower_bounds': [],
             'cases': [], 'stop_reason': None}
    status = 'profile_complete'
    try:
        start = clock()
        data = ProfilingData(data_dir, manifest_sha256)
        targets = data.validation_targets()
        state['load'] = clock() - start
        journal.write('load', seconds=state['load'])
        budget.check()

        budget.preflight_start = clock()
        for case in cases:
            record = preflight_case(data, case, targets, device, fingerprint, determinism, budget)
            _release(device)
            state['preflight'].append(record)
            bound = lower_bound(cases, state['preflight'], state['load'], budget.elapsed())
            state['lower_bounds'].append(bound)
            journal.write('preflight_case', record=record, lower_bound=bound)
            if bound > limits.threshold_seconds:
                raise Stop('preflight_exceeded',
                           f'lower-bound projection {bound:.1f} s over {limits.threshold_seconds} s')
        state['projection'] = projections(state['preflight'], state['load'],
                                          budget.elapsed())
        journal.write('projection', projection=state['projection'])
        if state['projection']['gate'] > limits.threshold_seconds:
            raise Stop('preflight_exceeded',
                       f"gate projection {state['projection']['gate']:.1f} s over "
                       f'{limits.threshold_seconds} s')
        budget.preflight_start = None

        for case in cases:
            trainer, build = _timed(
                lambda: build_case(data, case, targets, device, fingerprint, determinism), device)
            budget.check()
            record = {'method': case[0], 'batch_size': case[1], 'condition': case[2],
                      'build': build, 'epochs': []}
            state['cases'].append(record)
            journal.write('case_build', case=list(case), seconds=build)
            for epoch in range(1, EPOCHS + 1):
                epoch_record = profile_epoch(trainer, case, epoch, device, budget)
                record['epochs'].append(epoch_record)
                journal.write('epoch', case=list(case), record=epoch_record)
            del trainer
            _release(device)
    except Stop as stop:
        status, state['stop_reason'] = stop.status, stop.reason
    except Exception as exc:
        status, state['stop_reason'] = 'failed', f'{type(exc).__name__}: {str(exc)[:300]}'
    tail = journal.write('end', status=status, reason=state['stop_reason'])
    journal.close()
    result = {
        'schema_version': SCHEMA, 'status': status, 'stop_reason': state['stop_reason'],
        'category': 'Musical_Instruments', 'seed': SEED, 'epochs_per_case': EPOCHS,
        'cases_planned': [list(c) for c in cases], 'limits': asdict(limits),
        'load_seconds': state['load'], 'preflight': state['preflight'],
        'lower_bounds': state['lower_bounds'], 'projection': state['projection'],
        'cases': state['cases'], 'checkpoint_step': 'not_run_by_decision_2026-10-25',
        'environment': fingerprint, 'determinism': determinism, 'run_binding': run_binding,
        'journal_tail_sha256': tail, 'weights_kept': False, 'metric_values_reported': False,
        'test_execution': 'not_authorized', 'deployment_decision': 'NOT_AUTHORIZED',
        'remaining_gates': list(REMAINING_GATES),
    }
    with open(out_dir / RESULT_NAME, 'x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    return validate_result(result, out_dir / JOURNAL_NAME)


RESULT_KEYS = {
    'schema_version', 'status', 'stop_reason', 'category', 'seed', 'epochs_per_case',
    'cases_planned', 'limits', 'load_seconds', 'preflight', 'lower_bounds', 'projection',
    'cases', 'checkpoint_step', 'environment', 'determinism', 'run_binding',
    'journal_tail_sha256', 'weights_kept', 'metric_values_reported', 'test_execution',
    'deployment_decision', 'remaining_gates',
}


def _checks_pass(checks):
    return checks['metric_keys'] and checks['all_finite']


def validate_result(result, journal_path):
    """Collection-side check: the result must agree with its own journal and with itself."""
    if type(result) is not dict or set(result) != RESULT_KEYS:
        raise ValueError('result keys are not exactly the profile-result-v1 schema')
    if result['schema_version'] != SCHEMA or result['status'] not in STATUSES:
        raise ValueError('not a profile-result-v1 result')
    if (result['weights_kept'], result['metric_values_reported'], result['test_execution'],
            result['deployment_decision'], result['remaining_gates']) != (
            False, False, 'not_authorized', 'NOT_AUTHORIZED', REMAINING_GATES):
        raise ValueError('result claims an authorization or output it must not have')
    records, tail, cut_off = read_journal(journal_path)
    if cut_off or tail != result['journal_tail_sha256']:
        raise ValueError('result does not match the end of its progress journal')
    by_event = {}
    for r in records:
        by_event.setdefault(r['event'], []).append(r)
    end = by_event['end'][0]
    if (end['status'], end['reason']) != (result['status'], result['stop_reason']):
        raise ValueError('result status differs from the journal')
    if [r['record'] for r in by_event.get('preflight_case', [])] != result['preflight']:
        raise ValueError('preflight records differ from the journal')
    if [r['record'] for r in by_event.get('epoch', [])] != [
            e for c in result['cases'] for e in c['epochs']]:
        raise ValueError('epoch records differ from the journal')
    planned = len(result['cases_planned'])
    complete = (len(result['cases']) == planned
                and all(len(c['epochs']) == result['epochs_per_case'] for c in result['cases']))
    checks = [p['validation_checks'] for p in result['preflight']]
    checks += [e['validation']['checks'] for c in result['cases'] for e in c['epochs']]
    if result['status'] == 'profile_complete':
        if not complete or not all(_checks_pass(c) for c in checks):
            raise ValueError('profile_complete without every case, epoch and check')
        if result['projection']['gate'] > result['limits']['threshold_seconds']:
            raise ValueError('profile_complete although the gate projection is over threshold')
    elif result['status'] == 'preflight_exceeded':
        if result['cases']:
            raise ValueError('a case ran after the preflight stopped the run')
        over = (result['projection'] is not None
                and result['projection']['gate'] > result['limits']['threshold_seconds'])
        if not over and not (result['lower_bounds']
                             and result['lower_bounds'][-1] > result['limits']['threshold_seconds']):
            raise ValueError('preflight_exceeded without a projection over the threshold')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--accelerator', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    device = require_device(args.device, args.accelerator)
    determinism = configure_determinism() if device.type == 'cuda' else None
    result = run(args.data, args.manifest_sha256, device, args.out_dir, determinism=determinism)
    print(json.dumps({'status': result['status'], 'stop_reason': result['stop_reason']}))
    return 0 if result['status'] == 'profile_complete' else 1


if __name__ == '__main__':
    sys.exit(main())
