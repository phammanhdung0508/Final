"""GPU-READY parity run: accepted CPU models versus the EX01 path on a chosen device.

Run inside one session, so both sides share the same libraries (phase-2 runs 1 to 3).
Data are owned synthetic fixtures generated here; no Amazon or held-out data.
Tolerances are read from configs/gpu-ready-phase2-run-plan.json and never adjusted
after results. The result is evidence for review; it never marks a gate as passed.

Usage: CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.gpu_ready_parity \
           --device cuda:0 --accelerator gpu --plan configs/gpu-ready-phase2-run-plan.json \
           --out <fresh result path>
"""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from scripts.differentiable_models import (
    DifferentiableHeteroGNN, DifferentiableLightGCN, build_hetero_graph,
)
from scripts.gpu_runtime import (
    PROFILE_ID, configure_determinism, environment_fingerprint, require_determinism,
    require_device,
)
from scripts.checkpoint_identity import build_canonical_identity
from scripts.checkpoint_storage_cuda import publish_cuda_checkpoint, retrieve_cuda_checkpoint
from scripts.gpu_runtime import RuntimeRefusal
from scripts.learned_models import LightGCNGraph
from scripts.training_data import TrainingData
from scripts.training_harness import D2_KEYS, PrivateRNG, owned_digest
from scripts.recommendation_core import Evaluator, NegativeSampler
from scripts.recommendation_evaluation import TrainingContext
from scripts.training_harness import ADAM_OPTIONS, READINESS_GATES
from scripts.vector_trainer import VectorTrainer
from scripts.vectorized_evaluation import evaluate_model
from scripts.vectorized_models import VectorHeteroGNN, VectorLightGCN

RESULT_SCHEMA = 'amazon4u-H-result-cuda-v1'
IDENTITY_FIELDS = ('method', 'category', 'feedback_condition', 'seed', 'training_records',
                   'positive_history', 'all_rated_history', 'warm_items', 'graph_nodes',
                   'graph_edges', 'product_metadata', 'validation_inputs', 'resolved_settings')
ROOT = Path(__file__).resolve().parents[1]


def bundle_digest(root=ROOT):
    """SHA-256 over the path and bytes of every scripts/ and tests/ Python file."""
    h = hashlib.sha256()
    for path in sorted([*root.glob('scripts/*.py'), *root.glob('tests/*.py')]):
        rel = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        h.update(len(rel).to_bytes(8, 'big') + rel + len(data).to_bytes(8, 'big') + data)
    return h.hexdigest()


def plan_digest(plan):
    """SHA-256 of the run plan as canonical JSON (sorted keys)."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


# Same five gates as the frozen v1 list, except that the first names how GPU-READY is
# closed: by review of attached evidence, never by the run itself.
GPU_READINESS_GATES = [
    'GPU-READY acceptance by independent review of GR01 to GR10 evidence',
    *READINESS_GATES[1:],
]
# Adds the same u . b to every item score of a user, so BPR cancels it and its exact
# gradient is zero; Adam moves it by rounding noise. It is checked for that invariance.
RANKING_INVARIANT = {'Heterogeneous KG+GNN': 'self_biases_l1.Product'}
METHODS = ('LightGCN', 'Heterogeneous KG+GNN')
CONDITIONS = ('all_ratings', 'rating_ge_4')


def synthetic(seed, condition='rating_ge_4', n_users=24, n_items=30):
    """Owned synthetic fixture: interactions, category paths, brands, validation targets."""
    rng = random.Random(seed)
    users = [f'u{i:02d}' for i in range(n_users)]
    items = [f'p{i:02d}' for i in range(n_items)]
    records = []
    for u in users:
        for p in rng.sample(items, rng.randint(1, 8)):
            records.append((u, p, float(rng.choice([1, 2, 3, 4, 5, 5])), 0))
    context = TrainingContext(records, condition=condition)
    warm = context.sorted_warm_items
    roots = ['c:["Music"]', 'c:["Gear"]']
    leaves = ['c:["Music","Guitars"]', 'c:["Music","Drums"]', 'c:["Gear","Cables"]']
    categories = {p: rng.sample(roots + leaves, rng.randint(0, 2)) for p in warm}
    brands = {p: f'b:Brand{rng.randint(0, 4)}' for p in warm if rng.random() < 0.7}
    targets = []
    for u in users + ['unknown_user']:
        targets.append((u, rng.choice(items + ['cold_item']), float(rng.choice([2, 4, 5]))))
    return context, categories, brands, targets


def batch(context, seed, size=12):
    sampler = NegativeSampler(context)
    pairs = sorted((u, p) for u in context.users for p in context.positive_history.get(u, ()))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    triples, _ = sampler.prepare_batch(pairs[:size], rng)
    return triples


def as_index(triples, users, items, device=torch.device('cpu')):
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {p: i for i, p in enumerate(items)}
    return tuple(torch.tensor([m[t[k]] for t in triples], device=device)
                 for k, m in ((0, u_map), (1, i_map), (2, i_map)))


def model_pair(method, context, categories, brands, seed, device):
    """(accepted CPU model, vectorized model on `device`) with identical initial state."""
    if method == 'LightGCN':
        graph = LightGCNGraph(context)
        gen = torch.Generator().manual_seed(seed)
        u0 = torch.normal(0.0, 0.1, size=(len(graph.users), 64), generator=gen)
        i0 = torch.normal(0.0, 0.1, size=(len(graph.warm_items), 64), generator=gen)
        return DifferentiableLightGCN(graph, u0, i0), VectorLightGCN(graph, u0, i0, device)
    graph = build_hetero_graph(context, categories, brands)
    return DifferentiableHeteroGNN(graph, init_seed=seed), VectorHeteroGNN(graph, seed, device)


def _items(model):
    return model.graph.warm_items if hasattr(model.graph, 'warm_items') else model.graph.products


def compare(actual, expected, tol):
    """Largest differences and the elementwise |a-b| <= max_abs + rel*|b| verdict."""
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    diff = (a - b).abs()
    allowed = tol['max_abs'] + tol.get('rel', 0.0) * b.abs()
    return {'max_abs_diff': float(diff.max()) if diff.numel() else 0.0,
            'passed': bool((diff <= allowed).all())}


def _ref_representations(method, ref):
    if method == 'LightGCN':
        return ref.forward()
    h = ref.forward(None)
    return h['User'], h['Product']


def _adam_steps(model, step, n=5):
    opt = torch.optim.Adam(model.parameters(), **copy.deepcopy(ADAM_OPTIONS))
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        step()['total_loss'].backward()
        opt.step()
    return {k: v.detach().cpu() for k, v in model.named_parameters()}


def parity_case(method, seed, condition, device, tol):
    context, cats, brands, targets = synthetic(seed, condition)
    ref, vec = model_pair(method, context, cats, brands, seed, device)
    out = {'method': method, 'seed': seed, 'condition': condition}

    ref_u, ref_i = _ref_representations(method, ref)
    vec_u, vec_i = vec.representations()
    out['forward_scores'] = compare(vec_u @ vec_i.T, ref_u @ ref_i.T, tol['scores'])

    triples = batch(context, seed)
    users, pos, neg = map(list, zip(*triples))
    idx = as_index(triples, ref.graph.users, _items(ref), device)
    ref_loss = ref.compute_batch_loss(users, pos, neg)
    vec_loss = vec.compute_batch_loss(*idx)
    out['loss'] = {k: compare(vec_loss[k], ref_loss[k], {'max_abs': 0.0, 'rel': tol['loss']['rel']})
                   for k in ref_loss}
    ref_loss['total_loss'].backward()
    vec_loss['total_loss'].backward()
    ref_params = dict(ref.named_parameters())
    grads = {}
    for name, p in vec.named_parameters():
        r = ref_params[name].grad
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grads[name] = compare(g, r if r is not None else torch.zeros_like(g), tol['gradients'])
    out['gradients'] = {'passed': all(v['passed'] for v in grads.values()),
                        'max_abs_diff': max(v['max_abs_diff'] for v in grads.values()),
                        'failed': sorted(k for k, v in grads.items() if not v['passed'])}

    invariant = RANKING_INVARIANT.get(method)
    if invariant:
        out['ranking_invariant_parameter'] = {
            'name': invariant,
            'reference_gradient_max_abs': float(ref_params[invariant].grad.abs().max())}
    ref_after = _adam_steps(ref, lambda: ref.compute_batch_loss(users, pos, neg))
    vec_after = _adam_steps(vec, lambda: vec.compute_batch_loss(*idx))
    steps = {k: compare(v, ref_after[k], tol['parameters_after_5_adam_steps'])
             for k, v in vec_after.items() if k != invariant}
    out['parameters_after_5_adam_steps'] = {
        'passed': all(v['passed'] for v in steps.values()),
        'max_abs_diff': max(v['max_abs_diff'] for v in steps.values()),
        'failed': sorted(k for k, v in steps.items() if not v['passed'])}

    with torch.no_grad():
        ref.load_state_dict({**ref.state_dict(), **ref_after})
        expected = Evaluator(context).evaluate(copy.deepcopy(targets), ref, split='validation')
    actual = evaluate_model(vec, targets)
    out['top10_fixture_ranking'] = {'passed': [r['top_10'] for r in actual['per_user_results']]
                                    == [r['top_10'] for r in expected['per_user_results']]}
    out['passed'] = all(x['passed'] for x in (
        out['forward_scores'], out['gradients'], out['parameters_after_5_adam_steps'],
        out['top10_fixture_ranking'], *out['loss'].values()))
    return out


def checkpoint_fixture(method, device, fingerprint, determinism):
    """A trainer on the accepted D2 fixture, with every expectation D2 and D3 require.

    The fixtures live in the accepted D2/D3 test modules and are imported unchanged. For a
    CUDA run, the D1 identity records `execution_device = cuda:0`.
    """
    tests_dir = str(Path(__file__).resolve().parents[1] / 'tests')
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_checkpoint_storage import _fixture
    _, expectations = _fixture(hetero=method != 'LightGCN', initial=True)
    expectations = copy.deepcopy(expectations)
    identity = expectations['expected_identity']
    if str(device) != identity['resolved_settings']['execution']['execution_device']:
        fields = {k: copy.deepcopy(identity[k]) for k in IDENTITY_FIELDS}
        fields['resolved_settings']['execution']['execution_device'] = str(device)
        identity = build_canonical_identity(**fields).to_dict()
        expectations['expected_identity'] = identity
    data = TrainingData(copy.deepcopy(identity), expected_identity=identity,
                        resource_policy=expectations['resource_policy'])
    seed = data.identity.seed
    if method == 'LightGCN':
        graph = LightGCNGraph(data.context)
        gen = torch.Generator().manual_seed(seed)
        model = VectorLightGCN(graph, torch.normal(0.0, 0.1, (len(graph.users), 64), generator=gen),
                               torch.normal(0.0, 0.1, (len(graph.warm_items), 64), generator=gen),
                               device)
        orders = {'User': graph.users, 'Product': graph.warm_items}
    else:
        graph = build_hetero_graph(data.context, data.product_categories, data.product_brands)
        model = VectorHeteroGNN(graph, seed, device)
        orders = graph.node_ids
    if {k: list(v) for k, v in orders.items()} != expectations['expected_model']['row_order']:
        raise RuntimeRefusal('fixture row order differs from the model row order')
    training = data.identity.resolved_settings['training']
    trainer = VectorTrainer(model, data.validation_records,
                            rng=PrivateRNG(data.identity, expectations['expected_runtime']),
                            batch_size=training['batch_size'],
                            validation_frequency=training['validation_frequency'],
                            fingerprint=fingerprint, determinism=determinism)
    return trainer, expectations, identity


def restart_case(method, device, fingerprint, determinism, run_binding, epochs=4, split=2):
    """Uninterrupted versus publish, retrieve and restore at an epoch boundary; exact."""
    def fresh():
        return checkpoint_fixture(method, device, fingerprint, determinism)
    straight, expectations, identity = fresh()
    d2 = {k: expectations[k] for k in D2_KEYS}
    expected_device = {'fingerprint': fingerprint, 'determinism': determinism,
                       'run_binding': run_binding}
    logs = [straight.train_epoch() for _ in range(epochs)]
    first, _, _ = fresh()
    part = [first.train_epoch() for _ in range(split)]
    with tempfile.TemporaryDirectory() as tmp:
        os.chmod(tmp, 0o700)
        root = os.open(tmp, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            name = publish_cuda_checkpoint(
                root, 'restart.a4ucp', first.cuda_payload(expectations, identity, run_binding),
                expected_device=expected_device, **d2, io_policy=expectations['io_policy'])
            owned = retrieve_cuda_checkpoint(root, name, expected_device=expected_device, **d2,
                                             io_policy=expectations['io_policy'])
        finally:
            os.close(root)
    resumed, _, _ = fresh()
    resumed.restore_cuda(owned)
    part += [resumed.train_epoch() for _ in range(epochs - split)]
    on_device = all(p.device == torch.device(device) for p in resumed.model.parameters())
    a = straight.cuda_payload(expectations, identity, run_binding)
    b = resumed.cuda_payload(expectations, identity, run_binding)
    result = {'method': method, 'epochs': epochs, 'restored_after_epoch': split,
              'path': 'cuda-v1 payload, D2 core validation, D3 publish and retrieve',
              'logs_identical': logs == part, 'state_identical': states_equal(a, b),
              'validation_identical': straight.validation_report == resumed.validation_report,
              'parameters_on_device_after_restore': on_device}
    result['passed'] = all(result[k] for k in ('logs_identical', 'state_identical',
                                               'validation_identical',
                                               'parameters_on_device_after_restore'))
    return result


def _child(*args, env=None):
    """Run scripts.gpu_ready_child in a new process; its JSON output, or the failure."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'out.json')
        proc = subprocess.run([sys.executable, '-m', 'scripts.gpu_ready_child', *args, '--out', out],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0 or not os.path.exists(out):
            return {'child_error': f'exit {proc.returncode}: {proc.stderr[-1000:]}'}
        with open(out) as f:
            return json.load(f)


def fresh_process_restart_case(method, device_name, determinism, run_binding, epochs=4, split=2):
    """Restart across processes: the resuming process shares only the checkpoint bytes.

    `straight`, `first` and `resume` each run in a new Python process. `first` publishes a
    cuda-v1 checkpoint through D3 and exits; `resume` retrieves it, restores and finishes.
    """
    common = ['--device', device_name, '--method', method, '--run-binding', json.dumps(run_binding),
              '--determinism', 'off' if determinism is None else 'on',
              '--epochs', str(epochs), '--split', str(split)]
    with tempfile.TemporaryDirectory() as straight_dir, tempfile.TemporaryDirectory() as shared:
        for d in (straight_dir, shared):
            os.chmod(d, 0o700)
        straight = _child('--role', 'straight', '--dir', straight_dir, *common)
        first = _child('--role', 'first', '--dir', shared, *common)
        resumed = (_child('--role', 'resume', '--dir', shared, *common)
                   if 'child_error' not in first else {'child_error': 'first process failed'})
    result = {'method': method, 'epochs': epochs, 'restored_after_epoch': split,
              'mode': 'fresh_process',
              'path': 'cuda-v1 payload, D2 core validation, D3 publish and retrieve across processes'}
    errors = {k: v['child_error'] for k, v in (('straight', straight), ('first', first),
                                               ('resume', resumed)) if 'child_error' in v}
    if errors:
        return {**result, 'child_errors': errors, 'passed': False}
    result.update(
        distinct_processes=len({straight['pid'], first['pid'], resumed['pid']}) == 3,
        logs_identical=straight['logs'] == first['logs'] + resumed['logs'],
        state_identical=straight['state_digest'] == resumed['state_digest'],
        validation_identical=straight['validation_report'] == resumed['validation_report'],
        parameters_on_device_after_restore=resumed['on_device'])
    result['passed'] = all(result[k] for k in ('distinct_processes', 'logs_identical',
                                               'state_identical', 'validation_identical',
                                               'parameters_on_device_after_restore'))
    return result


def determinism_probes(device_name):
    """GR04 negative tests, each in its own process: the policy must make a documented
    nondeterministic operation raise, and the control must show the operation itself works.
    For cuBLAS the control keeps the policy on and sets CUBLAS_WORKSPACE_CONFIG."""
    from scripts.gpu_ready_child import PROBES
    on_cuda = device_name.startswith('cuda')
    without_config = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
    with_config = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    results = []
    for name, scope in PROBES.items():
        if scope == 'cuda' and not on_cuda:
            continue
        base = ['--device', device_name, '--probe', name]
        if name == 'cublas_matmul_without_workspace_config':
            policy = _child(*base, '--policy', 'on', env=without_config)
            control = _child(*base, '--policy', 'on', env=with_config)
        else:
            policy = _child(*base, '--policy', 'on')
            control = _child(*base, '--policy', 'off')
        caught = (policy.get('raised') is True and policy.get('type') == 'RuntimeError'
                  and 'deterministic' in policy.get('message', '').lower())
        results.append({'probe': name, 'policy': policy, 'control': control,
                        'expected_exception_caught': caught,
                        'control_succeeded': control.get('raised') is False,
                        'passed': caught and control.get('raised') is False})
    return results


def states_equal(a, b):
    """Exact equality via H-REF's semantic digest (dtype, layout, shape and bytes)."""
    return owned_digest(a) == owned_digest(b)


def run(device_name, accelerator, plan, *, determinism=None, run_binding=None):
    device = require_device(device_name, accelerator)
    fingerprint = environment_fingerprint(device)
    run_binding = run_binding or {'bundle_sha256': bundle_digest(),
                                  'run_spec_sha256': plan_digest(plan)}
    tol = plan['tolerances_float32']
    seeds = plan['fixtures']['seeds']
    parity = [parity_case(m, s, c, device, tol) for m in METHODS for s in seeds for c in CONDITIONS]
    restart = [restart_case(m, device, fingerprint, determinism, run_binding) for m in METHODS]
    restart += [fresh_process_restart_case(m, device_name, determinism, run_binding) for m in METHODS]
    probes = determinism_probes(device_name)
    if determinism is not None:
        require_determinism()
    passed = all(x['passed'] for x in parity + restart + probes)
    return {
        'schema_version': RESULT_SCHEMA, 'profile_id': PROFILE_ID,
        'status': 'parity_complete' if passed else 'parity_failed',
        'acceptance_scope': 'GPU-READY_evidence_only_requires_independent_review',
        'deployment_decision': 'NOT_AUTHORIZED',
        'device': device_name, 'accelerator': accelerator,
        'environment': fingerprint, 'determinism': determinism,
        'tolerances_float32': copy.deepcopy(tol), 'all_checks_passed': passed,
        'parity': parity, 'restart': restart, 'determinism_probes': probes,
        'run_binding': run_binding,
        'remaining_readiness_gates': list(GPU_READINESS_GATES),
        'data': 'owned_synthetic_fixtures_only', 'test_execution': 'not_authorized',
    }


RESULT_KEYS = {
    'schema_version', 'profile_id', 'status', 'acceptance_scope', 'deployment_decision',
    'device', 'accelerator', 'environment', 'determinism', 'tolerances_float32',
    'all_checks_passed', 'parity', 'restart', 'determinism_probes', 'run_binding',
    'remaining_readiness_gates', 'data', 'test_execution',
}


def validate_result(result):
    """Collection-side check of the cuda-v1 result. A v1 result can never pass."""
    if type(result) is not dict or set(result) != RESULT_KEYS:
        raise ValueError('result keys are not exactly the cuda-v1 schema')
    if result['schema_version'] != RESULT_SCHEMA or result['profile_id'] != PROFILE_ID:
        raise ValueError('not a GPU-READY cuda-v1 result')
    if result['remaining_readiness_gates'] != GPU_READINESS_GATES:
        raise ValueError('remaining readiness gates differ from the GPU-READY list')
    if result['deployment_decision'] != 'NOT_AUTHORIZED':
        raise ValueError('deployment decision must stay NOT_AUTHORIZED')
    passed = all(x['passed'] for x in
                 result['parity'] + result['restart'] + result['determinism_probes'])
    if result['all_checks_passed'] is not passed:
        raise ValueError('summary verdict disagrees with the individual checks')
    if result['status'] != ('parity_complete' if passed else 'parity_failed'):
        raise ValueError('status disagrees with the individual checks')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--accelerator', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f'refusing to overwrite {args.out}')
    with open(args.plan) as f:
        plan = json.load(f)
    determinism = configure_determinism()
    result = validate_result(run(args.device, args.accelerator, plan, determinism=determinism))
    with open(args.out, 'x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps({'status': result['status'], 'all_checks_passed': result['all_checks_passed']}))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
