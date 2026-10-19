"""Amazon4U H-REF: bounded CPU reference callers of the accepted scientific APIs."""
import copy
import hashlib
import math
import random
import struct
import sys

import numpy as np
import torch

from scripts.checkpoint_payload import validate_checkpoint_payload
from scripts.checkpoint_storage import acquire_sources, publish_checkpoint, retrieve_checkpoint
from scripts.differentiable_models import (
    DifferentiableHeteroGNN, DifferentiableLightGCN, build_hetero_graph,
)
from scripts.learned_models import LightGCNGraph
from scripts.recommendation_core import (
    Evaluator, NegativeSampler, PopularityScorer, select_kg_only_mixture,
)
from scripts.training_data import HarnessError, TrainingData, checked_policy, preflight


D2_KEYS = ('expected_identity', 'expected_provenance', 'expected_model',
           'expected_runtime', 'resource_policy')
# External closure, not an extension of the accepted D1 allowlist. Include the
# registered project sources/configs, D2, D3 and all six H files. Supplied bytes
# are declarations; checking them does not inspect or authenticate deployed code.
BUNDLE_PATHS = (
    'scripts/checkpoint_identity.py',
    'scripts/learned_models.py',
    'scripts/recommendation_core.py',
    'scripts/recommendation_evaluation.py',
    'scripts/differentiable_models.py',
    'scripts/build_kg.py',
    'scripts/audit_kg_inputs.py',
    'scripts/audit_candidates.py',
    'scripts/validate_kg_splits.py',
    'scripts/visualize_kg.py',
    'scripts/inspect_data.py',
    'scripts/download_5core.py',
    'configs/compute-plan.json',
    'configs/dataset-scope.json',
    'configs/evaluation-metrics.json',
    'configs/kg-v1.json',
    'configs/model-suite.json',
    'configs/protocol-lifecycle.json',
    'configs/reproducibility.json',
    'configs/training-semantics.json',
    'scripts/checkpoint_payload.py',
    'scripts/checkpoint_storage.py',
    'scripts/training_harness.py',
    'scripts/training_data.py',
    'scripts/training_entrypoint.py',
    'tests/test_training_harness.py',
    'tests/test_training_data.py',
    'docs/training-harness.md',
)
ADAM_OPTIONS = {
    'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8,
    'weight_decay': 0.0, 'amsgrad': False, 'maximize': False,
    'foreach': None, 'capturable': False, 'differentiable': False,
    'fused': None, 'decoupled_weight_decay': False,
}
ENVELOPE_KEYS = {
    'schema_version', 'run_id', 'approval_id', 'profile_id', 'method',
    'identity_hash', 'device', 'accelerator', 'bundle_sha256',
    'initial_state_sha256', 'determinism_policy',
}
READINESS_GATES = [
    'GPU-READY separately authorized and verified',
    'O-OFFLINE transport and packaging separately reviewed',
    'Production LightGCN initialization decision',
    'Approved environment and full-data feasibility profiling',
    'Comparative execution/search freeze and final-test authorization',
]


def owned_digest(value):
    """Semantic SHA-256 of OWNED/preflighted values, not a D3 physical-file hash.

    This is not an untrusted-input decoder. Public payload entry points perform D2
    preflight/validation before calling it. Dictionary insertion order is irrelevant;
    tensor dtype/layout/shape and exact bytes are included.
    """
    h = hashlib.sha256()

    def blob(tag, data):
        h.update(tag)
        h.update(struct.pack('!Q', len(data)))
        h.update(data)

    def visit(v):
        if v is None:
            h.update(b'N')
        elif type(v) is bool:
            h.update(b'T' if v else b'F')
        elif type(v) is int:
            blob(b'I', str(v).encode('ascii'))
        elif type(v) is float:
            blob(b'R', struct.pack('!d', v))
        elif type(v) is str:
            blob(b'S', v.encode('utf-8'))
        elif type(v) is bytes:
            blob(b'B', v)
        elif type(v) in (list, tuple):
            h.update(b'L' if type(v) is list else b'Q')
            h.update(struct.pack('!Q', len(v)))
            for item in v:
                visit(item)
        elif type(v) is dict:
            h.update(b'D')
            h.update(struct.pack('!Q', len(v)))
            for key in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(v[key])
        elif type(v) is torch.Tensor:
            if v.device.type != 'cpu' or v.requires_grad or v.grad_fn is not None:
                raise HarnessError('semantic digest requires detached CPU tensors')
            h.update(b'X')
            visit(str(v.dtype))
            visit(str(v.layout))
            visit(tuple(v.shape))
            if v.layout == torch.strided:
                blob(b'V', v.contiguous().numpy().tobytes())
            elif v.layout == torch.sparse_coo and v.is_coalesced():
                visit(v.indices())
                visit(v.values())
            else:
                raise HarnessError('unsupported digest tensor layout')
        else:
            raise HarnessError('unsupported owned digest value')
    visit(value)
    return h.hexdigest()


def bundle_manifest(sources):
    """Describe supplied closure bytes; never capture or import host source files."""
    if type(sources) is not dict or len(sources) != len(BUNDLE_PATHS):
        raise HarnessError('H bundle requires the complete explicit external closure')
    if any(type(k) is not str for k in sources) or set(sources) != set(BUNDLE_PATHS):
        raise HarnessError('unexpected or missing H bundle path')
    total = 0
    files = {}
    for path in sorted(sources):
        content = sources[path]
        if type(content) is not bytes or len(content) > 2097152:
            raise HarnessError('source packet exceeds the per-file bound')
        total += len(content)
        if total > 8388608:
            raise HarnessError('source packet exceeds the aggregate bound')
        files[path] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    return {'schema_version': 'amazon4u-H-bundle-v1', 'files': files,
            'sha256': owned_digest(files)}


def verify_bundle(sources, expected_manifest):
    actual = bundle_manifest(sources)
    if type(expected_manifest) is not dict or actual != expected_manifest:
        raise HarnessError('H bundle differs from independent expected closure')
    return actual['sha256']


def check_envelope(actual, expected, policy):
    preflight(actual, policy, 'H actual run envelope')
    preflight(expected, policy, 'H expected run envelope')
    if (type(actual) is not dict or type(expected) is not dict
            or set(actual) != ENVELOPE_KEYS or set(expected) != ENVELOPE_KEYS):
        raise HarnessError('invalid H run envelope schema')
    if actual != expected:
        raise HarnessError('run envelope differs from independent expectation')
    if (actual['schema_version'] != 'amazon4u-H-run-v1'
            or actual['profile_id'] != 'H-REF-synthetic-cpu-v1'
            or actual['determinism_policy'] != 'private_rng_bounded_cpu_no_global_setters'):
        raise HarnessError('unsupported execution profile')
    for key, value in actual.items():
        if type(value) is not str or not value or len(value) > 256:
            raise HarnessError('envelope values must be bounded nonempty strings')
    for key in ('identity_hash', 'bundle_sha256', 'initial_state_sha256'):
        value = actual[key]
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise HarnessError('invalid envelope digest')
    return copy.deepcopy(actual)


class PrivateRNG:
    """One owned object per generator ID; aliases are resolved by ID, not copied."""

    def __init__(self, identity, runtime):
        self.runtime = copy.deepcopy(runtime)
        self.roles = []
        for role in identity.resolved_settings['determinism']['rng_streams']:
            record = copy.deepcopy(role)
            record['generator_ids'] = list(runtime['rng_role_bindings'][role['name']])
            self.roles.append(record)
        self.objects = {}
        for gid, desc in runtime['rng_generator_bindings'].items():
            seed = desc['seed_material']
            if desc['worker_id'] is not None or type(seed) is not int:
                raise HarnessError('H-REF supports zero-worker integer-seeded private streams')
            codec = desc['codec']
            if codec == 'python.random.MT19937.v3':
                obj = random.Random(seed)
            elif codec == 'numpy.random.RandomState.MT19937.v1':
                obj = np.random.RandomState(seed)
            elif codec == 'numpy.random.PCG64.v1':
                obj = np.random.Generator(np.random.PCG64(seed))
            elif codec == 'torch.cpu.Generator.native.v1':
                obj = torch.Generator(device='cpu').manual_seed(seed)
            else:
                raise HarnessError('unsupported private RNG codec')
            self.objects[gid] = obj

    def role_id(self, role):
        ids = self.runtime['rng_role_bindings'][role]
        if len(ids) != 1:
            raise HarnessError('active H sampling roles require exactly one generator')
        return ids[0]

    def index(self, gid, length):
        if length <= 0:
            raise HarnessError('cannot draw from an empty pool')
        obj = self.objects[gid]
        codec = self.runtime['rng_generator_bindings'][gid]['codec']
        if codec == 'python.random.MT19937.v3':
            return obj.randrange(length)
        if codec == 'numpy.random.RandomState.MT19937.v1':
            return int(obj.randint(length))
        if codec == 'numpy.random.PCG64.v1':
            return int(obj.integers(length))
        return int(torch.randint(length, (1,), generator=obj, device='cpu').item())

    def choice_adapter(self, role):
        registry, gid = self, self.role_id(role)

        class Choice:
            def choice(self, pool):
                return pool[registry.index(gid, len(pool))]
        return Choice()

    def shuffle(self, values):
        gid = self.role_id('epoch_shuffle')
        obj = self.objects[gid]
        if type(obj) is random.Random:
            obj.shuffle(values)
        else:
            for i in range(len(values) - 1, 0, -1):
                j = self.index(gid, i + 1)
                values[i], values[j] = values[j], values[i]

    def capture(self):
        generators = []
        for gid, desc in self.runtime['rng_generator_bindings'].items():
            obj, codec = self.objects[gid], desc['codec']
            if codec == 'python.random.MT19937.v3':
                state = obj.getstate()
            elif codec == 'numpy.random.RandomState.MT19937.v1':
                raw = obj.get_state()
                state = (raw[0], [int(x) for x in raw[1]], int(raw[2]), int(raw[3]), float(raw[4]))
            elif codec == 'numpy.random.PCG64.v1':
                state = copy.deepcopy(obj.bit_generator.state)
            else:
                state = obj.get_state().clone()
            generators.append(dict(id=gid, state=state, **copy.deepcopy(desc)))
        return {'roles': copy.deepcopy(self.roles), 'generators': generators}

    def load_validated(self, rng):
        """Private setters only, after complete unchanged D2 validation."""
        for record in rng['generators']:
            obj, state, codec = self.objects[record['id']], record['state'], record['codec']
            if codec == 'python.random.MT19937.v3':
                obj.setstate(state)
            elif codec == 'numpy.random.RandomState.MT19937.v1':
                obj.set_state((state[0], np.array(state[1], dtype=np.uint32),
                               state[2], state[3], state[4]))
            elif codec == 'numpy.random.PCG64.v1':
                obj.bit_generator.state = copy.deepcopy(state)
            else:
                obj.set_state(state.clone())


class Selection:
    """Pure epoch/validation trace controller; no optimizer or evaluator mathematics."""

    def __init__(self, frequency, *, epoch=0, state=None):
        if type(frequency) is not int or not 1 <= frequency <= 100:
            raise HarnessError('invalid validation frequency')
        self.frequency, self.epoch = frequency, epoch
        self.state = copy.deepcopy(state) if state is not None else {
            'metric': 'validation_NDCG@10', 'history': [], 'best_score': None,
            'selected_epoch': None, 'bad_checks': 0, 'status': 'initial', 'stop_reason': None,
        }

    def advance(self, epoch, score=None):
        if self.state['status'] == 'stopped' or epoch != self.epoch + 1 or epoch > 100:
            raise HarnessError('nonconsecutive or terminal selection advance')
        scheduled = epoch % self.frequency == 0
        if scheduled:
            if type(score) is not float or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise HarnessError('scheduled validation requires finite eligible NDCG@10')
        elif score is not None:
            raise HarnessError('unscheduled selection score')
        next_state = copy.deepcopy(self.state)
        improved = False
        if scheduled:
            next_state['history'].append({'epoch': epoch, 'score': score})
            if next_state['best_score'] is None or score > next_state['best_score']:
                next_state.update(best_score=score, selected_epoch=epoch, bad_checks=0)
                improved = True
            else:
                next_state['bad_checks'] += 1
        patience = next_state['bad_checks'] == 10
        maximum = epoch == 100
        next_state['status'] = 'stopped' if patience or maximum else 'ready'
        next_state['stop_reason'] = (
            'patience_and_max_epochs' if patience and maximum else
            'patience' if patience else 'max_epochs' if maximum else None)
        self.state, self.epoch = next_state, epoch
        return improved


def validation_score(report):
    if (type(report) is not dict or report.get('split') != 'validation'
            or report['audit']['eligible_users'] <= 0):
        raise HarnessError('selection requires an eligible authoritative validation report')
    score = report['overall_metrics']['NDCG@10']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise HarnessError('validation NDCG@10 is unavailable or invalid')
    return float(score)


class HCPUAdam(torch.optim.Adam):
    """H-local eager CPU Adam; only the CUDA-capture health hook is replaced.

    Construction, step arithmetic, lazy state, parameter IDs and native state
    transfer are inherited unchanged. This is not a global Torch/CUDA bypass.
    """

    def _cuda_graph_capture_health_check(self):
        if torch.compiler.is_compiling():
            raise HarnessError('H CPU Adam requires eager execution')
        if self.defaults != ADAM_OPTIONS:
            raise HarnessError('H CPU Adam defaults/modes differ from exact approved options')
        if (getattr(self, 'grad_scale', None) is not None
                or getattr(self, 'found_inf', None) is not None):
            raise HarnessError('H CPU Adam does not support AMP execution modes')
        if len(self.param_groups) != 1:
            raise HarnessError('H CPU Adam requires one approved parameter group')
        for group in self.param_groups:
            options = {key: value for key, value in group.items() if key != 'params'}
            if options != ADAM_OPTIONS:
                raise HarnessError('H CPU Adam options/modes differ from exact approved options')
            for parameter in group['params']:
                if parameter.device.type != 'cpu':
                    raise HarnessError('H CPU Adam parameter must remain CPU')
                if parameter.grad is not None and parameter.grad.device.type != 'cpu':
                    raise HarnessError('H CPU Adam gradient must remain CPU')

        def check_state(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cpu':
                    raise HarnessError('H CPU Adam state tensor must remain CPU')
            elif isinstance(value, dict):
                for key, item in value.items():
                    check_state(key)
                    check_state(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check_state(item)
        # Include every existing state entry, not just currently active parameters.
        for parameter, state in self.state.items():
            check_state(parameter)
            if isinstance(parameter, torch.Tensor) and parameter.grad is not None:
                check_state(parameter.grad)
            check_state(state)


class CPUBackend:
    """Placement, numeric calls and native state transfer are outside the controller."""

    def capabilities(self):
        return {'device': 'cpu', 'accelerator': 'none', 'supports_training': True,
                'state_transfer': 'cpu_quiescent_epoch_v1'}

    def require(self, device, accelerator):
        capability = self.capabilities()
        if (capability.get('supports_training') is not True
                or device != capability.get('device')
                or accelerator != capability.get('accelerator')):
            raise HarnessError('unsupported device/accelerator; no fallback is permitted')
        return capability

    def verify_runtime(self, runtime):
        profile = runtime['runtime_profile']
        if (profile['profile_id'] != 'synthetic_profile_v1'
                or profile['python_version'] != '.'.join(map(str, sys.version_info[:3]))
                or profile['numpy_version'] != np.__version__
                or profile['torch_version'] != str(torch.__version__)
                or profile['adam_step_dtype'] != 'torch.float32'
                or profile['batch_policy'] != 'filter_before_batch'
                or torch.get_default_dtype() != torch.float32):
            raise HarnessError('unsupported or mismatched CPU fixture runtime')

    def make_model(self, data, light_initial=None):
        method = data.identity.method
        if method == 'LightGCN':
            if type(light_initial) is not dict or set(light_initial) != {'user_embeddings', 'item_embeddings'}:
                raise HarnessError('LightGCN requires explicit owned fixture tensors')
            preflight(light_initial, data.resource_policy, 'H LightGCN initialization')
            return DifferentiableLightGCN(
                LightGCNGraph(data.context), light_initial['user_embeddings'],
                light_initial['item_embeddings'])
        if method == 'Heterogeneous KG+GNN':
            if light_initial is not None:
                raise HarnessError('LightGCN initialization supplied for heterogeneous model')
            graph = build_hetero_graph(data.context, data.product_categories, data.product_brands)
            return DifferentiableHeteroGNN(graph, init_seed=data.identity.seed, embedding_dim=64)
        raise HarnessError('unsupported learned method')

    def snapshot_model(self, model):
        return {name: value.detach().clone() for name, value in model.state_dict().items()}

    def _require_optimizer(self, model, optimizer):
        if type(optimizer) is not HCPUAdam:
            raise HarnessError('H CPU backend requires its scoped CPU Adam adapter')
        optimizer._cuda_graph_capture_health_check()
        bound = [p for group in optimizer.param_groups for p in group['params']]
        parameters = list(model.parameters())
        if len(bound) != len(parameters) or any(a is not b for a, b in zip(bound, parameters)):
            raise HarnessError('H CPU Adam parameter binding differs from the model')

    def make_optimizer(self, model, runtime):
        groups = runtime['optimizer_groups']
        names = [name for name, _ in model.named_parameters()]
        if (len(groups) != 1 or list(groups[0]['parameter_names']) != names
                or groups[0]['options'] != ADAM_OPTIONS):
            raise HarnessError('unsupported H fixture Adam binding/options')
        optimizer = HCPUAdam(list(model.parameters()), **copy.deepcopy(ADAM_OPTIONS))
        self._require_optimizer(model, optimizer)
        return optimizer

    def update(self, model, optimizer, triples):
        if not triples:
            raise HarnessError('empty optimizer update refused')
        # Refuse incompatible placement, modes and binding before clearing existing
        # gradients, calling the model, creating lazy state or changing parameters.
        self._require_optimizer(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        users, positive, negative = map(list, zip(*triples))
        losses = model.compute_batch_loss(users, positive, negative, reg_coefficient=1e-5)
        losses['total_loss'].backward()
        active = [parameter for parameter in model.parameters() if parameter.grad is not None]
        if not active or any(not torch.isfinite(p.grad).all() for p in active):
            optimizer.zero_grad(set_to_none=True)
            raise HarnessError('missing or nonfinite gradients; optimizer not stepped')
        result = {key: float(value.detach().item()) for key, value in losses.items()}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return result

    def load_model(self, model, state):
        model.load_state_dict(state, strict=True, assign=False)

    def load_optimizer(self, optimizer, state):
        if type(optimizer) is not HCPUAdam:
            raise HarnessError('H CPU backend requires its scoped CPU Adam adapter')
        optimizer.load_state_dict(state)
        optimizer._cuda_graph_capture_health_check()

    def evaluate(self, data, model):
        with torch.no_grad():
            return Evaluator(data.context).evaluate(
                data.validation_records, model, split='validation')


def baseline_validation(data, method):
    """Separate deterministic routes: never construct an optimizer."""
    if method == 'CorePopularity':
        report = Evaluator(data.context).evaluate(
            data.validation_records, PopularityScorer(data.context), split='validation')
        return {'mixture': None, 'validation_report': report}
    if method == 'KG-only':
        mixture, trials = select_kg_only_mixture(
            data.context, data.product_categories, data.product_brands,
            data.validation_records)
        return {'mixture': mixture, 'validation_report': trials[mixture]}
    raise HarnessError('unknown deterministic baseline route')


class Harness:
    """An owned fixture trajectory. Restore returns a fresh handle, never mutates this one.

    A failed epoch is poisoned, not rolled back or reported as checkpointable. D2
    max_epochs=100/patience=10 remain fixed; the separate fixture execution cap is
    at most four actual Adam calls over the complete trajectory, including resume.
    """

    def __init__(self, data, expectations, *, backend, requested_device, accelerator,
                 run_envelope, expected_run_envelope, bundle_sources, expected_bundle,
                 light_initial=None, max_adam_steps=4):
        backend.require(requested_device, accelerator)
        if type(data) is not TrainingData:
            raise HarnessError('Harness requires the bounded owned data adapter')
        if type(max_adam_steps) is not int or not 1 <= max_adam_steps <= 4:
            raise HarnessError('fixture Adam cap must be 1..4')
        if type(expectations) is not dict or set(expectations) != set(D2_KEYS) | {'io_policy'}:
            raise HarnessError('all independent D2 and D3 policies are mandatory')
        policy = checked_policy(expectations['resource_policy'])
        for key in D2_KEYS:
            preflight(expectations[key], policy, 'H expectation ' + key)
        self.expectations = copy.deepcopy(expectations)
        self.d2 = {key: self.expectations[key] for key in D2_KEYS}
        self.backend, self.max_adam_steps = backend, max_adam_steps
        self.envelope = check_envelope(run_envelope, expected_run_envelope, policy)
        self.expected_bundle = copy.deepcopy(expected_bundle)
        digest = verify_bundle(bundle_sources, expected_bundle)
        if digest != self.envelope['bundle_sha256']:
            raise HarnessError('run/bundle binding mismatch')
        if (requested_device != self.envelope['device']
                or accelerator != self.envelope['accelerator']):
            raise HarnessError('run/device binding mismatch')
        backend.verify_runtime(self.expectations['expected_runtime'])
        self.data = data.owned_copy(expected_identity=self.expectations['expected_identity'])
        ident = self.data.identity
        if (self.envelope['identity_hash'] != ident.identity_hash
                or self.envelope['method'] != ident.method):
            raise HarnessError('run/input/method binding mismatch')
        settings = ident.resolved_settings
        if (settings['execution']['runtime_workers'] != 0
                or settings['execution']['execution_device'] != requested_device
                or settings['training']['max_epochs'] != 100
                or settings['training']['patience'] != 10
                or settings['model']['embedding_dim'] != 64
                or settings['training']['endpoint_regularization'] != 1e-5):
            raise HarnessError('identity does not select the bounded H fixture policy')
        self.sampler = NegativeSampler(self.data.context)
        self.model = backend.make_model(self.data, light_initial)
        initial = backend.snapshot_model(self.model)
        if owned_digest(initial) != self.envelope['initial_state_sha256']:
            raise HarnessError('initialization archive binding mismatch')
        self.optimizer = backend.make_optimizer(self.model, self.expectations['expected_runtime'])
        self.rng = PrivateRNG(ident, self.expectations['expected_runtime'])
        self.rng.role_id('epoch_shuffle')
        self.rng.role_id('negative_sampler')
        self.selection = Selection(settings['training']['validation_frequency'])
        self.progress = {'epoch': 0, 'global_step': 0,
                         'epoch_optimizer_steps': [], 'empty_pool_skips': []}
        self.source_order, self.best_model = None, None
        self.validation_report = None
        self.poisoned, self.busy = False, False
        self.capture_epoch_checkpoint()

    def _current_model(self):
        return {'epoch': self.progress['epoch'],
                'row_order': copy.deepcopy(self.expectations['expected_model']['row_order']),
                'state_dict': self.backend.snapshot_model(self.model)}

    def capture_epoch_checkpoint(self):
        if self.poisoned or self.busy:
            raise HarnessError('only successful quiescent epoch boundaries are checkpointable')
        epoch = self.progress['epoch']
        payload = {
            'schema_version': 'amazon4u-checkpoint-payload-v1',
            'capability': 'epoch_boundary_validation_only',
            'identity': self.data.identity.to_dict(),
            'provenance': copy.deepcopy(self.expectations['expected_provenance']),
            'runtime_profile': copy.deepcopy(self.expectations['expected_runtime']['runtime_profile']),
            'current_model': self._current_model(),
            'best_model': copy.deepcopy(self.best_model),
            'optimizer': self._optimizer_snapshot(),
            'rng': self.rng.capture(),
            'sampler': {
                'algorithm': self.expectations['expected_runtime']['sampler_algorithm'],
                'identity_hash': self.data.identity.identity_hash,
                'source_order': copy.deepcopy(self.source_order),
                'cursor': len(self.source_order) if epoch else 0,
                'next_epoch': epoch + 1, 'batch_cursor': 0, 'pending_batches': 0,
                'prefetch_pending': 0, 'inflight_workers': 0, 'worker_ids': [],
            },
            'progress': copy.deepcopy(self.progress),
            'selection': copy.deepcopy(self.selection.state),
            'scheduler': {'applicable': False, 'state': None,
                          'reason': 'not_used_constant_learning_rate'},
            'amp': {'applicable': False, 'state': None, 'reason': 'not_used_cpu_float32'},
        }
        return validate_checkpoint_payload(payload, **self.d2)

    def _optimizer_snapshot(self):
        state = copy.deepcopy(self.optimizer.state_dict())
        ids = [pid for group in state['param_groups'] for pid in group['params']]
        names = [name for name, _ in self.model.named_parameters()]
        bindings = []
        for pid, name in zip(ids, names):
            actual = state['state'].get(pid, {})
            updates = int(actual['step'].item()) if actual else 0
            bindings.append({'id': pid, 'name': name, 'updates': updates})
        return {'algorithm': 'Adam', 'state_dict': state, 'parameter_bindings': bindings}

    def train_epoch(self):
        if self.poisoned or self.busy or self.selection.state['status'] == 'stopped':
            raise HarnessError('trajectory is busy, failed or scientifically terminal')
        context = self.data.context
        trainable_count = sum(
            bool(context.warm_items - context.all_rated_history[user])
            for user, _ in self.data.positive_pairs)
        batch_size = self.data.identity.resolved_settings['training']['batch_size']
        count = (trainable_count + batch_size - 1) // batch_size
        if count <= 0 or self.progress['global_step'] + count > self.max_adam_steps:
            raise HarnessError('whole epoch would exceed the fixture Adam budget')
        self.busy = True
        try:
            order = list(self.data.positive_pairs)
            self.rng.shuffle(order)
            triples, skips = self.sampler.prepare_batch(
                order, self.rng.choice_adapter('negative_sampler'))
            if len(triples) != trainable_count:
                raise HarnessError('sampler trainable count disagreement')
            losses = []
            for start in range(0, len(triples), batch_size):
                losses.append(self.backend.update(
                    self.model, self.optimizer, triples[start:start + batch_size]))
            epoch = self.progress['epoch'] + 1
            report, score = None, None
            if epoch % self.selection.frequency == 0:
                report = self.backend.evaluate(self.data, self.model)
                score = validation_score(report)
            improved = self.selection.advance(epoch, score)
            self.progress['epoch'] = epoch
            self.progress['global_step'] += len(losses)
            self.progress['epoch_optimizer_steps'].append(len(losses))
            self.progress['empty_pool_skips'].append(skips)
            self.source_order = order
            if improved:
                self.best_model = self._current_model()
            if report is not None:
                self.validation_report = copy.deepcopy(report)
            self.busy = False
            self.capture_epoch_checkpoint()
            return {'epoch': epoch, 'optimizer_steps': len(losses),
                    'empty_pool_skips': skips, 'losses': losses,
                    'source_order': list(order)}
        except Exception:
            self.poisoned = True
            self.busy = False
            raise

    def restore_epoch_checkpoint(self, payload, *, actual_envelope,
                                 expected_checkpoint_digest, bundle_sources):
        check_envelope(actual_envelope, self.envelope, self.d2['resource_policy'])
        if verify_bundle(bundle_sources, self.expected_bundle) != self.envelope['bundle_sha256']:
            raise HarnessError('restore source closure mismatch')
        self.backend.require(self.envelope['device'], self.envelope['accelerator'])
        self.backend.verify_runtime(self.expectations['expected_runtime'])
        owned = validate_checkpoint_payload(payload, **self.d2)
        if (type(expected_checkpoint_digest) is not str
                or owned_digest(owned) != expected_checkpoint_digest):
            raise HarnessError('checkpoint differs from independently retained semantic digest')
        if owned['progress']['global_step'] > self.max_adam_steps:
            raise HarnessError('checkpoint exceeds fixture trajectory budget')
        # No active model, optimizer or RNG setter occurs below. All objects are fresh.
        staged = object.__new__(Harness)
        staged.expectations = copy.deepcopy(self.expectations)
        staged.d2 = {key: staged.expectations[key] for key in D2_KEYS}
        staged.backend = self.backend
        staged.max_adam_steps = self.max_adam_steps
        staged.envelope = copy.deepcopy(self.envelope)
        staged.expected_bundle = copy.deepcopy(self.expected_bundle)
        staged.data = self.data.owned_copy(expected_identity=staged.d2['expected_identity'])
        staged.sampler = NegativeSampler(staged.data.context)
        state = owned['current_model']['state_dict']
        light = ({key: state[key] for key in ('user_embeddings', 'item_embeddings')}
                 if staged.data.identity.method == 'LightGCN' else None)
        staged.model = self.backend.make_model(staged.data, light)
        self.backend.load_model(staged.model, state)
        staged.optimizer = self.backend.make_optimizer(staged.model, staged.d2['expected_runtime'])
        self.backend.load_optimizer(staged.optimizer, owned['optimizer']['state_dict'])
        staged.rng = PrivateRNG(staged.data.identity, staged.d2['expected_runtime'])
        staged.rng.load_validated(owned['rng'])
        staged.progress = copy.deepcopy(owned['progress'])
        staged.source_order = copy.deepcopy(owned['sampler']['source_order'])
        staged.selection = Selection(
            staged.data.identity.resolved_settings['training']['validation_frequency'],
            epoch=staged.progress['epoch'], state=owned['selection'])
        staged.best_model = copy.deepcopy(owned['best_model'])
        staged.validation_report = None
        staged.poisoned, staged.busy = False, False
        # Round-trip the fully staged native objects before returning a replacement.
        if owned_digest(staged.capture_epoch_checkpoint()) != owned_digest(owned):
            raise HarnessError('staged native restoration did not preserve complete state')
        return staged

    def acquire_registered_sources(self, source_roots, source_sizes, *, git_evidence,
                                   environment_records):
        return acquire_sources(
            source_roots, source_sizes, git_evidence=git_evidence,
            environment_records=environment_records,
            expected_provenance=self.expectations['expected_provenance'],
            io_policy=self.expectations['io_policy'])

    def publish_epoch_checkpoint(self, root_fd, name):
        owned = self.capture_epoch_checkpoint()
        published = publish_checkpoint(root_fd, name, owned, **self.d2,
                                       io_policy=self.expectations['io_policy'])
        return {'schema_version': 'amazon4u-H-checkpoint-receipt-v1', 'name': published,
                'checkpoint_digest': owned_digest(owned),
                'run_envelope': copy.deepcopy(self.envelope)}

    def retrieve_epoch_checkpoint(self, root_fd, name, *, actual_envelope,
                                  expected_checkpoint_digest, bundle_sources):
        check_envelope(actual_envelope, self.envelope, self.d2['resource_policy'])
        verify_bundle(bundle_sources, self.expected_bundle)
        owned = retrieve_checkpoint(root_fd, name, **self.d2,
                                    io_policy=self.expectations['io_policy'])
        return self.restore_epoch_checkpoint(
            owned, actual_envelope=actual_envelope,
            expected_checkpoint_digest=expected_checkpoint_digest,
            bundle_sources=bundle_sources)

    def selected_validation(self):
        if self.best_model is None or self.poisoned or self.busy:
            raise HarnessError('no successful selected validation snapshot')
        state = self.best_model['state_dict']
        light = ({key: state[key] for key in ('user_embeddings', 'item_embeddings')}
                 if self.data.identity.method == 'LightGCN' else None)
        model = self.backend.make_model(self.data, light)
        self.backend.load_model(model, state)
        return self.backend.evaluate(self.data, model)


class LiveSlot:
    """Optional explicit live-swap seam; assignment follows all staged checks."""

    def __init__(self, live):
        self.live = live

    def restore(self, payload, **kwargs):
        replacement = self.live.restore_epoch_checkpoint(payload, **kwargs)
        self.live = replacement
        return replacement
