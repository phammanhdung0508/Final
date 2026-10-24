"""GR06: the device checkpoint profile `amazon4u-checkpoint-payload-cuda-v1`.

A cuda-v1 payload is the accepted D2 v1 payload with three device-profile fields and
one extra `device` section:

- `schema_version` = `amazon4u-checkpoint-payload-cuda-v1`;
- `capability` = `epoch_boundary_same_environment_resume`;
- `amp.reason` = `not_used_<cpu|cuda>_float32`, matching the recorded device;
- `device` = environment fingerprint, determinism record, global generator states and
  the run binding (bundle and run-spec digests).

Validation never reimplements D2. The device section is checked here against an
independent expectation. The rest is checked by calling D2's public
`validate_checkpoint_payload` on the core view, which is the payload without `device`
and with the three profile fields set to their v1 values. Everything else must be
byte-identical, so any core field D2 rejects is rejected here.

The CUDA generator is never consumed by training (user rule, 2026-10-24), so the D1
`PyTorch_CUDA_if_applicable` stream stays inactive and D2 sees no active CUDA role. Its
state is kept in `device.global_rng` only so that restore leaves it exactly as it was.
Restore is valid only in the environment the fingerprint describes.
"""

import copy
import re

import torch

from scripts.checkpoint_payload import CheckpointValidationError, validate_checkpoint_payload
from scripts.gpu_runtime import EXPECTED_DETERMINISM, FINGERPRINT_SCHEMA

SCHEMA_VERSION = 'amazon4u-checkpoint-payload-cuda-v1'
CAPABILITY = 'epoch_boundary_same_environment_resume'
V1_PROFILE = {'schema_version': 'amazon4u-checkpoint-payload-v1',
              'capability': 'epoch_boundary_validation_only',
              'amp_reason': 'not_used_cpu_float32'}
DEVICE_KEYS = {'fingerprint', 'determinism', 'global_rng', 'run_binding'}
RUN_BINDING_KEYS = {'bundle_sha256', 'run_spec_sha256'}
FINGERPRINT_KEYS = {
    'schema_version', 'device', 'python_version', 'numpy_version', 'torch_version',
    'platform', 'cuda_runtime_version', 'cudnn_version', 'visible_device_count',
    'used_device_count', 'gpu_name', 'compute_capability', 'driver_version',
}
DEVICES = {'cpu': 'not_used_cpu_float32', 'cuda:0': 'not_used_cuda_float32'}
MAX_RNG_STATE_BYTES = 1 << 16
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _fail(message):
    raise CheckpointValidationError(message)


def _rng_state(value, role):
    if (type(value) is not torch.Tensor or value.device.type != 'cpu' or value.dtype != torch.uint8
            or value.dim() != 1 or not value.is_contiguous() or value.requires_grad
            or not 1 <= value.numel() <= MAX_RNG_STATE_BYTES):
        _fail(f'{role} must be a 1-D contiguous detached CPU uint8 tensor')


def validate_device_expectation(expected):
    """The independent expectation: fingerprint, determinism and run binding."""
    if type(expected) is not dict or set(expected) != {'fingerprint', 'determinism', 'run_binding'}:
        _fail('expected_device keys must be exactly fingerprint, determinism, run_binding')
    fp = expected['fingerprint']
    if type(fp) is not dict or set(fp) != FINGERPRINT_KEYS or fp['schema_version'] != FINGERPRINT_SCHEMA:
        _fail('expected fingerprint is not an amazon4u-gpu-environment-v1 record')
    if fp['device'] not in DEVICES:
        _fail(f'unsupported checkpoint device {fp["device"]!r}')
    cuda = fp['device'] == 'cuda:0'
    counts = (fp['visible_device_count'], fp['used_device_count'])
    if any(type(c) is not int for c in counts) or (counts[1] != 1 if cuda else counts != (0, 0)):
        _fail('fingerprint device counts disagree with the recorded device')
    if cuda and (counts[0] < 1 or expected['determinism'] != EXPECTED_DETERMINISM):
        _fail('a CUDA checkpoint requires the fixed determinism policy and a visible device')
    if expected['determinism'] is not None and type(expected['determinism']) is not dict:
        _fail('expected determinism must be a record or None')
    binding = expected['run_binding']
    if (type(binding) is not dict or set(binding) != RUN_BINDING_KEYS
            or any(type(v) is not str or not _DIGEST.fullmatch(v) for v in binding.values())):
        _fail('run_binding must hold exactly two lowercase SHA-256 digests')
    return fp


def _validate_device_section(device, expected):
    fp = validate_device_expectation(expected)
    if type(device) is not dict or set(device) != DEVICE_KEYS:
        _fail('payload.device keys must be exactly ' + ', '.join(sorted(DEVICE_KEYS)))
    for key in ('fingerprint', 'determinism', 'run_binding'):
        if device[key] != expected[key]:
            _fail(f'payload.device.{key} differs from the independent expectation; resume refused')
    rng = device['global_rng']
    want = {'torch_cpu', 'torch_cuda'} if fp['device'] == 'cuda:0' else {'torch_cpu'}
    if type(rng) is not dict or set(rng) != want:
        _fail(f'payload.device.global_rng keys must be exactly {sorted(want)}')
    _rng_state(rng['torch_cpu'], 'device.global_rng.torch_cpu')
    if 'torch_cuda' in rng:
        states = rng['torch_cuda']
        if type(states) is not list or len(states) != fp['visible_device_count']:
            _fail('device.global_rng.torch_cuda must hold one state per visible device')
        for i, state in enumerate(states):
            _rng_state(state, f'device.global_rng.torch_cuda[{i}]')
        if len({s.numel() for s in states}) != 1:
            _fail('CUDA generator states must have one common length')
    return DEVICES[fp['device']]


def core_view(payload):
    """The D2 v1 payload this cuda-v1 payload claims to contain."""
    core = {k: v for k, v in payload.items() if k != 'device'}
    core['schema_version'] = V1_PROFILE['schema_version']
    core['capability'] = V1_PROFILE['capability']
    core['amp'] = {**payload['amp'], 'reason': V1_PROFILE['amp_reason']}
    return core


def validate_cuda_checkpoint_payload(payload, *, expected_device, expected_identity,
                                     expected_provenance, expected_model, expected_runtime,
                                     resource_policy):
    """Validate a cuda-v1 payload; return an owned copy or raise CheckpointValidationError."""
    if type(payload) is not dict or set(payload) != set(V1_KEYS) | {'device'}:
        _fail('cuda-v1 payload keys must be the v1 keys plus device')
    if payload['schema_version'] != SCHEMA_VERSION or payload['capability'] != CAPABILITY:
        _fail('not an amazon4u-checkpoint-payload-cuda-v1 epoch-boundary payload')
    amp_reason = _validate_device_section(payload['device'], expected_device)
    amp = payload['amp']
    if type(amp) is not dict or amp.get('reason') != amp_reason:
        _fail(f'payload.amp.reason must be {amp_reason!r} for the recorded device')
    owned_core = validate_checkpoint_payload(
        core_view(payload), expected_identity=expected_identity,
        expected_provenance=expected_provenance, expected_model=expected_model,
        expected_runtime=expected_runtime, resource_policy=resource_policy)
    owned = dict(owned_core)
    owned['schema_version'] = SCHEMA_VERSION
    owned['capability'] = CAPABILITY
    owned['amp'] = {**owned_core['amp'], 'reason': amp_reason}
    device = payload['device']
    owned['device'] = {
        'fingerprint': copy.deepcopy(device['fingerprint']),
        'determinism': copy.deepcopy(device['determinism']),
        'run_binding': copy.deepcopy(device['run_binding']),
        'global_rng': {k: ([s.clone() for s in v] if isinstance(v, list) else v.clone())
                       for k, v in device['global_rng'].items()},
    }
    return owned


def build_cuda_payload(core, *, device_record):
    """Wrap a D2 v1-shaped core with the device profile. Validation happens separately."""
    payload = copy.deepcopy(core)
    if payload.get('schema_version') != V1_PROFILE['schema_version']:
        _fail('core must be a v1-shaped payload')
    payload['schema_version'] = SCHEMA_VERSION
    payload['capability'] = CAPABILITY
    payload['amp'] = {**payload['amp'], 'reason': DEVICES[device_record['fingerprint']['device']]}
    payload['device'] = copy.deepcopy(device_record)
    return payload


V1_KEYS = ('schema_version', 'capability', 'identity', 'provenance', 'runtime_profile',
           'current_model', 'best_model', 'optimizer', 'rng', 'sampler', 'progress',
           'selection', 'scheduler', 'amp')
