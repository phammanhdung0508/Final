"""GPU-READY runtime: explicit device selection, determinism and environment evidence.

Nothing here falls back to another device. A CUDA request without a usable GPU, a
missing `CUBLAS_WORKSPACE_CONFIG`, or any changed determinism setting is an error.
See docs/gpu-ready-design.md (GR01, GR04, GR05, GR06).
"""

import os
import platform
import sys

import numpy as np
import torch

PROFILE_ID = 'gpu-ready-synthetic-cuda-v1'
FINGERPRINT_SCHEMA = 'amazon4u-gpu-environment-v1'
CUBLAS_WORKSPACE = ':4096:8'


class RuntimeRefusal(RuntimeError):
    """The requested device or determinism policy cannot be honored."""


def require_device(device, accelerator):
    """Return the torch device for an explicit request, or refuse."""
    if (device, accelerator) == ('cpu', 'none'):
        return torch.device('cpu')
    if (device, accelerator) == ('cuda:0', 'gpu'):
        if not torch.cuda.is_available():
            raise RuntimeRefusal('CUDA requested but unavailable; no CPU fallback')
        return torch.device('cuda:0')
    raise RuntimeRefusal(f'unsupported device/accelerator {device!r}/{accelerator!r}')


def configure_determinism():
    """Apply the fixed policy; CUDA ops without a deterministic kernel will raise."""
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != CUBLAS_WORKSPACE:
        raise RuntimeRefusal(f'CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE} before CUDA starts')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return determinism_record()


def determinism_record():
    return {
        'use_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
    }


EXPECTED_DETERMINISM = {
    'use_deterministic_algorithms': True, 'deterministic_warn_only': False,
    'CUBLAS_WORKSPACE_CONFIG': CUBLAS_WORKSPACE, 'cudnn_deterministic': True,
    'cudnn_benchmark': False, 'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
    'float32_matmul_precision': 'highest',
}


def require_determinism():
    record = determinism_record()
    if record != EXPECTED_DETERMINISM:
        raise RuntimeRefusal(f'determinism settings changed: {record!r}')
    return record


def environment_fingerprint(device):
    """Observed environment; a metadata label alone is not evidence (GR05)."""
    cuda = device.type == 'cuda'
    record = {
        'schema_version': FINGERPRINT_SCHEMA,
        'device': str(device),
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'numpy_version': np.__version__,
        'torch_version': str(torch.__version__),
        'platform': platform.platform(),
        'cuda_runtime_version': torch.version.cuda if cuda else None,
        'cudnn_version': torch.backends.cudnn.version() if cuda else None,
        'visible_device_count': torch.cuda.device_count() if cuda else 0,
        'used_device_count': 1 if cuda else 0,
        'gpu_name': torch.cuda.get_device_name(device) if cuda else None,
        'compute_capability': list(torch.cuda.get_device_capability(device)) if cuda else None,
        'driver_version': _driver_version() if cuda else None,
    }
    return record


def _driver_version():
    try:
        import subprocess
        out = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return out.strip().splitlines()[0]
    except Exception as exc:  # the fingerprint must say why it is missing
        return f'unavailable: {type(exc).__name__}'


GLOBAL_RNG_SEED = 0


def pin_global_generators():
    """Seed the global generators to one fixed value at process start.

    PyTorch seeds the global CPU generator randomly in every new process. Training never
    consumes the global generators, so after this call every run process holds the same
    states, and a checkpoint written in one process can be resumed in another. Call it
    before building the trainer in any process that writes or resumes a checkpoint.
    """
    torch.manual_seed(GLOBAL_RNG_SEED)
    return GLOBAL_RNG_SEED


def global_rng_states(device):
    """Global generators that the run must never consume: CPU, and CUDA if used."""
    states = {'torch_cpu': torch.get_rng_state().clone()}
    if device.type == 'cuda':
        states['torch_cuda'] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return states


def require_rng_unchanged(before, after, epoch):
    """User rule (2026-10-24): GPU random state must not change during an epoch."""
    if before.keys() != after.keys():
        raise RuntimeRefusal('global RNG inventory changed')
    for key in before:
        a, b = before[key], after[key]
        same = (torch.equal(a, b) if isinstance(a, torch.Tensor)
                else len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b)))
        if not same:
            raise RuntimeRefusal(f'global {key} RNG state changed during epoch {epoch}')
