"""Run the accepted D1/D2 beside the current ones and compare every outcome.

`tests/accepted_checkpoint/` holds byte-exact copies of the accepted
`checkpoint_identity.py`, `checkpoint_payload.py` and `checkpoint_storage.py`; `accepted_modules()` checks them
against the recorded O-OFFLINE hashes, then loads them so the D2 copy binds to the D1
copy. `outcome()` reduces a call to what a caller can observe: the returned value
(identities as `to_dict()`, payloads by H-REF's `owned_digest`) or the exception's class
name and message. `dual()` wraps the current public entry points so each call also runs
the accepted code on the same arguments and records any difference in `MISMATCHES`.
"""

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_DIR = Path(__file__).resolve().parent / 'accepted_checkpoint'
MISMATCHES = []
CURRENT_TEST = [None]
D1_FUNCTIONS = ('_validate_clean_text', 'validate_category_id', 'validate_brand_id',
                'compute_data_fingerprint', 'compute_graph_fingerprint',
                'compute_validation_fingerprint', 'build_canonical_identity',
                'validate_canonical_identity')
D2_FUNCTIONS = ('validate_checkpoint_payload', 'preflight_resource_accounting',
                '_compute_condition_positive_and_empty_pool_pairs')
_loaded = None


def recorded_hashes():
    recorded = json.loads((ROOT / 'configs/o-offline-implementation.json').read_text())
    return recorded['source_copy_manifest_sha256']


def accepted_modules():
    """(accepted D1, accepted D2) loaded from the pinned copies; refuses changed copies."""
    global _loaded
    if _loaded is None:
        pins = recorded_hashes()
        for name in ('checkpoint_identity.py', 'checkpoint_payload.py'):
            actual = hashlib.sha256((ACCEPTED_DIR / name).read_bytes()).hexdigest()
            if actual != pins[f'scripts/{name}']:
                raise AssertionError(f'accepted copy {name} differs from its recorded hash')
        d1 = _load('accepted_checkpoint_identity', ACCEPTED_DIR / 'checkpoint_identity.py')
        # The D2 copy imports D1 as `checkpoint_identity` or `scripts.checkpoint_identity`.
        names = ('checkpoint_identity', 'scripts.checkpoint_identity')
        current = {n: sys.modules.get(n) for n in names}
        sys.modules.update(dict.fromkeys(names, d1))
        try:
            d2 = _load('accepted_checkpoint_payload', ACCEPTED_DIR / 'checkpoint_payload.py')
        finally:
            for n, module in current.items():
                if module is None:
                    del sys.modules[n]
                else:
                    sys.modules[n] = module
        _loaded = d1, d2
    return _loaded


def accepted_storage():
    """The accepted D3 module from its pinned copy; it binds to the current D1/D2."""
    pins = recorded_hashes()
    path = ACCEPTED_DIR / 'checkpoint_storage.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != pins['scripts/checkpoint_storage.py']:
        raise AssertionError('accepted copy checkpoint_storage.py differs from its recorded hash')
    return sys.modules.get('accepted_checkpoint_storage') or _load('accepted_checkpoint_storage', path)


def _load(name, path):
    """Import a copy; undo its import-time `setattr` on the `scripts` package."""
    import scripts
    attribute = path.stem
    saved = scripts.__dict__.get(attribute)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            scripts.__dict__.pop(attribute, None)
        else:
            setattr(scripts, attribute, saved)
    return module


def _canonical(value):
    """A comparable, tensor-free form of a returned value (exact types, bytes and order)."""
    import torch
    if isinstance(value, torch.Tensor):
        dense = value.coalesce() if value.layout == torch.sparse_coo else value
        parts = ((dense.indices(), dense.values()) if value.layout == torch.sparse_coo else (dense,))
        return ('tensor', str(value.dtype), str(value.layout), tuple(value.shape), value.requires_grad,
                tuple(hashlib.sha256(p.detach().contiguous().cpu().numpy().tobytes()).hexdigest() for p in parts))
    if dataclasses.is_dataclass(value) and hasattr(value, 'to_dict'):
        return ('identity', _canonical(value.to_dict()))
    if isinstance(value, dict):
        return (type(value).__name__, tuple((_canonical(k), _canonical(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return (type(value).__name__, tuple(_canonical(v) for v in items))
    if isinstance(value, int) and not isinstance(value, bool):
        return (type(value).__name__, hex(value))
    return (type(value).__name__, repr(value))


def _observable(value):
    return _canonical(value)


def outcome(fn, *args, **kwargs):
    try:
        return ('ok', _observable(fn(*args, **kwargs)))
    except Exception as exc:
        return ('raised', type(exc).__name__, str(exc))


def to_accepted_identity(value):
    """A current CanonicalResumeIdentity as the accepted class; anything else unchanged."""
    import scripts.checkpoint_identity as d1
    if type(value) is d1.CanonicalResumeIdentity:
        old = accepted_modules()[0].CanonicalResumeIdentity
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        return _bypass_init(old, fields)
    return value


def _bypass_init(cls, fields):
    obj = object.__new__(cls)
    for k, v in fields.items():
        object.__setattr__(obj, k, v)
    return obj


@contextlib.contextmanager
def dual():
    """Patch the current D1/D2 entry points to also run the accepted code and compare."""
    import scripts.checkpoint_identity as d1
    import scripts.checkpoint_payload as d2
    old_d1, old_d2 = accepted_modules()
    saved = []

    def patch(module, name, accepted):
        current = getattr(module, name)
        saved.append((module, name, current))

        def both(*args, **kwargs):
            converted = [to_accepted_identity(a) for a in args]
            converted_kw = {k: to_accepted_identity(v) for k, v in kwargs.items()}
            expected = outcome(accepted, *converted, **converted_kw)
            try:
                result = current(*args, **kwargs)
            except Exception as exc:
                got = ('raised', type(exc).__name__, str(exc))
                if got != expected:
                    MISMATCHES.append((CURRENT_TEST[0], name, expected, got))
                raise
            got = ('ok', _observable(result))
            if got != expected:
                MISMATCHES.append((CURRENT_TEST[0], name, expected, got))
            return result
        setattr(module, name, both)

    for name in D1_FUNCTIONS:
        patch(d1, name, getattr(old_d1, name))
    for name in D2_FUNCTIONS:
        patch(d2, name, getattr(old_d2, name))
    try:
        yield MISMATCHES
    finally:
        for module, name, current in reversed(saved):
            setattr(module, name, current)


FROZEN_SUITES = ('test_checkpoint_identity', 'test_checkpoint_payload', 'test_checkpoint_storage',
                 'test_checkpoint_storage_equivalence',
                 'test_checkpoint_cuda', 'test_checkpoint_real_data')


def main():
    """Run the frozen checkpoint suites with every D1/D2 call dual-run; print mismatches."""
    import unittest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(ROOT))
    with dual() as mismatches:
        suite = unittest.defaultTestLoader.loadTestsFromNames(FROZEN_SUITES)
        class Result(unittest.TextTestResult):
            def startTest(self, test):
                CURRENT_TEST[0] = test.id()
                super().startTest(test)
        result = unittest.TextTestRunner(verbosity=0, resultclass=Result).run(suite)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'mismatches': len(mismatches),
              'mismatch_detail': [repr(m)[:900] for m in mismatches[:50]]}
    print(json.dumps(report, indent=1))
    return 0 if result.wasSuccessful() and not mismatches else 1


if __name__ == '__main__':
    sys.exit(main())
