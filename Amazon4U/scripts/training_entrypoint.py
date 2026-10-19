"""Explicit H-side callable entrypoint; no CLI, host capture or transport runner."""
import copy

from scripts.training_data import HarnessError, TrainingData, preflight
from scripts.training_harness import (
    CPUBackend, Harness, READINESS_GATES, baseline_validation,
    check_envelope, verify_bundle,
)

SPEC_KEYS = {
    'schema_version', 'run_id', 'approval_id', 'stage', 'mode', 'data_kind',
    'method', 'device', 'accelerator', 'profile_id', 'epochs', 'max_adam_steps',
}
METHODS = {'LightGCN', 'Heterogeneous KG+GNN', 'CorePopularity', 'KG-only'}


def validate_run_spec(actual, *, expected_spec, authorization, resource_policy, backend):
    for role, value in [('actual run spec', actual), ('expected run spec', expected_spec),
                        ('authorization', authorization)]:
        preflight(value, resource_policy, role)
    if (type(actual) is not dict or type(expected_spec) is not dict
            or set(actual) != SPEC_KEYS or set(expected_spec) != SPEC_KEYS):
        raise HarnessError('run spec fields are not exactly the H allowlist')
    if actual != expected_spec:
        raise HarnessError('run spec differs from independent approved specification')
    if (type(authorization) is not dict or set(authorization) != {
            'schema_version', 'run_id', 'approval_id', 'scope', 'max_adam_steps'}):
        raise HarnessError('explicit independent execution authorization is required')
    if (authorization['schema_version'] != 'amazon4u-H-authorization-v1'
            or authorization['scope'] != 'tiny_synthetic_cpu_backward_and_private_restore'
            or authorization['run_id'] != actual['run_id']
            or authorization['approval_id'] != actual['approval_id']
            or type(authorization['max_adam_steps']) is not int
            or not 0 <= authorization['max_adam_steps'] <= 4):
        raise HarnessError('authorization does not approve this bounded fixture run')
    if (actual['schema_version'] != 'amazon4u-H-spec-v1'
            or actual['stage'] != 'training_from_scratch'
            or actual['mode'] != 'fixture_verification'
            or actual['data_kind'] != 'owned_synthetic_packet'
            or actual['profile_id'] != 'H-REF-synthetic-cpu-v1'
            or actual['method'] not in METHODS):
        raise HarnessError('real data, profiling, LM stages and final test are not authorized')
    for key in SPEC_KEYS - {'epochs', 'max_adam_steps'}:
        if type(actual[key]) is not str or not actual[key] or len(actual[key]) > 256:
            raise HarnessError('run spec strings must be bounded and nonempty')
    if (type(actual['epochs']) is not int or not 0 <= actual['epochs'] <= 4
            or type(actual['max_adam_steps']) is not int
            or not 0 <= actual['max_adam_steps'] <= authorization['max_adam_steps']):
        raise HarnessError('invalid fixture epoch/update budget')
    baseline = actual['method'] in {'CorePopularity', 'KG-only'}
    if baseline:
        if actual['epochs'] != 0 or actual['max_adam_steps'] != 0:
            raise HarnessError('deterministic baselines cannot request neural updates')
    elif actual['epochs'] < 1 or actual['max_adam_steps'] < 1:
        raise HarnessError('learned fixture run must explicitly budget actual updates')
    backend.require(actual['device'], actual['accelerator'])
    return copy.deepcopy(actual)


def execute_fixture(spec, *, expected_spec, authorization, data, expectations,
                    run_envelope, expected_run_envelope, bundle_sources, expected_bundle,
                    light_initial=None, backend=None):
    """Return a versioned result or propagate a failure; never manufacture success.

    Data, source packets, policies and authorization are supplied independently by
    the trusted caller. There are no filenames, test labels, credentials or mounts
    in the run spec. This function does not publish artifacts or launch processes.
    """
    backend = CPUBackend() if backend is None else backend
    if type(expectations) is not dict or 'resource_policy' not in expectations:
        raise HarnessError('independent resource expectations are mandatory')
    approved = validate_run_spec(
        spec, expected_spec=expected_spec, authorization=authorization,
        resource_policy=expectations['resource_policy'], backend=backend)
    if type(data) is not TrainingData:
        raise HarnessError('bounded owned fixture data required')
    envelope = check_envelope(
        run_envelope, expected_run_envelope, expectations['resource_policy'])
    for key in ('run_id', 'approval_id', 'method', 'device', 'accelerator', 'profile_id'):
        if envelope[key] != approved[key]:
            raise HarnessError('approved specification/run envelope mismatch')
    if verify_bundle(bundle_sources, expected_bundle) != envelope['bundle_sha256']:
        raise HarnessError('approved run/source closure mismatch')
    if envelope['identity_hash'] != data.identity.identity_hash:
        raise HarnessError('approved run/input identity mismatch')
    if approved['method'] in {'CorePopularity', 'KG-only'}:
        backend.verify_runtime(expectations['expected_runtime'])
        owned_data = data.owned_copy(expected_identity=expectations['expected_identity'])
        if light_initial is not None:
            raise HarnessError('baseline must not receive neural initialization tensors')
        outcome = baseline_validation(owned_data, approved['method'])
        epoch, steps, selection = 0, 0, None
        report, mixture = outcome['validation_report'], outcome['mixture']
        status = 'deterministic_validation_complete'
    else:
        harness = Harness(
            data, expectations, backend=backend, requested_device=approved['device'],
            accelerator=approved['accelerator'], run_envelope=envelope,
            expected_run_envelope=expected_run_envelope, bundle_sources=bundle_sources,
            expected_bundle=expected_bundle, light_initial=light_initial,
            max_adam_steps=approved['max_adam_steps'])
        context = harness.data.context
        total = sum(bool(context.warm_items - context.all_rated_history[u])
                    for u, _ in harness.data.positive_pairs)
        size = harness.data.identity.resolved_settings['training']['batch_size']
        if approved['epochs'] * ((total + size - 1) // size) > approved['max_adam_steps']:
            raise HarnessError('complete requested fixture run exceeds its approved Adam budget')
        for _ in range(approved['epochs']):
            if harness.selection.state['status'] == 'stopped':
                break
            harness.train_epoch()
        epoch, steps = harness.progress['epoch'], harness.progress['global_step']
        selection = copy.deepcopy(harness.selection.state)
        report = harness.selected_validation() if harness.best_model is not None else None
        mixture = None
        status = ('scientific_terminal' if selection['status'] == 'stopped'
                  else 'fixture_budget_complete')
    return {
        'schema_version': 'amazon4u-H-result-v1',
        'status': status, 'acceptance_scope': 'H-REF_CPU_reference_fixture_only',
        'deployment_decision': 'NOT_AUTHORIZED', 'run_spec': approved,
        'run_envelope': envelope, 'device_capability': backend.capabilities(),
        'method': approved['method'], 'epoch': epoch, 'actual_Adam_steps': steps,
        'selection': selection, 'mixture': mixture,
        'validation_report': copy.deepcopy(report),
        'remaining_readiness_gates': list(READINESS_GATES),
        'test_execution': 'not_authorized',
    }


def main():
    raise HarnessError('CLI/host input acquisition and Package O transport are not authorized by H-REF')
