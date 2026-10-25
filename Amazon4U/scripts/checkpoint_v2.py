"""Checkpoint identity v2: checkpoints store fingerprints, not the training-data copy.

Plan: docs/checkpoint-identity-v2-plan.md (agreed by the user, 2026-10-25). D1, D2 and D3
v1 are unchanged; v2 sits beside them, and each version refuses the other's files.

- The identity in a v2 payload is a reference: the D1 header, the five D1 fingerprints and
  the resolved settings (`identity_reference`). It holds no records, histories, graph or
  metadata.
- At validation, the caller passes a trusted `CanonicalResumeIdentity` built in this process
  from hash-verified data. The reference must match it field for field, with its settings
  and identity hash recomputed; the trusted identity's content then serves every check that
  reads content (sampler domain, empty-pool pairs, row order).
- Every other payload check is D2 v1's. `_validate_semantic_payload_v2` is derived from v1's
  function by the substitutions in tests/checkpoint_v2_derivation.py, and a test keeps the
  two in step.
- The cuda-v2 profile and publish/retrieve wrap these exactly as GR06 wraps v1: same device
  section checks, same D3 encoding, root anchoring and exclusive link.
"""

import copy
import hashlib
import os
from typing import Any, Dict, Tuple

import torch

from scripts import checkpoint_storage as d3
from scripts.checkpoint_identity import (
    CanonicalResumeIdentity, CheckpointError, CheckpointIdentityMismatchError,
    CheckpointValidationError, _canonical_json_bytes, _validate_identity_header,
    compute_settings_fingerprint,
)
from scripts.checkpoint_payload import (
    ADAM_EXPECTED_BETAS, ADAM_EXPECTED_EPS, ADAM_EXPECTED_LR, ADAM_EXPECTED_WEIGHT_DECAY,
    MAX_EPOCHS_LIMIT, MAX_PATIENCE_CHECKS, NODE_TYPES, PAYLOAD_CAPABILITY, PAYLOAD_REQUIRED_KEYS,
    RELATION_SCHEMAS, SUPPORTED_BATCH_POLICIES, SUPPORTED_STEP_DTYPES, _check_strict_int, _safe_repr,
    _make_owned_output_copy, _safe_sort_keys, _validate_adam_optimizer,
    _validate_expected_model_schema, _validate_expected_runtime_schema,
    _validate_model_state_dict, _validate_rng_state_dict, _validate_sampler_and_progress,
    _validate_selection_and_best_model, preflight_resource_accounting, validate_resource_policy,
    validate_source_provenance,
)
from scripts.checkpoint_payload_cuda import CAPABILITY, DEVICES, V1_KEYS, _validate_device_section

IDENTITY_SCHEMA_V2 = 'amazon4u-checkpoint-identity-v2'
PAYLOAD_SCHEMA_VERSION_V2 = 'amazon4u-checkpoint-payload-v2'
CUDA_SCHEMA_V2 = 'amazon4u-checkpoint-payload-cuda-v2'
HEADER = ('method', 'category', 'feedback_condition', 'seed')
FINGERPRINTS = ('data_fingerprint', 'graph_fingerprint', 'validation_fingerprint',
                'settings_fingerprint', 'identity_hash')
REFERENCE_KEYS = frozenset({'schema_version', *HEADER, *FINGERPRINTS, 'resolved_settings'})


def identity_reference(identity):
    """The v2 identity stored in a checkpoint: header, fingerprints and settings."""
    if type(identity) is not CanonicalResumeIdentity:
        raise CheckpointValidationError('identity_reference needs a CanonicalResumeIdentity')
    return {'schema_version': IDENTITY_SCHEMA_V2,
            **{k: getattr(identity, k) for k in (*HEADER, *FINGERPRINTS)},
            'resolved_settings': copy.deepcopy(identity.resolved_settings)}


def validate_identity_reference(raw, trusted):
    """Check a stored v2 reference against the trusted identity; return the trusted one."""
    if type(trusted) is not CanonicalResumeIdentity:
        raise CheckpointValidationError(
            f'expected_identity must be a trusted CanonicalResumeIdentity, got {type(trusted).__name__}')
    if type(raw) is not dict:
        raise CheckpointValidationError('payload.identity must be a dict')
    if any(type(k) is not str for k in raw) or set(raw) != REFERENCE_KEYS:
        raise CheckpointValidationError(
            f'payload.identity keys must be exactly {sorted(REFERENCE_KEYS)}, got {_safe_sort_keys(raw)}')
    if raw['schema_version'] != IDENTITY_SCHEMA_V2:
        raise CheckpointValidationError(
            f'payload.identity.schema_version {_safe_repr(raw["schema_version"])} != {IDENTITY_SCHEMA_V2!r}')
    try:
        _validate_identity_header(*(raw[k] for k in HEADER), {k: raw[k] for k in FINGERPRINTS},
                                  role='payload.identity')
        if type(raw['resolved_settings']) is not dict:
            raise CheckpointValidationError('payload.identity.resolved_settings must be a dict')
        if compute_settings_fingerprint(raw['resolved_settings']) != raw['settings_fingerprint']:
            raise CheckpointIdentityMismatchError(
                'Falsified or corrupt settings_fingerprint in payload.identity')
        combined = {'schema': 'amazon4u-canonical-identity-v1',
                    **{k: raw[k] for k in (*HEADER, *FINGERPRINTS[:4])}}
        if hashlib.sha256(_canonical_json_bytes(combined, role='identity hash')).hexdigest() != raw['identity_hash']:
            raise CheckpointIdentityMismatchError('Falsified or corrupt identity_hash in payload.identity')
    except CheckpointError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        raise CheckpointValidationError(f'Malformed payload.identity: {exc}') from exc
    # resolved_settings is bound by settings_fingerprint, recomputed above.
    mismatches = [k for k in (*HEADER, *FINGERPRINTS)
                  if raw[k] != getattr(trusted, k) or type(raw[k]) is not type(getattr(trusted, k))]
    if mismatches:
        raise CheckpointIdentityMismatchError(
            'Checkpoint identity differs from the trusted data: ' + ', '.join(mismatches))
    return trusted


# DERIVED from checkpoint_payload._validate_semantic_payload; see tests/checkpoint_v2_derivation.py
def _validate_semantic_payload_v2(
    payload: Dict[str, Any],
    *,
    expected_identity: CanonicalResumeIdentity,
    expected_provenance: Dict[str, Any],
    expected_model: Dict[str, Any],
    expected_runtime: Dict[str, Any],
) -> None:
    """Validate all payload envelope, identity, provenance, model, optimizer, rng, sampler, progress, and selection fields."""
    if type(payload) is not dict:
        raise CheckpointValidationError(f"payload must be an exact dict, got {_safe_repr(payload)}")
    if set(payload.keys()) != set(PAYLOAD_REQUIRED_KEYS):
        missing = _safe_sort_keys(set(PAYLOAD_REQUIRED_KEYS) - set(payload.keys()))
        extra = _safe_sort_keys(set(payload.keys()) - set(PAYLOAD_REQUIRED_KEYS))
        raise CheckpointValidationError(f"payload keys mismatch. Missing: {missing}, Extra: {extra}")

    if payload["schema_version"] != PAYLOAD_SCHEMA_VERSION_V2:
        raise CheckpointValidationError(
            f"payload.schema_version mismatch: {_safe_repr(payload['schema_version'])} != {PAYLOAD_SCHEMA_VERSION_V2!r}"
        )
    if payload["capability"] != PAYLOAD_CAPABILITY:
        raise CheckpointValidationError(
            f"payload.capability mismatch: {_safe_repr(payload['capability'])} != {PAYLOAD_CAPABILITY!r}"
        )

    # Inapplicable framework components (scheduler, amp)
    sch = payload["scheduler"]
    if type(sch) is not dict or set(sch.keys()) != {"applicable", "state", "reason"}:
        raise CheckpointValidationError("payload.scheduler keys must be exactly applicable, state, reason")
    if sch["applicable"] is not False or sch["state"] is not None or sch["reason"] != "not_used_constant_learning_rate":
        raise CheckpointValidationError(f"payload.scheduler must be inactive constant learning rate record: {_safe_repr(sch)}")

    amp = payload["amp"]
    if type(amp) is not dict or set(amp.keys()) != {"applicable", "state", "reason"}:
        raise CheckpointValidationError("payload.amp keys must be exactly applicable, state, reason")
    if amp["applicable"] is not False or amp["state"] is not None or amp["reason"] != "not_used_cpu_float32":
        raise CheckpointValidationError(f"payload.amp must be inactive CPU float32 record: {_safe_repr(amp)}")

    # Validate expected descriptors schemas
    _validate_expected_model_schema(expected_model)
    _validate_expected_runtime_schema(expected_runtime)
    exp_method = expected_model["method"]
    exp_row_order = expected_model["row_order"]
    exp_buffer_state = expected_model["buffer_state"]

    # Identity v2: the stored reference must match the caller's trusted identity. The
    # trusted identity's content serves every content check below.
    actual_identity = validate_identity_reference(payload["identity"], expected_identity)
    validate_source_provenance(payload["provenance"], expected_provenance)

    if exp_method != actual_identity.method:
        raise CheckpointValidationError(
            f"expected_model.method {exp_method!r} != identity method {actual_identity.method!r}"
        )

    # Bind resolved settings coherently
    res_model = actual_identity.resolved_settings.get("model", {})
    if res_model.get("architecture") != actual_identity.method:
        raise CheckpointValidationError(f"Identity resolved architecture {res_model.get('architecture')!r} != method {actual_identity.method!r}")
    if res_model.get("embedding_dim") != 64:
        raise CheckpointValidationError(f"Identity resolved embedding_dim {res_model.get('embedding_dim')} != 64")
    exp_layers = 3 if exp_method == "LightGCN" else 2
    if res_model.get("num_layers") != exp_layers:
        raise CheckpointValidationError(f"Identity resolved num_layers {res_model.get('num_layers')} != {exp_layers}")

    res_train = actual_identity.resolved_settings.get("training", {})
    if res_train.get("optimizer") != "Adam":
        raise CheckpointValidationError("Identity resolved optimizer must be Adam")
    if res_train.get("learning_rate") != ADAM_EXPECTED_LR:
        raise CheckpointValidationError(f"Identity resolved lr {res_train.get('learning_rate')} != {ADAM_EXPECTED_LR}")
    if res_train.get("weight_decay") != ADAM_EXPECTED_WEIGHT_DECAY:
        raise CheckpointValidationError("Identity resolved weight_decay must be 0.0")
    if tuple(res_train.get("adam_betas", ())) != ADAM_EXPECTED_BETAS:
        raise CheckpointValidationError(f"Identity resolved betas {res_train.get('adam_betas')} != {ADAM_EXPECTED_BETAS}")
    if res_train.get("adam_epsilon") != ADAM_EXPECTED_EPS:
        raise CheckpointValidationError(f"Identity resolved eps {res_train.get('adam_epsilon')} != {ADAM_EXPECTED_EPS}")
    if res_train.get("max_epochs") != MAX_EPOCHS_LIMIT:
        raise CheckpointValidationError(f"Identity resolved max_epochs {res_train.get('max_epochs')} != {MAX_EPOCHS_LIMIT}")
    if res_train.get("patience") != MAX_PATIENCE_CHECKS:
        raise CheckpointValidationError(f"Identity resolved patience {res_train.get('patience')} != {MAX_PATIENCE_CHECKS}")

    # Validate row order membership against retained identity
    validated_param_shapes: Dict[str, Tuple[int, ...]] = {}

    if exp_method == "LightGCN":
        if set(exp_row_order.keys()) != {"User", "Product"}:
            raise CheckpointValidationError("LightGCN expected_model.row_order must contain User and Product")
        u_list = exp_row_order["User"]
        p_list = exp_row_order["Product"]
        if sorted(u_list) != sorted(actual_identity.positive_history.keys()):
            raise CheckpointValidationError("LightGCN row_order['User'] does not match identity users")
        if sorted(p_list) != sorted(actual_identity.warm_items):
            raise CheckpointValidationError("LightGCN row_order['Product'] does not match identity warm_items")
        validated_param_shapes["user_embeddings"] = (len(u_list), 64)
        validated_param_shapes["item_embeddings"] = (len(p_list), 64)

    elif exp_method == "Heterogeneous KG+GNN":
        if set(exp_row_order.keys()) != set(NODE_TYPES):
            raise CheckpointValidationError(f"HeteroGNN expected_model.row_order must contain all node types: {NODE_TYPES}")
        for nt in NODE_TYPES:
            nt_list = exp_row_order[nt]
            if sorted(nt_list) != sorted(actual_identity.graph_nodes.get(nt, ())):
                raise CheckpointValidationError(f"HeteroGNN row_order[{nt!r}] does not match identity graph_nodes")

        validated_param_shapes["user_embeddings"] = (len(exp_row_order["User"]), 64)
        validated_param_shapes["product_embeddings"] = (len(exp_row_order["Product"]), 64)
        validated_param_shapes["category_embeddings"] = (len(exp_row_order["CategoryPath"]), 64)
        validated_param_shapes["brand_embeddings"] = (len(exp_row_order["Brand"]), 64)
        for rel in RELATION_SCHEMAS:
            validated_param_shapes[f"relation_weights_l0.{rel}"] = (64, 64)
            validated_param_shapes[f"relation_weights_l1.{rel}"] = (64, 64)
        for nt in NODE_TYPES:
            validated_param_shapes[f"self_weights_l0.{nt}"] = (64, 64)
            validated_param_shapes[f"self_biases_l0.{nt}"] = (64,)
            validated_param_shapes[f"self_weights_l1.{nt}"] = (64, 64)
            validated_param_shapes[f"self_biases_l1.{nt}"] = (64,)

    # Runtime profile validation
    rt_profile = payload["runtime_profile"]
    if type(rt_profile) is not dict:
        raise CheckpointValidationError("payload.runtime_profile must be a dict")
    expected_rt_keys = {"profile_id", "python_version", "numpy_version", "torch_version", "adam_step_dtype", "batch_policy"}
    if set(rt_profile.keys()) != expected_rt_keys:
        raise CheckpointValidationError(f"runtime_profile keys mismatch: {_safe_sort_keys(rt_profile)}")

    exp_rt_profile = expected_runtime["runtime_profile"]
    if type(exp_rt_profile) is not dict or set(exp_rt_profile.keys()) != expected_rt_keys:
        raise CheckpointValidationError("expected_runtime.runtime_profile malformed")

    for k in expected_rt_keys:
        if rt_profile[k] != exp_rt_profile[k]:
            raise CheckpointValidationError(f"runtime_profile.{k} {_safe_repr(rt_profile[k])} != expected {_safe_repr(exp_rt_profile[k])}")

    adam_step_dtype = rt_profile["adam_step_dtype"]
    if adam_step_dtype not in SUPPORTED_STEP_DTYPES:
        raise CheckpointValidationError(f"Unsupported adam_step_dtype: {_safe_repr(adam_step_dtype)}")

    # Non-fused Adam scalar initialization policy: float64 iff default is float64, otherwise float32
    expected_step_dtype = "torch.float64" if torch.get_default_dtype() == torch.float64 else "torch.float32"
    if adam_step_dtype != expected_step_dtype:
        raise CheckpointValidationError(
            f"adam_step_dtype {_safe_repr(adam_step_dtype)} does not match PyTorch default scalar policy {_safe_repr(expected_step_dtype)}"
        )

    batch_policy = rt_profile["batch_policy"]
    if batch_policy not in SUPPORTED_BATCH_POLICIES:
        raise CheckpointValidationError(f"Unsupported batch_policy: {_safe_repr(batch_policy)}")

    # Compare versions against provenance environment records
    env_rec = payload["provenance"]["environment"]
    if rt_profile["python_version"] != env_rec["python_version"]:
        raise CheckpointValidationError(
            f"runtime_profile python_version {_safe_repr(rt_profile['python_version'])} != provenance {_safe_repr(env_rec['python_version'])}"
        )
    if rt_profile["torch_version"] != env_rec["torch_version"]:
        raise CheckpointValidationError(
            f"runtime_profile torch_version {_safe_repr(rt_profile['torch_version'])} != provenance {_safe_repr(env_rec['torch_version'])}"
        )
    lib_vers = env_rec.get("library_versions", {})
    if "numpy" in lib_vers and rt_profile["numpy_version"] != lib_vers["numpy"]:
        raise CheckpointValidationError(
            f"runtime_profile numpy_version {_safe_repr(rt_profile['numpy_version'])} != provenance {_safe_repr(lib_vers['numpy'])}"
        )

    # Sampler & Progress validation
    P, K = _validate_sampler_and_progress(payload["sampler"], payload["progress"], actual_identity, expected_runtime)

    # Current model validation
    cur_model = payload["current_model"]
    if type(cur_model) is not dict or set(cur_model.keys()) != {"epoch", "row_order", "state_dict"}:
        raise CheckpointValidationError("current_model must be an exact dict with epoch, row_order, state_dict")
    cm_epoch = _check_strict_int(cur_model["epoch"], "current_model.epoch")
    if cm_epoch != payload["progress"]["epoch"]:
        raise CheckpointValidationError(
            f"current_model.epoch ({_safe_repr(cm_epoch)}) != progress.epoch ({_safe_repr(payload['progress']['epoch'])})"
        )
    if cur_model["row_order"] != exp_row_order:
        raise CheckpointValidationError("current_model.row_order does not match expected_model.row_order")

    _validate_model_state_dict(
        cur_model["state_dict"],
        exp_method,
        exp_row_order,
        exp_buffer_state,
        role="current_model",
    )

    # Selection history & Best model snapshot validation
    _validate_selection_and_best_model(
        payload["selection"],
        payload["best_model"],
        cur_model,
        payload["progress"],
        actual_identity,
        expected_model,
    )

    # Adam optimizer validation with exact parameter shapes
    _validate_adam_optimizer(
        payload["optimizer"],
        expected_runtime["optimizer_groups"],
        adam_step_dtype,
        payload["progress"]["global_step"],
        exp_method,
        validated_param_shapes=validated_param_shapes,
    )

    # RNG roles and codecs validation
    _validate_rng_state_dict(payload["rng"], actual_identity, expected_runtime)
# END DERIVED


def validate_checkpoint_payload_v2(payload, *, expected_identity, expected_provenance,
                                   expected_model, expected_runtime, resource_policy):
    """D2 v1's public validation for a v2 payload; returns an owned validated snapshot.

    The order is v1's. `expected_identity` is the trusted in-process identity object, so it
    is type-checked instead of walked by the resource preflight.
    """
    parsed_policy = validate_resource_policy(resource_policy)
    preflight_resource_accounting(payload, parsed_policy, role="payload")
    if type(expected_identity) is not CanonicalResumeIdentity:
        raise CheckpointValidationError(
            f"expected_identity must be a trusted CanonicalResumeIdentity, got {_safe_repr(expected_identity)}")
    preflight_resource_accounting(expected_provenance, parsed_policy, role="expected_provenance")
    preflight_resource_accounting(expected_model, parsed_policy, role="expected_model")
    preflight_resource_accounting(expected_runtime, parsed_policy, role="expected_runtime")
    if type(expected_provenance) is not dict:
        raise CheckpointValidationError(
            f"expected_provenance must be an exact dict, got {_safe_repr(expected_provenance)}")
    _validate_expected_model_schema(expected_model)
    _validate_expected_runtime_schema(expected_runtime)
    expectations = dict(expected_identity=expected_identity, expected_provenance=expected_provenance,
                        expected_model=expected_model, expected_runtime=expected_runtime)
    _validate_semantic_payload_v2(payload, **expectations)
    owned_snapshot = _make_owned_output_copy(payload)
    preflight_resource_accounting(owned_snapshot, parsed_policy, role="owned_snapshot")
    _validate_semantic_payload_v2(owned_snapshot, **expectations)
    return owned_snapshot


def _fail(message):
    raise CheckpointValidationError(message)


def core_view_v2(payload):
    """The D2 v2 payload this cuda-v2 payload claims to contain."""
    core = {k: v for k, v in payload.items() if k != 'device'}
    core['schema_version'] = PAYLOAD_SCHEMA_VERSION_V2
    core['capability'] = PAYLOAD_CAPABILITY
    core['amp'] = {**payload['amp'], 'reason': 'not_used_cpu_float32'}
    return core


def validate_cuda_checkpoint_payload_v2(payload, *, expected_device, expected_identity,
                                        expected_provenance, expected_model, expected_runtime,
                                        resource_policy):
    """GR06's cuda validation, for cuda-v2 payloads and the v2 core validator."""
    if type(payload) is not dict or set(payload) != set(V1_KEYS) | {'device'}:
        _fail('cuda-v2 payload keys must be the v2 keys plus device')
    if payload['schema_version'] != CUDA_SCHEMA_V2 or payload['capability'] != CAPABILITY:
        _fail('not an amazon4u-checkpoint-payload-cuda-v2 epoch-boundary payload')
    amp_reason = _validate_device_section(payload['device'], expected_device)
    amp = payload['amp']
    if type(amp) is not dict or amp.get('reason') != amp_reason:
        _fail(f'payload.amp.reason must be {amp_reason!r} for the recorded device')
    owned_core = validate_checkpoint_payload_v2(
        core_view_v2(payload), expected_identity=expected_identity,
        expected_provenance=expected_provenance, expected_model=expected_model,
        expected_runtime=expected_runtime, resource_policy=resource_policy)
    owned = dict(owned_core)
    owned['schema_version'] = CUDA_SCHEMA_V2
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


def build_cuda_payload_v2(core, *, device_record):
    """Wrap a D2 v2 core with the device profile. Validation happens separately."""
    payload = copy.deepcopy(core)
    if payload.get('schema_version') != PAYLOAD_SCHEMA_VERSION_V2:
        _fail('core must be a v2 payload')
    payload['schema_version'] = CUDA_SCHEMA_V2
    payload['capability'] = CAPABILITY
    payload['amp'] = {**payload['amp'], 'reason': DEVICES[device_record['fingerprint']['device']]}
    payload['device'] = copy.deepcopy(device_record)
    return payload


def publish_checkpoint_v2(root_fd, name, payload, *, expected_device, expected_identity,
                          expected_provenance, expected_model, expected_runtime,
                          resource_policy, io_policy):
    """GR06's publish with the v2 validator: validate, encode, exclusive link (D3)."""
    p = d3._policy(io_policy)
    name = d3._name(name)
    with d3._os_errors(), d3._Root(root_fd) as root:
        owned = validate_cuda_checkpoint_payload_v2(
            payload, expected_device=expected_device, expected_identity=expected_identity,
            expected_provenance=expected_provenance, expected_model=expected_model,
            expected_runtime=expected_runtime, resource_policy=resource_policy)
        data = d3._encode(owned, p)
        fd = d3._anonymous_temp(root.fd)
        try:
            d3._write_all(fd, data)
            if os.fstat(fd).st_size != len(data):
                raise d3.CheckpointSecurityError("incomplete checkpoint write")
            os.fsync(fd)
            d3._directory(os.fstat(root.fd))
            d3._link_fd(fd, root.fd, name)
            return name
        finally:
            os.close(fd)


def retrieve_checkpoint_v2(root_fd, name, *, expected_device, expected_identity,
                           expected_provenance, expected_model, expected_runtime,
                           resource_policy, io_policy):
    """GR06's retrieve with the v2 validator: bounded read and decode, then validate (D3)."""
    p = d3._policy(io_policy)
    name = d3._name(name)
    with d3._os_errors(), d3._Root(root_fd) as root, d3._Anchor(root) as anchor:
        fd, size = anchor.file(root.fd, name, p["max_file_bytes"])
        anchor.check()
        data = d3._read_fd(fd, size)
        anchor.check()
        decoded = d3._decode(data, p)
        result = validate_cuda_checkpoint_payload_v2(
            decoded, expected_device=expected_device, expected_identity=expected_identity,
            expected_provenance=expected_provenance, expected_model=expected_model,
            expected_runtime=expected_runtime, resource_policy=resource_policy)
        anchor.check()
        return result
