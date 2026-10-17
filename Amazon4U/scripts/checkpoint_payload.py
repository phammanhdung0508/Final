"""Amazon4U milestone I1: pure checkpoint payload and state validation contracts.

Provides:
- validate_checkpoint_payload: public entry point for validating decoded in-memory
  checkpoint payloads against mandatory expected identity, provenance, model,
  runtime, and resource policy descriptors.
- Resource preflight and accounting according to the frozen technical contract.
- Complete LightGCN and HeteroGNN model/tensor schema, device, layout, and content validation.
- Adam optimizer state, group, moment, and lazy initialization validation.
- RNG roles, shared generators, and native state codecs validation.
- Sampler domain, condition-positive pairs, batch policy, and progress validation.
- Selection history, strict improvement, patience, and best-model snapshot validation.
"""

from collections import defaultdict
import copy
import hashlib
import json
import math
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

# Canonicalize module in sys.modules so scripts.checkpoint_payload and checkpoint_payload
# share the exact same module object and class identities.
if __name__ == "scripts.checkpoint_payload":
    sys.modules.setdefault("checkpoint_payload", sys.modules[__name__])
elif __name__ == "checkpoint_payload":
    sys.modules.setdefault("scripts.checkpoint_payload", sys.modules[__name__])
    if "scripts" in sys.modules:
        setattr(sys.modules["scripts"], "checkpoint_payload", sys.modules[__name__])

try:
    import scripts as _scripts_pkg  # type: ignore
except ModuleNotFoundError:
    _scripts_pkg = None
if _scripts_pkg is not None:
    setattr(_scripts_pkg, "checkpoint_payload", sys.modules[__name__])

try:
    from checkpoint_identity import (
        CHECKPOINT_SCHEMA_VERSION,
        CONFIGURED_RNG_COMPONENTS,
        NODE_TYPES,
        RELATION_SCHEMAS,
        CanonicalResumeIdentity,
        CheckpointError,
        CheckpointIdentityMismatchError,
        CheckpointProvenanceError,
        CheckpointSecurityError,
        CheckpointValidationError,
        validate_canonical_identity,
        validate_source_provenance,
    )
except ModuleNotFoundError:
    from scripts.checkpoint_identity import (
        CHECKPOINT_SCHEMA_VERSION,
        CONFIGURED_RNG_COMPONENTS,
        NODE_TYPES,
        RELATION_SCHEMAS,
        CanonicalResumeIdentity,
        CheckpointError,
        CheckpointIdentityMismatchError,
        CheckpointProvenanceError,
        CheckpointSecurityError,
        CheckpointValidationError,
        validate_canonical_identity,
        validate_source_provenance,
    )


# ---------------------------------------------------------------------------
# Constants & Frozen Technical Contract Boundaries
# ---------------------------------------------------------------------------

PAYLOAD_SCHEMA_VERSION = "amazon4u-checkpoint-payload-v1"
PAYLOAD_CAPABILITY = "epoch_boundary_validation_only"

RESOURCE_POLICY_SCHEMA_VERSION = "amazon4u-payload-resources-v1"
SYNTHETIC_RESOURCE_PROFILE_ID = "amazon4u-payload-synthetic-v1"

PAYLOAD_REQUIRED_KEYS: Tuple[str, ...] = (
    "schema_version",
    "capability",
    "identity",
    "provenance",
    "runtime_profile",
    "current_model",
    "best_model",
    "optimizer",
    "rng",
    "sampler",
    "progress",
    "selection",
    "scheduler",
    "amp",
)

RUNTIME_POLICY_REQUIRED_KEYS: Set[str] = {
    "schema_version",
    "profile_id",
    "kind",
    "max_input_bytes",
    "max_output_bytes",
    "max_container_depth",
    "max_visited_values",
}

EXPECTED_MODEL_REQUIRED_KEYS: Set[str] = {"method", "row_order", "buffer_state"}

EXPECTED_RUNTIME_REQUIRED_KEYS: Set[str] = {
    "runtime_profile",
    "optimizer_groups",
    "rng_role_bindings",
    "rng_generator_bindings",
    "sampler_algorithm",
}

SUPPORTED_CODECS: Tuple[str, ...] = (
    "python.random.MT19937.v3",
    "numpy.random.RandomState.MT19937.v1",
    "numpy.random.PCG64.v1",
    "torch.cpu.Generator.native.v1",
)

SUPPORTED_METHODS: Tuple[str, ...] = ("LightGCN", "Heterogeneous KG+GNN")
SUPPORTED_BATCH_POLICIES: Tuple[str, ...] = ("filter_before_batch", "batch_then_filter")
SUPPORTED_STEP_DTYPES: Tuple[str, ...] = ("torch.float32", "torch.float64")

ADAM_EXPECTED_LR = 0.001
ADAM_EXPECTED_BETAS = (0.9, 0.999)
ADAM_EXPECTED_EPS = 1e-08
ADAM_EXPECTED_WEIGHT_DECAY = 0.0

SELECTION_METRIC = "validation_NDCG@10"
MAX_PATIENCE_CHECKS = 10
MAX_EPOCHS_LIMIT = 100


# ---------------------------------------------------------------------------
# Diagnostics & Safe Formatting Helpers
# ---------------------------------------------------------------------------

def _safe_type_name(t: Any) -> str:
    """Safely return type name without executing user metaclass hooks."""
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "str"
    if t is bytes:
        return "bytes"
    if t is bool:
        return "bool"
    if t is type(None):
        return "NoneType"
    if t is dict:
        return "dict"
    if t is list:
        return "list"
    if t is tuple:
        return "tuple"
    if t is set:
        return "set"
    if t is frozenset:
        return "frozenset"
    if t is torch.Tensor:
        return "torch.Tensor"
    return "<unsupported object>"


def _safe_sort_keys(keys: Iterable[Any]) -> List[str]:
    """Render a bounded key sample; sort only that sample, never all input keys.

    At most five source entries are inspected and four 192-character
    representations are retained. This helper is diagnostic only.
    """
    rendered: List[str] = []
    for index, key in enumerate(keys):
        if index == 4:
            rendered.append("<additional keys omitted>")
            break
        rendered.append(_safe_repr(key)[:192])
    return sorted(rendered)


def _safe_repr(val: Any) -> str:
    """Format a value safely without executing user hooks or triggering int conversion errors."""
    t = type(val)
    if t is bool:
        return "True" if val else "False"
    if t is int:
        if val.bit_length() > 256:
            return f"<large int ({val.bit_length()} bits)>"
        return str(val)
    if t is float:
        if math.isnan(val):
            return "nan"
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
        return str(val)
    if t is str:
        # Bound both inspected source text and escaped diagnostic output.
        sample = val[:97] + "..." if len(val) > 100 else val
        return repr(sample)[:160]
    if t is bytes:
        if len(val) > 50:
            return repr(val[:47] + b"...")
        return repr(val)
    if val is None:
        return "None"
    if t is dict:
        return f"<dict (len {len(val)})>"
    if t is list:
        return f"<list (len {len(val)})>"
    if t is tuple:
        return f"<tuple (len {len(val)})>"
    if t is torch.Tensor:
        return f"<Tensor ndim={val.dim()} dtype={val.dtype} device={val.device}>"
    return f"<{_safe_type_name(t)}>"


def _check_clean_text(val: Any, role: str) -> str:
    """Check that val is strictly an exact string without surrogate characters or unstripped whitespace."""
    if type(val) is not str:
        raise CheckpointValidationError(f"{role} must be a string, got {_safe_repr(val)}")
    try:
        val.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointValidationError(f"{role} contains invalid Unicode (surrogate characters): {exc}") from exc
    if not val or val.strip() != val:
        raise CheckpointValidationError(f"{role} must be a non-empty clean string, got {_safe_repr(val)}")
    return val


def _check_strict_int(val: Any, role: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Validate strict exact integer excluding bools, floats, and subclasses."""
    if type(val) is not int or type(val) is bool:
        raise CheckpointValidationError(f"{role} must be an integer (bool excluded), got {_safe_repr(val)}")
    if min_val is not None and val < min_val:
        raise CheckpointValidationError(f"{role} must be >= {_safe_repr(min_val)}, got {_safe_repr(val)}")
    if max_val is not None and val > max_val:
        raise CheckpointValidationError(f"{role} must be <= {_safe_repr(max_val)}, got {_safe_repr(val)}")
    return val


def _check_strict_bool(val: Any, role: str) -> bool:
    """Validate strict exact boolean."""
    if type(val) is not bool:
        raise CheckpointValidationError(f"{role} must be a strict boolean, got {_safe_repr(val)}")
    return val


def _check_strict_float(val: Any, role: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Validate strict finite float excluding bools, ints, and subclasses."""
    if type(val) is not float:
        raise CheckpointValidationError(f"{role} must be a float, got {_safe_repr(val)}")
    if not math.isfinite(val):
        raise CheckpointValidationError(f"{role} must be finite, got {_safe_repr(val)}")
    if min_val is not None and val < min_val:
        raise CheckpointValidationError(f"{role} must be >= {_safe_repr(min_val)}, got {_safe_repr(val)}")
    if max_val is not None and val > max_val:
        raise CheckpointValidationError(f"{role} must be <= {_safe_repr(max_val)}, got {_safe_repr(val)}")
    return val


def _check_strict_number(val: Any, role: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Union[int, float]:
    """Validate strict exact integer or float excluding bools and subclasses."""
    t = type(val)
    if (t is not int and t is not float) or t is bool:
        raise CheckpointValidationError(f"{role} must be a number (int or float, bool excluded), got {_safe_repr(val)}")
    if t is float:
        if not math.isfinite(val):
            raise CheckpointValidationError(f"{role} must be finite, got {_safe_repr(val)}")
    if min_val is not None and val < min_val:
        raise CheckpointValidationError(f"{role} must be >= {_safe_repr(min_val)}, got {_safe_repr(val)}")
    if max_val is not None and val > max_val:
        raise CheckpointValidationError(f"{role} must be <= {_safe_repr(max_val)}, got {_safe_repr(val)}")
    return val


def _ceil_div(a: int, b: int) -> int:
    """Exact integer ceil division for positive integers a and b."""
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Resource Policy & Content Accounting
# ---------------------------------------------------------------------------

def validate_resource_policy(policy: Any) -> Dict[str, Any]:
    """Validate structure, keys, types, and values of the trusted resource policy."""
    if type(policy) is not dict:
        raise CheckpointValidationError(f"resource_policy must be an exact dict, got {_safe_repr(policy)}")
    for k in policy.keys():
        if type(k) is not str:
            raise CheckpointValidationError(f"resource_policy keys must be strings, got {_safe_repr(k)}")

    policy_keys = set(policy.keys())
    if policy_keys != RUNTIME_POLICY_REQUIRED_KEYS:
        missing = _safe_sort_keys(RUNTIME_POLICY_REQUIRED_KEYS - policy_keys)
        extra = _safe_sort_keys(policy_keys - RUNTIME_POLICY_REQUIRED_KEYS)
        raise CheckpointValidationError(
            f"resource_policy keys mismatch. Missing: {missing}, Extra: {extra}"
        )

    schema = policy["schema_version"]
    if type(schema) is not str or schema != RESOURCE_POLICY_SCHEMA_VERSION:
        raise CheckpointValidationError(
            f"resource_policy.schema_version mismatch: {_safe_repr(schema)} != {RESOURCE_POLICY_SCHEMA_VERSION!r}"
        )

    profile_id = _check_clean_text(policy["profile_id"], "resource_policy.profile_id")
    kind = policy["kind"]
    if type(kind) is not str or kind not in ("synthetic", "runtime"):
        raise CheckpointValidationError(
            f"resource_policy.kind must be 'synthetic' or 'runtime', got {_safe_repr(kind)}"
        )

    max_in = _check_strict_int(policy["max_input_bytes"], "resource_policy.max_input_bytes", min_val=1)
    max_out = _check_strict_int(policy["max_output_bytes"], "resource_policy.max_output_bytes", min_val=1)
    max_depth = _check_strict_int(policy["max_container_depth"], "resource_policy.max_container_depth", min_val=1)
    max_visits = _check_strict_int(policy["max_visited_values"], "resource_policy.max_visited_values", min_val=1)

    if kind == "synthetic":
        if profile_id != SYNTHETIC_RESOURCE_PROFILE_ID:
            raise CheckpointValidationError(
                f"resource_policy.profile_id for synthetic kind must be {SYNTHETIC_RESOURCE_PROFILE_ID!r}, got {_safe_repr(profile_id)}"
            )
        if (max_in, max_out, max_depth, max_visits) != (2097152, 2097152, 32, 16384):
            raise CheckpointValidationError(
                f"Synthetic resource policy constants mismatch: expected (2097152, 2097152, 32, 16384), "
                f"got ({_safe_repr(max_in)}, {_safe_repr(max_out)}, {_safe_repr(max_depth)}, {_safe_repr(max_visits)})"
            )

    return {
        "schema_version": schema,
        "profile_id": profile_id,
        "kind": kind,
        "max_input_bytes": max_in,
        "max_output_bytes": max_out,
        "max_container_depth": max_depth,
        "max_visited_values": max_visits,
    }


def _tensor_descriptor(t: torch.Tensor) -> Tuple[Any, ...]:
    """Build exact technical descriptor tuple for dense or sparse COO tensor."""
    if t.layout == torch.strided:
        return (
            str(t.dtype),
            str(t.layout),
            str(t.device),
            tuple(t.shape),
            tuple(t.stride()),
            int(t.storage_offset()),
        )
    elif t.layout == torch.sparse_coo:
        if not t.is_coalesced():
            raise CheckpointValidationError("Sparse COO tensor must be coalesced")
        idx_desc = _tensor_descriptor(t.indices())
        val_desc = _tensor_descriptor(t.values())
        return (
            str(t.dtype),
            str(t.layout),
            str(t.device),
            tuple(t.shape),
            bool(t.is_coalesced()),
            idx_desc,
            val_desc,
        )
    else:
        raise CheckpointValidationError(f"Unsupported tensor layout: {t.layout}")


def preflight_resource_accounting(
    root_obj: Any,
    policy: Dict[str, Any],
    role: str = "payload",
) -> Tuple[int, int, int, int]:
    """Preflight and account for metadata charge M, distinct backing storage S, projected output C, container depth, and visits.

    Rejects custom objects, cycles, oversized containers, and unsupported structures before allocation.
    Returns (M, S, C, visited_values).
    """
    max_in = policy["max_input_bytes"]
    max_out = policy["max_output_bytes"]
    max_depth = policy["max_container_depth"]
    max_visits = policy["max_visited_values"]

    visited_count = 0
    metadata_bytes = 0
    input_storage_bytes = 0
    output_storage_bytes = 0
    seen_storage_ids: Set[Any] = set()
    kept_storages: List[Any] = []
    active_cycle_ids: Set[int] = set()

    def _traverse(node: Any, container_depth: int) -> None:
        nonlocal visited_count, metadata_bytes, input_storage_bytes, output_storage_bytes

        node_type = type(node)
        # 1. Reject unsupported types and subclasses immediately before calling any hooks
        if not (
            node_type is int
            or node_type is float
            or node_type is str
            or node_type is bytes
            or node_type is bool
            or node is None
            or node_type is dict
            or node_type is list
            or node_type is tuple
            or node_type is torch.Tensor
        ):
            raise CheckpointValidationError(
                f"{role} contains unsupported type or subclass {_safe_type_name(node_type)}"
            )

        visited_count += 1
        if visited_count > max_visits:
            raise CheckpointValidationError(
                f"{role} exceeded max_visited_values ({max_visits}): reached {visited_count}"
            )

        # Container depth check applies when node is a container
        if node_type is dict or node_type is list or node_type is tuple:
            if container_depth > max_depth:
                raise CheckpointValidationError(
                    f"{role} exceeded max_container_depth ({max_depth}): reached {container_depth}"
                )

        if node is None or node_type is bool:
            metadata_bytes += 1

        elif node_type is int:
            bit_len = node.bit_length()
            charge = 1 + max(1, (bit_len + 7) // 8)
            metadata_bytes += charge

        elif node_type is float:
            if not math.isfinite(node):
                raise CheckpointValidationError(f"{role} contains non-finite float: {_safe_repr(node)}")
            metadata_bytes += 8

        elif node_type is str:
            if len(node) > max_in:
                raise CheckpointValidationError(f"{role} contains oversized string exceeding input limit")
            try:
                encoded_bytes = len(node.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise CheckpointValidationError(f"{role} contains invalid Unicode (surrogate characters): {exc}") from exc
            metadata_bytes += 8 + encoded_bytes

        elif node_type is bytes:
            metadata_bytes += 8 + len(node)

        elif node_type is dict:
            node_id = id(node)
            if node_id in active_cycle_ids:
                raise CheckpointValidationError(f"{role} contains cyclic reference in dict")
            active_cycle_ids.add(node_id)
            try:
                metadata_bytes += 8
                # Check keys type safety before any set/hashing/sorting operations
                first_k_type = None
                for k in node.keys():
                    k_t = type(k)
                    if not (k_t is str or k_t is int or k_t is float or k_t is bytes or k_t is bool or k_t is tuple):
                        raise CheckpointValidationError(f"{role} dict key has unsupported type: {_safe_type_name(k_t)}")
                    if first_k_type is None:
                        first_k_type = k_t
                    elif k_t is not first_k_type:
                        raise CheckpointValidationError(
                            f"{role} dict contains mixed key types"
                        )
                for k, v in node.items():
                    k_t = type(k)
                    _traverse(k, container_depth + 1 if (k_t is dict or k_t is list or k_t is tuple) else container_depth)
                    v_t = type(v)
                    _traverse(v, container_depth + 1 if (v_t is dict or v_t is list or v_t is tuple) else container_depth)
            finally:
                active_cycle_ids.remove(node_id)

        elif node_type is list or node_type is tuple:
            node_id = id(node)
            if node_id in active_cycle_ids:
                t_name = "list" if node_type is list else "tuple"
                raise CheckpointValidationError(f"{role} contains cyclic reference in {t_name}")
            active_cycle_ids.add(node_id)
            try:
                metadata_bytes += 8
                for item in node:
                    item_t = type(item)
                    _traverse(item, container_depth + 1 if (item_t is dict or item_t is list or item_t is tuple) else container_depth)
            finally:
                active_cycle_ids.remove(node_id)

        elif node_type is torch.Tensor:
            if node.device.type != "cpu":
                raise CheckpointValidationError(
                    f"{role} tensor must be on CPU device, got {node.device.type}: {_safe_repr(node)}"
                )
            if node.layout not in (torch.strided, torch.sparse_coo):
                raise CheckpointValidationError(
                    f"{role} unsupported tensor layout {node.layout}: {_safe_repr(node)}"
                )

            # Account for tensor storage
            if node.layout == torch.strided:
                st = node.untyped_storage()
                kept_storages.append(st)
                st_id = getattr(st, "_cdata", id(st))
                nbytes = st.nbytes()
                if nbytes > 0 and st_id not in seen_storage_ids:
                    seen_storage_ids.add(st_id)
                    input_storage_bytes += nbytes

                output_storage_bytes += node.numel() * node.element_size()

            elif node.layout == torch.sparse_coo:
                if not node.is_coalesced():
                    raise CheckpointValidationError(f"{role} sparse COO tensor must be coalesced before preflight")
                idx_t = node.indices()
                val_t = node.values()

                for sub_t in (idx_t, val_t):
                    st = sub_t.untyped_storage()
                    kept_storages.append(st)
                    st_id = getattr(st, "_cdata", id(st))
                    nbytes = st.nbytes()
                    if nbytes > 0 and st_id not in seen_storage_ids:
                        seen_storage_ids.add(st_id)
                        input_storage_bytes += nbytes

                output_storage_bytes += (idx_t.numel() * idx_t.element_size()) + (val_t.numel() * val_t.element_size())

            # Tensor descriptor replaces tensor in metadata traversal
            # Descriptor tuple is a container at container_depth + 1
            desc_depth = container_depth + 1
            if desc_depth > max_depth:
                raise CheckpointValidationError(
                    f"{role} exceeded max_container_depth ({max_depth}): tensor descriptor reached {desc_depth}"
                )
            desc = _tensor_descriptor(node)
            metadata_bytes += 8
            for elem in desc:
                elem_t = type(elem)
                _traverse(elem, desc_depth + 1 if (elem_t is dict or elem_t is list or elem_t is tuple) else desc_depth)

        # Check bounds during traversal
        if metadata_bytes + input_storage_bytes > max_in:
            raise CheckpointValidationError(
                f"{role} input accounted bytes ({metadata_bytes + input_storage_bytes}) exceeded limit ({max_in})"
            )
        if metadata_bytes + output_storage_bytes > max_out:
            raise CheckpointValidationError(
                f"{role} projected output bytes ({metadata_bytes + output_storage_bytes}) exceeded limit ({max_out})"
            )

    root_t = type(root_obj)
    root_depth = 1 if (root_t is dict or root_t is list or root_t is tuple) else 0
    _traverse(root_obj, container_depth=root_depth)

    return (metadata_bytes, input_storage_bytes, output_storage_bytes, visited_count)


# ---------------------------------------------------------------------------
# Model State Validation (LightGCN & HeteroGNN)
# ---------------------------------------------------------------------------

def _get_hetero_expected_param_names() -> List[str]:
    """Return all 36 HeteroGNN parameter names in exact accepted registration order."""
    names: List[str] = [
        "user_embeddings",
        "product_embeddings",
        "category_embeddings",
        "brand_embeddings",
    ]
    for rel in RELATION_SCHEMAS:
        names.append(f"relation_weights_l0.{rel}")
    for rel in RELATION_SCHEMAS:
        names.append(f"relation_weights_l1.{rel}")
    for ntype in NODE_TYPES:
        names.append(f"self_weights_l0.{ntype}")
    for ntype in NODE_TYPES:
        names.append(f"self_biases_l0.{ntype}")
    for ntype in NODE_TYPES:
        names.append(f"self_weights_l1.{ntype}")
    for ntype in NODE_TYPES:
        names.append(f"self_biases_l1.{ntype}")
    return names


def _validate_tensor_properties(
    tensor: torch.Tensor,
    name: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: torch.dtype = torch.float32,
    expected_layout: torch.layout = torch.strided,
    role: str = "model",
) -> None:
    """Validate device, layout, dtype, shape, finiteness, and detachment of a tensor."""
    if type(tensor) is not torch.Tensor:
        raise CheckpointValidationError(f"{role} parameter/buffer {name!r} must be exact torch.Tensor, got {_safe_repr(tensor)}")
    if tensor.device.type != "cpu":
        raise CheckpointValidationError(f"{role} {name!r} must be on CPU, got {tensor.device.type}")
    if tensor.layout != expected_layout:
        raise CheckpointValidationError(f"{role} {name!r} layout mismatch: {tensor.layout} != {expected_layout}")
    if tensor.dtype != expected_dtype:
        raise CheckpointValidationError(f"{role} {name!r} dtype mismatch: {tensor.dtype} != {expected_dtype}")
    if tuple(tensor.shape) != expected_shape:
        raise CheckpointValidationError(f"{role} {name!r} shape mismatch: {tuple(tensor.shape)} != {expected_shape}")
    if tensor.grad_fn is not None or tensor.requires_grad:
        raise CheckpointValidationError(f"{role} {name!r} must be detached and not require grad")
    if expected_layout == torch.strided:
        if not torch.isfinite(tensor).all():
            raise CheckpointValidationError(f"{role} {name!r} contains non-finite values (NaN or Inf)")


def _validate_sparse_adj_buffer(
    adj: torch.Tensor,
    expected_adj: torch.Tensor,
    num_users: int,
    num_items: int,
    role: str = "current_model",
) -> None:
    """Validate LightGCN sparse COO adj_user_item buffer against trusted graph contents."""
    if type(adj) is not torch.Tensor:
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' must be exact torch.Tensor, got {_safe_repr(adj)}")
    if adj.layout != torch.sparse_coo:
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' must be sparse COO, got {adj.layout}")
    if adj.device.type != "cpu":
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' must be on CPU, got {adj.device.type}")
    if adj.dtype != torch.float32:
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' values must be float32, got {adj.dtype}")
    if tuple(adj.shape) != (num_users, num_items):
        raise CheckpointValidationError(
            f"{role} buffer 'adj_user_item' shape mismatch: {tuple(adj.shape)} != {(num_users, num_items)}"
        )
    if not adj.is_coalesced():
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' must be coalesced")
    if adj.grad_fn is not None or adj.requires_grad:
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' must be detached and not require grad")

    indices = adj.indices()
    values = adj.values()

    if indices.dtype != torch.int64:
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' indices must be int64, got {indices.dtype}")
    if indices.device.type != "cpu":
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' indices must be on CPU")
    if values.dtype != torch.float32:
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' values must be float32, got {values.dtype}")
    if not torch.isfinite(values).all():
        raise CheckpointValidationError(f"{role} buffer 'adj_user_item' contains non-finite values")

    nnz = values.numel()
    if indices.shape != (2, nnz):
        raise CheckpointValidationError(
            f"{role} buffer 'adj_user_item' indices shape mismatch: {tuple(indices.shape)} != (2, {nnz})"
        )

    # Validate canonical strictly ascending lexicographic order and uniqueness of indices
    if nnz > 0:
        row_idx = indices[0]
        col_idx = indices[1]

        # Check bounds
        if (row_idx < 0).any() or (row_idx >= num_users).any():
            raise CheckpointValidationError(f"{role} buffer 'adj_user_item' row indices out of range [0, {num_users - 1}]")
        if (col_idx < 0).any() or (col_idx >= num_items).any():
            raise CheckpointValidationError(f"{role} buffer 'adj_user_item' col indices out of range [0, {num_items - 1}]")

        for k in range(nnz - 1):
            r_k = int(row_idx[k].item())
            c_k = int(col_idx[k].item())
            r_next = int(row_idx[k + 1].item())
            c_next = int(col_idx[k + 1].item())
            if not (r_k < r_next or (r_k == r_next and c_k < c_next)):
                raise CheckpointValidationError(
                    f"{role} buffer 'adj_user_item' indices not in canonical strictly ascending order or contains duplicate index at index {k}"
                )

    # Match trusted expected_adj exactly without normalizing a bad descriptor into validity
    if type(expected_adj) is not torch.Tensor:
        raise CheckpointValidationError("expected_model buffer_state['adj_user_item'] must be exact torch.Tensor")
    if expected_adj.layout != torch.sparse_coo:
        raise CheckpointValidationError("expected_model buffer_state['adj_user_item'] must be sparse COO tensor")
    if expected_adj.device.type != "cpu":
        raise CheckpointValidationError("expected_model buffer_state['adj_user_item'] must be on CPU")
    if expected_adj.dtype != torch.float32:
        raise CheckpointValidationError("expected_model buffer_state['adj_user_item'] must have float32 values")
    if not expected_adj.is_coalesced():
        raise CheckpointValidationError("expected_model buffer_state['adj_user_item'] must be coalesced")
    if expected_adj.requires_grad or expected_adj.grad_fn is not None:
        raise CheckpointValidationError("expected_model buffer_state['adj_user_item'] must be detached")
    if tuple(expected_adj.shape) != tuple(adj.shape):
        raise CheckpointValidationError(
            f"{role} buffer 'adj_user_item' shape {tuple(adj.shape)} != expected {tuple(expected_adj.shape)}"
        )

    exp_indices = expected_adj.indices()
    exp_values = expected_adj.values()
    if not torch.equal(indices, exp_indices):
        raise CheckpointValidationError(
            f"{role} buffer 'adj_user_item' indices do not match trusted graph contents"
        )
    # Bitwise equality check on values
    if not (values.contiguous().view(torch.int32) == exp_values.contiguous().view(torch.int32)).all():
        raise CheckpointValidationError(
            f"{role} buffer 'adj_user_item' values do not match trusted graph contents"
        )


def _validate_model_state_dict(
    state_dict: Any,
    method: str,
    expected_row_order: Dict[str, List[str]],
    expected_buffer_state: Dict[str, Any],
    role: str = "current_model",
) -> None:
    """Validate that state_dict matches complete registration recipe for the method."""
    if type(state_dict) is not dict:
        raise CheckpointValidationError(f"{role}.state_dict must be a plain dict, got {_safe_repr(state_dict)}")

    if method == "LightGCN":
        num_users = len(expected_row_order["User"])
        num_items = len(expected_row_order["Product"])

        # Exactly 2 parameters and 1 registered buffer
        expected_keys = ["user_embeddings", "item_embeddings", "adj_user_item"]
        if list(state_dict.keys()) != expected_keys:
            raise CheckpointValidationError(
                f"{role}.state_dict keys mismatch for LightGCN. Got {_safe_sort_keys(state_dict)}, expected {_safe_sort_keys(expected_keys)}"
            )

        _validate_tensor_properties(state_dict["user_embeddings"], "user_embeddings", (num_users, 64), role=role)
        _validate_tensor_properties(state_dict["item_embeddings"], "item_embeddings", (num_items, 64), role=role)

        exp_adj = expected_buffer_state.get("adj_user_item")
        if exp_adj is None:
            raise CheckpointValidationError("expected_model buffer_state is missing 'adj_user_item'")
        _validate_sparse_adj_buffer(state_dict["adj_user_item"], exp_adj, num_users, num_items, role=role)

    elif method == "Heterogeneous KG+GNN":
        expected_keys = _get_hetero_expected_param_names()
        if list(state_dict.keys()) != expected_keys:
            missing = set(expected_keys) - set(state_dict.keys())
            extra = set(state_dict.keys()) - set(expected_keys)
            raise CheckpointValidationError(
                f"{role}.state_dict keys mismatch for HeteroGNN. Missing: {_safe_sort_keys(missing)}, Extra: {_safe_sort_keys(extra)}"
            )

        num_u = len(expected_row_order["User"])
        num_p = len(expected_row_order["Product"])
        num_c = len(expected_row_order["CategoryPath"])
        num_b = len(expected_row_order["Brand"])

        _validate_tensor_properties(state_dict["user_embeddings"], "user_embeddings", (num_u, 64), role=role)
        _validate_tensor_properties(state_dict["product_embeddings"], "product_embeddings", (num_p, 64), role=role)
        _validate_tensor_properties(state_dict["category_embeddings"], "category_embeddings", (num_c, 64), role=role)
        _validate_tensor_properties(state_dict["brand_embeddings"], "brand_embeddings", (num_b, 64), role=role)

        for rel in RELATION_SCHEMAS:
            _validate_tensor_properties(state_dict[f"relation_weights_l0.{rel}"], f"relation_weights_l0.{rel}", (64, 64), role=role)
            _validate_tensor_properties(state_dict[f"relation_weights_l1.{rel}"], f"relation_weights_l1.{rel}", (64, 64), role=role)

        for ntype in NODE_TYPES:
            _validate_tensor_properties(state_dict[f"self_weights_l0.{ntype}"], f"self_weights_l0.{ntype}", (64, 64), role=role)
            _validate_tensor_properties(state_dict[f"self_biases_l0.{ntype}"], f"self_biases_l0.{ntype}", (64,), role=role)
            _validate_tensor_properties(state_dict[f"self_weights_l1.{ntype}"], f"self_weights_l1.{ntype}", (64, 64), role=role)
            _validate_tensor_properties(state_dict[f"self_biases_l1.{ntype}"], f"self_biases_l1.{ntype}", (64,), role=role)

    else:
        raise CheckpointValidationError(f"Unsupported model method: {method!r}")


# ---------------------------------------------------------------------------
# Adam Optimizer Validation
# ---------------------------------------------------------------------------

def _validate_adam_optimizer(
    optimizer: Any,
    expected_groups: List[Dict[str, Any]],
    adam_step_dtype: str,
    global_step: int,
    model_method: str,
    validated_param_shapes: Dict[str, Tuple[int, ...]],
) -> None:
    """Validate Adam optimizer groups, bindings, options, moments, step precision, and lazy updates."""
    if type(optimizer) is not dict:
        raise CheckpointValidationError(f"optimizer must be an exact dict, got {_safe_repr(optimizer)}")
    expected_opt_keys = {"algorithm", "state_dict", "parameter_bindings"}
    if set(optimizer.keys()) != expected_opt_keys:
        missing_k = _safe_sort_keys(expected_opt_keys - set(optimizer.keys()))
        extra_k = _safe_sort_keys(set(optimizer.keys()) - expected_opt_keys)
        raise CheckpointValidationError(f"optimizer keys mismatch. Missing: {missing_k}, Extra: {extra_k}")

    if optimizer["algorithm"] != "Adam":
        raise CheckpointValidationError(f"optimizer.algorithm must be 'Adam', got {_safe_repr(optimizer['algorithm'])}")

    state_dict = optimizer["state_dict"]
    if type(state_dict) is not dict or set(state_dict.keys()) != {"state", "param_groups"}:
        raise CheckpointValidationError("optimizer.state_dict must be an exact dict with keys 'state' and 'param_groups'")

    state = state_dict["state"]
    param_groups = state_dict["param_groups"]
    if type(state) is not dict:
        raise CheckpointValidationError(f"optimizer.state_dict.state must be a dict, got {_safe_repr(state)}")
    if type(param_groups) is not list:
        raise CheckpointValidationError(f"optimizer.state_dict.param_groups must be a list, got {_safe_repr(param_groups)}")

    if len(param_groups) != len(expected_groups):
        raise CheckpointValidationError(
            f"optimizer.state_dict.param_groups length {len(param_groups)} != expected {len(expected_groups)}"
        )

    expected_all_params: List[str] = []
    for eg in expected_groups:
        expected_all_params.extend(eg["parameter_names"])

    num_total_params = len(expected_all_params)
    if model_method == "LightGCN" and num_total_params != 2:
        raise CheckpointValidationError(f"LightGCN must have exactly 2 optimizer parameters, got {_safe_repr(num_total_params)}")
    elif model_method == "Heterogeneous KG+GNN" and num_total_params != 36:
        raise CheckpointValidationError(f"HeteroGNN must have exactly 36 optimizer parameters, got {_safe_repr(num_total_params)}")

    # Validate state keys: every key must be an integer in 0..num_total_params - 1
    for k in state.keys():
        if type(k) is not int or type(k) is bool or k < 0 or k >= num_total_params:
            raise CheckpointValidationError(f"optimizer.state_dict.state contains invalid or orphan ID {_safe_repr(k)}")

    # Validate parameter_bindings
    bindings = optimizer["parameter_bindings"]
    if type(bindings) is not list or len(bindings) != num_total_params:
        raise CheckpointValidationError(
            f"optimizer.parameter_bindings must be an exact list of length {_safe_repr(num_total_params)}, got {_safe_repr(bindings)}"
        )

    binding_by_id: Dict[int, Dict[str, Any]] = {}
    for idx, b in enumerate(bindings):
        if type(b) is not dict or set(b.keys()) != {"id", "name", "updates"}:
            raise CheckpointValidationError(f"parameter_bindings[{idx}] keys must be exactly id, name, updates")
        p_id = b["id"]
        p_name = b["name"]
        p_updates = b["updates"]

        _check_strict_int(p_id, f"parameter_bindings[{idx}].id", min_val=0, max_val=num_total_params - 1)
        _check_clean_text(p_name, f"parameter_bindings[{idx}].name")
        _check_strict_int(p_updates, f"parameter_bindings[{idx}].updates", min_val=0)

        if p_id != idx:
            raise CheckpointValidationError(f"parameter_bindings[{idx}].id must be packed in order {idx}, got {_safe_repr(p_id)}")
        if p_name != expected_all_params[idx]:
            raise CheckpointValidationError(
                f"parameter_bindings[{idx}].name {_safe_repr(p_name)} does not match expected {_safe_repr(expected_all_params[idx])}"
            )
        binding_by_id[p_id] = b

    # Validate param_groups
    current_packed_id = 0
    for g_idx, (pg, eg) in enumerate(zip(param_groups, expected_groups)):
        if type(pg) is not dict:
            raise CheckpointValidationError(f"param_groups[{g_idx}] must be an exact dict")
        for k in pg.keys():
            if type(k) is not str:
                raise CheckpointValidationError(f"param_groups[{g_idx}] keys must be strings, got {_safe_repr(k)}")

        expected_opts = eg.get("options")
        if type(expected_opts) is not dict:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options must be a dict")

        has_param_names = "param_names" in expected_opts
        required_pg_keys = {
            "params", "lr", "betas", "eps", "weight_decay", "amsgrad",
            "maximize", "foreach", "capturable", "differentiable", "fused",
            "decoupled_weight_decay",
        }
        if has_param_names:
            required_pg_keys.add("param_names")

        if set(pg.keys()) != required_pg_keys:
            missing_k = _safe_sort_keys(required_pg_keys - set(pg.keys()))
            extra_k = _safe_sort_keys(set(pg.keys()) - required_pg_keys)
            raise CheckpointValidationError(f"param_groups[{g_idx}] keys mismatch. Missing: {missing_k}, Extra: {extra_k}")

        p_list = pg["params"]
        if type(p_list) is not list:
            raise CheckpointValidationError(f"param_groups[{g_idx}].params must be an exact list, got {_safe_repr(p_list)}")
        for pid in p_list:
            if type(pid) is not int or type(pid) is bool:
                raise CheckpointValidationError(f"param_groups[{g_idx}].params must contain exact ints (bool excluded), got {_safe_repr(pid)}")

        expected_group_pnames = eg["parameter_names"]
        expected_group_ids = list(range(current_packed_id, current_packed_id + len(expected_group_pnames)))
        if list(p_list) != expected_group_ids:
            raise CheckpointValidationError(
                f"param_groups[{g_idx}].params IDs {_safe_sort_keys(p_list)} do not match expected {_safe_sort_keys(expected_group_ids)}"
            )
        current_packed_id += len(expected_group_pnames)

        if has_param_names:
            act_pnames = pg["param_names"]
            trusted_pnames = expected_opts["param_names"]
            if (type(trusted_pnames) is not list and type(trusted_pnames) is not tuple):
                raise CheckpointValidationError(
                    f"expected_runtime.optimizer_groups[{g_idx}].options.param_names must be an exact list or tuple"
                )
            if type(act_pnames) is not type(trusted_pnames):
                raise CheckpointValidationError(
                    f"param_groups[{g_idx}].param_names must have the same concrete list or tuple type as trusted options"
                )
            for name_index, name in enumerate(act_pnames):
                _check_clean_text(name, f"param_groups[{g_idx}].param_names[{name_index}]")
            for name_index, name in enumerate(trusted_pnames):
                _check_clean_text(name, f"expected_runtime.optimizer_groups[{g_idx}].options.param_names[{name_index}]")
            if act_pnames != trusted_pnames:
                raise CheckpointValidationError(
                    f"param_groups[{g_idx}].param_names does not match the ordered trusted optional names"
                )

        # Compare exact options
        lr = _check_strict_number(pg["lr"], f"param_groups[{g_idx}].lr")
        if lr != expected_opts.get("lr") or lr != ADAM_EXPECTED_LR:
            raise CheckpointValidationError(f"param_groups[{g_idx}].lr {_safe_repr(lr)} != expected {_safe_repr(expected_opts.get('lr'))}")

        betas = pg["betas"]
        if type(betas) is not tuple or len(betas) != 2:
            raise CheckpointValidationError(f"param_groups[{g_idx}].betas must be an exact 2-element tuple, got {_safe_repr(betas)}")
        b1 = _check_strict_number(betas[0], f"param_groups[{g_idx}].betas[0]")
        b2 = _check_strict_number(betas[1], f"param_groups[{g_idx}].betas[1]")

        exp_betas = expected_opts.get("betas")
        if type(exp_betas) is not tuple or len(exp_betas) != 2:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options['betas'] must be an exact 2-element tuple, got {_safe_repr(exp_betas)}")
        exp_b1 = _check_strict_number(exp_betas[0], f"expected_runtime.optimizer_groups[{g_idx}].options.betas[0]")
        exp_b2 = _check_strict_number(exp_betas[1], f"expected_runtime.optimizer_groups[{g_idx}].options.betas[1]")

        if (b1, b2) != ADAM_EXPECTED_BETAS or (b1, b2) != (exp_b1, exp_b2):
            raise CheckpointValidationError(f"param_groups[{g_idx}].betas ({_safe_repr(b1)}, {_safe_repr(b2)}) != {ADAM_EXPECTED_BETAS}")

        eps = _check_strict_number(pg["eps"], f"param_groups[{g_idx}].eps")
        if eps != ADAM_EXPECTED_EPS or eps != expected_opts.get("eps"):
            raise CheckpointValidationError(f"param_groups[{g_idx}].eps {_safe_repr(eps)} != {ADAM_EXPECTED_EPS}")

        wd = _check_strict_number(pg["weight_decay"], f"param_groups[{g_idx}].weight_decay")
        if wd != ADAM_EXPECTED_WEIGHT_DECAY or wd != expected_opts.get("weight_decay"):
            raise CheckpointValidationError(f"param_groups[{g_idx}].weight_decay {_safe_repr(wd)} != {ADAM_EXPECTED_WEIGHT_DECAY}")

        for flag_name in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"):
            flag_val = _check_strict_bool(pg[flag_name], f"param_groups[{g_idx}].{flag_name}")
            exp_flag = expected_opts.get(flag_name)
            if flag_val is not False or flag_val != exp_flag:
                raise CheckpointValidationError(f"param_groups[{g_idx}].{flag_name} must be False in this CPU profile")

        act_foreach = pg["foreach"]
        exp_foreach = expected_opts.get("foreach")
        if (act_foreach is not None and type(act_foreach) is not bool) or act_foreach != exp_foreach:
            raise CheckpointValidationError(f"param_groups[{g_idx}].foreach {_safe_repr(act_foreach)} != expected {_safe_repr(exp_foreach)}")

        act_fused = pg["fused"]
        exp_fused = expected_opts.get("fused")
        if (act_fused is not None and type(act_fused) is not bool) or act_fused != exp_fused or (act_fused is not None and act_fused is not False):
            raise CheckpointValidationError(f"param_groups[{g_idx}].fused {_safe_repr(act_fused)} != expected {_safe_repr(exp_fused)}")

    # Validate state per parameter
    target_step_dtype = torch.float32 if adam_step_dtype == "torch.float32" else torch.float64

    total_updates_sum = 0
    for p_id, b in binding_by_id.items():
        updates = b["updates"]
        p_name = b["name"]
        if p_name not in validated_param_shapes:
            raise CheckpointValidationError(f"Unknown parameter name {_safe_repr(p_name)} in optimizer parameter_bindings")
        exp_shape = validated_param_shapes[p_name]
        total_updates_sum += updates

        if updates > global_step:
            raise CheckpointValidationError(
                f"Parameter {p_name!r} updates ({_safe_repr(updates)}) exceeds global_step ({_safe_repr(global_step)})"
            )

        if p_id not in state:
            if updates == 0:
                continue
            raise CheckpointValidationError(
                f"Parameter {p_name!r} has updates={_safe_repr(updates)} > 0 but state is omitted"
            )

        p_state = state[p_id]
        if type(p_state) is not dict:
            raise CheckpointValidationError(
                f"Parameter {p_name!r} state must be an exact dict, got {_safe_repr(p_state)}"
            )

        if updates == 0:
            if p_state == {}:
                continue
            # Complete initialized zero state
            if set(p_state.keys()) != {"step", "exp_avg", "exp_avg_sq"}:
                raise CheckpointValidationError(
                    f"Parameter {p_name!r} has updates=0 but non-empty state does not match complete zero state keys"
                )
            step_t = p_state["step"]
            if type(step_t) is not torch.Tensor or step_t.dim() != 0:
                raise CheckpointValidationError(f"Parameter {p_name!r} step must be 0-D tensor")
            if step_t.device.type != "cpu" or step_t.dtype != target_step_dtype:
                raise CheckpointValidationError(f"Parameter {p_name!r} zero step must be CPU {target_step_dtype}")
            if step_t.requires_grad or step_t.grad_fn is not None:
                raise CheckpointValidationError(f"Parameter {p_name!r} step must be detached")
            if step_t.item() != 0:
                raise CheckpointValidationError(f"Parameter {p_name!r} step must be 0 for updates=0")

            for m_name in ("exp_avg", "exp_avg_sq"):
                m_t = p_state[m_name]
                if type(m_t) is not torch.Tensor:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} must be exact torch.Tensor")
                if m_t.device.type != "cpu" or m_t.dtype != torch.float32 or m_t.layout != torch.strided:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} must be CPU strided float32")
                if tuple(m_t.shape) != exp_shape:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} shape mismatch: {tuple(m_t.shape)} != {exp_shape}")
                if m_t.requires_grad or m_t.grad_fn is not None:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} must be detached")
                if (m_t != 0).any():
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} must be identically zero for zero state")

        else:
            # updates > 0: must have step, exp_avg, exp_avg_sq
            if p_state is None or type(p_state) is not dict or set(p_state.keys()) != {"step", "exp_avg", "exp_avg_sq"}:
                raise CheckpointValidationError(
                    f"Parameter {p_name!r} has updates={_safe_repr(updates)} > 0 but missing or malformed state fields"
                )

            step_t = p_state["step"]
            if type(step_t) is not torch.Tensor:
                raise CheckpointValidationError(f"Parameter {p_name!r} step must be exact torch.Tensor")
            if step_t.device.type != "cpu":
                raise CheckpointValidationError(f"Parameter {p_name!r} step must be on CPU")
            if step_t.dim() != 0:
                raise CheckpointValidationError(f"Parameter {p_name!r} step must be 0-D tensor, got dim {step_t.dim()}")
            if step_t.dtype != target_step_dtype:
                raise CheckpointValidationError(
                    f"Parameter {p_name!r} step dtype {step_t.dtype} != expected {target_step_dtype}"
                )
            if step_t.requires_grad or step_t.grad_fn is not None:
                raise CheckpointValidationError(f"Parameter {p_name!r} step must be detached")

            # Check precision: float32 exact integer representability
            # int(float(updates)) == updates
            step_val = step_t.item()
            if not math.isfinite(step_val) or step_val < 0:
                raise CheckpointValidationError(f"Parameter {p_name!r} step value must be finite non-negative")

            # Validate that step integer value can be represented exactly in target_step_dtype
            if target_step_dtype == torch.float32:
                # Float32 has 24 bits of mantissa: values where int(float(updates)) != updates cannot be represented
                try:
                    f_val = float(updates)
                    roundtrip = int(torch.tensor(f_val, dtype=torch.float32).item())
                    if roundtrip != updates:
                        raise CheckpointValidationError(
                            f"Parameter {p_name!r} updates ({_safe_repr(updates)}) cannot be represented exactly in float32"
                        )
                except OverflowError:
                    raise CheckpointValidationError(
                        f"Parameter {p_name!r} updates ({_safe_repr(updates)}) overflows float32"
                    )

            if step_val != updates:
                raise CheckpointValidationError(
                    f"Parameter {p_name!r} step value ({_safe_repr(step_val)}) != updates ({_safe_repr(updates)})"
                )

            # Validate exp_avg and exp_avg_sq
            for m_name in ("exp_avg", "exp_avg_sq"):
                m_t = p_state[m_name]
                if type(m_t) is not torch.Tensor:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} must be exact torch.Tensor")
                if m_t.device.type != "cpu":
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} must be on CPU")
                if m_t.layout != torch.strided:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} layout must be strided, got {m_t.layout}")
                if m_t.dtype != torch.float32:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} dtype must be float32, got {m_t.dtype}")
                if tuple(m_t.shape) != exp_shape:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} shape mismatch: {tuple(m_t.shape)} != {exp_shape}")
                if m_t.requires_grad or m_t.grad_fn is not None:
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} must be detached")
                if not torch.isfinite(m_t).all():
                    raise CheckpointValidationError(f"Parameter {p_name!r} {m_name} contains non-finite values (NaN or Inf)")

            # Second moment must be strictly non-negative
            if (p_state["exp_avg_sq"] < 0).any():
                raise CheckpointValidationError(f"Parameter {p_name!r} exp_avg_sq contains negative values")

    if global_step == 0:
        if total_updates_sum != 0:
            raise CheckpointValidationError(
                f"global_step=0 but total parameter updates={_safe_repr(total_updates_sum)} > 0"
            )
    else:
        if total_updates_sum < global_step:
            raise CheckpointValidationError(
                f"Sum of parameter updates ({_safe_repr(total_updates_sum)}) < global_step ({_safe_repr(global_step)})"
            )


# ---------------------------------------------------------------------------
# RNG & Codecs Validation
# ---------------------------------------------------------------------------

def _validate_rng_state_dict(
    rng_dict: Any,
    expected_identity: CanonicalResumeIdentity,
    expected_runtime: Dict[str, Any],
) -> None:
    """Validate RNG roles, generator registry, native state codecs, seed matching, and worker contexts."""
    if type(rng_dict) is not dict:
        raise CheckpointValidationError("rng must be an exact dict")
    if set(rng_dict.keys()) != {"roles", "generators"}:
        missing_rk = _safe_sort_keys({"roles", "generators"} - set(rng_dict.keys()))
        extra_rk = _safe_sort_keys(set(rng_dict.keys()) - {"roles", "generators"})
        raise CheckpointValidationError(f"rng keys mismatch. Missing: {missing_rk}, Extra: {extra_rk}")

    roles = rng_dict["roles"]
    generators = rng_dict["generators"]
    if type(roles) is not list or type(generators) is not list:
        raise CheckpointValidationError("rng.roles and rng.generators must be exact lists")

    # Validate roles: exact 10 CONFIGURED_RNG_COMPONENTS in order
    if len(roles) != len(CONFIGURED_RNG_COMPONENTS):
        raise CheckpointValidationError(
            f"rng.roles count ({len(roles)}) != expected ({len(CONFIGURED_RNG_COMPONENTS)})"
        )

    expected_streams = expected_identity.resolved_settings["determinism"]["rng_streams"]
    expected_stream_by_name = {s["name"]: s for s in expected_streams}
    expected_role_bindings = expected_runtime["rng_role_bindings"]
    expected_gen_bindings = expected_runtime["rng_generator_bindings"]

    active_generator_ids: Set[str] = set()

    for idx, r in enumerate(roles):
        if type(r) is not dict:
            raise CheckpointValidationError(f"rng.roles[{idx}] must be a dict")
        expected_role_keys = {"name", "applicable", "algorithm", "seed", "derivation", "reason", "generator_ids"}
        if set(r.keys()) != expected_role_keys:
            missing_rok = _safe_sort_keys(expected_role_keys - set(r.keys()))
            extra_rok = _safe_sort_keys(set(r.keys()) - expected_role_keys)
            raise CheckpointValidationError(f"rng.roles[{idx}] keys mismatch. Missing: {missing_rok}, Extra: {extra_rok}")

        r_name = r["name"]
        if r_name != CONFIGURED_RNG_COMPONENTS[idx]:
            raise CheckpointValidationError(
                f"rng.roles[{idx}] name {_safe_repr(r_name)} does not match expected order {_safe_repr(CONFIGURED_RNG_COMPONENTS[idx])}"
            )

        exp_s = expected_stream_by_name[r_name]
        _check_strict_bool(r["applicable"], f"rng.roles[{idx}].applicable")

        if r_name == "PyTorch_CUDA_if_applicable" and r["applicable"]:
            raise CheckpointValidationError("PyTorch_CUDA_if_applicable role cannot be active in this CPU profile")

        if r["applicable"]:
            _check_strict_int(r["seed"], f"rng.roles[{idx}].seed", min_val=0)
        else:
            if r["seed"] is not None:
                raise CheckpointValidationError(f"rng.roles[{idx}] is inactive but seed is not None")

        for f in ("applicable", "algorithm", "seed", "derivation", "reason"):
            if r[f] != exp_s[f]:
                raise CheckpointValidationError(
                    f"rng.roles[{idx}] ({r_name}) field {f!r} {_safe_repr(r[f])} != expected {_safe_repr(exp_s[f])}"
                )

        g_ids = r["generator_ids"]
        if type(g_ids) is not list:
            raise CheckpointValidationError(f"rng.roles[{idx}] generator_ids must be an exact list, got {_safe_repr(g_ids)}")

        # Refusal of duplicates within role
        seen_r_gids = set()
        for gid in g_ids:
            if type(gid) is not str or not gid:
                raise CheckpointValidationError(f"rng.roles[{idx}] generator_id must be non-empty string")
            if gid in seen_r_gids:
                raise CheckpointValidationError(f"Duplicate generator ID {_safe_repr(gid)} within role {r_name!r}")
            seen_r_gids.add(gid)

        if not r["applicable"]:
            if len(g_ids) != 0:
                raise CheckpointValidationError(
                    f"rng.roles[{idx}] ({r_name}) is inactive (applicable=False) but generator_ids is not empty"
                )
            if len(expected_role_bindings[r_name]) != 0:
                raise CheckpointValidationError(
                    f"expected_runtime.rng_role_bindings[{r_name!r}] must be empty for inactive role"
                )
        else:
            exp_g_ids = expected_role_bindings[r_name]
            if list(g_ids) != list(exp_g_ids):
                raise CheckpointValidationError(
                    f"rng.roles[{idx}] ({_safe_repr(r_name)}) generator_ids {_safe_sort_keys(g_ids)} != expected {_safe_sort_keys(exp_g_ids)}"
                )
            if len(g_ids) == 0:
                raise CheckpointValidationError(
                    f"rng.roles[{idx}] ({r_name}) is active but generator_ids is empty"
                )
            for gid in g_ids:
                active_generator_ids.add(gid)

    # Validate generators
    gen_by_id: Dict[str, Dict[str, Any]] = {}
    runtime_workers = expected_identity.resolved_settings["execution"]["runtime_workers"]

    for idx, g in enumerate(generators):
        if type(g) is not dict:
            raise CheckpointValidationError(f"rng.generators[{idx}] must be a dict")
        expected_g_keys = {"id", "codec", "worker_id", "seed_material", "state"}
        if set(g.keys()) != expected_g_keys:
            missing_gk = _safe_sort_keys(expected_g_keys - set(g.keys()))
            extra_gk = _safe_sort_keys(set(g.keys()) - expected_g_keys)
            raise CheckpointValidationError(f"rng.generators[{idx}] keys mismatch. Missing: {missing_gk}, Extra: {extra_gk}")

        gid = _check_clean_text(g["id"], f"rng.generators[{idx}].id")
        if gid in gen_by_id:
            raise CheckpointValidationError(f"Duplicate generator ID {_safe_repr(gid)} in rng.generators")
        gen_by_id[gid] = g

        codec = g["codec"]
        if codec not in SUPPORTED_CODECS:
            raise CheckpointValidationError(f"rng.generators[{idx}] ({_safe_repr(gid)}) unsupported codec: {_safe_repr(codec)}")

        exp_g_desc = expected_gen_bindings.get(gid)
        if exp_g_desc is None:
            raise CheckpointValidationError(f"Orphan generator ID {_safe_repr(gid)} not in expected_runtime.rng_generator_bindings")
        if type(exp_g_desc) is not dict or set(exp_g_desc.keys()) != {"codec", "worker_id", "seed_material"}:
            raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] malformed schema")

        if codec != exp_g_desc["codec"]:
            raise CheckpointValidationError(f"rng.generators[{idx}] ({_safe_repr(gid)}) codec {_safe_repr(codec)} != expected {_safe_repr(exp_g_desc['codec'])}")

        wid = g["worker_id"]
        if wid is not None:
            _check_strict_int(wid, f"rng.generators[{idx}].worker_id", min_val=0)
            if runtime_workers == 0:
                raise CheckpointValidationError(f"Generator {_safe_repr(gid)} has worker_id {_safe_repr(wid)} but runtime_workers is 0")
            if wid >= runtime_workers:
                raise CheckpointValidationError(f"Generator {_safe_repr(gid)} worker_id {_safe_repr(wid)} >= runtime_workers {_safe_repr(runtime_workers)}")

        if wid != exp_g_desc["worker_id"]:
            raise CheckpointValidationError(f"rng.generators[{idx}] ({_safe_repr(gid)}) worker_id {_safe_repr(wid)} != expected {_safe_repr(exp_g_desc['worker_id'])}")

        seed_mat = g["seed_material"]
        exp_seed_mat = exp_g_desc["seed_material"]

        if codec == "torch.cpu.Generator.native.v1":
            if type(seed_mat) is not int or type(seed_mat) is bool or seed_mat < 0:
                raise CheckpointValidationError(
                    f"Generator {_safe_repr(gid)} Torch seed_material must be exact non-negative integer, got {_safe_repr(seed_mat)}"
                )
            if seed_mat > 0xFFFFFFFFFFFFFFFF:
                raise CheckpointValidationError(
                    f"Generator {_safe_repr(gid)} Torch seed exceeds uint64: {_safe_repr(seed_mat)}"
                )
            if type(exp_seed_mat) is not int or type(exp_seed_mat) is bool or exp_seed_mat < 0:
                raise CheckpointValidationError(
                    f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material must be exact non-negative integer, got {_safe_repr(exp_seed_mat)}"
                )
            if exp_seed_mat > 0xFFFFFFFFFFFFFFFF:
                raise CheckpointValidationError(
                    f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material exceeds uint64: {_safe_repr(exp_seed_mat)}"
                )

        elif codec == "python.random.MT19937.v3":
            if type(seed_mat) is not int or type(seed_mat) is bool or seed_mat < 0:
                raise CheckpointValidationError(
                    f"Generator {_safe_repr(gid)} Python seed_material must be exact non-negative integer, got {_safe_repr(seed_mat)}"
                )
            if type(exp_seed_mat) is not int or type(exp_seed_mat) is bool or exp_seed_mat < 0:
                raise CheckpointValidationError(
                    f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material must be exact non-negative integer, got {_safe_repr(exp_seed_mat)}"
                )

        elif codec in ("numpy.random.RandomState.MT19937.v1", "numpy.random.PCG64.v1"):
            if type(seed_mat) is list:
                if wid is None:
                    raise CheckpointValidationError(f"Generator {_safe_repr(gid)} word list seed only allowed for worker contexts")
                if not seed_mat:
                    raise CheckpointValidationError(f"Generator {_safe_repr(gid)} seed_material word list cannot be empty")
                for w_idx, w in enumerate(seed_mat):
                    _check_strict_int(w, f"rng.generators[{idx}] seed_material word[{w_idx}]", min_val=0, max_val=0xFFFFFFFF)
                if type(exp_seed_mat) is not list or not exp_seed_mat:
                    raise CheckpointValidationError(
                        f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material must be non-empty list of words"
                    )
                for w_idx, w in enumerate(exp_seed_mat):
                    _check_strict_int(w, f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] seed word[{w_idx}]", min_val=0, max_val=0xFFFFFFFF)
            else:
                _check_strict_int(seed_mat, f"rng.generators[{idx}].seed_material", min_val=0)
                _check_strict_int(exp_seed_mat, f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material", min_val=0)
                if codec == "numpy.random.RandomState.MT19937.v1":
                    if seed_mat > 0xFFFFFFFF:
                        raise CheckpointValidationError(f"Generator {_safe_repr(gid)} RandomState seed exceeds uint32: {_safe_repr(seed_mat)}")
                    if exp_seed_mat > 0xFFFFFFFF:
                        raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] RandomState seed exceeds uint32: {_safe_repr(exp_seed_mat)}")

        if wid is None:
            # Codec type/range checks above precede every parent equality.
            # Parent/local material is scalar even for NumPy codecs.
            _check_strict_int(seed_mat, f"rng.generators[{idx}].seed_material", min_val=0)
            _check_strict_int(exp_seed_mat, f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material", min_val=0)
            bound_roles = [r for r in roles if gid in r.get("generator_ids", [])]
            for r in bound_roles:
                r_seed = r["seed"]
                if r_seed != seed_mat:
                    raise CheckpointValidationError(
                        f"Generator {_safe_repr(gid)} seed_material {_safe_repr(seed_mat)} does not match bound role {_safe_repr(r['name'])} seed {_safe_repr(r_seed)}"
                    )

        if seed_mat != exp_seed_mat:
            raise CheckpointValidationError(
                f"rng.generators[{idx}] ({_safe_repr(gid)}) seed_material {_safe_repr(seed_mat)} != expected {_safe_repr(exp_seed_mat)}"
            )

        # Validate codec native state representation without altering global live state
        state_val = g["state"]
        _validate_single_codec_state(codec, state_val, seed_mat, gid)

    # Active generator IDs must match gen_by_id keys and expected_gen_bindings keys exactly
    if set(gen_by_id.keys()) != active_generator_ids:
        extra_gens = _safe_sort_keys(set(gen_by_id.keys()) - active_generator_ids)
        missing_gens = _safe_sort_keys(active_generator_ids - set(gen_by_id.keys()))
        raise CheckpointValidationError(
            f"rng.generators mismatch against active bound roles. Missing: {missing_gens}, Extra: {extra_gens}"
        )
    if set(expected_gen_bindings.keys()) != active_generator_ids:
        extra_exp = _safe_sort_keys(set(expected_gen_bindings.keys()) - active_generator_ids)
        missing_exp = _safe_sort_keys(active_generator_ids - set(expected_gen_bindings.keys()))
        raise CheckpointValidationError(
            f"expected_runtime.rng_generator_bindings mismatch against active bound roles. Missing: {missing_exp}, Extra: {extra_exp}"
        )

    # Family & Cardinality constraints
    # Python: exactly 1 generator of Python codec
    py_gids = roles[0]["generator_ids"]
    if len(py_gids) != 1 or gen_by_id[py_gids[0]]["codec"] != "python.random.MT19937.v3" or gen_by_id[py_gids[0]]["worker_id"] is not None:
        raise CheckpointValidationError("Python role must bind exactly one parent python.random.MT19937.v3 generator")

    # NumPy: exactly 1 generator of NumPy codec
    np_gids = roles[1]["generator_ids"]
    if len(np_gids) != 1 or gen_by_id[np_gids[0]]["codec"] not in ("numpy.random.RandomState.MT19937.v1", "numpy.random.PCG64.v1") or gen_by_id[np_gids[0]]["worker_id"] is not None:
        raise CheckpointValidationError("NumPy role must bind exactly one parent NumPy generator")

    # PyTorch_CPU: exactly 1 generator of Torch CPU codec
    torch_gids = roles[2]["generator_ids"]
    if len(torch_gids) != 1 or gen_by_id[torch_gids[0]]["codec"] != "torch.cpu.Generator.native.v1" or gen_by_id[torch_gids[0]]["worker_id"] is not None:
        raise CheckpointValidationError("PyTorch_CPU role must bind exactly one parent torch.cpu.Generator.native.v1 generator")

    # DataLoader_generator: exactly 1 generator of Torch CPU codec
    dl_gids = roles[8]["generator_ids"]
    if len(dl_gids) != 1 or gen_by_id[dl_gids[0]]["codec"] != "torch.cpu.Generator.native.v1" or gen_by_id[dl_gids[0]]["worker_id"] is not None:
        raise CheckpointValidationError("DataLoader_generator role must bind exactly one parent torch.cpu.Generator.native.v1 generator")

    # initialization role: if applicable, exactly 1 generator of Torch CPU codec
    if roles[4]["applicable"]:
        init_gids = roles[4]["generator_ids"]
        if len(init_gids) != 1 or gen_by_id[init_gids[0]]["codec"] != "torch.cpu.Generator.native.v1" or gen_by_id[init_gids[0]]["worker_id"] is not None:
            raise CheckpointValidationError("initialization role must bind exactly one parent torch.cpu.Generator.native.v1 generator")

    # DataLoader_workers role
    worker_role_gids = roles[9]["generator_ids"]
    if runtime_workers == 0:
        if len(worker_role_gids) != 0:
            raise CheckpointValidationError("DataLoader_workers must be empty when runtime_workers == 0")
    else:
        # Exactly one Python, one NumPy, and one PyTorch CPU generator per worker ID in 0..W-1
        expected_worker_gen_count = 3 * runtime_workers
        if len(worker_role_gids) != expected_worker_gen_count:
            raise CheckpointValidationError(
                f"DataLoader_workers must contain exactly 3 * W ({expected_worker_gen_count}) generators, got {len(worker_role_gids)}"
            )
        for w in range(runtime_workers):
            w_codecs = [gen_by_id[gid]["codec"] for gid in worker_role_gids if gen_by_id[gid]["worker_id"] == w]
            has_py = any(c == "python.random.MT19937.v3" for c in w_codecs)
            has_np = any(c in ("numpy.random.RandomState.MT19937.v1", "numpy.random.PCG64.v1") for c in w_codecs)
            has_torch = any(c == "torch.cpu.Generator.native.v1" for c in w_codecs)
            if not (has_py and has_np and has_torch and len(w_codecs) == 3):
                raise CheckpointValidationError(
                    f"DataLoader_workers for worker {w} must contain exactly one Python, one NumPy, and one TorchCPU generator"
                )


def _validate_single_codec_state(codec: str, state: Any, seed_mat: Any, gid: str) -> None:
    """Validate native state representation and apply to throwaway private generator."""
    # This argument is used only in diagnostics in this function. Render it
    # before interpolating it into messages or roles passed to _check_*.
    gid = _safe_repr(gid)
    if codec == "python.random.MT19937.v3":
        if type(state) is not tuple or len(state) != 3:
            raise CheckpointValidationError(f"Generator {gid!r} MT19937.v3 state must be 3-tuple, got {_safe_repr(state)}")
        if type(state[0]) is not int or type(state[0]) is bool or state[0] != 3:
            raise CheckpointValidationError(f"Generator {gid!r} MT19937.v3 state[0] must be int 3, got {_safe_repr(state[0])}")
        internal = state[1]
        if type(internal) is not tuple or len(internal) != 625:
            raise CheckpointValidationError(f"Generator {gid!r} MT19937.v3 internal state must be 625-tuple of ints")
        for i in range(624):
            w = internal[i]
            if type(w) is not int or type(w) is bool or w < 0 or w > 0xFFFFFFFF:
                raise CheckpointValidationError(f"Generator {gid!r} MT19937.v3 internal word[{i}] out of uint32 bounds: {_safe_repr(w)}")
        pos = internal[624]
        if type(pos) is not int or type(pos) is bool or pos < 0 or pos > 624:
            raise CheckpointValidationError(f"Generator {gid!r} MT19937.v3 index out of range [0, 624]: {_safe_repr(pos)}")
        gauss = state[2]
        if gauss is not None:
            _check_strict_float(gauss, f"Generator {gid!r} gauss")

        # Private throwaway validation
        try:
            r = random.Random(0)
            r.setstate(state)
        except Exception as exc:
            raise CheckpointValidationError(f"Invalid Python random state for generator {gid!r}: {_safe_repr(str(exc))}") from exc

    elif codec == "numpy.random.RandomState.MT19937.v1":
        if type(state) is not tuple or len(state) != 5:
            raise CheckpointValidationError(f"Generator {gid!r} RandomState state must be 5-tuple, got {_safe_repr(state)}")
        if state[0] != "MT19937":
            raise CheckpointValidationError(f"Generator {gid!r} RandomState[0] must be 'MT19937', got {_safe_repr(state[0])}")
        keys = state[1]
        if type(keys) is not list or len(keys) != 624:
            raise CheckpointValidationError(f"Generator {gid!r} RandomState keys must be list of 624 uint32 ints")
        for i, k in enumerate(keys):
            if type(k) is not int or type(k) is bool or k < 0 or k > 0xFFFFFFFF:
                raise CheckpointValidationError(f"Generator {gid!r} RandomState key[{i}] out of uint32 bounds: {_safe_repr(k)}")
        pos = _check_strict_int(state[2], f"Generator {gid!r} position", min_val=0, max_val=624)
        has_gauss = _check_strict_int(state[3], f"Generator {gid!r} has_gauss", min_val=0, max_val=1)
        cached_gauss = _check_strict_float(state[4], f"Generator {gid!r} cached_gauss")

        # Private throwaway validation
        try:
            rs = np.random.RandomState(0)
            key_arr = np.array(keys, dtype=np.uint32)
            rs.set_state(("MT19937", key_arr, pos, has_gauss, cached_gauss))
        except Exception as exc:
            raise CheckpointValidationError(f"Invalid NumPy RandomState for generator {gid!r}: {_safe_repr(str(exc))}") from exc

    elif codec == "numpy.random.PCG64.v1":
        if type(state) is not dict or set(state.keys()) != {"bit_generator", "state", "has_uint32", "uinteger"}:
            raise CheckpointValidationError(f"Generator {gid!r} PCG64 state dict keys mismatch")
        if state["bit_generator"] != "PCG64":
            raise CheckpointValidationError(f"Generator {gid!r} bit_generator must be 'PCG64', got {_safe_repr(state['bit_generator'])}")
        s_inner = state["state"]
        if type(s_inner) is not dict or set(s_inner.keys()) != {"state", "inc"}:
            raise CheckpointValidationError(f"Generator {gid!r} PCG64 inner state must contain state and inc")
        st_val = _check_strict_int(s_inner["state"], f"Generator {gid!r} PCG64 state", min_val=0, max_val=(1 << 128) - 1)
        inc_val = _check_strict_int(s_inner["inc"], f"Generator {gid!r} PCG64 inc", min_val=0, max_val=(1 << 128) - 1)
        if inc_val % 2 != 1:
            raise CheckpointValidationError(f"Generator {gid!r} PCG64 inc must be odd, got {_safe_repr(inc_val)}")
        _check_strict_int(state["has_uint32"], f"Generator {gid!r} has_uint32", min_val=0, max_val=1)
        _check_strict_int(state["uinteger"], f"Generator {gid!r} uinteger", min_val=0, max_val=0xFFFFFFFF)

        # Private throwaway validation
        try:
            pcg = np.random.PCG64(0)
            pcg.state = state
        except Exception as exc:
            raise CheckpointValidationError(f"Invalid NumPy PCG64 state for generator {gid!r}: {_safe_repr(str(exc))}") from exc

    elif codec == "torch.cpu.Generator.native.v1":
        if type(state) is not torch.Tensor:
            raise CheckpointValidationError(f"Generator {gid!r} native state must be exact torch.Tensor, got {_safe_repr(state)}")
        if state.device.type != "cpu" or state.dtype != torch.uint8 or state.dim() != 1 or not state.is_contiguous():
            raise CheckpointValidationError(f"Generator {gid!r} native state must be 1-D contiguous CPU uint8 tensor")

        dummy_gen = torch.Generator(device="cpu")
        expected_bytes = dummy_gen.get_state().numel()
        if state.numel() != expected_bytes:
            raise CheckpointValidationError(
                f"Generator {gid!r} native state length {state.numel()} != expected {expected_bytes}"
            )

        # Private throwaway validation
        try:
            dummy_gen.set_state(state)
        except Exception as exc:
            raise CheckpointValidationError(f"Invalid torch CPU generator native state for generator {gid!r}: {_safe_repr(str(exc))}") from exc

        # Check initial_seed matches seed_material (always integer for Torch)
        if type(seed_mat) is int:
            init_s = dummy_gen.initial_seed()
            if init_s != seed_mat:
                raise CheckpointValidationError(
                    f"Generator {gid!r} torch initial_seed {_safe_repr(init_s)} != declared seed_material {_safe_repr(seed_mat)}"
                )


# ---------------------------------------------------------------------------
# Sampler & Epoch Progress Validation
# ---------------------------------------------------------------------------

def _compute_condition_positive_and_empty_pool_pairs(
    identity: CanonicalResumeIdentity,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Compute complete condition-positive pairs P and empty-negative-pool pairs K."""
    P: Set[Tuple[str, str]] = set()
    for u, pos_items in identity.positive_history.items():
        for item in pos_items:
            P.add((u, item))

    warm_items_set = set(identity.warm_items)
    K: Set[Tuple[str, str]] = set()

    for u in identity.positive_history:
        all_rated = set(identity.all_rated_history.get(u, ()))
        eligible_negatives = warm_items_set - all_rated
        if len(eligible_negatives) == 0:
            for item in identity.positive_history[u]:
                K.add((u, item))

    return P, K


def _validate_sampler_and_progress(
    sampler: Any,
    progress: Any,
    identity: CanonicalResumeIdentity,
    expected_runtime: Dict[str, Any],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Validate quiescent sampler state, condition-positive domain, batch policies, and progress."""
    if type(sampler) is not dict:
        raise CheckpointValidationError("sampler must be an exact dict")
    expected_sampler_keys = {
        "algorithm", "identity_hash", "source_order", "cursor", "next_epoch",
        "batch_cursor", "pending_batches", "prefetch_pending", "inflight_workers", "worker_ids",
    }
    if set(sampler.keys()) != expected_sampler_keys:
        missing_sk = _safe_sort_keys(expected_sampler_keys - set(sampler.keys()))
        extra_sk = _safe_sort_keys(set(sampler.keys()) - expected_sampler_keys)
        raise CheckpointValidationError(f"sampler keys mismatch. Missing: {missing_sk}, Extra: {extra_sk}")

    if type(progress) is not dict:
        raise CheckpointValidationError("progress must be an exact dict")
    expected_progress_keys = {"epoch", "global_step", "epoch_optimizer_steps", "empty_pool_skips"}
    if set(progress.keys()) != expected_progress_keys:
        missing_pk = _safe_sort_keys(expected_progress_keys - set(progress.keys()))
        extra_pk = _safe_sort_keys(set(progress.keys()) - expected_progress_keys)
        raise CheckpointValidationError(f"progress keys mismatch. Missing: {missing_pk}, Extra: {extra_pk}")

    # Check sampler algorithm and identity hash
    exp_alg = expected_runtime["sampler_algorithm"]
    if sampler["algorithm"] != exp_alg:
        raise CheckpointValidationError(f"sampler.algorithm {_safe_repr(sampler['algorithm'])} != expected {_safe_repr(exp_alg)}")
    if sampler["identity_hash"] != identity.identity_hash:
        raise CheckpointValidationError("sampler.identity_hash mismatch against validated identity hash")

    # Quiescent boundary checks
    for zero_field in ("batch_cursor", "pending_batches", "prefetch_pending", "inflight_workers"):
        val = _check_strict_int(sampler[zero_field], f"sampler.{zero_field}")
        if val != 0:
            raise CheckpointValidationError(
                f"Quiescent epoch boundary violated: sampler.{zero_field} must be 0, got {_safe_repr(val)}"
            )

    w_ids = sampler["worker_ids"]
    runtime_workers = identity.resolved_settings["execution"]["runtime_workers"]
    if type(w_ids) is not list:
        raise CheckpointValidationError("sampler.worker_ids must be an exact list")
    if len(w_ids) != runtime_workers:
        raise CheckpointValidationError(f"sampler.worker_ids length {_safe_repr(len(w_ids))} != expected runtime_workers {_safe_repr(runtime_workers)}")
    for idx, wid in enumerate(w_ids):
        if type(wid) is not int or type(wid) is bool or wid != idx:
            raise CheckpointValidationError(f"sampler.worker_ids[{idx}] must be exact int {idx}, got {_safe_repr(wid)}")

    # Compute pairs P and K
    P, K = _compute_condition_positive_and_empty_pool_pairs(identity)
    N = len(P)
    T = len(P - K)
    if T <= 0:
        raise CheckpointValidationError(f"Untrainable run: condition-positive trainable pairs T={_safe_repr(T)} <= 0")

    # Validate progress counters
    max_epochs = identity.resolved_settings["training"]["max_epochs"]
    epoch = _check_strict_int(progress["epoch"], "progress.epoch", min_val=0, max_val=max_epochs)
    global_step = _check_strict_int(progress["global_step"], "progress.global_step", min_val=0)

    opt_steps = progress["epoch_optimizer_steps"]
    pool_skips = progress["empty_pool_skips"]
    if type(opt_steps) is not list or len(opt_steps) != epoch:
        raise CheckpointValidationError(f"progress.epoch_optimizer_steps must be exact list of length {_safe_repr(epoch)}, got {_safe_repr(opt_steps)}")
    if type(pool_skips) is not list or len(pool_skips) != epoch:
        raise CheckpointValidationError(f"progress.empty_pool_skips must be exact list of length {_safe_repr(epoch)}, got {_safe_repr(pool_skips)}")

    for e_idx, (steps, skips) in enumerate(zip(opt_steps, pool_skips)):
        _check_strict_int(steps, f"progress.epoch_optimizer_steps[{e_idx}]", min_val=1)
        _check_strict_int(skips, f"progress.empty_pool_skips[{e_idx}]", min_val=0)
        if skips != len(K):
            raise CheckpointValidationError(
                f"progress.empty_pool_skips[{e_idx}] {_safe_repr(skips)} != |K| {_safe_repr(len(K))}"
            )

    if sum(opt_steps) != global_step:
        raise CheckpointValidationError(
            f"Sum of epoch_optimizer_steps ({_safe_repr(sum(opt_steps))}) != global_step ({_safe_repr(global_step)})"
        )

    # Check cursor and next_epoch
    cursor = _check_strict_int(sampler["cursor"], "sampler.cursor", min_val=0)
    next_epoch = _check_strict_int(sampler["next_epoch"], "sampler.next_epoch", min_val=1)
    source_order = sampler["source_order"]

    if epoch == 0:
        if global_step != 0:
            raise CheckpointValidationError(f"epoch=0 requires global_step=0, got {_safe_repr(global_step)}")
        if source_order is not None:
            raise CheckpointValidationError(f"epoch=0 requires source_order=None, got {_safe_repr(source_order)}")
        if cursor != 0:
            raise CheckpointValidationError(f"epoch=0 requires cursor=0, got {_safe_repr(cursor)}")
        if next_epoch != 1:
            raise CheckpointValidationError(f"epoch=0 requires next_epoch=1, got {_safe_repr(next_epoch)}")
    else:
        if cursor != N:
            raise CheckpointValidationError(f"Completed epoch requires cursor=N ({_safe_repr(N)}), got {_safe_repr(cursor)}")
        if next_epoch != epoch + 1:
            raise CheckpointValidationError(f"Completed epoch requires next_epoch=epoch+1 ({_safe_repr(epoch + 1)}), got {_safe_repr(next_epoch)}")

        if type(source_order) is not list or len(source_order) != N:
            raise CheckpointValidationError(f"source_order must be exact list of length N ({_safe_repr(N)}), got {_safe_repr(source_order)}")

        # Convert elements to tuple pairs
        norm_source_order: List[Tuple[str, str]] = []
        for p_idx, pair in enumerate(source_order):
            if type(pair) is not tuple or len(pair) != 2:
                raise CheckpointValidationError(f"source_order[{p_idx}] must be a 2-tuple (user_id, product_id), got {_safe_repr(pair)}")
            u_id = _check_clean_text(pair[0], f"source_order[{p_idx}].user_id")
            p_id = _check_clean_text(pair[1], f"source_order[{p_idx}].product_id")
            norm_source_order.append((u_id, p_id))

        if set(norm_source_order) != P or len(norm_source_order) != N:
            raise CheckpointValidationError("source_order must be a valid permutation of all condition-positive pairs P")

        # Validate batch policy using exact integer ceil arithmetic
        batch_policy = expected_runtime["runtime_profile"]["batch_policy"]
        batch_size = identity.resolved_settings["training"]["batch_size"]

        if batch_policy == "filter_before_batch":
            expected_steps_per_epoch = _ceil_div(T, batch_size)
            for e_idx, st in enumerate(opt_steps):
                if st != expected_steps_per_epoch:
                    raise CheckpointValidationError(
                        f"epoch_optimizer_steps[{e_idx}] {_safe_repr(st)} != expected {_safe_repr(expected_steps_per_epoch)} under filter_before_batch"
                    )

        elif batch_policy == "batch_then_filter":
            # For last completed epoch, compute exact chunk updates from retained source_order
            last_epoch_steps = 0
            for start_idx in range(0, N, batch_size):
                chunk = norm_source_order[start_idx : start_idx + batch_size]
                if any(pair not in K for pair in chunk):
                    last_epoch_steps += 1

            if opt_steps[-1] != last_epoch_steps:
                raise CheckpointValidationError(
                    f"Last epoch optimizer steps ({_safe_repr(opt_steps[-1])}) != derived batch_then_filter steps ({_safe_repr(last_epoch_steps)})"
                )

            # Older epochs must lie in [ceil(T/B), min(T, ceil(N/B))] using exact integer ceil
            min_allowed = _ceil_div(T, batch_size)
            max_allowed = min(T, _ceil_div(N, batch_size))
            for e_idx in range(epoch - 1):
                st = opt_steps[e_idx]
                if st < min_allowed or st > max_allowed:
                    raise CheckpointValidationError(
                        f"epoch_optimizer_steps[{e_idx}] {_safe_repr(st)} out of bounds [{_safe_repr(min_allowed)}, {_safe_repr(max_allowed)}]"
                    )

    return P, K


# ---------------------------------------------------------------------------
# Selection History & Terminal Semantics Validation
# ---------------------------------------------------------------------------

def _validate_selection_and_best_model(
    selection: Any,
    best_model: Any,
    current_model: Any,
    progress: Dict[str, Any],
    identity: CanonicalResumeIdentity,
    expected_model: Dict[str, Any],
) -> None:
    """Validate validation selection history, strict improvement, bad checks, and best model snapshot."""
    if type(selection) is not dict:
        raise CheckpointValidationError("selection must be an exact dict")
    expected_selection_keys = {
        "metric", "history", "best_score", "selected_epoch", "bad_checks", "status", "stop_reason",
    }
    if set(selection.keys()) != expected_selection_keys:
        missing_sk = _safe_sort_keys(expected_selection_keys - set(selection.keys()))
        extra_sk = _safe_sort_keys(set(selection.keys()) - expected_selection_keys)
        raise CheckpointValidationError(f"selection keys mismatch. Missing: {missing_sk}, Extra: {extra_sk}")

    if selection["metric"] != SELECTION_METRIC:
        raise CheckpointValidationError(f"selection.metric must be {SELECTION_METRIC!r}, got {_safe_repr(selection['metric'])}")

    val_freq = identity.resolved_settings["training"]["validation_frequency"]
    max_epochs = identity.resolved_settings["training"]["max_epochs"]
    if val_freq > max_epochs:
        raise CheckpointValidationError(f"validation_frequency ({val_freq}) > max_epochs ({max_epochs}): at least one check required")

    epoch = progress["epoch"]

    history = selection["history"]
    if type(history) is not list:
        raise CheckpointValidationError(f"selection.history must be an exact list, got {_safe_repr(history)}")

    # Determine scheduled check epochs <= epoch
    scheduled_epochs: List[int] = []
    ep_cursor = val_freq
    while ep_cursor <= epoch and ep_cursor <= max_epochs:
        scheduled_epochs.append(ep_cursor)
        ep_cursor += val_freq

    if len(history) != len(scheduled_epochs):
        raise CheckpointValidationError(
            f"selection.history length ({_safe_repr(len(history))}) != scheduled validation checks count ({_safe_repr(len(scheduled_epochs))})"
        )

    # Recompute selection step-by-step
    recomputed_best_score: Optional[float] = None
    recomputed_selected_epoch: Optional[int] = None
    recomputed_bad_checks = 0
    terminal_check_epoch: Optional[int] = None

    for idx, (entry, sched_ep) in enumerate(zip(history, scheduled_epochs)):
        if type(entry) is not dict or set(entry.keys()) != {"epoch", "score"}:
            raise CheckpointValidationError(f"selection.history[{idx}] keys must be exactly epoch and score")
        e_val = _check_strict_int(entry["epoch"], f"selection.history[{idx}].epoch")
        if e_val != sched_ep:
            raise CheckpointValidationError(
                f"selection.history[{idx}].epoch {_safe_repr(e_val)} != scheduled epoch {_safe_repr(sched_ep)}"
            )
        s_val = _check_strict_float(entry["score"], f"selection.history[{idx}].score", min_val=0.0, max_val=1.0)

        if idx == 0:
            recomputed_best_score = s_val
            recomputed_selected_epoch = e_val
            recomputed_bad_checks = 0
        else:
            assert recomputed_best_score is not None
            if s_val > recomputed_best_score:
                recomputed_best_score = s_val
                recomputed_selected_epoch = e_val
                recomputed_bad_checks = 0
            else:
                recomputed_bad_checks += 1
                if recomputed_bad_checks == MAX_PATIENCE_CHECKS:
                    terminal_check_epoch = e_val
                    if idx != len(history) - 1:
                        raise CheckpointValidationError(
                            "History contains additional checks recorded after bad_checks reached patience limit (10)"
                        )
                    break

    # If patience stopped, training must terminate at that exact check epoch
    if terminal_check_epoch is not None and epoch != terminal_check_epoch:
        raise CheckpointValidationError(
            f"Terminal patience stop occurred at epoch {_safe_repr(terminal_check_epoch)}; current epoch {_safe_repr(epoch)} cannot exceed it"
        )

    # Check payload selection fields against recomputed values
    actual_best_score = selection["best_score"]
    actual_selected_epoch = selection["selected_epoch"]
    actual_bad_checks = _check_strict_int(selection["bad_checks"], "selection.bad_checks", min_val=0, max_val=MAX_PATIENCE_CHECKS)

    if recomputed_selected_epoch is None:
        if actual_selected_epoch is not None:
            raise CheckpointValidationError(f"No checks scheduled yet: selected_epoch must be None, got {_safe_repr(actual_selected_epoch)}")
        if actual_best_score is not None:
            raise CheckpointValidationError(f"No checks scheduled yet: best_score must be None, got {_safe_repr(actual_best_score)}")
        if actual_bad_checks != 0:
            raise CheckpointValidationError(f"No checks scheduled yet: bad_checks must be 0, got {_safe_repr(actual_bad_checks)}")
    else:
        _check_strict_int(actual_selected_epoch, "selection.selected_epoch", min_val=1)
        if actual_selected_epoch != recomputed_selected_epoch:
            raise CheckpointValidationError(
                f"selection.selected_epoch {_safe_repr(actual_selected_epoch)} != recomputed {_safe_repr(recomputed_selected_epoch)}"
            )
        act_sc = _check_strict_float(actual_best_score, "selection.best_score", min_val=0.0, max_val=1.0)
        assert recomputed_best_score is not None
        if act_sc != recomputed_best_score:
            raise CheckpointValidationError(
                f"selection.best_score {_safe_repr(act_sc)} != recomputed {_safe_repr(recomputed_best_score)}"
            )
        if actual_bad_checks != recomputed_bad_checks:
            raise CheckpointValidationError(
                f"selection.bad_checks {_safe_repr(actual_bad_checks)} != recomputed {_safe_repr(recomputed_bad_checks)}"
            )

    # Determine status and stop_reason
    is_patience_stopped = recomputed_bad_checks == MAX_PATIENCE_CHECKS
    is_max_epochs_stopped = epoch == max_epochs

    if is_patience_stopped and is_max_epochs_stopped:
        expected_status = "stopped"
        expected_stop_reason = "patience_and_max_epochs"
    elif is_patience_stopped:
        expected_status = "stopped"
        expected_stop_reason = "patience"
    elif is_max_epochs_stopped:
        expected_status = "stopped"
        expected_stop_reason = "max_epochs"
    elif epoch == 0:
        expected_status = "initial"
        expected_stop_reason = None
    else:
        expected_status = "ready"
        expected_stop_reason = None

    if selection["status"] != expected_status:
        raise CheckpointValidationError(
            f"selection.status {_safe_repr(selection['status'])} != expected {_safe_repr(expected_status)}"
        )
    if selection["stop_reason"] != expected_stop_reason:
        raise CheckpointValidationError(
            f"selection.stop_reason {_safe_repr(selection['stop_reason'])} != expected {_safe_repr(expected_stop_reason)}"
        )

    # Best model validation
    if recomputed_selected_epoch is None:
        if best_model is not None:
            raise CheckpointValidationError(f"Before first check, best_model must be None, got {_safe_repr(best_model)}")
    else:
        if best_model is None:
            raise CheckpointValidationError(f"After validation check, best_model must not be None (selected epoch {_safe_repr(recomputed_selected_epoch)})")
        if type(best_model) is not dict or set(best_model.keys()) != {"epoch", "row_order", "state_dict"}:
            raise CheckpointValidationError("best_model must be an exact dict with epoch, row_order, state_dict")

        bm_epoch = _check_strict_int(best_model["epoch"], "best_model.epoch")
        if bm_epoch != recomputed_selected_epoch:
            raise CheckpointValidationError(
                f"best_model.epoch ({_safe_repr(bm_epoch)}) does not match recomputed selected epoch ({_safe_repr(recomputed_selected_epoch)})"
            )
        if best_model["row_order"] != expected_model["row_order"]:
            raise CheckpointValidationError("best_model.row_order does not match expected_model.row_order")

        _validate_model_state_dict(
            best_model["state_dict"],
            expected_model["method"],
            expected_model["row_order"],
            expected_model["buffer_state"],
            role="best_model",
        )

        # When current_model and best_model share the exact same epoch, their state_dicts must be bitwise identical
        if current_model["epoch"] == bm_epoch:
            cur_sd = current_model["state_dict"]
            best_sd = best_model["state_dict"]
            for k in cur_sd:
                t_cur = cur_sd[k]
                t_best = best_sd[k]
                if t_cur.layout == torch.strided:
                    # Check bitwise representation to distinguish +0.0 from -0.0
                    if not (t_cur.contiguous().view(torch.int32) == t_best.contiguous().view(torch.int32)).all():
                        raise CheckpointValidationError(
                            f"best_model and current_model share epoch {bm_epoch} but tensor {k!r} contents differ"
                        )
                elif t_cur.layout == torch.sparse_coo:
                    if not torch.equal(t_cur.indices(), t_best.indices()):
                        raise CheckpointValidationError(
                            f"best_model and current_model share epoch {bm_epoch} but sparse buffer {k!r} indices differ"
                        )
                    if not (t_cur.values().contiguous().view(torch.int32) == t_best.values().contiguous().view(torch.int32)).all():
                        raise CheckpointValidationError(
                            f"best_model and current_model share epoch {bm_epoch} but sparse buffer {k!r} values differ"
                        )


# ---------------------------------------------------------------------------
# Owned Output Copy Constructor
# ---------------------------------------------------------------------------

def _make_owned_output_copy(obj: Any) -> Any:
    """Recursively clone containers and detached compact copies of tensors."""
    if type(obj) is torch.Tensor:
        if obj.layout == torch.strided:
            return obj.detach().clone().contiguous()
        elif obj.layout == torch.sparse_coo:
            return torch.sparse_coo_tensor(
                obj.indices().detach().clone(),
                obj.values().detach().clone(),
                obj.shape,
                is_coalesced=True,
            )
        else:
            raise CheckpointValidationError(f"Cannot copy unsupported tensor layout: {obj.layout}")

    elif type(obj) is dict:
        return {k: _make_owned_output_copy(v) for k, v in obj.items()}

    elif type(obj) is list:
        return [_make_owned_output_copy(elem) for elem in obj]

    elif type(obj) is tuple:
        return tuple(_make_owned_output_copy(elem) for elem in obj)

    else:
        # Immutable primitives
        return obj


def _validate_expected_model_schema(expected_model: Any) -> None:
    """Validate structure, types, and schema of expected_model before any copying, hashing, or traversal."""
    if type(expected_model) is not dict:
        raise CheckpointValidationError(
            f"expected_model must be an exact dict, got {_safe_repr(expected_model)}"
        )
    for k in expected_model.keys():
        if type(k) is not str:
            raise CheckpointValidationError(f"expected_model keys must be strings, got {_safe_repr(k)}")
    if set(expected_model.keys()) != EXPECTED_MODEL_REQUIRED_KEYS:
        missing = _safe_sort_keys(EXPECTED_MODEL_REQUIRED_KEYS - set(expected_model.keys()))
        extra = _safe_sort_keys(set(expected_model.keys()) - EXPECTED_MODEL_REQUIRED_KEYS)
        raise CheckpointValidationError(f"expected_model keys mismatch. Missing: {missing}, Extra: {extra}")

    method = _check_clean_text(expected_model["method"], "expected_model.method")
    if method not in SUPPORTED_METHODS:
        raise CheckpointValidationError(f"expected_model.method unsupported: {_safe_repr(method)}")

    row_order = expected_model["row_order"]
    if type(row_order) is not dict:
        raise CheckpointValidationError(f"expected_model.row_order must be a dict, got {_safe_repr(row_order)}")
    for rk in row_order.keys():
        if type(rk) is not str:
            raise CheckpointValidationError(f"expected_model.row_order keys must be strings, got {_safe_repr(rk)}")

    if method == "LightGCN":
        if set(row_order.keys()) != {"User", "Product"}:
            raise CheckpointValidationError(f"LightGCN expected_model.row_order keys mismatch: {_safe_sort_keys(row_order)}")
        for rk in ("User", "Product"):
            rlist = row_order[rk]
            if (type(rlist) is not list and type(rlist) is not tuple) or len(rlist) == 0:
                raise CheckpointValidationError(f"expected_model.row_order[{rk!r}] must be a non-empty sequence")
            seen_ids = set()
            for item in rlist:
                clean_item = _check_clean_text(item, f"expected_model.row_order[{rk!r}] item")
                if clean_item in seen_ids:
                    raise CheckpointValidationError(f"Duplicate ID {_safe_repr(clean_item)} in expected_model.row_order[{rk!r}]")
                seen_ids.add(clean_item)

    elif method == "Heterogeneous KG+GNN":
        if set(row_order.keys()) != set(NODE_TYPES):
            raise CheckpointValidationError(f"HeteroGNN expected_model.row_order keys mismatch: {_safe_sort_keys(row_order)}")
        for rk in NODE_TYPES:
            rlist = row_order[rk]
            if type(rlist) is not list and type(rlist) is not tuple:
                raise CheckpointValidationError(f"expected_model.row_order[{rk!r}] must be a sequence")
            # User and Product MUST be non-empty; CategoryPath and Brand MAY be empty with (0, 64) embeddings
            if rk in ("User", "Product") and len(rlist) == 0:
                raise CheckpointValidationError(f"expected_model.row_order[{rk!r}] must be a non-empty sequence")
            seen_ids = set()
            for item in rlist:
                clean_item = _check_clean_text(item, f"expected_model.row_order[{rk!r}] item")
                if clean_item in seen_ids:
                    raise CheckpointValidationError(f"Duplicate ID {_safe_repr(clean_item)} in expected_model.row_order[{rk!r}]")
                seen_ids.add(clean_item)

    buffer_state = expected_model["buffer_state"]
    if type(buffer_state) is not dict:
        raise CheckpointValidationError(f"expected_model.buffer_state must be a dict, got {_safe_repr(buffer_state)}")
    for bk in buffer_state.keys():
        if type(bk) is not str:
            raise CheckpointValidationError(f"expected_model.buffer_state keys must be strings, got {_safe_repr(bk)}")

    if method == "LightGCN":
        if set(buffer_state.keys()) != {"adj_user_item"}:
            raise CheckpointValidationError(
                f"LightGCN expected_model.buffer_state must contain exactly 'adj_user_item', got {_safe_sort_keys(buffer_state)}"
            )
        adj = buffer_state["adj_user_item"]
        if type(adj) is not torch.Tensor or adj.layout != torch.sparse_coo:
            raise CheckpointValidationError("expected_model buffer_state['adj_user_item'] must be a sparse COO tensor")
    elif method == "Heterogeneous KG+GNN":
        if len(buffer_state) != 0:
            raise CheckpointValidationError(f"HeteroGNN expected_model.buffer_state must be empty dict, got {_safe_sort_keys(buffer_state)}")


def _validate_expected_runtime_schema(expected_runtime: Any) -> None:
    """Validate structure, types, and schema of expected_runtime before any copying, hashing, or traversal."""
    if type(expected_runtime) is not dict:
        raise CheckpointValidationError(
            f"expected_runtime must be an exact dict, got {_safe_repr(expected_runtime)}"
        )
    for k in expected_runtime.keys():
        if type(k) is not str:
            raise CheckpointValidationError(f"expected_runtime keys must be strings, got {_safe_repr(k)}")
    if set(expected_runtime.keys()) != EXPECTED_RUNTIME_REQUIRED_KEYS:
        missing = _safe_sort_keys(EXPECTED_RUNTIME_REQUIRED_KEYS - set(expected_runtime.keys()))
        extra = _safe_sort_keys(set(expected_runtime.keys()) - EXPECTED_RUNTIME_REQUIRED_KEYS)
        raise CheckpointValidationError(f"expected_runtime keys mismatch. Missing: {missing}, Extra: {extra}")

    # Validate runtime_profile
    exp_rt_profile = expected_runtime["runtime_profile"]
    if type(exp_rt_profile) is not dict:
        raise CheckpointValidationError("expected_runtime.runtime_profile must be a dict")
    exp_rt_keys = {"profile_id", "python_version", "numpy_version", "torch_version", "adam_step_dtype", "batch_policy"}
    if set(exp_rt_profile.keys()) != exp_rt_keys:
        missing_rt = _safe_sort_keys(exp_rt_keys - set(exp_rt_profile.keys()))
        extra_rt = _safe_sort_keys(set(exp_rt_profile.keys()) - exp_rt_keys)
        raise CheckpointValidationError(f"expected_runtime.runtime_profile keys mismatch. Missing: {missing_rt}, Extra: {extra_rt}")

    _check_clean_text(exp_rt_profile["profile_id"], "expected_runtime.runtime_profile.profile_id")
    _check_clean_text(exp_rt_profile["python_version"], "expected_runtime.runtime_profile.python_version")
    _check_clean_text(exp_rt_profile["numpy_version"], "expected_runtime.runtime_profile.numpy_version")
    _check_clean_text(exp_rt_profile["torch_version"], "expected_runtime.runtime_profile.torch_version")
    step_dt = _check_clean_text(exp_rt_profile["adam_step_dtype"], "expected_runtime.runtime_profile.adam_step_dtype")
    if step_dt not in SUPPORTED_STEP_DTYPES:
        raise CheckpointValidationError(f"expected_runtime.runtime_profile.adam_step_dtype unsupported: {_safe_repr(step_dt)}")
    bp = _check_clean_text(exp_rt_profile["batch_policy"], "expected_runtime.runtime_profile.batch_policy")
    if bp not in SUPPORTED_BATCH_POLICIES:
        raise CheckpointValidationError(f"expected_runtime.runtime_profile.batch_policy unsupported: {_safe_repr(bp)}")

    _check_clean_text(expected_runtime["sampler_algorithm"], "expected_runtime.sampler_algorithm")

    groups = expected_runtime["optimizer_groups"]
    if type(groups) is not list or not groups:
        raise CheckpointValidationError("expected_runtime.optimizer_groups must be a non-empty list")
    seen_params = set()
    for g_idx, g in enumerate(groups):
        if type(g) is not dict:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}] must be a dict")
        if set(g.keys()) != {"parameter_names", "options"}:
            missing_gk = _safe_sort_keys({"parameter_names", "options"} - set(g.keys()))
            extra_gk = _safe_sort_keys(set(g.keys()) - {"parameter_names", "options"})
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}] keys mismatch. Missing: {missing_gk}, Extra: {extra_gk}")

        pnames = g["parameter_names"]
        if (type(pnames) is not list and type(pnames) is not tuple) or not pnames:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].parameter_names must be a non-empty sequence")
        for pn in pnames:
            clean_pn = _check_clean_text(pn, f"expected_runtime.optimizer_groups[{g_idx}].parameter_names item")
            if clean_pn in seen_params:
                raise CheckpointValidationError(f"Parameter {_safe_repr(clean_pn)} duplicated across optimizer groups")
            seen_params.add(clean_pn)

        opts = g["options"]
        if type(opts) is not dict:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options must be a dict")
        req_opts = {
            "lr", "betas", "eps", "weight_decay", "amsgrad",
            "maximize", "foreach", "capturable", "differentiable", "fused",
            "decoupled_weight_decay",
        }
        has_param_names = "param_names" in opts
        allowed_opts = req_opts | ({"param_names"} if has_param_names else set())
        if set(opts.keys()) != allowed_opts:
            missing_ok = _safe_sort_keys(req_opts - set(opts.keys()))
            extra_ok = _safe_sort_keys(set(opts.keys()) - allowed_opts)
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options keys mismatch. Missing: {missing_ok}, Extra: {extra_ok}")

        _check_strict_number(opts["lr"], f"expected_runtime.optimizer_groups[{g_idx}].options.lr", min_val=0.0)
        betas = opts["betas"]
        if type(betas) is not tuple or len(betas) != 2:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options.betas must be an exact 2-element tuple, got {_safe_repr(betas)}")
        _check_strict_number(betas[0], f"expected_runtime.optimizer_groups[{g_idx}].options.betas[0]")
        _check_strict_number(betas[1], f"expected_runtime.optimizer_groups[{g_idx}].options.betas[1]")
        _check_strict_number(opts["eps"], f"expected_runtime.optimizer_groups[{g_idx}].options.eps", min_val=0.0)
        _check_strict_number(opts["weight_decay"], f"expected_runtime.optimizer_groups[{g_idx}].options.weight_decay", min_val=0.0)

        for flag_name in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"):
            _check_strict_bool(opts[flag_name], f"expected_runtime.optimizer_groups[{g_idx}].options.{flag_name}")

        foreach_val = opts["foreach"]
        if foreach_val is not None and type(foreach_val) is not bool:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options.foreach must be None or bool, got {_safe_repr(foreach_val)}")

        fused_val = opts["fused"]
        if fused_val is not None and fused_val is not False:
            raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options.fused must be None or False, got {_safe_repr(fused_val)}")

        if has_param_names:
            pnames_opt = opts["param_names"]
            if type(pnames_opt) is not list and type(pnames_opt) is not tuple:
                raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options.param_names must be an exact list or tuple")
            for name_index, name in enumerate(pnames_opt):
                _check_clean_text(name, f"expected_runtime.optimizer_groups[{g_idx}].options.param_names[{name_index}]")
            if len(pnames_opt) != len(pnames) or any(actual != expected for actual, expected in zip(pnames_opt, pnames)):
                raise CheckpointValidationError(f"expected_runtime.optimizer_groups[{g_idx}].options.param_names does not match parameter_names")

    # Validate rng_role_bindings: must have all 10 CONFIGURED_RNG_COMPONENTS
    role_bindings = expected_runtime["rng_role_bindings"]
    if type(role_bindings) is not dict:
        raise CheckpointValidationError("expected_runtime.rng_role_bindings must be a dict")
    if set(role_bindings.keys()) != set(CONFIGURED_RNG_COMPONENTS):
        missing_rb = _safe_sort_keys(set(CONFIGURED_RNG_COMPONENTS) - set(role_bindings.keys()))
        extra_rb = _safe_sort_keys(set(role_bindings.keys()) - set(CONFIGURED_RNG_COMPONENTS))
        raise CheckpointValidationError(f"expected_runtime.rng_role_bindings keys mismatch. Missing: {missing_rb}, Extra: {extra_rb}")

    for rk, gids in role_bindings.items():
        if type(gids) is not list:
            raise CheckpointValidationError(f"expected_runtime.rng_role_bindings[{_safe_repr(rk)}] must be a list of IDs")
        seen_gids = set()
        for gid in gids:
            clean_gid = _check_clean_text(gid, f"expected_runtime.rng_role_bindings[{rk!r}] generator_id")
            if clean_gid in seen_gids:
                raise CheckpointValidationError(f"Duplicate generator ID {_safe_repr(clean_gid)} in expected_runtime.rng_role_bindings[{rk!r}]")
            seen_gids.add(clean_gid)

    # Validate rng_generator_bindings
    gen_bindings = expected_runtime["rng_generator_bindings"]
    if type(gen_bindings) is not dict:
        raise CheckpointValidationError("expected_runtime.rng_generator_bindings must be a dict")
    for gid, gdesc in gen_bindings.items():
        _check_clean_text(gid, "expected_runtime.rng_generator_bindings key")
        if type(gdesc) is not dict or set(gdesc.keys()) != {"codec", "worker_id", "seed_material"}:
            raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] malformed schema")
        codec = gdesc["codec"]
        if codec not in SUPPORTED_CODECS:
            raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] unsupported codec {_safe_repr(codec)}")
        wid = gdesc["worker_id"]
        if wid is not None:
            _check_strict_int(wid, f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].worker_id", min_val=0)
        s_mat = gdesc["seed_material"]
        if codec == "torch.cpu.Generator.native.v1":
            if type(s_mat) is not int or type(s_mat) is bool or s_mat < 0:
                raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material must be exact non-negative int")
            if s_mat > 0xFFFFFFFFFFFFFFFF:
                raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material exceeds uint64")
        elif codec == "python.random.MT19937.v3":
            if type(s_mat) is not int or type(s_mat) is bool or s_mat < 0:
                raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material must be exact non-negative int")
        elif codec in ("numpy.random.RandomState.MT19937.v1", "numpy.random.PCG64.v1"):
            if type(s_mat) is list:
                if wid is None:
                    raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] word list only allowed for worker contexts")
                if not s_mat:
                    raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] seed word list cannot be empty")
                for w in s_mat:
                    _check_strict_int(w, f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] seed word", min_val=0, max_val=0xFFFFFFFF)
            else:
                _check_strict_int(s_mat, f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}].seed_material", min_val=0)
                if codec == "numpy.random.RandomState.MT19937.v1" and s_mat > 0xFFFFFFFF:
                    raise CheckpointValidationError(f"expected_runtime.rng_generator_bindings[{_safe_repr(gid)}] RandomState seed exceeds uint32")


# ---------------------------------------------------------------------------
# Semantic Validation Phase
# ---------------------------------------------------------------------------

def _validate_semantic_payload(
    payload: Dict[str, Any],
    *,
    expected_identity: Dict[str, Any],
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

    if payload["schema_version"] != PAYLOAD_SCHEMA_VERSION:
        raise CheckpointValidationError(
            f"payload.schema_version mismatch: {_safe_repr(payload['schema_version'])} != {PAYLOAD_SCHEMA_VERSION!r}"
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

    # Reconstruct and validate Identity & Provenance through accepted D1 APIs
    raw_actual_id = payload["identity"]
    if type(raw_actual_id) is not dict:
        raise CheckpointValidationError("payload.identity must be a dict")
    actual_identity = CanonicalResumeIdentity.from_dict(raw_actual_id)
    exp_identity = CanonicalResumeIdentity.from_dict(expected_identity)

    validate_canonical_identity(actual_identity, exp_identity)
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


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def validate_checkpoint_payload(
    payload: Any,
    *,
    expected_identity: Any,
    expected_provenance: Any,
    expected_model: Any,
    expected_runtime: Any,
    resource_policy: Any,
) -> Dict[str, Any]:
    """Validate decoded in-memory checkpoint payload against mandatory expected contracts.

    Requires keyword arguments for serialized expected_identity, expected_provenance,
    expected_model, expected_runtime, and resource_policy.
    Returns an independently owned validated state snapshot eligible for restoration,
    or raises CheckpointValidationError.
    """
    # 1. Validate resource policy schema and bounds
    parsed_policy = validate_resource_policy(resource_policy)

    # 2. Complete bounded generic type/resource preflight BEFORE schema traversal, D1 calls, or copying
    preflight_resource_accounting(payload, parsed_policy, role="payload")
    preflight_resource_accounting(expected_identity, parsed_policy, role="expected_identity")
    preflight_resource_accounting(expected_provenance, parsed_policy, role="expected_provenance")
    preflight_resource_accounting(expected_model, parsed_policy, role="expected_model")
    preflight_resource_accounting(expected_runtime, parsed_policy, role="expected_runtime")

    # 3. Validate expected descriptors types and schemas
    if type(expected_identity) is not dict:
        raise CheckpointValidationError(
            f"expected_identity must be an exact serialized dict, got {_safe_repr(expected_identity)}"
        )
    if type(expected_provenance) is not dict:
        raise CheckpointValidationError(
            f"expected_provenance must be an exact dict, got {_safe_repr(expected_provenance)}"
        )
    _validate_expected_model_schema(expected_model)
    _validate_expected_runtime_schema(expected_runtime)

    # 4. Complete RAW semantic validation on raw payload
    _validate_semantic_payload(
        payload,
        expected_identity=expected_identity,
        expected_provenance=expected_provenance,
        expected_model=expected_model,
        expected_runtime=expected_runtime,
    )

    # 5. Make owned output copy
    owned_snapshot = _make_owned_output_copy(payload)

    # 6. Validate owned snapshot semantics and resource preflight
    preflight_resource_accounting(owned_snapshot, parsed_policy, role="owned_snapshot")
    _validate_semantic_payload(
        owned_snapshot,
        expected_identity=expected_identity,
        expected_provenance=expected_provenance,
        expected_model=expected_model,
        expected_runtime=expected_runtime,
    )

    # 7. Return independently owned validated snapshot
    return owned_snapshot
