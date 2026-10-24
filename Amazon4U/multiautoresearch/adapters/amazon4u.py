"""Amazon4U O-OFFLINE orchestration adapter.

Stdlib-only profile, pure renderer, bounded manifests, streaming hydration,
typed transport, immutable state management, and bounded artifact collection.
"""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import uuid
import zlib

# ---------------------------------------------------------------------------
# Schema Constants (OP01, OP11, OP13, OP17)
# ---------------------------------------------------------------------------

SCHEMA_O_SPEC = "amazon4u-O-spec-v1"
SCHEMA_H_SPEC = "amazon4u-H-spec-v1"
SCHEMA_PLAN = "amazon4u-O-plan-v1"
SCHEMA_O_AUTHORIZATION = "amazon4u-O-authorization-v1"
SCHEMA_H_AUTHORIZATION = "amazon4u-H-authorization-v1"
SCHEMA_RECORD = "amazon4u-O-record-v1"
SCHEMA_MANIFEST = "amazon4u-O-manifest-v1"
SCHEMA_ENVELOPE = "amazon4u-H-run-v1"
SCHEMA_H_RUN_ENVELOPE = "amazon4u-H-run-v1"

ENVELOPE_KEYS = frozenset({
    "schema_version", "run_id", "approval_id", "profile_id", "method",
    "identity_hash", "device", "accelerator", "bundle_sha256",
    "initial_state_sha256", "determinism_policy",
})

H_RESULT_KEYS = frozenset({
    "schema_version", "status", "acceptance_scope", "deployment_decision",
    "run_spec", "run_envelope", "device_capability", "method", "epoch",
    "actual_Adam_steps", "selection", "mixture", "validation_report",
    "remaining_readiness_gates", "test_execution",
})

# Backward compatibility / strict separation
SCHEMA_SPEC = SCHEMA_O_SPEC
SCHEMA_AUTHORIZATION = SCHEMA_O_AUTHORIZATION

READINESS_GATES = [
    "GPU-READY separately authorized and verified",
    "O-OFFLINE transport and packaging separately reviewed",
    "Production LightGCN initialization decision",
    "Approved environment and full-data feasibility profiling",
    "Comparative execution/search freeze and final-test authorization",
]

METHODS = {"LightGCN", "Heterogeneous KG+GNN", "CorePopularity", "KG-only"}
ALLOWED_STAGES = {"training_from_scratch"}
FORBIDDEN_STAGES = {"prepare", "eval", "evaluate", "final_test", "LMstage", "pretraining", "sft"}
ALLOWED_MODES = {"fixture_verification"}
ALLOWED_DATA_KINDS = {"owned_synthetic_packet"}
ALLOWED_PROFILE_IDS = {"H-REF-synthetic-cpu-v1", "amazon4u"}

SPEC_KEYS = {
    "schema_version", "run_id", "approval_id", "stage", "mode", "data_kind",
    "method", "device", "accelerator", "profile_id", "epochs", "max_adam_steps",
}

# Registered evaluation metrics (Configs evaluation-metrics.json)
FROZEN_REGISTERED_SEVEN_METRICS = (
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "HitRate@5",
    "HitRate@10",
    "HitRate@20",
    "CatalogCoverage@10",
)
REGISTERED_SEVEN_METRICS = frozenset(FROZEN_REGISTERED_SEVEN_METRICS)

LEGITIMATE_H_STATUSES = {
    "deterministic_validation_complete",
    "fixture_budget_complete",
    "scientific_terminal",
}

FORBIDDEN_PROXY_KEYS = {
    "eval_score",
    "val_bpb",
    "recall",
    "precision",
    "mrr",
    "map",
    "recall_10",
    "precision_10",
    "mrr_10",
    "map_10",
    "train_loss",
    "learning_rate",
}

# Strict budget caps
MAX_FILE_BYTES = 2097152       # 2 MiB per file
MAX_AGGREGATE_BYTES = 8388608  # 8 MiB aggregate
MAX_CONCURRENT_FILES = 128     # 128 files maximum
MAX_PAGES = 8                  # 8 paged listings maximum
MAX_FAKE_POLLS = 5             # 5 polls maximum
MAX_CHUNK_BYTES = 65536        # 64 KiB chunk size
MAX_ENCODED_BYTES = 4194304    # 4 MiB encoded payload cap


def validate_raw_relative_path(rel_path: str) -> None:
    """Validate relative path strictly on raw string BEFORE Path normalization.

    Rejects backslashes, drive letters, leading slashes, traversal (..),
    relative dot aliases (.), empty components, and duplicate separators (//).
    """
    if not isinstance(rel_path, str) or not rel_path:
        raise Amazon4UError("Path must be a non-empty string")
    if "\\" in rel_path or ":" in rel_path:
        raise Amazon4UError(f"Backslash or drive separator not permitted: {rel_path}")
    if rel_path.startswith("/"):
        raise Amazon4UError(f"Absolute path not permitted: {rel_path}")
    raw_parts = rel_path.split("/")
    if any(part in ("..", ".", "") for part in raw_parts):
        raise Amazon4UError(f"Path traversal, relative alias, or duplicate separator not permitted: {rel_path}")

VALID_STATUS_TOKENS = {
    "queued",
    "running",
    "complete",
    "error",
    "cancelacknowledged",
    "canceled",
    "submission_unknown",
    "poll_deadline_exceeded",
}
TERMINAL_STATUS_TOKENS = {
    "complete",
    "error",
    "cancelacknowledged",
    "canceled",
    "submission_unknown",
    "poll_deadline_exceeded",
}

# The 28 canonical H bundle paths that must be preserved
H_BUNDLE_PATHS = (
    "scripts/checkpoint_identity.py",
    "scripts/learned_models.py",
    "scripts/recommendation_core.py",
    "scripts/recommendation_evaluation.py",
    "scripts/differentiable_models.py",
    "scripts/build_kg.py",
    "scripts/audit_kg_inputs.py",
    "scripts/audit_candidates.py",
    "scripts/validate_kg_splits.py",
    "scripts/visualize_kg.py",
    "scripts/inspect_data.py",
    "scripts/download_5core.py",
    "configs/compute-plan.json",
    "configs/dataset-scope.json",
    "configs/evaluation-metrics.json",
    "configs/kg-v1.json",
    "configs/model-suite.json",
    "configs/protocol-lifecycle.json",
    "configs/reproducibility.json",
    "configs/training-semantics.json",
    "scripts/checkpoint_payload.py",
    "scripts/checkpoint_storage.py",
    "scripts/training_harness.py",
    "scripts/training_data.py",
    "scripts/training_entrypoint.py",
    "tests/test_training_harness.py",
    "tests/test_training_data.py",
    "docs/training-harness.md",
)

# Additional O outer closure paths
O_ADDITIONAL_CLOSURE_PATHS = (
    "multiautoresearch/post-training/scripts/kaggle_job.py",
    "multiautoresearch/adapters/__init__.py",
    "multiautoresearch/adapters/amazon4u.py",
    "multiautoresearch/adapters/amazon4u_kernel.py",
)

REQUIRED_CLOSURE_PATHS = tuple(sorted(set(H_BUNDLE_PATHS) | set(O_ADDITIONAL_CLOSURE_PATHS)))


class Amazon4UError(ValueError):
    """Explicit Amazon4U boundary, authority, format, or limit violation."""


class SubmissionUnknownError(Amazon4UError):
    """Submission outcome is ambiguous; automatic resubmission is prohibited."""


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _rename_no_replace_linux(src: Path, dst: Path) -> bool:
    """Attempt atomic rename with RENAME_NOREPLACE using libc.renameat2 if available."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except (AttributeError, OSError, TypeError):
        return False

    if not hasattr(libc, "renameat2"):
        return False

    try:
        libc.renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        libc.renameat2.restype = ctypes.c_int
        src_bytes = os.fsencode(src)
        dst_bytes = os.fsencode(dst)
        ret = libc.renameat2(_AT_FDCWD, src_bytes, _AT_FDCWD, dst_bytes, _RENAME_NOREPLACE)
        if ret == 0:
            return True
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))
        if err in (errno.ENOSYS, errno.EINVAL):
            return False
        raise OSError(err, os.strerror(err), str(dst))
    except (AttributeError, TypeError):
        return False


def atomic_move_no_replace(src: Path, dst: Path) -> None:
    """Atomically move src to dst without clobbering an existing destination.

    Uses Linux renameat2(RENAME_NOREPLACE) if supported, or POSIX os.link + os.unlink fallback for files.
    Never overwrites destination; fails closed with FileExistsError if dst exists.
    """
    if dst.exists():
        raise FileExistsError(errno.EEXIST, f"Destination file already exists; overwrite forbidden: {dst}")

    if _rename_no_replace_linux(src, dst):
        return

    # POSIX link + unlink atomic no-replace fallback (src and dst must be on same filesystem)
    if not src.is_dir():
        try:
            os.link(src, dst)
        except FileExistsError:
            raise
        except OSError as err:
            if err.errno == errno.EEXIST:
                raise FileExistsError(errno.EEXIST, f"Destination file already exists: {dst}") from err
            raise

        try:
            os.unlink(src)
        except OSError:
            pass
        return

    if dst.exists():
        raise FileExistsError(errno.EEXIST, f"Destination directory already exists: {dst}")
    raise OSError(errno.ENOSYS, "Atomic directory no-replace requires renameat2(RENAME_NOREPLACE) on this platform")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePacket:
    """A bounded source file packet with normalized relative path and content."""
    rel_path: str
    content: bytes

    def __post_init__(self) -> None:
        validate_raw_relative_path(self.rel_path)
        if not isinstance(self.content, bytes):
            raise Amazon4UError("Source packet content must be bytes")
        if len(self.content) > MAX_FILE_BYTES:
            raise Amazon4UError(f"Source packet exceeds maximum file size {MAX_FILE_BYTES}: {len(self.content)}")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def size(self) -> int:
        return len(self.content)


@dataclass
class RenderedPlan:
    """Deterministic in-memory bundle representation."""
    spec: dict[str, object]
    authorization: dict[str, object]
    metadata: dict[str, object]
    files: dict[str, str]  # rel_path -> base64 gzip payload
    kernel_script: str
    outer_manifest: dict[str, object]
    plan_digest: str
    owner: str
    slug: str
    attempt_id: str
    rendered_at: str


@dataclass(frozen=True)
class ApprovedOfflineContext:
    """Retained independent O-owned orchestration context."""
    canonical_h_spec: dict[str, object]
    h_authorization: dict[str, object]
    expected_h_envelope: dict[str, object]
    expected_h_bundle: dict[str, object]
    expected_o_outer: dict[str, object]
    expected_input_identity: dict[str, object]
    expected_o_environment: dict[str, object]
    submission_binding: dict[str, object]
    expected_output_inventory: dict[str, object]
    limits: dict[str, object]


def _h_bundle_files_digest(files: dict[str, object]) -> str:
    """H owned_digest encoding restricted to validated manifest dict/str/int values."""
    h = hashlib.sha256()

    def blob(tag, data):
        h.update(tag)
        h.update(struct.pack("!Q", len(data)))
        h.update(data)

    def visit(value):
        if type(value) is dict:
            h.update(b"D")
            h.update(struct.pack("!Q", len(value)))
            for key in sorted(value):
                visit(key)
                visit(value[key])
        elif type(value) is int:
            blob(b"I", str(value).encode("ascii"))
        elif type(value) is str:
            blob(b"S", value.encode("utf-8"))
        else:
            raise Amazon4UError("Invalid H manifest digest value")

    visit(files)
    return h.hexdigest()


def _retain_h_bundle_sources(sources, expected_bundle) -> dict[str, bytes]:
    """Retain physical H bytes before any provider callback or filesystem mutation."""
    if type(sources) is not dict or not sources:
        raise Amazon4UError("H sources require a nonempty dictionary")
    retained = dict(sources)
    files = expected_bundle["files"]
    if set(retained) != set(files):
        raise Amazon4UError("H sources must match the exact retained H28 manifest")
    for path, content in retained.items():
        if type(content) is not bytes or len(content) != files[path]["bytes"]:
            raise Amazon4UError("H source type or physical length mismatch")
        if hashlib.sha256(content).hexdigest() != files[path]["sha256"]:
            raise Amazon4UError("H source physical SHA256 mismatch")
    return retained


def validate_canonical_h_envelope(
    envelope: dict[str, object],
    *,
    expected_spec: dict[str, object] | None = None,
    expected_bundle: dict[str, object] | None = None,
    expected_envelope: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate canonical H run envelope matching frozen check_envelope and execute_fixture."""
    if type(envelope) is not dict:
        raise Amazon4UError("Run envelope must be a plain dictionary")
    if set(envelope.keys()) != ENVELOPE_KEYS:
        missing = ENVELOPE_KEYS - set(envelope.keys())
        extra = set(envelope.keys()) - ENVELOPE_KEYS
        raise Amazon4UError(f"Run envelope must have exact 11 keys. Missing: {missing}, Extra: {extra}")
    for key, val in envelope.items():
        if type(val) is not str or not val or len(val) > 256:
            raise Amazon4UError(f"Run envelope value for '{key}' must be a bounded nonempty plain string <= 256 characters")
    if envelope["schema_version"] != SCHEMA_H_RUN_ENVELOPE:
        raise Amazon4UError(f"Run envelope schema_version must be '{SCHEMA_H_RUN_ENVELOPE}', got '{envelope['schema_version']}'")
    if envelope["profile_id"] != "H-REF-synthetic-cpu-v1":
        raise Amazon4UError(f"Run envelope profile_id must be 'H-REF-synthetic-cpu-v1', got '{envelope['profile_id']}'")
    if envelope["determinism_policy"] != "private_rng_bounded_cpu_no_global_setters":
        raise Amazon4UError(f"Run envelope determinism_policy must be 'private_rng_bounded_cpu_no_global_setters', got '{envelope['determinism_policy']}'")
    for key in ("identity_hash", "bundle_sha256", "initial_state_sha256"):
        val = envelope[key]
        if len(val) != 64 or any(c not in "0123456789abcdef" for c in val):
            raise Amazon4UError(f"Run envelope '{key}' must be a 64-character lowercase hexadecimal digest")
    if expected_spec is not None:
        for key in ("run_id", "approval_id", "method", "device", "accelerator", "profile_id"):
            if envelope[key] != expected_spec.get(key):
                raise Amazon4UError(f"Run envelope '{key}' ({envelope[key]}) mismatch with spec ({expected_spec.get(key)})")
    if expected_bundle is not None:
        exp_bundle_sha = expected_bundle.get("sha256")
        if exp_bundle_sha and envelope["bundle_sha256"] != exp_bundle_sha:
            raise Amazon4UError("Run envelope bundle_sha256 mismatch with expected_bundle sha256")
    if expected_envelope is not None:
        if envelope != expected_envelope:
            raise Amazon4UError("Run envelope differs from independently expected run envelope")
    return copy.deepcopy(envelope)


def validate_approved_offline_context(context: ApprovedOfflineContext | dict[str, object] | None) -> ApprovedOfflineContext:
    """Validate and retain independent ApprovedOfflineContext safely copied against caller mutation."""
    if context is None:
        raise Amazon4UError("Mandatory independent ApprovedOfflineContext required; cannot be None")

    if isinstance(context, dict):
        required_fields = {
            "canonical_h_spec", "h_authorization", "expected_h_envelope",
            "expected_h_bundle", "expected_o_outer", "expected_input_identity",
            "expected_o_environment", "submission_binding", "expected_output_inventory",
            "limits",
        }
        missing = required_fields - set(context.keys())
        if missing:
            raise Amazon4UError(f"Missing required ApprovedOfflineContext fields: {missing}")
        raw_ctx = context
    elif isinstance(context, ApprovedOfflineContext) or getattr(getattr(context, "__class__", None), "__name__", "") == "ApprovedOfflineContext" or hasattr(context, "canonical_h_spec"):
        raw_ctx = {
            "canonical_h_spec": getattr(context, "canonical_h_spec", None),
            "h_authorization": getattr(context, "h_authorization", None),
            "expected_h_envelope": getattr(context, "expected_h_envelope", None),
            "expected_h_bundle": getattr(context, "expected_h_bundle", None),
            "expected_o_outer": getattr(context, "expected_o_outer", None),
            "expected_input_identity": getattr(context, "expected_input_identity", None),
            "expected_o_environment": getattr(context, "expected_o_environment", None),
            "submission_binding": getattr(context, "submission_binding", None),
            "expected_output_inventory": getattr(context, "expected_output_inventory", None),
            "limits": getattr(context, "limits", None),
        }
        for k, v in raw_ctx.items():
            if v is None:
                raise Amazon4UError(f"Missing required ApprovedOfflineContext attribute: {k}")
    else:
        raise Amazon4UError("context must be an ApprovedOfflineContext or dictionary")

    # Safely deep-copy all dictionaries against caller aliases/mutation
    ctx = ApprovedOfflineContext(
        canonical_h_spec=copy.deepcopy(raw_ctx["canonical_h_spec"]),
        h_authorization=copy.deepcopy(raw_ctx["h_authorization"]),
        expected_h_envelope=copy.deepcopy(raw_ctx["expected_h_envelope"]),
        expected_h_bundle=copy.deepcopy(raw_ctx["expected_h_bundle"]),
        expected_o_outer=copy.deepcopy(raw_ctx["expected_o_outer"]),
        expected_input_identity=copy.deepcopy(raw_ctx["expected_input_identity"]),
        expected_o_environment=copy.deepcopy(raw_ctx["expected_o_environment"]),
        submission_binding=copy.deepcopy(raw_ctx["submission_binding"]),
        expected_output_inventory=copy.deepcopy(raw_ctx["expected_output_inventory"]),
        limits=copy.deepcopy(raw_ctx["limits"]),
    )

    validate_h_spec(ctx.canonical_h_spec)

    if type(ctx.h_authorization) is not dict:
        raise Amazon4UError("h_authorization must be a plain dictionary")
    for key in ("schema_version", "run_id", "approval_id", "scope"):
        value = ctx.h_authorization.get(key)
        if type(value) is not str or not value or len(value) > 256:
            raise Amazon4UError("h_authorization must contain bounded nonempty plain strings")
    if ctx.h_authorization.get("schema_version") != SCHEMA_H_AUTHORIZATION:
        raise Amazon4UError(f"h_authorization schema_version must be '{SCHEMA_H_AUTHORIZATION}', got '{ctx.h_authorization.get('schema_version')}'")
    validate_amazon4u_authorization(ctx.h_authorization, actual_spec=ctx.canonical_h_spec)

    # Distinct H28 manifest validation
    h_bundle = ctx.expected_h_bundle
    if type(h_bundle) is not dict or set(h_bundle) != {"schema_version", "files", "sha256"}:
        raise Amazon4UError("expected_h_bundle must have exactly schema_version, files, sha256")
    if h_bundle.get("schema_version") != "amazon4u-H-bundle-v1":
        raise Amazon4UError("expected_h_bundle schema_version must be 'amazon4u-H-bundle-v1'")
    h_files = h_bundle.get("files")
    if type(h_files) is not dict:
        raise Amazon4UError("expected_h_bundle 'files' must be a dictionary")
    if set(h_files.keys()) != set(H_BUNDLE_PATHS):
        missing_h = set(H_BUNDLE_PATHS) - set(h_files.keys())
        extra_h = set(h_files.keys()) - set(H_BUNDLE_PATHS)
        raise Amazon4UError(f"expected_h_bundle must match exactly 28 H bundle paths. Missing: {missing_h}, Extra: {extra_h}")
    for p, info in h_files.items():
        if (
            type(info) is not dict
            or set(info) != {"bytes", "sha256"}
            or "sha256" not in info
            or not isinstance(info["sha256"], str)
            or len(info["sha256"]) != 64
            or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None
            or "bytes" not in info
            or type(info["bytes"]) is not int
            or isinstance(info["bytes"], bool)
            or info["bytes"] < 0
            or info["bytes"] > MAX_FILE_BYTES
        ):
            raise Amazon4UError(f"expected_h_bundle file entry invalid for {p}: requires int 'bytes' and 64-char hex 'sha256'")
    if sum(info["bytes"] for info in h_files.values()) > MAX_AGGREGATE_BYTES:
        raise Amazon4UError("expected_h_bundle exceeds the aggregate H source bound")
    h_digest = h_bundle.get("sha256")
    if not isinstance(h_digest, str) or re.fullmatch(r"[0-9a-f]{64}", h_digest) is None:
        raise Amazon4UError("expected_h_bundle must supply 64-character hex 'sha256' overall digest")
    if h_digest != _h_bundle_files_digest(h_files):
        raise Amazon4UError("expected_h_bundle semantic SHA256 differs from owned_digest(files)")

    # Canonical 11-field H run envelope validation matching frozen check_envelope / execute_fixture
    validate_canonical_h_envelope(
        ctx.expected_h_envelope,
        expected_spec=ctx.canonical_h_spec,
        expected_bundle=ctx.expected_h_bundle,
    )

    # Distinct O32 outer closure manifest validation
    o_outer = ctx.expected_o_outer
    if not isinstance(o_outer, dict):
        raise Amazon4UError("expected_o_outer must be a dictionary")
    o_files = o_outer.get("files")
    if not isinstance(o_files, dict):
        raise Amazon4UError("expected_o_outer 'files' must be a dictionary")
    if set(o_files) != set(REQUIRED_CLOSURE_PATHS):
        raise Amazon4UError("expected_o_outer must contain exactly the 32 closure paths")
    for info in o_files.values():
        if (type(info) is not dict or set(info) != {"size", "sha256"}
                or type(info["size"]) is not int or not 0 <= info["size"] <= MAX_FILE_BYTES
                or type(info["sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None):
            raise Amazon4UError("expected_o_outer file metadata violates type, size or SHA bounds")
    total = sum(info["size"] for info in o_files.values())
    if total > MAX_AGGREGATE_BYTES:
        raise Amazon4UError("expected_o_outer exceeds the aggregate source bound")
    if (o_outer.get("schema_version") != SCHEMA_MANIFEST
            or type(o_outer.get("file_count")) is not int or o_outer["file_count"] != len(o_files)
            or type(o_outer.get("aggregate_size")) is not int or o_outer["aggregate_size"] != total
            or o_outer.get("manifest_digest") != hashlib.sha256(json.dumps(o_files, sort_keys=True).encode("utf-8")).hexdigest()):
        raise Amazon4UError("expected_o_outer manifest schema, counts or digest mismatch")
    if "generated_runtime" in o_outer:
        grt = o_outer["generated_runtime"]
        if not isinstance(grt, dict) or grt.get("code_file") != "kernel.py" or "sha256" not in grt:
            raise Amazon4UError("expected_o_outer generated_runtime invalid")
    if "environment" in o_outer:
        o_env = o_outer["environment"]
        if not isinstance(o_env, dict) or o_env.get("device") != "cpu" or o_env.get("accelerator") not in ("none", ""):
            raise Amazon4UError("expected_o_outer environment violates CPU-only scope")

    # Input identity validation
    inp_id = ctx.expected_input_identity
    if not isinstance(inp_id, dict):
        raise Amazon4UError("expected_input_identity must be a dictionary")
    if "spec" not in inp_id or "authorization" not in inp_id:
        raise Amazon4UError("expected_input_identity must contain 'spec' and 'authorization'")
    if inp_id["spec"] != ctx.canonical_h_spec:
        raise Amazon4UError("expected_input_identity spec mismatch with canonical_h_spec")
    if inp_id["authorization"] != ctx.h_authorization:
        raise Amazon4UError("expected_input_identity authorization mismatch with h_authorization")

    sb = ctx.submission_binding
    if not isinstance(sb, dict):
        raise Amazon4UError("submission_binding must be a dictionary")
    if "version" not in sb or type(sb["version"]) is not int or isinstance(sb["version"], bool) or sb["version"] < 1:
        raise Amazon4UError("submission_binding must supply positive integer version")
    if "timeout" not in sb or type(sb["timeout"]) is not int or isinstance(sb["timeout"], bool) or sb["timeout"] < 1:
        raise Amazon4UError("submission_binding must supply positive integer timeout")

    env = ctx.expected_o_environment
    if not isinstance(env, dict):
        raise Amazon4UError("expected_o_environment must be a dictionary")
    if env.get("device", "cpu") != "cpu" or env.get("cuda_available", False) or env.get("accelerator", "none") not in ("none", ""):
        raise Amazon4UError("expected_o_environment violates CPU-only offline scope")
    caps = env.get("capabilities")
    if not isinstance(caps, dict):
        raise Amazon4UError("expected_o_environment must supply capabilities dictionary")
    required_caps = {
        "package", "version", "nativeIO", "ABI", "dependency",
        "CPU", "runtime", "resource", "callable", "bootstrap",
    }
    missing_caps = required_caps - set(caps.keys())
    if missing_caps:
        raise Amazon4UError(f"expected_o_environment capabilities missing required keys: {missing_caps}")

    # Output inventory validation
    inv = ctx.expected_output_inventory
    if not isinstance(inv, dict) or not inv:
        raise Amazon4UError("expected_output_inventory must be a non-empty dictionary")
    required_inv_roles = {"plan", "h_result", "run_envelope", "source_manifest", "log", "checkpoint"}
    missing_roles = required_inv_roles - set(inv.keys())
    if missing_roles:
        raise Amazon4UError(f"expected_output_inventory missing required roles: {missing_roles}")

    limits = ctx.limits
    if not isinstance(limits, dict):
        raise Amazon4UError("limits must be a dictionary")
    if "max_file_bytes" not in limits or type(limits["max_file_bytes"]) is not int or isinstance(limits["max_file_bytes"], bool) or limits["max_file_bytes"] <= 0:
        raise Amazon4UError("limits max_file_bytes must be a positive integer")
    if "max_aggregate_bytes" not in limits or type(limits["max_aggregate_bytes"]) is not int or isinstance(limits["max_aggregate_bytes"], bool) or limits["max_aggregate_bytes"] <= 0:
        raise Amazon4UError("limits max_aggregate_bytes must be a positive integer")

    return ctx


class BoundedLogReader:
    """Reader protocol interface for streaming kernel logs."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self._closed = False

    def read(self, max_bytes: int, deadline: float | None = None, clock_fn=None, cancellation=None) -> bytes:
        if self._closed:
            raise Amazon4UError("Log reader already closed")
        now = clock_fn() if clock_fn else time.monotonic()
        if deadline is not None and now > deadline:
            raise Amazon4UError("Deadline exceeded reading log chunk")
        if cancellation is not None and cancellation():
            raise Amazon4UError("Operation canceled")
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        now_after = clock_fn() if clock_fn else time.monotonic()
        if deadline is not None and now_after > deadline:
            raise Amazon4UError("Deadline exceeded after reading log chunk")
        if cancellation is not None and cancellation():
            raise Amazon4UError("Operation canceled after reading log chunk")
        if len(chunk) > max_bytes:
            raise Amazon4UError(f"Provider returned chunk size {len(chunk)} exceeding requested max {max_bytes}")
        return chunk

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Spec & Authorization Validation (OP01, OP06, OP18)
# ---------------------------------------------------------------------------

def validate_h_spec(
    spec: dict[str, object],
    *,
    expected_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate canonical H run specification matching scripts/training_entrypoint.py."""
    if type(spec) is not dict:
        raise Amazon4UError("H run spec must be a plain dictionary")

    if set(spec.keys()) != SPEC_KEYS:
        raise Amazon4UError(f"Run spec keys must match exact H allowlist; got: {set(spec.keys())}")

    if spec["schema_version"] != SCHEMA_H_SPEC:
        raise Amazon4UError(f"Unsupported H run spec schema_version: {spec['schema_version']}; expected {SCHEMA_H_SPEC}")

    if spec.get("stage") != "training_from_scratch":
        raise Amazon4UError(f"Unauthorized stage '{spec.get('stage')}'; only training_from_scratch authorized")

    if spec.get("mode") != "fixture_verification":
        raise Amazon4UError(f"Mode must be 'fixture_verification', got: '{spec.get('mode')}'")

    if spec.get("data_kind") != "owned_synthetic_packet":
        raise Amazon4UError(f"Data kind must be 'owned_synthetic_packet', got: '{spec.get('data_kind')}'")

    method = str(spec.get("method", ""))
    if method not in METHODS:
        raise Amazon4UError(f"Method '{method}' not in allowed methods: {METHODS}")

    device = spec.get("device")
    if device != "cpu":
        raise Amazon4UError(f"Only CPU device authorized in O-OFFLINE scope; requested: '{device}'")

    accelerator = spec.get("accelerator")
    if accelerator != "none":
        raise Amazon4UError(f"Only 'none' accelerator authorized in CPU reference scope; requested: '{accelerator}'")

    profile_id = str(spec.get("profile_id", ""))
    if profile_id != "H-REF-synthetic-cpu-v1":
        raise Amazon4UError(f"Invalid H profile_id '{profile_id}'; must be 'H-REF-synthetic-cpu-v1'")

    for k in SPEC_KEYS - {"epochs", "max_adam_steps"}:
        v = spec[k]
        if type(v) is not str or not v or len(v) > 256:
            raise Amazon4UError(f"Spec string field '{k}' must be non-empty and <= 256 chars")

    epochs = spec.get("epochs")
    max_adam_steps = spec.get("max_adam_steps")

    # Strict type checks: Python bool is a subclass of int, so reject bool-as-int explicitly
    if type(epochs) is not int or isinstance(epochs, bool):
        raise Amazon4UError("epochs must be an integer, not boolean or other type")
    if type(max_adam_steps) is not int or isinstance(max_adam_steps, bool):
        raise Amazon4UError("max_adam_steps must be an integer, not boolean or other type")

    if not 0 <= epochs <= 4:
        raise Amazon4UError(f"epochs out of range [0, 4]: {epochs}")

    if method in ("CorePopularity", "KG-only"):
        if epochs != 0 or max_adam_steps != 0:
            raise Amazon4UError(f"Deterministic baseline {method} cannot request updates (epochs={epochs}, steps={max_adam_steps})")
    else:
        if epochs < 1 or max_adam_steps < 1:
            raise Amazon4UError(f"Learned method {method} must budget >= 1 epoch and step")

    if expected_spec is not None:
        if spec != expected_spec:
            raise Amazon4UError("Run spec differs from independently expected approved spec")

    return copy.deepcopy(spec)


def get_canonical_h_spec(spec: dict[str, object]) -> dict[str, object]:
    """Extract or derive canonical H run specification from a validated spec dictionary."""
    if not isinstance(spec, dict):
        raise Amazon4UError("Spec must be a dictionary")
    schema_ver = spec.get("schema_version")
    if schema_ver == SCHEMA_H_SPEC:
        return validate_h_spec(spec)
    if schema_ver == SCHEMA_O_SPEC:
        validated_o = validate_amazon4u_spec(spec)
        return copy.deepcopy(validated_o["h_spec"])
    raise Amazon4UError(
        f"Cannot derive canonical H spec from unsupported schema_version: {schema_ver!r}; "
        f"only canonical H ('{SCHEMA_H_SPEC}') or validated O spec wrapper ('{SCHEMA_O_SPEC}') supported"
    )


def validate_amazon4u_spec(
    spec: dict[str, object],
    *,
    expected_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate Amazon4U run specification (either canonical H spec or O spec wrapper)."""
    if not isinstance(spec, dict):
        raise Amazon4UError("Run spec must be a dictionary")

    schema_ver = spec.get("schema_version")
    if schema_ver == SCHEMA_H_SPEC:
        return validate_h_spec(spec, expected_spec=expected_spec)

    if schema_ver == SCHEMA_O_SPEC:
        required_o_keys = {
            "schema_version", "run_id", "approval_id", "stage", "mode", "data_kind",
            "method", "device", "accelerator", "profile_id", "epochs", "max_adam_steps",
        }
        allowed_o_keys = required_o_keys | {"h_spec"}
        if not required_o_keys.issubset(set(spec.keys())) or not set(spec.keys()).issubset(allowed_o_keys):
            raise Amazon4UError(f"O spec keys invalid: {set(spec.keys())}")

        stage = str(spec.get("stage", ""))
        if stage in FORBIDDEN_STAGES or stage not in ALLOWED_STAGES:
            raise Amazon4UError(f"Forbidden or unauthorized stage '{stage}'")

        if spec.get("mode") not in ALLOWED_MODES:
            raise Amazon4UError(f"Mode must be 'fixture_verification', got: '{spec.get('mode')}'")

        if spec.get("data_kind") not in ALLOWED_DATA_KINDS:
            raise Amazon4UError(f"Data kind must be 'owned_synthetic_packet', got: '{spec.get('data_kind')}'")

        if spec.get("method") not in METHODS:
            raise Amazon4UError(f"Method '{spec.get('method')}' not in allowed methods")

        if str(spec.get("device", "")).lower() != "cpu":
            raise Amazon4UError(f"Only CPU device authorized in O-OFFLINE scope; requested: '{spec.get('device')}'")

        if str(spec.get("accelerator", "")).lower() not in ("none", ""):
            raise Amazon4UError(f"Only 'none' accelerator authorized; requested: '{spec.get('accelerator')}'")

        if spec.get("profile_id") not in ALLOWED_PROFILE_IDS:
            raise Amazon4UError(f"Invalid profile_id '{spec.get('profile_id')}'")

        epochs = spec.get("epochs")
        max_adam_steps = spec.get("max_adam_steps")
        if type(epochs) is not int or isinstance(epochs, bool):
            raise Amazon4UError("epochs must be an int")
        if type(max_adam_steps) is not int or isinstance(max_adam_steps, bool):
            raise Amazon4UError("max_adam_steps must be an int")

        if spec.get("method") in ("CorePopularity", "KG-only"):
            if epochs != 0 or max_adam_steps != 0:
                raise Amazon4UError("Baselines cannot request updates")
        else:
            if epochs < 1 or max_adam_steps < 1:
                raise Amazon4UError("Learned methods must budget >= 1 epoch and step")

        res = copy.deepcopy(spec)
        if "h_spec" in spec and isinstance(spec["h_spec"], dict):
            h_spec = validate_h_spec(spec["h_spec"])
            for field_name in ("run_id", "approval_id", "method", "epochs", "max_adam_steps", "data_kind"):
                if spec.get(field_name) != h_spec.get(field_name):
                    raise Amazon4UError(f"O spec field '{field_name}' differs from h_spec: {spec.get(field_name)} != {h_spec.get(field_name)}")
            res["h_spec"] = h_spec
        else:
            canonical_h = {k: spec[k] for k in SPEC_KEYS if k in spec}
            canonical_h["schema_version"] = SCHEMA_H_SPEC
            canonical_h["profile_id"] = "H-REF-synthetic-cpu-v1"
            res["h_spec"] = validate_h_spec(canonical_h)

        if expected_spec is not None:
            if expected_spec.get("schema_version") == SCHEMA_H_SPEC:
                if res["h_spec"] != expected_spec:
                    raise Amazon4UError("Canonical H spec differs from expected spec")
            elif spec != expected_spec:
                raise Amazon4UError("Run spec differs from expected spec")

        return res

    raise Amazon4UError(f"Unsupported spec schema_version: {schema_ver}")


def validate_amazon4u_authorization(
    auth: dict[str, object],
    *,
    actual_spec: dict[str, object] | None = None,
    expected_auth: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate independent execution authorization."""
    if not isinstance(auth, dict):
        raise Amazon4UError("Authorization must be a dictionary")

    required_keys = {"schema_version", "run_id", "approval_id", "scope", "max_adam_steps"}
    if set(auth.keys()) != required_keys:
        raise Amazon4UError(f"Authorization keys must match exact allowlist; got: {set(auth.keys())}")

    schema_ver = auth["schema_version"]
    if schema_ver not in (SCHEMA_H_AUTHORIZATION, SCHEMA_O_AUTHORIZATION):
        raise Amazon4UError(f"Unsupported authorization schema_version: {schema_ver}")

    if auth["scope"] != "tiny_synthetic_cpu_backward_and_private_restore":
        raise Amazon4UError(f"Authorization scope '{auth['scope']}' not authorized")

    max_steps = auth["max_adam_steps"]
    if type(max_steps) is not int or isinstance(max_steps, bool) or not 0 <= max_steps <= 4:
        raise Amazon4UError("Authorization max_adam_steps must be int in range 0..4 (no bool-as-int)")

    if actual_spec is not None:
        if auth["run_id"] != actual_spec.get("run_id"):
            raise Amazon4UError("Authorization run_id does not match run spec")
        if auth["approval_id"] != actual_spec.get("approval_id"):
            raise Amazon4UError("Authorization approval_id does not match run spec")
        if int(actual_spec.get("max_adam_steps", 0)) > max_steps:
            raise Amazon4UError("Run spec max_adam_steps exceeds authorized budget")

    if expected_auth is not None:
        if auth != expected_auth:
            raise Amazon4UError("Authorization differs from independently expected authorization")

    return copy.deepcopy(auth)


def validate_canonical_h_authorization(
    auth: dict[str, object],
    *,
    actual_spec: dict[str, object] | None = None,
    expected_auth: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate canonical H execution authorization matching scripts/training_entrypoint.py."""
    if not isinstance(auth, dict) or type(auth) is not dict:
        raise Amazon4UError("Authorization must be a plain dictionary")

    required_keys = {"schema_version", "run_id", "approval_id", "scope", "max_adam_steps"}
    if set(auth.keys()) != required_keys:
        raise Amazon4UError(f"Canonical H authorization keys must match exact allowlist; got: {set(auth.keys())}")

    schema_ver = auth.get("schema_version")
    if schema_ver != SCHEMA_H_AUTHORIZATION:
        raise Amazon4UError(
            f"Canonical H authorization requires '{SCHEMA_H_AUTHORIZATION}'; "
            f"cannot accept O transport authority or unsupported schema: {schema_ver!r}"
        )

    for key in ("schema_version", "run_id", "approval_id", "scope"):
        value = auth.get(key)
        if type(value) is not str or not value or len(value) > 256:
            raise Amazon4UError(f"Canonical H authorization {key} must be bounded nonempty plain string")

    if auth.get("scope") != "tiny_synthetic_cpu_backward_and_private_restore":
        raise Amazon4UError(f"Authorization scope must be 'tiny_synthetic_cpu_backward_and_private_restore'; got {auth.get('scope')!r}")

    max_steps = auth.get("max_adam_steps")
    if type(max_steps) is not int or isinstance(max_steps, bool) or not 0 <= max_steps <= 4:
        raise Amazon4UError("Authorization max_adam_steps must be int in range 0..4 (no bool-as-int)")

    if actual_spec is not None:
        if auth.get("run_id") != actual_spec.get("run_id"):
            raise Amazon4UError("Authorization run_id does not match run spec")
        if auth.get("approval_id") != actual_spec.get("approval_id"):
            raise Amazon4UError("Authorization approval_id does not match run spec")
        spec_steps = actual_spec.get("max_adam_steps")
        if type(spec_steps) is int and not isinstance(spec_steps, bool) and spec_steps > max_steps:
            raise Amazon4UError("Run spec max_adam_steps exceeds authorized budget")

    if expected_auth is not None:
        if auth != expected_auth:
            raise Amazon4UError("Authorization differs from independently expected authorization")

    return copy.deepcopy(auth)


def get_canonical_h_authorization(auth: dict[str, object]) -> dict[str, object]:
    """Extract canonical H execution authorization matching scripts/training_entrypoint.py."""
    if not isinstance(auth, dict):
        raise Amazon4UError("Authorization must be a dictionary")
    schema_ver = auth.get("schema_version")
    if schema_ver == SCHEMA_H_AUTHORIZATION:
        return validate_canonical_h_authorization(auth)
    if "h_authorization" in auth and isinstance(auth["h_authorization"], dict):
        return get_canonical_h_authorization(auth["h_authorization"])
    raise Amazon4UError(
        f"Canonical H authorization requires '{SCHEMA_H_AUTHORIZATION}'; "
        f"cannot manufacture or relabel O transport authorization '{schema_ver}'"
    )


# ---------------------------------------------------------------------------
# Source Manifest & Outer Closure (OP03)
# ---------------------------------------------------------------------------

def build_source_manifest(
    packets: list[SourcePacket],
    *,
    require_full_closure: bool = False,
) -> dict[str, object]:
    """Build external O outer source manifest separating source hashes from manifest hash."""
    if len(packets) > MAX_CONCURRENT_FILES:
        raise Amazon4UError(f"Number of source packets ({len(packets)}) exceeds max files ({MAX_CONCURRENT_FILES})")

    files_manifest: dict[str, dict[str, object]] = {}
    aggregate_size = 0
    seen_paths: set[str] = set()

    for pkt in sorted(packets, key=lambda p: p.rel_path):
        validate_raw_relative_path(pkt.rel_path)
        if pkt.rel_path in seen_paths:
            raise Amazon4UError(f"Duplicate source packet path detected: {pkt.rel_path}")
        seen_paths.add(pkt.rel_path)

        aggregate_size += pkt.size
        if aggregate_size > MAX_AGGREGATE_BYTES:
            raise Amazon4UError(f"Aggregate source packet size exceeds limit: {aggregate_size} > {MAX_AGGREGATE_BYTES}")

        files_manifest[pkt.rel_path] = {
            "sha256": pkt.sha256,
            "size": pkt.size,
        }

    if require_full_closure:
        missing = set(REQUIRED_CLOSURE_PATHS) - seen_paths
        extra = seen_paths - set(REQUIRED_CLOSURE_PATHS)
        if missing or extra:
            raise Amazon4UError(f"Manifest source closure incomplete. Missing: {missing}, Extra: {extra}")

    manifest_payload: dict[str, object] = {
        "schema_version": SCHEMA_MANIFEST,
        "file_count": len(files_manifest),
        "aggregate_size": aggregate_size,
        "files": files_manifest,
    }
    # Payload digest covers the files entries without self-referential manifest hash
    manifest_digest = hashlib.sha256(
        json.dumps(files_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_payload["manifest_digest"] = manifest_digest
    return manifest_payload


def verify_source_manifest(
    actual: dict[str, object],
    expected: dict[str, object],
) -> bool:
    """Verify that actual source manifest matches independently expected manifest."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise Amazon4UError("Manifests must be dictionaries")

    actual_files = actual.get("files")
    expected_files = expected.get("files")
    if not isinstance(actual_files, dict) or not isinstance(expected_files, dict):
        raise Amazon4UError("Manifest 'files' must be a dictionary")

    if set(actual_files.keys()) != set(expected_files.keys()):
        missing = set(expected_files.keys()) - set(actual_files.keys())
        extra = set(actual_files.keys()) - set(expected_files.keys())
        raise Amazon4UError(f"Manifest path mismatch. Missing: {missing}, Extra: {extra}")

    for path, exp_info in expected_files.items():
        act_info = actual_files[path]
        if act_info.get("sha256") != exp_info.get("sha256"):
            raise Amazon4UError(f"Hash mismatch for {path}: expected {exp_info.get('sha256')}, got {act_info.get('sha256')}")
        if act_info.get("size") != exp_info.get("size"):
            raise Amazon4UError(f"Size mismatch for {path}: expected {exp_info.get('size')}, got {act_info.get('size')}")

    if actual.get("manifest_digest") != expected.get("manifest_digest"):
        raise Amazon4UError("Manifest digest mismatch between actual and expected")

    # Outer closure: check generated_runtime and environment if expected
    if "generated_runtime" in expected:
        act_rt = actual.get("generated_runtime")
        exp_rt = expected.get("generated_runtime")
        if act_rt != exp_rt:
            raise Amazon4UError(f"Generated runtime mismatch: expected {exp_rt}, got {act_rt}")

    if "environment" in expected:
        act_env = actual.get("environment")
        exp_env = expected.get("environment")
        if act_env != exp_env:
            raise Amazon4UError(f"Environment expectations mismatch: expected {exp_env}, got {act_env}")

    return True


# ---------------------------------------------------------------------------
# Pure In-Memory Plan Rendering (OP02, OP06, OP18)
# ---------------------------------------------------------------------------

def encode_bytes(data: bytes) -> str:
    """Gzip compress (mtime=0) and base64 encode bytes."""
    import gzip
    gz = gzip.compress(data, mtime=0)
    return base64.b64encode(gz).decode("ascii")


def render_plan(
    spec: dict[str, object],
    *,
    authorization: dict[str, object],
    expected_spec: dict[str, object],
    expected_sources: dict[str, object],
    source_packets: list[SourcePacket],
    expected_auth: dict[str, object] | None = None,
    expected_runtime: dict[str, object] | None = None,
    expected_environment: dict[str, object] | None = None,
    context: ApprovedOfflineContext | None = None,
    owner: str = "amazon4u",
    slug: str | None = None,
    attempt_id: str = "attempt-01",
    timeout: int = 7200,
    accelerator: str = "none",
) -> RenderedPlan:
    """Pure, deterministic in-memory render of execution plan and kernel bundle.

    Requires independently supplied approved spec, authorization, expected sources,
    and source packets. No self-approval or synthesizing expectations.
    Tripwire protection: does NOT access Path.home(), os.environ credentials,
    CLI, subprocess, network, or filesystem.
    """
    if context is None:
        raise Amazon4UError("Mandatory independent ApprovedOfflineContext required for render_plan; self-derived fallbacks forbidden")
    valid_ctx = validate_approved_offline_context(context)
    if expected_spec is not None and expected_spec != valid_ctx.canonical_h_spec:
        raise Amazon4UError("expected_spec does not match context canonical_h_spec")
    expected_spec = valid_ctx.canonical_h_spec
    expected_auth = valid_ctx.h_authorization
    expected_sources = valid_ctx.expected_o_outer
    timeout = valid_ctx.submission_binding.get("timeout", timeout)
    accelerator = valid_ctx.expected_o_environment.get("accelerator", accelerator)

    if spec is None or expected_spec is None:
        raise Amazon4UError("spec and expected_spec must be independently supplied")
    if authorization is None:
        raise Amazon4UError("authorization must be independently supplied; cannot synthesize")
    if expected_sources is None:
        raise Amazon4UError("expected_sources is mandatory; cannot synthesize")
    if source_packets is None:
        raise Amazon4UError("source_packets is mandatory")

    valid_spec = validate_amazon4u_spec(spec, expected_spec=expected_spec)
    valid_auth = validate_amazon4u_authorization(
        authorization,
        actual_spec=valid_spec,
        expected_auth=expected_auth,
    )

    if type(timeout) is not int or isinstance(timeout, bool) or timeout <= 0:
        raise Amazon4UError("timeout must be a positive integer")
    accel_norm = str(accelerator).lower() if accelerator else "none"
    if accel_norm not in ("none", ""):
        raise Amazon4UError(f"accelerator must be 'none' or '' in CPU reference scope, got: {accelerator}")

    manifest = build_source_manifest(source_packets, require_full_closure=True)

    if slug is None:
        raw_slug = f"amazon4u-{str(valid_spec['method']).lower().replace(' ', '-').replace('+', '-')}-{valid_spec['run_id']}"
        slug = re.sub(r"[^a-z0-9]+", "-", raw_slug).strip("-")[:48]

    # Deterministic metadata: private=true, internet=false, gpu=false, explicit timeout
    metadata: dict[str, object] = {
        "id": f"{owner}/{slug}",
        "title": f"Amazon4U {valid_spec['method']} {valid_spec['run_id']}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "",
        "accelerator": "none",
        "timeout": timeout,
    }

    # Files dictionary: relative path -> base64 payload
    files_payload: dict[str, str] = {}
    for pkt in sorted(source_packets, key=lambda p: p.rel_path):
        files_payload[pkt.rel_path] = encode_bytes(pkt.content)

    # Autonomous generated kernel runner script
    kernel_script = build_amazon4u_kernel_script(valid_spec, valid_auth, str(manifest["manifest_digest"]))

    # Outer manifest incorporating full closure, generated runtime, and environment roles
    outer_manifest = copy.deepcopy(manifest)
    outer_manifest["generated_runtime"] = {
        "code_file": "kernel.py",
        "sha256": hashlib.sha256(kernel_script.encode("utf-8")).hexdigest(),
        "size": len(kernel_script.encode("utf-8")),
    }
    outer_manifest["environment"] = {
        "device": "cpu",
        "accelerator": "none",
        "network": "disabled",
    }

    # Construct THEN compare: verify outer_manifest against expected_sources and context.expected_o_outer
    verify_source_manifest(outer_manifest, expected_sources)
    verify_source_manifest(outer_manifest, valid_ctx.expected_o_outer)

    # Verify generated_runtime and environment against independent expectations unconditionally
    exp_rt = expected_runtime or valid_ctx.expected_o_outer.get("generated_runtime")
    if exp_rt is None:
        raise Amazon4UError("expected_runtime is mandatory; cannot synthesize")
    if outer_manifest["generated_runtime"] != exp_rt:
        raise Amazon4UError(f"Generated runtime does not match expected_runtime: {exp_rt}")

    exp_env = expected_environment or valid_ctx.expected_o_outer.get("environment") or valid_ctx.expected_o_environment
    if exp_env is None:
        raise Amazon4UError("expected_environment is mandatory; cannot synthesize")
    for k in ("device", "accelerator", "network"):
        if k in exp_env and outer_manifest["environment"].get(k) != exp_env.get(k):
            raise Amazon4UError(f"Environment field '{k}' mismatch: expected {exp_env.get(k)}, got {outer_manifest['environment'].get(k)}")

    # Compute deterministic plan digest covering spec, auth, manifest_digest, metadata, attempt, timeout
    plan_identity = {
        "spec": valid_spec,
        "authorization": valid_auth,
        "manifest_digest": manifest["manifest_digest"],
        "metadata": metadata,
        "attempt_id": attempt_id,
        "timeout": timeout,
    }
    plan_digest = hashlib.sha256(json.dumps(plan_identity, sort_keys=True).encode("utf-8")).hexdigest()

    return RenderedPlan(
        spec=valid_spec,
        authorization=valid_auth,
        metadata=metadata,
        files=files_payload,
        kernel_script=kernel_script,
        outer_manifest=outer_manifest,
        plan_digest=plan_digest,
        owner=owner,
        slug=slug,
        attempt_id=attempt_id,
        rendered_at="2026-10-19T00:00:00Z",  # fixed deterministic timestamp
    )


_GENERATED_PREFLIGHT = r'''
import copy
import hashlib
import struct
from dataclasses import make_dataclass

def _h_files_digest(files):
    h = hashlib.sha256()
    def blob(tag, data):
        h.update(tag)
        h.update(struct.pack("!Q", len(data)))
        h.update(data)
    def visit(v):
        if type(v) is dict:
            h.update(b"D")
            h.update(struct.pack("!Q", len(v)))
            for key in sorted(v):
                visit(key)
                visit(v[key])
        elif type(v) is int:
            blob(b"I", str(v).encode("ascii"))
        elif type(v) is str:
            blob(b"S", v.encode("utf-8"))
        else:
            raise ValueError("Invalid H manifest digest value")
    visit(files)
    return h.hexdigest()

def _retain_generated_context(context):
    def need(condition, reason):
        if not condition:
            raise ValueError(reason)
    fields = (
        "canonical_h_spec", "h_authorization", "expected_h_envelope",
        "expected_h_bundle", "expected_o_outer", "expected_input_identity",
        "expected_o_environment", "submission_binding", "expected_output_inventory",
        "limits",
    )
    need(context is not None, "Mandatory independent context required")
    raw = context if isinstance(context, dict) else {k: getattr(context, k, None) for k in fields}
    need(all(k in raw and raw[k] is not None for k in fields), "Missing mandatory context fields")
    c = copy.deepcopy({k: raw[k] for k in fields})
    s = c["canonical_h_spec"]
    need(type(s) is dict and set(s) == set(SPEC_KEYS), "Invalid canonical H spec keys")
    need(all(type(s[k]) is str and 0 < len(s[k]) <= 256 for k in set(SPEC_KEYS) - {"epochs", "max_adam_steps"}), "Invalid H spec strings")
    need(s["schema_version"] == "amazon4u-H-spec-v1" and s["stage"] == "training_from_scratch", "Invalid H schema/stage")
    need(s["mode"] == "fixture_verification" and s["data_kind"] == "owned_synthetic_packet", "Invalid H mode/data kind")
    need(s["method"] in ("LightGCN", "Heterogeneous KG+GNN", "CorePopularity", "KG-only"), "Invalid H method")
    need(s["device"] == "cpu" and s["accelerator"] == "none", "Invalid H device/accelerator")
    need(s["profile_id"] == "H-REF-synthetic-cpu-v1", "Invalid H profile")
    need(type(s["epochs"]) is int and type(s["max_adam_steps"]) is int and 0 <= s["epochs"] <= 4, "Invalid H integer budgets")
    baseline = s["method"] in ("CorePopularity", "KG-only")
    need((s["epochs"] == 0 and s["max_adam_steps"] == 0) if baseline else (s["epochs"] >= 1 and s["max_adam_steps"] >= 1), "Invalid method budgets")
    a = c["h_authorization"]
    need(type(a) is dict and set(a) == {"schema_version", "run_id", "approval_id", "scope", "max_adam_steps"}, "Invalid authorization keys")
    need(all(type(a[k]) is str and 0 < len(a[k]) <= 256 for k in ("schema_version", "run_id", "approval_id", "scope")), "Invalid authorization strings")
    need(a["schema_version"] == "amazon4u-H-authorization-v1", "Invalid authorization schema")
    need(a["scope"] == "tiny_synthetic_cpu_backward_and_private_restore", "Invalid authorization scope")
    need(type(a["max_adam_steps"]) is int and 0 <= a["max_adam_steps"] <= 4, "Invalid authorization budget")
    need(a["run_id"] == s["run_id"] and a["approval_id"] == s["approval_id"], "Authorization identity mismatch")
    need(s["max_adam_steps"] <= a["max_adam_steps"], "Unauthorized H budget")
    b = c["expected_h_bundle"]
    need(type(b) is dict and set(b) == {"schema_version", "files", "sha256"} and b["schema_version"] == "amazon4u-H-bundle-v1", "Invalid canonical H manifest")
    f = b["files"]
    need(type(f) is dict and set(f) == set(H_BUNDLE_PATHS), "Expected H manifest must contain exactly H28")
    def sha(v):
        return type(v) is str and len(v) == 64 and all(ch in "0123456789abcdef" for ch in v)
    for meta in f.values():
        need(type(meta) is dict and set(meta) == {"bytes", "sha256"}, "Invalid H file metadata keys")
        need(type(meta["bytes"]) is int and 0 <= meta["bytes"] <= MAX_FILE_BYTES and sha(meta["sha256"]), "Invalid H file length/SHA")
    need(sum(m["bytes"] for m in f.values()) <= MAX_AGGREGATE_BYTES, "H aggregate source bound exceeded")
    need(sha(b["sha256"]) and b["sha256"] == _h_files_digest(f), "Invalid H semantic manifest SHA")
    e = c["expected_h_envelope"]
    env_keys = {"schema_version", "run_id", "approval_id", "profile_id", "method", "identity_hash", "device", "accelerator", "bundle_sha256", "initial_state_sha256", "determinism_policy"}
    need(type(e) is dict and set(e) == env_keys, "Invalid expected H envelope keys")
    need(all(type(e[k]) is str and 0 < len(e[k]) <= 256 for k in env_keys), "Invalid expected H envelope strings")
    need(e["schema_version"] == "amazon4u-H-run-v1", "Invalid expected H envelope schema")
    need(e["profile_id"] == "H-REF-synthetic-cpu-v1", "Invalid expected H envelope profile")
    need(e["determinism_policy"] == "private_rng_bounded_cpu_no_global_setters", "Invalid expected H envelope determinism policy")
    need(sha(e["identity_hash"]) and sha(e["bundle_sha256"]) and sha(e["initial_state_sha256"]), "Invalid expected H envelope digests")
    need(e["run_id"] == s["run_id"] and e["approval_id"] == s["approval_id"], "Expected H envelope identity mismatch")
    need(e["method"] == s["method"] and e["device"] == s["device"] and e["accelerator"] == s["accelerator"] and e["profile_id"] == s["profile_id"], "Expected H envelope spec mismatch")
    need(e["bundle_sha256"] == b["sha256"], "Expected H envelope bundle SHA mismatch")
    o = c["expected_o_outer"]
    need(isinstance(o, dict) and isinstance(o.get("files"), dict), "Invalid O outer manifest")
    need(set(o["files"]) == set(REQUIRED_CLOSURE_PATHS), "Expected exactly the 32 O closure paths")
    for meta in o["files"].values():
        need(type(meta) is dict and set(meta) == {"size", "sha256"}, "Invalid O file metadata keys")
        need(type(meta["size"]) is int and 0 <= meta["size"] <= MAX_FILE_BYTES and sha(meta["sha256"]), "Invalid O file length/SHA")
    total = sum(meta["size"] for meta in o["files"].values())
    need(total <= MAX_AGGREGATE_BYTES, "O aggregate source bound exceeded")
    need(o.get("schema_version") == "amazon4u-O-manifest-v1", "Invalid O manifest schema")
    need(type(o.get("file_count")) is int and o["file_count"] == len(o["files"]), "Invalid O file count")
    need(type(o.get("aggregate_size")) is int and o["aggregate_size"] == total, "Invalid O aggregate size")
    need(o.get("manifest_digest") == hashlib.sha256(json.dumps(o["files"], sort_keys=True).encode("utf-8")).hexdigest(), "Invalid O manifest digest")
    if "generated_runtime" in o:
        r = o["generated_runtime"]
        need(isinstance(r, dict) and r.get("code_file") == "kernel.py" and "sha256" in r, "Invalid generated runtime metadata")
    if "environment" in o:
        r = o["environment"]
        need(isinstance(r, dict) and r.get("device") == "cpu" and r.get("accelerator") in ("none", ""), "Invalid O outer environment")
    i = c["expected_input_identity"]
    need(isinstance(i, dict) and i.get("spec") == s and i.get("authorization") == a, "Input identity mismatch")
    sb = c["submission_binding"]
    need(isinstance(sb, dict) and all(type(sb.get(k)) is int and sb[k] > 0 for k in ("version", "timeout")), "Invalid submission binding")
    env = c["expected_o_environment"]
    need(isinstance(env, dict) and env.get("device", "cpu") == "cpu" and not env.get("cuda_available", False) and env.get("accelerator", "none") in ("none", ""), "Invalid expected O environment")
    caps = env.get("capabilities")
    need(isinstance(caps, dict) and {"package", "version", "nativeIO", "ABI", "dependency", "CPU", "runtime", "resource", "callable", "bootstrap"}.issubset(caps), "Missing expected capabilities")
    inv = c["expected_output_inventory"]
    need(isinstance(inv, dict) and {"plan", "h_result", "run_envelope", "source_manifest", "log", "checkpoint"}.issubset(inv), "Missing output inventory roles")
    limits = c["limits"]
    need(isinstance(limits, dict) and all(type(limits.get(k)) is int and limits[k] > 0 for k in ("max_file_bytes", "max_aggregate_bytes")), "Invalid context limits")
    cls = make_dataclass("ApprovedOfflineContext", [(k, dict) for k in fields], frozen=True)
    return cls(**c)

def _retain_generated_sources(sources, bundle):
    if type(sources) is not dict or not sources or set(sources) != set(bundle["files"]):
        raise ValueError("H sources require the nonempty exact H28 map")
    retained = dict(sources)
    for path, content in retained.items():
        meta = bundle["files"][path]
        if type(content) is not bytes or len(content) != meta["bytes"] or hashlib.sha256(content).hexdigest() != meta["sha256"]:
            raise ValueError("H source physical bytes differ from retained manifest")
    return retained
'''


def build_amazon4u_kernel_script(
    spec: dict[str, object],
    auth: dict[str, object],
    manifest_digest: str,
) -> str:
    """Generate self-contained standalone Python script with bounded streaming hydration."""
    spec_json = json.dumps(spec)
    auth_json = json.dumps(auth)

    return f'''#!/usr/bin/env python3
\"\"\"Amazon4U Autonomous Kernel Runner — Bounded Offline Hydration & Staged Execution.\"\"\"
import base64
import json
import os
from pathlib import Path
import sys
import zlib

SPEC = json.loads({spec_json!r})
AUTH = json.loads({auth_json!r})
MANIFEST_DIGEST = {manifest_digest!r}

MAX_FILE_BYTES = {MAX_FILE_BYTES}
MAX_AGGREGATE_BYTES = {MAX_AGGREGATE_BYTES}
MAX_CHUNK_BYTES = {MAX_CHUNK_BYTES}
MAX_ENCODED_BYTES = {MAX_ENCODED_BYTES}

SPEC_KEYS = {tuple(sorted(SPEC_KEYS))!r}
H_BUNDLE_PATHS = {H_BUNDLE_PATHS!r}
REQUIRED_CLOSURE_PATHS = {REQUIRED_CLOSURE_PATHS!r}
{_GENERATED_PREFLIGHT}

def streaming_hydrate(
    rel_path: str,
    b64_payload: str,
    target_dir: Path,
    current_total: int,
    expected_size: int,
    expected_sha256: str,
) -> int:
    if not isinstance(rel_path, str) or not rel_path or "\\\\" in rel_path or ":" in rel_path or rel_path.startswith("/"):
        raise ValueError(f"Path format invalid, absolute, or backslash: {{rel_path}}")
    raw_parts = rel_path.split("/")
    if any(part in ("..", ".", "") for part in raw_parts):
        raise ValueError(f"Path traversal or empty parts escape detected: {{rel_path}}")
    if len(b64_payload) > MAX_ENCODED_BYTES:
        raise ValueError(f"Payload for {{rel_path}} exceeds maximum encoded bytes {{MAX_ENCODED_BYTES}}")

    dst_path = target_dir / rel_path
    if dst_path.exists():
        raise ValueError(f"Destination file already exists; overwrite forbidden: {{dst_path}}")
    if any(parent.is_symlink() for parent in dst_path.parents):
        raise ValueError(f"Ancestor symlink forbidden: {{dst_path}}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        compressed = base64.b64decode(b64_payload, validate=True)
    except Exception as exc:
        raise ValueError(f"Malformed base64 payload for {{rel_path}}: {{exc}}")
    if len(compressed) > MAX_FILE_BYTES:
        raise ValueError(f"Compressed payload for {{rel_path}} exceeds file limit {{MAX_FILE_BYTES}}")

    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
    chunk_size = MAX_CHUNK_BYTES
    file_bytes = 0
    import hashlib
    hasher = hashlib.sha256()

    temp_path = dst_path.with_name(f".tmp_{{dst_path.name}}_{{os.getpid()}}")
    created_temp = False
    try:
        temp_fd = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created_temp = True
        with os.fdopen(temp_fd, "wb") as out_f:
            offset = 0
            while offset < len(compressed) or dobj.unconsumed_tail:
                if dobj.unconsumed_tail:
                    chunk = dobj.unconsumed_tail
                else:
                    chunk = compressed[offset:offset + chunk_size]
                    offset += len(chunk)
                rem_f = MAX_FILE_BYTES - file_bytes
                rem_a = MAX_AGGREGATE_BYTES - (current_total + file_bytes)
                probe = max(0, min(rem_f, rem_a, chunk_size)) + 1
                data = dobj.decompress(chunk, probe)
                if data:
                    file_bytes += len(data)
                    if file_bytes > MAX_FILE_BYTES:
                        raise ValueError(f"File {{rel_path}} exceeded maximum bytes {{MAX_FILE_BYTES}}")
                    if current_total + file_bytes > MAX_AGGREGATE_BYTES:
                        raise ValueError(f"Aggregate limit {{MAX_AGGREGATE_BYTES}} exceeded during hydration")
                    hasher.update(data)
                    out_f.write(data)

            while not dobj.eof:
                rem_f = MAX_FILE_BYTES - file_bytes
                rem_a = MAX_AGGREGATE_BYTES - (current_total + file_bytes)
                probe = max(0, min(rem_f, rem_a, chunk_size)) + 1
                data = dobj.decompress(b"", probe)
                if not data:
                    break
                file_bytes += len(data)
                if file_bytes > MAX_FILE_BYTES:
                    raise ValueError(f"File {{rel_path}} exceeded maximum bytes {{MAX_FILE_BYTES}}")
                if current_total + file_bytes > MAX_AGGREGATE_BYTES:
                    raise ValueError(f"Aggregate limit {{MAX_AGGREGATE_BYTES}} exceeded during hydration")
                hasher.update(data)
                out_f.write(data)

        if not dobj.eof:
            raise ValueError(f"Truncated compressed stream for {{rel_path}}")
        if dobj.unused_data:
            raise ValueError(f"Trailing data after stream EOF for {{rel_path}}")
        if file_bytes != expected_size:
            raise ValueError(f"Size mismatch for {{rel_path}}: expected {{expected_size}}, got {{file_bytes}}")
        if hasher.hexdigest() != expected_sha256:
            raise ValueError(f"Checksum mismatch for {{rel_path}}: expected {{expected_sha256}}, got {{hasher.hexdigest()}}")

        if dst_path.exists():
            raise ValueError(f"Destination file appeared concurrently: {{dst_path}}")
        try:
            os.link(str(temp_path), str(dst_path))
            os.unlink(str(temp_path))
            created_temp = False
        except FileExistsError:
            raise ValueError(f"Destination file appeared concurrently: {{dst_path}}")
    finally:
        if created_temp and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return file_bytes

def main(custom_workdir: Path | None = None, custom_outdir: Path | None = None, callable_fn = None, fake_bootstrap = None, checkpoint_verifier = None, context = None, capability_provider = None) -> int:
    try:
        context = _retain_generated_context(context)
    except Exception as exc:
        print("Refusal: ApprovedOfflineContext: " + str(exc), file=sys.stderr)
        return 1
    if not callable(callable_fn) or type(fake_bootstrap) is not dict:
        print("Refusal: Authorized callable and bootstrap required", file=sys.stderr)
        return 1
    required_bootstrap_keys = {{"expected_spec", "authorization", "data", "expectations", "run_envelope", "expected_run_envelope", "bundle_sources", "expected_bundle"}}
    if not required_bootstrap_keys.issubset(fake_bootstrap):
        print("Refusal: Missing mandatory bootstrap fields", file=sys.stderr)
        return 1
    fake_bootstrap = dict(fake_bootstrap)
    # Compare the presented metadata, then retain it BEFORE provider callbacks.
    # Opaque data/expectations are not deep-copied or initialized by this adapter.
    try:
        metadata = (
            ("expected_spec", context.canonical_h_spec),
            ("authorization", context.h_authorization),
            ("run_envelope", context.expected_h_envelope),
            ("expected_run_envelope", context.expected_h_envelope),
            ("expected_bundle", context.expected_h_bundle),
        )
        for key, expected in metadata:
            actual = fake_bootstrap[key]
            if type(actual) is not dict or actual != expected:
                raise ValueError("Canonical bootstrap metadata mismatch: " + key)
            if key != "expected_bundle" and any(type(actual[k]) is not type(v) for k, v in expected.items()):
                raise ValueError("Canonical bootstrap metadata type mismatch: " + key)
            fake_bootstrap[key] = copy.deepcopy(actual)
    except Exception as exc:
        print("Refusal: Preflight canonical bootstrap metadata: " + str(exc), file=sys.stderr)
        return 1
    try:
        fake_bootstrap["bundle_sources"] = _retain_generated_sources(fake_bootstrap["bundle_sources"], context.expected_h_bundle)
    except Exception as exc:
        print("Refusal: H bundle source coherence invalid: " + str(exc), file=sys.stderr)
        return 1

    env_work = os.environ.get("AMAZON4U_WORKDIR")
    env_out = os.environ.get("AMAZON4U_OUTDIR")
    if custom_workdir is None and not env_work:
        print("Refusal: No authorized workdir provided; default /kaggle paths forbidden before verification", file=sys.stderr)
        return 1
    if custom_outdir is None and not env_out:
        print("Refusal: No authorized outdir provided; default /kaggle paths forbidden before verification", file=sys.stderr)
        return 1
    if callable_fn is None and fake_bootstrap is None:
        print("Refusal: Absence of authorized runtime bootstrap/callable fails closed", file=sys.stderr)
        return 1

    workdir = custom_workdir or Path(env_work)
    outdir = custom_outdir or Path(env_out)
    if outdir.exists():
        print("Refusal: Output directory already exists; adoption of pre-existing directory or overwrite forbidden", file=sys.stderr)
        return 1

    # Stage 1: Preflight validation BEFORE creating directories, logs, hydration, or target module imports
    if context is not None:
        ctx_spec = getattr(context, "canonical_h_spec", None) or (context.get("canonical_h_spec") if isinstance(context, dict) else None)
        ctx_auth = getattr(context, "h_authorization", None) or (context.get("h_authorization") if isinstance(context, dict) else None)
        if ctx_spec != SPEC:
            print("Refusal: Preflight context spec mismatch", file=sys.stderr)
            return 1
        if ctx_auth != AUTH:
            print("Refusal: Preflight context authorization mismatch", file=sys.stderr)
            return 1

    expected_caps = context.expected_o_environment["capabilities"]

    actual_caps = None
    if capability_provider is not None:
        try:
            actual_caps = capability_provider()
        except Exception:
            print("Refusal: Capability provider raised error", file=sys.stderr)
            return 1
    elif fake_bootstrap is not None:
        actual_caps = fake_bootstrap.get("actual_capabilities")

    required_caps = {{
        "package", "version", "nativeIO", "ABI", "dependency",
        "CPU", "runtime", "resource", "callable", "bootstrap",
    }}
    if actual_caps is None or expected_caps is None or not isinstance(actual_caps, dict) or not isinstance(expected_caps, dict):
        print("Refusal: Preflight capabilities missing or invalid", file=sys.stderr)
        return 1
    if not required_caps.issubset(set(expected_caps.keys())):
        print("Refusal: Preflight expected capabilities missing required keys", file=sys.stderr)
        return 1
    for k, exp_val in expected_caps.items():
        if actual_caps.get(k) != exp_val:
            print(f"Refusal: Capability mismatch for {{k}}: expected {{exp_val}}, got {{actual_caps.get(k)}}", file=sys.stderr)
            return 1

    if fake_bootstrap is not None:
        if fake_bootstrap.get("expected_spec") != SPEC:
            print("Refusal: Preflight spec mismatch", file=sys.stderr)
            return 1
        if "authorization" in fake_bootstrap and fake_bootstrap["authorization"] != AUTH:
            print("Refusal: Preflight authorization mismatch", file=sys.stderr)
            return 1
        if context is not None:
            ctx_h_bundle = getattr(context, "expected_h_bundle", None) or (context.get("expected_h_bundle") if isinstance(context, dict) else None)
            if ctx_h_bundle is not None and "expected_bundle" in fake_bootstrap:
                if fake_bootstrap["expected_bundle"] != ctx_h_bundle:
                    print("Refusal: Preflight expected_bundle mismatch with canonical H context", file=sys.stderr)
                    return 1
        expectations = fake_bootstrap.get("expectations")
        if not isinstance(expectations, dict) or "resource_policy" not in expectations:
            print("Refusal: Preflight expectations invalid", file=sys.stderr)
            return 1
        expected_runtime = expectations.get("expected_runtime", {{}})
        if not isinstance(expected_runtime, dict):
            print("Refusal: Preflight expected_runtime invalid", file=sys.stderr)
            return 1
        if expected_runtime.get("cuda_available", False) or expected_runtime.get("device", "cpu") != "cpu" or expected_runtime.get("accelerator", "none") not in ("none", ""):
            print("Refusal: Preflight CPU-only scope violated", file=sys.stderr)
            return 1
        run_env = fake_bootstrap.get("run_envelope")
        exp_run_env = fake_bootstrap.get("expected_run_envelope")
        if not isinstance(run_env, dict) or not isinstance(exp_run_env, dict):
            print("Refusal: Preflight run_envelope and expected_run_envelope must be dicts", file=sys.stderr)
            return 1
        if exp_run_env != context.expected_h_envelope:
            print("Refusal: Preflight expected_run_envelope mismatch with canonical context", file=sys.stderr)
            return 1
        env_keys = {{"schema_version", "run_id", "approval_id", "profile_id", "method", "identity_hash", "device", "accelerator", "bundle_sha256", "initial_state_sha256", "determinism_policy"}}
        if set(run_env.keys()) != env_keys:
            print("Refusal: Preflight run_envelope invalid keys", file=sys.stderr)
            return 1
        if run_env.get("schema_version") != "amazon4u-H-run-v1" or run_env.get("profile_id") != "H-REF-synthetic-cpu-v1" or run_env.get("determinism_policy") != "private_rng_bounded_cpu_no_global_setters":
            print("Refusal: Preflight run_envelope schema or profile invalid", file=sys.stderr)
            return 1
        if any(not isinstance(run_env.get(k), str) or not run_env.get(k) or len(run_env.get(k)) > 256 for k in env_keys):
            print("Refusal: Preflight run_envelope values must be bounded nonempty strings", file=sys.stderr)
            return 1
        import re
        if any(not re.fullmatch(r"[0-9a-f]{{64}}", str(run_env.get(k))) for k in ("identity_hash", "bundle_sha256", "initial_state_sha256")):
            print("Refusal: Preflight run_envelope digests invalid", file=sys.stderr)
            return 1
        if run_env != context.expected_h_envelope:
            print("Refusal: Preflight run_envelope mismatch with canonical context", file=sys.stderr)
            return 1

    # ALL PREFLIGHT VALIDATIONS COMPLETE BEFORE DIRECTORY CREATION OR HYDRATION
    bundle_files_json = workdir / "bundle_files.json"
    manifest_file = workdir / "manifest.json"
    if not bundle_files_json.is_file() or not manifest_file.is_file():
        print("Hydration refusal: Required bundle files/manifest missing", file=sys.stderr)
        return 2

    print("=== Amazon4U Bounded Offline Runner ===")
    print("Method: " + str(SPEC.get("method")))
    print("Run ID: " + str(SPEC.get("run_id")))
    sys.stdout.flush()

    # Hydrate files if not already verified
    if bundle_files_json.exists() and manifest_file.exists():
        already_hydrated = False
        try:
            man = json.loads(manifest_file.read_text(encoding="utf-8"))
            if man != context.expected_o_outer:
                raise ValueError("Manifest differs from independently retained O expectation")
            if man.get("manifest_digest") != MANIFEST_DIGEST:
                raise ValueError("Manifest digest mismatch in kernel runner")
            man_files = man.get("files", {{}})
            if not man_files:
                raise ValueError("Manifest files empty")
            import hashlib
            for r, exp in man_files.items():
                p = workdir / r
                if not p.is_file() or p.is_symlink() or any(parent.is_symlink() for parent in p.parents):
                    raise ValueError(f"File invalid or symlink: {{r}}")
                if p.stat().st_size != exp.get("size"):
                    raise ValueError(f"Size mismatch: {{r}}")
                if hashlib.sha256(p.read_bytes()).hexdigest() != exp.get("sha256"):
                    raise ValueError(f"SHA mismatch: {{r}}")
            if (workdir / "scripts" / "__init__.py").exists():
                raise ValueError("scripts/__init__.py must not exist")
            already_hydrated = True
        except Exception:
            already_hydrated = False

        if not already_hydrated:
            try:
                files_map = json.loads(bundle_files_json.read_text(encoding="utf-8"))
                man = json.loads(manifest_file.read_text(encoding="utf-8"))
                if man != context.expected_o_outer:
                    raise ValueError("Manifest differs from independently retained O expectation")
                man_files = man.get("files", {{}})
                if not isinstance(files_map, dict) or set(files_map) != set(man_files):
                    raise ValueError("Bundle files differ from retained O path inventory")
                workdir.mkdir(parents=True, exist_ok=True)
                total = 0
                for rel, b64 in sorted(files_map.items()):
                    exp = man_files.get(rel, {{}})
                    written = streaming_hydrate(rel, b64, workdir, total, exp.get("size", 0), exp.get("sha256", ""))
                    total += written
                # Post-hydration verification before importing
                import hashlib
                for r, exp in man_files.items():
                    p = workdir / r
                    if not p.is_file() or p.is_symlink() or any(parent.is_symlink() for parent in p.parents):
                        raise ValueError(f"Hydrated file invalid or symlink: {{r}}")
                    if p.stat().st_size != exp.get("size"):
                        raise ValueError(f"Size mismatch after hydration: {{r}}")
                    if hashlib.sha256(p.read_bytes()).hexdigest() != exp.get("sha256"):
                        raise ValueError(f"SHA mismatch after hydration: {{r}}")
                if (workdir / "scripts" / "__init__.py").exists():
                    raise ValueError("scripts/__init__.py must not exist")
            except Exception as exc:
                print(f"Hydration error in kernel runner: {{exc}}", file=sys.stderr)
                return 2

    # Kernel execution dispatches via multiautoresearch.adapters.amazon4u_kernel
    # Isolate relevant cached target package/modules before loading from fresh verified root
    orig_sys_path = list(sys.path)
    isolated_keys = [k for k in list(sys.modules) if k.startswith("multiautoresearch")]
    saved_modules = {{k: sys.modules.pop(k) for k in isolated_keys}}
    try:
        sys.path.insert(0, str(workdir))
        from multiautoresearch.adapters.amazon4u_kernel import run_kernel_stages
        rc = run_kernel_stages(
            workdir,
            outdir,
            SPEC,
            AUTH,
            h_callable=callable_fn,
            fake_bootstrap=fake_bootstrap,
            checkpoint_verifier=checkpoint_verifier,
            already_hydrated=True,
            context=context,
            capability_provider=capability_provider,
        )
        return rc
    except Exception as exc:
        print("Kernel execution error: " + str(exc), file=sys.stderr)
        return 1
    finally:
        sys.path[:] = orig_sys_path
        for k in list(sys.modules):
            if k.startswith("multiautoresearch") and k not in saved_modules:
                sys.modules.pop(k, None)
        for k, v in saved_modules.items():
            sys.modules[k] = v

if __name__ == "__main__":
    sys.exit(main())
'''


def write_bundle(plan: RenderedPlan, target_dir: Path) -> Path:
    """Write rendered bundle into a fresh, owned, initially non-existent directory.

    Refuses adoption of pre-existing directories (empty or populated) and overwrites.
    """
    if target_dir.exists():
        raise Amazon4UError(
            f"Target bundle directory '{target_dir}' already exists; adoption of pre-existing directory or overwrite refused."
        )
    target_dir.mkdir(parents=True, exist_ok=False)

    # Write kernel script
    (target_dir / "kernel.py").write_text(plan.kernel_script, encoding="utf-8")

    # Write kernel metadata
    (target_dir / "kernel-metadata.json").write_text(
        json.dumps(plan.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Write plan and manifest
    plan_info = {
        "schema_version": SCHEMA_PLAN,
        "spec": plan.spec,
        "authorization": plan.authorization,
        "plan_digest": plan.plan_digest,
        "attempt_id": plan.attempt_id,
        "rendered_at": plan.rendered_at,
    }
    (target_dir / "plan.json").write_text(
        json.dumps(plan_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / "manifest.json").write_text(
        json.dumps(plan.outer_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Write files payload in bundle directory
    bundle_files_dir = target_dir / "bundle_files.json"
    bundle_files_dir.write_text(
        json.dumps(plan.files, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return target_dir


# ---------------------------------------------------------------------------
# Incremental Bounded Streaming Hydration (OP04, I2)
# ---------------------------------------------------------------------------

def hydrate_packet(
    rel_path: str,
    b64_payload: str,
    target_dir: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_aggregate_bytes: int = MAX_AGGREGATE_BYTES,
    current_aggregate: int = 0,
) -> tuple[int, str]:
    """Hydrate a single compressed file incrementally with strict byte ceilings.

    Bounds allocation incrementally using decompress max_length probe.
    Validates expected checksum, size, base64 validity, EOF, and trailing data.
    Refuses overwriting existing files or traversing ancestor symlinks.
    """
    validate_raw_relative_path(rel_path)

    if expected_sha256 is None or expected_size is None:
        raise Amazon4UError(f"expected_sha256 and expected_size are mandatory for {rel_path}")

    if len(b64_payload) > MAX_ENCODED_BYTES:
        raise Amazon4UError(f"Payload for {rel_path} exceeds maximum encoded bytes {MAX_ENCODED_BYTES}")

    dst_path = target_dir / rel_path
    if dst_path.exists():
        raise Amazon4UError(f"Destination file already exists; overwrite forbidden: {dst_path}")
    if any(parent.is_symlink() for parent in dst_path.parents):
        raise Amazon4UError(f"Ancestor symlink forbidden: {dst_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        compressed_bytes = base64.b64decode(b64_payload, validate=True)
    except Exception as exc:
        raise Amazon4UError(f"Malformed base64 payload for {rel_path}") from exc

    if len(compressed_bytes) > max_file_bytes:
        raise Amazon4UError(f"Compressed payload for {rel_path} exceeds file limit {max_file_bytes}")

    # Streaming decompression bounding allocation incrementally
    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
    chunk_size = MAX_CHUNK_BYTES
    hasher = hashlib.sha256()
    file_bytes = 0

    temp_path = dst_path.with_name(f".{dst_path.name}.tmp_{uuid.uuid4().hex[:8]}")
    created_temp = False
    try:
        temp_fd = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        created_temp = True
        with os.fdopen(temp_fd, "wb") as out_fh:
            offset = 0
            while offset < len(compressed_bytes) or dobj.unconsumed_tail:
                if dobj.unconsumed_tail:
                    in_chunk = dobj.unconsumed_tail
                else:
                    in_chunk = compressed_bytes[offset:offset + chunk_size]
                    offset += len(in_chunk)
                rem_f = max_file_bytes - file_bytes
                rem_a = max_aggregate_bytes - (current_aggregate + file_bytes)
                probe_len = max(0, min(rem_f, rem_a, chunk_size)) + 1
                out_chunk = dobj.decompress(in_chunk, probe_len)
                if out_chunk:
                    file_bytes += len(out_chunk)
                    if file_bytes > max_file_bytes:
                        raise Amazon4UError(
                            f"Hydration of {rel_path} aborted: exceeded file limit {max_file_bytes} bytes"
                        )
                    if current_aggregate + file_bytes > max_aggregate_bytes:
                        raise Amazon4UError(
                            f"Hydration of {rel_path} aborted: exceeded aggregate limit {max_aggregate_bytes} bytes"
                        )
                    hasher.update(out_chunk)
                    out_fh.write(out_chunk)

            while not dobj.eof:
                rem_f = max_file_bytes - file_bytes
                rem_a = max_aggregate_bytes - (current_aggregate + file_bytes)
                probe_len = max(0, min(rem_f, rem_a, chunk_size)) + 1
                out_chunk = dobj.decompress(b"", probe_len)
                if not out_chunk:
                    break
                file_bytes += len(out_chunk)
                if file_bytes > max_file_bytes:
                    raise Amazon4UError(f"Hydration of {rel_path} aborted: exceeded file limit {max_file_bytes} bytes")
                if current_aggregate + file_bytes > max_aggregate_bytes:
                    raise Amazon4UError(f"Hydration of {rel_path} aborted: exceeded aggregate limit {max_aggregate_bytes} bytes")
                hasher.update(out_chunk)
                out_fh.write(out_chunk)

        if not dobj.eof:
            raise Amazon4UError(f"Truncated compressed stream for {rel_path}; EOF not reached")
        if dobj.unused_data:
            raise Amazon4UError(f"Trailing garbage data after stream EOF for {rel_path}")

        actual_sha256 = hasher.hexdigest()

        if file_bytes != expected_size:
            raise Amazon4UError(
                f"Declared size lie for {rel_path}: declared {expected_size}, actual {file_bytes}"
            )
        if actual_sha256 != expected_sha256:
            raise Amazon4UError(
                f"Checksum mismatch for {rel_path}: expected {expected_sha256}, got {actual_sha256}"
            )

        if dst_path.exists():
            raise Amazon4UError(f"Destination file appeared concurrently; collision refused: {dst_path}")
        try:
            atomic_move_no_replace(temp_path, dst_path)
            created_temp = False
        except FileExistsError as exc:
            raise Amazon4UError(f"Destination file appeared concurrently; collision refused: {dst_path}") from exc
    finally:
        if created_temp and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return file_bytes, actual_sha256


def hydrate_bundle(
    files: dict[str, str],
    target_dir: Path,
    *,
    manifest: dict[str, object] | None = None,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_aggregate_bytes: int = MAX_AGGREGATE_BYTES,
    max_files: int = MAX_CONCURRENT_FILES,
) -> dict[str, dict[str, object]]:
    """Hydrate full bundle files into target directory with aggregate checks."""
    if len(files) > max_files:
        raise Amazon4UError(f"Bundle file count ({len(files)}) exceeds maximum allowed ({max_files})")

    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_files = (manifest or {}).get("files", {}) if manifest else {}
    if manifest and isinstance(manifest_files, dict) and manifest_files:
        if set(files.keys()) != set(manifest_files.keys()):
            raise Amazon4UError("Bundle files keys mismatch manifest files keys")

    results: dict[str, dict[str, object]] = {}
    aggregate_bytes = 0

    for rel_path in sorted(files.keys()):
        payload = files[rel_path]
        expected_info = manifest_files.get(rel_path, {})
        exp_sha = expected_info.get("sha256")
        exp_sz = expected_info.get("size")
        if exp_sha is None or exp_sz is None:
            # If not in manifest, compute expected from packet payload or manifest is required
            raise Amazon4UError(f"Missing expected sha256 or size in manifest for {rel_path}")

        written, sha = hydrate_packet(
            rel_path,
            payload,
            target_dir,
            max_file_bytes=max_file_bytes,
            max_aggregate_bytes=max_aggregate_bytes,
            current_aggregate=aggregate_bytes,
            expected_sha256=exp_sha,
            expected_size=exp_sz,
        )
        aggregate_bytes += written
        results[rel_path] = {"size": written, "sha256": sha}

    # Ensure scripts directory remains a namespace package WITHOUT __init__.py
    scripts_init = target_dir / "scripts" / "__init__.py"
    if scripts_init.exists():
        raise Amazon4UError("Protected scripts namespace must NOT contain __init__.py")

    return results


# ---------------------------------------------------------------------------
# Fake Kaggle Transport (OP07, OP08, OP09, OP10, OP18)
# ---------------------------------------------------------------------------

class FakeKaggleTransport:
    """Mock transport interface for testing offline without network or CLI execution."""

    def __init__(
        self,
        *,
        acknowledged_version: int = 1,
        statuses: list[str] | None = None,
        logs_text: str = "Training finished successfully",
        log_stream_chunks: list[str] | None = None,
        log_availability_error: bool = False,
        paged_outputs: list[list[dict[str, object]]] | None = None,
        push_returncode: int = 0,
        push_printed_error: str | None = None,
        push_ambiguous_timeout: bool = False,
    ) -> None:
        self.acknowledged_version = acknowledged_version
        self.statuses = list(statuses or ["queued", "running", "complete"])
        self.status_index = 0
        self.logs_text = logs_text
        self.log_stream_chunks = log_stream_chunks
        self.log_availability_error = log_availability_error
        self.paged_outputs = paged_outputs or []
        self.push_returncode = push_returncode
        self.push_printed_error = push_printed_error
        self.push_ambiguous_timeout = push_ambiguous_timeout
        self.push_calls: list[dict[str, object]] = []
        self.status_calls: list[str] = []
        self.log_calls: list[str] = []

    def push(
        self,
        bundle_dir: Path,
        *,
        timeout: int | None = None,
        accelerator: str | None = None,
        plan_digest: str | None = None,
    ) -> dict[str, object]:
        self.push_calls.append({
            "bundle_dir": str(bundle_dir),
            "timeout": timeout,
            "accelerator": accelerator,
            "plan_digest": plan_digest,
        })

        if self.push_ambiguous_timeout:
            raise SubmissionUnknownError("Ambiguous submission timeout after push sent")

        if self.push_returncode != 0:
            return {
                "returncode": self.push_returncode,
                "error": "Simulated push failure",
                "acknowledged_version": None,
            }

        if self.push_printed_error:
            return {
                "returncode": 0,
                "stdout": f"Error: {self.push_printed_error}",
                "acknowledged_version": None,
            }

        meta_path = bundle_dir / "kernel-metadata.json"
        if not meta_path.exists():
            return {"returncode": 1, "error": "kernel-metadata.json missing"}

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        owner, slug = meta["id"].split("/")

        return {
            "returncode": 0,
            "owner": owner,
            "slug": slug,
            "acknowledged_version": self.acknowledged_version,
            "version_ref": f"{owner}/{slug}/v{self.acknowledged_version}",
            "timeout": timeout or 7200,
            "accelerator": accelerator or "none",
            "plan_digest": plan_digest,
            "stdout": f"Kernel version {self.acknowledged_version} successfully pushed.",
        }

    def status(self, kernel_ref: str, version: int | None = None) -> dict[str, object]:
        self.status_calls.append(kernel_ref)
        if self.status_index < len(self.statuses):
            st = self.statuses[self.status_index]
            self.status_index += 1
        else:
            st = self.statuses[-1]

        v = self.acknowledged_version if self.acknowledged_version is not None else version
        return {
            "returncode": 0,
            "status": st,
            "version": v,
            "kernel_ref": kernel_ref,
        }

    def open_log_reader(
        self,
        kernel_ref: str,
        version: int | None = None,
        deadline: float | None = None,
        cancellation: Any = None,
    ) -> dict[str, object]:
        self.log_calls.append(kernel_ref)
        if self.log_availability_error:
            return {
                "returncode": 1,
                "error": "Log stream unavailable on remote host",
                "reader": None,
                "version": version or self.acknowledged_version,
                "kernel_ref": kernel_ref,
            }
        chunks: list[bytes] = []
        if self.log_stream_chunks is not None:
            chunks = [c.encode("utf-8") if isinstance(c, str) else c for c in self.log_stream_chunks]
        elif self.logs_text:
            chunks = [self.logs_text.encode("utf-8")]
        reader = BoundedLogReader(chunks)
        return {
            "returncode": 0,
            "reader": reader,
            "version": version or self.acknowledged_version,
            "kernel_ref": kernel_ref,
        }

    def logs(
        self,
        kernel_ref: str,
        version: int | None = None,
        max_bytes: int = 65536,
    ) -> dict[str, object]:
        self.log_calls.append(kernel_ref)
        if self.log_availability_error:
            return {
                "returncode": 1,
                "error": "Log stream unavailable on remote host",
                "logs": "",
                "version": version or self.acknowledged_version,
                "kernel_ref": kernel_ref,
            }

        if self.log_stream_chunks is not None:
            # Combine stream chunks
            text = "".join(self.log_stream_chunks)
        else:
            text = self.logs_text

        # Redact synthetic secrets/tokens BEFORE persistence
        redacted = re.sub(
            r"(token|key|password|credential|bearer)\s*[:=]\s*[A-Za-z0-9_-]+",
            r"\1=[REDACTED]",
            text,
            flags=re.I,
        )
        raw_bytes = redacted.encode("utf-8")
        if len(raw_bytes) > max_bytes:
            redacted = raw_bytes[:max_bytes].decode("utf-8", errors="ignore") + "\n...[TRUNCATED]"

        return {
            "returncode": 0,
            "logs": redacted,
            "version": version or self.acknowledged_version,
            "kernel_ref": kernel_ref,
        }

    def list_output_page(
        self,
        kernel_ref: str,
        page_token: str | None = None,
    ) -> tuple[list[dict[str, object]], str | None]:
        page_idx = int(page_token) if page_token and page_token.isdigit() else 0
        if page_idx < len(self.paged_outputs):
            items = self.paged_outputs[page_idx]
            next_token = str(page_idx + 1) if (page_idx + 1) < len(self.paged_outputs) else None
            return items, next_token
        return [], None


def validate_submission_receipt(
    receipt: dict[str, object],
    *,
    expected_owner: str,
    expected_slug: str,
    expected_version: int,
    expected_timeout: int | None = None,
    expected_accelerator: str | None = None,
    expected_plan_digest: str | None = None,
) -> bool:
    """Validate submission receipt returned by Kaggle transport consumer."""
    if not isinstance(receipt, dict):
        raise Amazon4UError("Submission receipt must be a dictionary")

    if receipt.get("returncode") != 0:
        raise Amazon4UError(f"Submission failed with returncode {receipt.get('returncode')}: {receipt.get('error')}")

    if receipt.get("owner") != expected_owner:
        raise Amazon4UError(f"Receipt owner mismatch: expected {expected_owner}, got {receipt.get('owner')}")

    if receipt.get("slug") != expected_slug:
        raise Amazon4UError(f"Receipt slug mismatch: expected {expected_slug}, got {receipt.get('slug')}")

    if receipt.get("acknowledged_version") != expected_version:
        raise Amazon4UError(f"Receipt version mismatch: expected {expected_version}, got {receipt.get('acknowledged_version')}")

    expected_ref = f"{expected_owner}/{expected_slug}/v{expected_version}"
    if receipt.get("version_ref") != expected_ref:
        raise Amazon4UError(f"Receipt version_ref mismatch: expected {expected_ref}, got {receipt.get('version_ref')}")

    if expected_timeout is not None and receipt.get("timeout") != expected_timeout:
        raise Amazon4UError(f"Receipt timeout mismatch: expected {expected_timeout}, got {receipt.get('timeout')}")

    if expected_accelerator is not None and receipt.get("accelerator") != expected_accelerator:
        raise Amazon4UError(f"Receipt accelerator mismatch: expected {expected_accelerator}, got {receipt.get('accelerator')}")

    if expected_plan_digest is not None and receipt.get("plan_digest") != expected_plan_digest:
        raise Amazon4UError("Receipt plan_digest mismatch")

    return True


# ---------------------------------------------------------------------------
# Status State Machine (OP08)
# ---------------------------------------------------------------------------

def poll_kernel_status(
    kernel_ref: str,
    expected_version: int,
    transport: FakeKaggleTransport,
    *,
    max_polls: int = MAX_FAKE_POLLS,
    deadline_seconds: float = 300.0,
    poll_interval: float = 0.0,
    clock_fn = None,
    sleep_fn = None,
) -> str:
    """Bounded status state machine with exact token checks and fake clock injection.

    Never uses substring checks for terminal status.
    Enforces maximum poll iterations and deadline budgets.
    """
    if type(max_polls) is not int or isinstance(max_polls, bool) or max_polls <= 0:
        raise Amazon4UError(f"max_polls must be a positive int, got: {max_polls}")
    if not (isinstance(deadline_seconds, (int, float)) and deadline_seconds > 0):
        raise Amazon4UError(f"deadline_seconds must be a positive number, got: {deadline_seconds}")

    now_fn = clock_fn or time.monotonic
    wait_fn = sleep_fn or time.sleep

    polls = 0
    start_time = now_fn()

    while True:
        polls += 1
        if polls > max_polls:
            return "poll_deadline_exceeded"
        if (now_fn() - start_time) > deadline_seconds:
            return "poll_deadline_exceeded"

        res = transport.status(kernel_ref, version=expected_version)
        # Check deadline immediately after provider call
        if (now_fn() - start_time) > deadline_seconds:
            return "poll_deadline_exceeded"

        if res.get("returncode", 1) != 0:
            return "error"

        # Check version and ref
        if res.get("version") != expected_version:
            return "error"
        if res.get("kernel_ref") != kernel_ref:
            return "error"

        raw_status = str(res.get("status", "")).strip().lower()

        # Reject substring matching
        if raw_status not in VALID_STATUS_TOKENS:
            return "error"

        if raw_status in TERMINAL_STATUS_TOKENS:
            return raw_status

        if poll_interval > 0:
            wait_fn(poll_interval)
            if (now_fn() - start_time) > deadline_seconds:
                return "poll_deadline_exceeded"


# ---------------------------------------------------------------------------
# Bounded Log Consumer (OP09)
# ---------------------------------------------------------------------------

def consume_kernel_logs(
    kernel_ref: str,
    expected_version: int,
    transport: FakeKaggleTransport,
    *,
    max_bytes: int = 65536,
    deadline_seconds: float = 300.0,
    clock_fn: Any = None,
    cancellation: Any = None,
) -> dict[str, object]:
    """Consume bounded kernel logs with version checks, incremental reading, and credential redaction."""
    if type(expected_version) is not int or isinstance(expected_version, bool):
        raise Amazon4UError("expected_version must be an int (no bool-as-int)")

    now_fn = clock_fn or time.monotonic
    start_time = now_fn()
    deadline = start_time + deadline_seconds

    def redact_text(text: str) -> str:
        r = re.sub(r"(bearer\s+)[A-Za-z0-9_\.-]+", r"\\1[REDACTED]", text, flags=re.I)
        r = re.sub(r"(token|key|password|credential|secret|bearer)\s*[:=]\s*[A-Za-z0-9_\.-]+", r"\\1=[REDACTED]", r, flags=re.I)
        return r

    def cap_utf8(text: str) -> str:
        return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")

    if hasattr(transport, "open_log_reader"):
        reader = None
        try:
            reader_res = transport.open_log_reader(kernel_ref, version=expected_version, deadline=deadline, cancellation=cancellation)
            if reader_res.get("returncode") != 0:
                raw_err = redact_text(str(reader_res.get("error", "Log stream unavailable")))
                return {
                    "status": "unavailable",
                    "error": cap_utf8(raw_err),
                    "logs": "",
                    "version": expected_version,
                }
            if reader_res.get("version") != expected_version or reader_res.get("kernel_ref") != kernel_ref:
                return {
                    "status": "error",
                    "error": "Log version/ref mismatch",
                    "logs": "",
                    "version": reader_res.get("version"),
                }
            reader = reader_res.get("reader")
            raw_acc = bytearray()
            truncated = False
            while True:
                if now_fn() > deadline:
                    return {
                        "status": "error",
                        "error": "Log consumption deadline exceeded",
                        "logs": cap_utf8(redact_text(raw_acc.decode("utf-8", errors="ignore"))),
                        "version": expected_version,
                    }
                chunk = reader.read(MAX_CHUNK_BYTES, deadline=deadline, clock_fn=now_fn, cancellation=cancellation)
                if not chunk:
                    break
                if len(raw_acc) + len(chunk) > max_bytes:
                    allowed = max(0, max_bytes - len(raw_acc))
                    raw_acc.extend(chunk[:allowed])
                    truncated = True
                    break
                raw_acc.extend(chunk)

            text_logs = redact_text(raw_acc.decode("utf-8", errors="ignore"))
            raw_bytes = text_logs.encode("utf-8")
            truncated = truncated or len(raw_bytes) > max_bytes
            if truncated:
                marker = "\\n...[TRUNCATED_BY_CONSUMER]"
                marker_bytes = marker.encode("utf-8")
                if max_bytes < len(marker_bytes):
                    text_logs = marker_bytes[:max_bytes].decode("utf-8", errors="ignore")
                else:
                    keep = max_bytes - len(marker_bytes)
                    text_logs = raw_bytes[:keep].decode("utf-8", errors="ignore") + marker
            return {
                "status": "partial" if truncated else "available",
                "logs": text_logs,
                "version": expected_version,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": cap_utf8(redact_text(str(exc))),
                "logs": "",
                "version": expected_version,
            }
        finally:
            if reader is not None and hasattr(reader, "close"):
                reader.close()

    res = transport.logs(kernel_ref, version=expected_version, max_bytes=max_bytes)
    if res.get("returncode") != 0:
        raw_err = redact_text(str(res.get("error", "Log unavailable")))
        return {
            "status": "unavailable",
            "error": cap_utf8(raw_err),
            "logs": "",
            "version": expected_version,
        }

    if res.get("version") != expected_version or res.get("kernel_ref") != kernel_ref:
        return {
            "status": "error",
            "error": "Log version/ref mismatch",
            "logs": "",
            "version": res.get("version"),
        }

    logs = str(res.get("logs", ""))
    redacted = redact_text(logs)

    raw_bytes = redacted.encode("utf-8")
    truncated = False
    if len(raw_bytes) > max_bytes:
        truncated = True
        marker = "\\n...[TRUNCATED_BY_CONSUMER]"
        marker_bytes = marker.encode("utf-8")
        if max_bytes < len(marker_bytes):
            redacted = marker_bytes[:max_bytes].decode("utf-8", errors="ignore")
        else:
            keep_bytes = max_bytes - len(marker_bytes)
            redacted = raw_bytes[:keep_bytes].decode("utf-8", errors="ignore") + marker

    return {
        "status": "partial" if truncated else "available",
        "logs": redacted,
        "version": expected_version,
    }


# ---------------------------------------------------------------------------
# Bounded Paged Output Collection (OP10, OP11)
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACT_ROLES = {"plan", "h_result", "run_envelope", "source_manifest", "log", "checkpoint", "approval"}

def collect_outputs(
    kernel_ref: str,
    expected_version: int,
    transport: FakeKaggleTransport,
    destination_dir: Path,
    *,
    expected_spec: dict[str, object],
    expected_approval: dict[str, object],
    expected_manifest: dict[str, object] | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_output_inventory: dict[str, object] | None = None,
    expected_envelope: dict[str, object] | None = None,
    expected_run_envelope: dict[str, object] | None = None,
    max_pages: int = MAX_PAGES,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_aggregate_bytes: int = MAX_AGGREGATE_BYTES,
    max_files: int = MAX_CONCURRENT_FILES,
    deadline_seconds: float = 300.0,
    clock_fn: Any = None,
) -> dict[str, object]:
    """Collect kernel outputs into a fresh quarantine, counting bytes chunk-by-chunk.

    Refuses overwrite of existing destination (empty or foreign).
    Enforces positive finite deadline across all operations, checks elapsed time before/after
    provider calls, between chunks/files/pages, and before promotion/completion.
    Validates complete canonical 15-field H result, canonical 11-field envelope, required roles,
    and distinct checkpoint physical digest before promotion.
    """
    if destination_dir.exists():
        raise Amazon4UError(f"Destination directory '{destination_dir}' already exists; overwrite refused")

    if type(deadline_seconds) not in (int, float) or isinstance(deadline_seconds, bool) or not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        return {
            "status": "error",
            "reason": f"deadline_seconds must be a finite positive number, got {deadline_seconds!r}",
        }
    if type(max_pages) is not int or isinstance(max_pages, bool) or not 1 <= max_pages <= MAX_PAGES:
        return {"status": "error", "reason": f"max_pages must be an integer in 1..{MAX_PAGES}, got {max_pages!r}"}
    if type(max_files) is not int or isinstance(max_files, bool) or not 1 <= max_files <= MAX_CONCURRENT_FILES:
        return {"status": "error", "reason": f"max_files must be an integer in 1..{MAX_CONCURRENT_FILES}, got {max_files!r}"}
    if type(max_file_bytes) is not int or isinstance(max_file_bytes, bool) or not 1 <= max_file_bytes <= MAX_FILE_BYTES:
        return {"status": "error", "reason": f"max_file_bytes must be an integer in 1..{MAX_FILE_BYTES}, got {max_file_bytes!r}"}
    if type(max_aggregate_bytes) is not int or isinstance(max_aggregate_bytes, bool) or not 1 <= max_aggregate_bytes <= MAX_AGGREGATE_BYTES:
        return {"status": "error", "reason": f"max_aggregate_bytes must be an integer in 1..{MAX_AGGREGATE_BYTES}, got {max_aggregate_bytes!r}"}
    if clock_fn is not None and not callable(clock_fn):
        return {"status": "error", "reason": "clock_fn must be callable when provided"}

    # Early Expected Metadata Validation & Snapshot before any provider callbacks
    eff_expected_envelope = expected_envelope if expected_envelope is not None else expected_run_envelope
    if eff_expected_envelope is None or not isinstance(eff_expected_envelope, dict):
        return {"status": "error", "reason": "expected_envelope is mandatory and must be a dictionary"}
    if expected_spec is None or not isinstance(expected_spec, dict):
        return {"status": "error", "reason": "expected_spec is mandatory and must be a dictionary"}
    if expected_approval is None or not isinstance(expected_approval, dict):
        return {"status": "error", "reason": "expected_approval is mandatory and must be a dictionary"}
    if expected_manifest is None or not isinstance(expected_manifest, dict):
        return {"status": "error", "reason": "expected_manifest is mandatory and must be a dictionary"}
    if expected_output_inventory is None or not isinstance(expected_output_inventory, dict) or not expected_output_inventory:
        return {"status": "error", "reason": "expected_output_inventory is mandatory and cannot be empty"}

    try:
        snap_expected_spec = get_canonical_h_spec(expected_spec)
    except Exception as exc:
        return {"status": "error", "reason": f"expected_spec invalid: {exc}"}

    try:
        snap_expected_approval = validate_canonical_h_authorization(expected_approval, actual_spec=snap_expected_spec)
    except Exception as exc:
        return {"status": "error", "reason": f"expected_approval invalid: {exc}"}

    try:
        snap_expected_envelope = validate_canonical_h_envelope(
            eff_expected_envelope,
            expected_spec=snap_expected_spec,
        )
    except Exception as exc:
        return {"status": "error", "reason": f"expected_envelope invalid: {exc}"}

    snap_expected_manifest = copy.deepcopy(expected_manifest)
    snap_expected_inventory = copy.deepcopy(expected_output_inventory)
    snap_expected_checkpoint_sha = str(expected_checkpoint_sha256) if expected_checkpoint_sha256 is not None else None

    active_clock = time.monotonic if clock_fn is None else clock_fn

    def _sample_clock() -> float:
        try:
            val = active_clock()
        except Exception as exc:
            raise Amazon4UError(f"Clock function raised exception: {exc}") from exc
        if type(val) not in (int, float) or isinstance(val, bool) or not math.isfinite(val):
            raise Amazon4UError(f"Clock returned non-finite or invalid sample: {val!r}")
        return float(val)

    try:
        start_time = _sample_clock()
    except Amazon4UError as exc:
        return {"status": "error", "reason": str(exc)}

    dl_limit = float(deadline_seconds)

    def _deadline_exceeded() -> bool:
        try:
            now = _sample_clock()
        except Amazon4UError:
            return True
        return (now - start_time) >= dl_limit

    def _deadline_error(stage: str) -> dict[str, object]:
        try:
            now = _sample_clock()
            elapsed = now - start_time
        except Amazon4UError:
            elapsed = dl_limit
        return {
            "status": "error",
            "reason": f"Deadline exceeded ({stage}): elapsed {elapsed:.3f}s >= deadline {dl_limit:.3f}s",
        }

    if _deadline_exceeded():
        return _deadline_error("before_quarantine_creation")

    # Fresh quarantine directory with exclusive creation (no exist_ok adoption)
    quarantine = destination_dir.parent / f".quarantine_{uuid.uuid4().hex[:12]}"
    while quarantine.exists():
        quarantine = destination_dir.parent / f".quarantine_{uuid.uuid4().hex[:12]}"
    quarantine.mkdir(parents=True, mode=0o700, exist_ok=False)
    quarantine_created = True
    quarantine_stat = quarantine.stat()
    quarantine_dev = quarantine_stat.st_dev
    quarantine_ino = quarantine_stat.st_ino

    seen_tokens: set[str] = set()
    page_token: str | None = None
    pages = 0
    total_bytes = 0
    collected_files: dict[str, Path] = {}

    try:
        while True:
            if _deadline_exceeded():
                return _deadline_error("before_page_iteration")

            pages += 1
            if pages > max_pages:
                return {
                    "status": "error",
                    "reason": f"Page limit {max_pages} exceeded during output collection",
                }

            if page_token is not None:
                if page_token in seen_tokens:
                    return {
                        "status": "error",
                        "reason": f"Cycle detected in pagination token: {page_token}",
                    }
                seen_tokens.add(page_token)

            if _deadline_exceeded():
                return _deadline_error("before_list_output_page")

            items, page_token = transport.list_output_page(kernel_ref, page_token)

            # Check deadline immediately after provider return; late returns must not cause late writes
            if _deadline_exceeded():
                return _deadline_error("after_list_output_page")

            items_iter = iter(items)
            while True:
                if _deadline_exceeded():
                    return _deadline_error("before_item_next")
                try:
                    item = next(items_iter)
                except StopIteration:
                    if _deadline_exceeded():
                        return _deadline_error("after_items_stop_iteration")
                    break
                if _deadline_exceeded():
                    return _deadline_error("after_item_next")

                if len(collected_files) >= max_files:
                    return {
                        "status": "error",
                        "reason": f"File count limit {max_files} exceeded during collection",
                    }

                filename = str(item.get("name", ""))
                if not filename or filename.startswith("/") or "\\" in filename:
                    return {
                        "status": "error",
                        "reason": f"Path escape in output filename: {filename}",
                    }
                p = Path(filename)
                if p.is_absolute() or any(part in ("..", ".", "") for part in p.parts):
                    return {
                        "status": "error",
                        "reason": f"Path traversal in output filename: {filename}",
                    }

                out_p = quarantine / filename
                if out_p.exists():
                    return {
                        "status": "error",
                        "reason": f"Duplicate file in output collection: {filename}",
                    }
                out_p.parent.mkdir(parents=True, exist_ok=True)

                if _deadline_exceeded():
                    return _deadline_error(f"before_file_open_{filename}")

                chunks = item.get("chunks")
                if chunks is None:
                    raw = item.get("content", b"")
                    if isinstance(raw, str):
                        raw = raw.encode("utf-8")
                    if len(raw) > max_file_bytes:
                        return {
                            "status": "error",
                            "reason": f"File {filename} content exceeds max_file_bytes {max_file_bytes}",
                        }
                    if total_bytes + len(raw) > max_aggregate_bytes:
                        return {
                            "status": "error",
                            "reason": f"Aggregate collection size exceeds limit {max_aggregate_bytes}",
                        }
                    chunks = [raw[i:i + MAX_CHUNK_BYTES] for i in range(0, len(raw), MAX_CHUNK_BYTES)] if raw else [b""]

                file_size = 0
                chunks_iter = iter(chunks)
                with out_p.open("wb") as fh:
                    while True:
                        if _deadline_exceeded():
                            return _deadline_error(f"before_chunk_next_{filename}")
                        try:
                            chunk = next(chunks_iter)
                        except StopIteration:
                            if _deadline_exceeded():
                                return _deadline_error(f"after_chunks_stop_iteration_{filename}")
                            break
                        if _deadline_exceeded():
                            return _deadline_error(f"after_chunk_next_{filename}")

                        if len(chunk) > MAX_CHUNK_BYTES:
                            return {
                                "status": "error",
                                "reason": f"Chunk size {len(chunk)} exceeds max chunk bytes {MAX_CHUNK_BYTES}",
                            }
                        file_size += len(chunk)
                        if file_size > max_file_bytes:
                            return {
                                "status": "error",
                                "reason": f"File {filename} size {file_size} exceeds max_file_bytes {max_file_bytes}",
                            }
                        if total_bytes + len(chunk) > max_aggregate_bytes:
                            return {
                                "status": "error",
                                "reason": f"Aggregate collection size exceeds limit {max_aggregate_bytes}",
                            }
                        total_bytes += len(chunk)
                        fh.write(chunk)
                        if _deadline_exceeded():
                            return _deadline_error(f"after_chunk_write_{filename}")

                if _deadline_exceeded():
                    return _deadline_error(f"after_file_write_{filename}")

                collected_files[filename] = out_p

            if _deadline_exceeded():
                return _deadline_error("between_pages")

            if page_token is None:
                break

        if _deadline_exceeded():
            return _deadline_error("before_validation")

        # Check artifact completeness and crosslinks using pre-callback snapshots
        validation = validate_collected_artifacts(
            collected_files,
            expected_version=expected_version,
            expected_spec=snap_expected_spec,
            expected_approval=snap_expected_approval,
            expected_manifest=snap_expected_manifest,
            expected_checkpoint_sha256=snap_expected_checkpoint_sha,
            expected_output_inventory=snap_expected_inventory,
            expected_envelope=snap_expected_envelope,
        )
        if validation["status"] != "complete":
            return validation

        if _deadline_exceeded():
            return _deadline_error("before_promotion")

        # Promote quarantine to destination using genuine no-replace seam without rename fallback
        if destination_dir.exists():
            return {
                "status": "error",
                "reason": f"Destination directory appeared concurrently: {destination_dir}",
            }
        destination_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not _rename_no_replace_linux(quarantine, destination_dir):
                return {
                    "status": "error",
                    "reason": f"Native directory no-replace is unsupported on this platform; ordinary rename fallback is forbidden: {destination_dir}",
                }
            quarantine_created = False
        except FileExistsError:
            return {
                "status": "error",
                "reason": f"Destination directory appeared concurrently: {destination_dir}",
            }
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                return {
                    "status": "error",
                    "reason": f"Destination directory appeared concurrently: {destination_dir}",
                }
            return {
                "status": "error",
                "reason": f"Native directory no-replace failed: {exc}",
            }

        if _deadline_exceeded():
            return _deadline_error("before_completion")

        return {
            "status": "complete",
            "destination": str(destination_dir),
            "files": list(collected_files.keys()),
            "total_bytes": total_bytes,
            "version": expected_version,
            "roles": validation.get("roles", {}),
        }

    finally:
        if quarantine_created and quarantine.exists():
            try:
                # Safe owner and identity verification before rmtree; retain if safe owner cleanup cannot be established
                cur_stat = quarantine.stat()
                if (
                    not quarantine.is_symlink()
                    and cur_stat.st_uid == os.getuid()
                    and cur_stat.st_dev == quarantine_dev
                    and cur_stat.st_ino == quarantine_ino
                ):
                    shutil.rmtree(quarantine)
            except OSError:
                pass


def validate_collected_artifacts(
    files: dict[str, Path],
    *,
    expected_version: int,
    expected_spec: dict[str, object],
    expected_approval: dict[str, object],
    expected_manifest: dict[str, object] | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_output_inventory: dict[str, object] | None = None,
    expected_envelope: dict[str, object] | None = None,
    expected_run_envelope: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate completeness, roles, and cross-links among collected artifacts against expectations."""
    if type(expected_version) is not int or isinstance(expected_version, bool) or expected_version < 1:
        return {"status": "error", "reason": "expected_version must be a positive integer"}
    if expected_manifest is None or not isinstance(expected_manifest, dict):
        return {"status": "error", "reason": "expected_manifest is mandatory; cannot be None"}
    if expected_output_inventory is None or not isinstance(expected_output_inventory, dict) or not expected_output_inventory:
        return {"status": "error", "reason": "expected_output_inventory is mandatory and cannot be empty"}
    if expected_spec is None or not isinstance(expected_spec, dict):
        return {"status": "error", "reason": "expected_spec is mandatory and must be a dictionary"}
    if expected_approval is None or not isinstance(expected_approval, dict):
        return {"status": "error", "reason": "expected_approval is mandatory and must be a dictionary"}

    eff_expected_envelope = expected_envelope if expected_envelope is not None else expected_run_envelope
    if eff_expected_envelope is None or not isinstance(eff_expected_envelope, dict):
        return {"status": "error", "reason": "expected_envelope is mandatory; cannot be None"}

    try:
        canonical_exp_spec = get_canonical_h_spec(expected_spec)
    except Exception as exc:
        return {"status": "error", "reason": f"expected_spec invalid: {exc}"}

    try:
        valid_exp_auth = validate_canonical_h_authorization(expected_approval, actual_spec=canonical_exp_spec)
    except Exception as exc:
        return {"status": "error", "reason": f"expected_approval invalid: {exc}"}

    try:
        valid_exp_env = validate_canonical_h_envelope(
            eff_expected_envelope,
            expected_spec=canonical_exp_spec,
        )
    except Exception as exc:
        return {"status": "error", "reason": f"expected_envelope invalid: {exc}"}

    is_baseline = canonical_exp_spec.get("method") in ("CorePopularity", "KG-only")

    # Upfront exact required roles in expected_output_inventory
    required_inv_roles = {"plan", "h_result", "run_envelope", "source_manifest", "log", "checkpoint"}
    missing_inv_roles = required_inv_roles - set(expected_output_inventory.keys())
    if missing_inv_roles:
        return {"status": "error", "reason": f"expected_output_inventory missing required roles: {missing_inv_roles}"}

    # Validate inventory entries upfront
    ckpt_inv = expected_output_inventory.get("checkpoint")
    if is_baseline:
        if not isinstance(ckpt_inv, dict):
            return {"status": "error", "reason": "Baseline checkpoint inventory must be a complete metadata dictionary, bare markers forbidden"}
        if ckpt_inv.get("status") != "not_applicable_deterministic_baseline":
            return {"status": "error", "reason": "Baseline checkpoint inventory status must be 'not_applicable_deterministic_baseline'"}
        if ckpt_inv.get("applicable") is not False:
            return {"status": "error", "reason": "Baseline checkpoint inventory applicable must be False"}
        if ckpt_inv.get("method") != canonical_exp_spec.get("method"):
            return {"status": "error", "reason": f"Baseline checkpoint inventory method mismatch: {ckpt_inv.get('method')} != {canonical_exp_spec.get('method')}"}
        if type(ckpt_inv.get("epoch")) is not int or isinstance(ckpt_inv.get("epoch"), bool) or ckpt_inv["epoch"] != 0:
            return {"status": "error", "reason": "Baseline checkpoint inventory epoch must be integer 0"}
        if type(ckpt_inv.get("actual_Adam_steps")) is not int or isinstance(ckpt_inv.get("actual_Adam_steps"), bool) or ckpt_inv["actual_Adam_steps"] != 0:
            return {"status": "error", "reason": "Baseline checkpoint inventory actual_Adam_steps must be integer 0"}
        if ckpt_inv.get("selection") is not None:
            return {"status": "error", "reason": "Baseline checkpoint inventory selection must be None"}
        if expected_checkpoint_sha256 is not None and expected_checkpoint_sha256 != "not_applicable_deterministic_baseline":
            return {"status": "error", "reason": "Deterministic baseline cannot claim physical checkpoint SHA"}
    else:
        if not isinstance(ckpt_inv, dict):
            return {"status": "error", "reason": "Learned method checkpoint inventory must be a dictionary"}
        ckpt_fname = ckpt_inv.get("filename") or ckpt_inv.get("name")
        if not isinstance(ckpt_fname, str) or not ckpt_fname:
            return {"status": "error", "reason": "Learned checkpoint inventory missing filename"}
        if "size" not in ckpt_inv and "length" not in ckpt_inv:
            return {"status": "error", "reason": "Learned checkpoint inventory missing required 'size' field"}
        ckpt_sz = ckpt_inv.get("size") if "size" in ckpt_inv else ckpt_inv.get("length")
        if type(ckpt_sz) is not int or isinstance(ckpt_sz, bool) or ckpt_sz < 0 or ckpt_sz > MAX_FILE_BYTES:
            return {"status": "error", "reason": "Learned checkpoint inventory size must be non-negative bounded integer"}
        ckpt_sha_val = ckpt_inv.get("physical_sha256") or ckpt_inv.get("sha256")
        if not isinstance(ckpt_sha_val, str) or len(ckpt_sha_val) != 64 or any(c not in "0123456789abcdef" for c in ckpt_sha_val):
            return {"status": "error", "reason": "Learned checkpoint inventory physical_sha256 must be 64-character lowercase hex string"}
        if not expected_checkpoint_sha256 or not isinstance(expected_checkpoint_sha256, str) or len(expected_checkpoint_sha256) != 64:
            return {"status": "error", "reason": "expected_checkpoint_sha256 mandatory 64-character hex string for learned methods"}
        if ckpt_sha_val != expected_checkpoint_sha256:
            return {"status": "error", "reason": f"Checkpoint inventory physical SHA mismatch: expected {expected_checkpoint_sha256}, got {ckpt_sha_val}"}

    for role in ("plan", "h_result", "run_envelope", "source_manifest", "log"):
        item = expected_output_inventory.get(role)
        if not isinstance(item, dict):
            return {"status": "error", "reason": f"expected_output_inventory role '{role}' must be a dictionary"}
        fname = item.get("filename") or item.get("name")
        if not isinstance(fname, str) or not fname or ".." in fname or fname.startswith("/"):
            return {"status": "error", "reason": f"expected_output_inventory role '{role}' missing valid filename"}
        if "size" not in item and "length" not in item:
            return {"status": "error", "reason": f"expected_output_inventory role '{role}' missing required size field"}
        sz = item.get("size") if "size" in item else item.get("length")
        if type(sz) is not int or isinstance(sz, bool) or sz < 0 or sz > MAX_FILE_BYTES:
            return {"status": "error", "reason": f"expected_output_inventory role '{role}' size must be non-negative bounded integer"}
        sh = item.get("sha256") or item.get("physical_sha256")
        if sh is None or not isinstance(sh, str) or len(sh) != 64 or any(c not in "0123456789abcdef" for c in sh):
            return {"status": "error", "reason": f"expected_output_inventory role '{role}' sha256 must be 64-character lowercase hex string"}

    roles_found: dict[str, str] = {}
    roles_found["version"] = str(expected_version)

    role_alias_patterns: dict[str, tuple[str, ...]] = {
        "plan": ("plan.json",),
        "h_result": ("result.json", "h_result.json"),
        "run_envelope": ("run_envelope.json",),
        "source_manifest": ("manifest.json",),
        "checkpoint": ("checkpoint.bin", "checkpoint.pt", "checkpoint_payload.json"),
        "approval": ("authorization.json", "approval.json"),
    }

    # Detect all recognized candidate files in files per role
    role_candidates: dict[str, list[str]] = {}
    all_check_roles = ("plan", "h_result", "run_envelope", "source_manifest", "log", "checkpoint", "approval")
    for r in all_check_roles:
        exp_item = expected_output_inventory.get(r)
        exp_fname = (exp_item.get("filename") or exp_item.get("name")) if isinstance(exp_item, dict) else None

        matches = []
        for f in sorted(files.keys()):
            is_match = False
            if r in role_alias_patterns:
                is_match = f in role_alias_patterns[r]
            if is_match or (exp_fname is not None and f == exp_fname):
                matches.append(f)
        role_candidates[r] = matches

    # Detect and reject ambiguous aliases: if alternative recognized filenames for the same role exist in files
    for r, matches in role_candidates.items():
        if len(matches) > 1:
            return {
                "status": "error",
                "reason": f"Ambiguous artifact aliases for role '{r}': {matches}",
            }

    # Reject unapproved standalone authorization files when authorization is not in inventory
    if "approval" not in expected_output_inventory:
        unapproved_auth = [f for f in role_alias_patterns["approval"] if f in files]
        if unapproved_auth:
            return {
                "status": "error",
                "reason": f"Unapproved standalone authorization artifact present when not in expected_output_inventory: {unapproved_auth}",
            }

    # Bind roles_found to the approved inventory filename and reject alias mismatch
    for r in all_check_roles:
        matches = role_candidates.get(r, [])
        exp_item = expected_output_inventory.get(r)
        if isinstance(exp_item, dict) and (exp_item.get("filename") or exp_item.get("name")) is not None:
            exp_fname = exp_item.get("filename") or exp_item.get("name")
            if exp_fname in files:
                roles_found[r] = exp_fname
            elif matches:
                return {
                    "status": "error",
                    "reason": f"Role '{r}' artifact mismatch: inventory specifies '{exp_fname}', but found alias '{matches[0]}' in files",
                }

    if "plan" in roles_found:
        try:
            plan_data = json.loads(files[roles_found["plan"]].read_text(encoding="utf-8"))
        except Exception as exc:
            return {"status": "error", "reason": f"Invalid JSON in plan artifact: {exc}"}
        if not isinstance(plan_data, dict):
            return {"status": "error", "reason": "plan.json root must be a JSON dictionary"}
        if "approval" not in roles_found and "authorization" in plan_data:
            roles_found["approval"] = roles_found["plan"]

    if is_baseline:
        unapproved_ckpts = [f for f in role_alias_patterns["checkpoint"] if f in files]
        if unapproved_ckpts:
            return {
                "status": "error",
                "reason": f"Deterministic baseline cannot provide physical checkpoint file: {unapproved_ckpts}",
            }
        roles_found["checkpoint"] = "not_applicable_deterministic_baseline"
        roles_found["checkpoint_physical_sha256"] = "not_applicable_deterministic_baseline"

    missing_roles = REQUIRED_ARTIFACT_ROLES - set(roles_found.keys())
    if missing_roles:
        if "checkpoint" in missing_roles:
            return {
                "status": "unrecoverable",
                "reason": f"Missing required artifact roles: {missing_roles}",
                "roles": roles_found,
            }
        return {
            "status": "partial",
            "reason": f"Missing required artifact roles: {missing_roles}",
            "roles": roles_found,
        }

    try:
        result_data = json.loads(files[roles_found["h_result"]].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "reason": f"Invalid JSON in result artifact: {exc}"}
    if not isinstance(result_data, dict):
        return {"status": "error", "reason": "result.json root must be a JSON dictionary"}

    try:
        env_data = json.loads(files[roles_found["run_envelope"]].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "reason": f"Invalid JSON in run_envelope artifact: {exc}"}
    if not isinstance(env_data, dict):
        return {"status": "error", "reason": "run_envelope.json root must be a JSON dictionary"}

    try:
        manifest_data = json.loads(files[roles_found["source_manifest"]].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "reason": f"Invalid JSON in manifest artifact: {exc}"}
    if not isinstance(manifest_data, dict):
        return {"status": "error", "reason": "manifest.json root must be a JSON dictionary"}

    plan_spec = plan_data.get("spec")
    if not isinstance(plan_spec, dict):
        return {"status": "error", "reason": "Artifact plan spec must be a dictionary"}
    try:
        validate_h_spec(plan_spec, expected_spec=canonical_exp_spec)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"Artifact plan spec validation failed (spec does not match expected_spec): {exc}",
        }

    plan_auth = plan_data.get("authorization")
    if not isinstance(plan_auth, dict):
        return {"status": "error", "reason": "Artifact plan authorization must be a dictionary"}
    if plan_auth != valid_exp_auth:
        return {
            "status": "error",
            "reason": "Artifact plan authorization does not match expected_approval",
        }

    # Validate authorization structure
    if plan_auth.get("schema_version") != "amazon4u-H-authorization-v1":
        return {"status": "error", "reason": "Artifact plan authorization schema_version must be 'amazon4u-H-authorization-v1'"}
    if plan_auth.get("scope") != "tiny_synthetic_cpu_backward_and_private_restore":
        return {"status": "error", "reason": "Artifact plan authorization scope must be 'tiny_synthetic_cpu_backward_and_private_restore'"}
    auth_steps = plan_auth.get("max_adam_steps")
    if type(auth_steps) is not int or isinstance(auth_steps, bool) or not 0 <= auth_steps <= 4:
        return {"status": "error", "reason": "Artifact plan authorization max_adam_steps must be int 0..4"}
    if plan_auth.get("run_id") != canonical_exp_spec.get("run_id"):
        return {"status": "error", "reason": "Artifact plan authorization run_id mismatch with spec"}
    if plan_auth.get("approval_id") != canonical_exp_spec.get("approval_id"):
        return {"status": "error", "reason": "Artifact plan authorization approval_id mismatch with spec"}
    if canonical_exp_spec.get("max_adam_steps", 0) > auth_steps:
        return {"status": "error", "reason": "Artifact plan spec max_adam_steps exceeds authorized budget"}

    plan_version = plan_data.get("version")
    if type(plan_version) is not int or isinstance(plan_version, bool) or plan_version != expected_version:
        return {"status": "error", "reason": f"Plan version mismatch or missing: {plan_version!r} != {expected_version}"}

    # Validate canonical H run envelope
    try:
        validate_canonical_h_envelope(
            env_data,
            expected_spec=canonical_exp_spec,
            expected_envelope=valid_exp_env,
        )
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"Crosslink mismatch in run_envelope: {exc}",
        }

    # H-result validation: schema amazon4u-H-result-v1 with complete 15 fields, no transport version
    if not isinstance(result_data, dict):
        return {"status": "error", "reason": "result.json must be a dictionary"}
    if "version" in result_data:
        return {"status": "error", "reason": "Transport version must not be inserted into the frozen H-result object"}
    if result_data.get("schema_version") != "amazon4u-H-result-v1":
        return {"status": "error", "reason": f"Invalid result schema: {result_data.get('schema_version')}"}

    missing_result_fields = H_RESULT_KEYS - set(result_data.keys())
    extra_result_fields = set(result_data.keys()) - H_RESULT_KEYS
    if missing_result_fields or extra_result_fields:
        return {
            "status": "error",
            "reason": f"result.json must contain exact 15 canonical H fields; missing {missing_result_fields}, extra {extra_result_fields}",
        }

    if result_data.get("acceptance_scope") != "H-REF_CPU_reference_fixture_only":
        return {"status": "error", "reason": f"Invalid result acceptance_scope: {result_data.get('acceptance_scope')}"}
    if result_data.get("deployment_decision") != "NOT_AUTHORIZED":
        return {"status": "error", "reason": f"Invalid result deployment_decision: {result_data.get('deployment_decision')}"}
    if result_data.get("test_execution") != "not_authorized":
        return {"status": "error", "reason": f"Invalid result test_execution: {result_data.get('test_execution')}"}
    if result_data.get("remaining_readiness_gates") != READINESS_GATES:
        return {"status": "error", "reason": "result.json remaining_readiness_gates mismatch with frozen readiness gates"}

    dev_cap = result_data.get("device_capability")
    if not isinstance(dev_cap, dict):
        return {"status": "error", "reason": "result.json device_capability must be a dictionary"}
    if (
        dev_cap.get("device") != "cpu"
        or dev_cap.get("accelerator") != "none"
        or type(dev_cap.get("supports_training")) is not bool
        or dev_cap.get("supports_training") is not True
        or dev_cap.get("state_transfer") != "cpu_quiescent_epoch_v1"
    ):
        return {
            "status": "error",
            "reason": f"result.json device_capability mismatch: expected {{'device': 'cpu', 'accelerator': 'none', 'supports_training': True, 'state_transfer': 'cpu_quiescent_epoch_v1'}}, got {dev_cap}",
        }

    res_spec = result_data.get("run_spec")
    if not isinstance(res_spec, dict):
        return {
            "status": "error",
            "reason": "Crosslink mismatch in result.json run_spec vs expected_spec: not a dictionary",
        }
    try:
        validate_h_spec(res_spec, expected_spec=canonical_exp_spec)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"Crosslink mismatch in result.json run_spec vs expected_spec: {exc}",
        }

    res_env = result_data.get("run_envelope")
    if not isinstance(res_env, dict) or res_env != env_data:
        return {
            "status": "error",
            "reason": "Crosslink mismatch between result.json run_envelope and run_envelope.json",
        }

    if result_data.get("method") != canonical_exp_spec.get("method"):
        return {
            "status": "error",
            "reason": f"Crosslink mismatch in result.json method: {result_data.get('method')} != {canonical_exp_spec.get('method')}",
        }

    # Checkpoint applicability and verification
    approved_epochs = canonical_exp_spec["epochs"]
    approved_max_adam_steps = canonical_exp_spec["max_adam_steps"]
    auth_max_adam_steps = valid_exp_auth["max_adam_steps"]

    if is_baseline:
        if result_data.get("status") != "deterministic_validation_complete":
            return {"status": "error", "reason": "Deterministic baseline status must be 'deterministic_validation_complete'"}
        if (
            "epoch" not in result_data or type(result_data["epoch"]) is not int or isinstance(result_data["epoch"], bool) or result_data["epoch"] != 0
            or "actual_Adam_steps" not in result_data or type(result_data["actual_Adam_steps"]) is not int or isinstance(result_data["actual_Adam_steps"], bool) or result_data["actual_Adam_steps"] != 0
            or "selection" not in result_data or result_data["selection"] is not None
        ):
            return {"status": "error", "reason": "Deterministic baseline must explicitly have epoch=0, steps=0, selection=None (no defaulting)"}
        if expected_checkpoint_sha256 is not None and expected_checkpoint_sha256 != "not_applicable_deterministic_baseline":
            return {"status": "error", "reason": "Deterministic baseline cannot claim physical checkpoint SHA"}
    else:
        if result_data.get("status") not in ("fixture_budget_complete", "scientific_terminal"):
            return {"status": "error", "reason": f"Learned result status must be 'fixture_budget_complete' or 'scientific_terminal', got {result_data.get('status')}"}
        if roles_found.get("checkpoint") == "not_applicable_deterministic_baseline" or roles_found.get("checkpoint") not in files:
            return {"status": "unrecoverable", "reason": "Learned method requires physical checkpoint file"}
        if (
            "epoch" not in result_data
            or type(result_data["epoch"]) is not int
            or isinstance(result_data["epoch"], bool)
            or not 1 <= result_data["epoch"] <= approved_epochs
        ):
            return {"status": "error", "reason": f"Learned method must explicitly have epoch in 1..{approved_epochs}"}
        if (
            "actual_Adam_steps" not in result_data
            or type(result_data["actual_Adam_steps"]) is not int
            or isinstance(result_data["actual_Adam_steps"], bool)
            or not 1 <= result_data["actual_Adam_steps"] <= approved_max_adam_steps
            or result_data["actual_Adam_steps"] > auth_max_adam_steps
        ):
            return {"status": "error", "reason": f"Learned method must explicitly have actual_Adam_steps in 1..{approved_max_adam_steps} (authorized: {auth_max_adam_steps})"}
        if "selection" not in result_data or not isinstance(result_data["selection"], dict):
            return {"status": "error", "reason": "Learned method must explicitly have a dictionary selection"}
        if result_data.get("mixture") is not None:
            return {"status": "error", "reason": "Learned method mixture must be None"}

        ckpt_path = files[roles_found["checkpoint"]]
        ckpt_bytes = ckpt_path.read_bytes()
        physical_sha = hashlib.sha256(ckpt_bytes).hexdigest()
        roles_found["checkpoint_physical_sha256"] = physical_sha

        if not expected_checkpoint_sha256 or expected_checkpoint_sha256 == "not_applicable_deterministic_baseline":
            return {"status": "error", "reason": "expected_checkpoint_sha256 is mandatory for learned methods"}
        if physical_sha != expected_checkpoint_sha256:
            return {
                "status": "error",
                "reason": f"Checkpoint physical SHA does not match expected {expected_checkpoint_sha256}, got {physical_sha}",
            }

    # Verify no proxy metrics or unregistered metrics in validation_report
    val_rep = result_data.get("validation_report")
    if val_rep is not None:
        if not isinstance(val_rep, dict):
            return {"status": "error", "reason": "validation_report must be a dictionary or None"}
        for forbidden in FORBIDDEN_PROXY_KEYS:
            if forbidden in val_rep:
                return {"status": "error", "reason": f"Forbidden proxy metric '{forbidden}' in H validation report"}
        for sub_sec in ("overall_metrics", "metrics"):
            if sub_sec in val_rep:
                sec = val_rep[sub_sec]
                if not isinstance(sec, dict):
                    return {"status": "error", "reason": f"validation_report section '{sub_sec}' must be a dictionary"}
                for forbidden in FORBIDDEN_PROXY_KEYS:
                    if forbidden in sec:
                        return {"status": "error", "reason": f"Forbidden proxy metric '{forbidden}' in H validation report"}
                unreg = set(sec.keys()) - REGISTERED_SEVEN_METRICS
                if unreg:
                    return {"status": "error", "reason": f"Unregistered metric keys in {sub_sec}: {unreg}"}
                for k, val in sec.items():
                    if val is not None and val != "N/A":
                        if type(val) not in (int, float) or isinstance(val, bool) or not math.isfinite(val):
                            return {"status": "error", "reason": f"Metric {k} must be finite number or None/N/A, got {val}"}
        if "metrics" in val_rep and "overall_metrics" not in val_rep:
            return {"status": "error", "reason": "Validation report must use 'overall_metrics', not invented 'metrics'"}

    # Manifest verification (not digest-only)
    try:
        verify_source_manifest(manifest_data, expected_manifest)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"Manifest verification failed against expected_manifest: {exc}",
        }

    for role, exp_item in expected_output_inventory.items():
        if role in ("checkpoint", "checkpoint_physical_sha256") and is_baseline:
            continue
        if isinstance(exp_item, dict):
            exp_file = exp_item.get("filename") or exp_item.get("name")
            if not exp_file or not isinstance(exp_file, str):
                return {"status": "error", "reason": f"Inventory entry for role {role} missing valid filename"}
            if exp_file not in files:
                return {"status": "unrecoverable", "reason": f"Inventory file {exp_file} missing for role {role}"}

            exp_size = exp_item.get("size") if "size" in exp_item else exp_item.get("length")
            actual_size = files[exp_file].stat().st_size
            if actual_size != exp_size:
                return {"status": "error", "reason": f"Inventory size mismatch for {exp_file}: expected {exp_size}, got {actual_size}"}

            exp_sha = exp_item.get("sha256") or exp_item.get("physical_sha256")
            file_sha = hashlib.sha256(files[exp_file].read_bytes()).hexdigest()
            if file_sha != exp_sha:
                return {"status": "error", "reason": f"Inventory SHA mismatch for {exp_file}: expected {exp_sha}, got {file_sha}"}

    return {
        "status": "complete",
        "roles": roles_found,
    }


# ---------------------------------------------------------------------------
# Immutable State Management (OP12)
# ---------------------------------------------------------------------------

class Amazon4UStateManager:
    """Manages immutable run/attempt state files without mutating legacy state."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir

    def record_attempt(
        self,
        profile: str,
        run_id: str,
        attempt_id: str,
        state: dict[str, object],
    ) -> Path:
        if profile != "amazon4u":
            raise Amazon4UError(f"Amazon4UStateManager only handles profile 'amazon4u', got '{profile}'")

        # Validate bounded identity components (no path traversal, slashes, or special characters)
        ident_pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        if not ident_pattern.match(run_id):
            raise Amazon4UError(f"Invalid run_id format: '{run_id}'")
        if not ident_pattern.match(attempt_id):
            raise Amazon4UError(f"Invalid attempt_id format: '{attempt_id}'")

        self.state_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{run_id}_{attempt_id}.json"
        target_path = self.state_dir / filename

        state_copy = copy.deepcopy(state)
        state_copy["recorded_at"] = "2026-10-19T00:00:00Z"
        content_bytes = (json.dumps(state_copy, indent=2, sort_keys=True) + "\n").encode("utf-8")

        # Atomic exclusive create (O_CREAT | O_EXCL) to prevent overwrite
        try:
            fd = os.open(str(target_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with open(fd, "wb") as f:
                f.write(content_bytes)
        except FileExistsError:
            raise Amazon4UError(
                f"State record for {run_id}/{attempt_id} already exists at {target_path}; overwriting is prohibited"
            )

        return target_path

    def load_attempt(self, run_id: str, attempt_id: str) -> dict[str, object] | None:
        ident_pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        if not ident_pattern.match(run_id) or not ident_pattern.match(attempt_id):
            return None
        path = self.state_dir / f"{run_id}_{attempt_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Result Record Construction (OP13, OP15)
# ---------------------------------------------------------------------------

def verify_h_provenance(
    h_result: dict[str, object],
    *,
    expected_spec: dict[str, object] | None = None,
    expected_run_envelope: dict[str, object] | None = None,
) -> dict[str, object]:
    """Verify that an H execution result satisfies canonical H provenance and acceptance scope.

    Checks all required H record keys, exact acceptance_scope ('H-REF_CPU_reference_fixture_only'),
    deployment_decision 'NOT_AUTHORIZED', test_execution 'not_authorized', and absence of eval_score proxies.
    """
    if not isinstance(h_result, dict):
        raise Amazon4UError("H result must be a dictionary")
    if h_result.get("schema_version") != "amazon4u-H-result-v1":
        raise Amazon4UError(f"Invalid H result schema_version: {h_result.get('schema_version')}")

    required_h_keys = {
        "schema_version", "status", "acceptance_scope", "deployment_decision",
        "run_spec", "run_envelope", "device_capability", "method", "epoch",
        "actual_Adam_steps", "selection", "mixture", "validation_report",
        "remaining_readiness_gates", "test_execution",
    }
    missing_h = required_h_keys - set(h_result.keys())
    if missing_h:
        raise Amazon4UError(f"H result missing required H fields: {missing_h}")

    if h_result["acceptance_scope"] != "H-REF_CPU_reference_fixture_only":
        raise Amazon4UError(f"Invalid acceptance_scope: '{h_result['acceptance_scope']}'")
    if h_result["deployment_decision"] != "NOT_AUTHORIZED":
        raise Amazon4UError(f"deployment_decision must be 'NOT_AUTHORIZED', got: '{h_result['deployment_decision']}'")
    if h_result["test_execution"] != "not_authorized":
        raise Amazon4UError(f"test_execution must be 'not_authorized', got: '{h_result['test_execution']}'")

    val_report = h_result.get("validation_report")
    if val_report is not None:
        if not isinstance(val_report, dict):
            raise Amazon4UError("validation_report must be a dictionary or None")
        for forbidden in FORBIDDEN_PROXY_KEYS:
            if forbidden in val_report:
                raise Amazon4UError(f"Forbidden proxy metric '{forbidden}' in H validation report")
        for sub_sec in ("overall_metrics", "metrics"):
            if sub_sec in val_report:
                sec = val_report.get(sub_sec)
                if not isinstance(sec, dict):
                    raise Amazon4UError(f"validation_report section '{sub_sec}' must be a dictionary")
                for forbidden in FORBIDDEN_PROXY_KEYS:
                    if forbidden in sec:
                        raise Amazon4UError(f"Forbidden proxy metric '{forbidden}' in H validation report")
                unreg = set(sec.keys()) - REGISTERED_SEVEN_METRICS
                if unreg:
                    raise Amazon4UError(f"Unregistered metric keys in {sub_sec}: {unreg}")
                for k, val in sec.items():
                    if val is not None and val != "N/A":
                        if type(val) not in (int, float) or isinstance(val, bool) or not math.isfinite(val):
                            raise Amazon4UError(f"Metric {k} must be finite number or None/N/A, got {val}")
        if "metrics" in val_report and "overall_metrics" not in val_report:
            raise Amazon4UError("Validation report must use 'overall_metrics', not invented 'metrics'")

    if expected_spec is not None:
        h_run_spec = h_result.get("run_spec")
        if not isinstance(h_run_spec, dict):
            raise Amazon4UError("H result run_spec must be a dictionary")
        try:
            canonical_exp = get_canonical_h_spec(expected_spec)
        except Exception as exc:
            raise Amazon4UError(f"Invalid expected_spec in provenance verification: {exc}") from exc
        try:
            validate_h_spec(h_run_spec, expected_spec=canonical_exp)
        except Exception as exc:
            raise Amazon4UError(f"H result run_spec does not match expected_spec: {exc}") from exc

    if expected_run_envelope is not None:
        h_env = h_result.get("run_envelope")
        if not isinstance(h_env, dict):
            raise Amazon4UError("H result run_envelope must be a dictionary")
        if h_env != expected_run_envelope:
            raise Amazon4UError("H result run_envelope mismatch with expected_run_envelope")

    return copy.deepcopy(h_result)


def build_amazon4u_record(
    plan: dict[str, object],
    h_result: dict[str, object],
    *,
    evidence_kind: str = "simulated_offline_fixture",
    population_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Wrap H execution result in typed Amazon4U record without eval_score proxies."""
    if not isinstance(h_result, dict) or h_result.get("schema_version") != "amazon4u-H-result-v1":
        raise Amazon4UError("h_result must be a valid 'amazon4u-H-result-v1' dictionary")

    if evidence_kind != "simulated_offline_fixture":
        raise Amazon4UError(f"Unauthorized evidence_kind '{evidence_kind}'; must be 'simulated_offline_fixture'")

    spec = plan.get("spec", {})
    if not isinstance(spec, dict):
        raise Amazon4UError("plan must contain a spec dictionary")

    verify_h_provenance(h_result, expected_spec=spec)

    required_h_keys = {
        "schema_version", "status", "acceptance_scope", "deployment_decision",
        "run_spec", "run_envelope", "device_capability", "method", "epoch",
        "actual_Adam_steps", "selection", "mixture", "validation_report",
        "remaining_readiness_gates", "test_execution",
    }
    missing_h = required_h_keys - set(h_result.keys())
    if missing_h:
        raise Amazon4UError(f"h_result missing required H fields: {missing_h}")

    if h_result["acceptance_scope"] != "H-REF_CPU_reference_fixture_only":
        raise Amazon4UError(f"Invalid acceptance_scope: '{h_result['acceptance_scope']}'")
    if h_result["deployment_decision"] != "NOT_AUTHORIZED":
        raise Amazon4UError(f"deployment_decision must be 'NOT_AUTHORIZED', got: '{h_result['deployment_decision']}'")
    if h_result["test_execution"] != "not_authorized":
        raise Amazon4UError(f"test_execution must be 'not_authorized', got: '{h_result['test_execution']}'")

    h_run_spec = h_result.get("run_spec")
    if not isinstance(h_run_spec, dict):
        raise Amazon4UError("h_result run_spec must be a dictionary")
    canonical_plan_spec = get_canonical_h_spec(spec)
    try:
        validate_h_spec(h_run_spec, expected_spec=canonical_plan_spec)
    except Exception as exc:
        raise Amazon4UError(f"h_result run_spec does not match plan spec: {exc}") from exc

    h_env = h_result.get("run_envelope")
    if not isinstance(h_env, dict):
        raise Amazon4UError("h_result run_envelope must be a dictionary")
    if spec.get("run_id") and h_env.get("run_id") != spec.get("run_id"):
        raise Amazon4UError("h_result run_envelope run_id mismatch with plan spec")
    if spec.get("approval_id") and h_env.get("approval_id") != spec.get("approval_id"):
        raise Amazon4UError("h_result run_envelope approval_id mismatch with plan spec")

    if spec.get("method") and h_result.get("method") != spec.get("method"):
        raise Amazon4UError("h_result method mismatch with plan spec")

    h_status = h_result.get("status")
    if h_status not in LEGITIMATE_H_STATUSES:
        raise Amazon4UError(f"Illegitimate H status '{h_status}'; expected one of {LEGITIMATE_H_STATUSES}")

    method_name = spec.get("method")
    if method_name in ("CorePopularity", "KG-only"):
        if h_status != "deterministic_validation_complete":
            raise Amazon4UError(f"Deterministic baseline {method_name} must have status 'deterministic_validation_complete', got '{h_status}'")
    else:
        if h_status not in ("fixture_budget_complete", "scientific_terminal"):
            raise Amazon4UError(f"Learned method {method_name} status must be fixture_budget_complete or scientific_terminal, got '{h_status}'")

    # scientific_terminal is normal scientific stopping, not a failure
    if h_status in ("fixture_budget_complete", "deterministic_validation_complete", "scientific_terminal"):
        record_status = "simulated_offline_success"
    else:
        record_status = "simulated_offline_failure"

    if h_status == "deterministic_validation_complete":
        if (
            type(h_result.get("epoch")) is not int or isinstance(h_result.get("epoch"), bool) or h_result["epoch"] != 0
            or type(h_result.get("actual_Adam_steps")) is not int or isinstance(h_result.get("actual_Adam_steps"), bool) or h_result["actual_Adam_steps"] != 0
            or h_result.get("selection") is not None
        ):
            raise Amazon4UError("Deterministic baseline must have exact integer epoch=0, steps=0, selection=None")
    elif h_status in ("fixture_budget_complete", "scientific_terminal"):
        epoch_val = h_result.get("epoch")
        steps_val = h_result.get("actual_Adam_steps")
        if (
            type(epoch_val) is not int
            or isinstance(epoch_val, bool)
            or epoch_val < 1
            or type(steps_val) is not int
            or isinstance(steps_val, bool)
            or steps_val < 1
            or not isinstance(h_result.get("selection"), dict)
        ):
            raise Amazon4UError("Complete learned H run must have exact integer positive epoch, steps, and selection dict")

    # Validate population: must be synthetic in offline scope; reject fabricated Music_5core_sample
    if population_metadata is not None and population_metadata.get("dataset_split") == "Music_5core_sample":
        raise Amazon4UError("Fabricated Music_5core_sample population is forbidden")

    pop = population_metadata or {
        "kind": "synthetic_fixtures",
        "dataset": "Musical_Instruments",
        "split": "synthetic_offline_fixture",
    }
    if pop.get("kind") != "synthetic_fixtures":
        raise Amazon4UError("Real data population or fake-as-real is forbidden in offline scope")
    if pop.get("dataset_split") == "Music_5core_sample":
        raise Amazon4UError("Fabricated Music_5core_sample population is forbidden")

    # Validate validation_report: must use overall_metrics with the 7 frozen metrics, no proxies or recall/precision
    val_report = h_result.get("validation_report")
    if val_report is not None:
        if not isinstance(val_report, dict):
            raise Amazon4UError("validation_report must be a dictionary or None")
        if "metrics" in val_report and "overall_metrics" not in val_report:
            raise Amazon4UError("Validation report must use 'overall_metrics', not invented 'metrics'")
        for sub_sec in ("overall_metrics", "metrics"):
            if sub_sec in val_report:
                sec = val_report.get(sub_sec)
                if not isinstance(sec, dict):
                    raise Amazon4UError(f"validation_report section '{sub_sec}' must be a dictionary")

        overall_metrics = val_report.get("overall_metrics", {})

        all_report_keys = set(val_report.keys()) | set(overall_metrics.keys())
        if any(proxy in all_report_keys for proxy in FORBIDDEN_PROXY_KEYS):
            raise Amazon4UError("Forbidden proxy metric (eval_score / val_bpb / recall / precision / etc.) found in validation report")

        unregistered = set(overall_metrics.keys()) - REGISTERED_SEVEN_METRICS
        if unregistered:
            raise Amazon4UError(f"Unregistered metric keys found in overall_metrics: {unregistered}")

        if overall_metrics:
            missing_metrics = REGISTERED_SEVEN_METRICS - set(overall_metrics.keys())
            if missing_metrics:
                raise Amazon4UError(f"Missing required overall_metrics keys: {missing_metrics}")
            for k, val in overall_metrics.items():
                if val is not None and val != "N/A":
                    if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val):
                        raise Amazon4UError(f"Metric {k} must be finite number or None/N/A, got {val}")

    record_dict = {
        "schema_version": SCHEMA_RECORD,
        "profile": "amazon4u",
        "evidence_kind": evidence_kind,
        "run_id": spec.get("run_id"),
        "approval_id": spec.get("approval_id"),
        "method": spec.get("method"),
        "status": record_status,
        "population": pop,
        "h_result": copy.deepcopy(h_result),
        "readiness_gates": list(READINESS_GATES),
        "deployment_decision": "NOT_AUTHORIZED",
    }
    if isinstance(plan, dict) and "version" in plan:
        record_dict["version"] = plan["version"]
    return record_dict


# ---------------------------------------------------------------------------
# CLI Profile Dispatch (OP01, OP18)
# ---------------------------------------------------------------------------

def dispatch_amazon4u(
    args: argparse.Namespace,
    *,
    transport: FakeKaggleTransport | None = None,
    context: ApprovedOfflineContext | None = None,
) -> int:
    """Early profile dispatch hook for Amazon4U before legacy credentials/CLI/state resolution."""
    command = getattr(args, "command", "")

    # OP01: Validate profile
    profile = getattr(args, "profile", None)
    if profile != "amazon4u":
        raise Amazon4UError(f"dispatch_amazon4u called with non-amazon4u profile: '{profile}'")

    # Retained independent context loading and validation (S02)
    context_arg = getattr(args, "context", None)
    if context is None and context_arg is not None:
        p_ctx = Path(context_arg)
        if not p_ctx.exists():
            print(f"Error: Context file not found: {context_arg}", file=sys.stderr)
            return 1
        try:
            raw_ctx = json.loads(p_ctx.read_text(encoding="utf-8"))
            context = validate_approved_offline_context(raw_ctx)
        except Exception as exc:
            print(f"Error validating approved offline context: {exc}", file=sys.stderr)
            return 1
    elif context is not None:
        try:
            context = validate_approved_offline_context(context)
        except Exception as exc:
            print(f"Error validating approved offline context: {exc}", file=sys.stderr)
            return 1

    # OP18: Live execution refusal without authorized transport
    if transport is None and not getattr(args, "dry_run", False):
        print(
            "Error: Live Kaggle execution is NOT authorized in O-OFFLINE scope. "
            "Deployment NOT_AUTHORIZED; use --dry-run or offline fake transport.",
            file=sys.stderr,
        )
        return 1

    if context is None:
        print(
            "Error: Mandatory independent ApprovedOfflineContext (--context) required for Amazon4U; "
            "self-authenticating plan JSON without independent authority is strictly forbidden.",
            file=sys.stderr,
        )
        return 1

    spec_path = getattr(args, "spec", None) or getattr(args, "run_spec", None)
    approval_path = getattr(args, "approval", None)
    plan_path = getattr(args, "plan", None)

    if command == "launch":
        if not plan_path:
            print(
                "Error: Explicit independently supplied --plan is required for amazon4u launch; "
                "synthesizing authorization, stub sources, or self-expectations is strictly forbidden",
                file=sys.stderr,
            )
            return 1

        p_plan = Path(plan_path)
        if not p_plan.exists():
            print(f"Error: Plan file not found: {plan_path}", file=sys.stderr)
            return 1

        try:
            plan_data = json.loads(p_plan.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Error reading plan file: {exc}", file=sys.stderr)
            return 1

        raw_packets = plan_data.get("source_packets") or plan_data.get("sources")

        raw_spec = plan_data.get("spec") or context.canonical_h_spec
        raw_auth = plan_data.get("authorization") or context.h_authorization
        expected_spec = context.canonical_h_spec
        expected_auth = context.h_authorization
        expected_sources = context.expected_o_outer
        expected_output_inventory = context.expected_output_inventory
        expected_checkpoint_sha256 = expected_output_inventory.get("checkpoint_physical_sha256")
        expected_version = context.submission_binding.get("version")
        expected_timeout = context.submission_binding.get("timeout")
        expected_accelerator = context.expected_o_environment.get("accelerator", "none")
        expected_h_envelope = context.expected_h_envelope if context else (plan_data.get("expected_h_envelope") or plan_data.get("expected_envelope") if plan_data else None)

        # Verify command-line spec/approval match plan if provided
        if spec_path:
            p_s = Path(spec_path)
            if p_s.exists():
                user_spec = json.loads(p_s.read_text(encoding="utf-8"))
                if raw_spec is not None and user_spec != raw_spec:
                    print("Error: Command-line spec does not match plan spec", file=sys.stderr)
                    return 1
                raw_spec = user_spec

        if approval_path:
            p_a = Path(approval_path)
            if p_a.exists():
                user_auth = json.loads(p_a.read_text(encoding="utf-8"))
                if raw_auth is not None and user_auth != raw_auth:
                    print("Error: Command-line approval does not match plan authorization", file=sys.stderr)
                    return 1
                raw_auth = user_auth

        if raw_spec is None:
            print("Error: Plan must supply explicit spec", file=sys.stderr)
            return 1
        if raw_auth is None:
            print("Error: Plan must supply explicit authorization; synthesizing is forbidden", file=sys.stderr)
            return 1
        if expected_spec is None:
            print("Error: Plan must supply explicit independent expected_spec; self-defaulting to raw_spec is strictly forbidden", file=sys.stderr)
            return 1
        if expected_sources is None:
            print("Error: Plan must supply expected_sources or expected_manifest", file=sys.stderr)
            return 1

        if isinstance(raw_packets, list):
            packets = [
                SourcePacket(p["rel_path"], p["content"].encode("utf-8") if isinstance(p["content"], str) else p["content"])
                for p in raw_packets
            ]
        elif isinstance(raw_packets, dict):
            packets = [
                SourcePacket(rel, c.encode("utf-8") if isinstance(c, str) else c)
                for rel, c in raw_packets.items()
            ]
        else:
            print("Error: Plan must supply explicit source_packets or sources", file=sys.stderr)
            return 1

        try:
            valid_spec = validate_amazon4u_spec(raw_spec, expected_spec=expected_spec)
            valid_auth = validate_amazon4u_authorization(raw_auth, actual_spec=valid_spec)
            plan = render_plan(
                valid_spec,
                authorization=valid_auth,
                expected_spec=expected_spec,
                expected_sources=expected_sources,
                source_packets=packets,
                context=context,
                owner=getattr(args, "username", None) or plan_data.get("owner") or "amazon4u",
                slug=getattr(args, "slug", None) or plan_data.get("slug"),
                attempt_id=plan_data.get("attempt_id", "attempt-01"),
                timeout=expected_timeout,
                accelerator=expected_accelerator,
            )
        except Exception as exc:
            print(f"Error preparing Amazon4U plan: {exc}", file=sys.stderr)
            return 1

        if getattr(args, "dry_run", False):
            # Pure render-only even if --bundle-dir is supplied; no publication during dry-run
            print("Rendered Amazon4U plan successfully (dry-run pure render; no bundle publication).")
            return 0

        bundle_dir = getattr(args, "bundle_dir", None)
        if bundle_dir:
            try:
                write_bundle(plan, Path(bundle_dir))
                print(f"Rendered Amazon4U bundle successfully at {bundle_dir}")
            except Exception as exc:
                print(f"Error writing bundle: {exc}", file=sys.stderr)
                return 1
        else:
            print("Rendered Amazon4U plan successfully (in-memory).")

        # Reserve attempt before push
        if transport is not None:
            state_dir = Path("state") / "amazon4u"
            state_mgr = Amazon4UStateManager(state_dir)
            expected_ver = getattr(args, "version", None) or (context.submission_binding.get("version") if context else None) or plan_data.get("expected_version")
            if expected_ver is None:
                print("Error: Explicit version is required; guessing version 1 is forbidden", file=sys.stderr)
                return 1
            try:
                state_mgr.record_attempt(
                    "amazon4u",
                    str(valid_spec.get("run_id")),
                    str(plan.attempt_id),
                    {
                        "status": "reserved",
                        "plan_digest": plan.plan_digest,
                        "timeout": plan.metadata.get("timeout"),
                        "version": expected_ver,
                    },
                )
            except Exception as exc:
                print(f"Attempt reservation failed: {exc}", file=sys.stderr)
                return 1

            try:
                plan_timeout = plan.metadata.get("timeout", 7200)
                if not bundle_dir:
                    import tempfile
                    with tempfile.TemporaryDirectory() as td:
                        b_dir = Path(td) / "bundle"
                        write_bundle(plan, b_dir)
                        receipt = transport.push(
                            b_dir,
                            timeout=plan_timeout,
                            accelerator="none",
                            plan_digest=plan.plan_digest,
                        )
                else:
                    receipt = transport.push(
                        Path(bundle_dir),
                        timeout=plan_timeout,
                        accelerator="none",
                        plan_digest=plan.plan_digest,
                    )
            except SubmissionUnknownError as exc:
                print(f"Submission unknown error: {exc}; auto-repush is strictly prohibited", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Push failure: {exc}", file=sys.stderr)
                return 1

            expected_version = expected_ver
            try:
                validate_submission_receipt(
                    receipt,
                    expected_owner=plan.owner,
                    expected_slug=plan.slug,
                    expected_version=expected_version,
                    expected_timeout=plan.metadata.get("timeout"),
                    expected_accelerator="none",
                    expected_plan_digest=plan.plan_digest,
                )
            except Exception as exc:
                print(f"Submission receipt validation failed: {exc}", file=sys.stderr)
                return 1

            if not getattr(args, "detach", False):
                st = poll_kernel_status(f"{plan.owner}/{plan.slug}", expected_version, transport)
                if st == "complete":
                    out_dir = getattr(args, "output_dir", None) or (Path("output") / f"{plan.owner}_{plan.slug}")
                    col_res = collect_outputs(
                        f"{plan.owner}/{plan.slug}",
                        expected_version,
                        transport,
                        Path(out_dir),
                        expected_spec=valid_spec,
                        expected_approval=valid_auth,
                        expected_manifest=plan.outer_manifest,
                        expected_checkpoint_sha256=expected_checkpoint_sha256,
                        expected_output_inventory=expected_output_inventory,
                        expected_envelope=expected_h_envelope,
                    )
                    if col_res.get("status") == "complete":
                        return 0
                    print(f"Output collection error: {col_res.get('reason')}", file=sys.stderr)
                    return 1
                print(f"Kernel execution terminated with status '{st}'", file=sys.stderr)
                return 1

        return 0

    plan_obj = None
    if plan_path:
        p_plan = Path(plan_path)
        if p_plan.exists():
            try:
                plan_obj = json.loads(p_plan.read_text(encoding="utf-8"))
            except Exception:
                pass

    if command == "status":
        kernel_ref = getattr(args, "kernel", None) or (f"{plan_obj.get('owner', 'amazon4u')}/{plan_obj.get('slug')}" if plan_obj and plan_obj.get('slug') else None)
        version = getattr(args, "version", None) or (context.submission_binding.get("version") if context else None) or (plan_obj.get("expected_version") if plan_obj else None)
        if not kernel_ref:
            print("Error: Explicit kernel reference (--kernel or --plan) is required", file=sys.stderr)
            return 1
        if version is None:
            print("Error: Explicit version is required; guessing version 1 is forbidden", file=sys.stderr)
            return 1
        if transport is not None:
            st = poll_kernel_status(kernel_ref, version, transport, max_polls=1)
            print(f"Status for {kernel_ref}: {st}")
            return 0 if st in VALID_STATUS_TOKENS else 1
        print("Status requires transport in offline mode.", file=sys.stderr)
        return 1

    if command == "logs":
        kernel_ref = getattr(args, "kernel", None) or (f"{plan_obj.get('owner', 'amazon4u')}/{plan_obj.get('slug')}" if plan_obj and plan_obj.get('slug') else None)
        version = getattr(args, "version", None) or (context.submission_binding.get("version") if context else None) or (plan_obj.get("expected_version") if plan_obj else None)
        if not kernel_ref:
            print("Error: Explicit kernel reference (--kernel or --plan) is required", file=sys.stderr)
            return 1
        if version is None:
            print("Error: Explicit version is required; guessing version 1 is forbidden", file=sys.stderr)
            return 1
        if transport is not None:
            log_res = consume_kernel_logs(kernel_ref, version, transport)
            if log_res["status"] == "available":
                print(log_res["logs"])
                return 0
            print(f"Log error: {log_res.get('error')}", file=sys.stderr)
            return 1
        print("Logs require transport in offline mode.", file=sys.stderr)
        return 1

    if command == "output":
        kernel_ref = getattr(args, "kernel", None) or (f"{plan_obj.get('owner', 'amazon4u')}/{plan_obj.get('slug')}" if plan_obj and plan_obj.get('slug') else None)
        version = getattr(args, "version", None) or (context.submission_binding.get("version") if context else None) or (plan_obj.get("expected_version") if plan_obj else None)
        if not kernel_ref:
            print("Error: Explicit kernel reference (--kernel or --plan) is required; default placeholders forbidden", file=sys.stderr)
            return 1
        if version is None:
            print("Error: Explicit version (--version or context or plan expected_version) is required; default version forbidden", file=sys.stderr)
            return 1
        dest = getattr(args, "output_dir", None) or Path("output")
        if transport is not None:
            if not plan_obj and not spec_path and not context:
                print("Error: Independently supplied --plan, --context, or --spec/--approval required for output collection; fabricated placeholders forbidden", file=sys.stderr)
                return 1
            spec = context.canonical_h_spec if context else (plan_obj.get("spec", {}) if plan_obj else (json.loads(Path(spec_path).read_text(encoding="utf-8")) if spec_path else None))
            auth = context.h_authorization if context else (plan_obj.get("authorization", {}) if plan_obj else (json.loads(Path(approval_path).read_text(encoding="utf-8")) if approval_path else None))
            exp_sha = getattr(args, "expected_checkpoint_sha256", None) or (
                context.expected_output_inventory.get("checkpoint_physical_sha256") if context else None
            ) or (plan_obj.get("expected_checkpoint_sha256") if plan_obj else None)
            exp_inv = context.expected_output_inventory if context else (plan_obj.get("expected_output_inventory") if plan_obj else None)
            exp_man = context.expected_o_outer if context else (plan_obj.get("expected_sources") if plan_obj else None)
            exp_env = context.expected_h_envelope if context else (plan_obj.get("expected_h_envelope") or plan_obj.get("expected_envelope") if plan_obj else None)
            if not spec or not auth:
                print("Error: Valid spec and authorization required for output collection", file=sys.stderr)
                return 1
            res = collect_outputs(
                kernel_ref,
                version,
                transport,
                Path(dest),
                expected_spec=spec,
                expected_approval=auth,
                expected_manifest=exp_man,
                expected_checkpoint_sha256=exp_sha,
                expected_output_inventory=exp_inv,
                expected_envelope=exp_env,
            )
            return 0 if res.get("status") == "complete" else 1
        print("Output collection requires transport in offline mode.", file=sys.stderr)
        return 1

    print(f"Command '{command}' not implemented in offline dispatch.", file=sys.stderr)
    return 1
