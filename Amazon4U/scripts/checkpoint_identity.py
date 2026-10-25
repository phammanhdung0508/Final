"""Amazon4U milestone I1: pure checkpoint identity and source/environment provenance contracts.

Python standard library only.
Provides:
- CheckpointError, CheckpointValidationError, CheckpointIdentityMismatchError,
  CheckpointProvenanceError, CheckpointSecurityError: strict exception hierarchy.
- CanonicalResumeIdentity: structured, collision-free canonical identity container
  retaining actual caller-supplied training records, histories, warm pool, typed graph,
  permitted product metadata, validation selection inputs, and complete resolved settings.
- build_canonical_identity: builds CanonicalResumeIdentity from mandatory ACTUAL
  caller-supplied records; validates record/history membership, condition positivity,
  warm-pool consistency, graph endpoints, permitted metadata, and strict types;
  recomputes content fingerprints.
- validate_canonical_identity: validates actual identity by recomputing all fingerprints
  from actual retained content, detecting falsified fingerprints or tampering, and
  verifying exact match against mandatory trusted caller expectations.
- compute_data_fingerprint, compute_graph_fingerprint, compute_validation_fingerprint,
  compute_settings_fingerprint: pure deterministic SHA-256 fingerprint calculators.
- build_source_provenance: builds source and environment provenance from mandatory
  allowlisted source bytes, Git evidence, and environment records; validates allowlisted
  relative paths, bounds total source bytes to 2 MiB, and produces reconstructable snapshots.
- validate_source_provenance: validates source provenance by recomputing file sizes and
  SHA-256 hashes from actual content bytes, detecting tampering, and verifying exact
  agreement against mandatory trusted expected provenance.

Trust boundary notice:
Caller acquisition of actual evidence (source bytes, Git commit/status, environment) is a
documented trust boundary; this pure contract module does not prove origin, automatically
inspect the filesystem, run Git subprocesses, scan directories, execute shell commands, or
read external projects/credentials/datasets. Secure filesystem IO and validation-only harness
integration belong to later deliverables.
"""

from collections import defaultdict
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

# Canonicalize module in sys.modules so scripts.checkpoint_identity and checkpoint_identity
# share the exact same module object and class identities.
if __name__ == "scripts.checkpoint_identity":
    sys.modules.setdefault("checkpoint_identity", sys.modules[__name__])
elif __name__ == "checkpoint_identity":
    sys.modules.setdefault("scripts.checkpoint_identity", sys.modules[__name__])
    if "scripts" in sys.modules:
        setattr(sys.modules["scripts"], "checkpoint_identity", sys.modules[__name__])

try:
    import scripts as _scripts_pkg  # type: ignore
except ModuleNotFoundError:
    _scripts_pkg = None
if _scripts_pkg is not None:
    setattr(_scripts_pkg, "checkpoint_identity", sys.modules[__name__])


# ---------------------------------------------------------------------------
# Exceptions Hierarchy
# ---------------------------------------------------------------------------

class CheckpointError(Exception):
    """Base exception for all checkpoint recovery errors."""


class CheckpointValidationError(CheckpointError):
    """Raised when canonical identity, settings, records, or graph fail structural or semantic validation."""


class CheckpointIdentityMismatchError(CheckpointValidationError):
    """Raised when canonical identity fingerprints or retained content mismatch expected identity."""


class CheckpointProvenanceError(CheckpointValidationError):
    """Raised when source or environment provenance validation fails."""


class CheckpointSecurityError(CheckpointError):
    """Raised when path traversal, forbidden path, credential pattern, or security policy violation is detected."""


# ---------------------------------------------------------------------------
# Constants & Allowlisted Boundaries
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA_VERSION = "amazon4u-checkpoint-identity-v1"
PROVENANCE_SCHEMA_VERSION = "amazon4u-provenance-v1"
MAX_PROVENANCE_SOURCE_BYTES = 2 * 1024 * 1024  # 2 MiB ceiling

NODE_TYPES: Tuple[str, ...] = ("User", "Product", "CategoryPath", "Brand")

RELATION_SCHEMAS: Dict[str, Tuple[str, str]] = {
    "rated": ("User", "Product"),
    "rated_by": ("Product", "User"),
    "belongs_to": ("Product", "CategoryPath"),
    "contains_product": ("CategoryPath", "Product"),
    "has_brand": ("Product", "Brand"),
    "brand_of": ("Brand", "Product"),
    "category_child_of": ("CategoryPath", "CategoryPath"),
    "category_parent_of": ("CategoryPath", "CategoryPath"),
}

ALLOWLISTED_SOURCE_FILES: Tuple[str, ...] = (
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
)

ALLOWLISTED_CONFIG_FILES: Tuple[str, ...] = (
    "configs/compute-plan.json",
    "configs/dataset-scope.json",
    "configs/evaluation-metrics.json",
    "configs/kg-v1.json",
    "configs/model-suite.json",
    "configs/protocol-lifecycle.json",
    "configs/reproducibility.json",
    "configs/training-semantics.json",
)

ALL_ALLOWLISTED_PATHS: Set[str] = set(ALLOWLISTED_SOURCE_FILES) | set(ALLOWLISTED_CONFIG_FILES)
APPROVED_TOP_LEVEL_DIRS: Tuple[str, ...] = ("scripts", "configs")

DISALLOWED_PATH_SUBSTRINGS: Tuple[str, ...] = (
    "data/",
    ".git/",
    ".venv/",
    ".test-output/",
    "Movie4U",
    "multiautoresearch",
    ".worktrees",
)

CREDENTIAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r".*\.key$", re.IGNORECASE),
    re.compile(r".*\.pem$", re.IGNORECASE),
    re.compile(r".*token.*", re.IGNORECASE),
    re.compile(r".*secret.*", re.IGNORECASE),
    re.compile(r".*\.env.*", re.IGNORECASE),
    re.compile(r".*credential.*", re.IGNORECASE),
    re.compile(r".*password.*", re.IGNORECASE),
)

EXCLUDED_METADATA_FIELDS: Tuple[str, ...] = (
    "average_rating",
    "rating_number",
    "bestseller_rank",
    "store",
    "Manufacturer",
    "bought_together",
    "details",
)

REQUIRED_SETTINGS_SECTIONS: Tuple[str, ...] = (
    "protocol",
    "execution",
    "model",
    "training",
    "sampling",
    "determinism",
)

REQUIRED_SETTINGS_FIELDS: Dict[str, Tuple[str, ...]] = {
    "protocol": (
        "candidate_policy",
        "lifecycle_state",
        "protocol_version",
    ),
    "execution": (
        "cpu_threads",
        "execution_device",
        "execution_label",
        "runtime_mode",
        "runtime_workers",
        "worker_settings",
    ),
    "model": (
        "architecture",
        "embedding_dim",
        "initialization",
        "num_layers",
    ),
    "training": (
        "adam_betas",
        "adam_epsilon",
        "batch_size",
        "endpoint_regularization",
        "learning_rate",
        "loss_function",
        "max_epochs",
        "optimizer",
        "patience",
        "validation_frequency",
        "weight_decay",
    ),
    "sampling": (
        "negative_strategy",
        "negatives_per_positive",
        "neighbor_fanouts",
        "pool_policy",
    ),
    "determinism": (
        "deterministic_algorithms",
        "reproducibility_version",
        "rng_streams",
        "seed",
        "worker_derivation",
    ),
}

CONFIGURED_RNG_COMPONENTS: Tuple[str, ...] = (
    "Python",
    "NumPy",
    "PyTorch_CPU",
    "PyTorch_CUDA_if_applicable",
    "initialization",
    "epoch_shuffle",
    "negative_sampler",
    "neighbor_sampler",
    "DataLoader_generator",
    "DataLoader_workers",
)

RNG_STREAM_RECORD_KEYS: Set[str] = {
    "algorithm",
    "applicable",
    "derivation",
    "name",
    "reason",
    "seed",
}

WORKER_DERIVATION_KEYS: Set[str] = {
    "algorithm",
    "applicable",
    "base_seed",
    "derivation",
    "reason",
}

WORKER_SETTINGS_KEYS: Set[str] = {
    "multiprocessing_context",
    "persistent_workers",
    "prefetch_factor",
    "worker_init_function",
}

PROVENANCE_TOP_LEVEL_KEYS: Set[str] = {
    "schema_version",
    "total_source_bytes",
    "source_tree_hash",
    "source_files",
    "git",
    "environment",
}

GIT_EVIDENCE_KEYS: Set[str] = {"commit", "dirty", "tracked_diff", "untracked_sources"}

ENVIRONMENT_RECORD_KEYS: Set[str] = {
    "python_version",
    "platform_system",
    "platform_release",
    "platform_machine",
    "cpu_count",
    "torch_version",
    "library_versions",
}

FILE_ENTRY_KEYS: Set[str] = {"path", "size_bytes", "sha256", "content_bytes"}


# ---------------------------------------------------------------------------
# Safe Representation & Canonical JSON Serialization Helpers
# ---------------------------------------------------------------------------

def _safe_repr(val: Any) -> str:
    """Format a value for diagnostic messages without leaking ValueError on huge integers."""
    if isinstance(val, int) and not isinstance(val, bool):
        # Python 3.11+ limits integer string conversion to 4300 digits (~14284 bits).
        if val.bit_length() > 14000:
            return f"<oversized int ({val.bit_length()} bits)>"
    try:
        r = repr(val)
        if len(r) > 200:
            return r[:197] + "..."
        return r
    except Exception:
        return f"<{type(val).__name__} repr failed>"


_SURROGATES = re.compile("[\\ud800-\\udfff]")


def _validate_clean_text(val: Any, name: str, allow_empty: bool = False) -> str:
    """Validate string type, absence of lone surrogates/invalid Unicode, and whitespace padding."""
    if not isinstance(val, str):
        raise CheckpointValidationError(f"{name} must be a string, got {type(val).__name__}: {_safe_repr(val)}")
    try:
        val.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointValidationError(f"{name} contains invalid Unicode / lone surrogates: {exc}") from exc
    # Readiness C2: one regex search replaces the per-character loop for exact str. A str
    # subclass may override iteration, so it keeps the loop.
    if (_SURROGATES.search(val) is not None if type(val) is str
            else any(0xD800 <= ord(c) <= 0xDFFF for c in val)):
        raise CheckpointValidationError(f"{name} contains lone surrogates: {_safe_repr(val)}")
    if not allow_empty:
        if not val.strip() or val.strip() != val:
            raise CheckpointValidationError(f"{name} must not be empty or have leading/trailing whitespace: {_safe_repr(val)}")
    return val


def _canonical_json_bytes(payload: Any, role: str = "canonical encoding") -> bytes:
    """Encode payload to compact canonical JSON UTF-8 bytes, translating serialization errors to CheckpointValidationError."""
    try:
        json_str = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return json_str.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointValidationError(f"Invalid Unicode / lone surrogates in {role}: {exc}") from exc
    except (ValueError, OverflowError) as exc:
        raise CheckpointValidationError(f"Numeric or serialization conversion failure in {role}: {exc}") from exc


# ---------------------------------------------------------------------------
# Metadata and Identifier Validators
# ---------------------------------------------------------------------------

def validate_category_id(cat_id: str) -> List[str]:
    """Validate canonical KG-v1 CategoryPath identity: 'c:' followed by compact JSON array.

    Rejects non-strings, leading/trailing whitespace, missing 'c:' prefix, invalid JSON,
    empty arrays, non-string labels, and leading/trailing whitespace inside labels.
    """
    _validate_clean_text(cat_id, "CategoryPath ID")
    if not cat_id.startswith("c:"):
        raise CheckpointValidationError(f"CategoryPath ID must start with 'c:', got {_safe_repr(cat_id)}")

    payload = cat_id[2:]
    try:
        parsed = json.loads(payload)
    except Exception as exc:
        raise CheckpointValidationError(f"Invalid JSON payload in CategoryPath ID {_safe_repr(cat_id)}: {exc}") from exc

    if not isinstance(parsed, list):
        raise CheckpointValidationError(f"CategoryPath ID JSON payload must be a list, got {type(parsed).__name__}: {_safe_repr(cat_id)}")
    if len(parsed) == 0:
        raise CheckpointValidationError(f"CategoryPath ID must have at least one category element: {_safe_repr(cat_id)}")

    for idx, elem in enumerate(parsed):
        _validate_clean_text(elem, f"CategoryPath element at index {idx}")

    try:
        raw_repr = json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
        raw_repr.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointValidationError(f"CategoryPath ID formatting failed due to invalid Unicode: {exc}") from exc

    canonical_repr = f"c:{raw_repr}"
    if cat_id != canonical_repr:
        raise CheckpointValidationError(
            f"CategoryPath ID is not in canonical compact JSON representation: {_safe_repr(cat_id)} != {_safe_repr(canonical_repr)}"
        )

    return parsed


def format_canonical_category_id(path_elements: Sequence[str]) -> str:
    """Format a sequence of category path labels into canonical 'c:[...]' string."""
    if not isinstance(path_elements, (list, tuple)):
        raise CheckpointValidationError("path_elements must be a sequence of strings.")
    if len(path_elements) == 0:
        raise CheckpointValidationError("path_elements cannot be empty.")
    for idx, elem in enumerate(path_elements):
        _validate_clean_text(elem, f"Category path element at index {idx}")
    try:
        raw_repr = json.dumps(list(path_elements), separators=(',', ':'), ensure_ascii=False)
        raw_repr.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointValidationError(f"Category formatting failed due to invalid Unicode: {exc}") from exc
    return f"c:{raw_repr}"


def validate_brand_id(brand_id: str) -> str:
    """Validate canonical KG-v1 Brand identity: 'b:' followed by a non-blank string label.

    Rejects non-strings, leading/trailing whitespace, missing 'b:' prefix, empty labels,
    and malformed Unicode / lone surrogates.
    """
    _validate_clean_text(brand_id, "Brand ID")
    if not brand_id.startswith("b:"):
        raise CheckpointValidationError(f"Brand ID must start with 'b:', got {_safe_repr(brand_id)}")

    label = brand_id[2:]
    _validate_clean_text(label, "Brand ID label")
    return label


def format_canonical_brand_id(brand_label: str) -> str:
    """Format a raw brand label into canonical 'b:...' string."""
    _validate_clean_text(brand_label, "Brand label")
    return f"b:{brand_label}"


def validate_allowlisted_relative_path(rel_path: str) -> str:
    """Strictly validate and normalize an allowlisted relative script or config path.

    Enforces:
    - Non-empty relative path without leading/trailing whitespace.
    - No absolute paths.
    - No path traversal components ('..', '.', empty parts).
    - Must reside in approved top-level directories ('scripts/' or 'configs/').
    - Rejects disallowed substrings (data/, .git/, .venv/, etc.).
    - Rejects credential file patterns (*.key, *.pem, *token*, *secret*, etc.).
    - Must belong to the explicit registered allowlist.
    """
    if not isinstance(rel_path, str):
        raise CheckpointSecurityError(f"Path must be a string, got {type(rel_path).__name__}: {rel_path!r}")
    if not rel_path.strip() or rel_path.strip() != rel_path:
        raise CheckpointSecurityError(f"Path must not have leading or trailing whitespace: {rel_path!r}")

    p = Path(rel_path)
    if p.is_absolute() or rel_path.startswith("/") or rel_path.startswith("\\"):
        raise CheckpointSecurityError(f"Absolute paths are forbidden: {rel_path!r}")

    raw_parts = rel_path.replace("\\", "/").split("/")
    for part in raw_parts:
        if part in ("..", ".", ""):
            raise CheckpointSecurityError(f"Path traversal component {part!r} forbidden: {rel_path!r}")

    parts = p.parts
    if not parts or parts[0] not in APPROVED_TOP_LEVEL_DIRS:
        raise CheckpointSecurityError(
            f"Path must reside in approved directories {APPROVED_TOP_LEVEL_DIRS}, got: {rel_path!r}"
        )

    if "\\\\" in rel_path or "//" in rel_path:
        raise CheckpointSecurityError(f"Malformed slashes in path: {rel_path!r}")

    for sub in DISALLOWED_PATH_SUBSTRINGS:
        if sub in rel_path:
            raise CheckpointSecurityError(f"Path contains forbidden substring {sub!r}: {rel_path!r}")

    for pattern in CREDENTIAL_PATTERNS:
        if pattern.match(rel_path):
            raise CheckpointSecurityError(f"Potential credential path in allowlist: {rel_path!r}")

    normalized = str(Path(rel_path).as_posix())
    if normalized not in ALL_ALLOWLISTED_PATHS:
        raise CheckpointSecurityError(f"Path is not in allowlisted source/config set: {rel_path!r}")

    return normalized


# ---------------------------------------------------------------------------
# Common Header & Fingerprint Validation
# ---------------------------------------------------------------------------

def _validate_identity_header(
    method: Any,
    category: Any,
    feedback_condition: Any,
    seed: Any,
    fingerprints: Optional[Mapping[str, Any]] = None,
    role: str = "identity",
) -> None:
    """Validate common header fields and strict SHA-256 fingerprints across constructors, builders, and validators."""
    _validate_clean_text(method, f"Invalid {role}.method")
    _validate_clean_text(category, f"Invalid {role}.category")
    if feedback_condition not in ("all_ratings", "rating_ge_4"):
        raise CheckpointValidationError(
            f"Invalid {role}.feedback_condition: {_safe_repr(feedback_condition)}. Must be strictly 'all_ratings' or 'rating_ge_4'."
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CheckpointValidationError(
            f"Invalid {role}.seed: {_safe_repr(seed)}. Must be a non-negative integer and not boolean."
        )

    if fingerprints is not None:
        for name in (
            "data_fingerprint",
            "graph_fingerprint",
            "validation_fingerprint",
            "settings_fingerprint",
            "identity_hash",
        ):
            if name not in fingerprints:
                raise CheckpointValidationError(f"Missing {name} in {role}.")
            fp = fingerprints[name]
            if (
                not isinstance(fp, str)
                or len(fp) != 64
                or not all(c in "0123456789abcdef" for c in fp)
            ):
                raise CheckpointValidationError(
                    f"Invalid SHA-256 fingerprint for {role}.{name}: {_safe_repr(fp)}. Must be 64-character lowercase hexadecimal without padding."
                )


# ---------------------------------------------------------------------------
# Fingerprint Computation Functions
# ---------------------------------------------------------------------------

def compute_data_fingerprint(
    feedback_condition: str,
    users: Sequence[str],
    training_records: Sequence[Any],
    positive_history: Mapping[str, Sequence[str]],
    all_rated_history: Mapping[str, Sequence[str]],
    warm_items: Sequence[str],
) -> str:
    """Compute deterministic SHA-256 over training records, histories, and warm pool.

    Enforces:
    - feedback_condition strictly 'all_ratings' or 'rating_ge_4'.
    - Complete user and warm-item deduplication and non-empty validation.
    - Strict record format: user_id, parent_asin, finite rating (not bool), exact integer timestamp (not bool/float).
    - Duplicate (user_id, parent_asin) pairs rejected.
    - Exact history membership: positive history subset of all_rated_history.
    - Condition positivity consistency:
      all_ratings: positive_history equals all_rated_history.
      rating_ge_4: item is positive if and only if record rating >= 4.0.
    - Warm pool consistency:
      all_ratings: warm_items equals union of observed items.
      rating_ge_4: warm_items equals union of positive items.
    - Structured canonical JSON representation without delimiter collisions.
    - Rejection of same-count mutations across records, ratings, timestamps, and warm pool.
    """
    if feedback_condition not in ("all_ratings", "rating_ge_4"):
        raise CheckpointValidationError(f"Invalid feedback condition: {feedback_condition!r}")

    if not isinstance(users, (list, tuple)):
        raise CheckpointValidationError(f"users must be a list or tuple, got {type(users).__name__}")
    if len(users) == 0:
        raise CheckpointValidationError("users sequence cannot be empty.")
    seen_users: Set[str] = set()
    for u in users:
        _validate_clean_text(u, "user ID in users")
        if u in seen_users:
            raise CheckpointValidationError(f"Duplicate user ID in users sequence: {_safe_repr(u)}")
        seen_users.add(u)
    sorted_users = sorted(seen_users)

    if not isinstance(warm_items, (list, tuple)):
        raise CheckpointValidationError(f"warm_items must be a list or tuple, got {type(warm_items).__name__}")
    if len(warm_items) == 0:
        raise CheckpointValidationError("warm_items sequence cannot be empty.")
    seen_warm: Set[str] = set()
    for item in warm_items:
        _validate_clean_text(item, "item ID in warm_items")
        if item in seen_warm:
            raise CheckpointValidationError(f"Duplicate item ID in warm_items sequence: {_safe_repr(item)}")
        seen_warm.add(item)
    sorted_warm = sorted(seen_warm)

    if not isinstance(positive_history, (dict, Mapping)):
        raise CheckpointValidationError("positive_history must be a mapping.")
    if not isinstance(all_rated_history, (dict, Mapping)):
        raise CheckpointValidationError("all_rated_history must be a mapping.")

    # Validate keys FIRST before doing set operations or sorted()
    for k in positive_history.keys():
        _validate_clean_text(k, "user ID key in positive_history")
    for k in all_rated_history.keys():
        _validate_clean_text(k, "user ID key in all_rated_history")

    if set(positive_history.keys()) != seen_users:
        extra = set(positive_history.keys()) - seen_users
        missing = seen_users - set(positive_history.keys())
        raise CheckpointValidationError(f"positive_history keys mismatch. Extra: {sorted(extra)}, Missing: {sorted(missing)}")

    if set(all_rated_history.keys()) != seen_users:
        extra = set(all_rated_history.keys()) - seen_users
        missing = seen_users - set(all_rated_history.keys())
        raise CheckpointValidationError(f"all_rated_history keys mismatch. Extra: {sorted(extra)}, Missing: {sorted(missing)}")

    if not isinstance(training_records, (list, tuple)):
        raise CheckpointValidationError("training_records must be a sequence.")
    if len(training_records) == 0:
        raise CheckpointValidationError("training_records cannot be empty.")

    seen_pairs: Set[Tuple[str, str]] = set()
    user_record_items: Dict[str, Set[str]] = defaultdict(set)
    user_pos_record_items: Dict[str, Set[str]] = defaultdict(set)
    norm_records: List[Dict[str, Any]] = []

    for idx, r in enumerate(training_records):
        if isinstance(r, dict):
            allowed_record_keys = {"user_id", "parent_asin", "rating", "timestamp"}
            for rk in r.keys():
                if not isinstance(rk, str):
                    raise CheckpointValidationError(f"Record dict keys at index {idx} must be strings, got {type(rk).__name__}: {_safe_repr(rk)}")
            if set(r.keys()) != allowed_record_keys:
                extra_keys = set(r.keys()) - allowed_record_keys
                missing_keys = allowed_record_keys - set(r.keys())
                raise CheckpointValidationError(
                    f"Invalid keys in training record at index {idx}. Missing: {sorted(missing_keys)}, Extra: {sorted(extra_keys)}"
                )
            u_id = r["user_id"]
            p_id = r["parent_asin"]
            rating = r["rating"]
            ts = r["timestamp"]
        elif isinstance(r, (list, tuple)):
            if len(r) != 4:
                raise CheckpointValidationError(
                    f"Invalid training record at index {idx}: expected exact 4-tuple (user_id, parent_asin, rating, timestamp), got {len(r)} elements: {_safe_repr(r)}"
                )
            u_id, p_id, rating, ts = r[0], r[1], r[2], r[3]
        else:
            raise CheckpointValidationError(f"Invalid training record at index {idx}: {_safe_repr(r)}")

        _validate_clean_text(u_id, f"Invalid user_id in training record at index {idx}")
        _validate_clean_text(p_id, f"Invalid parent_asin in training record at index {idx}")

        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            raise CheckpointValidationError(
                f"Invalid rating in training record at index {idx}: {_safe_repr(rating)}. Must be a finite number and not boolean."
            )
        try:
            frating = float(rating)
        except (OverflowError, ValueError):
            raise CheckpointValidationError(f"Numeric overflow in rating at index {idx}: {_safe_repr(rating)}")
        if not math.isfinite(frating):
            raise CheckpointValidationError(
                f"Invalid rating in training record at index {idx}: {_safe_repr(rating)}. Must be a finite number and not boolean."
            )

        if isinstance(ts, bool) or not isinstance(ts, int):
            raise CheckpointValidationError(
                f"Invalid timestamp in training record at index {idx}: {_safe_repr(ts)}. Timestamps must be exact integers (not float, bool, or non-finite)."
            )

        pair = (u_id, p_id)
        if pair in seen_pairs:
            raise CheckpointValidationError(f"Duplicate training record pair detected at index {idx}: {pair}")
        seen_pairs.add(pair)

        if u_id not in seen_users:
            raise CheckpointValidationError(f"Training record user {u_id!r} not present in declared users sequence.")

        user_record_items[u_id].add(p_id)
        if feedback_condition == "all_ratings" or frating >= 4.0:
            user_pos_record_items[u_id].add(p_id)

        norm_records.append({
            "user": u_id,
            "item": p_id,
            "rating": frating,
            "timestamp": int(ts),
        })

    for u in sorted_users:
        all_hist_items = all_rated_history[u]
        if not isinstance(all_hist_items, (list, tuple, set, frozenset)):
            raise CheckpointValidationError(f"all_rated_history[{u!r}] must be a collection of item strings.")
        all_hist_list = list(all_hist_items)
        seen_all_items: Set[str] = set()
        for item in all_hist_list:
            _validate_clean_text(item, f"Invalid item ID in all_rated_history for user {u!r}")
            if item in seen_all_items:
                raise CheckpointValidationError(f"Duplicate items in all_rated_history for user {u!r}: {_safe_repr(item)}")
            seen_all_items.add(item)
        all_hist_set = seen_all_items

        rec_items = user_record_items.get(u, set())
        if all_hist_set != rec_items:
            raise CheckpointValidationError(
                f"all_rated_history for user {u!r} does not match training records. Diff: {all_hist_set ^ rec_items}"
            )

        pos_hist_items = positive_history[u]
        if not isinstance(pos_hist_items, (list, tuple, set, frozenset)):
            raise CheckpointValidationError(f"positive_history[{u!r}] must be a collection of item strings.")
        pos_hist_list = list(pos_hist_items)
        seen_pos_items: Set[str] = set()
        for item in pos_hist_list:
            _validate_clean_text(item, f"Invalid item ID in positive_history for user {u!r}")
            if item in seen_pos_items:
                raise CheckpointValidationError(f"Duplicate items in positive_history for user {u!r}: {_safe_repr(item)}")
            seen_pos_items.add(item)
        pos_hist_set = seen_pos_items

        if not pos_hist_set.issubset(all_hist_set):
            raise CheckpointValidationError(
                f"Positive history for user {u!r} contains items not in all_rated_history: {pos_hist_set - all_hist_set}"
            )

        expected_pos_set = user_pos_record_items.get(u, set())
        if pos_hist_set != expected_pos_set:
            raise CheckpointValidationError(
                f"positive_history for user {u!r} inconsistent with condition {feedback_condition!r}. "
                f"Expected {sorted(expected_pos_set)}, got {sorted(pos_hist_set)}"
            )

    all_observed_items: Set[str] = {r["item"] for r in norm_records}
    all_positive_items: Set[str] = {r["item"] for r in norm_records if feedback_condition == "all_ratings" or r["rating"] >= 4.0}

    if feedback_condition == "all_ratings":
        if seen_warm != all_observed_items:
            raise CheckpointValidationError(
                f"In 'all_ratings', warm_items must equal all observed items. Diff: {seen_warm ^ all_observed_items}"
            )
    else:
        if seen_warm != all_positive_items:
            raise CheckpointValidationError(
                f"In 'rating_ge_4', warm_items must equal condition-positive items. Diff: {seen_warm ^ all_positive_items}"
            )

    norm_records.sort(key=lambda r: (r["user"], r["item"], r["rating"], r["timestamp"]))

    canonical_positive = [
        {"user": u, "items": sorted(set(positive_history[u]))}
        for u in sorted_users
    ]
    canonical_all_rated = [
        {"user": u, "items": sorted(set(all_rated_history[u]))}
        for u in sorted_users
    ]

    payload: Dict[str, Any] = {
        "schema": "amazon4u-canonical-data-v1",
        "condition": feedback_condition,
        "users": sorted_users,
        "warm_items": sorted_warm,
        "positive_history": canonical_positive,
        "all_rated_history": canonical_all_rated,
        "training_records": norm_records,
    }

    canon_bytes = _canonical_json_bytes(payload, role="data fingerprint")
    return hashlib.sha256(canon_bytes).hexdigest()


def compute_graph_fingerprint(
    graph_nodes: Mapping[str, Sequence[str]],
    graph_edges: Sequence[Any],
    product_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    """Compute deterministic SHA-256 over typed graph nodes, relations, edges, and permitted metadata.

    Enforces:
    - Declared node types in NODE_TYPES ("User", "Product", "CategoryPath", "Brand").
    - CategoryPath nodes validated via validate_category_id ('c:[...]').
    - Brand nodes validated via validate_brand_id ('b:...').
    - Non-empty node IDs, no duplicate node IDs per type.
    - Edges 6-tuples (src_type, src_id, relation, dst_type, dst_id, weight).
    - Relation schemas matching RELATION_SCHEMAS.
    - Edge endpoints existing in declared node sets.
    - Finite float edge weights without lossy rounding or delimiter collisions.
    - No duplicate edges.
    - Product metadata is MANDATORY, complete for all declared Product nodes, and canonical.
    - Permitted product metadata: canonical CategoryPath IDs array and explicit canonical Brand ID ('b:...') or None.
    - Rejection of unknown/excluded fields (average_rating, rating_number, store, Manufacturer, etc.).
    """
    if not isinstance(graph_nodes, (dict, Mapping)) or not graph_nodes:
        raise CheckpointValidationError("graph_nodes must be a non-empty mapping.")

    # Validate all graph_nodes keys FIRST before sorting
    for ntype in graph_nodes.keys():
        if not isinstance(ntype, str):
            raise CheckpointValidationError(f"graph_nodes keys must be strings, got {type(ntype).__name__}: {_safe_repr(ntype)}")
        if ntype not in NODE_TYPES:
            raise CheckpointValidationError(f"Invalid node type in graph_nodes: {ntype!r}. Must be one of {NODE_TYPES}")

    node_sets: Dict[str, Set[str]] = {}
    # Readiness C2: validate_category_id depends only on the value of an exact str, so a
    # value that passed once in this call is not parsed again.
    valid_categories: Set[str] = set()
    for ntype in sorted(graph_nodes.keys()):
        nids = graph_nodes[ntype]
        if not isinstance(nids, (list, tuple, set, frozenset)):
            raise CheckpointValidationError(f"graph_nodes[{ntype!r}] must be a sequence of node IDs.")
        nid_list = list(nids)
        seen_in_type: Set[str] = set()
        for nid in nid_list:
            _validate_clean_text(nid, f"Invalid node ID in type {ntype!r}")
            if nid in seen_in_type:
                raise CheckpointValidationError(f"Duplicate node ID in type {ntype!r}: {_safe_repr(nid)}")
            if ntype == "CategoryPath":
                validate_category_id(nid)
                if type(nid) is str:
                    valid_categories.add(nid)
            elif ntype == "Brand":
                validate_brand_id(nid)
            seen_in_type.add(nid)
        node_sets[ntype] = seen_in_type

    if not isinstance(graph_edges, (list, tuple)):
        raise CheckpointValidationError("graph_edges must be a sequence of 6-tuples or edge dicts.")

    seen_edges: Set[Tuple[str, str, str, str, str]] = set()
    canon_edges: List[Dict[str, Any]] = []

    for idx, edge in enumerate(graph_edges):
        if isinstance(edge, (list, tuple)):
            if len(edge) != 6:
                raise CheckpointValidationError(
                    f"Invalid edge at index {idx} (expected exact 6-tuple src_type, src_id, relation, dst_type, dst_id, weight): {_safe_repr(edge)}"
                )
            src_type, src_id, relation, dst_type, dst_id, weight = edge[0], edge[1], edge[2], edge[3], edge[4], edge[5]
        elif isinstance(edge, dict):
            for ek in edge.keys():
                if not isinstance(ek, str):
                    raise CheckpointValidationError(f"Edge dict keys at index {idx} must be strings, got {type(ek).__name__}: {_safe_repr(ek)}")
            allowed_edge_keys = {"src_type", "src_id", "relation", "dst_type", "dst_id", "weight"}
            if set(edge.keys()) != allowed_edge_keys:
                raise CheckpointValidationError(
                    f"Invalid keys in edge dict at index {idx}: missing={sorted(allowed_edge_keys - set(edge.keys()))}, extra={sorted(set(edge.keys()) - allowed_edge_keys)}"
                )
            src_type = edge["src_type"]
            src_id = edge["src_id"]
            relation = edge["relation"]
            dst_type = edge["dst_type"]
            dst_id = edge["dst_id"]
            weight = edge["weight"]
        else:
            raise CheckpointValidationError(f"Invalid edge at index {idx}: {_safe_repr(edge)}")

        for val, name in (
            (src_type, "src_type"),
            (src_id, "src_id"),
            (relation, "relation"),
            (dst_type, "dst_type"),
            (dst_id, "dst_id"),
        ):
            _validate_clean_text(val, f"Invalid {name} in edge at index {idx}")

        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise CheckpointValidationError(f"Invalid edge weight at index {idx}: {_safe_repr(weight)}. Must be a finite number and not boolean.")
        try:
            fweight = float(weight)
        except (OverflowError, ValueError):
            raise CheckpointValidationError(f"Numeric overflow in edge weight at index {idx}: {_safe_repr(weight)}")
        if not math.isfinite(fweight):
            raise CheckpointValidationError(f"Invalid edge weight at index {idx}: {_safe_repr(weight)}. Must be a finite number and not boolean.")

        if relation not in RELATION_SCHEMAS:
            raise CheckpointValidationError(f"Unknown relation {relation!r} in edge at index {idx}. Must be in {list(RELATION_SCHEMAS.keys())}")

        expected_src_type, expected_dst_type = RELATION_SCHEMAS[relation]
        if src_type != expected_src_type or dst_type != expected_dst_type:
            raise CheckpointValidationError(
                f"Relation schema mismatch in edge at index {idx}: relation {relation!r} requires "
                f"({expected_src_type} -> {expected_dst_type}), got ({src_type} -> {dst_type})"
            )

        if src_type not in node_sets or src_id not in node_sets[src_type]:
            raise CheckpointValidationError(f"Edge source endpoint {src_type}:{src_id} at index {idx} not found in declared graph_nodes.")
        if dst_type not in node_sets or dst_id not in node_sets[dst_type]:
            raise CheckpointValidationError(f"Edge destination endpoint {dst_type}:{dst_id} at index {idx} not found in declared graph_nodes.")

        edge_key = (src_type, src_id, relation, dst_type, dst_id)
        if edge_key in seen_edges:
            raise CheckpointValidationError(f"Duplicate graph edge detected at index {idx}: {edge_key}")
        seen_edges.add(edge_key)

        canon_edges.append({
            "src_type": src_type,
            "src_id": src_id,
            "relation": relation,
            "dst_type": dst_type,
            "dst_id": dst_id,
            "weight": fweight,
        })

    canon_edges.sort(key=lambda e: (e["src_type"], e["src_id"], e["relation"], e["dst_type"], e["dst_id"], e["weight"]))

    # Product metadata validation (MANDATORY, complete, canonical)
    if product_metadata is None or not isinstance(product_metadata, (dict, Mapping)):
        raise CheckpointValidationError("product_metadata is mandatory and must be a mapping.")

    for pid in product_metadata.keys():
        _validate_clean_text(pid, "product ID key in product_metadata")

    product_nodes = node_sets.get("Product", set())
    if set(product_metadata.keys()) != product_nodes:
        missing_p = product_nodes - set(product_metadata.keys())
        extra_p = set(product_metadata.keys()) - product_nodes
        raise CheckpointValidationError(
            f"product_metadata keys must match declared Product nodes exactly. Missing: {sorted(missing_p)}, Extra: {sorted(extra_p)}"
        )

    canon_metadata: List[Dict[str, Any]] = []
    allowed_meta_keys = {"categories", "brand"}

    for pid in sorted(product_metadata.keys()):
        entry = product_metadata[pid]
        if not isinstance(entry, (dict, Mapping)):
            raise CheckpointValidationError(f"product_metadata[{pid!r}] must be a mapping.")

        for k in entry.keys():
            if not isinstance(k, str):
                raise CheckpointValidationError(f"product_metadata[{pid!r}] keys must be strings, got {type(k).__name__}: {_safe_repr(k)}")
            if k in EXCLUDED_METADATA_FIELDS:
                raise CheckpointValidationError(
                    f"Excluded metadata field {k!r} detected in product_metadata for item {pid!r}. "
                    "Amazon4U protocol excludes review-derived, aggregate, and non-allowlisted metadata."
                )
            if k not in allowed_meta_keys:
                raise CheckpointValidationError(f"Unknown or non-allowlisted metadata field {k!r} for item {pid!r}")

        if not allowed_meta_keys.issubset(set(entry.keys())):
            missing_keys = allowed_meta_keys - set(entry.keys())
            raise CheckpointValidationError(f"product_metadata[{pid!r}] missing explicit keys {sorted(missing_keys)}")

        cats = entry["categories"]
        if cats is None:
            raise CheckpointValidationError(f"categories for item {pid!r} cannot be None; use empty list for absence.")
        if isinstance(cats, (str, bytes)):
            raise CheckpointValidationError(f"categories for item {pid!r} must be a sequence of categories, not a string.")
        if not isinstance(cats, (list, tuple, set, frozenset)):
            raise CheckpointValidationError(f"categories for item {pid!r} must be a sequence.")

        norm_cats: List[str] = []
        seen_item_cats: Set[str] = set()
        for c in cats:
            if not isinstance(c, str):
                raise CheckpointValidationError(
                    f"Category in product_metadata for item {pid!r} must be canonical CategoryPath ID string ('c:[...]'), got {type(c).__name__}: {_safe_repr(c)}"
                )
            if type(c) is not str or c not in valid_categories:
                validate_category_id(c)
                if type(c) is str:
                    valid_categories.add(c)
            if c in seen_item_cats:
                raise CheckpointValidationError(f"Duplicate category {c!r} for item {pid!r}")
            seen_item_cats.add(c)
            norm_cats.append(c)
        norm_cats.sort()

        brand = entry["brand"]
        norm_brand: Optional[str] = None
        if brand is not None:
            if not isinstance(brand, str):
                raise CheckpointValidationError(f"Brand for item {pid!r} must be string or None, got {type(brand).__name__}: {_safe_repr(brand)}")
            validate_brand_id(brand)
            norm_brand = brand

        canon_metadata.append({
            "product_id": pid,
            "categories": norm_cats,
            "brand": norm_brand,
        })

    canon_nodes = [
        {"type": ntype, "ids": sorted(node_sets[ntype])}
        for ntype in sorted(node_sets.keys())
    ]

    payload: Dict[str, Any] = {
        "schema": "amazon4u-canonical-graph-v1",
        "nodes": canon_nodes,
        "edges": canon_edges,
        "metadata": canon_metadata,
    }

    canon_bytes = _canonical_json_bytes(payload, role="graph fingerprint")
    return hashlib.sha256(canon_bytes).hexdigest()


def compute_validation_fingerprint(validation_inputs: Mapping[str, Any]) -> str:
    """Compute deterministic SHA-256 over validation selection targets and records.

    Enforces:
    - validation_inputs must be a non-empty mapping with 'targets' and/or 'records'.
    - Rejects unknown fields.
    - Records must be exact 4-tuples or dicts with exact 4 keys: user_id, parent_asin, rating, timestamp.
    - No silent defaulting of missing rating/timestamp; no > 4 element tuples.
    - Strict types: finite numbers for ratings (not bool), exact ints for timestamps (not bool/float).
    - Rejects duplicate target pairs or non-finite ratings.
    - Strict agreement: when both 'targets' and 'records' are supplied, target items and record items must match exactly.
    - Deterministic canonical JSON sorting without delimiter collisions.
    """
    if not isinstance(validation_inputs, (dict, Mapping)) or not validation_inputs:
        raise CheckpointValidationError("validation_inputs must be a non-empty mapping.")

    for k in validation_inputs.keys():
        if not isinstance(k, str):
            raise CheckpointValidationError(f"validation_inputs keys must be strings, got {type(k).__name__}: {_safe_repr(k)}")

    allowed_keys = {"targets", "records"}
    if not set(validation_inputs.keys()).issubset(allowed_keys):
        unknown = set(validation_inputs.keys()) - allowed_keys
        raise CheckpointValidationError(f"Unknown keys in validation_inputs: {sorted(unknown)}")

    targets = validation_inputs.get("targets")
    records = validation_inputs.get("records")

    if targets is None and records is None:
        raise CheckpointValidationError("validation_inputs must specify at least one of 'targets' or 'records'.")

    target_user_items: Dict[str, Set[str]] = {}
    canon_targets: List[Dict[str, Any]] = []
    if targets is not None:
        if not isinstance(targets, (dict, Mapping)):
            raise CheckpointValidationError("validation targets must be a mapping of user_id to target item IDs.")
        for u in targets.keys():
            _validate_clean_text(u, "Invalid validation target user ID")
        for u in sorted(targets.keys()):
            t_items = targets[u]
            if not isinstance(t_items, (list, tuple, set, frozenset)):
                raise CheckpointValidationError(f"validation targets for user {u!r} must be a sequence of item strings.")
            t_list = list(t_items)
            seen_t: Set[str] = set()
            for item in t_list:
                _validate_clean_text(item, f"Invalid target item ID for user {u!r}")
                if item in seen_t:
                    raise CheckpointValidationError(f"Duplicate target item {_safe_repr(item)} for validation user {u!r}.")
                seen_t.add(item)
            target_user_items[u] = seen_t
            canon_targets.append({
                "user": u,
                "target_items": sorted(seen_t),
            })

    record_user_items: Dict[str, Set[str]] = defaultdict(set)
    canon_records: List[Dict[str, Any]] = []
    if records is not None:
        if not isinstance(records, (list, tuple)):
            raise CheckpointValidationError("validation records must be a sequence.")
        seen_v_pairs: Set[Tuple[str, str]] = set()
        for idx, r in enumerate(records):
            if isinstance(r, dict):
                for rk in r.keys():
                    if not isinstance(rk, str):
                        raise CheckpointValidationError(f"Validation record dict keys at index {idx} must be strings, got {type(rk).__name__}: {_safe_repr(rk)}")
                allowed_v_keys = {"user_id", "parent_asin", "rating", "timestamp"}
                if set(r.keys()) != allowed_v_keys:
                    missing_k = allowed_v_keys - set(r.keys())
                    extra_k = set(r.keys()) - allowed_v_keys
                    raise CheckpointValidationError(
                        f"Validation record dict at index {idx} must have exact 4 keys. Missing: {sorted(missing_k)}, Extra: {sorted(extra_k)}"
                    )
                u_id = r["user_id"]
                p_id = r["parent_asin"]
                rating = r["rating"]
                ts = r["timestamp"]
            elif isinstance(r, (list, tuple)):
                if len(r) != 4:
                    raise CheckpointValidationError(
                        f"Validation record tuple at index {idx} must have exactly 4 elements (user_id, parent_asin, rating, timestamp), got {len(r)}: {_safe_repr(r)}"
                    )
                u_id, p_id, rating, ts = r[0], r[1], r[2], r[3]
            else:
                raise CheckpointValidationError(f"Invalid validation record format at index {idx}: {_safe_repr(r)}")

            _validate_clean_text(u_id, f"Invalid user_id in validation record at index {idx}")
            _validate_clean_text(p_id, f"Invalid parent_asin in validation record at index {idx}")
            if isinstance(rating, bool) or not isinstance(rating, (int, float)):
                raise CheckpointValidationError(f"Invalid rating in validation record at index {idx}: {_safe_repr(rating)}")
            try:
                frating = float(rating)
            except (OverflowError, ValueError):
                raise CheckpointValidationError(f"Numeric overflow in rating in validation record at index {idx}: {_safe_repr(rating)}")
            if not math.isfinite(frating):
                raise CheckpointValidationError(f"Invalid rating in validation record at index {idx}: {_safe_repr(rating)}")
            if isinstance(ts, bool) or not isinstance(ts, int):
                raise CheckpointValidationError(f"Invalid timestamp in validation record at index {idx}: {_safe_repr(ts)}")

            pair = (u_id, p_id)
            if pair in seen_v_pairs:
                raise CheckpointValidationError(f"Duplicate validation record pair at index {idx}: {pair}")
            seen_v_pairs.add(pair)

            record_user_items[u_id].add(p_id)
            canon_records.append({
                "user": u_id,
                "item": p_id,
                "rating": frating,
                "timestamp": int(ts),
            })

        canon_records.sort(key=lambda r: (r["user"], r["item"], r["rating"], r["timestamp"]))

    # Agreement check between targets and records when both are supplied
    if targets is not None and records is not None:
        if set(target_user_items.keys()) != set(record_user_items.keys()):
            missing_u = set(target_user_items.keys()) ^ set(record_user_items.keys())
            raise CheckpointValidationError(
                f"Validation targets users and records users mismatch: {sorted(missing_u)}"
            )
        for u, t_set in target_user_items.items():
            r_set = record_user_items[u]
            if t_set != r_set:
                raise CheckpointValidationError(
                    f"Validation targets items do not match records items for user {u!r}. Diff: {t_set ^ r_set}"
                )

    payload: Dict[str, Any] = {
        "schema": "amazon4u-canonical-validation-v1",
        "targets": canon_targets,
        "records": canon_records,
    }

    canon_bytes = _canonical_json_bytes(payload, role="validation fingerprint")
    return hashlib.sha256(canon_bytes).hexdigest()


def _validate_settings_section(sec_name: str, sec_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a specific resolved settings section against REQUIRED_SETTINGS_FIELDS."""
    if not isinstance(sec_dict, (dict, Mapping)) or not sec_dict:
        raise CheckpointValidationError(f"Section {sec_name!r} in resolved_settings must be a non-empty mapping.")

    for k in sec_dict.keys():
        if not isinstance(k, str):
            raise CheckpointValidationError(f"Section {sec_name!r} keys must be strings, got {type(k).__name__}: {k!r}")

    expected_fields = set(REQUIRED_SETTINGS_FIELDS[sec_name])
    actual_fields = set(sec_dict.keys())
    if actual_fields != expected_fields:
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        raise CheckpointValidationError(
            f"Section {sec_name!r} fields mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
        )

    canon_sec: Dict[str, Any] = {}

    def _check_str(field: str, val: Any) -> str:
        return _validate_clean_text(val, f"Field {sec_name}.{field}")

    def _check_int(field: str, val: Any, min_val: int = 0) -> int:
        if isinstance(val, bool) or not isinstance(val, int):
            raise CheckpointValidationError(
                f"Field {sec_name}.{field} must be an integer (not boolean), got {_safe_repr(val)}"
            )
        if val < min_val:
            raise CheckpointValidationError(
                f"Field {sec_name}.{field} must be >= {min_val}, got {_safe_repr(val)}"
            )
        return val

    def _check_float(field: str, val: Any, min_val: Optional[float] = None, strictly_positive: bool = False) -> float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise CheckpointValidationError(
                f"Field {sec_name}.{field} must be a number (not boolean), got {_safe_repr(val)}"
            )
        try:
            fval = float(val)
        except (OverflowError, ValueError):
            raise CheckpointValidationError(
                f"Numeric overflow in field {sec_name}.{field}: {_safe_repr(val)}"
            )
        if not math.isfinite(fval):
            raise CheckpointValidationError(
                f"Field {sec_name}.{field} must be finite, got {_safe_repr(val)}"
            )
        if strictly_positive and fval <= 0.0:
            raise CheckpointValidationError(
                f"Field {sec_name}.{field} must be strictly positive, got {_safe_repr(val)}"
            )
        if min_val is not None and fval < min_val:
            raise CheckpointValidationError(
                f"Field {sec_name}.{field} must be >= {min_val}, got {_safe_repr(val)}"
            )
        return fval

    def _check_bool(field: str, val: Any) -> bool:
        if not isinstance(val, bool):
            raise CheckpointValidationError(
                f"Field {sec_name}.{field} must be a strict boolean, got {_safe_repr(val)}"
            )
        return val

    if sec_name == "protocol":
        canon_sec["candidate_policy"] = _check_str("candidate_policy", sec_dict["candidate_policy"])
        canon_sec["lifecycle_state"] = _check_str("lifecycle_state", sec_dict["lifecycle_state"])
        canon_sec["protocol_version"] = _check_str("protocol_version", sec_dict["protocol_version"])
    elif sec_name == "execution":
        canon_sec["cpu_threads"] = _check_int("cpu_threads", sec_dict["cpu_threads"], min_val=1)
        canon_sec["execution_device"] = _check_str("execution_device", sec_dict["execution_device"])
        canon_sec["execution_label"] = _check_str("execution_label", sec_dict["execution_label"])
        canon_sec["runtime_mode"] = _check_str("runtime_mode", sec_dict["runtime_mode"])
        runtime_workers = _check_int("runtime_workers", sec_dict["runtime_workers"], min_val=0)
        canon_sec["runtime_workers"] = runtime_workers

        ws = sec_dict["worker_settings"]
        if not isinstance(ws, (dict, Mapping)):
            raise CheckpointValidationError(f"Field execution.worker_settings must be a mapping, got {type(ws).__name__}")
        for wsk in ws.keys():
            if not isinstance(wsk, str):
                raise CheckpointValidationError(f"execution.worker_settings keys must be strings, got {type(wsk).__name__}: {_safe_repr(wsk)}")
        if set(ws.keys()) != WORKER_SETTINGS_KEYS:
            missing_ws = WORKER_SETTINGS_KEYS - set(ws.keys())
            extra_ws = set(ws.keys()) - WORKER_SETTINGS_KEYS
            raise CheckpointValidationError(
                f"execution.worker_settings fields mismatch. Missing: {sorted(missing_ws)}, Extra: {sorted(extra_ws)}"
            )

        canon_ws: Dict[str, Any] = {}
        if runtime_workers == 0:
            if ws["prefetch_factor"] is not None:
                raise CheckpointValidationError(
                    f"execution.worker_settings.prefetch_factor must be None when runtime_workers == 0, got {_safe_repr(ws['prefetch_factor'])}"
                )
            if not isinstance(ws["persistent_workers"], bool) or ws["persistent_workers"] is not False:
                raise CheckpointValidationError(
                    f"execution.worker_settings.persistent_workers must be strictly False when runtime_workers == 0, got {_safe_repr(ws['persistent_workers'])}"
                )
            if ws["multiprocessing_context"] is not None:
                raise CheckpointValidationError(
                    f"execution.worker_settings.multiprocessing_context must be None when runtime_workers == 0, got {_safe_repr(ws['multiprocessing_context'])}"
                )
            if ws["worker_init_function"] is not None:
                raise CheckpointValidationError(
                    f"execution.worker_settings.worker_init_function must be None when runtime_workers == 0, got {_safe_repr(ws['worker_init_function'])}"
                )
            canon_ws = {
                "multiprocessing_context": None,
                "persistent_workers": False,
                "prefetch_factor": None,
                "worker_init_function": None,
            }
        else:
            pf = ws["prefetch_factor"]
            if isinstance(pf, bool) or not isinstance(pf, int) or pf <= 0:
                raise CheckpointValidationError(
                    f"execution.worker_settings.prefetch_factor must be a positive integer when runtime_workers > 0, got {_safe_repr(pf)}"
                )
            pw = ws["persistent_workers"]
            if not isinstance(pw, bool):
                raise CheckpointValidationError(
                    f"execution.worker_settings.persistent_workers must be a strict boolean, got {_safe_repr(pw)}"
                )
            mc = _validate_clean_text(ws["multiprocessing_context"], "execution.worker_settings.multiprocessing_context")
            wif = _validate_clean_text(ws["worker_init_function"], "execution.worker_settings.worker_init_function")
            canon_ws = {
                "multiprocessing_context": mc,
                "persistent_workers": pw,
                "prefetch_factor": pf,
                "worker_init_function": wif,
            }
        canon_sec["worker_settings"] = {k: canon_ws[k] for k in sorted(canon_ws.keys())}
    elif sec_name == "model":
        canon_sec["architecture"] = _check_str("architecture", sec_dict["architecture"])
        canon_sec["embedding_dim"] = _check_int("embedding_dim", sec_dict["embedding_dim"], min_val=1)
        canon_sec["initialization"] = _check_str("initialization", sec_dict["initialization"])
        canon_sec["num_layers"] = _check_int("num_layers", sec_dict["num_layers"], min_val=0)
    elif sec_name == "training":
        betas = sec_dict["adam_betas"]
        if isinstance(betas, (str, bytes, dict)) or not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise CheckpointValidationError(
                f"Field training.adam_betas must be a 2-element sequence of floats, got {_safe_repr(betas)}"
            )
        canon_betas: List[float] = []
        for b_idx, b_val in enumerate(betas):
            if isinstance(b_val, bool) or not isinstance(b_val, (int, float)):
                raise CheckpointValidationError(
                    f"training.adam_betas[{b_idx}] must be a finite number (not boolean), got {_safe_repr(b_val)}"
                )
            try:
                fb = float(b_val)
            except (OverflowError, ValueError):
                raise CheckpointValidationError(f"Numeric overflow in training.adam_betas[{b_idx}]: {_safe_repr(b_val)}")
            if not math.isfinite(fb):
                raise CheckpointValidationError(
                    f"training.adam_betas[{b_idx}] must be a finite number (not boolean), got {_safe_repr(b_val)}"
                )
            if not (0.0 < fb < 1.0):
                raise CheckpointValidationError(
                    f"training.adam_betas[{b_idx}] must be strictly between 0 and 1, got {_safe_repr(b_val)}"
                )
            canon_betas.append(fb)
        canon_sec["adam_betas"] = canon_betas

        canon_sec["adam_epsilon"] = _check_float("adam_epsilon", sec_dict["adam_epsilon"], strictly_positive=True)
        canon_sec["batch_size"] = _check_int("batch_size", sec_dict["batch_size"], min_val=1)
        canon_sec["endpoint_regularization"] = _check_float("endpoint_regularization", sec_dict["endpoint_regularization"], min_val=0.0)
        canon_sec["learning_rate"] = _check_float("learning_rate", sec_dict["learning_rate"], strictly_positive=True)
        canon_sec["loss_function"] = _check_str("loss_function", sec_dict["loss_function"])
        canon_sec["max_epochs"] = _check_int("max_epochs", sec_dict["max_epochs"], min_val=1)
        canon_sec["optimizer"] = _check_str("optimizer", sec_dict["optimizer"])
        canon_sec["patience"] = _check_int("patience", sec_dict["patience"], min_val=1)
        canon_sec["validation_frequency"] = _check_int("validation_frequency", sec_dict["validation_frequency"], min_val=1)
        canon_sec["weight_decay"] = _check_float("weight_decay", sec_dict["weight_decay"], min_val=0.0)
    elif sec_name == "sampling":
        canon_sec["negative_strategy"] = _check_str("negative_strategy", sec_dict["negative_strategy"])
        canon_sec["negatives_per_positive"] = _check_int("negatives_per_positive", sec_dict["negatives_per_positive"], min_val=1)

        # neighbor_fanouts: sequence of non-negative ints, preserving order
        fanouts = sec_dict["neighbor_fanouts"]
        if isinstance(fanouts, (str, bytes, dict)) or not isinstance(fanouts, (list, tuple)):
            raise CheckpointValidationError(
                f"Field sampling.neighbor_fanouts must be a list or tuple of integers, got {_safe_repr(fanouts)}"
            )
        canon_fanouts: List[int] = []
        for f_idx, f_val in enumerate(fanouts):
            if isinstance(f_val, bool) or not isinstance(f_val, int) or f_val < 0:
                raise CheckpointValidationError(
                    f"sampling.neighbor_fanouts[{f_idx}] must be an integer >= 0 (not boolean), got {_safe_repr(f_val)}"
                )
            canon_fanouts.append(int(f_val))
        canon_sec["neighbor_fanouts"] = canon_fanouts

        canon_sec["pool_policy"] = _check_str("pool_policy", sec_dict["pool_policy"])
    elif sec_name == "determinism":
        canon_sec["deterministic_algorithms"] = _check_bool("deterministic_algorithms", sec_dict["deterministic_algorithms"])
        canon_sec["reproducibility_version"] = _check_str("reproducibility_version", sec_dict["reproducibility_version"])
        det_seed = _check_int("seed", sec_dict["seed"], min_val=0)
        canon_sec["seed"] = det_seed

        # rng_streams: list/tuple of records with EXACT keys {name, applicable, algorithm, seed, derivation, reason}
        streams = sec_dict["rng_streams"]
        if isinstance(streams, (str, bytes, dict)) or not isinstance(streams, (list, tuple)):
            raise CheckpointValidationError(
                f"Field determinism.rng_streams must be a sequence of stream records, got {_safe_repr(streams)}"
            )
        if len(streams) == 0:
            raise CheckpointValidationError("determinism.rng_streams cannot be empty.")

        canon_streams: List[Dict[str, Any]] = []
        seen_stream_names: Set[str] = set()
        for s_idx, s_rec in enumerate(streams):
            if not isinstance(s_rec, (dict, Mapping)):
                raise CheckpointValidationError(
                    f"determinism.rng_streams[{s_idx}] must be a mapping, got {type(s_rec).__name__}: {_safe_repr(s_rec)}"
                )
            for rk in s_rec.keys():
                if not isinstance(rk, str):
                    raise CheckpointValidationError(
                        f"determinism.rng_streams[{s_idx}] record keys must be strings, got {type(rk).__name__}: {_safe_repr(rk)}"
                    )
            if set(s_rec.keys()) != RNG_STREAM_RECORD_KEYS:
                missing_rk = RNG_STREAM_RECORD_KEYS - set(s_rec.keys())
                extra_rk = set(s_rec.keys()) - RNG_STREAM_RECORD_KEYS
                raise CheckpointValidationError(
                    f"determinism.rng_streams[{s_idx}] record keys mismatch. Missing: {sorted(missing_rk)}, Extra: {sorted(extra_rk)}"
                )

            s_name = s_rec["name"]
            if not isinstance(s_name, str) or s_name not in CONFIGURED_RNG_COMPONENTS:
                raise CheckpointValidationError(
                    f"Invalid stream name in determinism.rng_streams[{s_idx}]: {_safe_repr(s_name)}. "
                    f"Must be one of {CONFIGURED_RNG_COMPONENTS}."
                )
            if s_name in seen_stream_names:
                raise CheckpointValidationError(
                    f"Duplicate stream name in determinism.rng_streams: {_safe_repr(s_name)}"
                )
            seen_stream_names.add(s_name)

            applicable = s_rec["applicable"]
            if not isinstance(applicable, bool):
                raise CheckpointValidationError(
                    f"determinism.rng_streams[{s_idx}] ({s_name}).applicable must be a strict boolean, got {_safe_repr(applicable)}"
                )

            canon_s: Dict[str, Any] = {
                "name": s_name,
                "applicable": applicable,
            }
            if applicable:
                alg = _validate_clean_text(s_rec["algorithm"], f"determinism.rng_streams[{s_idx}] ({s_name}).algorithm")
                sd = s_rec["seed"]
                if isinstance(sd, bool) or not isinstance(sd, int) or sd < 0:
                    raise CheckpointValidationError(
                        f"determinism.rng_streams[{s_idx}] ({s_name}).seed must be a non-negative integer when applicable=True, got {_safe_repr(sd)}"
                    )
                drv = _validate_clean_text(s_rec["derivation"], f"determinism.rng_streams[{s_idx}] ({s_name}).derivation")
                if s_rec["reason"] is not None:
                    raise CheckpointValidationError(
                        f"determinism.rng_streams[{s_idx}] ({s_name}).reason must be None when applicable=True, got {_safe_repr(s_rec['reason'])}"
                    )
                canon_s["algorithm"] = alg
                canon_s["seed"] = sd
                canon_s["derivation"] = drv
                canon_s["reason"] = None
            else:
                if s_rec["algorithm"] is not None:
                    raise CheckpointValidationError(
                        f"determinism.rng_streams[{s_idx}] ({s_name}).algorithm must be None when applicable=False, got {_safe_repr(s_rec['algorithm'])}"
                    )
                if s_rec["seed"] is not None:
                    raise CheckpointValidationError(
                        f"determinism.rng_streams[{s_idx}] ({s_name}).seed must be None when applicable=False, got {_safe_repr(s_rec['seed'])}"
                    )
                if s_rec["derivation"] is not None:
                    raise CheckpointValidationError(
                        f"determinism.rng_streams[{s_idx}] ({s_name}).derivation must be None when applicable=False, got {_safe_repr(s_rec['derivation'])}"
                    )
                rsn = _validate_clean_text(s_rec["reason"], f"determinism.rng_streams[{s_idx}] ({s_name}).reason")
                canon_s["algorithm"] = None
                canon_s["seed"] = None
                canon_s["derivation"] = None
                canon_s["reason"] = rsn

            canon_streams.append({k: canon_s[k] for k in sorted(canon_s.keys())})

        if set(seen_stream_names) != set(CONFIGURED_RNG_COMPONENTS):
            missing_streams = set(CONFIGURED_RNG_COMPONENTS) - seen_stream_names
            extra_streams = seen_stream_names - set(CONFIGURED_RNG_COMPONENTS)
            raise CheckpointValidationError(
                f"determinism.rng_streams components mismatch. Missing: {sorted(missing_streams)}, Extra: {sorted(extra_streams)}"
            )

        canon_sec["rng_streams"] = canon_streams

        # worker_derivation: EXACT record with applicable, algorithm, base_seed, derivation, reason
        wd = sec_dict["worker_derivation"]
        if not isinstance(wd, (dict, Mapping)):
            raise CheckpointValidationError(
                f"Field determinism.worker_derivation must be a mapping, got {type(wd).__name__}: {_safe_repr(wd)}"
            )
        for wdk in wd.keys():
            if not isinstance(wdk, str):
                raise CheckpointValidationError(
                    f"determinism.worker_derivation keys must be strings, got {type(wdk).__name__}: {_safe_repr(wdk)}"
                )
        if set(wd.keys()) != WORKER_DERIVATION_KEYS:
            missing_wd = WORKER_DERIVATION_KEYS - set(wd.keys())
            extra_wd = set(wd.keys()) - WORKER_DERIVATION_KEYS
            raise CheckpointValidationError(
                f"determinism.worker_derivation fields mismatch. Missing: {sorted(missing_wd)}, Extra: {sorted(extra_wd)}"
            )

        wd_applicable = wd["applicable"]
        if not isinstance(wd_applicable, bool):
            raise CheckpointValidationError(
                f"determinism.worker_derivation.applicable must be a strict boolean, got {_safe_repr(wd_applicable)}"
            )

        canon_wd: Dict[str, Any] = {
            "applicable": wd_applicable,
        }
        if wd_applicable:
            w_alg = _validate_clean_text(wd["algorithm"], "determinism.worker_derivation.algorithm")
            w_bs = wd["base_seed"]
            if isinstance(w_bs, bool) or not isinstance(w_bs, int) or w_bs < 0:
                raise CheckpointValidationError(
                    f"determinism.worker_derivation.base_seed must be a non-negative integer when applicable=True, got {_safe_repr(w_bs)}"
                )
            w_drv = _validate_clean_text(wd["derivation"], "determinism.worker_derivation.derivation")
            if wd["reason"] is not None:
                raise CheckpointValidationError(
                    f"determinism.worker_derivation.reason must be None when applicable=True, got {_safe_repr(wd['reason'])}"
                )
            canon_wd["algorithm"] = w_alg
            canon_wd["base_seed"] = w_bs
            canon_wd["derivation"] = w_drv
            canon_wd["reason"] = None
        else:
            if wd["algorithm"] is not None:
                raise CheckpointValidationError(
                    f"determinism.worker_derivation.algorithm must be None when applicable=False, got {_safe_repr(wd['algorithm'])}"
                )
            if wd["base_seed"] is not None:
                raise CheckpointValidationError(
                    f"determinism.worker_derivation.base_seed must be None when applicable=False, got {_safe_repr(wd['base_seed'])}"
                )
            if wd["derivation"] is not None:
                raise CheckpointValidationError(
                    f"determinism.worker_derivation.derivation must be None when applicable=False, got {_safe_repr(wd['derivation'])}"
                )
            w_rsn = _validate_clean_text(wd["reason"], "determinism.worker_derivation.reason")
            canon_wd["algorithm"] = None
            canon_wd["base_seed"] = None
            canon_wd["derivation"] = None
            canon_wd["reason"] = w_rsn

        canon_sec["worker_derivation"] = {k: canon_wd[k] for k in sorted(canon_wd.keys())}
    return {k: canon_sec[k] for k in sorted(canon_sec.keys())}


def _validate_resolved_settings_consistency(canon_settings: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate cross-section consistency among execution and determinism settings."""
    if "execution" in canon_settings and "determinism" in canon_settings:
        rw = canon_settings["execution"]["runtime_workers"]
        wd = canon_settings["determinism"]["worker_derivation"]
        det_seed = canon_settings["determinism"]["seed"]
        rng_streams = canon_settings["determinism"]["rng_streams"]

        # Applicability must agree with runtime_workers > 0
        if rw > 0 and not wd["applicable"]:
            raise CheckpointValidationError("determinism.worker_derivation must be applicable when execution.runtime_workers > 0.")
        if rw == 0 and wd["applicable"]:
            raise CheckpointValidationError("determinism.worker_derivation cannot be applicable when execution.runtime_workers == 0.")

        # Active base_seed must agree with determinism.seed
        if wd["applicable"]:
            if wd["base_seed"] != det_seed:
                raise CheckpointValidationError(
                    f"determinism.worker_derivation.base_seed ({_safe_repr(wd['base_seed'])}) must agree with determinism.seed ({_safe_repr(det_seed)})."
                )

        # DataLoader_workers stream applicability must also agree
        for s in rng_streams:
            if s["name"] == "DataLoader_workers":
                if s["applicable"] != wd["applicable"]:
                    raise CheckpointValidationError(
                        f"determinism.rng_streams['DataLoader_workers'].applicable ({s['applicable']}) must agree with worker_derivation.applicable ({wd['applicable']})."
                    )
                break


def compute_settings_fingerprint(resolved_settings: Mapping[str, Any]) -> str:
    """Compute deterministic SHA-256 over resolved scientific training, execution, and model settings.

    Enforces:
    - Non-empty mapping whose top-level keys match REQUIRED_SETTINGS_SECTIONS exactly:
      ('protocol', 'execution', 'model', 'training', 'sampling', 'determinism').
    - Each section must match REQUIRED_SETTINGS_FIELDS exactly with strict types.
    - Rejection of missing/extra sections, missing/extra fields, placeholders,
      and malformed or non-finite numbers.
    """
    if not isinstance(resolved_settings, (dict, Mapping)) or not resolved_settings:
        raise CheckpointValidationError("resolved_settings must be a non-empty mapping.")

    for sec in resolved_settings.keys():
        if not isinstance(sec, str):
            raise CheckpointValidationError(f"resolved_settings keys must be strings, got {type(sec).__name__}: {sec!r}")

    required_sections = set(REQUIRED_SETTINGS_SECTIONS)
    actual_sections = set(resolved_settings.keys())
    if actual_sections != required_sections:
        missing_sec = required_sections - actual_sections
        extra_sec = actual_sections - required_sections
        raise CheckpointValidationError(
            f"resolved_settings must contain all 6 mandatory sections. Missing: {sorted(missing_sec)}, Extra: {sorted(extra_sec)}"
        )

    canon_settings: Dict[str, Dict[str, Any]] = {}
    for sec_name in REQUIRED_SETTINGS_SECTIONS:
        sec_val = resolved_settings[sec_name]
        canon_settings[sec_name] = _validate_settings_section(sec_name, sec_val)

    _validate_resolved_settings_consistency(canon_settings)

    payload: Dict[str, Any] = {
        "schema": "amazon4u-canonical-settings-v1",
        "settings": {sec: canon_settings[sec] for sec in sorted(canon_settings.keys())},
    }

    canon_bytes = _canonical_json_bytes(payload, role="settings fingerprint")
    return hashlib.sha256(canon_bytes).hexdigest()


# ---------------------------------------------------------------------------
# CanonicalResumeIdentity Container & Validators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalResumeIdentity:
    """Deterministic, content-grounded canonical resume identity.

    Retains actual training records, positive history, condition warm pool,
    typed graph structure, validation selection inputs, and resolved scientific
    settings alongside their deterministic SHA-256 fingerprints.

    Note on mutability:
    This dataclass is shallowly frozen via frozen=True (top-level attributes cannot be reassigned).
    However, nested Python containers (dict, list) remain mutable if accessed directly. Callers
    must treat snapshots as isolated copies and revalidate with validate_canonical_identity.
    """

    method: str
    category: str
    feedback_condition: str
    seed: int
    data_fingerprint: str
    graph_fingerprint: str
    validation_fingerprint: str
    settings_fingerprint: str
    identity_hash: str
    training_records: Tuple[Dict[str, Any], ...]
    positive_history: Dict[str, Tuple[str, ...]]
    all_rated_history: Dict[str, Tuple[str, ...]]
    warm_items: Tuple[str, ...]
    graph_nodes: Dict[str, Tuple[str, ...]]
    graph_edges: Tuple[Dict[str, Any], ...]
    product_metadata: Dict[str, Dict[str, Any]]
    validation_inputs: Dict[str, Any]
    resolved_settings: Dict[str, Any]

    def __post_init__(self) -> None:
        _validate_identity_header(
            self.method,
            self.category,
            self.feedback_condition,
            self.seed,
            {
                "data_fingerprint": self.data_fingerprint,
                "graph_fingerprint": self.graph_fingerprint,
                "validation_fingerprint": self.validation_fingerprint,
                "settings_fingerprint": self.settings_fingerprint,
                "identity_hash": self.identity_hash,
            },
            role="CanonicalResumeIdentity",
        )

        if not self.training_records or not self.positive_history or not self.all_rated_history or not self.warm_items:
            raise CheckpointValidationError("CanonicalResumeIdentity must retain actual data records and histories.")
        if not self.graph_nodes or not self.graph_edges:
            raise CheckpointValidationError("CanonicalResumeIdentity must retain actual graph nodes and edges.")
        if not self.product_metadata:
            raise CheckpointValidationError("CanonicalResumeIdentity must retain actual product metadata.")
        if not self.validation_inputs:
            raise CheckpointValidationError("CanonicalResumeIdentity must retain validation inputs.")
        if not self.resolved_settings:
            raise CheckpointValidationError("CanonicalResumeIdentity must retain resolved settings.")

    def to_dict(self) -> Dict[str, Any]:
        """Return an isolated dictionary snapshot of the identity."""
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "method": self.method,
            "category": self.category,
            "feedback_condition": self.feedback_condition,
            "seed": self.seed,
            "data_fingerprint": self.data_fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "validation_fingerprint": self.validation_fingerprint,
            "settings_fingerprint": self.settings_fingerprint,
            "identity_hash": self.identity_hash,
            "training_records": [copy.deepcopy(r) for r in self.training_records],
            "positive_history": {k: list(v) for k, v in self.positive_history.items()},
            "all_rated_history": {k: list(v) for k, v in self.all_rated_history.items()},
            "warm_items": list(self.warm_items),
            "graph_nodes": {k: list(v) for k, v in self.graph_nodes.items()},
            "graph_edges": [copy.deepcopy(e) for e in self.graph_edges],
            "product_metadata": copy.deepcopy(self.product_metadata),
            "validation_inputs": copy.deepcopy(self.validation_inputs),
            "resolved_settings": copy.deepcopy(self.resolved_settings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalResumeIdentity":
        """Reconstruct CanonicalResumeIdentity from a dictionary, strictly validating all records and fingerprints."""
        if not isinstance(data, (dict, Mapping)):
            raise CheckpointValidationError(f"Expected mapping for CanonicalResumeIdentity, got {type(data).__name__}")

        for k in data.keys():
            if not isinstance(k, str):
                raise CheckpointValidationError(
                    f"CanonicalResumeIdentity dictionary keys must be strings, got {type(k).__name__}: {k!r}"
                )

        schema = data.get("schema_version")
        if schema != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointValidationError(f"Schema version mismatch: expected {CHECKPOINT_SCHEMA_VERSION!r}, got {schema!r}")

        expected_keys = {
            "schema_version",
            "method",
            "category",
            "feedback_condition",
            "seed",
            "data_fingerprint",
            "graph_fingerprint",
            "validation_fingerprint",
            "settings_fingerprint",
            "identity_hash",
            "training_records",
            "positive_history",
            "all_rated_history",
            "warm_items",
            "graph_nodes",
            "graph_edges",
            "product_metadata",
            "validation_inputs",
            "resolved_settings",
        }
        if set(data.keys()) != expected_keys:
            missing = expected_keys - set(data.keys())
            extra = set(data.keys()) - expected_keys
            raise CheckpointValidationError(
                f"CanonicalResumeIdentity keys mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        _validate_identity_header(
            data["method"],
            data["category"],
            data["feedback_condition"],
            data["seed"],
            {
                "data_fingerprint": data["data_fingerprint"],
                "graph_fingerprint": data["graph_fingerprint"],
                "validation_fingerprint": data["validation_fingerprint"],
                "settings_fingerprint": data["settings_fingerprint"],
                "identity_hash": data["identity_hash"],
            },
            role="from_dict",
        )

        if not isinstance(data["training_records"], (list, tuple)):
            raise CheckpointValidationError(f"training_records in serialized identity must be a sequence, got {type(data['training_records']).__name__}")
        if not isinstance(data["positive_history"], (dict, Mapping)):
            raise CheckpointValidationError(f"positive_history in serialized identity must be a mapping, got {type(data['positive_history']).__name__}")
        if not isinstance(data["all_rated_history"], (dict, Mapping)):
            raise CheckpointValidationError(f"all_rated_history in serialized identity must be a mapping, got {type(data['all_rated_history']).__name__}")
        if not isinstance(data["warm_items"], (list, tuple, set, frozenset)):
            raise CheckpointValidationError(f"warm_items in serialized identity must be a sequence, got {type(data['warm_items']).__name__}")
        if not isinstance(data["graph_nodes"], (dict, Mapping)):
            raise CheckpointValidationError(f"graph_nodes in serialized identity must be a mapping, got {type(data['graph_nodes']).__name__}")
        if not isinstance(data["graph_edges"], (list, tuple)):
            raise CheckpointValidationError(f"graph_edges in serialized identity must be a sequence, got {type(data['graph_edges']).__name__}")
        if not isinstance(data["product_metadata"], (dict, Mapping)):
            raise CheckpointValidationError(f"product_metadata in serialized identity must be a mapping, got {type(data['product_metadata']).__name__}")
        if not isinstance(data["validation_inputs"], (dict, Mapping)):
            raise CheckpointValidationError(f"validation_inputs in serialized identity must be a mapping, got {type(data['validation_inputs']).__name__}")
        if not isinstance(data["resolved_settings"], (dict, Mapping)):
            raise CheckpointValidationError(f"resolved_settings in serialized identity must be a mapping, got {type(data['resolved_settings']).__name__}")

        identity = build_canonical_identity(
            method=data["method"],
            category=data["category"],
            feedback_condition=data["feedback_condition"],
            seed=data["seed"],
            training_records=data["training_records"],
            positive_history=data["positive_history"],
            all_rated_history=data["all_rated_history"],
            warm_items=data["warm_items"],
            graph_nodes=data["graph_nodes"],
            graph_edges=data["graph_edges"],
            product_metadata=data["product_metadata"],
            validation_inputs=data["validation_inputs"],
            resolved_settings=data["resolved_settings"],
        )

        for fp_field in (
            "data_fingerprint",
            "graph_fingerprint",
            "validation_fingerprint",
            "settings_fingerprint",
            "identity_hash",
        ):
            if data[fp_field] != getattr(identity, fp_field):
                raise CheckpointIdentityMismatchError(
                    f"Falsified or corrupt {fp_field} in serialized identity: stored {data[fp_field]!r} != "
                    f"recomputed {getattr(identity, fp_field)!r}"
                )

        return identity


def build_canonical_identity(
    method: str,
    category: str,
    feedback_condition: str,
    seed: int,
    training_records: Sequence[Any],
    positive_history: Mapping[str, Sequence[str]],
    all_rated_history: Mapping[str, Sequence[str]],
    warm_items: Sequence[str],
    graph_nodes: Mapping[str, Sequence[str]],
    graph_edges: Sequence[Any],
    product_metadata: Mapping[str, Mapping[str, Any]],
    validation_inputs: Mapping[str, Any],
    resolved_settings: Mapping[str, Any],
) -> CanonicalResumeIdentity:
    """Build a validated CanonicalResumeIdentity from mandatory actual records and content.

    Computes deterministic content fingerprints across all components and constructs
    the complete collision-free identity hash with canonical normalization.
    """
    _validate_identity_header(method, category, feedback_condition, seed, role="build_canonical_identity")

    if not isinstance(positive_history, (dict, Mapping)) or not positive_history:
        raise CheckpointValidationError("positive_history must be a non-empty mapping.")
    for pk in positive_history.keys():
        if not isinstance(pk, str) or not pk.strip() or pk.strip() != pk:
            raise CheckpointValidationError(f"Invalid user ID in positive_history: {pk!r}")
    users = sorted(positive_history.keys())

    # 1. Data fingerprint
    dfp = compute_data_fingerprint(
        feedback_condition=feedback_condition,
        users=users,
        training_records=training_records,
        positive_history=positive_history,
        all_rated_history=all_rated_history,
        warm_items=warm_items,
    )

    # 2. Graph fingerprint
    gfp = compute_graph_fingerprint(
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        product_metadata=product_metadata,
    )

    # 3. Validation inputs fingerprint
    vfp = compute_validation_fingerprint(validation_inputs)

    # 4. Resolved settings fingerprint
    sfp = compute_settings_fingerprint(resolved_settings)

    # 5. Composite identity hash
    combined = {
        "schema": "amazon4u-canonical-identity-v1",
        "method": method,
        "category": category,
        "feedback_condition": feedback_condition,
        "seed": seed,
        "data_fingerprint": dfp,
        "graph_fingerprint": gfp,
        "validation_fingerprint": vfp,
        "settings_fingerprint": sfp,
    }
    canon_bytes = _canonical_json_bytes(combined, role="identity hash")
    identity_hash = hashlib.sha256(canon_bytes).hexdigest()

    # Canonicalize and isolate all retained collections
    norm_records: List[Dict[str, Any]] = []
    for r in training_records:
        if isinstance(r, dict):
            u_id, p_id, rating, ts = r["user_id"], r["parent_asin"], r["rating"], r["timestamp"]
        else:
            u_id, p_id, rating, ts = r[0], r[1], r[2], r[3]
        norm_records.append({"user_id": u_id, "parent_asin": p_id, "rating": float(rating), "timestamp": int(ts)})
    norm_records.sort(key=lambda r: (r["user_id"], r["parent_asin"], r["rating"], r["timestamp"]))

    norm_edges: List[Dict[str, Any]] = []
    for e in graph_edges:
        if isinstance(e, dict):
            st, si, rel, dt, di, w = e["src_type"], e["src_id"], e["relation"], e["dst_type"], e["dst_id"], e["weight"]
        else:
            st, si, rel, dt, di, w = e[0], e[1], e[2], e[3], e[4], e[5]
        norm_edges.append({
            "src_type": st,
            "src_id": si,
            "relation": rel,
            "dst_type": dt,
            "dst_id": di,
            "weight": float(w),
        })
    norm_edges.sort(key=lambda e: (e["src_type"], e["src_id"], e["relation"], e["dst_type"], e["dst_id"], e["weight"]))

    # Canonicalize product metadata
    norm_metadata: Dict[str, Dict[str, Any]] = {}
    for pid in sorted(product_metadata.keys()):
        entry = product_metadata[pid]
        cats = entry.get("categories", [])
        norm_cats = sorted(cats)
        brand = entry.get("brand")
        norm_metadata[pid] = {
            "categories": norm_cats,
            "brand": brand,
        }

    # Canonicalize validation inputs
    norm_val_inputs: Dict[str, Any] = {}
    if "targets" in validation_inputs and validation_inputs["targets"] is not None:
        norm_val_inputs["targets"] = {
            u: sorted(validation_inputs["targets"][u])
            for u in sorted(validation_inputs["targets"].keys())
        }
    if "records" in validation_inputs and validation_inputs["records"] is not None:
        val_recs: List[Dict[str, Any]] = []
        for r in validation_inputs["records"]:
            if isinstance(r, dict):
                val_recs.append({
                    "user_id": r["user_id"],
                    "parent_asin": r["parent_asin"],
                    "rating": float(r["rating"]),
                    "timestamp": int(r["timestamp"]),
                })
            else:
                val_recs.append({
                    "user_id": r[0],
                    "parent_asin": r[1],
                    "rating": float(r[2]),
                    "timestamp": int(r[3]),
                })
        val_recs.sort(key=lambda r: (r["user_id"], r["parent_asin"], r["rating"], r["timestamp"]))
        norm_val_inputs["records"] = val_recs

    # Canonicalize resolved settings
    norm_settings = {sec: _validate_settings_section(sec, resolved_settings[sec]) for sec in sorted(resolved_settings.keys())}
    _validate_resolved_settings_consistency(norm_settings)

    return CanonicalResumeIdentity(
        method=method,
        category=category,
        feedback_condition=feedback_condition,
        seed=seed,
        data_fingerprint=dfp,
        graph_fingerprint=gfp,
        validation_fingerprint=vfp,
        settings_fingerprint=sfp,
        identity_hash=identity_hash,
        training_records=tuple(norm_records),
        positive_history={k: tuple(sorted(set(positive_history[k]))) for k in sorted(positive_history.keys())},
        all_rated_history={k: tuple(sorted(set(all_rated_history[k]))) for k in sorted(all_rated_history.keys())},
        warm_items=tuple(sorted(set(warm_items))),
        graph_nodes={k: tuple(sorted(set(graph_nodes[k]))) for k in sorted(graph_nodes.keys())},
        graph_edges=tuple(norm_edges),
        product_metadata=norm_metadata,
        validation_inputs=norm_val_inputs,
        resolved_settings=norm_settings,
    )


def validate_canonical_identity(
    actual: CanonicalResumeIdentity,
    expected: CanonicalResumeIdentity,
) -> None:
    """Validate canonical resume identity against trusted caller expectations.

    Preflights both actual and expected identities, recomputes all fingerprints
    from retained actual content to detect tampering, and verifies exact match
    against trusted caller expectations.
    """
    _validate_identity_pair(actual, expected, content_verified=False)


def _validate_identity_pair(
    actual: CanonicalResumeIdentity,
    expected: CanonicalResumeIdentity,
    *,
    content_verified: bool,
) -> None:
    """validate_canonical_identity; with content_verified, minus three repeated recomputes.

    Readiness C2. CanonicalResumeIdentity.from_dict recomputes the data, graph and
    validation fingerprints from the serialized content and refuses any that differ from
    the stored ones. The identity it returns holds that content as build_canonical_identity
    normalizes it (sorted, deduplicated, ratings float, timestamps int), which none of the
    three fingerprints distinguishes, so recomputing them from the retained content cannot
    fail for a from_dict result that nothing has changed since. D2 passes
    content_verified=True only for such a pair; every other check here still runs.
    """
    if not isinstance(actual, CanonicalResumeIdentity):
        raise CheckpointValidationError(f"actual must be CanonicalResumeIdentity, got {type(actual).__name__}")
    if not isinstance(expected, CanonicalResumeIdentity):
        raise CheckpointValidationError(f"expected must be CanonicalResumeIdentity, got {type(expected).__name__}")

    try:
        # Preflight all header and fingerprint fields on both actual and expected
        for role, obj in (("actual", actual), ("expected", expected)):
            _validate_identity_header(
                obj.method,
                obj.category,
                obj.feedback_condition,
                obj.seed,
                {
                    "data_fingerprint": obj.data_fingerprint,
                    "graph_fingerprint": obj.graph_fingerprint,
                    "validation_fingerprint": obj.validation_fingerprint,
                    "settings_fingerprint": obj.settings_fingerprint,
                    "identity_hash": obj.identity_hash,
                },
                role=role,
            )
            if not isinstance(obj.positive_history, (dict, Mapping)):
                raise CheckpointValidationError(f"{role}.positive_history must be a mapping, got {type(obj.positive_history).__name__}")
            if not isinstance(obj.all_rated_history, (dict, Mapping)):
                raise CheckpointValidationError(f"{role}.all_rated_history must be a mapping, got {type(obj.all_rated_history).__name__}")
            for pk in obj.positive_history.keys():
                _validate_clean_text(pk, f"{role}.positive_history key")
            for ak in obj.all_rated_history.keys():
                _validate_clean_text(ak, f"{role}.all_rated_history key")
            if not isinstance(obj.training_records, (list, tuple)):
                raise CheckpointValidationError(f"{role}.training_records must be a sequence, got {type(obj.training_records).__name__}")
            if not isinstance(obj.warm_items, (list, tuple, set, frozenset)):
                raise CheckpointValidationError(f"{role}.warm_items must be a sequence, got {type(obj.warm_items).__name__}")
            if not isinstance(obj.graph_nodes, (dict, Mapping)):
                raise CheckpointValidationError(f"{role}.graph_nodes must be a mapping, got {type(obj.graph_nodes).__name__}")
            if not isinstance(obj.graph_edges, (list, tuple)):
                raise CheckpointValidationError(f"{role}.graph_edges must be a sequence, got {type(obj.graph_edges).__name__}")
            if not isinstance(obj.product_metadata, (dict, Mapping)):
                raise CheckpointValidationError(f"{role}.product_metadata must be a mapping, got {type(obj.product_metadata).__name__}")
            if not isinstance(obj.validation_inputs, (dict, Mapping)):
                raise CheckpointValidationError(f"{role}.validation_inputs must be a mapping, got {type(obj.validation_inputs).__name__}")
            if not isinstance(obj.resolved_settings, (dict, Mapping)):
                raise CheckpointValidationError(f"{role}.resolved_settings must be a mapping, got {type(obj.resolved_settings).__name__}")

        # 1. Preflight and recompute actual fingerprints
        if not content_verified:
            actual_users = sorted(actual.positive_history.keys())
            recomputed_dfp = compute_data_fingerprint(
                feedback_condition=actual.feedback_condition,
                users=actual_users,
                training_records=actual.training_records,
                positive_history=actual.positive_history,
                all_rated_history=actual.all_rated_history,
                warm_items=actual.warm_items,
            )
            if actual.data_fingerprint != recomputed_dfp:
                raise CheckpointIdentityMismatchError(
                    f"Falsified or corrupt data_fingerprint in actual identity: {actual.data_fingerprint} != {recomputed_dfp}"
                )

            recomputed_gfp = compute_graph_fingerprint(
                graph_nodes=actual.graph_nodes,
                graph_edges=actual.graph_edges,
                product_metadata=actual.product_metadata,
            )
            if actual.graph_fingerprint != recomputed_gfp:
                raise CheckpointIdentityMismatchError(
                    f"Falsified or corrupt graph_fingerprint in actual identity: {actual.graph_fingerprint} != {recomputed_gfp}"
                )

            recomputed_vfp = compute_validation_fingerprint(actual.validation_inputs)
            if actual.validation_fingerprint != recomputed_vfp:
                raise CheckpointIdentityMismatchError(
                    f"Falsified or corrupt validation_fingerprint in actual identity: {actual.validation_fingerprint} != {recomputed_vfp}"
                )

        recomputed_sfp = compute_settings_fingerprint(actual.resolved_settings)
        if actual.settings_fingerprint != recomputed_sfp:
            raise CheckpointIdentityMismatchError(
                f"Falsified or corrupt settings_fingerprint in actual identity: {actual.settings_fingerprint} != {recomputed_sfp}"
            )

        actual_combined = {
            "schema": "amazon4u-canonical-identity-v1",
            "method": actual.method,
            "category": actual.category,
            "feedback_condition": actual.feedback_condition,
            "seed": actual.seed,
            "data_fingerprint": actual.data_fingerprint,
            "graph_fingerprint": actual.graph_fingerprint,
            "validation_fingerprint": actual.validation_fingerprint,
            "settings_fingerprint": actual.settings_fingerprint,
        }
        canon_bytes = _canonical_json_bytes(actual_combined, role="actual identity hash")
        recomputed_hash = hashlib.sha256(canon_bytes).hexdigest()
        if actual.identity_hash != recomputed_hash:
            raise CheckpointIdentityMismatchError(
                f"Falsified or corrupt identity_hash in actual identity: {actual.identity_hash} != {recomputed_hash}"
            )

        # 2. Preflight and recompute expected fingerprints
        if not content_verified:
            exp_users = sorted(expected.positive_history.keys())
            exp_dfp = compute_data_fingerprint(
                feedback_condition=expected.feedback_condition,
                users=exp_users,
                training_records=expected.training_records,
                positive_history=expected.positive_history,
                all_rated_history=expected.all_rated_history,
                warm_items=expected.warm_items,
            )
            if expected.data_fingerprint != exp_dfp:
                raise CheckpointIdentityMismatchError(
                    f"Corrupt or malformed expected identity: data_fingerprint {expected.data_fingerprint} != {exp_dfp}"
                )

            exp_gfp = compute_graph_fingerprint(
                graph_nodes=expected.graph_nodes,
                graph_edges=expected.graph_edges,
                product_metadata=expected.product_metadata,
            )
            if expected.graph_fingerprint != exp_gfp:
                raise CheckpointIdentityMismatchError(
                    f"Corrupt or malformed expected identity: graph_fingerprint {expected.graph_fingerprint} != {exp_gfp}"
                )

            exp_vfp = compute_validation_fingerprint(expected.validation_inputs)
            if expected.validation_fingerprint != exp_vfp:
                raise CheckpointIdentityMismatchError(
                    f"Corrupt or malformed expected identity: validation_fingerprint {expected.validation_fingerprint} != {exp_vfp}"
                )

        exp_sfp = compute_settings_fingerprint(expected.resolved_settings)
        if expected.settings_fingerprint != exp_sfp:
            raise CheckpointIdentityMismatchError(
                f"Corrupt or malformed expected identity: settings_fingerprint {expected.settings_fingerprint} != {exp_sfp}"
            )

        exp_combined = {
            "schema": "amazon4u-canonical-identity-v1",
            "method": expected.method,
            "category": expected.category,
            "feedback_condition": expected.feedback_condition,
            "seed": expected.seed,
            "data_fingerprint": expected.data_fingerprint,
            "graph_fingerprint": expected.graph_fingerprint,
            "validation_fingerprint": expected.validation_fingerprint,
            "settings_fingerprint": expected.settings_fingerprint,
        }
        exp_canon_bytes = _canonical_json_bytes(exp_combined, role="expected identity hash")
        exp_id_hash = hashlib.sha256(exp_canon_bytes).hexdigest()
        if expected.identity_hash != exp_id_hash:
            raise CheckpointIdentityMismatchError(
                f"Corrupt or malformed expected identity: identity_hash {expected.identity_hash} != {exp_id_hash}"
            )

        # 3. Compare actual against expected
        mismatches: List[str] = []
        for field in (
            "method",
            "category",
            "feedback_condition",
            "seed",
            "data_fingerprint",
            "graph_fingerprint",
            "validation_fingerprint",
            "settings_fingerprint",
            "identity_hash",
            "training_records",
            "positive_history",
            "all_rated_history",
            "warm_items",
            "graph_nodes",
            "graph_edges",
            "product_metadata",
            "validation_inputs",
            "resolved_settings",
        ):
            act_val = getattr(actual, field)
            exp_val = getattr(expected, field)
            if act_val != exp_val:
                mismatches.append(f"{field} (actual != expected)")

        if mismatches:
            raise CheckpointIdentityMismatchError(
                f"Canonical resume identity mismatch against expected: {', '.join(mismatches)}"
            )
    except CheckpointError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        raise CheckpointValidationError(f"Malformed retained structure in identity: {exc}") from exc


# ---------------------------------------------------------------------------
# Source & Environment Provenance
# ---------------------------------------------------------------------------

def _preflight_provenance_structure_and_bounds(
    provenance: Mapping[str, Any],
    role: str,
) -> int:
    """Preflight and validate complete structure, keys, types, and byte bounds of provenance.

    Performs complete schema and boundary validation with ZERO SHA-256 hashing.
    Returns total accumulated source bytes.
    """
    if not isinstance(provenance, (dict, Mapping)):
        raise CheckpointProvenanceError(f"{role} provenance must be a mapping, got {type(provenance).__name__}")

    for k in provenance.keys():
        if not isinstance(k, str):
            raise CheckpointProvenanceError(f"{role} provenance top-level keys must be strings, got {type(k).__name__}: {k!r}")

    if set(provenance.keys()) != PROVENANCE_TOP_LEVEL_KEYS:
        missing = PROVENANCE_TOP_LEVEL_KEYS - set(provenance.keys())
        extra = set(provenance.keys()) - PROVENANCE_TOP_LEVEL_KEYS
        raise CheckpointProvenanceError(
            f"{role} provenance top-level keys mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
        )

    schema = provenance["schema_version"]
    if schema != PROVENANCE_SCHEMA_VERSION:
        raise CheckpointProvenanceError(f"{role} provenance schema_version mismatch: {schema!r} != {PROVENANCE_SCHEMA_VERSION!r}")

    total_bytes = provenance["total_source_bytes"]
    if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 0:
        raise CheckpointProvenanceError(f"{role} total_source_bytes must be a non-negative integer, got {total_bytes!r}")

    if total_bytes > MAX_PROVENANCE_SOURCE_BYTES:
        raise CheckpointProvenanceError(
            f"{role} total_source_bytes exceeds 2 MiB limit ({MAX_PROVENANCE_SOURCE_BYTES} bytes): got {total_bytes} bytes."
        )

    source_files = provenance["source_files"]
    if not isinstance(source_files, (dict, Mapping)) or not source_files:
        raise CheckpointProvenanceError(f"{role} provenance source_files must be a non-empty mapping.")

    accumulated_bytes = 0
    for path_key, entry in source_files.items():
        if not isinstance(path_key, str):
            raise CheckpointProvenanceError(f"{role} source_files keys must be strings, got {type(path_key).__name__}: {path_key!r}")
        norm_path = validate_allowlisted_relative_path(path_key)
        if not isinstance(entry, (dict, Mapping)):
            raise CheckpointProvenanceError(f"{role} source_files[{path_key!r}] must be a mapping.")
        for ek in entry.keys():
            if not isinstance(ek, str):
                raise CheckpointProvenanceError(f"{role} source_files[{path_key!r}] entry keys must be strings, got {type(ek).__name__}: {ek!r}")
        if set(entry.keys()) != FILE_ENTRY_KEYS:
            missing_e = FILE_ENTRY_KEYS - set(entry.keys())
            extra_e = set(entry.keys()) - FILE_ENTRY_KEYS
            raise CheckpointProvenanceError(
                f"{role} source_files[{path_key!r}] keys mismatch. Missing: {sorted(missing_e)}, Extra: {sorted(extra_e)}"
            )

        entry_path = entry["path"]
        if entry_path != norm_path:
            raise CheckpointProvenanceError(f"{role} entry['path'] {entry_path!r} does not match key {norm_path!r}")

        content_bytes = entry["content_bytes"]
        if not isinstance(content_bytes, bytes):
            raise CheckpointProvenanceError(f"{role} content_bytes for {norm_path} must be bytes, got {type(content_bytes).__name__}")

        sz = entry["size_bytes"]
        if isinstance(sz, bool) or not isinstance(sz, int) or sz < 0:
            raise CheckpointProvenanceError(f"{role} size_bytes for {norm_path} must be a non-negative int, got {sz!r}")

        actual_sz = len(content_bytes)
        if sz != actual_sz:
            raise CheckpointProvenanceError(f"{role} declared size_bytes {sz} != actual bytes {actual_sz} for {norm_path}")

        accumulated_bytes += actual_sz
        if accumulated_bytes > MAX_PROVENANCE_SOURCE_BYTES:
            raise CheckpointProvenanceError(
                f"{role} total actual source bytes exceed 2 MiB limit: {accumulated_bytes} bytes."
            )

        sha = entry["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
            raise CheckpointProvenanceError(f"{role} invalid sha256 for {norm_path}: {sha!r}")

    if accumulated_bytes != total_bytes:
        raise CheckpointProvenanceError(
            f"{role} declared total_source_bytes ({total_bytes}) does not match sum of file bytes ({accumulated_bytes})."
        )

    # Validate Git evidence
    git_evidence = provenance["git"]
    if not isinstance(git_evidence, (dict, Mapping)):
        raise CheckpointProvenanceError(f"{role} git evidence must be a mapping.")
    for gk in git_evidence.keys():
        if not isinstance(gk, str):
            raise CheckpointProvenanceError(f"{role} git evidence keys must be strings, got {type(gk).__name__}: {gk!r}")
    if set(git_evidence.keys()) != GIT_EVIDENCE_KEYS:
        missing_g = GIT_EVIDENCE_KEYS - set(git_evidence.keys())
        extra_g = set(git_evidence.keys()) - GIT_EVIDENCE_KEYS
        raise CheckpointProvenanceError(f"{role} git keys mismatch. Missing: {sorted(missing_g)}, Extra: {sorted(extra_g)}")

    commit = git_evidence["commit"]
    if not isinstance(commit, str) or len(commit) != 40 or not all(c in "0123456789abcdefABCDEF" for c in commit):
        raise CheckpointProvenanceError(
            f"{role} Git commit hash must be a 40-character hex string, got {commit!r}"
        )

    dirty = git_evidence["dirty"]
    if not isinstance(dirty, bool):
        raise CheckpointProvenanceError(f"{role} dirty flag in Git evidence must be strict boolean, got {type(dirty).__name__}: {dirty!r}")

    tracked_diff = git_evidence["tracked_diff"]
    if not isinstance(tracked_diff, str):
        raise CheckpointProvenanceError(f"{role} tracked_diff must be a string, got {type(tracked_diff).__name__}")

    untracked_sources = git_evidence["untracked_sources"]
    if not isinstance(untracked_sources, (list, tuple)):
        raise CheckpointProvenanceError(f"{role} untracked_sources must be a sequence, got {type(untracked_sources).__name__}")

    seen_untracked: Set[str] = set()
    for u_path in untracked_sources:
        if not isinstance(u_path, str) or not u_path.strip() or u_path.strip() != u_path:
            raise CheckpointProvenanceError(f"{role} invalid path in untracked_sources: {u_path!r}")
        norm_u = validate_allowlisted_relative_path(u_path)
        if norm_u in seen_untracked:
            raise CheckpointProvenanceError(f"{role} duplicate untracked source path: {norm_u!r}")
        seen_untracked.add(norm_u)
        if norm_u not in source_files:
            raise CheckpointProvenanceError(
                f"{role} untracked source {norm_u!r} must have corresponding reconstructable bytes in source_files."
            )

    if not dirty:
        if tracked_diff != "":
            raise CheckpointProvenanceError(f"{role} git claims clean tree (dirty=False) but contains non-empty tracked_diff.")
        if len(untracked_sources) > 0:
            raise CheckpointProvenanceError(f"{role} git claims clean tree (dirty=False) but contains untracked_sources.")
    else:
        if tracked_diff == "" and len(untracked_sources) == 0:
            raise CheckpointProvenanceError(f"{role} git claims dirty tree (dirty=True) but diff and untracked_sources are both empty.")

    # Validate Environment records
    env = provenance["environment"]
    if not isinstance(env, (dict, Mapping)):
        raise CheckpointProvenanceError(f"{role} environment must be a mapping.")
    for ek in env.keys():
        if not isinstance(ek, str):
            raise CheckpointProvenanceError(f"{role} environment keys must be strings, got {type(ek).__name__}: {ek!r}")
    if set(env.keys()) != ENVIRONMENT_RECORD_KEYS:
        missing_env = ENVIRONMENT_RECORD_KEYS - set(env.keys())
        extra_env = set(env.keys()) - ENVIRONMENT_RECORD_KEYS
        raise CheckpointProvenanceError(f"{role} environment keys mismatch. Missing: {sorted(missing_env)}, Extra: {sorted(extra_env)}")

    for str_field in ("python_version", "platform_system", "platform_release", "platform_machine", "torch_version"):
        val = env[str_field]
        if not isinstance(val, str) or not val.strip():
            raise CheckpointProvenanceError(f"{role} environment {str_field!r} cannot be empty or non-string: {val!r}")

    cpu_cnt = env["cpu_count"]
    if isinstance(cpu_cnt, bool) or not isinstance(cpu_cnt, int) or cpu_cnt <= 0:
        raise CheckpointProvenanceError(f"{role} cpu_count in environment must be a positive integer, got {cpu_cnt!r}")

    lib_vers = env["library_versions"]
    if not isinstance(lib_vers, (dict, Mapping)) or not lib_vers:
        raise CheckpointProvenanceError(f"{role} library_versions in environment must be a non-empty mapping.")
    for lib_k, lib_v in lib_vers.items():
        if not isinstance(lib_k, str) or not lib_k.strip() or not isinstance(lib_v, str) or not lib_v.strip():
            raise CheckpointProvenanceError(f"{role} invalid library version entry: {lib_k!r}: {lib_v!r}")

    return accumulated_bytes


def build_source_provenance(
    source_files: Mapping[str, bytes],
    git_evidence: Mapping[str, Any],
    environment_records: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a pure, reconstructable source and environment provenance record.

    Enforces:
    - Explicit allowlisted relative scripts/configs paths -> actual bytes.
    - Total source bytes bounded to 2 MiB ceiling with pre-conversion validation.
    - Path traversal, absolute paths, credentials, and disallowed paths rejected.
    - Explicit Git commit (40-char hex), dirty flag (strict bool), tracked diff,
      and untracked-source records without fabricated clean defaults.
    - Untracked sources must be allowlisted relative paths present in source_files with bytes.
    - Mandatory nonempty environment records (Python, platform, CPU, PyTorch, library versions).
    - Deeply isolated snapshots returned (no aliasing with caller inputs).
    """
    if not isinstance(source_files, (dict, Mapping)) or not source_files:
        raise CheckpointProvenanceError("source_files must be a non-empty mapping of relative path to bytes.")

    # 1. Preflight aggregate sizes, types, and paths BEFORE any hashing
    total_bytes = 0
    validated_files: Dict[str, bytes] = {}
    for rel_path, content_bytes in source_files.items():
        if not isinstance(rel_path, str):
            raise CheckpointProvenanceError(f"Source file path must be a string, got {type(rel_path).__name__}: {rel_path!r}")
        norm_path = validate_allowlisted_relative_path(rel_path)
        if not isinstance(content_bytes, bytes):
            raise CheckpointProvenanceError(f"Source file {rel_path!r} content must be bytes, got {type(content_bytes).__name__}")
        total_bytes += len(content_bytes)
        if total_bytes > MAX_PROVENANCE_SOURCE_BYTES:
            raise CheckpointProvenanceError(
                f"Total source bytes exceed 2 MiB limit ({MAX_PROVENANCE_SOURCE_BYTES} bytes): got {total_bytes} bytes."
            )
        validated_files[norm_path] = content_bytes

    # 2. Preflight Git evidence BEFORE hashing
    if not isinstance(git_evidence, (dict, Mapping)):
        raise CheckpointProvenanceError("git_evidence must be a mapping.")
    for gk in git_evidence.keys():
        if not isinstance(gk, str):
            raise CheckpointProvenanceError(f"git_evidence keys must be strings, got {type(gk).__name__}: {gk!r}")
    if set(git_evidence.keys()) != GIT_EVIDENCE_KEYS:
        missing = GIT_EVIDENCE_KEYS - set(git_evidence.keys())
        extra = set(git_evidence.keys()) - GIT_EVIDENCE_KEYS
        raise CheckpointProvenanceError(f"git_evidence keys mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}")

    commit = git_evidence["commit"]
    if not isinstance(commit, str) or len(commit) != 40 or not all(c in "0123456789abcdefABCDEF" for c in commit):
        raise CheckpointProvenanceError(f"Invalid Git commit hash in git_evidence: {commit!r}. Must be a 40-char hex string.")

    dirty = git_evidence["dirty"]
    if not isinstance(dirty, bool):
        raise CheckpointProvenanceError(f"Invalid dirty flag in git_evidence: {dirty!r}. Must be a strict boolean (not int).")

    tracked_diff = git_evidence["tracked_diff"]
    if not isinstance(tracked_diff, str):
        raise CheckpointProvenanceError(f"tracked_diff in git_evidence must be a string, got {type(tracked_diff).__name__}")

    untracked_sources = git_evidence["untracked_sources"]
    if not isinstance(untracked_sources, (list, tuple)):
        raise CheckpointProvenanceError(f"untracked_sources in git_evidence must be a sequence, got {type(untracked_sources).__name__}")

    norm_untracked: List[str] = []
    seen_untracked: Set[str] = set()
    for u_path in untracked_sources:
        if not isinstance(u_path, str) or not u_path.strip() or u_path.strip() != u_path:
            raise CheckpointProvenanceError(f"Invalid path in untracked_sources: {u_path!r}")
        norm_u = validate_allowlisted_relative_path(u_path)
        if norm_u in seen_untracked:
            raise CheckpointProvenanceError(f"Duplicate untracked source path: {norm_u!r}")
        seen_untracked.add(norm_u)
        if norm_u not in validated_files:
            raise CheckpointProvenanceError(
                f"Untracked source {norm_u!r} must have corresponding reconstructable bytes in source_files."
            )
        norm_untracked.append(norm_u)

    if not dirty:
        if tracked_diff != "":
            raise CheckpointProvenanceError("git_evidence claims clean tree (dirty=False) but contains non-empty tracked_diff.")
        if len(norm_untracked) > 0:
            raise CheckpointProvenanceError("git_evidence claims clean tree (dirty=False) but contains untracked_sources.")
    else:
        if tracked_diff == "" and len(norm_untracked) == 0:
            raise CheckpointProvenanceError("git_evidence claims dirty tree (dirty=True) but diff and untracked_sources are both empty.")

    # 3. Preflight environment records BEFORE hashing
    if not isinstance(environment_records, (dict, Mapping)):
        raise CheckpointProvenanceError("environment_records must be a mapping.")
    for ek in environment_records.keys():
        if not isinstance(ek, str):
            raise CheckpointProvenanceError(f"environment_records keys must be strings, got {type(ek).__name__}: {ek!r}")
    if set(environment_records.keys()) != ENVIRONMENT_RECORD_KEYS:
        missing_env = ENVIRONMENT_RECORD_KEYS - set(environment_records.keys())
        extra_env = set(environment_records.keys()) - ENVIRONMENT_RECORD_KEYS
        raise CheckpointProvenanceError(f"environment_records keys mismatch. Missing: {sorted(missing_env)}, Extra: {sorted(extra_env)}")

    for str_field in ("python_version", "platform_system", "platform_release", "platform_machine", "torch_version"):
        val = environment_records[str_field]
        if not isinstance(val, str) or not val.strip():
            raise CheckpointProvenanceError(f"Environment field {str_field!r} cannot be empty or non-string: {val!r}")

    cpu_cnt = environment_records["cpu_count"]
    if isinstance(cpu_cnt, bool) or not isinstance(cpu_cnt, int) or cpu_cnt <= 0:
        raise CheckpointProvenanceError(f"cpu_count in environment_records must be a positive integer, got {cpu_cnt!r}")

    lib_vers = environment_records["library_versions"]
    if not isinstance(lib_vers, (dict, Mapping)) or not lib_vers:
        raise CheckpointProvenanceError("library_versions in environment_records must be a non-empty mapping.")
    for lib_k, lib_v in lib_vers.items():
        if not isinstance(lib_k, str) or not lib_k.strip() or not isinstance(lib_v, str) or not lib_v.strip():
            raise CheckpointProvenanceError(f"Invalid library version entry: {lib_k!r}: {lib_v!r}")

    # 4. NOW compute SHA256 hashes and tree hash
    file_records: Dict[str, Dict[str, Any]] = {}
    tree_hash_entries: List[Tuple[str, int, str]] = []
    for norm_path in sorted(validated_files.keys()):
        content_bytes = validated_files[norm_path]
        sz = len(content_bytes)
        sha = hashlib.sha256(content_bytes).hexdigest()
        file_records[norm_path] = {
            "path": norm_path,
            "size_bytes": sz,
            "sha256": sha,
            "content_bytes": content_bytes,
        }
        tree_hash_entries.append((norm_path, sz, sha))

    tree_canon_bytes = _canonical_json_bytes(tree_hash_entries, role="source tree hash")
    tree_hash = hashlib.sha256(tree_canon_bytes).hexdigest()


    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "total_source_bytes": total_bytes,
        "source_tree_hash": tree_hash,
        "source_files": file_records,
        "git": {
            "commit": commit,
            "dirty": dirty,
            "tracked_diff": tracked_diff,
            "untracked_sources": list(norm_untracked),
        },
        "environment": {
            "python_version": environment_records["python_version"],
            "platform_system": environment_records["platform_system"],
            "platform_release": environment_records["platform_release"],
            "platform_machine": environment_records["platform_machine"],
            "cpu_count": cpu_cnt,
            "torch_version": environment_records["torch_version"],
            "library_versions": dict(lib_vers),
        },
    }


def validate_source_provenance(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Validate source and environment provenance against trusted caller expectations.

    Preflights both actual and expected provenance objects against the strict schema
    and aggregate byte bounds with ZERO hashing. After preflight passes on both,
    recomputes file sizes and SHA-256 hashes from actual content bytes, detects tampering,
    and compares all fields against trusted caller expectations.
    """
    try:
        # Step 1: Preflight BOTH actual AND expected with ZERO hashing!
        _preflight_provenance_structure_and_bounds(actual, "actual")
        _preflight_provenance_structure_and_bounds(expected, "expected")

        # Step 2: Hashing and validation of actual
        act_files = actual["source_files"]
        for path, entry in act_files.items():
            actual_sha = hashlib.sha256(entry["content_bytes"]).hexdigest()
            if entry["sha256"] != actual_sha:
                raise CheckpointProvenanceError(f"actual declared sha256 {entry['sha256']} != recomputed {actual_sha} for {path}")

        act_tree_entries = [
            (path, act_files[path]["size_bytes"], act_files[path]["sha256"])
            for path in sorted(act_files.keys())
        ]
        act_tree_bytes = _canonical_json_bytes(act_tree_entries, role="actual source tree hash")
        recomputed_act_tree_hash = hashlib.sha256(act_tree_bytes).hexdigest()
        if actual["source_tree_hash"] != recomputed_act_tree_hash:
            raise CheckpointProvenanceError(
                f"actual source_tree_hash mismatch: declared {actual['source_tree_hash']} != recomputed {recomputed_act_tree_hash}"
            )

        # Step 3: Hashing and validation of expected
        exp_files = expected["source_files"]
        for path, entry in exp_files.items():
            exp_sha = hashlib.sha256(entry["content_bytes"]).hexdigest()
            if entry["sha256"] != exp_sha:
                raise CheckpointProvenanceError(f"expected declared sha256 {entry['sha256']} != recomputed {exp_sha} for {path}")

        exp_tree_entries = [
            (path, exp_files[path]["size_bytes"], exp_files[path]["sha256"])
            for path in sorted(exp_files.keys())
        ]
        exp_tree_bytes = _canonical_json_bytes(exp_tree_entries, role="expected source tree hash")
        recomputed_exp_tree_hash = hashlib.sha256(exp_tree_bytes).hexdigest()
        if expected["source_tree_hash"] != recomputed_exp_tree_hash:
            raise CheckpointProvenanceError(
                f"expected source_tree_hash mismatch: declared {expected['source_tree_hash']} != recomputed {recomputed_exp_tree_hash}"
            )

        # Step 4: Compare actual vs expected
        if actual["total_source_bytes"] != expected["total_source_bytes"]:
            raise CheckpointProvenanceError(
                f"total_source_bytes mismatch: actual={actual['total_source_bytes']} != expected={expected['total_source_bytes']}"
            )

        if actual["source_tree_hash"] != expected["source_tree_hash"]:
            raise CheckpointProvenanceError(
                f"source_tree_hash mismatch: actual={actual['source_tree_hash']} != expected={expected['source_tree_hash']}"
            )

        if set(act_files.keys()) != set(exp_files.keys()):
            missing_f = set(exp_files.keys()) - set(act_files.keys())
            extra_f = set(act_files.keys()) - set(exp_files.keys())
            raise CheckpointProvenanceError(f"Provenance source_files set mismatch. Missing: {sorted(missing_f)}, Extra: {sorted(extra_f)}")

        for path in sorted(exp_files.keys()):
            act_ent = act_files[path]
            exp_ent = exp_files[path]
            if act_ent["path"] != exp_ent["path"]:
                raise CheckpointProvenanceError(f"Path mismatch for {path}: actual={act_ent['path']}, expected={exp_ent['path']}")
            if act_ent["size_bytes"] != exp_ent["size_bytes"]:
                raise CheckpointProvenanceError(f"Size mismatch for {path}: actual={act_ent['size_bytes']}, expected={exp_ent['size_bytes']}")
            if act_ent["sha256"] != exp_ent["sha256"]:
                raise CheckpointProvenanceError(f"SHA-256 mismatch for {path}: actual={act_ent['sha256']}, expected={exp_ent['sha256']}")
            if act_ent["content_bytes"] != exp_ent["content_bytes"]:
                raise CheckpointProvenanceError(f"Content bytes mismatch for {path}")

        act_git = actual["git"]
        exp_git = expected["git"]
        for git_k in ("commit", "dirty", "tracked_diff", "untracked_sources"):
            if act_git[git_k] != exp_git[git_k]:
                raise CheckpointProvenanceError(f"Git evidence mismatch on {git_k}: actual={act_git[git_k]!r}, expected={exp_git[git_k]!r}")

        act_env = actual["environment"]
        exp_env = expected["environment"]
        for env_k in ("python_version", "platform_system", "platform_release", "platform_machine", "cpu_count", "torch_version"):
            if act_env[env_k] != exp_env[env_k]:
                raise CheckpointProvenanceError(f"Environment record mismatch on {env_k}: actual={act_env[env_k]!r}, expected={exp_env[env_k]!r}")

        if act_env["library_versions"] != exp_env["library_versions"]:
            raise CheckpointProvenanceError(
                f"library_versions mismatch: actual={act_env['library_versions']!r}, expected={exp_env['library_versions']!r}"
            )
    except CheckpointError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        raise CheckpointProvenanceError(f"Malformed provenance structure: {exc}") from exc
