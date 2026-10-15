# Milestone I1: Pure Checkpoint Identity and Provenance Contracts

## 1. Scope, Status, and Architectural Boundary

This document defines the pure canonical identity and source/environment provenance contracts for the Amazon4U recommendation benchmarking platform.

**Status:** Implementation complete for Deliverable 1; **pending Pi independent review and acceptance**. This deliverable does NOT constitute milestone acceptance, full harness completion, training authorization, or deployment approval (`DEPLOYMENT_NOT_AUTHORIZED`). Held-out evaluation is NOT_AUTHORIZED for anyone by this task; any later evaluation needs separately predeclared policy and explicit authorization.

Following the project's user-approved architectural decomposition (after exhausted task `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a` reached `ESCALATE`), checkpoint recovery is structured as exactly **three substantive deliverables**:
1. **Pure Identity and Provenance Contracts (This Deliverable)**: In-memory canonical identity representation, collision-free deterministic content fingerprinting, strict source and environment provenance contracts, and adversarial contract verification.
2. **Payload and State Validation (Deliverable 2, Unaccepted/Pending)**: Checkpoint payload integrity, PyTorch model weights-only CPU tensor validation, RNG state contracts, and optimizer parameter state consistency.
3. **Secure I/O and Integration (Deliverable 3, Unaccepted/Pending)**: Exclusive atomic no-clobber publication, secure provenance evidence acquisition adapters, and validation-only integration with the training harness (strictly prohibiting in-place replacement of completed artifacts or unvalidated live restoration).

### Pure Contract Boundary
This module ([`scripts/checkpoint_identity.py`](../scripts/checkpoint_identity.py)) is implemented strictly with the Python standard library. It:
- **Never** accesses the live filesystem, scans directories, or executes shell commands.
- **Never** interacts with Git subprocesses directly or reads credentials/datasets.
- **Never** allocates PyTorch tensors, runs optimizer updates, or performs model training.
- **Does not** provide live checkpoint capture, restoration, serialization, or file I/O APIs.
- Operates entirely on caller-supplied in-memory data structures and bytes.
- Neither grants nor implies authorization for model training, profiling, or deployment (`DEPLOYMENT_NOT_AUTHORIZED`). Held-out evaluation remains NOT_AUTHORIZED for anyone by this task.

---

## 2. Trust Boundary and Evidence Model

The contracts enforce a clear separation between **evidence acquisition** and **contract validation**:

```
+-------------------------------------------------------------+
|                     Caller / Runner                         |
|  - Acquires raw source files, Git commit, environment state |
|  - Supplies actual runtime records, graph, settings         |
|  - Supplies trusted expected identity & provenance objects  |
+-------------------------------------------------------------+
                              |
                              v  (Caller Evidence Trust Boundary)
+-------------------------------------------------------------+
|            scripts/checkpoint_identity.py (Pure)            |
|  - Validates schema, types, allowlists, and boundaries       |
|  - Recomputes fingerprints from retained actual content     |
|  - Validates internal consistency (tamper detection)        |
|  - Verifies exact match against trusted expected objects    |
+-------------------------------------------------------------+
```

### Trust Guarantees and Boundary Scope
- **Caller Acquisition Trust Boundary**: The pure contracts **cannot prove the physical origin** of caller-supplied evidence; acquiring authentic source bytes, valid Git metadata, and environment records is the caller's documented responsibility. The contracts do not automatically inspect the filesystem, run Git subprocesses, verify physical execution hardware, or enforce runtime determinism.
- **Tamper Detection Scope**: Tamper detection is strictly scoped to **deterministic canonical recomputation** from retained content and comparison against an **independent, unchanged, trusted expected object**. There is **no universal guarantee** if a caller aliases or mutates both actual and expected evidence objects, nor can standard-library pure contracts intercept arbitrary, malicious in-memory Python object mutations across unverified references.
- **Preflighting and Domain Error Translation**: Both actual and expected objects are strictly preflighted: malformed, incomplete, or bypassed objects cannot evade validation. All corrupted objects raise domain errors (`CheckpointValidationError`, `CheckpointIdentityMismatchError`, `CheckpointProvenanceError`, or `CheckpointSecurityError`), preventing incidental `KeyError`, `TypeError`, or `AttributeError` escapes. Specifically, mapping keys for `positive_history` and `all_rated_history` in both actual and expected roles are explicitly preflighted for clean-text properties before any sorting, ensuring precise diagnostics (`actual.positive_history key`, `expected.positive_history key`) and avoiding incidental `TypeError` on mixed key types.
- **Malformed Unicode and Lone Surrogate Domain Behavior**: Literal lone-surrogate code points (U+D800..U+DFFF) in `CategoryPath` labels, brand labels, header fields, or settings text are strictly rejected in both validators and formatters with `CheckpointValidationError`. All canonical serialization maintains `ensure_ascii=False` without replacing code points or escaping aliases (e.g., `'c:["\u00e9"]'` is rejected in favor of literal compact UTF-8 `'c:["é"]'`). Canonical JSON encoding translates any underlying `UnicodeEncodeError` into `CheckpointValidationError`.
- **Unsupported Numeric Serialization and Overflow Handling**: Numbers in training records, edge weights, validation ratings, settings, and timestamps are bounded and verified for finite representation. Numbers such as $10^{400}$ (which has 401 decimal digits and overflows finite float conversion, but does not exceed 4300 digits) are checked during float validation to reject non-finite and overflowing values. Very large integers such as $10^{5000}$ (5001 decimal digits) exceed Python 3.11+'s default native 4300-digit string/JSON conversion setting (`sys.get_int_max_str_digits()`); `_safe_repr` truncates diagnostic printing (for values exceeding 14,000 bits) strictly for safe message generation without deciding validity, while canonical JSON serialization translates actual native conversion failures (`ValueError`, `OverflowError`) into `CheckpointValidationError`. There is no arbitrary 14,000-bit validity cap: native-supported exact integers (such as $2^{14000}$, which has 14,001 bits and 4,215 decimal digits, below the 4300-digit conversion limit) are fully retained without float conversion, and unsupported integers fail only when the current native converter cannot encode them, without altering global interpreter limits.
- **Safe Worker-Seed Mismatch Diagnostics**: In `_validate_resolved_settings_consistency`, cross-section consistency diagnostics between `determinism.worker_derivation.base_seed` and `determinism.seed` format both values using `_safe_repr`. When an oversized integer exceeding native string-conversion limits (such as $10^{5000}$ with 5001 decimal digits) appears in either field in an active positive-worker configuration, the mismatch diagnostic safely formats the value (e.g., `<oversized int (16610 bits)>`) without leaking native `ValueError` prior to constructing `CheckpointValidationError`. Regression coverage in `test_oversized_worker_seed_mismatch_diagnostics` verifies both uncoupled mismatch directions across `compute_settings_fingerprint`, `build_canonical_identity`, `CanonicalResumeIdentity.from_dict` (via in-memory mutated snapshot), and both `actual` and `expected` roles in `validate_canonical_identity`.


---

## 3. Public Canonical Schemas

### 3.1 Canonical Identity (`amazon4u-checkpoint-identity-v1`)

A [`CanonicalResumeIdentity`](../scripts/checkpoint_identity.py) object is the definitive representation of a training run's identity. It binds:
1. **Method, Category, Feedback Condition, and Seed**:
   - `method`: non-empty string without whitespace padding (e.g., `"LightGCN"`).
   - `category`: non-empty string matching the dataset scope (e.g., `"Musical_Instruments"`).
   - `feedback_condition`: strictly `"all_ratings"` or `"rating_ge_4"`.
   - `seed`: exact integer (strictly not boolean).
2. **Deterministic Content Fingerprints (SHA-256)**:
   - `data_fingerprint`: SHA-256 over training records, positive/all-rated histories, and condition warm pool.
   - `graph_fingerprint`: SHA-256 over typed graph nodes, edges, relations, and permitted product metadata.
   - `validation_fingerprint`: SHA-256 over validation selection inputs (targets and records).
   - `settings_fingerprint`: SHA-256 over resolved scientific training, execution, and model settings.
   - `identity_hash`: SHA-256 over the composite canonical identity payload.
3. **Retained Actual Canonical Content**:
   - `training_records`: tuple of sorted canonical record mappings `{"user_id": str, "parent_asin": str, "rating": float, "timestamp": int}`.
   - `positive_history`: mapping of `user_id` to sorted tuple of positive item IDs.
   - `all_rated_history`: mapping of `user_id` to sorted tuple of all-rated item IDs.
   - `warm_items`: sorted tuple of condition-warm item IDs.
   - `graph_nodes`: mapping of node type (`"User"`, `"Product"`, `"CategoryPath"`, `"Brand"`) to sorted tuple of typed node IDs.
   - `graph_edges`: tuple of sorted canonical edge mappings `{"src_type", "src_id", "relation", "dst_type", "dst_id", "weight"}`.
   - `product_metadata`: mapping of `product_id` to canonical metadata `{"categories": list(sorted_category_ids), "brand": Optional[str]}`.
   - `validation_inputs`: canonical validation inputs `{"targets": {user: list(sorted_items)}, "records": list(sorted_records)}`.
   - `resolved_settings`: mapping of the 6 mandatory resolved semantic sections.

### 3.2 Canonical Data Content (`amazon4u-canonical-data-v1`)
Enforces:
- Exact record format: `user_id` (str), `parent_asin` (str), `rating` (finite float, not bool), `timestamp` (exact integer, not bool/float).
- Duplicate `(user_id, parent_asin)` pairs are strictly rejected.
- Positive history is an exact subset of all-rated history.
- **Condition Positivity**:
  - In `"all_ratings"`: `positive_history` equals `all_rated_history`.
  - In `"rating_ge_4"`: an item is in `positive_history` if and only if its record rating is $\ge 4.0$.
- **Warm Pool Consistency**:
  - In `"all_ratings"`: `warm_items` equals the set of all observed items in training records.
  - In `"rating_ge_4"`: `warm_items` equals the set of all condition-positive items.

### 3.3 Canonical Graph Content (`amazon4u-canonical-graph-v1`)
Enforces:
- Declared node types must belong to `("User", "Product", "CategoryPath", "Brand")`.
- `CategoryPath` IDs must use canonical compact UTF-8 JSON prefix `c:[...]` (e.g., `'c:["Electronics","Headphones"]'`). Delimiter-separated strings (such as `'Electronics > Headphones'`) are rejected.
- `Brand` IDs must use canonical prefix `b:...` with non-blank label (e.g., `'b:Sony'`).
- Edge relations must match the allowlisted relation schema:
  - `"rated"`: User $\to$ Product
  - `"rated_by"`: Product $\to$ User
  - `"belongs_to"`: Product $\to$ CategoryPath
  - `"contains_product"`: CategoryPath $\to$ Product
  - `"has_brand"`: Product $\to$ Brand
  - `"brand_of"`: Brand $\to$ Product
  - `"category_child_of"`: CategoryPath $\to$ CategoryPath
  - `"category_parent_of"`: CategoryPath $\to$ CategoryPath
- All edge endpoints must exist in declared node sets. Edge weights must be finite numbers (not bool). Duplicate edges are rejected.
- **Product Metadata**:
  - Must be explicitly provided for all declared `Product` nodes (complete coverage, no missing products, no undeclared products).
  - Permitted fields: `"categories"` (sequence of canonical `CategoryPath` IDs) and `"brand"` (canonical `b:...` ID or explicit `None`).
  - Excluded/review-derived metadata fields (`average_rating`, `rating_number`, `bestseller_rank`, `store`, `Manufacturer`, `bought_together`, `details`) and any unknown fields are strictly rejected.

### 3.4 Canonical Validation Selection Inputs (`amazon4u-canonical-validation-v1`)
Enforces a strict complete selection-input schema:
- `validation_inputs` must contain `"targets"` and/or `"records"` (rejects unknown keys).
- `"targets"`: mapping of validation `user_id` to a sequence of target `parent_asin` strings. No duplicate target items per user.
- `"records"`: sequence of validation records. Each record must strictly have 4 fields: `(user_id, parent_asin, rating, timestamp)` as a 4-tuple or dict with those exact keys. No defaulting of rating or timestamp; no tuples with $> 4$ fields. Finite ratings (not bool), exact integer timestamps (not bool/float).
- **Agreement**: when both `"targets"` and `"records"` are supplied, they must agree completely: the set of users in `"targets"` must match `"records"`, and for every user, the set of target items must match the record items.

### 3.5 Resolved Semantic Settings (`amazon4u-canonical-settings-v1`)
Requires all **6 mandatory resolved sections**, recording full configuration with strict schemas without freezing real pending execution settings:
1. `"protocol"`:
   - `candidate_policy`: string (e.g., `"full_candidate_ranking"`)
   - `lifecycle_state`: string (e.g., `"IMPLEMENTATION_REGRESSION_VERIFICATION"`)
   - `protocol_version`: string (e.g., `"amazon4u-protocol-v2.0.0"`)
2. `"execution"`:
   - `cpu_threads`: positive integer (strictly not boolean, e.g., `4`), recording caller evidence of CPU thread limits.
   - `execution_device`: string (e.g., `"cpu"`)
   - `execution_label`: string (e.g., `"synthetic_contract_verification"`)
   - `runtime_mode`: string (e.g., `"benchmark_verification"`)
   - `runtime_workers`: non-negative integer (e.g., `0`)
   - `worker_settings`: EXACT record mapping `{"prefetch_factor", "persistent_workers", "multiprocessing_context", "worker_init_function"}`:
     - For `runtime_workers == 0`: strictly `prefetch_factor: None, persistent_workers: False, multiprocessing_context: None, worker_init_function: None`.
     - For `runtime_workers > 0`: strictly positive integer `prefetch_factor`, strict boolean `persistent_workers`, and non-empty unpadded strings for `multiprocessing_context` and `worker_init_function`.
3. `"model"`:
   - `architecture`: string (e.g., `"LightGCN"`)
   - `embedding_dim`: integer $\ge 1$ (e.g., `64`)
   - `initialization`: string (e.g., `"normal_0_0.1_xavier_uniform_gain_1"`)
   - `num_layers`: integer $\ge 0$ (e.g., `3`)
4. `"training"`:
   - `adam_betas`: sequence of 2 floats in $(0, 1)$ (e.g., `[0.9, 0.999]`)
   - `adam_epsilon`: strictly positive finite float (e.g., `1e-08`)
   - `batch_size`: integer $\ge 1$ (e.g., `128`)
   - `endpoint_regularization`: non-negative finite float (e.g., `1e-05`)
   - `learning_rate`: strictly positive finite float (e.g., `0.001`)
   - `loss_function`: string (e.g., `"BPR"`)
   - `max_epochs`: integer $\ge 1$ (e.g., `100`)
   - `optimizer`: string (e.g., `"Adam"`)
   - `patience`: integer $\ge 1$ (e.g., `10`)
   - `validation_frequency`: integer $\ge 1$ (e.g., `1`)
   - `weight_decay`: non-negative finite float (e.g., `0.0`)
5. `"sampling"`:
   - `negative_strategy`: string (e.g., `"uniform_eligible"`)
   - `negatives_per_positive`: integer $\ge 1$ (e.g., `1`)
   - `neighbor_fanouts`: sequence of non-negative integers (e.g., `[15, 10]`)
   - `pool_policy`: string (e.g., `"condition_warm_excluding_rated"`)
6. `"determinism"`:
   - `deterministic_algorithms`: strict boolean (e.g., `True`)
   - `reproducibility_version`: string (e.g., `"amazon4u-reproducibility-v1"`)
   - `rng_streams`: list/tuple of exactly 10 structured stream records for configured components (`Python`, `NumPy`, `PyTorch_CPU`, `PyTorch_CUDA_if_applicable`, `initialization`, `epoch_shuffle`, `negative_sampler`, `neighbor_sampler`, `DataLoader_generator`, `DataLoader_workers`). Each record contains:
     - `name`: string matching the registered component name. Missing, unknown, or duplicate names are rejected.
     - `applicable`: strict boolean.
     - If `applicable == True`: `algorithm` and `derivation` are non-empty unpadded strings, `seed` is a non-negative integer (strictly not boolean), and `reason` is strictly `None`.
     - If `applicable == False`: `algorithm`, `seed`, and `derivation` are strictly `None`, and `reason` is a non-empty unpadded string.
     List order represents semantic registration order; equivalent tuple/list carriers normalize equally, but permuting records alters the settings fingerprint.
   - `seed`: integer $\ge 0$ (e.g., `42`)
   - `worker_derivation`: EXACT record mapping `{"applicable", "algorithm", "base_seed", "derivation", "reason"}`:
     - `applicable`: strict boolean, required to agree with `(runtime_workers > 0)`.
     - If `applicable == True`: `algorithm` and `derivation` are non-empty unpadded strings, `base_seed` is a non-negative integer equal to `determinism.seed`, and `reason` is strictly `None`.
     - If `applicable == False`: `algorithm`, `base_seed`, and `derivation` are strictly `None`, and `reason` is a non-empty unpadded string.
     - `DataLoader_workers.applicable` must agree with `worker_derivation.applicable`.

Missing sections, unknown sections, missing fields, unknown fields, and invalid types are strictly rejected without silent defaulting.

---

## 4. Source and Environment Provenance (`amazon4u-provenance-v1`)

The provenance builder and validator guarantee exact, reconstructable source and environment capture.

### 4.1 Allowlisted Relative Paths
Only relative paths in approved top-level directories (`scripts/` and `configs/`) matching the registered allowlist are permitted:
- Allowlisted scripts: `scripts/checkpoint_identity.py`, `scripts/learned_models.py`, `scripts/recommendation_core.py`, `scripts/recommendation_evaluation.py`, `scripts/differentiable_models.py`, `scripts/build_kg.py`, `scripts/audit_kg_inputs.py`, `scripts/audit_candidates.py`, `scripts/validate_kg_splits.py`, `scripts/visualize_kg.py`, `scripts/inspect_data.py`, `scripts/download_5core.py`.
- Allowlisted configs: `configs/compute-plan.json`, `configs/dataset-scope.json`, `configs/evaluation-metrics.json`, `configs/kg-v1.json`, `configs/model-suite.json`, `configs/protocol-lifecycle.json`, `configs/reproducibility.json`, `configs/training-semantics.json`.

Strictly rejected:
- Absolute paths.
- Path traversal components (`..`, `.`, empty parts).
- Disallowed paths (`data/`, `.git/`, `.venv/`, `.test-output/`, `Movie4U`, `multiautoresearch`, `.worktrees`).
- Credential file patterns (`*.key`, `*.pem`, `*token*`, `*secret*`, `*.env*`, `*credential*`, `*password*`).

### 4.2 Bounded Source Bytes
- Total source bytes are capped at **2 MiB** (`MAX_PROVENANCE_SOURCE_BYTES = 2,097,152` bytes).
- Aggregate size and element types are preflighted **before** any hashing or byte processing to prevent memory or computation exhaustion.
- Reconstructable `content_bytes` are preserved alongside `size_bytes` and `sha256` for every allowlisted file.

### 4.3 Git Evidence
- `commit`: full 40-character hexadecimal Git commit hash.
- `dirty`: strict boolean (not integer).
- `tracked_diff`: diff string.
- `untracked_sources`: sequence of allowlisted relative paths for untracked files.
  - Every path in `untracked_sources` must be valid, unique, and present in `source_files` with reconstructable bytes.
- Consistency:
  - If `dirty == False`: `tracked_diff` must be `""` and `untracked_sources` must be empty.
  - If `dirty == True`: `tracked_diff != ""` or `len(untracked_sources) > 0`.
  - No fabricated clean defaults are permitted.

### 4.4 Environment Records
Requires complete, typed records:
- `python_version`, `platform_system`, `platform_release`, `platform_machine`, `torch_version`: non-empty strings.
- `cpu_count`: positive integer (strictly not boolean).
- `library_versions`: non-empty mapping of library name to version string.

---

## 5. Memory Model and Snapshot Isolation

### Shallow Freezing vs. Deep Snapshot Isolation
- `CanonicalResumeIdentity` is decorated with `@dataclass(frozen=True)`. This provides shallow immutability: top-level attributes cannot be reassigned.
- However, nested Python containers (`dict`, `list`) remain mutable if directly accessed.
- To prevent accidental mutation and aliasing:
  - `build_canonical_identity` stores canonical sorted tuples and deeply isolated copies of all inputs.
  - `CanonicalResumeIdentity.to_dict()` returns an isolated snapshot via deep copy.
   - Callers must **never assume unforgeable containers**; any canonical identity object must be verified with `validate_canonical_identity` before use. Post-mutation validation detects tampering even if performed on nested structures.

### Ordering Invariance and Ordered-List Exceptions
Collections of unordered entities are stored and compared in canonical sorted representation:
- Permuting input lists, tuples, history keys, graph nodes, edges, validation records, or product metadata produces identical canonical representations, identical fingerprints, and equal `CanonicalResumeIdentity` instances (`ident1 == ident2` and `validate_canonical_identity(ident1, ident2)` passes).
- **Ordered-List Exceptions**:
  - `sampling.neighbor_fanouts`: layer fanout order is strictly semantic (e.g., `[15, 10]` vs `[10, 15]`).
  - `determinism.rng_streams`: component stream registration sequence is strictly semantic.
  Permuting elements in these ordered sequences alters the canonical settings representation and changes `settings_fingerprint`. Equivalent tuple vs. list carriers normalize equally.

---

## 6. Public API Summary

| Function / Class | Description |
| :--- | :--- |
| [`CanonicalResumeIdentity`](../scripts/checkpoint_identity.py) | Shallowly frozen container retaining canonical content and fingerprints. |
| [`build_canonical_identity`](../scripts/checkpoint_identity.py) | Constructs and validates `CanonicalResumeIdentity` from actual records. |
| [`validate_canonical_identity`](../scripts/checkpoint_identity.py) | Recomputes fingerprints from actual content and checks match against expected. |
| [`compute_data_fingerprint`](../scripts/checkpoint_identity.py) | Computes deterministic SHA-256 for training records, histories, warm pool. |
| [`compute_graph_fingerprint`](../scripts/checkpoint_identity.py) | Computes deterministic SHA-256 for graph nodes, edges, relations, metadata. |
| [`compute_validation_fingerprint`](../scripts/checkpoint_identity.py) | Computes deterministic SHA-256 for validation targets and records. |
| [`compute_settings_fingerprint`](../scripts/checkpoint_identity.py) | Computes deterministic SHA-256 for the 6 resolved settings sections. |
| [`build_source_provenance`](../scripts/checkpoint_identity.py) | Builds reconstructable provenance from allowlisted source bytes, Git, and env. |
| [`validate_source_provenance`](../scripts/checkpoint_identity.py) | Preflights and verifies provenance records against expected provenance. |
| [`validate_category_id`](../scripts/checkpoint_identity.py) | Validates canonical CategoryPath ID (`c:[...]`). |
| [`format_canonical_category_id`](../scripts/checkpoint_identity.py) | Formats sequence of category labels into `c:[...]`. |
| [`validate_brand_id`](../scripts/checkpoint_identity.py) | Validates canonical Brand ID (`b:...`). |
| [`format_canonical_brand_id`](../scripts/checkpoint_identity.py) | Formats brand label into `b:...`. |
| [`validate_allowlisted_relative_path`](../scripts/checkpoint_identity.py) | Validates path against security rules and allowlist. |

---

## 7. Next Steps in Milestone I1 Architectural Decomposition

As approved in the architectural decomposition:
1. **Deliverable 1 (Current Deliverable)**: Pure identity and provenance contracts (implemented, verified with unit and adversarial tests in [`tests/test_checkpoint_identity.py`](../tests/test_checkpoint_identity.py), pending Pi independent review and acceptance).
2. **Deliverable 2 (Next Deliverable)**: Checkpoint payload schema, weights-only CPU state dict validation, RNG seed sequence contracts, and optimizer state verification.
3. **Deliverable 3 (Final Deliverable)**: Exclusive atomic no-clobber publication, secure filesystem acquisition adapters, and validation-only harness integration (strictly avoiding replacement of completed artifacts or live restoration).
