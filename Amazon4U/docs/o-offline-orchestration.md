# Amazon4U O-OFFLINE Orchestration and Packaging Architecture

## Expected O closure preflight correction

Production and generated context preflight now require exactly the 32 O closure
paths. Each physical descriptor has a plain integer size within 2 MiB and a hex
SHA-256, with an 8 MiB aggregate ceiling. Manifest schema, counts and digest must
agree with those descriptors. This occurs before runtime callbacks or hydration.

Two in-memory controls reproduced eight failing subcases before the change:
extra paths, boolean sizes, oversized files and oversized aggregate expectations
at both validators. Those controls and twelve nearby methods passed after the
change. Evidence is in `data/reports/o-offline-context-closure-tdd-v1/`. These
focused checks do not replace complete O-only or inherited full verification.

## Log output byte limits verified before final source freeze

The final review reproduced two OP09 output-cap violations with valid tiny fake
logs. Redaction expanded a seven-byte input to thirteen output bytes, and slicing
an unavailable diagnostic by characters retained fourteen UTF-8 bytes under a
seven-byte limit. Both new controls failed before the consumer change.

The consumer now caps redacted output in encoded bytes, preserves complete UTF-8
characters, marks capped successful streams partial, and retains primary failure
statuses for unavailable or error diagnostics. The two new controls and sixteen
nearby methods passed together without failures, errors or skips. Evidence is in
`data/reports/o-offline-log-cap-tdd-v1/`. This is focused evidence, not a full O or
H regression. Scientific code and the withdrawn hydration finding are unchanged.

## Generated hydration finding retracted after executable checks

The user requested a TDD correction for `G-HYDRATE-01`. The three new generated
hydration tests passed before any production edit. Examination of the actual
generated function confirmed that its no-output drain-loop break already exists.
The previous claim that this guard was missing was a reviewer error. The earlier
standard-library codec observation did not establish a defect in the generated
function. No duplicate production guard or timeout change was applied.

Three additive regression methods now exercise complete input, truncated input
and trailing data through the actual generated `streaming_hydrate` function.
The tests compile only that function from the generated source; they do not run
kernel main, the injected preflight template or H/model code. The truncated-input
control wraps the real tiny codec with a bounded no-progress safety fuse.
Valid bytes and rejection cleanup are checked directly.

All sixteen selected methods passed, including the thirteen nearby existing
controls. There were no failures, errors, skips, native-provider calls or guard
violations. Evidence is in `data/reports/o-offline-generated-hydration-tdd-v1/`.
The earlier hypothesis and planning evidence remain preserved, with an additive
correction in `data/reports/o-offline-verification-planning-v1/finding-correction-v1.json`.
Whole O-OFFLINE review and verification remain incomplete. Deployment remains
`NOT_AUTHORIZED`.

## Direct TDD correction of review findings

The user requested direct TDD repair of the review's four actionable findings.
Strict canonical specification and H authorization validation are restored from
reviewed native15 behavior. Unsupported wrappers are not relabeled, O transport
authority cannot satisfy H preflight, and `train_loss` and `learning_rate` remain
forbidden report keys. The inventory-binding and actual nested-specification and
report-section checks from the preceding repair are retained.

Seven new regression methods cover approved result aliases, alias substitution,
actual boolean specification fields, malformed report sections, unsupported
expected specifications, pre-callback refusal, and forbidden report-root metrics.
The red run observed ten failing subcases with no errors or skips. The first
green run passed all seven methods. Four nearby existing checks then exposed a
malformed-plan classification problem. Moving the existing plan-root check before
approval discovery fixed it without changing any existing test assertions.
The final bounded run passed all eleven selected methods with no failures,
errors, or skips. Native-provider guards recorded no calls or violations; each
fresh owned scratch directory was empty afterward.

Evidence and the rerunnable focused test launcher are in
`data/reports/o-offline-direct-tdd-v1/`. These are synthetic focused checks, not a
full O or H regression. The historical O-only slot was not used. Original reports,
terminal decisions, consumed budgets, inherited tests, accepted scientific code,
and the native-provider guard remain preserved. This correction does not grant
whole-package O-OFFLINE acceptance, main integration, production or GPU readiness,
or deployment. Deployment remains `NOT_AUTHORIZED`.

The authority and implementation narratives below are preserved historical records.

## Authority and Scope

This document records the targeted contract repair of Package O (Orchestration and Packaging) under the user-approved exceptional native contract repair of the SAME existing Amazon4U O-OFFLINE candidate, the 18 OP requirements, and Pi's independent adjudication of SDK findings F01–F10.

- **Implementation Status**: Source-only contract repair by Worker across the seven allowed Git-relative paths; all project unit tests and full suite regressions remain **STRICTLY UNEXECUTED BY WORKER**. Independent review and candidate benchmark discovery are reserved exclusively for Pi.
- **Task-Only Offline Scope**: Verified in candidate source against fake transport, simulated clocks (zero real sleeps), and synthetic fixtures only.
- **Deployment Decision**: `NOT_AUTHORIZED`. Offline fake-transport verification does NOT establish remote Kaggle API readiness, cloud credentials, GPU hardware capability, production bootstrap/data loading, durable storage retention, or deployment permission.
- **Scope Protection**: Exactly seven candidate paths are modified/created. All accepted H files (6 files), 31 scientific models, evaluator, D1–D3 components, 412 baseline tests (35 ledgers, 3564 expanded cases), and legacy unprofiled runners remain untouched and byte-identical.

---

## Contract Repairs and Adjudicated Decisions

### 1. ApprovedOfflineContext and Role Bindings Before Effects (F01, OP01)
`ApprovedOfflineContext` binds the operator profile, run spec, authorization receipt, and caller role before any filesystem, network, or transport effects occur. Enforces fail-closed validation: unapproved or mismatched contexts refuse immediately.

### 2. Pi Baseline Checkpoint Applicability Ruling (F03, OP11)
- **Canonical Deterministic Baselines** (`CorePopularity`, `KG-only`): Have zero epochs, zero update steps, and `selection=None`. Their checkpoint role is explicitly and logically satisfied by independent expected `not_applicable_deterministic_baseline` metadata. Physical checkpoint files or dummy bytes are neither required nor permitted.
- **Learned Methods** (`LightGCN`, `Heterogeneous KG+GNN`): Strictly require physical checkpoint files on disk and trusted `expected_checkpoint_sha256` matching caller expectations.

### 3. Canonical H Argument and Provenance Coherence (F02, OP05, OP13)
Canonical H spec transformation (`get_canonical_h_spec`) and authorization extraction (`get_canonical_h_authorization`) strictly preserve exact H callable and schema boundaries. Full provenance verification (`verify_h_provenance`) checks all 14 required H record keys, exact acceptance scope (`H-REF_CPU_reference_fixture_only`), and `NOT_AUTHORIZED` deployment decisions.

### 4. Manifest Separation: H28 vs O32 Outer Closure (F04, OP03)
H expects exactly 28 files (`H_BUNDLE_PATHS` / `BUNDLE_PATHS`) in its `expected_bundle`. O outer manifest contains 32 paths (`REQUIRED_CLOSURE_PATHS`) plus `generated_runtime` and `environment`. The kernel runner publishes the O outer manifest to `manifest.json` while maintaining H's `expected_bundle` distinct. In `render_plan`, `verify_source_manifest` evaluates the outer manifest after runtime and environment bindings are attached.

### 5. Attempt Reservation Before Push and No Auto-Repush (F06, OP07, OP12)
`Amazon4UStateManager.record_attempt` reserves the attempt state with status `"reserved"` before `transport.push` is called. If the push raises `SubmissionUnknownError`, the error is reported and automatic repushing is strictly prohibited to prevent duplicate submissions.

### 6. Bounded Log Reader and Honest Status (F08, OP09)
`BoundedLogReader` provides incremental, deadline-bounded reading of log chunks with strict byte capping. `consume_kernel_logs` redacts credentials before checking caps, handles `open_log_reader` transport interfaces, and reports honest `"available"`, `"partial"`, or `"unavailable"` statuses.

### 7. Fixed OP05 Test Fixtures (F07, OP05)
`test_op05_staged_failure_propagation` and `test_op05_injected_callable_exception_propagates` use `encode_bytes(b"# test\n")` (base64-encoded gzip with mtime=0) in `bundle_payload`, matching Stage 2 hydration requirements and allowing staged failures to reach callable, checkpoint, and publication stages.

---

## Architectural Design

### 1. Single Facade with Early Profile Dispatch (`kaggle_job.py`)

The existing runner (`multiautoresearch/post-training/scripts/kaggle_job.py`) serves as the single facade for both legacy post-training jobs and Amazon4U execution:
- **Early profile dispatch**: `kaggle_job.py` inspects `--profile` and configuration flags (`--spec`, `--run-spec`, `--approval`, `--plan`) BEFORE resolving credentials, CLI binaries, or mutable job state.
- **Subparser Default Handling**: Common profile options on subparsers use `default=argparse.SUPPRESS` so that specifying `--profile amazon4u` before a subcommand (e.g. `kaggle_job.py --profile amazon4u launch`) is not overwritten or erased by subparser defaults.
- **Fail-Closed Gate**: If Amazon-specific flags are passed without `--profile amazon4u`, execution refuses immediately with exit code 1 to prevent accidental legacy fallback or parameter misinterpretation.
- **Legacy Preservation**: When no profile is provided, frozen legacy behavior is preserved verbatim.

### 2. Stdlib-Only Orchestration Adapter (`amazon4u.py`)

Implemented exclusively with Python standard library modules (`argparse`, `base64`, `copy`, `ctypes`, `dataclasses`, `errno`, `hashlib`, `json`, `os`, `pathlib`, `re`, `shutil`, `sys`, `time`, `uuid`, `zlib`):
- **Pure In-Memory Render (`render_plan`)**: Deterministic generation of bundle files, kernel script, and metadata without disk IO, network calls, subprocesses, or credential queries. Requires independently supplied approved spec, authorization, expected sources, and source packets; rejects self-approval or synthesizing expectations.
- **Strict Dry-Run Semantics**: In `dispatch_amazon4u`, `--dry-run` performs pure in-memory plan rendering even when `--bundle-dir` is passed; bundle directory writing is an explicit, non-dry-run action into a fresh, non-existent path.
- **Complete Outer Closure Manifest (`build_source_manifest`, `verify_source_manifest`)**: Binds all 32 required closure paths (the 28 canonical H bundle paths plus 4 O runner and adapter modules). Separates source payload hashes from manifest digest to prevent self-referential hashing loops. Caller retains outer closure and environment expectations independently.
- **Bounded Streaming Hydration (`hydrate_packet`, `hydrate_bundle`, `streaming_hydrate`)**: Uses `zlib.decompressobj(16 + zlib.MAX_WBITS)` to stream-decompress chunks. Bounds allocation incrementally using remaining budget probes (`max_length`), checks strict base64 decoding (`validate=True`), enforces mandatory checksums and sizes, and drains pending buffer strictly via `dobj.decompress(b"", remaining)`. Unbounded `flush()` has been completely removed from the hardcap path. Refuses to overwrite existing files or traverse ancestor symlinks.
- **Atomic No-Clobber Move (`atomic_move_no_replace`)**: Uses Linux `renameat2` with `RENAME_NOREPLACE` (with `os.link` + `os.unlink` fallback on systems without renameat2), failing closed with `FileExistsError` without clobbering existing target files.
- **Safe Owned Temporary File Lifecycle**: Tracks explicit `created_temp` creation flags using `O_CREAT | O_EXCL` file creation, ensuring pre-existing foreign temporary files are never unlinked in `finally` blocks.
- **Exact Status State Machine (`poll_kernel_status`)**: Enforces registered exact status tokens (`queued`, `running`, `complete`, `error`, `cancelacknowledged`, `canceled`, `submission_unknown`, `poll_deadline_exceeded`). Strictly rejects substring matching (such as `"task could not complete"` being misidentified as complete). Injects simulated clocks (`FakeClock`) to avoid real `time.sleep` during tests.
- **Bounded Log Consumer (`consume_kernel_logs`)**: Redacts secret credentials (`token=`, `bearer `, `password`, `key=`) BEFORE saving or returning logs. Enforces hard byte caps so output never exceeds `max_bytes` even with truncation marker, safely handling tiny caps smaller than the marker text.
- **Paged Quarantine Collection (`collect_outputs`)**: Downloads outputs into a fresh, newly-owned quarantine directory across paged listings (`exist_ok=False`). Verifies owner UID, device, and inode before any cleanup, detects token cycles, checks per-chunk (64 KiB) and aggregate byte caps, validates required roles, and verifies crosslinks before promotion. Refuses if destination directory already exists.
- **Artifact Role & Loss Semantics (`validate_collected_artifacts`)**: Checks presence of required artifact roles. Missing required checkpoint role yields status `"unrecoverable"`; missing other roles yields status `"partial"`. Validates artifact versions against `expected_version`, compares method against `expected_spec`, and compares physical checkpoint SHA-256 against caller expectation.
- **Immutable State Management (`Amazon4UStateManager`)**: Stores run/attempt history under `run_id_attempt_id.json` keys using atomic exclusive creation (`O_CREAT | O_EXCL`), refusing collision or overwrite, and leaving legacy `LAST_JOB_PATH` untouched.
- **Typed Result Wrapping (`build_amazon4u_record`)**: Wraps H result in `amazon4u-O-record-v1` with explicit `evidence_kind: 'simulated_offline_fixture'`, preserving H keys intact and forbidding `eval_score` or `val_bpb` stand-in proxies. Validates crosslinks to plan spec, enforces exact integer types (`type(v) is int and not isinstance(v, bool)`) for epoch and steps, and verifies finite values in all 7 frozen registered metrics. Scientific stopping (`scientific_terminal`) represents legitimate scientific stopping (not error), categorized as `simulated_offline_success`.

### 3. Thin Kernel Runtime Adapter (`amazon4u_kernel.py`)

Executes inside the Kaggle environment (or simulated offline runtime):
- **Preflight Validation (Stage 1)**: Validates spec, authorization, runtime environment (CPU reference, CUDA prohibited), and run envelope crosslinks BEFORE creating directories or writing log files.
- **Mandatory Hydration (Stage 2)**: Hydrates bundle files; refuses if `bundle_files.json` is missing rather than silently skipping. When `already_hydrated=True`, strictly verifies regular file type, absence of symlinks/ancestor symlinks, exact file size, SHA-256 against `manifest.json`, and full closure BEFORE inserting `workdir` into `sys.path` or importing kernel stages.
- **Exact H Callable Handoff (Stage 3)**: Hands off exact 11 arguments matching `scripts.training_entrypoint:execute_fixture`: `spec`, `expected_spec`, `authorization`, `data`, `expectations`, `run_envelope`, `expected_run_envelope`, `bundle_sources`, `expected_bundle`, `light_initial`, `backend`. Catches callable exceptions and returns exit code 3.
- **Checkpoint Verification (Stage 4)**: Verifies checkpoint integrity with consumer check of expected metadata (SHA-256, run ID) and stages checkpoint via exclusive temporary file with `atomic_move_no_replace`.
- **Artifact Publication (Stage 5)**: Publishes verified artifacts to output directory.
- **Staged Failure Propagation**: Propagates primary nonzero exit codes (1 for preflight, 2 for hydration, 3 for callable, 4 for checkpoint, 5 for publish). No success is permitted from skipped stages. Controlled test fixtures provide independent valid fixtures per stage so that each named stage failure is reached and tested directly.

---

## OP01–OP18 Requirement and Control Matrix

| ID | Part / Basis | Implemented Positive Control | Implemented Negative Control | Guard Outcome | Test Location |
|---|---|---|---|---|---|
| **OP01** | Explicit profile & authority dispatch | `--profile amazon4u` before or after subcommand dispatches to O | Unknown profile, Amazon flags without profile, bool-as-int, invalid epoch budgets | Refuses before credentials/CLI/state access | `test_kaggle_orchestration.py` |
| **OP02** | Pure render & owned bundle write | `render_plan` returns identical in-memory bundle and digest | Tripwires on `Path.home`/env/network; non-empty target directory | Collision refused without deletion/mutation | `test_kaggle_orchestration.py` |
| **OP03** | Complete source closure & layout | Manifest incorporates 28 H paths + 4 O paths matching literal oracle | Missing H/O module, extra module, absolute path, traversal `..`, duplicate path | Path escapes and incomplete closure rejected | `test_amazon4u_bundle.py` |
| **OP04** | Incremental bounded hydration | Stream-decompress chunk-by-chunk with allocation budget probe | Decoded bytes > file limit, aggregate limit, declared size lie, truncated stream, trailing data, existing file overwrite | Aborts before overlimit allocation/write | `test_amazon4u_bundle.py` |
| **OP05** | H callable bridge & staged failure | Passes exact 11 arguments to callable; staged execution | Preflight, hydration, callable exception, checkpoint, publish failure; direct script no-op | Nonzero primary failure propagated; no entry call after preflight failure | `test_kaggle_orchestration.py` |
| **OP06** | Environment & capability gates | CPU profile and expected dependencies pass preflight | Requesting `cuda` or `gpu` in CPU reference scope | Fails closed before work; no driver probes | `test_kaggle_orchestration.py` |
| **OP07** | Submission acknowledgment & timeout | Push passes `-t` and accelerator; validates acknowledgment receipt | Push error with exit code 0; ambiguous timeout; mismatched owner/slug/version/plan_digest | Marks `submission_unknown`; no auto repush | `test_kaggle_orchestration.py` |
| **OP08** | Exact status state machine | `queued` -> `running` -> `complete` transitions with fake clock | Status containing "complete" inside error text; mismatched version in reply; deadline exceeded | No substring success; bounds poll count | `test_kaggle_orchestration.py` |
| **OP09** | Bounded credential-safe logs | Sensitive patterns (`token=`, `bearer `, `password`, `key=`) redacted | Log availability error vs job completion; version mismatch | Secrets redacted; log availability failure reported | `test_kaggle_orchestration.py` |
| **OP10** | Paged bounded collection | Multi-page collection into fresh quarantine chunk-by-chunk | Cycle in pagination tokens; chunk/file byte cap; existing destination directory | Aborts before writing past limits | `test_amazon4u_bundle.py` |
| **OP11** | Artifact completeness & crosslinks | Complete inventory (`plan`, `result`, `envelope`, `manifest`, `log`, `checkpoint`, `approval`) | Missing checkpoint role; missing non-checkpoint role; crosslink mismatch; forged result set | Missing checkpoint marked `unrecoverable`; missing non-checkpoint marked `partial`; forged/mismatched marked `error` | `test_amazon4u_bundle.py` |
| **OP12** | Immutable state preservation | Separate attempt files (`run_A_attempt_01.json`) via `O_CREAT | O_EXCL` | Overwriting existing attempt state file; path escapes in `run_id` or `attempt_id` | Overwrite refused; earlier attempts intact | `test_kaggle_orchestration.py` |
| **OP13** | Typed results boundary | Wraps H result in `amazon4u-O-record-v1`; `scientific_terminal` yields success record | `eval_score` or `val_bpb` stand-in proxies; fake-as-real claims; invented status | Preserves scientific H result verbatim; normal scientific stopping recorded as success | `test_kaggle_orchestration.py` |
| **OP14** | Legacy compatibility | Legacy parser defaults and choices preserved | Non-zero training exit code in generated legacy `run_cmd`; leakage of Amazon routing | Non-zero exit code propagates through generated legacy runner; legacy unprofiled | `test_kaggle_orchestration.py` |
| **OP15** | Honest durability & readiness | Readiness gates retained verbatim; failure status explicit | Claiming D3 fsync equals remote durability | Readiness gates retained; `NOT_AUTHORIZED` | `test_kaggle_orchestration.py` |
| **OP16** | Safe generated program tripwires | In-process execution of actual generated script with fake runtime/callable using separate fresh fixtures | Checkpoint verifier rejection in separate negative fixture; forbidden seams (subprocess, socket) tripwire-protected | Generated script takes positive and failure paths without workspace collision | `test_kaggle_orchestration.py` |
| **OP17** | Whole-suite evidence preservation | Retains all 412 baseline tests and schemas | Weakening assertions or editing baseline | Full suite evidence preserved | `test_kaggle_orchestration.py` |
| **OP18** | Source/runtime vs operational authority | Pure render and fake job flow work offline | Calling live launch without authorized transport | Live launch refused before side effects | `test_kaggle_orchestration.py` |

---

## Test Execution Safety Rules

1. **Zero Numeric Imports**: Neither `test_kaggle_orchestration.py` nor `test_amazon4u_bundle.py` imports `torch`, `numpy`, or model modules.
2. **File-Based Import for Runner**: `test_kaggle_orchestration.py` loads `kaggle_job.py` dynamically via `importlib.util.spec_from_file_location` to prevent invalid package namespace imports.
3. **Simulated Clock**: Polling and deadline loops utilize `FakeClock` with zero real `time.sleep` calls.
4. **Fresh Fixtures**: All file fixtures use `tempfile.TemporaryDirectory` under approved `TMPDIR`.
5. **Resource Bounds**: All simulated payloads and test fixtures adhere to <= 2 MiB per file, <= 8 MiB concurrent, <= 128 files, <= 8 mock pages, <= 5 fake polls, and <= 64 KiB chunks.
6. **Preserved Baseline**: The 412 baseline tests remain completely unchanged.

---

## Worker Implementation Status & Checkpoints (Source-Only Corrective 60m Task)

### Checkpoint 01: Initial Restoration & Baseline Audit (00m elapsed, ~2026-10-20T09:36:00Z)
- **Restoration**: Restored exact native7 candidate files into all seven allowed paths from `inputs/current-candidate/Amazon4U/...`; verified SHA-256 digests against `restored-candidate-sha256.json`.
- **Worker Execution Boundary**: Source-only AST/read/patch operations strictly active. Zero project/H/Torch/NumPy imports, zero test execution, zero runtime probes (no zlib/gzip/base64/parser semantic probes).
- **Known Remaining Implementation & Control Work**:
  1. Fix generated/runtime output ownership contradiction (N7-S01): unified single fresh-owned output lifecycle in `amazon4u.py` and `amazon4u_kernel.py`.
  2. Complete mandatory independent `ApprovedOfflineContext` / capabilities wiring across all callers (render, runner, dispatch, generated runtime, poll, log, collection) without fallback to unauthenticated/self-authorizing JSON (N7-S02).
  3. Complete trusted output inventory and canonical validation: require physical checkpoint SHA-256 for learned methods, explicit `not_applicable_deterministic_baseline` for canonical baselines, eliminate catch-all canonical-spec fallback bypasses (N7-S03).
  4. Update `tests/test_kaggle_orchestration.py` and `tests/test_amazon4u_bundle.py` to add independent oracle test sources, canonical positive fixtures with physical SHA, full canonical H spec fixtures, while preserving all 479 test IDs and 412 baseline tests (N7-S04).
- **Unexecuted Pi Gates**:
  - Independent Pi source AST/hash/export/ID metadata verification.
  - Pi O-only preflight test execution (requires separate authorization; unreleased).
  - Full suite regression (requires separate authorization; unreleased).
  - Deployment decision remains `NOT_AUTHORIZED`.

### Checkpoint 02: Progress Audit & Defect Root Causes Pinpointed (15m elapsed, ~2026-10-20T09:47:00Z)
- **Status & Progress**:
  - Analyzed N7-S01: Output directory collision pinpointed in `amazon4u.py` line 1041 (`outdir.mkdir(...)` in generated script) vs `amazon4u_kernel.py` line 195 (`outdir.mkdir(...)` in `run_kernel_stages` stage 1). Establishing single fresh-owned output lifecycle by having `run_kernel_stages` create `outdir` after stage 1 preflight passes, removing premature creation from generated script.
  - Analyzed N7-S02: Mandatory `ApprovedOfflineContext` / capabilities wiring. In `dispatch_amazon4u`, removing fallback to self-authorizing plan JSON; requiring independent `--context` or context object. Adding `context` parameter to `render_plan` and forwarding `context` and `capability_provider` through generated `main` into `run_kernel_stages`.
  - Analyzed N7-S03: Enforcing mandatory `expected_manifest` and `expected_output_inventory` in `validate_collected_artifacts`. Removing catch-all `except Exception:` fallback in canonical spec validation so full canonical H schema is enforced without raw-spec fallback bypass.
  - Analyzed N7-S04: Preparing fixture updates in `tests/test_amazon4u_bundle.py` and `tests/test_kaggle_orchestration.py` to provide full canonical H specs and mandatory physical checkpoint SHA-256 for learned methods, plus independent oracle test source.
- **Worker Execution Boundary**: Source-only AST/read/patch operations strictly maintained. Zero tests, zero imports, zero runtime probes.
- **Unexecuted Pi Gates**: Unchanged; all verification and execution reserved for Pi.

### Checkpoint 03: Implementation Refactoring Planned & Underway (26m elapsed, ~2026-10-20T09:59:00Z)
- **Status & Progress**:
  - `amazon4u.py`:
    1. Fixed N7-S01: Removed premature `outdir.mkdir` from generated script `main()`. `run_kernel_stages` owns `outdir` creation in stage 1 preflight.
    2. Fixed N7-S02: Updated generated `main` signature and call to accept and forward `context` and `capability_provider` to `run_kernel_stages`. Updated `render_plan` to accept `context: ApprovedOfflineContext | None = None` and bind canonical spec/auth/outer expectations. Updated `dispatch_amazon4u` to strictly require `--context` for Amazon4U profile, eliminating unauthenticated fallback.
    3. Fixed N7-S03: Enforced mandatory `expected_manifest` and `expected_output_inventory` in `validate_collected_artifacts`. Enforced strict baseline logical checkpoint role checking against inventory without bare-token bypass. Removed catch-all canonical spec validation bypass.
  - `amazon4u_kernel.py`:
    1. Wired capability checking in preflight to require matching fake capability dictionary across all 10 required keys (`package`, `version`, `nativeIO`, `ABI`, `dependency`, `CPU`, `runtime`, `resource`, `callable`, `bootstrap`).
    2. Retained stage 1 ownership of `outdir` creation.
  - `tests/test_kaggle_orchestration.py` and `tests/test_amazon4u_bundle.py`:
    1. Adding independent oracle test source `_oracle_encode_bytes` (stdlib gzip + base64) for OP05 without production encoder self-assertion.
    2. Adding stage-entry and callcount assertions to OP05.
    3. Updating OP10/OP11 test fixtures to supply full canonical H specs, expected manifests, and expected output inventory with physical SHA-256 for learned methods and explicit `not_applicable_deterministic_baseline` for baselines.
    4. Updating OP16 in-process execution test with full capabilities bootstrap, clean `sys.path`/`sys.modules` restoration in `finally`.
    5. Preserving all 479 test IDs and 412 baseline tests byte-for-byte in outside modules.
- **Worker Execution Boundary**: Strict source-only mode. Zero test execution or probes.
- **Unexecuted Pi Gates**: Unchanged; all verification and execution reserved for Pi.

### Checkpoint 04: Implementation Complete & Static Verification Audited (41m elapsed, ~2026-10-20T10:14:00Z)
- **Status & Completed Corrections**:
  - **N7-S01 (Output Ownership Conflict)**: Resolved. In `amazon4u.py`, removed premature `outdir.mkdir` call from generated kernel script `main()`. Single fresh-owned lifecycle is maintained: `run_kernel_stages` in `amazon4u_kernel.py` performs preflight validation and exclusively creates `outdir` (with `exist_ok=False`) before any output writing.
  - **N7-S02 (Independent Context & Capabilities Wiring)**: Resolved.
    - Wired `ApprovedOfflineContext` through `dispatch_amazon4u` (strictly requiring `--context` for Amazon4U profile, refusing unauthenticated fallback), `render_plan` (binding canonical H spec, authorization, outer manifest, environment, timeout, and accelerator), and generated `main()` (accepting and forwarding `context` and `capability_provider` to `run_kernel_stages`).
    - Preflight capability checks in `run_kernel_stages` verify all 10 required keys (`package`, `version`, `nativeIO`, `ABI`, `dependency`, `CPU`, `runtime`, `resource`, `callable`, `bootstrap`) against expected capabilities before hydration or execution.
  - **N7-S03 (Trusted Output Inventory & Canonical H Spec Validation)**: Resolved.
    - In `validate_collected_artifacts`, enforced mandatory `expected_manifest` and `expected_output_inventory` (fail closed if missing).
    - Enforced mandatory `expected_checkpoint_sha256` matching physical file SHA-256 for learned methods (`LightGCN`, `HeteroGNN`).
    - Enforced explicit `not_applicable_deterministic_baseline` declaration in `expected_output_inventory` for canonical baselines (`CorePopularity`, `KGOnlyScorer`), refusing bare-token shortcuts or unverified missing files.
    - Removed catch-all exception bypass around canonical H spec verification, ensuring that full canonical H schemas are strictly enforced.
  - **N7-S04 (Test Suite & Independent Oracle Updates)**: Resolved.
    - Added independent stdlib oracle encoder `_oracle_encode_bytes` (gzip + base64) to `test_kaggle_orchestration.py` without production encoder self-assertions (addressing N7-P01).
    - Added call-count tracking to `test_op05_staged_failure_propagation` verifying 0 calls after preflight, hydration, and callable failure, and sequential calls on checkpoint/publish stages.
    - Added full 10-key capability dictionaries to `fake_bootstrap` fixtures in `test_op05` and `test_op16`.
    - Restored `sys.path` and isolated `sys.modules` in `test_op16`'s `finally` block.
    - Updated OP10/OP11 test fixtures in `test_amazon4u_bundle.py` with full canonical H specs, expected manifests, and expected output inventory with physical checkpoint SHA-256 for learned methods and explicit `not_applicable_deterministic_baseline` for baselines.
  - **Static & Preservation Audit Results**:
    - AST parsing passes cleanly (`ast.parse`) for all 6 Python files in allowed paths.
    - Test ID preservation: Exactly 479 test IDs found, 0 missing, 0 extra relative to `prior479-test-ids.json`.
    - Candidate file size limits: All 7 candidate files are <= 2 MiB (maximum candidate size is ~124 KiB).
    - Git modification isolation: Only the 7 allowed paths have changes; all 31 inherited Python files (and 412 baseline tests) are 100% preserved.
- **Worker Execution Boundary**:
  - Source-only mode strictly honored. Zero runtime probes executed. Zero test executions performed.
- **Implementation Status**:
  - `implementation_ready_for_Pi_review`.
- **Known Limitations & Unexecuted Pi Gates**:
  - Worker does not run unit, smoke, or regression tests; all test execution is reserved for Pi.
  - Independent Pi source AST/hash/export/metadata verification gate is unexecuted.
  - Pi O-only test execution gate is unexecuted (requires separate Pi authorization).
  - Pi full suite regression gate (479 tests) is unexecuted (requires separate Pi authorization).
  - Deployment decision remains strictly `NOT_AUTHORIZED`.






## Native9 Bounded Four-Category Correction (N8-S01..N8-S04)

### Checkpoint 01: Candidate Restored & Initial Audit (0m elapsed, ~2026-10-20T12:45:20Z)
- **Status & Restored Baseline**:
  - Restored candidate files from inputs/current-candidate/Amazon4U/ to the seven allowed paths.
  - Verified SHA-256 for all 7 candidate files against restored-candidate-sha256.json (all matched).
  - Clean bootstrap HEAD pre-O source preserved; native8 latest candidate is the starting baseline for this 60m corrective task.
- **Worker Execution Boundary**:
  - Strict source-only mode: zero runtime probes, no imports of project/framework/torch/H modules, no test execution.
  - Focus restricted exclusively to the approved four categories (N8-S01, N8-S02, N8-S03, N8-S04) and caller-callee compatibility audit.
- **Plan for Correction**:
  - N8-S01: Enforce mandatory retained context, distinct H28/O32 manifests, generated/environment expectations without permissive fallbacks.
  - N8-S02: Artifact inventory validation rejecting empty/partial dicts, enforcing roles/types/lengths/physical hashes/versions/approval links, and bound baseline N/A metadata.
  - N8-S03: Ensure mandatory preflight checks before filesystem mutations and before hydrated target imports/execution; update OP16 origin tests with tripwires and verified execution markers under isolated modules.
  - N8-S04: Fix missing-plan negative test to supply valid context and assert intended missing-plan failure stage.
  - Caller-callee compatibility audit: Verify and record compatibility across all definitions, callers, templates, mocks, and fixtures.

### Checkpoint 02: Design Finalized & Core Edits Prepared (12m elapsed, ~2026-10-20T12:56:30Z)
- **Status & Progress**:
  - Detailed root-cause analysis completed for all four approved categories (N8-S01..N8-S04) and caller-callee compatibility.
  - N8-S01: validate_approved_offline_context updated to deep-copy context, enforce distinct H28 (H_BUNDLE_PATHS) vs O32 (REQUIRED_CLOSURE_PATHS) manifests, complete 10-key capabilities, and full schemas; render_plan strictly requires non-None context and unconditionally verifies generated_runtime and environment expectations.
  - N8-S02: validate_collected_artifacts updated to reject partial/empty inventories, enforce full approval equality (not just matching run_id), require mandatory integer version matching, verify manifest with verify_source_manifest, treat size/length with explicit key/type checks (allowing 0), and enforce bound baseline N/A metadata vs physical checkpoint SHA-256 for learned methods.
  - N8-S03: build_amazon4u_kernel_script generated main() performs capability and preflight checks before any workdir creation or hydration or module imports; OP16 test in test_kaggle_orchestration.py installs tripwires before setup/render/write, isolates sys.modules before importing from workdir, uses valid hydrated O source, and verifies canonical origin and execution marker.
  - N8-S04: test_op01_launch_without_plan_refused passes valid context to isolate the intended missing-plan rejection before forbidden effects.
- **Worker Execution Boundary**:
  - Strict source-only mode honored. Zero test executions or runtime probes.

### Checkpoint 03: Implementation Applied & Test Fixtures Aligned (30m elapsed, ~2026-10-20T13:14:00Z)
- **Status & Completed Source Changes**:
  - `multiautoresearch/adapters/amazon4u.py`:
    - `validate_approved_offline_context`: Fails closed on `None`; deepcopies context to prevent caller mutation; verifies distinct H28 vs O32 closures; enforces `expected_h_envelope`, `expected_input_identity`, `expected_o_environment` (10 required capability keys), `submission_binding`, non-empty `expected_output_inventory`, and `limits`.
    - `render_plan`: Fails closed on `context is None`; unconditionally verifies `outer_manifest["generated_runtime"]` and `outer_manifest["environment"]` against independent expectations without permissive fallbacks.
    - `build_amazon4u_kernel_script`: Generated `main()` performs capability and preflight checks BEFORE `workdir.mkdir`, BEFORE hydration, and BEFORE importing runtime. Isolates `sys.modules` from `multiautoresearch` before importing from `workdir` and restores in `finally`.
    - `validate_collected_artifacts`: Enforces exact approval equality (`plan_auth == expected_approval`); requires mandatory integer versions in plan and result; validates manifest with `verify_source_manifest`; validates size/length with explicit key/type checks (allowing 0); validates physical SHA-256 (64 hex characters); enforces bound baseline N/A metadata vs physical checkpoint SHA for learned methods.
  - `multiautoresearch/adapters/amazon4u_kernel.py`:
    - Added module-level `_LATEST_EXECUTION_MARKER: dict[str, object] | None = None`.
    - In `run_kernel_stages`, records execution marker (`origin_workdir`, `origin_outdir`, `module_file`, `sha256`).
    - In artifact publication, sets `"version"` in `plan_summary` and `h_result` and attaches `"execution_marker"` to `amazon4u_record.json`.
  - `tests/test_kaggle_orchestration.py`:
    - Added `H_BUNDLE_PATHS` import and defined `sample_h_bundle_packets()` (28 paths).
    - Updated `sample_approved_context()` to supply distinct 28-path H manifest (`expected_h_bundle`), 32-path O outer manifest (`expected_o_outer` with generated runtime and environment), and full 6-role output inventory.
    - Updated `test_op01_launch_without_plan_refused`: supplies valid context and asserts missing-plan rejection cleanly (closing N8-S04).
    - Updated `test_op02` and `test_op06` tests to pass `context=ctx`.
    - Updated `test_op16_generated_fake_program_execution_and_tripwires`: installs tripwires before setup/render; uses valid hydrated O source from repo files; isolates `sys.modules`; checks canonical origin (`imported_mod.__file__` in `workdir_pos`) and execution marker; restores exact `sys.modules` and `sys.path` in `finally` (closing N8-S03).
  - `tests/test_amazon4u_bundle.py`:
    - Updated `test_op10_paged_bounded_collection_positive`: added `"version": 1` to plan and result, `"files": {}, "file_count": 0, "aggregate_size": 0` to `exp_man`.
    - Updated `test_op11_artifact_completeness_positive`: added `"version": 1` to plan, result, baseline plan, and baseline result; added `"files": {}, "file_count": 0, "aggregate_size": 0` to `exp_man`; updated baseline checkpoint inventory to bound logical N/A dict (`status="not_applicable_deterministic_baseline"`, `applicable=False`, `method="CorePopularity"`, `epoch=0`, `actual_Adam_steps=0`, `selection=None`).
    - Updated `test_op11_missing_required_role_marked_partial_negative`, `test_op11_mismatched_crosslink_negative`, `test_op11_internally_consistent_forged_set_refused`: added `"version": 1` and `"files": {}` to manifests.
- **Worker Execution Boundary**:
  - Strict source-only mode honored. Zero test executions or runtime probes.

### Checkpoint 04: Source & AST Audit Verified (45m elapsed, ~2026-10-20T13:25:00Z)
- **Status & Verification Benchmark**:
  - Code freeze achieved prior to 50m ceiling.
  - Candidate AST validation: All 6 Python candidate files parse with zero syntax errors via `ast.parse`.
  - Test ID inventory: Exactly 479 static test IDs confirmed matching `inputs/prior479-test-ids.json` (0 missing, 0 added).
  - Inherited baseline integrity: All 31 inherited Python files retain byte-identical SHA-256 hashes matching `plan.json['inherited_python_sha256']`.
  - Static exports: All declared exports in `multiautoresearch/adapters/__init__.py` resolve statically to definitions in `amazon4u.py` and `amazon4u_kernel.py`.
  - Allowed paths: Only the 7 authorized files modified or created. No changes outside allowed boundaries.
- **Implementation Status**: `implementation_ready_for_Pi_review`.
- **Known Limitations & Reserved Pi Authority**:
  - Implementation Worker is strictly source-only; no unit, smoke, or full regression tests have been executed by Worker.
  - Pi independent verification gates remain unexecuted (source AST/hash/export benchmark, O-only suite, 479-test full regression suite).
  - Deployment decision remains strictly `NOT_AUTHORIZED`.

---

## Mandatory Caller–Callee Source/AST Compatibility Audit Matrix

| Contract / Interface | Definition Location | Relevant Callers & Call Sites | Fixture Prerequisites & Provenance | Source / AST Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `validate_approved_offline_context(context, ...)` | `multiautoresearch/adapters/amazon4u.py:314` | `render_plan` (`amazon4u.py:882`), `test_kaggle_orchestration.py:sample_approved_context`, `test_op01`, `test_op02`, `test_op06`, `test_op16` | Retained context with `expected_h_bundle` (28 files), `expected_o_outer` (32 files), 10 environment capabilities, 6-role inventory, limits. | Deep-copies context; rejects None/missing keys; checks distinct H vs O sets; verifies capability keys statically. | COMPATIBLE / IMPLEMENTED |
| `render_plan(h_plan, outer_manifest, context)` | `multiautoresearch/adapters/amazon4u.py:870` | `test_kaggle_orchestration.py:test_op02`, `test_op06`, `test_op16` | Valid H-plan dict, 32-file outer manifest with `generated_runtime` and `environment`, valid context dict. | Rejects None context; validates outer manifest against context expectations unconditionally; returns `(o_plan, kernel_script)`. | COMPATIBLE / IMPLEMENTED |
| `validate_collected_artifacts(files, *, expected_version, expected_spec, expected_approval, ...)` | `multiautoresearch/adapters/amazon4u.py:2156` | `collect_outputs` (`amazon4u.py:2102`), `test_amazon4u_bundle.py:test_op11_*` (6 test methods) | `files` dict mapping names to Paths; `expected_version` int; `expected_spec` dict; `expected_approval` dict; `expected_manifest` with files dict; `expected_output_inventory` dict. | Exact approval equality (`plan_auth == expected_approval`); integer version validation in plan and result; manifest verified via `verify_source_manifest`; size/length validated with explicit key/type checks; 64-char hex physical SHA; bound baseline N/A dict. | COMPATIBLE / IMPLEMENTED |
| `collect_outputs(kernel_ref, expected_version, transport, destination_dir, ...)` | `multiautoresearch/adapters/amazon4u.py:1968` | `test_amazon4u_bundle.py:test_op10_*` (5 test methods) | `FakeKaggleTransport` with paged outputs; non-existing destination Path; complete expected spec, approval, manifest, inventory. | Quarantined download; byte counting; calls `validate_collected_artifacts`; atomically promotes via `_rename_no_replace_linux`; safe owner cleanup. | COMPATIBLE / IMPLEMENTED |
| Generated Kernel Preflight & Runtime Execution Isolation | `build_amazon4u_kernel_script` in `amazon4u.py:930` & `amazon4u_kernel.py` | Generated script `main()` executed in kernel environment; verified in `test_kaggle_orchestration.py:test_op16_*` | Valid 32-file closure in repo; `sample_approved_context()`; fake provider for capabilities. | Preflight & capability checks in generated script placed BEFORE `workdir.mkdir`, BEFORE hydration, and BEFORE target imports; `sys.modules` isolated before import; canonical origin and `_LATEST_EXECUTION_MARKER` verified; modules and path restored in `finally`. | COMPATIBLE / IMPLEMENTED |
| Missing-Plan Negative Fixture (`test_op01_launch_without_plan_refused`) | `tests/test_kaggle_orchestration.py:157` | Test suite runner (future Pi execution) | Valid `sample_approved_context()`, valid auth, missing plan. | Supplies complete context; asserts missing-plan rejection cleanly without weakening context guard. | COMPATIBLE / IMPLEMENTED |
| Deterministic Baseline Checkpoint N/A Receipt Control | `validate_collected_artifacts` & `test_op11_artifact_completeness_positive` | `collect_outputs`, artifact validation callers | Deterministic baseline spec (`CorePopularity` / `KG-only`), `epoch=0`, `actual_Adam_steps=0`, `selection=None`, bound N/A dict in inventory. | Enforces bound logical N/A metadata dict in inventory; rejects physical checkpoint files or physical SHA claims for baselines; validates roles receipt. | COMPATIBLE / IMPLEMENTED |



## Native10 Bounded Four-Category Correction (N8-S01..N8-S04)

### Checkpoint 01: Candidate Restored & Baseline Integrity Verified (0m elapsed, ~2026-10-20T14:14:24Z)
- **Status & Restored Baseline**:
  - Restored all 7 candidate files from `data/reports/o-offline-compat-verification-v1/inputs/current-candidate/Amazon4U/` to the seven authorized allowed paths.
  - Verified SHA-256 for all 7 candidate files against `restored-candidate-sha256.json` (all 7 matched exactly).
  - Pinned clean root `/home/sunf/FSB/Final/Amazon4U/.worktrees/o-offline-compat-correction-v1`; project workdir `Amazon4U/`.
  - Clean bootstrap HEAD pre-O source preserved; native9 candidate is the starting baseline for this 60m corrective task.
- **Worker Execution Boundary**:
  - Strict source-only mode: zero runtime probes, no imports or execution of repository/project Python modules (including Pi verification infrastructure / `data.reports`), no test execution.
  - Work bounded strictly to the four approved categories: N8-S01, N8-S02, N8-S03, N8-S04, and verified caller-callee compatibility audit.

### Checkpoint 02: Implementation Applied & Aligned Across Candidate Files (14m elapsed, ~2026-10-20T14:28:00Z)
- **Status & Completed Source Changes**:
  - **1/N8-S01 (Canonical H Bundle Manifest Schema & Oracle)**:
    - In `multiautoresearch/adapters/amazon4u.py` (`validate_approved_offline_context`), removed `size` demand from H manifest validation; strictly validated `schema_version == "amazon4u-H-bundle-v1"`, exactly 28 `H_BUNDLE_PATHS`, file entries `{bytes: int >= 0, sha256: 64-hex}`, and overall `sha256` 64-hex string. Supported reconstructed class identity across module reloads.
    - In `tests/test_kaggle_orchestration.py`, implemented `_oracle_owned_digest(files)` matching frozen H `owned_digest` using stdlib `hashlib` and `struct` (big-endian 64-bit int length prefixes + UTF-8 path bytes + physical sha256 hex bytes) without importing H or modifying frozen contracts.
    - In `tests/test_kaggle_orchestration.py`, implemented `sample_h_bundle_manifest(packets)` generating canonical `amazon4u-H-bundle-v1` manifest with `bytes` and `_oracle_owned_digest`, used across context fixtures.
  - **2/N8-S02 (Direct Inventory Fail-Closed & O-Owned Transport Version)**:
    - In `multiautoresearch/adapters/amazon4u.py` (`validate_collected_artifacts`), enforced upfront fail-closed validation of non-empty `expected_output_inventory` requiring all 6 roles (`plan`, `h_result`, `run_envelope`, `source_manifest`, `log`, `checkpoint`). Enforced complete logical N/A metadata dict for baselines (`status="not_applicable_deterministic_baseline"`, `applicable=False`, `method`, `epoch=0`, `actual_Adam_steps=0`, `selection=None`) without physical checkpoint file or SHA. Enforced physical checkpoint file and 64-hex SHA for learned methods.
    - In `multiautoresearch/adapters/amazon4u.py` (`validate_collected_artifacts`), validated transport integer `version` in `plan.json["version"]`; validated `result.json` against canonical 15-field `amazon4u-H-result-v1` without requiring or expecting `version` in H-result.
    - In `multiautoresearch/adapters/amazon4u_kernel.py` (Stage 5), removed `h_result["version"] = ver` mutation; published clean 15-field `h_result` to `result.json`; bound `ver` strictly in `plan_summary` and `o_record["version"]`; recorded `_LATEST_EXECUTION_MARKER`.
    - In `multiautoresearch/adapters/amazon4u.py` (`build_amazon4u_record`), bound `"version": plan.get("version")` into O-owned record without mutating `h_result`.
    - In `tests/test_amazon4u_bundle.py`, removed `"version": 1` from all mock H-results (`res_dict`, `res_base_dict`), preserving canonical 15-field H contract.
    - In `tests/test_amazon4u_bundle.py` (`test_op11_missing_required_role_marked_partial_negative`), updated `exp_inv` to contain complete 6-role inventory with valid metadata, preserving both assertions (`unrecoverable` and `partial`).
  - **3/N8-S03 (Immutability, Real Cache Pre-Isolation, Runtime Origin & Capabilities)**:
    - In `tests/test_kaggle_orchestration.py` (`test_op16_real_execution_and_cache_origin_isolation`), eliminated `ctx.expected_o_outer = exp_o_outer` assignment (zero frozen attribute assignments); updated `sample_approved_context` to accept `expected_o_outer` at construction.
    - In `tests/test_kaggle_orchestration.py`, updated `sample_caps` in `sample_approved_context` and `test_op05` to match retained canonical capability values (`simulated`, `frozen`, `reference`, `offline`, `bounded`, `execute_fixture`, `synthetic`).
    - In `tests/test_kaggle_orchestration.py` (`test_op16`), added pre-import cache isolation popping `multiautoresearch` modules from `sys.modules` before `kernel_main`; captured runtime origin during active execution via `mock_callable`; verified both `captured_origin` and persistent `amazon4u_record.json["execution_marker"]`; restored `sys.modules` and `sys.path` in `finally`.
  - **4/N8-S04 (Preserved Missing-Plan Negative Prerequisite Validity)**:
    - In `tests/test_kaggle_orchestration.py` (`test_op01_launch_without_plan_refused`), supplied valid `ApprovedOfflineContext` with canonical capabilities and H bundle manifest, while cleanly omitting plan file from bundle to isolate missing-plan rejection before forbidden filesystem effects.

### Checkpoint 03: Static Verification, Test ID Preservation & Audit Matrix (27m elapsed, ~2026-10-20T14:41:00Z)
- **Status & Verification Benchmark**:
  - Code freeze achieved at 27m, well before 50m ceiling.
  - Candidate AST validation: All 6 Python candidate files parse with zero syntax errors via `ast.parse`.
  - Frozen dataclass assignments: AST check confirms 0 unintended assignments to `ctx.*` or frozen dataclass instances.
  - Test ID inventory: Exactly 479 static test IDs confirmed matching `inputs/prior479-test-ids.json` (0 missing, 0 added).
  - Inherited baseline integrity: All 31 inherited Python files retain byte-identical SHA-256 hashes matching HEAD baseline.
  - Static exports: All declared exports in `multiautoresearch/adapters/__init__.py` resolve statically to definitions in `amazon4u.py` and `amazon4u_kernel.py`.
  - Allowed paths: Only the 7 authorized files modified or created. No changes outside allowed boundaries.
- **Implementation Status**: `implementation_ready_for_Pi_review`.
- **Known Limitations & Reserved Pi Authority**:
  - Implementation Worker is strictly source-only; no unit, smoke, or full regression tests have been executed by Worker (`tests_executed=[]`).
  - Independent Pi source AST/hash/export benchmark gate is unexecuted.
  - Pi O-only test execution gate is unexecuted (requires separate Pi authorization).
  - Pi full suite regression gate (479 tests) is unexecuted (requires separate Pi authorization).
  - Deployment decision remains strictly `NOT_AUTHORIZED`.

---

## Mandatory Caller–Callee Source/AST Compatibility Audit Matrix (Native10 Corrected)

| Contract / Interface | Definition Location | Signature & Return Type | Relevant Callers & Call Sites | Fixture Prerequisites & Provenance | Source / AST Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `validate_approved_offline_context(context)` | `multiautoresearch/adapters/amazon4u.py:314` | `(context: ApprovedOfflineContext \| dict[str, object]) -> ApprovedOfflineContext` | `render_plan` (`amazon4u.py:889`), `run_amazon4u_submission` (`amazon4u.py:1367`), `test_kaggle_orchestration.py:sample_approved_context`, `test_op01`, `test_op02`, `test_op06`, `test_op16` | Retained context with `expected_h_bundle` (28 files, `schema_version="amazon4u-H-bundle-v1"`, `bytes`, `sha256`), `expected_o_outer` (32 files, `amazon4u-source-manifest-v1`, `size`, `sha256`), 10 environment capabilities, 6-role inventory, limits. | Deep-copies context; rejects None/missing keys; checks distinct H vs O sets; verifies capability keys statically. Supports reconstructed class identity across module reloads. | COMPATIBLE / IMPLEMENTED |
| `render_plan(spec, *, authorization=None, expected_spec=None, expected_sources=None, source_packets=None, expected_auth=None, expected_runtime=None, expected_environment=None, context=None, owner="sunf", slug="amazon4u-offline-run", attempt_id="attempt_01", timeout=3600, accelerator="none")` | `multiautoresearch/adapters/amazon4u.py:848` | `(spec: dict[str, object], *, ...) -> RenderedPlan` (returns `RenderedPlan` dataclass with `spec`, `authorization`, `metadata`, `files`, `kernel_script`, `outer_manifest`, `plan_digest`, `owner`, `slug`, `attempt_id`, `rendered_at`) | `tests/test_kaggle_orchestration.py`: `test_op02_pure_render_reproducibility` (line 242), `test_op02_render_plan_missing_expectations_refused` (lines 271, 303, 311), `test_op06_complete_kernel_metadata_envelope` (line 411), `test_op07_timeout_and_accelerator_plumbing` (line 461), `test_op16_real_execution_and_cache_origin_isolation` (line 1250); `multiautoresearch/adapters/amazon4u.py:run_amazon4u_submission` (line 1400) | Valid `spec` dict, valid `authorization` dict, `expected_sources` matching 32-file outer manifest, valid non-None `context` (`ApprovedOfflineContext`). | Rejects None context; validates outer manifest against expected context expectations unconditionally; embeds outer manifest digest into generated kernel script; returns `RenderedPlan`. | COMPATIBLE / IMPLEMENTED |
| `validate_collected_artifacts(files, *, expected_version, expected_spec, expected_approval, expected_manifest, expected_checkpoint_sha256=None, expected_output_inventory=None)` | `multiautoresearch/adapters/amazon4u.py:2170` | `(files: dict[str, Path], *, ...) -> dict[str, object]` | `collect_outputs` (`amazon4u.py:2116`), `tests/test_amazon4u_bundle.py`: `test_op11_artifact_completeness_positive` (line 399), `test_op11_unexpected_file_unquarantined_negative` (line 587), `test_op11_corrupt_plan_digest_negative` (line 692), `test_op11_invalid_metadata_envelope_negative` (line 845), `test_op11_missing_required_role_marked_partial_negative` (line 934) | `files` dict mapping names to Paths; `expected_version` int; `expected_spec` dict; `expected_approval` dict; `expected_manifest` with files dict; `expected_output_inventory` requiring all 6 roles. | Fail-closed upfront validation: rejects None/empty/partial inventories; exact approval equality (`plan_auth == expected_approval`); integer version validation in plan and result; manifest verified via `verify_source_manifest`; size/length validated with explicit key/type checks; 64-char hex physical SHA for learned methods; complete logical N/A metadata dict bound for baselines. | COMPATIBLE / IMPLEMENTED |
| `collect_outputs(kernel_ref, expected_version, transport, destination_dir, *, expected_spec, expected_approval, expected_manifest, expected_checkpoint_sha256=None, expected_output_inventory=None, max_pages=8, max_file_bytes=2*1024*1024, max_aggregate_bytes=8*1024*1024, max_files=128, deadline_seconds=300.0, clock_fn=None)` | `multiautoresearch/adapters/amazon4u.py:1982` | `(kernel_ref, expected_version, transport, destination_dir, *, ...) -> dict[str, object]` | `multiautoresearch/adapters/amazon4u.py:run_amazon4u_submission`; `tests/test_amazon4u_bundle.py`: `test_op10_quarantined_output_staging_and_atomic_promotion` (line 120), `test_op10_missing_file_refused_atomic_rollback` (line 180), `test_op10_aggregate_byte_limit_enforced` (line 240), `test_op10_file_count_limit_enforced` (line 300), `test_op10_timeout_enforced` (line 350) | `FakeKaggleTransport` with paged outputs; non-existing destination Path; complete expected spec, approval, manifest, inventory. | Quarantined download; byte counting; calls `validate_collected_artifacts`; atomically promotes via `_rename_no_replace_linux`; safe owner cleanup on failure. | COMPATIBLE / IMPLEMENTED |
| `build_amazon4u_kernel_script(spec, auth, manifest_digest)` | `multiautoresearch/adapters/amazon4u.py:996` | `(spec: dict[str, object], auth: dict[str, object], manifest_digest: str) -> str` | `render_plan` (`amazon4u.py:910`), `tests/test_kaggle_orchestration.py:test_op06_complete_kernel_metadata_envelope` | Valid `spec`, `auth`, and 64-hex `manifest_digest`. | Generates standalone kernel script. Generated `main()` includes early capability preflight checks placed BEFORE `workdir.mkdir`, BEFORE hydration, and BEFORE importing targets. Pre-isolates `sys.modules` from `multiautoresearch` before importing from `workdir` and restores in `finally`. | COMPATIBLE / IMPLEMENTED |
| `run_kernel_stages(workdir, outdir, spec, auth, *, h_callable=None, fake_bootstrap=False, fail_stage=None, checkpoint_verifier=None, already_hydrated=False, context=None, capability_provider=None)` | `multiautoresearch/adapters/amazon4u_kernel.py:77` | `(workdir: Path, outdir: Path, spec: dict[str, object], auth: dict[str, object], *, ...) -> int` | Generated script `main()`; `tests/test_kaggle_orchestration.py:test_op16_real_execution_and_cache_origin_isolation` | Valid paths, `spec`, `auth`, callable, context, capability provider. | Executes 5 kernel stages. In Stage 4, executes H callable and receives pure 15-field H result without mutating it. In Stage 5, writes clean 15-field `h_result` to `result.json`, binds transport version in `plan.json` and `amazon4u_record.json`, and records `_LATEST_EXECUTION_MARKER`. | COMPATIBLE / IMPLEMENTED |
| `build_amazon4u_record(plan, h_result, *, evidence_kind="reference_run", population_metadata=None)` | `multiautoresearch/adapters/amazon4u.py:2559` | `(plan: dict[str, object], h_result: dict[str, object], *, ...) -> dict[str, object]` | `multiautoresearch/adapters/amazon4u.py:run_amazon4u_submission` (line 1445), `multiautoresearch/adapters/amazon4u_kernel.py:run_kernel_stages` Stage 5 (line 218), `tests/test_amazon4u_bundle.py` | Valid `plan` dict, valid 15-field `h_result` dict. | Builds O audit record. Binds `"version": plan.get("version")` into O-owned record without modifying `h_result`. | COMPATIBLE / IMPLEMENTED |
| `_oracle_owned_digest(files)` & `sample_h_bundle_manifest(packets)` | `tests/test_kaggle_orchestration.py:92-126` | `(files: dict[str, dict[str, object]]) -> str`, `(packets: dict[str, bytes] \| None = None) -> dict[str, object]` | `tests/test_kaggle_orchestration.py:sample_approved_context` | 28 H bundle files or packets. | Independent stdlib oracle matching frozen H `owned_digest` using big-endian 64-bit int length prefixes + UTF-8 path bytes + physical sha256 hex bytes without importing H or modifying frozen H contracts. Generates canonical `amazon4u-H-bundle-v1` manifest. | COMPATIBLE / IMPLEMENTED |
| Missing-Plan Negative Fixture (`test_op01_launch_without_plan_refused`) | `tests/test_kaggle_orchestration.py:169` | `() -> None` | Unittest suite runner (Pi execution) | Valid `sample_approved_context()`, valid auth, bundle omitting plan file. | Supplies complete context with canonical capabilities and H manifest; asserts missing-plan rejection cleanly without weakening context guard. | COMPATIBLE / IMPLEMENTED |
| Deterministic Baseline vs Learned Checkpoint Verification | `multiautoresearch/adapters/amazon4u.py:2300-2340` (inside `validate_collected_artifacts`) | inline verification logic | `collect_outputs`, `tests/test_amazon4u_bundle.py:test_op11_*` | Deterministic baseline spec (`CorePopularity` / `KG-only`) vs learned method spec (`LightGCN` / `HeterogeneousKGNN`). | For deterministic baselines, requires bound logical N/A metadata dict (`status="not_applicable_deterministic_baseline"`, `applicable=False`, `method`, `epoch=0`, `actual_Adam_steps=0`, `selection=None`) and rejects physical checkpoint files or physical SHA. For learned methods, requires valid physical checkpoint file and 64-hex SHA matching expected checkpoint SHA. | COMPATIBLE / IMPLEMENTED |

---

## Native11 Bounded Targeted Recovery (N10-S01 and N10-S02)

### Checkpoint 01: Candidate Restored & Root Causes Pinpointed (00m elapsed, ~2026-10-20T15:28:00Z)
- **Status & Restored Baseline**:
  - Restored all 7 candidate files from `data/reports/o-offline-targeted-verification-v1/inputs/current-candidate/Amazon4U/` to the seven authorized allowed paths.
  - Verified SHA-256 for all 7 candidate files against `restored-candidate-sha256.json` (all 7 matched bit-for-bit).
  - Pinned clean root `/home/sunf/FSB/Final/Amazon4U/.worktrees/o-offline-targeted-runtime-recovery-v1`; project workdir `Amazon4U/`.
  - Clean bootstrap HEAD pre-O source preserved; native10 candidate is the starting baseline for this 60m corrective task.
- **Worker Execution Boundary**:
  - Strict source-only mode: zero runtime probes, no imports or execution of repository/project Python modules (including Pi verification infrastructure / `data.reports`), no test execution (`tests_executed=[]`).
  - Work bounded strictly to the two approved root-cause categories: N10-S01, N10-S02, and directly affected caller/fixture prerequisites.

### Checkpoint 02: Implementation Applied & Aligned Across Candidate Files (15m elapsed, ~2026-10-20T15:43:00Z)
- **Status & Completed Source Changes**:
  - **1/N10-S01 (Canonical H Expectation Handoff & Coherence Guard)**:
    - In `multiautoresearch/adapters/amazon4u_kernel.py`:
      - Stage 1 Preflight: Required independent `ApprovedOfflineContext` (`if context is None: return 1`), eliminating `context=None` bypass.
      - Reject inconsistent bootstrap expectations before effects: compared `fake_bootstrap["expected_bundle"]` against `valid_ctx.expected_h_bundle` and refused with exit code 1 on mismatch; verified `expected_run_envelope` against `valid_ctx.expected_h_envelope`; verified run envelope `bundle_sha256` against `valid_ctx.expected_h_bundle["sha256"]`; verified `bundle_sources` coherence against `valid_ctx.expected_h_bundle["files"]`.
      - Stage 3 Callable Handoff: Bound `expected_bundle` strictly to `valid_ctx.expected_h_bundle` (canonical H28 manifest: `schema_version="amazon4u-H-bundle-v1"`, `bytes`, `sha256`), along with `expected_spec=valid_ctx.canonical_h_spec`, `authorization=valid_ctx.h_authorization`, and `expected_run_envelope=valid_ctx.expected_h_envelope` forwarded to `adapt_h_callable`.
    - In `multiautoresearch/adapters/amazon4u.py` (generated script template `main()`):
      - In preflight, enforced consistency check rejecting `fake_bootstrap["expected_bundle"] != ctx_h_bundle` before effects.
    - In `tests/test_kaggle_orchestration.py`:
      - In `test_op16_generated_fake_program_execution_and_tripwires`: Updated `mock_bs` to provide canonical `ctx.expected_h_bundle` (H28) rather than `plan.outer_manifest` (O32), valid 28-file `bundle_sources`, coherent envelope `bundle_sha256`, and `version=1`.
      - In `mock_callable`: Asserted actual independently expected H28 schema (`amazon4u-H-bundle-v1`), exactly 28 files with `bytes` and `sha256`, matching `ctx.expected_h_bundle["sha256"]`.
      - Added negative subcase in `test_op16` proving that supplying `plan.outer_manifest` (O32) as `expected_bundle` is rejected before effects with exit code 1 and no output directory created.
      - In `test_op05_h_callable_bridge_exact_arguments_positive`: Provided `ctx.expected_h_bundle` and asserted received `expected_bundle == ctx.expected_h_bundle`.
  - **2/N10-S02 (Truthful Negative-Stage Controls & Prerequisites)**:
    - In `tests/test_kaggle_orchestration.py` (`test_op05_staged_failure_propagation`):
      - Stage 3: Separated into 3a (explicit `fail_stage="callable"`) returning 3, and 3b (precise stage 3 invalid-H-result negative control: callable returns `bad_h_result` with forbidden proxy metric `eval_score=0.99`, rejected by `verify_h_provenance` at Stage 3, returning exit code 3). Asserted callable was invoked, but Stage 4 checkpoint was NOT written (`checkpoint.bin` absent) and Stage 5 publication was NOT reached (`result.json`, `plan.json`, `amazon4u_record.json` absent).
      - Stage 4: Preserved 4a (`fail_stage="checkpoint"`) and 4b (checkpoint verifier rejection) returning exit code 4, asserting `checkpoint.bin` is not written.
      - Stage 5: Implemented distinct meaningful Stage 5 publication failure control with valid preceding H output (`good_callable`), valid checkpoint data and verifier (`checkpoint_verifier=lambda b: True`), valid version (`version=1`), valid context (`context=ctx`), and valid expectations (`fake_bootstrap=bs`). Used explicit fake publication-error seam `fail_stage="publish"`, returning exit code 5. Asserted preceding stages succeeded (`checkpoint.bin` was verified and written) and publication artifacts were NOT published.
      - Updated all earlier-stage fixtures in `test_op05_staged_failure_propagation`, `test_op05_injected_callable_exception_propagates`, and `test_op06_preflight_refuses_gpu_expected_runtime` with valid canonical context (`context=ctx`), version, and canonical bundle expectations.
      - Preserved valid missing-plan negative (`test_op01_launch_without_plan_refused`) with exact error message check.

### Checkpoint 03: Source/AST Verification & Test ID Audit (25m elapsed, ~2026-10-20T15:53:00Z)
- **Status & Verification Benchmark**:
  - Code freeze achieved well before the 50m ceiling.
  - Candidate AST validation: All 6 Python candidate files parse with zero syntax errors via `ast.parse`.
  - Test ID inventory: Exactly 479 static test IDs confirmed (42 in `test_kaggle_orchestration.py`, 25 in `test_amazon4u_bundle.py`, 412 inherited baseline). Exactly 0 missing, 0 added.
  - Inherited baseline integrity: All 31 inherited Python files retain byte-identical SHA-256 hashes matching HEAD baseline.
  - Allowed paths: Only the 7 authorized files modified or created. Zero changes outside allowed boundaries.
- **Implementation Status**: `implementation_ready_for_Pi_review`.
- **Known Limitations & Reserved Pi Authority**:
  - Implementation Worker is strictly source-only; zero unit, smoke, or full regression tests have been executed by Worker (`tests_executed=[]`).
  - Independent Pi source AST/hash/export benchmark gate is unexecuted.
  - Independent Pi O-only test execution gate is unexecuted (reserved for Pi after source safety review).
  - Pi full suite regression gate (479 tests) is unexecuted.
  - Deployment decision remains strictly `NOT_AUTHORIZED`.

---

## Mandatory Caller–Callee Source/AST Compatibility Audit Matrix (Native11 Corrected)

| Contract / Interface | Definition Location | Signature & Return Type | Relevant Callers & Call Sites | Fixture Prerequisites & Provenance | Source / AST Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `run_kernel_stages(workdir, outdir, spec, auth, *, h_callable=None, fake_bootstrap=False, fail_stage=None, checkpoint_verifier=None, already_hydrated=False, context=None, capability_provider=None)` | `multiautoresearch/adapters/amazon4u_kernel.py:77` | `(workdir: Path, outdir: Path, spec: dict[str, object], auth: dict[str, object], *, ...) -> int` | Generated script `main()`; `tests/test_kaggle_orchestration.py`: `test_op05_staged_failure_propagation` (7 call sites), `test_op05_injected_callable_exception_propagates`, `test_op05_direct_script_no_op_refused`, `test_op06_preflight_refuses_gpu_expected_runtime`, `test_op16_generated_fake_program_execution_and_tripwires` (via `kernel_main`) | Valid paths, `spec`, `auth`, `context` (`ApprovedOfflineContext` with `expected_h_bundle` 28 files, `expected_o_outer` 32 files), `fake_bootstrap` matching canonical context. | In Stage 1, enforces mandatory `ApprovedOfflineContext` (`context=None` returns 1) and rejects inconsistent bootstrap expectations before effects. In Stage 3, binds `expected_bundle` to `valid_ctx.expected_h_bundle`. Rejects invalid H result provenance at Stage 3 returning 3 before checkpoint or publication. In Stage 5, checks version from context/bs, handles explicit fake publication seam `fail_stage="publish"` returning 5. | COMPATIBLE / IMPLEMENTED |
| `adapt_h_callable(callable_fn, spec, *, expected_spec, authorization, data, expectations, run_envelope, expected_run_envelope, bundle_sources, expected_bundle, light_initial=None, backend=None)` | `multiautoresearch/adapters/amazon4u_kernel.py:37` | `(callable_fn, spec, *, ...) -> dict[str, Any]` | `multiautoresearch/adapters/amazon4u_kernel.py:run_kernel_stages` Stage 3; `tests/test_kaggle_orchestration.py:test_op05_h_callable_bridge_exact_arguments_positive` | 1 positional + 10 keyword arguments matching `scripts.training_entrypoint:execute_fixture`. `expected_bundle` bound to canonical H28 manifest (`amazon4u-H-bundle-v1`, `bytes`, `sha256`). | Forwards exact 11 arguments. Fails closed if `spec.data_kind != "owned_synthetic_packet"`. Validated against `mock_h_callable` receiving exact canonical H28 manifest. | COMPATIBLE / IMPLEMENTED |
| `mock_callable in test_op16` | `tests/test_kaggle_orchestration.py:1305` | `(s, **kwargs) -> dict[str, object]` | `kernel_main` in `test_op16_generated_fake_program_execution_and_tripwires` | Runtime forwards canonical context expectations via `run_kernel_stages`. | Asserts `received_bundle["schema_version"] == "amazon4u-H-bundle-v1"`, `len(files) == 28`, files have `bytes` (int) and `sha256` (64-hex), digest matches `ctx.expected_h_bundle["sha256"]`, and `received_bundle == ctx.expected_h_bundle`. Captures execution origin. | COMPATIBLE / IMPLEMENTED |
| Stage 3 Invalid-H-Result Negative Fixture | `tests/test_kaggle_orchestration.py:734` | inline in `test_op05_staged_failure_propagation` | Unittest suite runner (future Pi execution) | Valid `sample_approved_context()`, valid bootstrap, `bad_h_result` with forbidden proxy metric `eval_score=0.99`. | `run_kernel_stages` returns 3 at Stage 3 provenance verification. Asserts `rc == 3`, checkpoint is NOT written, and publication artifacts do NOT exist. | COMPATIBLE / IMPLEMENTED |
| Stage 5 Publication Failure Control | `tests/test_kaggle_orchestration.py:768` | inline in `test_op05_staged_failure_propagation` | Unittest suite runner (future Pi execution) | Valid `sample_approved_context()`, valid bootstrap with `version=1`, valid H callable, valid checkpoint verifier, `fail_stage="publish"`. | `run_kernel_stages` returns 5 at Stage 5 publication entry. Asserts preceding callable succeeded, checkpoint WAS verified and written, but publication artifacts do NOT exist. | COMPATIBLE / IMPLEMENTED |
| Inconsistent Bootstrap Negative Fixture | `tests/test_kaggle_orchestration.py:1376` | inline in `test_op16_generated_fake_program_execution_and_tripwires` | Unittest suite runner (future Pi execution) | `mock_bs` with `expected_bundle = plan.outer_manifest` (O32) vs context retaining H28 manifest. | `kernel_main` returns 1 in preflight before effects. Asserts `rc == 1` and output directory is NOT created. | COMPATIBLE / IMPLEMENTED |
| `build_amazon4u_kernel_script(spec, auth, manifest_digest)` | `multiautoresearch/adapters/amazon4u.py:996` | `(spec: dict[str, object], auth: dict[str, object], manifest_digest: str) -> str` | `render_plan` (`amazon4u.py:910`), `tests/test_kaggle_orchestration.py:test_op06_complete_kernel_metadata_envelope` | Valid `spec`, `auth`, and 64-hex `manifest_digest`. | Generated script `main()` enforces preflight bootstrap consistency check against `context.expected_h_bundle` before effects. | COMPATIBLE / IMPLEMENTED |


---

## Native12 Exceptional Native Correction: H01 & H02 Canonical Metadata Contracts & Fixture Alignment

### 1. Root Cause & Problem Resolution Summary
Under latest user direction and Pi management, Native12 addresses the two source-confirmed metadata contract gaps identified in `Pi-H-boundary-review.json`:
- **H01**: Replaced the invented 3-field `amazon4u-H-envelope-v1` schema with the canonical 11-field `amazon4u-H-run-v1` run envelope required by frozen `scripts/training_harness.py check_envelope` and `execute_fixture`. All 11 keys (`schema_version`, `run_id`, `approval_id`, `profile_id`, `method`, `identity_hash`, `device`, `accelerator`, `bundle_sha256`, `initial_state_sha256`, `determinism_policy`) are strictly validated against non-empty string bounds (<= 256 chars), 64-character lowercase hexadecimal digests (`identity_hash`, `bundle_sha256`, `initial_state_sha256`), frozen profile (`H-REF-synthetic-cpu-v1`), and frozen determinism policy (`private_rng_bounded_cpu_no_global_setters`). Crosslinks against approved spec and semantic H28 manifest digest are enforced before any hydration, directory creation, or target module import.
- **H02**: Separated generic O transport authorization from independent canonical H authorization. In both host (`validate_approved_offline_context`) and generated preflight (`_retain_generated_context` inside `build_amazon4u_kernel_script`), `h_authorization` strictly requires `amazon4u-H-authorization-v1` with exact 5 keys (`schema_version`, `run_id`, `approval_id`, `scope`, `max_adam_steps`). Generic O authorization tags (`amazon4u-O-authorization-v1`) are refused at the H-owned context and handoff, and `get_canonical_h_authorization` refuses to silently relabel or manufacture H approval from O transport tags.
- **Metadata Retention**: `run_kernel_stages` and generated `main()` preflight snapshot and deeply copy the validated actual run envelope, expected envelope, H authorization, and H28 sources before provider callbacks. Stage 3 callable handoff and Stage 5 artifact publication use these retained values, ensuring protection against callback mutation.

### 2. Caller–Callee Audit Matrix (Native12)

| Contract / Interface | Definition Location | Signature & Return Type | Relevant Callers & Call Sites | Fixture Prerequisites & Provenance | Source / AST Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `validate_canonical_h_envelope(envelope, *, expected_spec=None, expected_bundle=None, expected_envelope=None)` | `multiautoresearch/adapters/amazon4u.py:365` | `(dict[str, object], *, ...) -> dict[str, object]` | `validate_approved_offline_context`, `multiautoresearch/adapters/amazon4u_kernel.py:run_kernel_stages` Stage 1 preflight | Envelope dictionary with exact 11 keys, `amazon4u-H-run-v1`, `H-REF-synthetic-cpu-v1`, `private_rng_bounded_cpu_no_global_setters`, 64-hex digests. | Validates exact 11 keys, bounded strings <= 256 chars, 64-char lowercase hex digests, and spec/bundle crosslinks. Returns deepcopy. | COMPATIBLE / IMPLEMENTED |
| `validate_approved_offline_context(context)` | `multiautoresearch/adapters/amazon4u.py:408` | `(ApprovedOfflineContext \| dict \| None) -> ApprovedOfflineContext` | `dispatch_amazon4u`, `run_kernel_stages`, unit test fixtures | Independent context object or dict with 10 fields. | Requires `h_authorization` schema `amazon4u-H-authorization-v1` (H02). Validates H28 manifest, then validates `expected_h_envelope` via `validate_canonical_h_envelope` (H01). Retains frozen dataclass. | COMPATIBLE / IMPLEMENTED |
| `get_canonical_h_authorization(auth)` | `multiautoresearch/adapters/amazon4u.py:823` | `(dict[str, object]) -> dict[str, object]` | Host dispatch and authorization verification callers | Authorization dict | Validates `amazon4u-H-authorization-v1`; refuses to manufacture or relabel O authorization tags. | COMPATIBLE / IMPLEMENTED |
| `_retain_generated_context(context)` in `_GENERATED_PREFLIGHT` | `multiautoresearch/adapters/amazon4u.py:1121` | `(context) -> ApprovedOfflineContext` | Generated kernel script `main()` | `context` passed to kernel runner | Strictly requires `a["schema_version"] == "amazon4u-H-authorization-v1"` (H02) and exact 11-field canonical envelope `amazon4u-H-run-v1` with 64-hex digests matching spec and bundle (H01). | COMPATIBLE / IMPLEMENTED |
| `run_kernel_stages(...)` | `multiautoresearch/adapters/amazon4u_kernel.py:78` | `(...) -> int` | Generated script `main()`; OP05, OP06, OP16 tests | Valid paths, `spec`, `auth`, `context`, `fake_bootstrap` | Stage 1 validates both `run_envelope` and `expected_run_envelope` against canonical metadata before effects; retains deepcopies. Stage 3 hands off retained envelope, expected envelope, and authorization to callable. Stage 5 publishes `retained_run_envelope` to `run_envelope.json`. | COMPATIBLE / IMPLEMENTED |
| `sample_canonical_h_envelope(spec=None, bundle_manifest=None)` | `tests/test_kaggle_orchestration.py:158` | `(...) -> dict[str, str]` | `sample_h_result`, `sample_approved_context`, test fixtures | Valid spec and bundle manifest | Supplies exact 11-field canonical `amazon4u-H-run-v1` envelope with 64-hex digests and spec crosslinks. | COMPATIBLE / IMPLEMENTED |
| Callback Mutation Test Control | `tests/test_kaggle_orchestration.py:1076` | inline in `test_op05_h_sources_preflight_and_retained_handoff` | Test suite runner | Valid context, bootstrap with provider mutating sources, envelope, and authorization | Asserts callable receives retained envelope, authority, and sources; publication writes retained envelope to `run_envelope.json`. | COMPATIBLE / IMPLEMENTED |
| OP16 Negative Controls Matrix | `tests/test_kaggle_orchestration.py:1580` | inline in `test_op16_generated_fake_program_execution_and_tripwires` | Test suite runner | Contexts and bootstraps with legacy envelope schema, missing envelope field, malformed digest, O authorization tag | Asserts `kernel_main` returns 1 in preflight before mkdir, write, log, hydration, target module import, or callable entry. | COMPATIBLE / IMPLEMENTED |

### 3. Static Test ID & File Integrity Verification
- **Test ID Count**: Exactly 480 test IDs retained (68 in O test files: 43 in `test_kaggle_orchestration.py`, 25 in `test_amazon4u_bundle.py`; 412 inherited baseline). Exactly 0 missing, 0 dropped, 0 renamed.
- **Inherited Baseline Integrity**: All 31 inherited Python files retain byte-identical SHA-256 hashes matching HEAD baseline.
- **Allowed Scope**: Only allowed candidate paths modified (`amazon4u.py`, `amazon4u_kernel.py`, `test_kaggle_orchestration.py`, `o-offline-orchestration.md`). `kaggle_job.py` and `adapters/__init__.py` restored byte-identically and preserved.
- **Worker Discipline**: Strict source-only; no unit tests, smoke tests, or full regressions executed (`tests_executed=[]`).
- **Deployment Status**: Strictly `NOT_AUTHORIZED`.


### Pi deterministic integration after native12 source review

Native12's generated caller was still late for actual bootstrap envelope checks and only shallow-copied nested bootstrap metadata;its source report and terminal review remain unchanged in the predecessor archive. This derived Pi-authored integration compares canonical spec,H authorization,actual/expected11-field envelopes,and H28 expectation against the independently retained context,then snapshots the presented metadata **before** any provider callback. Opaque data/expectations are not deep-copied or initialized. Existing post-provider CPU/runtime checks remain before hydration/imports. Canonical H spec/envelope/authorization ownership now matches frozen H's exact plain-dict/plain-string input types;generic O transport schemas remain distinct and unchanged.

Existing fake controls now prove malformed actual metadata never calls the provider/H entry/hydrator or creates/writes targets. The positive callback empties the original nested bootstrap spec/authority/envelopes/H28/source mappings and mutates the caller context;the H argument assertions and published envelope still have the original independent canonical values. These source changes are not a measured PASS. Fresh same-artifact SOURCE and the one carried globally guarded O-only child remain required;all ten original criteria/OP01–OP18/full verification remain mandatory. No H imports/updates,model calls,main promotion or deployment follows from this integration.

Canonical H CPU device metadata is exact `cpu`/`none`,matching read-only frozen `CPUBackend.require`;no normalization or empty-accelerator fallback. Existing generated fixture finally restores every original affected module object/key,includingNone sentinels. Fresh Pi source/O preservation also records strict Git/input-directory timestamps;the initial derived-repository baseline retains its original current-new-Git timestamp qualification. No old evidence/verifier changes.

---

## O-OFFLINE Collection Three Gaps Repair: Deadlines, Typed Artifacts, and No-Overwrite Promotion

### 1. Authority, Scope, and Worker Discipline

Under the latest user authorization ("Ok focus handle three collection gaps. When you finish handle 3 gaps report back to me. decision is your no need ESCALATE to me"), this phase implements exclusively the three assigned collection gaps in Package O:
1. **OP10 Deadlines**: Active deadline validation, safe clock sampling, parameter bounds with hard security ceilings, and multi-point cooperative checks across provider/iterator operations, between chunks/files/pages, and before promotion/completion.
2. **OP11 Typed Artifacts**: Complete frozen 15-field H result, canonical 11-field H run envelope, canonical H authorization, exact CPU capability validation, registered evaluation metrics validation, mandatory physical inventory sizes and 64-character lowercase hexadecimal SHA-256 validation, and physical checkpoint SHA verification directly against expected physical SHA.
3. **OP12 No-Overwrite Promotion**: Elimination of ordinary rename fallback in production; fail-closed no-replace promotion using `_rename_no_replace_linux` without `promotion_fn` bypass; test modeling via isolated test fixture helper in `test_amazon4u_bundle.py`.

Worker discipline remains strictly source-only:
- **Zero test execution**: `tests_executed = []`. No unit tests, integration tests, smoke tests, CLI tests, or compileall executed by Worker.
- **Allowed paths**: Edits restricted to the authorized candidate paths (`amazon4u.py`, `test_amazon4u_bundle.py`, `o-offline-orchestration.md`).
- **Byte-identical preservation**: Runner (`kaggle_job.py`), exports (`__init__.py`), kernel adapter (`amazon4u_kernel.py`), and Kaggle orchestration tests (`test_kaggle_orchestration.py`) remain byte-identical to the pinned restoration base (`inputs/restoration-base-sha256.json`).
- **Deployment Status**: Strictly `NOT_AUTHORIZED`.

### 2. Gap 1 Implementation: Deadlines, Safe Clock Sampling, and Bounded Parameters (OP10)

In `multiautoresearch/adapters/amazon4u.py` (`collect_outputs`):
- **Pre-Mutation Parameter Validation & Hard Ceilings**:
  - `deadline_seconds`: Validated as a positive, finite real number (`isinstance(deadline_seconds, (int, float)) and not isinstance(deadline_seconds, bool) and math.isfinite(deadline_seconds) and deadline_seconds > 0.0`). Rejects non-positive, NaN, infinity, or boolean types immediately before quarantine directory creation.
  - `clock_fn`: Validated as callable (`callable(clock_fn)`) or None (defaults to `time.monotonic`).
  - Hard security ceilings enforced before mutations:
    - `max_pages`: integer in `1..MAX_PAGES` (8)
    - `max_files`: integer in `1..MAX_CONCURRENT_FILES` (128)
    - `max_file_bytes`: integer in `1..MAX_FILE_BYTES` (2,097,152 = 2 MiB)
    - `max_aggregate_bytes`: integer in `1..MAX_AGGREGATE_BYTES` (8,388,608 = 8 MiB)
  - Chunk reads are bounded by `MAX_CHUNK_BYTES` (65,536 = 64 KiB).
- **Safe Clock Sampler (`_sample_clock`)**:
  - Enforces that all clock samples are finite non-bool floats (`isinstance(now, (int, float)) and not isinstance(now, bool) and math.isfinite(now)`).
  - If a custom clock returns NaN, Inf, a boolean, or raises an exception, collection halts immediately and fails closed with an error status (`"Clock returned non-finite or invalid sample"` or `"Clock function raised exception"`).
- **Pre-Callback Snapshot of Expected Metadata**:
  - Validates and deep-copies canonical expected spec, expected authorization, expected envelope, manifest, and output inventory *before* any provider listing callback.
  - Prohibits transport callbacks from mutating the caller's expectations or manufacturing expectations from actual downloaded files.
- **Multi-Point Cooperative Deadline Checks**:
  The deadline `end_time = start_time + deadline_seconds` is actively checked at every latency seam:
  1. Before creating the fresh quarantine directory.
  2. Before calling `transport.list_output_page`.
  3. Immediately after `transport.list_output_page` returns (late provider returns refuse writes and fail closed).
  4. Between item iterations in a page.
  5. Before item chunk iteration.
  6. Between chunk iterations.
  7. After chunk write.
  8. After file write.
  9. Between listing pages.
  10. Before calling `validate_collected_artifacts`.
  11. Before directory promotion (`Deadline exceeded (before_promotion)`).
  12. Before returning complete status.
- **Quarantine Cleanup & Destination Safety**:
  - When a timeout or error occurs, owned quarantine directory cleanup is strictly guarded by UID, device, inode, and symlink checks in `finally`.
  - Pre-existing or already-promoted destination directories are NEVER deleted or rolled back to hide a failure.

### 3. Gap 2 Implementation: Typed Artifacts and Independent Expectations (OP11)

In `multiautoresearch/adapters/amazon4u.py` (`validate_collected_artifacts`):
- **Complete Frozen 15-Field H Result**:
  - Validates exact 15 canonical H result keys (`H_RESULT_KEYS`): `schema_version`, `status`, `acceptance_scope`, `deployment_decision`, `run_spec`, `run_envelope`, `device_capability`, `method`, `epoch`, `actual_Adam_steps`, `selection`, `mixture`, `validation_report`, `remaining_readiness_gates`, `test_execution`.
  - Schema must be exact `"amazon4u-H-result-v1"`.
  - Acceptance scope must be exact `"H-REF_CPU_reference_fixture_only"`.
  - Deployment decision must be exact `"NOT_AUTHORIZED"`.
  - Rejects transport metadata injection (e.g. transport `version` injected into `h_result` is strictly forbidden).
  - Requires `method` to match `run_spec` and `run_envelope`.
  - Strictly validates `device_capability` matching frozen CPU specification: `{'device': 'cpu', 'accelerator': 'none', 'supports_training': True, 'state_transfer': 'cpu_quiescent_epoch_v1'}`. Rejects non-boolean `supports_training` (e.g. integer 1).
  - Validates `epoch` and `actual_Adam_steps` as non-boolean integers within approved spec budgets and authorization ceilings.
  - Validates `validation_report`: metric sections (`overall_metrics`, `metrics`) must be dictionaries containing only finite numeric values for the frozen seven registered metrics (`REGISTERED_SEVEN_METRICS`: `NDCG@5`, `NDCG@10`, `NDCG@20`, `HitRate@5`, `HitRate@10`, `HitRate@20`, `CatalogCoverage@10`; NO MRR, Recall, Precision, or training metrics). Malformed section types, unregistered metric keys, and forbidden proxy metrics (`eval_score`, `val_bpb`, `train_loss`, `learning_rate`, etc.) are strictly rejected.
- **Canonical 11-Field H Run Envelope**:
  - Requires mandatory non-None `expected_envelope` matching canonical `amazon4u-H-run-v1` envelope schema.
  - Validates all 11 keys (`schema_version`, `run_id`, `approval_id`, `profile_id`, `method`, `identity_hash`, `device`, `accelerator`, `bundle_sha256`, `initial_state_sha256`, `determinism_policy`).
  - Enforces string bounds (<= 256 chars), 64-character lowercase hexadecimal digests (`identity_hash`, `bundle_sha256`, `initial_state_sha256`), frozen profile (`H-REF-synthetic-cpu-v1`), device `cpu`, accelerator `none`, and determinism policy (`private_rng_bounded_cpu_no_global_setters`).
  - Bidirectional crosslinks: `result.run_id == envelope.run_id`, `result.approval_id == envelope.approval_id`, `result.method == envelope.method`, `result.identity_hash == envelope.identity_hash`, `result.device == envelope.device`, `result.accelerator == envelope.accelerator`, `result.determinism_policy == envelope.determinism_policy`.
  - Expected envelope propagated through `ApprovedOfflineContext`, `dispatch_amazon4u` (both `launch` and `output` branches), and `collect_outputs`.
- **Canonical H Authorization & Canonical Spec**:
  - Requires canonical H authorization: exact 5 keys (`schema_version`, `approval_id`, `run_id`, `scope`, `max_adam_steps`). Schema must be `"amazon4u-H-authorization-v1"`, scope must be `"tiny_synthetic_cpu_backward_and_private_restore"`, bounded strings, and integer `max_adam_steps` in `0..4` (rejects booleans). Rejects generic O authority.
  - Canonical H spec: `get_canonical_h_spec` only accepts canonical `amazon4u-H-spec-v1` or wrapped under validated `amazon4u-offline-spec-v1` (`o_spec["h_spec"]`). Rejects unsupported full-key spec schemas instead of normalizing them into H.
- **Physical Inventory & Checkpoint Verification**:
  - `expected_output_inventory`: Requires explicit non-negative integer `size` and 64-character lowercase hexadecimal `sha256` for every physical role. Missing or invalid size/hash fields fail closed upfront.
  - Strict JSON root type validation: plan, result, envelope, and manifest files must have dictionary JSON roots.
  - Plan version and budgets reject booleans masquerading as integers. Required `expected_version` is an explicit keyword-only positive integer without default.
  - Verifies actual file size on disk matches declared inventory size.
  - Computes and verifies actual SHA-256 on disk matches declared inventory `sha256`.
  - Verifies actual physical checkpoint SHA directly against `expected_checkpoint_sha256`. Checkpoint physical SHA operates in a distinct verification domain from semantic identity/manifest digests without invented blanket hex-inequality rules.
  - Deterministic baselines (`CorePopularity`, `KG-only`): Checkpoint role is satisfied by `not_applicable_deterministic_baseline` without physical file requirement.
  - Learned methods (`LightGCN`, `Heterogeneous KG+GNN`): Require valid physical checkpoint file and matching physical SHA-256.

### 4. Gap 3 Implementation: Fail-Closed No-Overwrite Promotion Seam (OP10, OP12)

In `multiautoresearch/adapters/amazon4u.py` (`collect_outputs`):
- **Elimination of Rename Fallback in Production**:
  - The vulnerable check-then-rename fallback (`quarantine.rename(dest_path)`) has been completely eliminated.
  - Production `collect_outputs` no longer exposes a `promotion_fn` bypass parameter.
  - `collect_outputs` invokes `_rename_no_replace_linux(quarantine_dir, dest_path)` directly.
  - If the native provider is unavailable or returns `False`, `collect_outputs` returns `{"status": "error", "reason": f"Native directory no-replace is unsupported on this platform; ordinary rename fallback is forbidden: {destination_dir}"}`.
  - If collision occurs (`FileExistsError` or `errno.EEXIST`), `collect_outputs` catches the error and returns `{"status": "error", "reason": f"Destination directory appeared concurrently: {destination_dir}"}` without modifying, deleting, or adopting the existing destination.
  - If another `OSError` occurs during promotion, `collect_outputs` returns `{"status": "error", "reason": f"Native directory no-replace failed: {exc}"}`.
  - `_rename_no_replace_linux` isolates `ctypes.CDLL` loading from syscall invocation, ensuring genuine `FileExistsError` and `OSError` propagate cleanly instead of being swallowed.
- **Safe Test Promotion Seam (`fake_directory_promotion_seam`)**:
  - Defined locally in `tests/test_amazon4u_bundle.py` as an explicit test fixture helper (not shipped in production).
  - Models supported success in tests by mocking `_rename_no_replace_linux` via `unittest.mock.patch(..., side_effect=fake_directory_promotion_seam)`.
  - Verifies source existence, regular directory type, no symlink, owner UID matching parent owner UID, destination non-existence, and performs `src.rename(dst)` only after all preconditions hold.
  - Raises `FileExistsError` if destination exists.
  - Operates safely under test tripwires and Pi's `NativeProviderGuard`.

### 5. Caller–Callee Audit Matrix (Three Gaps Correction)

| Contract / Interface | Definition Location | Signature & Return Type | Relevant Callers & Call Sites | Fixture Prerequisites & Provenance | Source / AST Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `collect_outputs(kernel_ref, expected_version, transport, destination_dir, *, ...)` | `multiautoresearch/adapters/amazon4u.py:2334` | `(kernel_ref: str, expected_version: int, transport: FakeKaggleTransport, destination_dir: Path, *, expected_spec: dict[str, object], expected_approval: dict[str, object], expected_manifest: dict[str, object] | None = None, expected_checkpoint_sha256: str | None = None, expected_output_inventory: dict[str, object] | None = None, expected_envelope: dict[str, object] | None = None, expected_run_envelope: dict[str, object] | None = None, max_pages: int = 8, max_file_bytes: int = 2097152, max_aggregate_bytes: int = 8388608, max_files: int = 128, deadline_seconds: float = 300.0, clock_fn: Any = None) -> dict[str, object]` | `dispatch_amazon4u` (output branch); `tests/test_amazon4u_bundle.py`: 22 collection test methods | Valid transport, non-existent destination path, positive finite deadline, canonical expected spec, envelope, authorization, inventory. | Enforces finite positive deadline; enforces hard security ceilings (8 pages, 128 files, 2 MiB file, 8 MiB aggregate); checks deadline at 12 distinct seams; snapshots metadata pre-callback; eliminates rename fallback and `promotion_fn` bypass; uses `_rename_no_replace_linux`; fails closed on `False`, `EEXIST`, or `OSError`. | COMPATIBLE / IMPLEMENTED |
| `validate_collected_artifacts(files, *, expected_version, expected_spec, expected_approval, ...)` | `multiautoresearch/adapters/amazon4u.py:2686` | `(files: dict[str, Path], *, expected_version: int, expected_spec: dict[str, object], expected_approval: dict[str, object], expected_manifest: dict[str, object] | None = None, expected_checkpoint_sha256: str | None = None, expected_output_inventory: dict[str, object] | None = None, expected_envelope: dict[str, object] | None = None, expected_run_envelope: dict[str, object] | None = None) -> dict[str, object]` | `collect_outputs`; unit test fixtures in `tests/test_amazon4u_bundle.py` | Complete quarantine artifacts directory, canonical spec, envelope, auth, expected inventory | Validates exact 15-field `amazon4u-H-result-v1`; canonical 11-field `amazon4u-H-run-v1`; canonical 5-key H authorization; exact CPU capability; registered 7 evaluation metrics (`NDCG@5/10/20`, `HitRate@5/10/20`, `CatalogCoverage@10`); mandatory physical inventory sizes and hashes; distinct checkpoint physical SHA; rejects proxy metrics and non-dict metric sections. | COMPATIBLE / IMPLEMENTED |
| `dispatch_amazon4u(args, transport=None)` | `multiautoresearch/adapters/amazon4u.py:1924` | `(args: argparse.Namespace, transport: FakeKaggleTransport | None = None) -> int` | CLI / Facade entrypoint (`kaggle_job.py`); `tests/test_kaggle_orchestration.py:1842` | Namespace with valid profile, spec, approval, plan, destination | Forwards `expected_envelope=ctx.expected_h_envelope` from validated context to `collect_outputs` in both launch and output branches. | COMPATIBLE / IMPLEMENTED |
| `_rename_no_replace_linux(src, dst)` | `multiautoresearch/adapters/amazon4u.py:2145` | `(src: Path, dst: Path) -> bool` | `collect_outputs` promotion seam | Fresh quarantine source directory, destination directory path | Invokes Linux `renameat2` with `RENAME_NOREPLACE` via libc ctypes; propagates `FileExistsError` and `OSError` cleanly; returns `False` if libc unavailable or function absent. | COMPATIBLE / IMPLEMENTED |
| `fake_directory_promotion_seam(src, dst)` | `tests/test_amazon4u_bundle.py:82` | `(src: Path, dst: Path) -> bool` | `tests/test_amazon4u_bundle.py` test fixtures | Fresh owned source directory, non-existent destination directory | Test-only fixture helper modeling supported promotion success under `_rename_no_replace_linux` patch. Checks source existence, dir type, no symlink, owner UID; checks destination non-existence; raises `FileExistsError` on collision; returns `True`. | TEST FIXTURE ONLY |

### 6. Static Test Inventory & Integrity Verification

- **Prior Test ID Inventory Preserved**: All 480 test IDs from `prior480-test-ids.json` are retained with unchanged identifiers (43 in `test_kaggle_orchestration.py`, 25 original in `test_amazon4u_bundle.py`, 412 inherited baseline). Total test methods across suites: 501 (412 baseline, 46 in `test_amazon4u_bundle.py`, 43 in `test_kaggle_orchestration.py`).
- **Test Fixture Enhancements**: All test methods in `test_amazon4u_bundle.py` retained and adapted to use canonical test fixtures (`_canonical_test_spec`, `_canonical_test_auth`, `_canonical_test_env`, `_canonical_test_manifest`, `_canonical_test_inventory`).
- **Repurposed Physical Checksum Test**: `test_op11_checkpoint_physical_sha_confusion_rejected_negative` repurposed to test wrong-role semantic digest as expected physical checksum fails with physical SHA mismatch, while correct physical checksum passes, without mocking `hashlib.sha256`.
- **Targeted Test Controls Preserved (8 controls)**:
  1. `test_op10_callback_metadata_mutation_prevention_control`: proves callback mutating caller's `expected_approval` during listing cannot alter pre-callback snapshot.
  2. `test_op10_parameter_bounds_ceilings_negative`: tests hard ceilings on `max_pages`, `max_files`, `max_file_bytes`, `max_aggregate_bytes`.
  3. `test_op10_invalid_clock_samples_fail_closed_negative`: tests clock returning NaN, Inf, bool, string, or raising exception fails closed.
  4. `test_op11_spec_or_auth_bool_budget_rejected_negative`: tests `epochs=True` and `max_adam_steps=True` are rejected as booleans with required `expected_version`.
  5. `test_op11_device_capability_supports_training_int_rejected_negative`: tests `supports_training=1` is rejected.
  6. `test_op11_budget_exceeded_rejected_negative`: tests `actual_Adam_steps > spec['max_adam_steps']` is rejected.
  7. `test_op11_json_root_non_dict_rejected_negative`: tests non-dict JSON roots in plan and result fail closed with required `expected_version`.
  8. `test_op10_promotion_collision_refused_negative`: exercises EEXIST seam call proof, False (unsupported) seam call proof, other OSError seam call proof, and concurrent appearance precheck collision proof without ordinary rename fallback.
- **Inherited Baseline Integrity**: All 31 inherited Python files retain byte-identical SHA-256 hashes matching HEAD baseline.
- **Restoration Base Integrity**: `kaggle_job.py`, `__init__.py`, `amazon4u_kernel.py`, and `test_kaggle_orchestration.py` remain byte-identical to `inputs/restoration-base/Amazon4U/...` per `inputs/restoration-base-sha256.json`.
- **Worker Discipline**: Strict source-only mode maintained; zero tests executed (`tests_executed=[]`). No measured PASS is claimed; independent Pi review and verification benchmark remain required.
- **Deployment Status**: Strictly `NOT_AUTHORIZED`.
