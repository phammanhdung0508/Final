# Protocol version, lifecycle and authorization

## Current record — user approved

- Protocol: **`amazon4u-protocol-v1.0.0`**.
- State: **`BASELINE_PROTOCOL_FROZEN`**.
- Execution revision: **not assigned** (`null`); propose `exec-v1` only when its
  settings and tuning plan are approved and frozen.
- Models/evaluator/training controls: implementation and verification pending.
- Profiling and comparative training: **not authorized**.
- Deployment: **NOT_AUTHORIZED**.

The user approved applying the discussed lifecycle and version policy. This is a
baseline documentation/configuration freeze, not acceptance of unimplemented
software and not permission to run optimizer updates. The authoritative machine
record is [protocol-lifecycle.json](../configs/protocol-lifecycle.json).

**Rationale:** a single DRAFT label conflated agreed scientific decisions with
unmeasured resource settings. Separate identifiers make readiness and permission
explicit without silently changing the approved experiment.

## Lifecycle and transition gates

| State | Entry requirements / permitted interpretation |
|---|---|
| DRAFT | Scientific decisions still being proposed |
| BASELINE_PROTOCOL_FROZEN | Approved baseline recorded and versioned; scoped implementation and non-training regression work can be requested |
| IMPLEMENTATION_REGRESSION_VERIFICATION | Implement baseline and independently verify semantics before profiling; tests with optimizer updates need separate approval |
| BOUNDED_PROFILING_AUTHORIZED | Explicit approved profiling scope, limits, provisional settings and outputs, plus passing prerequisite regression checks |
| EXECUTION_SETTINGS_AND_TUNING_PLAN_FROZEN | Approved measured execution settings, remaining initialization choices, validation schedule and exact tuning policy recorded |
| COMPARATIVE_TRAINING_AUTHORIZED | Independent implementation/integration verification and explicit authorization identifying versions, groups, runs and budgets |
| EXPERIMENTS_RUNNING | Execute only the authorized plan; preserve provenance, failures, checkpoints and registered selection rules |
| RESULTS_FROZEN_ANALYSIS | Freeze selected artifacts and results, perform registered reporting/uncertainty analysis; no automatic retuning after test exposure |

A state transition alone neither changes the protocol version nor grants any
permission. Record separate scoped authorizations; the label and flags must agree.
If a gate fails, stop and record the blocker rather than automatically advancing.
Deployment approval is separate from all research lifecycle states and task ACCEPT.

The baseline freezes scope, split/candidate/relevance rules, core suite, approved
architectures/scoring, Block 2 numerical recipe, graph semantics, metrics and
Block 3 reproducibility/bootstrap decisions. It **does not** freeze unresolved
MF/LightGCN initialization, execution/sampling/fanout/inference, batch sizes,
validation frequency or tuning budgets. Filling these declared pending slots
requires approval and a frozen execution revision before comparative training;
changing an already frozen scientific rule additionally requires a protocol bump.
The remaining choices are listed in the lifecycle and training configurations.

## Bounded profiling — authorization still pending

Start on Musical_Instruments only after a separate approval records:

- Methods and feedback conditions, hardware/environment, time/compute ceilings,
  permitted optimizer updates, provisional initialization/batch/fanout settings,
  and fresh output/temp paths.
- Full-data/full-ranking scope, training inputs and validation only; no test model
  evaluation, held-out-based selection or performance-driven hyperparameter tuning.
- Regression evidence for masking, sampling exclusions, scoring/ranking, fallback,
  regularization, deterministic controls and method-specific graph semantics.
- Measurements of epoch/validation time and peak memory, with provenance and an
  explicit statement that provisional settings are not comparative-run settings.

Previously audited published test splits are not unseen data. No learned test
results have been exposed; record any future exposure in amendments. Profiling
feasibility on Musical_Instruments does not establish feasibility on larger groups.

## Version policy and amendments

Use `amazon4u-protocol-vMAJOR.MINOR.PATCH`:

- **Major:** changes to splits, scope, relevance, candidates or core comparison design.
- **Minor:** approved experiment additions or substantive methodological amendments.
- **Patch:** clarifications with no change to experimental behavior.

Component identifiers such as `kg-v1` and `reproducibility-v1` remain component/schema
versions; they do not replace the global protocol version. Execution revisions
(`exec-v1`, `exec-v2`, ...) identify complete approved settings and tuning plans;
never use them to hide a scientific amendment or authorize a test-driven rerun.
Execution revisions and authorization records have their own hashes and are
separate from the baseline snapshot.

Preserve each frozen version's source snapshot and manifest under
`configs/protocols/<version>/`. Do not overwrite an earlier snapshot to accommodate
later edits. Append amendment history with prior/new versions, changed rules,
rationale, user approval, evidence references and whether held-out results were
already exposed. Post-exposure scientific changes are labeled new/exploratory;
reused test splits must never be described as unseen.

The initial v1.0.0 snapshot captures baseline configs and normative experiment
documents, with SHA-256 hashes and source Git revision. It excludes mutable lifecycle
state and execution/authorization records to avoid circular hashes and accidental
version bumps on state transitions. The snapshot is a local artifact, not a commit;
no staging, commit or push is authorized by this approval.

## Required run provenance

Each profiling, tuning or final run must record protocol version and baseline
manifest hash, lifecycle state at start, frozen execution revision (or explicitly
provisional profiling settings), applicable scoped authorization and its hash,
resolved configuration hashes, source Git revision plus dirty diff/untracked source
snapshots, environment, data/split/condition identity, seeds and run purpose.
Before a comparative run, check configs against the approved baseline/execution
records. Mismatches block the run; do not silently update manifests to match code.
Implementation of these checks and run-manifest generation remains pending.

## Verification status

Documentation/configuration only. Added read-only regression checks in
`tests/test_protocol_lifecycle.py` for version/state separation, false authorization
flags, preserved pending choices and snapshot hashes. The full existing fixture
suite plus four lifecycle checks passed: **12 tests** using
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -v`.
The overlap-detection fixture deliberately emits a failing validation result as
its expected negative case; the test itself passed. No models or optimizer
updates were run. JSON syntax, cross-document consistency, changed-path scope and
whitespace are independently reviewed before task acceptance. This record does
not claim model correctness, performed profiling or training.

**Pi review: ACCEPT.** Acceptance scope: lifecycle/version documentation,
configuration, preserved baseline snapshot and read-only lifecycle regression checks.
Evidence: 12 passing fixture tests, valid JSON, verified source/snapshot SHA-256
matches, valid local documentation links and clean `git diff --check` for Amazon4U.
All new changes are inside Amazon4U; unrelated pre-existing workspace changes are
preserved. **Deployment decision: NOT_AUTHORIZED.** No profiling, model training,
optimizer updates, staging, commits or pushes were performed.
