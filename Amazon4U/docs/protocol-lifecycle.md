# Protocol version, lifecycle and authorization

## Current record, implementation and regression verification

- Protocol: **`amazon4u-protocol-v2.0.0`**, following the user-approved
  [model-suite removal](bpr-mf-removal-amendment.md); v1.0.0 remains preserved.
- State: **`IMPLEMENTATION_REGRESSION_VERIFICATION`**.
- Execution revision: **not assigned** (`null`); propose `exec-v1` only when its
  settings and tuning plan are approved and frozen.
- Evaluator: preserved artifact **ACCEPT (task_only)**, task
  `2562b8ec-cf57-4ace-b348-9fbd6c994262`; shared engine in
  `scripts/recommendation_evaluation.py` integrated into main
  `scripts/recommendation_core.py` with guarded KG fallback. Integrated main suite:
  **113 tests, zero failures/errors/skips**. See [integration evidence](evaluator-integration.md).
- Remaining I1: Popularity/KG-only scorers and non-training sampler/loss/regularization
  reference utilities are independently verified and **ACCEPTED (task_only)**
  under task `cc64f3cb-9a4f-40aa-abbf-5ced6c1bd000`; benchmark evidence:
  **115 tests, zero failures/errors/skips**.
- Step 3a/3b learned-model foundations are independently verified and **ACCEPTED
  (task_only)**: LightGCN foundation under task
  `8153df65-e2e2-4571-a4ce-340d0ffeb1c2` and Heterogeneous KG+GNN foundation
  under task `d35b3d7c-3cd6-4f8e-8ecf-adcb963f4b93`. Code lives in
  `scripts/learned_models.py` (re-exported in `scripts/recommendation_core.py`)
  with synthetic correctness tests in `tests/test_learned_models.py` and
  `tests/test_heterogeneous_kgnn.py`; regression suite: **163 tests, zero
  failures/errors/skips**. Optimizer training loops, real-data training, profiling
  and deployment remain NOT_AUTHORIZED. See [the implementation roadmap](implementation-roadmap.md).
- Differentiable CPU model/graph recovery: frozen verification **ACCEPT (task_only)**
  under `f709d062-ae0e-4eb6-960c-f94728180d1b`, followed by separately recorded
  Pi-local main import integration and **204 passing tests**. Frozen source-task
  ESCALATE remains preserved; this is not full-harness acceptance.
- Checkpoint/provenance task `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a`:
  **ESCALATE / not accepted**; isolated draft only, 168 tests/6 errors and missing
  completion/doc evidence after timeouts. Operator-authorized same-task reopening
  allowed final revision 2, which timed out without changing the draft. Both
  revisions are now used; final independent decision
  `agy-evidence-411171fa-c296-40bc-9346-667b9d7c5b97` is ESCALATE, task_only not
  accepted, deployment NOT_AUTHORIZED. No automatic retry or task/budget reset.
  See [harness review](training-harness-review.md).
- Profiling and comparative training: **not authorized**.

- Deployment: **NOT_AUTHORIZED**.

The active scientific protocol is `amazon4u-protocol-v2.0.0`. The original v1.0.0
snapshot remains byte-for-byte intact. The major amendment removes the MF baseline
and metadata variant, retaining BPR loss for both remaining learned models. Milestone I1
implements core evaluation and non-training reference logic without optimizer updates or
learned embeddings. The authoritative machine record is
[protocol-lifecycle.json](../configs/protocol-lifecycle.json).

**Rationale:** milestone I1 provides pure Python reference utilities and evaluation
primitives with zero external dependencies. State transitions separate approved baseline
specifications, code implementation verification, profiling authorization, and comparative
training runs.

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
LightGCN initialization, execution/sampling/fanout/inference, batch sizes,
validation frequency or tuning budgets. Filling these declared pending slots
requires approval and a frozen execution revision before comparative training;
changing an already frozen scientific rule also requires a protocol bump.
The remaining choices are listed in the lifecycle and training configurations.

## Bounded profiling, authorization still pending

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

### Historical baseline-freeze verification (v1.0.0)

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

### Historical milestone I1 worker submission (not independent acceptance)

Milestone I1 implementation complete: reusable shared evaluator, deterministic Popularity
and KG-only scorers, on-demand uniform negative sampling, stable BPR loss, and endpoint
regularization in `scripts/recommendation_core.py`. Regression test suite in
`tests/test_recommendation_core.py` exercises hand-calculated ranking metrics, exact ties,
history filters, shortage reporting, corrupt profile detection, cold metadata invariance,
multi-positive asymmetric user-profile oracle, macro metrics with multi-user K boundaries,
low-rated validation record filtering, negative sampling preflight and unknown user rejection,
numerical failure handling (difference overflow, squared norm overflow, softplus),
and strict held-out isolation.

Known limitations: this is a reference fixture core, not a full-data production adapter or training pipeline; no cross-platform bitwise guarantee from math.fsum alone.

Full verification suite execution using ORIGINAL approved wrapper:
```sh
.venv/bin/python -B -c "import os,tempfile,unittest,json,sys; from pathlib import Path; p=Path('.test-output/recommendation-core-v1').resolve(); p.mkdir(parents=True,exist_ok=True); os.environ['TMPDIR']=str(p); tempfile.tempdir=None; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); r=unittest.TextTestRunner(verbosity=2).run(suite); (p/'fixture-result.json').write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'successful':r.wasSuccessful()},indent=2)+'\n'); sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)"
```

Diagnostic artifact: `.test-output/recommendation-core-v1/fixture-result.json` (note: existing literal backslash-n suffix in the wrapper makes this a diagnostic text artifact, not strict JSON; original command and output preserved as approved).
Result: **63 tests run, 0 failures, 0 errors, 0 skipped, successful: true**.

Invocation timeout history:
- Rev0: 10-minute timeout during initial implementation exploration.
- Rev1: 10-minute timeout during initial revision.
- Rev2: targeted fixes completed promptly; 63 unit tests pass cleanly in ~1.8s.

**Historical Worker review status: implemented_pending_Pi_review.** This original
submission is preserved, not retroactively accepted. The subsequent evaluator-only
acceptance and main integration are recorded below; remaining I1 review stays separate.
**Deployment decision: NOT_AUTHORIZED.** No model training, optimizer updates, profiling,
staging, commits or pushes were performed.

### Current evaluator acceptance and main integration

Pi independently accepted the preserved evaluator artifact under task
`2562b8ec-cf57-4ace-b348-9fbd6c994262`, revision 0; acceptance_scope **task_only**,
deployment **NOT_AUTHORIZED**. Decision:
`agy-evidence-f04ab70a-955f-4b89-8da0-2a484c1a4740`.
Artifact benchmark: 113 tests passed. After byte-identical mechanical source/test
integration into main, Pi verified the 63-test pre-integration baseline and
**113-test integrated suite** (zero failures/errors/skips, 1.334 seconds).
Evidence: `data/reports/evaluator-main-integration-v1-provenance/` and
[evaluator-integration.md](evaluator-integration.md).

The mutable lifecycle record now distinguishes this verified evaluator component
from remaining-I1 review and learned-model/harness work. A component-progress event
is appended to its history, without changing the lifecycle state, protocol version,
execution revision, pending execution choices or authorization flags. Frozen
snapshots/manifests and historical ESCALATEs remain untouched. Musical_Instruments
profiling preparation is a later deliverable; execution is not yet authorized.

Following the user's status-consistency correction, README, this document and
`configs/protocol-lifecycle.json` were synchronized. Independent validation matched
component task/counts to preserved integration evidence and confirmed prior history,
scientific/version/state/execution fields and authorization flags unchanged. Full
main regressions passed again: **113 tests, zero failures/errors/skips**, 2.547 seconds;
fixture scratch cleaned and `git diff --check` passed. These are status-record edits,
not a new implementation acceptance or execution authorization.

### User-approved v2.0.0 suite amendment

The user explicitly removed BPR-MF entirely, including the metadata variant, and
approved Pi performing this config/documentation cleanup directly rather than
launching an agy task. No runnable MF model existed; BPR loss/reference utilities
remain unchanged for LightGCN and Heterogeneous KG+GNN. Active core membership is
Popularity, LightGCN, KG-only and Heterogeneous KG+GNN. The paired bootstrap plan
now has three baseline contrasts and 36 descriptive intervals; two learned core
methods imply 36 final seed fits before tuning, not a permission to execute them.

`configs/protocols/v2.0.0/` provides a new immutable snapshot/manifest with predecessor
provenance; v1.0.0 and historical lifecycle events remain unchanged. The active
protocol version and baseline pointer change, not the lifecycle state, execution
revision, authorization flags, data scope, graph/evaluator rules or remaining
training recipe. See [the amendment record](bpr-mf-removal-amendment.md).

### Current remaining-I1 acceptance

Pi independently accepted the remaining Milestone I1 reference components under
Task `cc64f3cb-9a4f-40aa-abbf-5ced6c1bd000`, revision 0; acceptance_scope
**task_only**, deployment **NOT_AUTHORIZED**. Decision:
`agy-evidence-8cfc6826-f8ae-4ea6-be7c-dc1c838cd036`. The accepted scope covers
PopularityScorer, KGOnlyScorer, NegativeSampler, `bpr_loss`/`stable_softplus` and
`endpoint_regularization` only. It does not accept learned models, full training
loops, profiling execution or deployment.

Pi reviewed the actual implementation in `scripts/recommendation_core.py`, shared
context/evaluator dependencies in `scripts/recommendation_evaluation.py`, focused
regression tests in `tests/test_recommendation_core.py` and v2 protocol/config
records. Independent benchmark verification used
`PYTHONDONTWRITEBYTECODE=1 /home/sunf/FSB/Final/Amazon4U/.venv/bin/python -m unittest discover -s tests`
and passed **115 tests, zero failures/errors/skips**, 1.515 seconds; evidence
`agy-evidence-db13c5d9-6593-49d8-8586-3ff66cd9f6f3`. Prior setup/environment
ESCALATEs remain preserved as noncritical recovery history.

The next implementation gate is full training-loop/execution integration and then
bounded Musical_Instruments profiling-plan preparation, both still requiring their
separate gates. Profiling and training execution remain unauthorized; no model
training, optimizer updates or Musical_Instruments profiling has run.

### Current step 3a LightGCN foundation acceptance

Pi independently accepted the learned-model foundation and core LightGCN reference
implementation under Task `8153df65-e2e2-4571-a4ce-340d0ffeb1c2`, revision 0;
acceptance_scope **task_only**, deployment **NOT_AUTHORIZED**. Decision:
`agy-evidence-1ad0bf36-5bef-4661-824d-c7b6dfef8295`. The accepted scope covers
`LightGCNConfig`, `LightGCNGraph`, `propagate_lightgcn`, `LightGCNScorer`,
`compute_bpr_batch_loss`, `TrainingHarnessGuard` and checkpoint/provenance
manifest helpers.

Pi reviewed the final implementation in `scripts/learned_models.py`, re-exports
in `scripts/recommendation_core.py`, synthetic tests in `tests/test_learned_models.py`
and mutable status records. Independent benchmark verification used
`PYTHONDONTWRITEBYTECODE=1 /home/sunf/FSB/Final/Amazon4U/.venv/bin/python -m unittest discover -s tests`
and passed **140 tests, zero failures/errors/skips**, 1.542 seconds; evidence
`agy-evidence-81230b37-ea8a-40f5-a8f9-613e4f95420b`. Historical Step 3a
implementation/recovery ESCALATEs remain preserved as noncritical recovery history.

This acceptance does not authorize real-data training, optimizer updates,
profiling, Kaggle execution, comparative training or deployment.

### Current step 3b heterogeneous KG+GNN foundation acceptance

Pi independently accepted the synthetic Heterogeneous KG+GNN reference encoder
foundation under Task `d35b3d7c-3cd6-4f8e-8ecf-adcb963f4b93`, revision 0;
acceptance_scope **task_only**, deployment **NOT_AUTHORIZED**. Decision:
`agy-evidence-94db11ba-75bf-43f4-a255-7a975db8cede`. The accepted scope covers
`HeterogeneousKGNNConfig`, `HeterogeneousKGNNGraph`, `build_heterogeneous_graph`,
`mask_heterogeneous_batch_edges`, `propagate_heterogeneous_kgnn`,
`HeterogeneousKGNNScorer` and `compute_heterogeneous_bpr_batch_loss`.

Pi reviewed the final implementation in `scripts/learned_models.py`, re-exports
in `scripts/recommendation_core.py` and synthetic tests in
`tests/test_heterogeneous_kgnn.py`. Independent benchmark verification used
`PYTHONDONTWRITEBYTECODE=1 /home/sunf/FSB/Final/Amazon4U/.venv/bin/python -m unittest discover -s tests`
and passed **163 tests, zero failures/errors/skips**, 2.703 seconds; evidence
`agy-evidence-bca4a17c-806d-4bee-935b-9206520885be`. Historical Step 3b timeout
ESCALATE remains preserved as noncritical recovery history.

This acceptance does not authorize real-data training, optimizer updates,
profiling, Kaggle execution, comparative training or deployment.

### User-approved v2.1.0 amendment (2026-10-26)

`amazon4u-protocol-v2.1.0` is a minor, methodological amendment:

- the learning rate is tuned over 0.0005, 0.001 and 0.002;
- the execution choices that were still pending are frozen: batch sizes, validation every
  epoch, LightGCN initialization, full KG + GNN neighborhoods and exact LightGCN
  propagation;
- the tuning seed and trial-reuse rules are fixed.

See [training semantics](training-semantics.md#protocol-v210-amendment-execution-choices-and-learning-rate-tuning-2026-10-26)
and the [learning-rate amendment](learning-rate-tuning-amendment.md).

- **Recorded before any comparative training, tuning or held-out result.**
- **Earlier snapshots are intact.** The v2.0.0 snapshot and manifest are unchanged, and
  v2.1.0's snapshot under `configs/protocols/v2.1.0/` names v2.0.0 as its predecessor.
- **The lifecycle state is unchanged.** It stays `IMPLEMENTATION_REGRESSION_VERIFICATION`.
- **Nothing is authorized.** Training, profiling and deployment authorizations are
  unchanged: none, none and `NOT_AUTHORIZED`.
