# User-approved implementation sequence and gates

**Current Worker governance:** [native agy through the delegation extension is the
implementation default](worker-delegation-policy.md). Pi plans/scopes/orchestrates,
independently verifies and decides acceptance. Native recovery maxima are separately
authorized **2 + 2 + 2**, with cumulative history and root-cause review after each
substantive ESCALATE; #3 requires a recommendation,not automatic fallback. Every SDK
text-only generation request needs explicit user approval. Standing noncritical
recovery discretion and historical SDK preferences do not expand task/round budgets,
change channels silently or authorize execution. Historical approvals below remain
phase-scoped; the new policy resets none of their counters or decisions.

The user approved proceeding in order 1 → 4, including scoped delegation to agy
without repeated confirmation. Pi plans and independently reviews each subsystem;
agy implements/fixes; acceptance criteria and original verification commands are
pinned per task. Noncritical setup fixes do not need repeated lead approval.
No historical ESCALATE decision is erased, and mandatory evidence cannot be waived.

**Standing user authorization (recorded after verification-only escalation):** the
user prioritizes the final result, implementation + regression testing + profiling
on Musical_Instruments. Pi handles in-scope noncritical recovery without repeated
approval requests, retaining the agreed agy/Pi loop, preservation, prospective
scope/output policies and bounded revision limits. See
[the standing recovery authorization](delegation-and-execution.md#standing-authorization-noncritical-recovery-without-repeated-questions).
Critical scientific/security/data/resource decisions and execution permissions not
already granted still require the user. This does not erase historical ESCALATEs or
implicitly authorize training/profiling execution or deployment.

1. Integrate the independently accepted evaluator into the main Amazon4U checkout,
   preserving current source and compatible KG-only semantics; independently run
   regression verification before claiming main-checkout integration acceptance.
2. Separately verify remaining I1 Popularity/KG-only scorers and reference sampling,
   BPR loss and endpoint regularization. Existing broad I1 remains unaccepted until
   those scopes have their own independent evidence.
3. Implement and independently verify learned models and training harness under
   the frozen protocol, including heterogeneous positive-edge masking, fixed-graph
   LightGCN and reproducibility/checkpoint contracts. No real model training is
   authorized. Synthetic optimizer-update tests require explicit scoped approval
   before execution; non-training reference fixtures may proceed.
4. Prepare a bounded Musical_Instruments profiling plan for explicit execution
   approval: hardware, time/RAM/VRAM ceilings, provisional configuration, optimizer
   permissions, validation-only measurements and fresh outputs. Preparation is not
   permission to execute profiling or comparative training.

Deployment remains **NOT_AUTHORIZED**. No implementation commits or pushes are
included in this approval. Frozen scientific snapshots/configs remain unchanged.

## Historical Step 1 architecture and preservation (HEAD at that milestone)

Current main HEAD: `ed60be08f3d1f22a9d76eecbee282415f95a6a5c`. Pi preserved and
byte-verified all 229 tracked Amazon4U files (6,349,614 bytes) before modifications:
`data/reports/evaluator-integration-provenance/20261011T175759Z/`.
Archive SHA-256: `76508021f4d9383c723073b7126dcbd3027bf59ed76d297bfad8d60fb0bf2ece`.
Existing untracked worktrees, sibling changes and ignored real datasets remain
untouched. No new source-capture commit is needed: implementation review uses a
local clone of existing HEAD at `.worktrees/evaluator-integration-review/`.
The initial sparse checkout failed evidence preflight because the monitor requires
all tracked outside-project sentinel files to exist. Task
`d0a8d369-f283-4a35-8ad7-b27f30370e4d` remains ESCALATE/unaccepted, deployment
NOT_AUTHORIZED; no Worker ran. Under the lead's standing noncritical-setup approval,
Pi materialized the remaining tracked checkout for a separately scoped fresh
submission. Movie4U files there are unmodified baseline sentinels only; Worker may
not edit or execute them. Originals in the main sibling project are untouched.
Previously authorized bounded recursive sentinel hashing is retained. Pi alone
prepared that isolated Git checkout; Worker Git metadata edits forbidden. This
mechanical preflight correction does not waive mandatory benchmark/acceptance gates.

The accepted reference evaluator is in
`.worktrees/evaluator-hardening-clean/Amazon4U/scripts/recommendation_evaluation.py`
with its 33-test suite and task-only acceptance record in
[the evaluator review](evaluator-hardening-review.md). It must not be modified.

Integration design: one shared TrainingContext/ScoringFailureError and evaluator
candidate/filtering/ranking/metrics implementation. Preserve existing core import
paths for compatibility rather than leave two divergent evaluation engines.
The base evaluator continues rejecting arbitrary provider fallback claims. A narrow
core-specific extension must preserve the frozen KG-only whole-user
`unusable_metadata_profile` fallback without enabling arbitrary scorers, corrupt
profiles or users with usable positive-weighted profiles to claim fallback. Validate
same context, finite valid profiles and condition-positive popularity scores;
never convert scorer failures into fallback. Keep scorer/loss/sampler mathematics
unchanged in step 1; their standalone verification belongs to step 2.

Independent synthetic checks must cover both import modes (`scripts.*` and existing
scripts-directory imports), identity of shared exceptions/context, inherited
candidate hardening through the public core API, strict validation history,
registered KG fallback/subgroups and mixture selection. Existing tests and all
accepted adversarial fixtures must remain passing. No scientific-rule amendment.

## Current status

### Current H-REF — corrected six-file candidate, ESCALATE pending regression

The recorded sequence used **four SDK text-only generation requests**, not native agy
invocations. Request3 delivered six proposed files; request4 corrected four files and
retained training_data.py/training_entrypoint.py byte-identically. Pi reviewed the
actual changes and relevant callers/native boundary: B1's scoped CPU Adam guard and
F2–F5's checkpoint-name,exception,fallback and source-closure corrections are statically
addressed,not runtime-verified. No accepted model/evaluator/D1–D3 or methodology change.
Latest selected review: `data/reports/training-harness-h-ref-v1/correction-v1/review.json`.
Selected proposal: `data/reports/training-harness-h-ref-v1/correction-v1/correction/preview/Amazon4U/`.

The candidate remains **unapplied to isolated/main code and unaccepted**. The original
389-test dependency baseline passed (57.732s unittest,64.917s process,661983232bytes
peakRSS;3564expanded cases). **Zero H candidate suites or new numeric updates ran.**
Static discovery predicts412 methods,not a PASS. H2–H8 require actual mandatory evidence.

Next gate,only if separately authorized: mechanical isolated application of the exact
six-file proposal,fresh safety/preservation review and one original full regression
under **240s / 2GiB** and unchanged log/scratch/fixture limits. Then Pi assesses all
original H1–H9 and current limitations before any acceptance. No new generation is
needed just to verify the candidate. No new native/SDK call,regression,main integration
or experiment is authorized by this status update. The four SDK requests remain
consumed; prospective native 2 + 2 + 2 does not reset/replenish that history.
Acceptance_scope: **H-REF CPU reference,not accepted**; deployment **NOT_AUTHORIZED**.
O-OFFLINE remains unassigned;GPU-READY and full-data profiling remain separate gates.
[Authority and complete phase history](training-harness-h-ref.md).

### Historical first two-SDK-request H result — ESCALATE/no files, not latest

User approved H only withdevice interfaces. Fresh389baselineOK64.917s process,
57.732s unittest/661983232bytespeakRSS,3564expanded pass. The two SDK responses returned
no files:missingfrozencontracts,thenloss-returnkeyabsentfromPi's freshAPI excerpt.
Pi owns contextomission;no modelbug/CPUharnessinfeasibility established. At that stop,
no newtests,updates/restore or candidatefullsuite;isolated316-filebaseline unchanged.
`data/reports/training-harness-h-ref-v1/review.json` retains that ESCALATE and two used
requests. Later separately bounded continuations delivered the proposal above,without
rewriting this original decision/plan or reclassifying SDK requests as native agy.

### Historical Training Harness preparation proposal — O-OFFLINE still unassigned

The proposal below predates the later H-REF phases. Its preparation-only statements
and proposed paths/budgets are historical,not the current H candidate state or a
standing SDK channel. Current policy and the latest H review above govern next steps.

User-agreed clarification: **CPU reference first,not permanent CPU architecture**;
see [device-transition gate](training-harness-device-transition.md). H-REF andO-OFFLINE
can establish limited milestones,but GPU-READY remains required before planned GPU
profiling. Design shared device/capability/transfer seams now;do not rewrite the GNN
or launch GPU work now. Numeric parity,determinism,GPU RNG/checkpoint/runtime and
actual Kaggle compatibility must later be verified;full-data feasibility still needs
bounded Musical_Instruments profiling. No old criteria/budgets/report resets.

The user authorized preparing the requirement/evidence matrix and bounded plan under
**reuse orchestration,not inherited training methodology**. See
[the plan](training-harness-preparation.md),[34-part matrix](training-harness-requirements.md)
and `configs/training-harness-preparation.json`. Existing managed Kaggle runner is the
foundation;propose explicit Amazon4U profile/targeted replacements,not a competing
framework. All new paths/Worker budgets/synthetic optimizer-restore tests/240s limits
are proposals only. No production/legacy replacement or execution now.

First implementation scope proposed: authoritative CPU models plus state/data/evaluator
integration,and offline packaging/entrypoint/failure/status/artifact orchestration.
CPU/GPU readiness,LightGCN initialization,full source closure outside D1's20paths,
Kaggle compatibility/durable checkpoint storage and remaining execution choices are
explicit gates—not solved by selecting Kaggle GPU metadata. Stage names are recommender
training from scratch,validation selection,frozen final test andpost-selection reporting;
not fine-tuning or self-supervised pre-training. Preserve D1–D3/code/evidence/history.
After separate implementation/main-integration approval and readiness review,prepare
bounded full-scope Musical_Instruments profiling;then Toys_and_Games/Electronics checks,
execution/tuning freeze and separately authorized comparative/final-test work.

### Active checkpoint position — D3 main integration ACCEPT, isolated phase retired

The user separately authorized decommissioning isolated D3 implementation and bounded
main integration. Both original D3 trees/evidence remain unchanged,audit-only;no delete,
move,chmod,Gitmetadata change or additional isolated execution. Pi copied only the
exact3accepted D3files after355main baselineOK56.293s;fullmain389OK57.452s process,
0failures/errors/skips,under240s each. All355IDs/bytes/3564expanded/34D3methods preserved.
Complete MI1–MI5/IO1–IO8/current8orderedlimitation adjudication: **ACCEPT/task_only** in
`data/reports/checkpoint-storage-main-integration/review.json`;see
`docs/checkpoint-storage-main-integration.md`. Old mainD3doc archived;transferred API
artifact exact,with Worker submission-status history preserved andthisreceipt current.
FourWorker attempts/sevenoldfullruns plus2mainruns retained,nine total,no reset/retry.
Deployment NOT_AUTHORIZED;no further automatic tests/Worker/corrections,profiling/training,
heldout/remote,stage/commit/push/release/rollback. Researchexecution still separate gates.

Previous isolated independent D3 decision is **ACCEPT/task_only** after the explicitly authorized
Pi-only sourceFIFO oracle correction:389OK,0failures/errors/skips,53.292s unittest/
55.890s process under240s. All355 accepted dependencies/IDs and3564expanded cases
preserved;production andcandidateAPI docs unchanged. FullIO1–IO8 and8exactordered
limitation reviews completed. See
`data/reports/checkpoint-storage-implementation/oracle-only-correction-v1/review.json`.
AllpriorESCALATEs/failed389 and4Worker/7conservativefullrun history remainunchanged.
Deployment NOT_AUTHORIZED;D3 main source/test integration/regression requires separate
approval. No profiling/training,heldout,stage/commit/push or additional cycles. The
following entries preserve the implementation authority and earlier gate history.

Latest user approval supersedes the preparation-only hold below: execute D3 within
IO1–IO8 using implementation-first, evidence-driven correction. Worker may propose
and implement scoped technical choices; Pi verifies source and tests, targets concrete
obstacles and never claims unproved guarantees or executes unsafe IO tests. The bounded
plan is `data/reports/checkpoint-storage-implementation/plan.json`. D1/D2 remain closed
and protected; no controller investigation, main transfer or research authorization.
The historical pre-Worker ESCALATE after two120s timeouts remains in
`data/reports/checkpoint-storage-implementation/review.json`; no D2 defect was shown.
Runtime investigation is closed: no profiler, diagnostics, sudo, C++ trace, thread
experiment or accepted-test edit. User subsequently approved the D3-only240s ceiling
in `checkpoint-storage-execution-budget-amendment.md`; fresh unchanged355 baseline
passed55.645s process. Original120s outcomes remain unchanged.

Initial full D3 regression:379tests,1failure/2errors/0skips,55.993s process; all355
inherited tests and3564 expanded cases pass. New source fixture modes caused early
refusals. One consolidated revision proposed targeted permission, inode-publication,
O_PATH-classification and byte-order corrections, but remains **unapplied/unexecuted**:
its source baseline hash is malformed, and the assigned IO2 parent-race guarantee is
still unresolved. Both response slots used. Final **ESCALATE/task_only/not accepted**:
`data/reports/checkpoint-storage-implementation/final-review-evidence/review.json`.
No automatic third request, Pi production/hash fix, budget reset, main transfer or
research execution; deployment NOT_AUTHORIZED. Retain both proposals and old reports.
Earlier preparation status below is retained as history, not an active prohibition.

D1 identity/provenance and D2 payload validation are accepted and integrated in main.
D2's separate integration decision is
`data/reports/checkpoint-payload-main-integration/review.json`:355 passing tests,
zero failures/errors/skips,56.643s complete process within120s, task_only and
NOT_AUTHORIZED for deployment. Later historical entries below remain historical
records; their exhausted tasks are not reopened or reset.

The user has concluded D2 and directed **D3 preparation only**. No additional D2
correction, re-audit or security-controller investigation is planned absent a newly
identified concrete error. Preserve accepted D1/D2/model code/tests and their reports.
D2 validates checkpoint mappings **after decoding**; D3 owns secure checkpoint
storage/retrieval, decoding-time resource bounds, allowlisted physical acquisition,
no-clobber publication/cleanup and integration with D1/D2.

See [the D3 scope/evidence draft](checkpoint-storage.md) and
`configs/checkpoint-storage-scope.json`. Finalize format/decoder bounds, IO policy,
root/race/platform contract, publication/failure semantics and exact APIs/tests/setup
before implementation authorization. No D3 Worker, source implementation, checkpoint
IO or fresh regression execution is authorized by this preparation. Keep one
substantive D3 deliverable and the established text-only channel; no tooling detour.
Training, profiling and deployment remain separately unauthorized.

### Preserved earlier milestone records

Step 1: **evaluator integrated and regression-verified in main**. Preserved
artifact task `2562b8ec-cf57-4ace-b348-9fbd6c994262` received recorded
**ACCEPT/task_only/NOT_AUTHORIZED**; decision
`agy-evidence-f04ab70a-955f-4b89-8da0-2a484c1a4740`. Pi mechanically integrated
its five accepted source/test/doc files after preservation, without new code
retuning. Main baseline 63 tests passed; integrated main **113 tests passed**,
zero failures/errors/skips, 1.334 seconds. Other pre-existing tracked Amazon4U bytes,
unrelated Git status records and HEAD/index were unchanged; no sibling edits.
See `evaluator-integration.md`, `evaluator-verification-runtime-separated.md` and
`data/reports/evaluator-main-integration-v1-provenance/` for complete evidence.
Historical ESCALATEs remain unchanged; runtime-separated outputs and report closure
were handled autonomously under standing noncritical recovery authorization.

Step 2: **complete and accepted task-only**. Remaining-I1 PopularityScorer,
KGOnlyScorer, NegativeSampler, BPR loss/reference utilities and endpoint
regularization were independently verified under task
`cc64f3cb-9a4f-40aa-abbf-5ced6c1bd000`; decision
`agy-evidence-8cfc6826-f8ae-4ea6-be7c-dc1c838cd036`; benchmark evidence
`agy-evidence-db13c5d9-6593-49d8-8586-3ff66cd9f6f3`: **115 tests passed**,
zero failures/errors/skips. Historical noncritical recovery ESCALATEs remain
preserved and do not change this final Step 2 acceptance.

Step 3: **Step 3a complete and accepted task-only**. Learned-model training
harness foundation, LightGCN fixed Option B graph construction, exact full-batch
propagation, dot-product candidate scoring, composed BPR batch loss, endpoint
regularization, training harness guards against empty-batch/unauthorized updates,
and serializable checkpoint/provenance contracts are implemented in
`scripts/learned_models.py` (and re-exported in `scripts/recommendation_core.py`)
with 25 synthetic unit tests in `tests/test_learned_models.py`. Final acceptance
task `8153df65-e2e2-4571-a4ce-340d0ffeb1c2`; decision
`agy-evidence-1ad0bf36-5bef-4661-824d-c7b6dfef8295`; benchmark evidence
`agy-evidence-81230b37-ea8a-40f5-a8f9-613e4f95420b`: **140 tests passed**,
zero failures/errors/skips. LightGCNConfig validates exact 3 propagation layers;
`propagate_lightgcn` validates exact key matching for `graph.users` and
`graph.warm_items`, rejecting missing/unknown IDs; `TrainingHarnessGuard` fails
closed unconditionally for real data (even if `allow_synthetic_updates=True`) and
requires `allow_synthetic_updates=True` for non-real-data reference validation.
Historical Step 3a recovery ESCALATEs remain preserved.

Step 3b: **complete and accepted task-only**. The synthetic Heterogeneous KG+GNN
reference encoder foundation is implemented in `scripts/learned_models.py` and
re-exported in `scripts/recommendation_core.py`, with tests in
`tests/test_heterogeneous_kgnn.py`. Final acceptance task
`d35b3d7c-3cd6-4f8e-8ecf-adcb963f4b93`; decision
`agy-evidence-94db11ba-75bf-43f4-a255-7a975db8cede`; benchmark evidence
`agy-evidence-bca4a17c-806d-4bee-935b-9206520885be`: **163 tests passed**,
zero failures/errors/skips. Accepted scope covers the typed graph/config, batch
positive-edge masking helper, deterministic synthetic relation/self aggregation,
scorer, and BPR loss-composition reference utilities only. Historical Step 3b
timeout ESCALATE remains preserved.

Step 3c models/graphs recovery: **frozen verification ACCEPT / task_only /
NOT_AUTHORIZED**, task `f709d062-ae0e-4eb6-960c-f94728180d1b`, decision
`agy-evidence-bd7419a4-e9bb-4669-b261-4fb3064d5900`, benchmark
`agy-evidence-118bdab1-f944-443c-99e6-1b9410fe28f6` (180 clone tests passed).
The differentiable CPU PyTorch models/training-only graphs and 40 tests are now
in main; a Pi-local import compatibility repair and fresh-process test bring
main regression to **204 passed**, zero failures/errors/skips. See
`differentiable-models-recovery.md`, `training-harness-review.md` and
`data/reports/differentiable-models-main-integration/`. The preserved frozen
candidate is unchanged and historical source ESCALATEs remain intact.

The **full training harness remains unfinished/unaccepted**. Checkpoint/provenance
task `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a` is **ESCALATE**, preserved in isolation:
168 tests/6 errors, required doc/report absent. After user-approved shared-tool
repair, the operator reopened the same task and Pi launched final revision 2;
it timed out, leaving the draft unchanged. Both revisions are now used. Complete
independent source/test/dependency review confirms unresolved identity/provenance/
state/IO blockers; original benchmark still fails. Final decision:
`agy-evidence-411171fa-c296-40bc-9346-667b9d7c5b97`, task_only not accepted,
deployment NOT_AUTHORIZED. No automatic retry, replacement task or budget reset. See `checkpoint-recovery-plan.md`
and `checkpoint-final-review.md`. Remaining separate work is checkpoint closure
and training-loop/continuation tests,
followed by integrated review. Real optimizer updates, full training loops,
profiling, real-data training execution and deployment remain NOT_AUTHORIZED
except any separately predeclared bounded synthetic test scope. Step 4 remains queued: bounded
Musical_Instruments profiling plan preparation. No model training, optimizer
updates or profiling has run.

