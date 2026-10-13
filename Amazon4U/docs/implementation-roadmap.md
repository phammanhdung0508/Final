# User-approved implementation sequence and gates

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

## Step 1 architecture and preservation

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

