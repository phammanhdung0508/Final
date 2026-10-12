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

Step 2: next, separate remaining-I1 scorer/sampler/reference-loss verification;
passing evaluator integration does not by itself accept that scope. Steps 3–4:
queued, not completed. Profiling/training execution unauthorized; deployment
NOT_AUTHORIZED. No model training, optimizer updates or profiling has run.
