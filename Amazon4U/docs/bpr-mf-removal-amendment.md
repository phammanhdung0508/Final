# User-approved removal of BPR-MF

Status: **ACCEPT, task_only, user-authorized Pi-direct metadata cleanup**.
Deployment: **NOT_AUTHORIZED**. No agy task was launched.
User explicitly confirmed: "remove BPR-MF entirely." This follows discussion of
removal versus retaining it as an optional baseline. It authorizes removal from
the active study, including BPR-MF + metadata; it does not authorize deleting
historical evidence or rewriting the v1.0.0 frozen baseline.

## Scientific amendment

The active core becomes Popularity, LightGCN, KG-only and Heterogeneous KG+GNN.
No optional BPR-MF model or metadata variant remains. Retain the matched
heterogeneous encoder without metadata as the existing optional, unconfigured
control; do not add replacement baselines. BPR ranking loss and endpoint
regularization remain unchanged for LightGCN and Heterogeneous KG+GNN.

Changing frozen core-comparison membership is a **major** amendment:
`amazon4u-protocol-v1.0.0` → `amazon4u-protocol-v2.0.0`. Preserve the existing
`configs/protocols/v1.0.0/` tree byte-for-byte. Create a separate v2.0.0 snapshot
and SHA-256 manifest from the amended normative source files, with truthful
current source HEAD/dirty provenance, user approval and predecessor references.
Append lifecycle/amendment history; do not rewrite earlier events or evidence.
No learned model/test results have been exposed; publisher split/metadata audits
have already occurred and do not become unseen data. No results-driven selection.

Remove the KG+GNN-minus-BPR-MF bootstrap contrast. The remaining three contrasts
against Popularity/LightGCN/KG-only, two metrics, three categories and two feedback
conditions yield **36**, not 48, planned descriptive intervals. Preserve seeds,
10,000 resamples, seed 12345, pairing, metrics and all remaining uncertainty rules.
Only two learned core methods remain, so run-budget examples must be recomputed
without changing authorized resource ceilings, reserve percentage or granting runs.
Remove MF-only pending initialization/configuration slots; LightGCN initialization
remains pending. All splits, categories, feedback conditions, candidates, graph
semantics, encoders, remaining model specifications, numerical training recipe,
evaluator/scorer/sampler/loss code and authorization gates otherwise stay unchanged.

## Implementation/review policy

The initial plan was agy implementation in an isolated project snapshot; Pi
preserved 242 current-main files (7,989,954 bytes), verified the archive and ran a
clean 113-test baseline. **No agy task was launched.** The user then explicitly
approved Pi doing the cleanup directly because no runnable MF model exists. This
scoped exception applies to config/documentation/snapshot changes and matching
regression assertions, not future learned-model implementation. Pi reviews actual
changes, affected configs/docs/tests and frozen manifests, and verifies the
synthetic suite against the preserved pre-change baseline. No new implementation
commits, staging, pushes, source retuning, real-data/model/optimizer execution,
profiling, held-out experiments or deployment. Max two scoped revisions; no
replacement loop for exhausted substantive defects. Existing evaluator acceptance
and historical ESCALATEs retain their original scopes.

Pi's permitted edits are active configs/docs/AGENTS/README, focused lifecycle
regression tests and the new `configs/protocols/v2.0.0/` snapshot.
No implementation changes under scripts or framework trees. Preserve v1 and all
historical worktrees/data/evidence. Fresh external runtime and fixture scratch are
separate: coding-tool runtime logs/locks/feature configuration permitted in runtime,
fixture files ephemeral and cleaned. Compileall/bytecode forbidden. Only synthetic unittest verification and non-training metadata/hash checks may
execute; no optimizer or real-data model execution. Independent Pi baseline/review evidence goes to
`data/reports/bpr-mf-removal-provenance/`. No blanket ignore-based permission.

The direct amendment requires passing main regressions, independent old/new
snapshot hash/semantic checks and an explicit recorded Pi task-only review before
claiming completion. Frozen snapshots exclude mutable lifecycle/review records;
the new manifest links this amendment without snapshotting its mutable status. Profiling/training flags remain false,
execution_revision null, lifecycle IMPLEMENTATION_REGRESSION_VERIFICATION,
deployment NOT_AUTHORIZED. This amendment does not accept remaining I1/model/harness
implementation or authorize the final Musical_Instruments profiling workload.

## Recorded verification

Pi recorded the direct-review decision in
`data/reports/bpr-mf-removal-provenance/pi-direct-review.json`; this is a scoped
Pi-direct exception, not an independent authorship/delegation review. Main
regressions: **115 tests, zero failures/errors/skips**, 1.933 seconds; pre-change
isolated baseline: 113 passing. Both frozen manifests and their exact file sets
were verified; all 17 original v1 files and prior lifecycle events remain intact.
The implementation queue also removes the future MF module; its original I1
verification results remain explicitly historical, not a retroactive acceptance.
No implementation code, model execution, scope sampling or authorization changed.
