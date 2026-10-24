# Evaluator integration recovery authorization

## Agreed scope (user: "ok, i agree. please continue")

Pi may delegate a focused recovery of evaluator integration, correcting the
always-present core fallback reason schema, adding exact regression coverage,
correcting draft documentation, and independently re-verifying all original
integration criteria. This explicit approval permits a fresh scoped task after
the exhausted historical task; it does not retrospectively accept that task.
Pi retains acceptance authority. Main integration still requires this gate to
pass; no training, profiling, held-out evaluation, deployment, staging, commits,
pushes or rollback are authorized by this recovery approval.

Historical task `303fdf53-8c5b-4013-86be-4fec983c809a` remains ESCALATE,
acceptance_scope task_only, deployment_decision NOT_AUTHORIZED. See
`evaluator-integration-review.md` for the original findings and benchmark.
Historical standalone evaluator ACCEPT and older ESCALATE decisions are unchanged.

## Preservation and workspace

Pi archived and independently byte/hash-verified six final draft artifacts before
starting recovery. Archive and per-file SHA-256 manifest:
`data/reports/evaluator-integration-recovery-provenance/20261011-recovery/`.
Archive SHA-256:
`7a6d478dd1a2b7c2b0be6d7aa8a05d11f13bdd74f05dc8d54ea81a371767b55f`.
Original escalated worktree remains read-only. Archive evidence was copied into
the new clone's protected data directory.

Fresh clean local clone of existing HEAD `ed60be08f3d1f22a9d76eecbee282415f95a6a5c`:
`.worktrees/evaluator-integration-recovery/Amazon4U/`. No new baseline commit.
Full tracked checkout is materialized for sentinel monitoring, not sparse.
Outside-project tracked files in the isolated clone are immutable baseline
sentinels, not editable source or execution targets. Main and sibling originals
are untouched. Pi configured generated test-output Git exclusion before submission.
Previously approved recursive sentinel hashing budgets remain: 128 MiB/file,
512 MiB total, 60 seconds/file, 180 seconds/snapshot. Frozen protocols, all data,
and baseline test output are protected; this is not an OS sandbox.

## Implementation and verification contract

agy must first copy the five existing draft source/test/doc files from the
historical worktree, preserving all baseline and accepted fixtures, then make
focused fixes only. Allowed Git-root-relative files:

- `Amazon4U/scripts/recommendation_evaluation.py`
- `Amazon4U/scripts/recommendation_core.py`
- `Amazon4U/tests/test_recommendation_evaluation.py`
- `Amazon4U/tests/test_evaluator_integration.py`
- `Amazon4U/docs/evaluator-integration.md`

Required: one shared engine/context/error/import identity; exact candidate/history,
finite-score and ranking/metric semantics; guarded same-context legitimate KG
fallback with exact popularity; core always emits both protocol fallback reasons
with count0/all-None metrics when absent, while strict standalone defaults and
provider fallback rejection remain unchanged. Exact empty/no-fallback/empty-positive-
only and mixed three-route/by-reason macro/coverage fixtures are mandatory.
Profile finiteness validation must not trust a provider-overridden validator;
review legitimate subtype support without making sandbox/out-of-band guarantees.
Correct shortage, count, tracked-sentinel and pending-decision documentation.

Original 63 baseline tests plus 21 accepted shared tests and 13 integration tests
must be retained and pass, alongside new fixtures. Baseline and final synthetic
suite use the same standard-library unittest discovery wrapper and main `.venv`
Python with `-B`; reject failures/errors/skips. Task wrapper output/TMPDIR:
`Amazon4U/.test-output/evaluator-integration-recovery-v1/`; separate protected
pre-submission baseline output:
`Amazon4U/.test-output/evaluator-integration-recovery-baseline/`.
No datasets, optimizer updates or profiling in verification. No deployment policy:
NOT_AUTHORIZED. Maximum two scoped revisions, then ESCALATE if unmet/incomplete.
Initial status: authorized, independent verification pending; no completion claim.

## Independent outcome

Task `8aa6d9b3-2366-4bf5-943e-78e1d7ad60fd`, revision 1: **ESCALATE**,
acceptance_scope **task_only**, deployment_decision **NOT_AUTHORIZED**.
Decision evidence: `agy-evidence-2305da6d-ed6b-41e0-b3bf-99687c43bd97`.
This is not a revision-cap escalation: one revision remains, but a known
execution/output-policy violation cannot be erased by another source revision.

Pi's original approved benchmark independently passed **113 tests, zero
failures/errors/skips**, 1.414 seconds. Full output (25,093 characters) read:
`agy-evidence-02b45ffa-aac9-4947-83da-c93ccd8136e2`.
All four implementation/regression criteria are supported by independent source,
caller/dependency and exact-oracle review. The absent core fallback-reason schema
regression is fixed; standalone strict defaults remain unchanged. Corrupt/missing
profile and provider-overridden-validator claims fail. Both condition/split pairs
and all three routes/two reasons have fixed metric/coverage oracles. Original
fixtures preserved: 63 baseline + 21 shared + 29 integration = 113.

**Blocking fifth criterion:** independent raw SQLite tool-call decoding revealed
Worker executed `python -m compileall -q scripts tests` in revision 0 step 75 and
revision 1 step 57. This was outside the approved exact wrapper and generated
15 ignored `scripts/__pycache__/*.pyc` / `tests/__pycache__/*.pyc` files totaling
407,773 bytes outside the declared output path. The standard-ignore monitor
reported no unauthorized monitored changes because these paths were ignored;
that is not evidence that no extra writes occurred. Pi did not delete the caches,
retroactively amend the artifact policy, accept the task, or copy sources to main.
No training/profiling commands were observed in the bounded captured action audit;
this does not establish sandboxing or complete historical timeout execution.

The final post-benchmark monitored snapshot is
`agy-evidence-208258f7-109a-4a2a-868c-fdd993318074`, hashing 150 protected files,
86,293,638 bytes completely, no Git metadata/outside-sentinel changes.
Pi preserved actual SQLite DB bytes and exact original JSON tool arguments for
both completed current-task invocations, including `finish`, rather than relying
on action candidates. All 80/66 rows framed and typed; 39/32 tool-call wrappers;
1,737,246/1,696,588 raw payload bytes; no nested trajectories. Sample row hashes
match immutable tool evidence; other hashes retained for comparison. Read-only
source DB sizes/mtimes/inodes were stable during capture. Initial decoder stopped
on the previously unlisted `finish` tool name, then was rerun with that explicit
reporting action recognized; only the complete successful records support review.
No historical timed-out invocation was reclassified as complete.

All preservation artifacts are under
`data/reports/evaluator-integration-recovery-provenance/20261011-recovery/`:
- `worker-revision-0.db`, `worker-revision-1.db`
- `worker-revision-0-decoded-actions.json`, `worker-revision-1-decoded-actions.json`
- `undeclared-bytecode.tar.gz`
- `current-artifacts-and-bytecode-finding.json` (source/cache hashes and timestamps)

Current implementation is preserved in the recovery clone, unaccepted. Main source
and sibling projects remain untouched; no commits, pushes, deployment or profiling.
Historical original draft, standalone ACCEPT and ESCALATE decisions unchanged.
Recommend explicit approval for a **clean verification-only scoped task**, with
this source snapshot preserved and every generated path predeclared, rather than
another reimplementation. No automatic replacement, rollback, or policy waiver.
