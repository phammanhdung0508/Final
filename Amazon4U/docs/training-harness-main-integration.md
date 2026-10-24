# H-REF main integration, ESCALATE, not accepted

User approved the [bounded integration plan](training-harness-main-integration-preparation.md).
Pi performed the approved main-specific verification setup and mechanical six-file
transfer. No Worker/SDK was invoked. **The six H files are now present in main, byte-identical
to the accepted isolated artifact, but main integration is NOT accepted.** Do not treat
presence/importability or isolated acceptance as production/experiment readiness.

Authority, effective plan, complete logs, independent reconciliation and decision:
`../data/reports/training-harness-main-integration-v1/`.
See especially `root-cause-review.md`, `interrupted-run-reconciliation.json`, `review.json`.
The approved preparation plan is retained unchanged, not rewritten to match this result.

## What passed

- Fresh bindings:all322accepted isolated files and26main inherited Python modules exact;
  the intervening user commit added only the approved preparation document/configuration.
- One mock-only invocation:9checks PASS, no target imports or real cache mkdir/rmdir.
- Full unchanged main baseline:389PASS/0failures/errors/skips, 93.561s unittest,
  99.647s child, 762454016bytes peakRSS; 34D3methods/35ledgers3564expanded cases.
  Overall resources, scope, owned-empty-cache cleanup and empty scratch PASS.
- Exact six-file exclusive transfer; no implementation/test/dependency edits or overwrite.

## Why main acceptance stopped

The sole post-transfer412invocation was interrupted by Pi's verifier, not a reported
model assertion. Live scratch-size polling raced normal temporary-file deletion:
`FileNotFoundError` from `rglob('*')` followed by `lstat()` on a deleted Parquet file.
The child was settled by the supervisor's finally path. Actual logs contain285started
methods/284completed OK, 34D3methods and3564successful expanded cases, but no footer and
no H test method reached. Final child time/RSS/exit receipt was not written; unknown,
not inferred passing. There is no completed412result.

The owned empty cache was removed correctly; one other EMPTY temporary test directory
remained after interruption. It is preserved, not adopted or recursively deleted.
Required-empty scratch and overall verification failed. Pi owns the monitor/preflight
omission. The unchanged H implementation is not established defective by this result.

## Current boundaries and next decision

**ESCALATE/task_only_main_H_REF_CPU_reference_not_accepted.** All old reports, including
the separate isolated H-REF ACCEPT and native task3606687c-beba-4d3f-bd24-03ae5238da37's
historical ESCALATE/failed verification, remain unchanged. No native reopening, replacement
task, receipt substitution or extension change. No automatic fix, rerun, cleanup or rollback.

HistoricalSDK4/native1/fullH4 plus2new main invocations =6H-associated full invocations;
all2main slots consumed, 0remaining. The interrupted run adds0successful Hsteps; prior
inferred72remain unchanged. A new Pi-only verifier correction, targeted mock controls
and one separately allocated412run are recommended ONLY for separate approval. No new
Worker/model change is presently justified. Detailed evidence/material recovery changes
and authorization requirements are in the RCA; no recovery has been implemented.

The transferred `training-harness.md` intentionally retains SDK proposal-era status
paragraphs; it is the exact accepted API/methodology artifact, not this main outcome.
This separate document is the current main-integration status.

**Deployment NOT_AUTHORIZED.** O-OFFLINE, GPU-READY, production initialization/physical
source collection, real-data profiling/comparative/final-test execution remain separately
gated. No commit, push, release or rollback was authorized or performed.
