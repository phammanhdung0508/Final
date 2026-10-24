# Independent evaluator integration review

## Recorded decision

- Task: `303fdf53-8c5b-4013-86be-4fec983c809a`, revision 2.
- Decision: **ESCALATE**, acceptance_scope: **task_only**.
- deployment_decision: **NOT_AUTHORIZED** (no deployment policy).
- Decision evidence: `agy-evidence-cfe0e25e-e123-4bde-be04-82cffc1f5598`.
- Project: Amazon4U. Implementation remains isolated at
  `.worktrees/evaluator-integration-review/Amazon4U/`; nothing copied into main source.
- Three Worker invocations timed out; declared two revisions exhausted.
  No automatic replacement task or source correction after this decision.

## Independent verification

Pi inspected diff-first, read actual complete source/functions/dependencies,
all copied accepted fixtures and integration tests, relevant original callers/tests,
and the Worker documentation and generated diagnostic JSON. Pi ran only the
original predeclared full-suite benchmark using the main project's Python with
`-B`, outputs and TMPDIR restricted to the declared clone output directory.

Result: **97 tests, zero failures/errors/skips**, 1.988 seconds.
Evidence: `agy-evidence-0cf92425-7f5b-44a1-80ff-34f215ab4a6a`, full output read.
Partition: 51 original core + 21 shared evaluator + 13 integration + 12 other.
The accepted standalone suite's previously reported 33 included those 12 other
fixtures; the copied shared file has 21 methods, not 33. All 63 baseline tests
remain, and the copied accepted fixture SHA-256 is unchanged:
`5d522eddc5c16bfd0b487a6dda84d900b5a99aec092471ed95f570d6fff837b1`.

Post-benchmark inspection: `agy-evidence-aef2ffee-adb5-47da-9759-40dac2062f27`.
It reported only five allowed source/test/doc files plus the declared result JSON,
no unauthorized monitored changes or Git metadata edits, and complete protected
hashing (151 files, 87,533,473 bytes). `git diff --check` passed. This is monitored
snapshot evidence, not an OS sandbox or exhaustive Worker execution evidence.
Worker timeouts prevent a complete action/execution claim. Reviewed fixtures and
the approved benchmark contain no training or real-data profiling.

## Acceptance blockers

1. **Public core report-schema regression.** Original core `Evaluator` always
   emitted both `empty_positive_history` and `unusable_metadata_profile` fallback
   by-reason entries, even with zero users (count 0 and all-None metrics).
   New shared `evaluate` enumerates observed reasons union only
   `empty_positive_history`; the core subclass does not extend that default.
   Thus `report['subgroups']['fallback']['by_reason']['unusable_metadata_profile']`
   is missing for empty/no-metadata-fallback populations. This violates the original
   criterion to preserve the public core API/defaults/statistics. The passing
   fixtures do not assert this absent-case schema.
2. Integration documentation falsely describes candidate shortages as raising
   `ScoringFailureError` (they are counted, without padding/errors), misstates the
   per-file test counts, and calls materialized baseline sentinels untracked when
   they are tracked clone files. It is a Worker draft, not accepted evidence.
3. Exact mixed KG fallback macro/subgroup/coverage and zero-count reason-schema
   fixtures need strengthening. Many new assertions only check not-None/counts.
   The scorer-owned overridable profile validator also merits explicit adversarial
   subclass/override review; no blanket security guarantee is established.
4. The incomplete Worker invocation histories are preserved as such; passing
   current source benchmarks does not establish exhaustive Worker execution.

Single-engine identity and candidate/history isolation are supported by source
review and passing fixtures. Guarded ordinary KG metadata fallback/rejection
fixtures pass. Neither those positives nor historical standalone acceptance
waives the report-schema regression or the unresolved scoped review findings.

## Preserved artifact hashes

- `scripts/recommendation_core.py`:
  `1bba78394628c763f1d01ad4d8cc66d37f4f3f4a540edeb0b5316683b7e0517e`
- `scripts/recommendation_evaluation.py`:
  `c6c8c7be5a9de590e8a137c43fb5650f53af9c2f2855c385c52187853af3775a`
- `tests/test_evaluator_integration.py`:
  `4b9934a06a93fdf79cd8cb5555757be6bc36d226fe03685305a858b081460c8d`
- Worker `docs/evaluator-integration.md`:
  `589aadd7881aedf8d09f3f95ae36719ed2df17e33aa0687df70b372a39ad8ad8`

## Sequence status

Step 1 is **not accepted or integrated**. Steps 2-4 have not proceeded past this
gate. No training/profiling, deployment, commits, pushes or rollback authorized.
The standing approval allowed routine setup and scoped revisions, not bypassing
acceptance gates or automatically replacing an exhausted task.

A future explicitly approved recovery would preserve this state and original
criteria, add a shared reason-schema extension preserving strict standalone
semantics and core's two default reasons, strengthen exact/absence fixtures,
correct draft documentation, and obtain fresh complete independent review.
Historical sparse preflight `d0a8d369-f283-4a35-8ad7-b27f30370e4d` remains
ESCALATE. Prior standalone task `0f2f0b7d-6e2a-4726-9e6e-83a73e8da423` remains
its original task-only ACCEPT/NOT_AUTHORIZED. Earlier broad I1/evaluator/audit
ESCALATE records are not superseded.
