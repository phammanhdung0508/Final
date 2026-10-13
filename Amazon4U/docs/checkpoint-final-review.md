# Checkpoint recovery — final revision exhausted, not accepted

Decision: **ESCALATE / not accepted**.
Acceptance scope: **task_only (not accepted)**.
Deployment decision: **NOT_AUTHORIZED**.
Task: `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a`.
Final revision-2 decision: `agy-evidence-411171fa-c296-40bc-9346-667b9d7c5b97`.
Both corrective revisions are now used. No automatic retry, replacement task or
revision reset is authorized. The draft remains isolated; nothing transferred to main.

Historical revision-1 ESCALATE evidence:
`agy-evidence-267a7fbd-265f-4d48-8e45-76e8999552c6`.
This supplemental Pi review does not replace the task's immutable history.

## Historical review before supported same-task reopening

### User authorization and lifecycle blocker

The user explicitly requested the final corrective checkpoint revision and
independent ACCEPT/ESCALATE decision. Pi read the entire preserved revision-1
source (1,738 lines), tests (829 lines), and relevant manifest helper contracts,
then submitted consolidated final revision feedback using `delegation_review`.
The tool returned **"Task is already finalized"**: the earlier ESCALATE was
terminal. No revision-2 Worker was launched and the remaining numerical slot was
not consumed. Pi's earlier statement that this task could simply resume its last
revision was incorrect for this tool lifecycle.

`delegation_verify(stage="benchmark")` also refused with **"Task is not pending
review"**. Pi did not reset task state, edit shared tooling, create replacement
source tasks or revise the draft directly. Using the remaining slot requires a
supported reopening of this same task while preserving its history and budget;
project authorization alone does not provide a tool operation to do that.
This describes the earlier blocked attempt; the subsequently authorized final
revision is recorded below.

## Independent recheck

Pi ran the exact previously approved Python unittest command directly from the
pinned isolated Amazon4U checkout, with the original scratch directory and
`PYTHONDONTWRITEBYTECODE=1`. Result: **168 tests, 6 errors**, exit 1.
This is supplemental independent evidence, not a successful `delegation_verify`
record and not acceptance of an unexecuted final revision.

Errors: `test_atomic_no_clobber_exclusive_publication`,
`test_bounded_file_size_enforcement`,
`test_cpu_scope_scheduler_and_amp_absence_invariants`,
`test_optimizer_spec_and_moments_validation`,
`test_symlink_and_parent_symlink_refusal`, and
`test_zero_optimizer_steps_guarantee`.
Each uses a fixture manifest with best score 0.05 / selected epoch 2 while its
payload call uses None selection defaults. Fix the fixture consistency, not the
validation guard. Several intended negative tests can similarly pass by tripping
the selection guard before reaching the behavior they purport to exercise.

Fresh post-test inspection: `agy-evidence-51358e0b-9b75-46ce-800c-d107829ae87f`.
It reports the same two declared draft files, no monitored unauthorized writes,
no Git-metadata change, and complete protected/sentinel hashes. Empty Git diff
is expected for these untracked drafts; Pi read actual files, not just the diff.
No checkpoint source or tests were transferred into main.

## Criterion review and blocking findings

1. **Scope preservation: partially verified, overall unverified.** Monitored
   writes are limited to the two permitted draft files. Current tests/source use
   no optimizer steps or training loops. Historical Worker invocations timed out,
   so no claim of complete action-history verification is made.
2. **Canonical identity and reconstructable provenance: failed.** Training records
   remain optional; their pair set need not equal all observed history. Timestamp
   conversion truncates values. Source byte snapshots are optional in validation;
   Git dirty/diff/untracked evidence defaults to false/empty/clean rather than
   actual evidence. Loader source validation and expected provenance are optional.
   Required schema/bytes/hash/size/environment comparisons are incomplete.
3. **Safe complete payload validation: failed.** Optimizer hyperparameters and
   required moments/step are checked only if present; fractional tensor steps are
   truncated. Unexpected state at step zero and parameter-ID mapping are not fully
   validated. Python RNG element/range/Gaussian state and torch RNG length/state
   validity are incomplete. Sampler algorithm is not bound to an expected contract.
   Source/provenance validation can be bypassed via omitted optional arguments.
4. **Bounded/exclusive/path-safe IO: failed.** No-clobber publication and byte limits
   exist, but optional roots default broadly and lexical parent-path checks are
   vulnerable to replacement between validation and IO. O_NOFOLLOW protects only
   the final opened component, not mutable parent directories. Missing-parent
   creation/link publication must remain anchored to approved directory descriptors.
   Serialized-file size alone does not guarantee bounded reconstructed storage.
5. **Passing meaningful benchmark and accurate documentation: failed.** Six tests
   error; required `docs/checkpoint-recovery.md` and completed Worker report remain
   absent. Round-trip tests do not prove exact resumption. Shallow mapping proxies
   and a public token do not justify immutable/unforgeable-container claims.

## Required next boundary

Keep the draft isolated and original source/evidence intact. Do not grant ACCEPT
for a passing-fixture-only repair without completing identity/provenance/security
review and original criteria. No replacement task or revision-limit reset was
performed. A same-task tool-supported reopening would be required before using
the authorized final revision; shared tool changes are outside this Amazon4U task.
Main models/graphs integration is unchanged, with its prior 204 passing regressions.
Full harness, real-data training, profiling and deployment remain unauthorized.

## Final authorized revision and independent review

The user approved scoped shared-tool repair and explicitly confirmed reopening the
same task. Reopening preserved the original spec, criteria, paths, commands,
protected artifacts, history and max_revisions=2. Pi submitted consolidated REVISE
feedback; revision 2 launched in the original pinned clone. It timed out after
601,064 ms with SIGKILL/code null and ERROR/interrupted output, without a completed
Worker report. Execution evidence: `agy-evidence-0beacd61-cb89-49b0-b25e-866df395b428`.
No further corrective slot remains.

Pi followed the diff-first guide: the textual diff was empty because both drafts
are untracked, so Pi read the complete actual 1,738-line source and 829-line test
file, manifest dependencies/caller tests, TrainingContext and reproducibility
contracts. Final SHA-256s exactly equal the pre-revision reopened snapshot:

- Source: `cf48874330dc2f7207e7acd11d8a66de99df6abb12ccf4be9ff25bc2d2d47494`.
- Tests: `5d75fbe81716f7784d5dba0b3e457ad9118dcc3ff3ade098cf01256ad3c42c69`.

The Worker left these deliverables unchanged; required subsystem documentation is
still absent. Pi ran the original approved command through
`delegation_verify(stage="benchmark")`: **168 tests, 6 errors, exit 1**, unittest
runtime 3.065s (tool duration 5.734s), no output truncation. The complete immutable
raw log was read, including all six selection-fixture tracebacks. Benchmark:
`agy-evidence-402110be-9f88-4882-9c98-12a1fde2a51c`. No new source repair or optimizer
updates were performed by Pi.

Fresh post-benchmark inspection:
`agy-evidence-4224f211-2783-428a-b20b-15215b7d451c`. Only two declared draft files
are modified against baseline; unauthorized=[] and Git-metadata change=false.
Protected hashing completed for 218 files / 87,189,276 bytes; 130 outside sentinels
were monitored. No benchmark side-effect mutations were reported. This is not an
OS sandbox or a complete historical-action audit of the interrupted Worker.

| Original criterion | Final independent status |
|---|---|
| Scope / zero prohibited actions | Unverified overall: monitored scope clean and source/tests contain no optimizer loop, but interrupted action history is incomplete |
| Actual-content identity / reconstructable provenance | Failed: optional/incomplete records, truncated timestamps, optional source bytes/expected provenance and fabricated clean Git defaults remain |
| Complete safe payload validation | Failed: optional Adam moments/step/hyperparameters, fractional steps, incomplete RNG/sampler/ID contracts and mutable-container claims remain |
| Bounded exclusive/path-safe IO | Failed: parent-directory TOCTOU, broad optional roots, no decoded-storage preflight; intended boundary fixtures still error before IO |
| Passing meaningful benchmark / accurate documentation | Failed: six errors, missing doc/completed report; baseline passes do not establish subsystem acceptance |

Recorded final decision: **ESCALATE**, acceptance_scope **task_only (not accepted)**,
deployment_decision **NOT_AUTHORIZED**. No predeclared deployment policy, held-out
exposure or deployment evaluation exists. Prior escalations and frozen artifacts
are preserved. No checkpoint transfer to main, full-harness acceptance, training,
profiling, staging, commits or pushes follow. Further source work would require a
new explicitly approved consequential plan, not an automatic exhausted-task retry.
