# Proposed D3 execution-budget amendment v1

**Current status: APPROVED, prospective D3-only240-second ceiling.**
The user's subsequent **"Approve. please procced"** explicitly approves the proposal
below. Approval is frozen in
`data/reports/checkpoint-storage-implementation/budget-amendment-v1/approval.json`;
operative derived plan: `effective-plan-240.json` in that report directory.
Original proposal bytes, 120s plan/verifier/results and all budgets are preserved.
No retrospective acceptance: first run one fresh unchanged uninstrumented 355-test
baseline, then proceed Worker-led only if it passes.

The proposal-time statements below are retained as history, superseded only by this
explicit execution-limit approval, not by any substantive criterion or safety waiver.

The user recommended closing the runtime investigation, proposing a larger D3
execution ceiling, establishing an unchanged uninstrumented 355-test baseline under
the approved new ceiling, then proceeding with Worker-led D3 under IO1-IO8 if it passes.
The user explicitly described 240 seconds as a **proposed upper limit**, not expected
runtime, and required formal approval/documentation before an increase is used.
This record requests that approval; it does not infer it from the recommendation.

Companion: `configs/checkpoint-storage-execution-budget-amendment.json`.
Original D3 plan and historical 120s receipts:
`data/reports/checkpoint-storage-implementation/plan.json` and `review.json`.

## Close the runtime investigation

Conclude the investigation on the existing evidence. No more diagnostics, profiling,
C++ backtraces, sudo operations, thread-policy experiments or accepted-test edits.

The single authorized diagnostic measured the dense matrix at73.97s wall/73.72s CPU;
the optional-name matrix started at110.26s and had 9.74s before cutoff. These data do
not establish a hung/faulty optional-name test. Instrumentation overhead was not
calibrated; no claim is made that this explains every historical timing difference.
D2 remains concluded and accepted. Its implementation, tests/assertions, accepted
artifact documentation, acceptance decisions and original 120s evidence are unchanged.
Investigation evidence remains in `full-suite-diagnostic/` under the D3 report directory.
Closing investigation is not D3 acceptance or proof of an exact root cause.

## Exact amendment proposed

| Item | Existing limit | Proposed limit after approval |
|---|---:|---:|
| Fresh D3 dependency baseline: all 355 existing tests |120s |**240s per full-suite invocation** |
| D3 initial/revision/independent full regressions: all 355 plus additional D3 tests |120s |**240s per full-suite invocation** |

240 seconds is a hard execution **ceiling**, not a target, runtime prediction or
promise the suite will finish. It applies to the entire selected suite, including
interpreter startup, discovery and the additional D3 tests, not 240s per test or only
the original 355. Ordinary before/after evidence snapshots are separate bookkeeping,
not extra time granted to an executing test process. Termination/cleanup is recorded,
not permission to continue executing tests beyond the deadline.

The amendment is **D3-only and prospective**. It changes the execution-budget portion
of IO8 and the corresponding prospective planned-evidence limit, not the semantic
requirements of IO1-IO8. Preserve original criterion/plan texts and append the approved
amendment as explicit authority for future versioned commands/receipts. No historical
run exceeding 120s becomes compliant with its original 120s contract. D2's accepted
original criterion, prior timeouts/ESCALATEs and the non-acceptance diagnostic remain
exactly as recorded.

No other permission changes: no source/test/assertion weakening, subsets/skips,
new production allocation limits, unsafe IO fixtures, thread/environment retuning,
new controller/profiler work, deployment, live restoration, training/profiling or
main transfer. Preserve the existing Worker request/revision bounds, consumed slots,
execution history and bounded verification-run limits; no resets or indefinite retry.

## Execution sequence only after explicit approval

1. Freeze the approval and this amendment, preserving original 120s records. Declare
   a versioned verification receipt/entry point using the same Python, uninstrumented
   full-discovery body, pinned project/scratch, fixture limits and preservation checks;
   only the approved wall-clock ceiling changes. Do not overwrite the original plan
   or120s verifier/results. No instrumentation or resource/thread policy change.
2. Run **one fresh full 355-test baseline**, with all existing implementation/tests and
   assertions unchanged, under the 240s ceiling. Retain complete streams, exit/timeout
   status, elapsed time, identities and stable before/after snapshots. A pass requires
   completion, exit 0 and zero failures/errors/skips; this gate is preparation only.
3. If it passes, proceed with the already authorized **Worker-led D3 implementation**
   within IO1-IO8, unchanged text-only channel and bounded response/revision budget.
   Worker may propose technical solutions; Pi independently reviews complete source,
   callers/dependencies/tests and actual guarantees before executing safe fixtures.
4. Run the entire suite with added D3 tests under the same 240s ceiling. Preserve
   the original 355 tests/assertions; do not impose a355-test-only selection or a
   fabricated final test-count target. Passing baseline alone does not accept D3.
5. If the baseline or a mandatory later evaluation is incomplete/failing, preserve
   that result and identify the specific issue. No automatic further timeout increase,
   new task/budget reset, test pruning, unsafe test, or source retuning is authorized.

This proposal includes **no new execution now**. The current 120s ceiling remains
operative until explicit approval. Fresh baseline and Worker launch remain pending.

## Requested formal approval

> Approve D3 execution-budget amendment v1: a240-second wall-clock ceiling per full-suite invocation, covering the fresh unchanged 355-test baseline and subsequent full D3 suites including added tests. Keep implementation/tests/assertions, IO1-IO8 substantive requirements and all other safety/review/fixture/response-budget boundaries unchanged. Preserve original 120-second evidence without retroactive acceptance. After a passing fresh baseline, proceed with the already authorized Worker-led D3 process.

Acceptance scope remains **task_only** if later earned; deployment **NOT_AUTHORIZED**.

## Subsequent execution receipt, not D3 acceptance

Under the explicit approval above, fresh unchanged 355 baseline passed 55.645s process.
The initial full suite with added D3 tests completed 379 tests in55.993s process, with
1 failure/2 errors/0 skips; all 355 inherited tests and 3564 expanded cases pass. The one
consolidated revision is unapplied/unexecuted because its source baseline hash is
malformed; assigned IO2 parent-race refusal also remains unresolved. Initial+revision
responses are both consumed. **ESCALATE/task_only/not accepted**, deployment
NOT_AUTHORIZED: `data/reports/checkpoint-storage-implementation/final-review-evidence/review.json`.
No further automatic run, Worker, source/hash fix, limit increase or budget reset.
The original proposal and 120s outcomes above remain history, not retroactively accepted.
