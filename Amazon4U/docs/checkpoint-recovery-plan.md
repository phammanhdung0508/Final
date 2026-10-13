# Training harness recovery — checkpoint/provenance subsystem

Status: **final ESCALATE / not accepted**, task `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a`.
Final decision: `agy-evidence-411171fa-c296-40bc-9346-667b9d7c5b97`.
Both revisions are used. Original approved benchmark: 168 tests, 6 errors, 3.065s;
`agy-evidence-402110be-9f88-4882-9c98-12a1fde2a51c`. The operator reopened this same
task after approved shared-tool repair; final revision 2 timed out without changing
the drafts. Missing doc/report and identity/provenance/state/IO failures remain.
Nothing transferred to main. Acceptance_scope: task_only (not accepted);
deployment_decision: NOT_AUTHORIZED. No automatic retry/replacement/budget reset.
See `checkpoint-final-review.md` for all five criterion outcomes and fresh inspection.

## Preserved earlier recovery history

Historical revision-1 decision: `agy-evidence-267a7fbd-265f-4d48-8e45-76e8999552c6`.
Initial Worker and corrective revision 1 timed out; one of two revisions used.
Latest approved benchmark: 168 tests, 6 errors, 2.379s; evidence
`agy-evidence-041357ae-6be6-4d3f-b42b-6177ac52d694`. Required subsystem doc/report
is missing and final semantic/security review is incomplete. Draft preserved in
`.worktrees/harness-evidence-review/Amazon4U/`; nothing transferred to main.
This bounded stop does not exhaust the revision budget or justify replacement source
tasks. Continuation must first review the revised snapshot and retain the same task,
original criteria, commands and remaining single corrective revision.

The user subsequently explicitly authorized using the final checkpoint revision,
independent verification and a final ACCEPT or ESCALATE decision. Pi has read the
complete revision-1 source (1,738 lines), complete tests (829 lines) and manifest
caller/dependency contracts. The final feedback covers inconsistent selection
fixtures, incomplete mandatory provenance/source bytes and Git evidence, actual
training-record completeness, optimizer/RNG/sampler validation, IO path safety,
and missing documentation/completion report. Pi attempted revision 2 of the same task, but the tool refused with
"Task is already finalized": the earlier ESCALATE was terminal. No final revision
ran and no replacement task or budget reset was attempted. The earlier claim that
the same task could simply resume was incorrect for this tool lifecycle. The
verification tool also refused because the task is not pending review. Pi directly
reran the exact previously approved command: 168 tests, 6 errors. Decision remains
ESCALATE, task_only not accepted, deployment NOT_AUTHORIZED. A supported same-task
reopening is needed to use the numerical revision slot; no such operation is
provided by the current tool interface. See `checkpoint-final-review.md` for the
complete revised-source criterion review and supplemental evidence.

## User-approved shared-tool repair

The user explicitly approved scoped inspection/repair of the personal delegation
extension, followed by same-task reopening and the final revision. The extension
now implements an operator-only, UI-confirmed `/delegation-reopen` command. Local
repair review is ACCEPT, limited to shared tooling/synthetic tests (164 passed,
zero live Workers; deployment NOT_AUTHORIZED). It preserves task/history/spec/budget,
refuses exposed/exhausted/unsafe/stale tasks and requires corrective REVISE before
fresh verification/ACCEPT. Evidence:
`~/.pi/agent/extensions/antigravity/.runtime/reopen-repair-review.json`.

The current Pi runtime must `/reload` before this command exists. No checkpoint
reopening or final Worker revision has run yet. After reload, operator command:
`/delegation-reopen e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a User approved final checkpoint revision; preserve original scope and budget`.
The command prompts for confirmation and requests Pi continuation on the same task.
The one remaining final revision and all independent-review gates remain intact.
No ACCEPT for this subsystem or the full harness. The user's continue request resumes
that sequence; it does not authorize real-data training or profiling.

## Final revision authorization and session prerequisite

The user explicitly authorized using the single remaining corrective revision to
resolve the six fixture errors, missing subsystem documentation/completion report,
and incomplete final review, then rerun independent verification and let Pi decide
ACCEPT or ESCALATE. Original criteria, commands, limits and task identity remain
unchanged; no replacement source task or revision-budget reset is authorized.

In the current session, `delegation_status` returns `Unknown task` for
`e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a`; historical evidence references are also
unavailable through the delegation tools on this branch. The final revision has
not been launched or consumed. Resume the original task-bearing session
`01a13578-e189-72ed-8ab2-3a3a5a5bc712` (session file dated
`2026-10-13T13-34-36-937Z`) before submitting consolidated REVISE feedback.
Do not inject historical state, create a replacement task, or claim new acceptance
from chat/log summaries. Scope remains task-only; deployment NOT_AUTHORIZED.

## Final authorized continuation result

The task-bearing session was resumed, shared-tool repair loaded and the operator
confirmed reopening. The earlier unknown-task/lifecycle prerequisites above are
historical, not current blockers. Pi sent the consolidated corrective feedback
within the existing task; revision 2 consumed the final slot. The original
benchmark was independently run through the verification tool and failed. Pi
read all actual source/tests and relevant dependencies; final decision is ESCALATE,
not incomplete review awaiting another automatic source revision. Any consequential
follow-up requires a new user-approved plan preserving this exhausted history.
Training-loop/continuation work must not treat this checkpoint draft as accepted.

## Scope and boundaries

Implement one standalone CPU checkpoint/provenance subsystem in a pinned isolated
Amazon4U checkout. Deliverables: `scripts/checkpoint_recovery.py`,
`tests/test_checkpoint_recovery.py`, `docs/checkpoint-recovery.md`.
Do not revise model/graph source, frozen protocols or earlier evidence. Do not
implement an epoch loop, validation selection engine, optimizer update, data loader,
profiling runner or held-out experiment. This task provides serialization/validation
contracts only; exact optimizer continuation and full harness integration belong to
the subsequent training-loop subtask.

Frozen rules: `docs/reproducibility-and-uncertainty.md`, `configs/reproducibility.json`,
`docs/training-semantics.md`, and `configs/training-semantics.json`. Existing manifest
helpers are reference context, not adequate proof of complete resumability.

## Required contracts

- Identity must fingerprint actual permitted training records, positive/observed
  histories, warm IDs, typed graph/metadata and validation-selection inputs, not
  their counts. Preserve ID namespaces and deterministic canonical ordering;
  duplicate/malformed identities fail. Record resolved scientific settings and
  protocol/execution identifiers. A same-size/count identity change must reject resume.
- Source provenance is explicit allowlisted relative source/config snapshots with
  SHA-256s, tracked-diff/dirty provenance and reconstructable untracked bytes, plus
  supplied environment/version/hardware records. Never automatically capture
  credentials, arbitrary directories, siblings or generated datasets. Require
  nonempty valid fingerprints and mandatory provenance; verify claimed hashes.
- Serialize a versioned safe payload containing model, optimizer, RNG/sampler,
  epoch/step, best validation score/selected epoch and early-stopping counters.
  Scheduler/AMP are explicitly absent in CPU scope, not silently omitted when used.
  Use `weights_only=True` for loading; never fall back to unsafe pickle or invoke
  shell/eval. Fail closed on missing/extra schema fields, invalid types, nonfinite
  values, wrong shapes/dtypes/devices, unsupported versions, identity mismatch,
  and corrupt/truncated files. Only validated payloads are eligible for restoration;
  do not mutate a live model/optimizer before full validation.
- Publish checkpoints without overwriting completed artifacts. Use atomic exclusive
  publication; cleanup only owned temporary outputs. Bound file size before loading,
  validate paths/symlinks and do not traverse outside caller-approved output roots.
- Preserve current epoch-boundary-only capability label. Do not claim exact
  resumption until the later continuation oracle tests pass.

## Verification limits

CPU synthetic fixtures only: at most 8 users, 12 products, 8 metadata nodes/type,
64 dimensions; **zero optimizer steps**, no actual training, no CUDA. Test fixtures
may serialize small state mappings/CPU tensors and RNG state, including a synthetic
optimizer-state mapping, without running optimizer updates. No datasets accessed.
Maximum new fixture checkpoint 2 MiB, suite 120 seconds. Fresh generated outputs
only `Amazon4U/.test-output/checkpoint-recovery-v1/` in the isolated checkout;
all source, tests, baseline artifacts, Git metadata and sibling sentinels protected.
Use the predeclared original unittest command; never run compileall or create caches.

Pi runs baseline and independent approved benchmark, reads complete changed
functions plus callers/dependencies/tests, and records each criterion separately.
Up to two consolidated corrective revisions before exposure; no deployment policy
or held-out commands. Acceptance remains **task_only**, deployment
**NOT_AUTHORIZED**. Mandatory incomplete evidence requires ESCALATE; historical
failures are preserved. Main transfer follows independent acceptance and a separate
integration regression check.
