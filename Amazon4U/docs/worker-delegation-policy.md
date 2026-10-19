# Worker delegation, recovery and SDK fallback policy

## Authority, precedence and scope

This is the current user-approved **prospective governance policy** for Amazon4U,
including H-REF, O-OFFLINE and `multiautoresearch`. Machine-readable counterpart:
[`../configs/worker-delegation-policy.json`](../configs/worker-delegation-policy.json).
Exact authority and documentation-only verification:
`../data/reports/worker-delegation-governance-v1/`.

This correction authorizes documentation/configuration edits only: **no implementation,
native agy invocation, SDK generation, regression, experiment or deployment**. Policy
ceilings are not task or round authorizations. Existing tighter task budgets and safety
holds remain binding until appropriately and prospectively authorized otherwise.

For future channel/recovery decisions this policy supersedes historical preferences for
an "existing text-only channel". Earlier preparation documents and D2/D3/H-specific
approvals remain historical or scope-limited records, not standing SDK permission.
Original plans, criteria, reports, decisions, artifact hashes and consumed counts must
not be rewritten. Frozen science, security, project boundaries and execution gates are
not superseded. Preserve archived worktree instructions; explicitly supply current
governance in any later authorized handoff rather than editing historical snapshots.

## Roles and three distinct event types

- **Pi:** planning, orchestration, task scoping, handoff preparation, independent source
  and test review, independent verification and acceptance decisions.
- **Native agy:** implementation, corrections, tests and related documentation only within
  explicitly authorized paths, commands and resource/execution limits.
- **SDK text-only generation:** optional exceptional source of untrusted candidate text,
  not native agy and not an independently accepting reviewer.

Use these exact accounting labels:

| Event | Meaning | Accounting |
|---|---|---|
| `native_agy_worker_invocation` | Actual native agy CLI Worker invocation through the delegation extension | Native round and cumulative deliverable ledger |
| `sdk_text_only_generation_request` | Direct SDK/model generation, whether or not the prompt calls the role "agy" | Separate explicitly approved SDK request ledger |
| `pi_independent_verification` | Pi's independently authorized review/checks, including regression when permitted | Separate verification command/run/resource ledger |

A role name in a prompt does not launch agy or establish its account identity. An SDK
request has no native delegation task/inspection/verification receipt unless an actual
native invocation independently occurred; never fabricate or retrofit such receipts.
Started native invocations count even if they return no code, ask for context, fail in
orchestration, or encounter tooling problems. Classify the cause separately. A refusal
before native launch is a recorded blocker, not a fabricated native invocation. Issued
SDK requests, including failed/no-code responses, retain their own consumed accounting.
Pi verification is neither an implementation invocation nor a substitute for one.

## Default implementation channel

**Native agy Worker invocation is the default for substantive implementation tasks.**
Use the delegation extension's supported submission, inspection, original benchmark
verification and review workflow (`delegate_task`, `delegation_inspect`,
`delegation_verify`, `delegation_review`), with full evidence retrieval when needed.
Pi reads actual complete changed functions, relevant callers/dependencies/tests and
complete results; diff-first is never diff-only. Worker reports are untrusted.

Do not silently switch to SDK because it is convenient, context is missing, a tool fails,
or an earlier task used SDK. Repair Pi's handoff within existing authority or report the
actual blocker. Native delegation is not an OS sandbox or a promise to remove native
tools; declare permitted actions honestly and stop if required safety cannot be met.
Ordinary Pi-authored governance records and mechanical integration of reviewed patches
remain distinct from substantive implementation assigned to native agy.

## Maximum recovery structure: 2 + 2 + 2

| Round | Purpose | First invocation | Consolidated correction | If unresolved |
|---|---|---|---|---|
| 1 | Initial implementation | Attempt 1: initial implementation | Attempt 2: after Pi independent review | ESCALATE #1 |
| 2 | Targeted recovery of the existing candidate | Attempt 3: targeted continuation | Attempt 4: after independent review | ESCALATE #2 |
| 3 | Final recovery from unresolved findings | Attempt 5: targeted continuation | Attempt 6: final consolidated correction | ESCALATE #3 |

**Three rounds, maximum two native invocations per round, maximum six native invocations
per substantive task/deliverable.** These are maxima, not required counts and not blanket
permission for six calls. Individually authorize the task and each recovery round with
prospective invocation, time/resource, path, output and verification budgets. Keep both
a per-round counter and cumulative same-deliverable accounting. Do not fill unused slots
merely to reach two/six. Stop on ACCEPT, no credible resolution path, an authorization
boundary, or a critical safety issue. An early substantive stop still requires review.
The attempt numbers in the table label the maximum schedule. If a round uses fewer
than two calls, record the unused slots as unused: actual cumulative counts include
only started native invocations, not reserved slots. Do not fabricate consumption or
transfer an unused completed-round slot into another round.

Never reopen/reset a completed round's counter or erase failed attempts. Subsequent
separately authorized rounds continue the same candidate, criteria and task history;
new labels or task IDs must not disguise replacement tasks or evade cumulative limits.
The policy is not an instruction to change extension code, rewrite a stored specification,
set `max_revisions` to five, or bypass terminal/held-out/reopening gates. Use only supported,
explicitly authorized continuation mechanisms with linked immutable history. If a native
extension gate cannot express the proposed continuation, record that tooling boundary
and seek an operator decision; do not reset budgets or switch channels silently.
These are declarative governance obligations for Pi's round/cumulative ledger, not new
runtime enforcement or a claim that extension code implements automatic round release.

## Root-cause review after every substantive ESCALATE

Before proposing another round, Pi records:

1. Preserved candidate, original evidence/decisions and consumed budgets.
2. Substantiated root cause: Worker implementation; Pi handoff/orchestration;
   environment/tooling; verification; missing requirements; or another evidenced blocker.
3. Every remaining acceptance gap, with evidence and original criterion references.
4. What materially changes in the proposed next round, beyond simply trying again.
5. Whether further native agy work has a technically credible resolution path.
6. Additional scope, budget or execution approval needed beyond existing authority.

Worker implementation failures and Pi orchestration failures both consume actual native
invocations, but are not equivalent evidence of Worker capability. Preserve mixed causes
and uncertainty rather than guessing. Existing standing noncritical recovery discretion
remains usable only within its already granted scope, budgets and execution authority;
it does not release a new round, extra calls, experiments or SDK requests automatically.

A substantive ESCALATE is a distinct unresolved stop decision for the same deliverable,
linked to its round/evidence. Repeated status messages or documents of the same unresolved
decision are not extra ESCALATE occurrences. Do not reinterpret historic SDK decisions
as three new native recovery rounds.

## Mandatory recommendation after ESCALATE #3

After the third substantive ESCALATE, Pi performs a consolidated root-cause assessment
and presents a recommendation, with evidence, unresolved criteria, justification,
proposed scope/budget and consequences. The options are:

- Stop the task.
- Propose a separately user-authorized exceptional native agy continuation beyond the
  normal limit, preserving the cumulative ledger and explaining the exceptional basis.
- Recommend SDK text-only fallback under the explicit approval requirements below.

**ESCALATE #3 triggers review and recommendation, never automatic fallback execution.**
Native unavailability or a serious safety issue can justify recommending earlier fallback,
not executing it. A recommendation or an unused budget slot is not authorization.

## SDK fallback: explicit user approval before every request

Retain the existing SDK mechanism, but never use it as the default or rename it native
agy. Before **every** SDK fallback request Pi must obtain explicit user approval covering:

- Why native agy is unsuitable or unsuccessful, supported by actual evidence.
- The exact task and unresolved original acceptance criteria.
- The exact SDK model and channel/provider.
- Allowed source and output paths.
- Maximum number of SDK requests.
- Time, resource and verification limits.
- The independent review and acceptance procedure, including how non-native evidence
  will be recorded without inventing extension receipts.

A maximum request count is a ceiling, not inherited permission for a later request: each
request must be explicitly covered by user approval before it is sent. Old SDK approvals,
standing recovery discretion, tool errors and ESCALATE #3 do not supply that approval.
No automatic fallback, silent channel switch, budget transfer or inherited authorization.
Record SDK counts separately; they neither consume, replenish nor create native allowances.
All SDK-generated code remains an untrusted candidate requiring independent verification.

## Existing H-REF history is unchanged

The following application/verification-state snapshot is **as of policy adoption**,
not a live H status counter. Later separately authorized verification is recorded in
`training-harness-h-ref.md` and `../data/reports/training-harness-h-ref-v1/verification-v1/`.
That phase applied the existing candidate in isolation and ran412tests with24errors;
it added no SDK/native invocation or new round. Original four-SDK request history and
the adoption snapshot below are preserved,not reset or converted into native rounds.

The recorded H-REF sequence contains **four `sdk_text_only_generation_request` events**,
not four native agy invocations. Evidence is the original `execution.json` in:
`training-harness-h-ref-v1/initial/`, `revision/`,
`continuation-v1/implementation/` and `correction-v1/correction/`, under `data/reports/`.
Those records identify the SDK route and `native_agy: false`. Historical generic
"Worker" labels/count fields remain untouched; this is an additive terminology correction,
not a new native task ID, retroactive approval, acceptance or budget reset.

The four SDK requests remain consumed, no new native round/SDK request is authorized,
and the existing candidate remains unapplied/unaccepted pending its applicable gates.
The recorded H full dependency baseline is one `pi_independent_verification` regression;
there have been zero H candidate full regressions. Prior D1–D3 acceptance, counters,
reports and retired archives remain unchanged. O-OFFLINE remains separately unassigned.

## User-authorized final Pi H-REF deliverable review

For H-REF, the user explicitly authorized a separate append-only final Pi acceptance
record using existing independently verified evidence. See
`h-ref-pi-final-acceptance.md` and
`../data/reports/training-harness-h-ref-v1/pi-final-acceptance-v1/authority.txt`.
The hierarchy is **User authorization → Pi final acceptance authority → native agy
implementation/execution evidence**. Native agy is not the ultimate H-REF acceptor.

The final Pi record is **ACCEPT/task_only**, with deployment **NOT_AUTHORIZED**.
Assigned Pi-review/resource/adjudication duties are recorded as **fulfilled** only
with independent evidence, not out_of_scope. Original limitation/issue text and order,
all H1–H9 and budgets remain preserved. This disposition is authorized for the separate
Pi H-REF record; it does not alter the native tool schema or any original report.
Native task3606687c-beba-4d3f-bd24-03ae5238da37 remains historical ESCALATE,including
its failed original verification. No native ACCEPT,receipt rewrite,reopen/reassessment,
new delegation,extension change or bypass occurs. The historical outcome is not a
permanent veto over this explicitly authorized deliverable-level decision.

This is not blanket permission to disregard failed criteria or launch work. A substantive
unresolved requirement still blocks ACCEPT. No new Worker/SDK/implementation/regression/
main integration/experiment/deployment or Git action is authorized. Existing channel,
recovery,security and native-extension restrictions remain unchanged.

## Safeguards and reporting

Preserve frozen recommendation science/evaluation, accepted models/evaluator/D1–D3,
original criteria, complete evidence, project isolation and resource limits. Never weaken
criteria to save calls or force completion within six attempts. Review source and test
safety before execution; run only authorized original mandatory regression commands.
Missing, unsafe, stale, interrupted or unresolved mandatory evaluation blocks ACCEPT.
Adjudicate all current Worker limitations exactly/in order under the existing policy;
Worker scope labels do not waive assigned requirements.

Implementation, experiment execution, main integration and deployment require their
separate approvals. No Kaggle/GPU/training/profiling/deployment or automatic commit/push.
Task acceptance is not deployment approval; without a predeclared applicable deployment
policy the decision is NOT_AUTHORIZED. This document grants none of those permissions.

Report the actual channel, round, invocation/request counts, distinct ESCALATE decision,
root cause, preserved candidate/evidence, verification state and next authorization gate.
The intended loop is **Pi → Native agy (at most two per round) → Independent Verification**;
if unresolved, **ESCALATE → Root-Cause Review → Separately authorized next round**.
