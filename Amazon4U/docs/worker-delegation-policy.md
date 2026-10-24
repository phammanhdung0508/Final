# Implementation ownership and optional delegation

## Current user-approved policy

The user approved this change with the exact instruction:

> APPROVE, please remove mandatory Pi-agy delegation out of Amazon4U

One coding agent owns diagnosis, implementation, related tests, and local
verification by default. Pi may write and fix code directly within authorized
scope. A separate agy Worker, multi-agent system, or delegation-extension task is
not required. This applies throughout Amazon4U, including H-REF, O-OFFLINE, and
`multiautoresearch`.

The machine-readable policy is
[`../configs/worker-delegation-policy.json`](../configs/worker-delegation-policy.json).
The existing filenames remain stable for links. They no longer prescribe a
mandatory Pi-to-agy workflow.

## What this supersedes

This policy replaces the former native-agy implementation default, mandatory
planner/Worker role split, standing delegation preference, and prospective
`2 + 2 + 2` recovery process. No new corrective delegation task is required merely
to continue authorized direct work. Historical instructions that prescribe those
roles or channels are not standing requirements for future work.

The three previous policy documents are preserved in
`../data/reports/mandatory-delegation-retirement-v1/before/`.
Original task specifications, criteria, reports, decisions, failed attempts,
artifact hashes, and consumed budgets remain unchanged. This change does not
reopen terminal tasks or authorize further implementation or tests on a stopped
candidate. Existing task-specific holds and limits still apply.

## Direct work and verification

- Define the authorized scope, requirements, preservation rules, and checks.
- Keep one consolidated list of unresolved requirements for the deliverable.
- Diagnose and fix related defects together within the existing authority.
- Read complete changed functions and relevant callers, dependencies, and tests.
- Run the applicable authorized checks and inspect their actual results.
- Distinguish static inspection, executed checks, historical evidence, and work
  that remains unverified. Do not call self-review independent review.
- Retain any independent review required by an original task. A separate review
  at a meaningful milestone does not require delegating implementation.
- Do not claim acceptance when mandatory requirements or evaluations remain
  unresolved. No delegation receipt is required for ordinary direct work.

## Optional delegation

Delegation and second-model review remain optional. Neither is required for every
change, and this policy authorizes no new Worker or SDK calls.

If native delegation is explicitly selected and authorized, its original
submission, inspection, verification, limitation, revision, history, and acceptance
gates remain mandatory. Pi retains independent review and acceptance authority
for those tasks. Worker reports are untrusted evidence. Do not bypass a terminal
or exhausted task by relabeling it as direct work or creating a replacement task.

Direct SDK generation remains a separately approved exception. Each request
requires explicit user approval for its task, unresolved criteria, model and
channel, source and output paths, request ceiling, time and resource limits, and
independent review procedure. No native-agy failure is a prerequisite for ordinary
direct implementation. SDK approval is not inherited from historical requests.

Keep actual native invocations, SDK requests, and verification runs separate in
historical accounting. A prompt role name does not establish an execution channel.

## Standing autonomy for minor issues

User-approved on 2026-10-24: "I want agents to have the autonomy to resolve minor
issues, provided they can do so safely and within pre-approved parameters."

- Do without asking: in-scope noncritical setup and verification recoveries, small
  fixes inside an approved task, re-running an already approved check within its
  existing limits, and documentation corrections that do not change a decision.
- Ask first: new budgets or higher limits; frozen science, protocol or acceptance
  changes; real-data, GPU, Kaggle, profiling or held-out work; deleting or rewriting
  evidence; new packages, frameworks or dependencies; cross-project or shared-root
  changes; and stage, commit, push or deployment.
- Record each autonomous fix in the relevant document or report.
- Priority: implementation and regression testing, then Musical_Instruments profiling
  once its plan is approved.

## Unchanged boundaries

Frozen science, data integrity, leakage safeguards, original acceptance criteria,
security checks, project isolation, and applicable regression requirements remain
binding. No criteria are weakened to reduce coordination overhead.

This approval changes workflow policy only. It grants no O-OFFLINE correction,
main integration, full regression, real-data profiling, model training, GPU or
Kaggle execution, credential access, deployment, cleanup, commit, or push.

Task acceptance remains separate from deployment approval. Where held-out or
deployment evaluation is required, predeclare its policy and complete its gates.
Never retune automatically after held-out exposure. Without an applicable
predeclared policy and approval, deployment is `NOT_AUTHORIZED`.
