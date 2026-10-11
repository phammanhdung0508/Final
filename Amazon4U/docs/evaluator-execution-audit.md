# Evaluator execution audit — Track A

## Status

Pi performed read-only investigation of the existing records for task
`a7b233bd-e2fc-4ceb-8119-ea62bbcd0508` after the user approved separate evidence-audit
and evaluator-hardening tracks. No agy rerun, infrastructure modification,
training, profiling, acceptance reversal or deployment occurred in this audit.
The task remains **ESCALATE**, acceptance scope `task_only` (unaccepted),
deployment decision **NOT_AUTHORIZED**. Track B has not been launched by this
investigation; it must remain a separate scoped worker task.

## Evidence recovered and independently checked

The prior revision-1 invocation timed out and originally had an unavailable
trace in its execution evidence. Its settled database can now be captured
read-only; this recovery does not turn the timed-out execution into completion.

| Revision | Conversation | Raw rows | Payload bytes | Navigation action candidates |
| --- | --- | ---: | ---: | ---: |
| 0 | `e4d42f7f-2694-4e81-8536-9c5e052832ad` | 72 | 1,360,995 | 35 |
| 1 | `ac4aaea0-b267-4144-88c2-bce2bb86803c` | 84 | 1,787,022 | 41 |
| 2 | `b3363eb1-e2f1-4c74-b596-f37e659f6366` | 60 | 1,249,136 | 29 |

Pi re-read the settled databases using the actual capture implementation via
`python3 -I -B`, with no generated files. Revision-2 database SHA-256 matches
immutable captured evidence:
`7ef1ab0be3159256845c755bc817fa5d52e846bee37f0ba2d2e5b5cc7d91d433`.
Revision-0 and revision-1 database image hashes observed in this audit:
`178e5a31675b2852321e92e4910b4feb164ab11fae16f7d2dd6cb2641c1f4bf4` and
`8c22551ccba5020b74cf5ec1bf0dab66938b12c59b016e8a65fad4c15d7af552`.

All three captures report contiguous raw rows and no child-trajectory flags.
Navigation extraction found four command candidates in revision 0 (indices
19/27/39/67), four in revision 1 (25/59/63/81), and one in revision 2 (49).
Those command strings match the original approved unittest wrapper and project
working directory. These are candidate observations, not a completeness verdict.
No prohibited command was identified among these command candidates.

Evidence references:
- Latest revision-2 capture: `agy-evidence-6c7a10b7-437d-40ef-9b57-af1b99b31745`.
- Recovered revision-1 capture: `agy-evidence-6090631d-8e5c-4c2a-b8ff-dfbaca695f98`.
- Original final review: `agy-evidence-ea4761f8-3d73-4aee-ac7a-326cbb7ef7e0`.
- Independent passing benchmark: `agy-evidence-4367d871-5636-45b3-a55c-a3314824ca6e`.

## Why semantic completeness remains unresolved

Pi read the complete `trace_capture.py` and `trace.ts` implementations under
`/home/sunf/.pi/agent/extensions/antigravity/` without modifying them.

The collector explicitly returns `interpretation_complete: false`; this is a
fixed limitation, not a failed evaluator regression test. It captures raw bytes
and extracts dictionary-shaped JSON from generic protobuf strings for step type
132. It does not use an authoritative versioned action schema. String extraction
has size/depth/node/byte limits and silently abandons unsupported wire content;
zero `candidates_omitted` does not prove that every executable action was found.
The observed extraction counters did not reach the global node/byte caps, but
that alone does not eliminate the other omissions or establish semantics.

Pi independently parsed all revision-2 top-level wire envelopes: row types were
14 (one), 90 (one), 15 (29), and 132 (29), with structurally parseable top-level
fields. Structural parsing and numerical type counts do not establish the
meaning of every nested field or prove absence of additional execution paths.

## Separate tooling implementation outcome

User approved a separate Amazon4U-scoped audit-tooling task; installed extension,
CLI and conversation database modifications were excluded. Task
`fb50c21d-766a-4771-b2a7-396200852f76` ran in the detached clean worktree
`/home/sunf/FSB/Final/Amazon4U/.worktrees/trace-audit/Amazon4U`.
agy implemented `scripts/audit_worker_trace.py`,
`tests/test_worker_trace_audit.py` and `docs/worker-trace-audit.md` there.
No evaluator hardening was bundled into that task.

Pi reviewed actual complete source/tests/documentation and relevant dependencies,
and independently ran the original approved full-suite wrapper. Initial result:
40 tests with one failure; revision 1: 54 tests passed; final revision 2:
**54 tests passed, zero failures/errors/skips**, unittest 1.883s, command 3.103s.
Final benchmark evidence: `agy-evidence-560bb327-1d76-4a79-b996-db09e923fcfb`.
Protected snapshot/cache/data monitoring and Git metadata showed no unexpected
changes; this remains bounded monitoring, not an OS sandbox.

The final worker revision timed out after source changes, before required
regression fixtures and matching documentation were completed. Dedicated tests
for same-size input mutations, hard-ceiling limits and lying cumulative budgets
remain absent; documentation's format rule contradicts the final source.
Pi therefore recorded **ESCALATE — task_only, unaccepted**;
**deployment NOT_AUTHORIZED**, review evidence
`agy-evidence-3964e413-ba61-4ab8-b859-3666fbe4cdee`.
Both revision slots are exhausted. No automatic replacement task or acceptance.
Artifacts remain isolated and preserved, not integrated into the original checkout.

The tool checks self-consistent raw bytes and outer wire framing, explicitly
refuses semantic completeness without authoritative schema, and cannot close
historical Track A. This is partial infrastructure progress, not a schema-backed
execution decoder. Track B remains unstarted; profiling/training remain paused.

## Required next decision

A reliable audit needs an authoritative schema/export for the recorded CLI
version, complete action/result classification across all invocation rows, and
fail-closed handling of unknown types, truncation and child trajectories. Merely
flipping the completeness flag, trusting command candidates, or rerunning the
worker does not resolve historical evidence.

If tooling implementation is necessary, request separate approval for that
infrastructure task and its writable paths before delegation. Do not edit the
installed extension, the CLI, or conversation databases under this audit.

Separately, Track B must protect the evaluator's authoritative candidate sequence
against provider clear/remove/append mutation and add adversarial regression
fixtures. An agy implementation fix cannot by itself resolve Track A. Preserve
all historical ESCALATE decisions and state exactly what any future acceptance
covers. Model work and Musical_Instruments profiling remain paused.
