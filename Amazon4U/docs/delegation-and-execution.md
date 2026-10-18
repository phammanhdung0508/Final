# Delegation and execution agreement

## Status and scope

User-agreed operating roles and delegation rules for Amazon4U, recorded from
our discussion. Applies to Pi and agy, including work under `multiautoresearch/`.
This document records workflow decisions, not implementation acceptance or
permission to launch experiments. Preserve the frozen scientific protocol and
consult `protocol-lifecycle.md` for authorization gates.

## Roles

- **Pi = planner + orchestrator + independent reviewer.** Pi defines coherent
  task scope, original acceptance criteria, approved verification commands and
  revision limits; reviews actual code, callers, dependencies, tests and evidence;
  independently verifies results; and retains ACCEPT/REVISE/ESCALATE authority.
- **agy = implementation and explicitly authorized execution worker.** agy
  implements code, fixtures and related documentation in the pinned isolated
  worktree, and fixes consolidated Pi review findings within the revision limit.
  Worker reports are untrusted submissions, not independent acceptance.
- After explicit execution authorization, agy may submit only the specified
  Kaggle job, monitor it and collect logs, metrics, checkpoints and provenance.
  Pi reviews the evidence and decides the next step.
- agy cannot independently change the protocol, expand scope or budgets, tune
  on test data, promote models, deploy, launch subagents or start another research
  cycle. Neither agent stages, commits or pushes without explicit user request.

## Standing user direction: delegate bounded implementation

User reaffirmed: Pi should prioritize planning, assigning work, managing execution,
receiving results, coordinating corrections and independent verification—not routinely
spend its context authoring straightforward implementation. Within an authorized
implementation scope, delegate clear, independently verifiable simple/medium programming
work to agy: utility functions, unit tests and local bug fixes, with explicit inputs,
outputs, allowed paths, preservation rules and acceptance criteria. Bundle related
utilities/fixes with their tests into coherent bounded assignments rather than one
request per trivial edit. Independent subsystems remain separately assigned.

Pi retains consequential design/security/protocol decisions, full necessary code review,
independent verification and final acceptance authority. Worker reports remain untrusted;
delegation does not reduce review rigor. Pi may handle brief governance records and
mechanical application/integration of reviewed Worker patches; exceptions involving
Pi-authored implementation must be explicit, not a default substitute for Worker work.
Use scoped relevant context without omitting original requirements or necessary callers.

This is a work-allocation policy, not new execution authority or a budget reset. The
current Training Harness remains preparation-only: no Worker launch, implementation,
new tests/optimizer work or remote execution until its separate scope is approved.
Existing controller-free text-only channel and all review/safety/deployment gates remain.

## Worker dependency and scope-change approval

Standing user direction: if agy proposes a package, framework, architectural structure
or other change not covered by its assigned task, it must **ask Pi before implementing
or installing it**. Existing approved dependencies/structures may be used as declared;
an undeclared dependency, broad refactor, new service or structural expansion is not
implicitly permitted because it seems useful. Pause the affected work and provide the
purpose/version or design, affected paths/interfaces, installation/network/resource/
security impact and proposed verification; do not code first and disclose afterward.
The text-only Worker returns this request without host actions or unauthorized tools.

Pi approves, narrows or rejects ordinary implementation choices within the user's
broader authorized project scope; no repeated user approval is needed for those
choices. Before affected work resumes, record the decision and prospectively declare
any approved paths/dependencies/outputs/commands and verification changes. Preserve
original requirements, reports and history; approval does not silently rewrite criteria,
reset response/revision budgets or authorize an extra Worker invocation. If the approved
mechanism cannot resume within current authority/budget, stop and record that boundary.

This discretion does not authorize violating frozen science, security/data safeguards,
project boundaries or ungranted material resource/real-execution permissions. Current
Training Harness work is still preparation-only. No automatic install, remote job,
training/profiling, controller redesign or new Worker follows from this policy.

## Delegation principles

1. Never lower acceptance criteria solely to save tokens.
2. Do not delegate tiny few-line changes separately; incorporate small fixes in
   an existing substantive task's revision cycle.
3. Do not bundle independently implementable/verifiable subsystems into one task.

Use one coherent subsystem with its fixtures and relevant documentation per
substantive task. Consolidate review findings and correction feedback. Save
context through targeted evidence navigation, never incomplete review:
diff-first is not diff-only. Exhausted revisions or incomplete mandatory evidence
require escalation, not a fabricated completion. The standing authorization below
permits bounded noncritical recovery after recording ESCALATE; it never permits
resetting revision limits to keep retuning an exhausted source task.

## Operating loop

1. Pi plans and predeclares scope, criteria, commands, permitted outputs and limits.
2. agy implements the scoped deliverable in an isolated worktree.
3. Pi inspects actual changes and independently runs approved original checks.
4. agy addresses consolidated findings within the declared revision limit.
5. Pi records acceptance or escalation, with explicit acceptance scope and
   separate deployment decision.
6. Only after a separate explicit execution approval, agy runs the authorized
   Kaggle workload and returns complete artifacts and evidence.
7. Pi reviews results before proposing any further work.

Task/benchmark ACCEPT never implies deployment approval. Where deployment or
held-out generalization evaluation matters, predeclare its policy before task
submission; complete mandatory evaluation before final acceptance. No automatic
revision or retuning after held-out exposure. Without a predeclared deployment
policy, deployment is NOT_AUTHORIZED.

## Standing authorization: noncritical recovery without repeated questions

Status: **user-approved**, following the verification-only scratch-output
ESCALATE. The user explicitly requested durable recording:

> Again, please take note somewhere incase you keep forget, i only care about final result like 'implementation + regression testing + profiling on Musical_Instruments' if the ESCALATE is not critical. no need to ask me every time. you can decide by yourself as long as follow the rule loop we agree on.

The delivery goal is **implementation + regression testing + profiling on
Musical_Instruments**, not repeated approval exchanges about administrative issues.
This supplements the approved implementation sequence in `implementation-roadmap.md`.

- Pi classifies and records findings using their actual impact. A formal ESCALATE
  does not automatically require interrupting the user. Noncritical infrastructure,
  scratch/output-layout, setup and verification-evidence issues may be handled by
  Pi autonomously, including a prospectively scoped fresh verification task.
- Preserve source and original evidence/decisions first; record the recovery scope,
  permitted outputs, exact commands, budgets and revision limits before execution.
  Then follow the same agy implementation / Pi independent inspection, verification
  and acceptance loop. Never waive an unmet criterion or claim incomplete work is
  accepted. Fresh recovery does not retroactively authorize previous writes.
- The current scratch-runtime issue is within this standing noncritical recovery
  authorization. Pi may preserve its files, separate/predeclare infrastructure
  outputs and fixture scratch, and arrange fresh verification without asking again.
  Historical `07318909-7b11-4a46-810a-c11d6c2c8b3a` remains ESCALATE; no
  evaluator rewrite is justified merely by that infrastructure finding.
- Keep recoveries bounded by the declared limits. Do not evade exhausted source
  revisions through replacement tasks or run indefinite retries. If the issue
  cannot be resolved within the agreed limits, explain the consequential decision
  that actually requires the user rather than repeatedly asking to rerun checks.
- Ask for decisions affecting scientific semantics, leakage/test use, data integrity,
  security, project scope, material resource/cost limits, or execution permission
  not already granted. Passing tests do not make such issues noncritical.
- The goal includes Musical_Instruments profiling, but execution still follows the
  existing prerequisite, bounded-plan and explicit execution-authorization gates.
  This is not blanket permission for real-data training, optimizer updates,
  comparative/held-out experiments, deployment, commits or pushes.

This standing authorization supersedes the need for a new user confirmation for
otherwise in-scope noncritical recovery. It does **not** supersede mandatory
verification, frozen scientific rules, acceptance authority or historical records.

## Subsequent Package H authority and stopped result

The user subsequently approved bounded H-REF CPU-reference implementation with device
interfaces;that specific approval superseded earlier preparation-only statements for
H only. See `docs/training-harness-h-ref.md`. The two permitted text-only responses are
now consumed with no files:requested frozen context,thenmissing total-loss return key
in Pi's fresh context. Fresh389baseline passed,butno Himplementation/newtests/updates
or candidate verification occurred. Pi records ESCALATE/notaccepted andowns the handoff
omission;this is not a verified accepted-model defect. No third request,Pi substitute
implementation,source task replacement orbudgetreset automatically. Originalrequests/
reports/criteria andD3history remainintact. O/GPU/Kaggle/profiling/maintransfer notapproved.

## Lifecycle structure and execution backend

We intend to adapt `multiautoresearch`'s pre-training, post-training and inference
separation and Kaggle orchestration rather than duplicate its control plane.
Kaggle is the intended GPU execution backend; local activity focuses on planning,
implementation, review and authorized synthetic regression verification.

Exact Amazon-specific stage mappings and runner adaptation remain to be agreed
and implemented. Optional self-supervised KG pre-training is not part of the
frozen baseline. BPR models trained from scratch must not be mislabeled as
fine-tuning. Lifecycle directory names do not grant execution permission.

Existing LM/Movie4U runners, metrics and agent prompts are not authoritative for
Amazon4U. Verify the actual Kaggle runner and reconcile conflicting instructions
before using them; preserve the original framework context rather than silently
claiming it has already been adapted.

## Execution boundaries and historical approvals

The original role agreement alone did not authorize workloads or new Worker tasks.
Subsequent implementation-sequence approvals and the standing authorization above
permit their stated scoped delegations and noncritical recovery. They do not grant
blanket profiling, model training, optimizer updates, real-data experiments,
deployment or unrelated framework changes. Those retain their respective execution
approvals and prerequisite verification. Prior escalated tasks remain unaccepted;
no historical decision is superseded by the new operating authorization.
