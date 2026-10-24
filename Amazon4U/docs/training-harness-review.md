# Actual training harness, independent review outcome

Status: **ESCALATE / NOT ACCEPTED**. Lifecycle remains
`IMPLEMENTATION_REGRESSION_VERIFICATION`. This note does not authorize real-data
training, profiling, Kaggle, or deployment.

## Scope and evidence

- Task: `12299217-65f9-43dd-a96a-b77fd9200251`.
- Final decision: `agy-evidence-d45dd722-ed99-4004-979b-554b13b6073c`.
- `acceptance_scope`: `task_only` (no ACCEPT recorded).
- `deployment_decision`: `NOT_AUTHORIZED`; no deployment policy or held-out evaluation.
- Isolated clone: `Amazon4U/.worktrees/training-harness-review/Amazon4U/`.
- Drafts: `scripts/training_harness.py`, `tests/test_training_harness.py`,
  `docs/training-harness-implementation.md` in that clone only. None transferred
  into main implementation.
- Initial invocation and both allowed revisions timed out. Revision ceiling is
  exhausted; no automatic replacement task or execution expansion follows.
- Fresh independent benchmark: `agy-evidence-f539dc8d-51f8-41e9-af17-a4bbbdff7d94`:
  **159 tests passed**, no failures/errors/skips, 3.796 seconds unittest runtime.
  This is the clone's 140-test committed baseline plus 19 draft tests, not the
  main 163-test suite and not verification of integration into main.
- Original command, from clone's Amazon4U directory:
  `PYTHONDONTWRITEBYTECODE=1 /home/sunf/FSB/Final/Amazon4U/.venv/bin/python -m unittest discover -s tests -v`.
- Fresh inspection: `agy-evidence-57789041-9a4f-4dd8-935b-f22e329f1ccd`;
  three new files only, no unauthorized changes or Git-metadata mutation detected;
  sibling read-only sentinel hashing complete.

## Blocking findings

1. Heterogeneous category hierarchy parses delimiter strings instead of canonical
   KG-v1 JSON path-prefix identities; malformed metadata/masks are not reliably rejected.
2. Resume identity compares counts rather than hashes of actual training history,
   warm pool, typed graph, metadata and validation selection inputs.
3. Safe `weights_only=True` deserialization exists, but checkpoint semantic validation
   and required provenance are incomplete; empty hashes and missing source/environment
   evidence are accepted.
4. Terminal resume can reference an uninitialized epoch; input/CPU/determinism guards
   and skipped-example reporting need correction.
5. Claimed mathematical invariance/self-only tests merely assert finite results.
   Strict early stopping, selected weights, final optimizer state/order/RNG continuation
   and metadata-neighbor ranking gradients lack adequate independent oracles.
6. Draft documentation overstates exact resumption and complete provenance.

Pi read the complete draft source/tests/documentation and relevant existing callers
and dependencies. Passing tests do not resolve these correctness/integration gaps.
Worker timeouts leave execution-history completeness unverified; do not claim a
complete audit of all Worker actions. Independent tests were prospectively scoped to
small CPU synthetic fixtures only (8 users, 12 products, 8 metadata nodes/type, 64d,
12 epochs/fixture, 200 steps/suite, 120 seconds/suite; fresh generated fixture outputs).

## Preserved state and next decision

Existing main staged Step3b work remains intact. No new staging, commits or pushes
were performed in this harness attempt. Movie4U source was not changed. Prior setup
ESCALATEs (`a36fc1dd-9249-4f62-8770-7dc7530ae532`,
`f8d3819a-19be-408e-b2d3-90d5bdb4bb9c`,
`674e78f7-499d-4e4e-851d-c80a5ea8cdcb`) remain historical failures.

## User-approved smaller-scale recovery

The user explicitly approved proceeding with the smaller-scale recovery plan after
this ESCALATE. The original task and setup failures remain unaccepted; this approval
permits new bounded subsystem tasks, not indefinite retries of the exhausted bundle.

Sequence:
1. Differentiable models and training-only graphs, with meaningful numerical,
   hierarchy, masking and gradient oracles. No optimizer loop/checkpoint code.
2. Safe checkpoints, full identity/provenance validation and reproducible state.
3. Training loop, validation/checkpoint selection and exact continuation tests.
4. Independent integrated regression review before any main-code transfer or
   declaration that the harness is accepted.

Each task retains relevant original criteria, pinned clean clone, approved original
commands, bounded synthetic CPU fixtures, up to two corrective revisions, independent
Pi file/dependency review and task-only acceptance. No reduction in correctness gates.
Small synthetic forward/backward fixtures are allowed; actual optimizer tests in later
subsystems retain the previously scoped fixture/step/time/output bounds and must be
predeclared. No real-data training/profiling/Kaggle/deployment/commits/staging/pushes.

First recovery implementation uses a fresh clean clone under
`Amazon4U/.worktrees/harness-model-recovery/`, preserving the rejected draft. It adds
only `scripts/differentiable_models.py`, `tests/test_differentiable_models.py`, and
`docs/differentiable-models-recovery.md`. The clone is at committed Step3a baseline;
main staged Step3b files are not changed or copied over. The first benchmark therefore
must preserve its 140 baseline tests; main integration later must preserve all 163.
No optimizer steps or checkpoint serialization in this first subsystem. Required
model tests use at most 8 users/12 products/8 metadata nodes per type, 64 dimensions,
CPU and 120 seconds per suite; outputs only the fresh declared generated directory
`.test-output/differentiable-models` (prefer no file outputs).

### First recovery submission: infrastructure-blocked

Task `8a192755-46dc-407d-b395-c4db6742514a` could not start a Worker turn:
configured `gemini-3.1-pro-high` is no longer recognized by agy. Execution evidence
`agy-evidence-8da7f193-681a-468d-96bf-0ce619ae6c22` lists supported choices:
Gemini 3.8 Flash (High/Medium/Low), Claude Sonnet 4.6 (Thinking), Claude Opus 4.6
(Thinking), GPT-OSS 120B (Medium). The delegation tool exposes no model-selection
parameter; changing shared harness configuration is outside Amazon4U file scope.

Fresh inspection showed zero changes. Independent original benchmark preserved
140 baseline tests (1.595 seconds, OK), evidence
`agy-evidence-9717c5d6-cfc1-4fca-84b8-08d0faeadbd9`. Decision remains **ESCALATE**,
`agy-evidence-acdb65f7-55a4-4073-9e08-37d56632a8ca`, task-only scope, deployment
NOT_AUTHORIZED. No model/graph deliverable exists. Fix Worker model selection
before retrying under the approved bounded recovery; repeated retries with the
invalid configuration are not justified.

The user subsequently explicitly authorized switching the Worker to
`gemini-3.8-flash-high`. Pi changed only the two current runtime model pins in the
personal extension's `core.ts`, the corresponding four test files, and current
configuration documentation. Historical evidence/probes retain their original
identities. The local extension regression suite passed **144/144 tests** with no
model calls. This verifies the source change, not live Worker availability.
The running Pi extension still requires `/reload` (or restart) before delegation
can use the new pin. Resume the already-approved model/graph recovery after reload;
do not silently change criteria or mark the previous startup task accepted.

### Models/graphs recovery result after model switch

Task `ca73a153-6f02-45f3-b5d0-c4e93fffefb7` used the requested
`gemini-3.8-flash-high`; revision 1 completed successfully. The final revision 2
implemented additional parameter/adjacency validation and gradient/numeric tests,
but was killed at the 600-second Worker timeout before returning the required report.
Both source revisions are now exhausted; no automatic model retuning follows.

Fresh original-command Pi benchmark: **180 tests passed**, no failures/errors/skips,
1.624 seconds unittest runtime (140 committed clone baseline + 40 new tests), evidence
`agy-evidence-e7ad41dc-0663-43a2-874d-4f3271a795bd`. This is not main's 163-test
integration suite. Actual code review examined complete model operators, graph
construction, mathematical/masking/gradient fixtures and relevant dependencies.
Full final completion evidence and some final whole-snapshot coverage remain pending.
No optimizer loop or checkpoint code was run/implemented in this subsystem's scope.

Fresh inspection `agy-evidence-e9756a57-f386-4251-9c8d-bf73c6af59e5` captured
three new permitted files, no unauthorized/Git-metadata mutation, and complete
read-only sentinel hashes. The draft is preserved under the approved recovery clone.
No draft source/tests/docs were transferred into main implementation.

Final **ESCALATE / task_only / NOT_AUTHORIZED** decision:
`agy-evidence-7068a797-20b6-4102-a174-1bad8ba39466`.
Passing tests do not override the incomplete mandatory execution/report gate. Preserve
this tested snapshot; a future explicitly bounded evidence-completion plan should
finish independent final review and completion evidence without rewriting/retuning
models or retroactively accepting the interrupted task. Checkpoint/provenance and
loop subtasks remain pending. Main staged Step3b work and Movie4U remain unchanged.

## User-approved frozen verification-only recovery

The user approved proceeding with verification-only recovery and asked to continue
until final results. Historical source task `ca73a153` remains ESCALATE; there is no
new source-revision budget. Pi froze a byte-identical 87-file candidate (scripts,
tests, docs, configs, AGENTS, README, requirements) in the standard ignored report
location of a clean clone:
`Amazon4U/.worktrees/harness-evidence-review/Amazon4U/data/reports/harness-candidate/`.
Its sibling `harness-candidate-manifest.json` records all SHA-256s, compared directly
to the preserved source draft. No Git staging/commit is needed to freeze this artifact.

New recovery is read-only with empty allowed/generated paths; the candidate directory
and manifest are explicitly protected ignored artifacts. Worker only reads/reports;
Pi independently reviews actual candidate code, validates all frozen hashes, and runs
the same original unittest command from the candidate directory. External fixture
scratch is predeclared at `.worktrees/harness-evidence-temp/`, initially empty, outside
the clean clone. No optimizer steps, source writes, real data, retuning or held-out
execution. Acceptance, if earned, applies only to verification of this frozen
models/graphs snapshot; it never retroactively accepts the interrupted source task
or the complete training harness. Subsequent subsystem tasks retain their own gates.

Do not treat the previous draft or this approval as accepted or execution-ready.

## Frozen verification accepted; main model integration reviewed

Following the user's continue request, frozen verification task
`f709d062-ae0e-4eb6-960c-f94728180d1b` received **ACCEPT / task_only /
NOT_AUTHORIZED**, decision `agy-evidence-bd7419a4-e9bb-4669-b261-4fb3064d5900`.
Pi's approved benchmark ran **180 tests**, zero failures/errors/skips
(`agy-evidence-118bdab1-f944-443c-99e6-1b9410fe28f6`). Independent manifest
verification found exactly 87 files with no missing, extra or mismatched hashes.
Fresh inspection reported no monitored mutations. Setup-only preflight failures
remain historical; no model rewriting/retraining took place in the frozen task.

Pi transferred only the new model source, tests and subsystem documentation into
main Amazon4U, preserving Step 3b. The first main suite found a namespace-package
import compatibility regression absent from the older clone baseline. Pi repaired
package-attribute backfilling in the main copy (not the frozen candidate), added a
fresh-process regression for both import orders, and reran all main tests:
**204 passed**, no failures/errors/skips. Main integration evidence and exact
source/test hashes are in `data/reports/differentiable-models-main-integration/`.
This is a separately recorded Pi-local integration outcome, not an expansion of
the delegated frozen-snapshot ACCEPT.

Historical source task `ca73a153` and the original bundled harness remain
**ESCALATE**. Models/graphs verification and main integration do not accept the full
training harness. Remaining recovery: checkpoint/provenance subsystem, training-loop
and continuation tests, then integrated harness review.

### Checkpoint/provenance attempt, isolated draft not accepted

Task `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a` received **ESCALATE / task_only /
NOT_AUTHORIZED**, decision `agy-evidence-267a7fbd-265f-4d48-8e45-76e8999552c6`.
Initial Worker and corrective revision 1 each hit the 600-second timeout, with no
completed report. Initial independent benchmark: 162 tests, 3 errors. Revised
benchmark: **168 tests, 6 errors**, 2.379s, evidence
`agy-evidence-041357ae-6be6-4d3f-b42b-6177ac52d694`. The six errors are inconsistent
selection fixtures: manifest best score/epoch are present while payload defaults
are None. The consistency guard must not be weakened to silence them.
Required `docs/checkpoint-recovery.md` remains absent. Pi fully reviewed initial
source/tests and sent consolidated identity/provenance/state/IO security findings;
full revised-source review and mandatory completion evidence are unfinished.

No checkpoint files were copied into main. Isolated drafts remain in
`.worktrees/harness-evidence-review/Amazon4U/scripts/checkpoint_recovery.py` and
`tests/test_checkpoint_recovery.py` there. Inspection found no monitored
unauthorized or Git changes. **One of two corrective revisions used; one remains**:
this is a bounded stop, not revision exhaustion. Further work must retain this
task/budget, finish review and use actionable consolidated feedback; do not create
replacement source tasks to reset limits. See `checkpoint-recovery-plan.md`.

The user subsequently authorized the final corrective revision, but the tool
refused: **"Task is already finalized"**. No revision-2 Worker was launched. The
prior assertion that this terminal task could resume was incorrect. Pi finished
reading all revised source/tests and directly reran the previously approved
benchmark (168 tests, 6 errors); security/provenance/state blockers remain.
Decision remains ESCALATE. See `checkpoint-final-review.md`. A supported same-task
reopening is required to use its remaining numerical slot; no task reset,
replacement source task or shared-tool modification was performed.
No real-data training,
profiling, Kaggle execution, deployment, staging, commits or pushes are authorized.

### Final checkpoint revision after operator-authorized same-task reopening

Following the explicitly approved shared-tool repair, the operator reopened the
original task without resetting its history/specification/budget. Pi submitted
consolidated final feedback and launched revision 2. It timed out after 601,064 ms,
SIGKILL/code null, without a completed report or changes to the two draft files.
The required subsystem documentation is still absent. Both revisions are now used;
the earlier one-slot/lifecycle-blocker paragraphs above are preserved history,
not the current status.

Pi read the complete actual source/tests plus relevant callers/dependencies and
ran the original approved benchmark through `delegation_verify(stage="benchmark")`:
**168 tests, 6 errors**, exit 1, unittest runtime 3.065s. All six errors are the
same unresolved selection-fixture inconsistencies. Benchmark:
`agy-evidence-402110be-9f88-4882-9c98-12a1fde2a51c`; fresh inspection:
`agy-evidence-4224f211-2783-428a-b20b-15215b7d451c` (no monitored unauthorized or
Git changes, complete protected hashes). Actual identity/provenance/state/IO
security failures remain; interrupted action-history completeness is unverified.

Final decision: **ESCALATE**, `agy-evidence-411171fa-c296-40bc-9346-667b9d7c5b97`;
acceptance_scope **task_only (not accepted)**, deployment_decision **NOT_AUTHORIZED**.
See `checkpoint-final-review.md` for each original criterion. No checkpoint
transfer to main, automatic retry/replacement/budget reset or training-loop work
on an allegedly accepted checkpoint follows. A consequential follow-up needs a
new explicit plan and approval, preserving exhausted source-task history.
