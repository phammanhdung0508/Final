# Checkpoint recovery — approved three-deliverable boundary

Status: **user-approved decomposition, implementation/review pending**.
Acceptance remains task-only; deployment NOT_AUTHORIZED.

The user approved the consequential recovery recommendation after final checkpoint
ESCALATE and explicitly required that the following three deliverables are already
small enough: **do not subdivide them into micro-tasks**. Small fixes, additional
fixtures, documentation and review findings stay within the owning deliverable's
bounded revision cycle. No additional independent source tasks are created for
helper functions, fixture repair or individual security findings.

## Preserved decisions

- Step 3c differentiable models/graphs frozen verification is ACCEPT; main model
  integration was separately regression-verified with 204 tests. Do not rewrite it.
- Original checkpoint task `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a` remains final
  ESCALATE, with both revisions exhausted. Final decision:
  `agy-evidence-411171fa-c296-40bc-9346-667b9d7c5b97`.
- Its two drafts, missing documentation, six fixture errors and security/provenance
  findings are preserved. New work is an explicitly approved architectural
  decomposition, not another retry or retroactive acceptance of that task.
- No training-loop/continuation, optimizer updates, real-data access, CUDA execution,
  profiling, Kaggle, held-out experiments, deployment, staging, commits or pushes.

## Exactly three substantive deliverables

1. **Identity/provenance contracts**: one pure CPU/standard-library contract module,
   its adversarial tests and documentation. Canonicalize actual complete training
   records/histories/warm pool, typed graph, permitted metadata, validation selection
   inputs and resolved settings. Validate mandatory caller-supplied actual source
   bytes and explicit Git/environment evidence against mandatory trusted expectations.
   No checkpoint/tensor serialization, RNG/optimizer restoration or filesystem
   capture/publishing in this deliverable. Source/file acquisition belongs to the
   secure IO integration, not a fourth task. Pure contracts must not fabricate
   provenance or imply that caller-supplied evidence alone proves its origin.
2. **Payload validation**: one module/tests/documentation, depending on independently
   accepted identity/provenance contracts. Complete model/Adam/RNG/sampler/counter/
   selection/schema validation before eligible restore data is returned. No live
   restoration, optimizer updates or checkpoint filesystem implementation. Repair
   inconsistent selection fixtures within this boundary; never weaken the guard.
3. **Secure checkpoint IO and integration verification**: one module/tests/doc,
   depending on the first two accepted contracts. Required narrow caller-approved
   roots, descriptor-anchored traversal, exclusive atomic publication, safe bounded
   decoding including reconstructed storage, corruption/symlink/parent-race refusal,
   owned temporary cleanup and allowlisted actual source acquisition. Verify the
   complete serialization/validation pipeline together within this deliverable.
   Main transfer/regression is its integration gate, not another Worker source task.

Each deliverable is independently reviewed as a whole by Pi, using complete actual
functions and callers/dependencies/tests, original approved benchmark commands and
criterion-by-criterion decisions. Up to two consolidated corrective revisions per
new deliverable; these never reset the exhausted original task. A timeout or unmet
mandatory gate means ESCALATE, not acceptance or automatic further decomposition.
Later deliverables do not start against an unaccepted dependency.

## First-deliverable execution boundary

Files: `scripts/checkpoint_identity.py`, `tests/test_checkpoint_identity.py`,
`docs/checkpoint-identity.md` only. Use an independently prepared clean full clone
under `Amazon4U/.worktrees/checkpoint-contracts-review/`, with its canonical project
subdirectory pinned as working_directory/project_root. Git initialization by Pi's
clone setup is not permission for Worker Git metadata mutation. No commits.

Preserve a byte-identical read-only reference of the rejected two-file draft at
`data/reports/rejected-checkpoint-v2/` in this clone; explicitly protect it when
submitting. This is reference evidence, not accepted dependency/source.

Fresh synthetic scratch only: `Amazon4U/.test-output/checkpoint-identity-v1/` relative
to the clone Git root. No external temp directory or source/bytecode cache writes.
training_allowed=false. Small synthetic fixtures: at most 8 users, 12 products,
8 metadata nodes/type; no tensors/optimizer updates in the new tests. Original
baseline tests are preserved. Suite ceiling 120 seconds. Source byte fixtures are
bounded; future checkpoint payload/file limits remain 2MiB.

Pi verifies the clean clone baseline before submission and pins that exact Python
unittest discovery command, executable, absolute scratch directory, TMPDIR,
tempfile.tempdir and PYTHONDONTWRITEBYTECODE in the task specification. Benchmark
and revision runs use that command unchanged. No deployment policy/held-out commands.

## Progress

Approval and boundaries recorded. Pi prepared the clean clone at
`/home/sunf/FSB/Final/Amazon4U/.worktrees/checkpoint-contracts-review/` from existing
HEAD `fd15dc3b5fe665f45d8787ba779b9e9541690bde` without committing/staging. The two
protected rejected-draft copies match their original SHA-256s. The exact planned
benchmark preserved **204 baseline tests**, zero failures/errors/skips, 12.718s;
log: `data/reports/checkpoint-identity-setup/baseline.log` in main Amazon4U.
This is setup verification only, not first-deliverable acceptance.

First submission `5e641e60-0945-4e3f-a0ed-11593b3927ab` failed metadata preflight
before baseline capture/Worker launch: the 2MiB/file, 8MiB/aggregate monitoring
budget was too small for all monitored pre-existing source/sentinel content.
Inspection reported missing baseline and benchmark verification refused because
it was already ESCALATE. No source implementation or corrective revision occurred.
Preserve this setup failure; it does not create a fourth deliverable or exhaust a
source revision cycle. Under standing noncritical setup recovery authorization,
resubmit the same full first-deliverable specification/paths/commands/criteria,
using the previously authorized checkpoint monitoring budgets (128MiB/file,
512MiB/aggregate; 60s/file, 300s/snapshot). These are read-only evidence budgets,
NOT larger synthetic source/checkpoint fixture limits (still 2MiB).
No acceptance or deployment approval is implied by the correction.

### First deliverable — final independent outcome

Corrected whole-deliverable task: `2b5b2471-9f30-4e24-aaa6-b7653d359449`.
Final decision: **ESCALATE**, `agy-evidence-5a5f3a35-ea45-4f1f-8eff-dc6565a54cd0`.
Acceptance_scope: **task_only (not accepted)**; deployment_decision:
**NOT_AUTHORIZED**. Both corrective revisions are used. Initial invocation and
both revisions timed out at approximately 600 seconds without the required
completed Worker report. No further automatic retry, task replacement, revision
reset or micro-task decomposition is authorized.

Pi independently reviewed complete actual final source (1,975 lines), tests
(939 lines) and documentation (212 lines, unchanged since full revision-1 read),
plus relevant accepted typed-category/relation definitions and protocol contracts.
The textual diff is empty because these are untracked drafts; actual files were
read. Final SHA-256s:

- Source: `9a5a3cde799b4727e9028f7cc0b762e705ef3424e8be5014651df19e390d52e5`.
- Tests: `69e3b20ad7b6bffc558a91cabbbae17cf973497569df20ce089d8911c13832a0`.
- Doc: `5bf55245afc8d8eee56a8cd053099370b7262ddd03d2ae2d1f558d4b0714d09c`.

Original independent benchmark results:

| Stage | Result | Evidence |
|---|---|---|
| Initial implementation | 232 tests, 1 ordering error, 6.229s | `agy-evidence-07b89842-934d-4faf-85c2-4db1475b3ca9` |
| Revision 1 | 232 tests, 10 stale-fixture errors, 8.410s | `agy-evidence-5c9914ee-c0b2-44a4-9761-d763618155a4` |
| Final revision 2 | 232 tests, 1 mutation-fixture error, 8.424s | `agy-evidence-fa8d1001-edae-4bfd-a6a0-986457b598fd` |

All 204 existing baseline tests pass. The remaining new-test error changes a
validation target from p3 to p2 while leaving its record at p3; the agreement guard
correctly rejects this inconsistent fixture. Do not weaken the guard. Pi read all
final raw test results and the traceback. Verbatim paged logs are preserved in
`data/reports/checkpoint-identity-setup/` in main Amazon4U. Final Worker execution:
`agy-evidence-1bf0e04b-bf69-4e67-ab5c-770f7c88a48b`.

Final fresh inspection: `agy-evidence-8d90a7fc-a830-4789-86e2-ef9ec72890cc`:
exactly the three declared files, no monitored unauthorized or Git changes, complete
protected/sentinel hashing (132 files / 86,251,786 bytes; 130 outside sentinels).
Read-only process checks found no matching surviving unittest/Worker. These are
bounded monitoring observations, not a complete interrupted-action audit or OS sandbox.

| Original criterion | Pi outcome |
|---|---|
| Scope / zero prohibited actions | Unverified overall: monitored scope clean and pure new code/tests, interrupted historical actions incomplete |
| Complete canonical identity | Failed: improved content/ordering/type checks, but exact settings schema cannot record batch size, fanouts, initialization, full Adam settings or validation frequency; adversarial coverage incomplete |
| Bounded reconstructable source/environment provenance | Verified for pure caller-supplied evidence: strict schemas, mandatory expectations/bytes, paths, sizes/hashes/Git/environment and two-pass bounds; not proof of physical origin/acquisition |
| Pure isolated APIs / import and trust boundary | Verified for implemented APIs and snapshot revalidation; no capture/restore/pickle/optimizer APIs |
| Passing meaningful benchmark / accurate doc/completed report | Failed: one error, incomplete requested adversarial coverage, no completed report; doc falsely calls result complete/verified and describes unapproved replacement/live restoration intent |

Only **deliverable 1** was attempted; the other **two** remain not started because
the prerequisite is unaccepted. No identity source/tests/doc transferred into main.
Step 3c's prior accepted model/graph integration is untouched. The original
checkpoint task and setup-only failure are preserved, not retroactively accepted.

The three natural boundaries remain the approved plan. If further recovery is
considered, first diagnose why Worker turns repeatedly fail to finish/report; do
not respond with smaller source micro-tasks or silently increase runtime/revisions.
Shared-runtime changes or consequential source continuation need their appropriate
explicit approval. This note does not grant either.
Original final review: [checkpoint-final-review.md](checkpoint-final-review.md).
