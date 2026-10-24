# Runtime-separated evaluator verification

Status: prospectively authorized under the standing user approval in
`delegation-and-execution.md`; user subsequently said "ok, please proceed".
This is bounded, read-only verification of the preserved evaluator artifact,
not a source revision, retroactive waiver or profiling/training permission.
Historical task `07318909-7b11-4a46-810a-c11d6c2c8b3a` remains ESCALATE.

## Scope and output policy (declared before execution)

Use the existing clean isolated repository
`/home/sunf/FSB/Final/Amazon4U/.worktrees/evaluator-verification-only`, with
working_directory/project_root `Amazon4U/.verification-snapshot` inside it.
The entire snapshot is protected despite Git ignore status. Manifest SHA-256:
`65bf760f1532692295c5985badf702bc62434a9d4d5f27edb28846a90bca3916`.
No source, configuration, test, documentation or Git edits by Worker;
allowed_paths and test generated_paths are empty; read_only true.
Raw-byte recursive sentinel monitoring: 128 MiB/file, 512 MiB total,
60 seconds/file, 180 seconds/snapshot; no sibling writes.

Fresh external directories, both within the approved Amazon4U workspace:
- `/home/sunf/FSB/Final/Amazon4U/.verification-temp/evaluator-runtime-separated-v1/runtime`:
  first harness temporary directory. Coding-tool infrastructure may create/update
  its runtime logs, locks and feature configuration here. These are permitted
  current-task infrastructure outputs, not scientific artifacts. They need not be
  empty after the task. No user files exist here before baseline. Independent Pi
  inventory/preservation is capped at 8 MiB/file, 32 MiB total, 30 seconds.
- `/home/sunf/FSB/Final/Amazon4U/.verification-temp/evaluator-runtime-separated-v1/fixtures`:
  test TMPDIR only. Synthetic fixtures may create ephemeral temporary files and
  must clean them up; this directory must be empty after verification.

No other Worker output, compileall, bytecode, report files, subagents, extra shell
commands, optimizer/model training, real-data model evaluation, held-out evaluation,
Git operations or cleanup of historical files. Harness-managed invocation databases
and captured stdout/stderr are infrastructure evidence. Pi may write independent
review evidence in `data/reports/evaluator-runtime-separated-provenance/` and update
main review documents; these are not Worker writable paths. Preserve original
source, historical caches and scratch artifacts. No commits or pushes.

## Exact approved benchmark/worker command

```sh
cd /home/sunf/FSB/Final/Amazon4U/.worktrees/evaluator-verification-only/Amazon4U/.verification-snapshot && /home/sunf/FSB/Final/Amazon4U/.venv/bin/python -B -c "import os,tempfile,unittest,sys; os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.environ['TMPDIR']='/home/sunf/FSB/Final/Amazon4U/.verification-temp/evaluator-runtime-separated-v1/fixtures'; tempfile.tempdir=None; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); r=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)"
```

Worker runs this once and returns a short factual report through finish; file
views are permitted. Pi runs the same command for clean baseline and independent
benchmark. No source retuning, max_revisions 0. This does not reset an exhausted
implementation revision budget. The new policy addresses only infrastructure
outputs prospectively; historical criteria/results stay unchanged.

## Original semantic criteria retained

1. One shared engine/context/error, identical package/top-level imports in both
   orders, compatible core APIs and both default fallback reasons; standalone
   strict default, original scorer/sampler/loss math retained.
2. Authoritative candidates/private provider lists, history filtering, exact ties,
   finite/missing-score rejection, no padding, condition-specific warmth and
   registered macro/coverage metrics correct through both evaluator APIs.
3. Narrow same-context KG fallback with positive history, authoritative finite
   profile validation including unused channels, unusable positive-weight channels,
   exact popularity; spoof/corrupt/missing/wrong-context claims fail. Fixed mixed
   route/subgroup/by-reason oracles and validation-only mixture selection correct.
4. Exactly 113 retained synthetic regressions, zero failures/errors/skips under
   the approved command; no model training, optimizer updates or real-data model
   execution. Existing hand-calculated four-condition/split and three-route oracles
   independently reviewed; test fixtures depend on DuckDB/PyArrow as well as stdlib.
5. Complete current source/hash/action/benchmark evidence, no monitored mutation,
   only approved command, clean fixture scratch, infrastructure outputs confined
   to prospectively permitted runtime scratch and harness evidence. No acceptance
   of old execution, no main integration or sibling/historical changes by Worker.

Acceptance is task_only. Deployment NOT_AUTHORIZED (no deployment policy).
Passing this task verifies the immutable source artifact, not main integration,
remaining I1 acceptance, learned models or Musical_Instruments profiling execution.

## First runtime-separated result and bounded report closure

Task `179e1aaa-8a9c-483c-a537-2699edf4f208`: source/benchmark/output criteria
independently verified; 113 tests passed, no failures/errors/skips, 1.901 seconds.
Fixture scratch empty; five permitted runtime files / 463,680 bytes confined to
runtime. Independent raw action audit: eight rows, 144,398 payload bytes, one view,
one approved command and finish; no subtrajectories. Snapshot unchanged.

ACCEPT submission was rejected by the harness because Worker placed four scope,
role and dependency boundary notes in `known_limitations`. Those notes are not
current unmet implementation criteria, but Pi cannot rewrite the Worker report or
silently bypass the harness gate. Task had max_revisions 0; it remains **ESCALATE**,
not ACCEPT, decision `agy-evidence-bd721f7c-fc0a-41b0-b3e1-9e0524088c37`.
This is a noncritical report-closure problem, not a new source/output defect.

Under standing user authorization, predeclare **one** fresh report-closure
verification, no source retuning or further automatic replacement loop:
- Same protected immutable snapshot and five criteria above.
- Fresh scratch root
  `/home/sunf/FSB/Final/Amazon4U/.verification-temp/evaluator-runtime-separated-v2/`
  with `runtime` first and `fixtures` second, identical prospective output policy
  and budgets. Preserve all v1 files. Exact approved command above with only the
  test TMPDIR path changed from `evaluator-runtime-separated-v1/fixtures` to
  `evaluator-runtime-separated-v2/fixtures`.
- Pi baseline and independent benchmark use that same newly predeclared command.
- Max_revisions **1**, for report corrections only, never source edits or retuning.
  Worker must report genuine current unmet criteria in known_limitations/issues;
  authorization/deployment/dependency/role boundaries go in summary, not as bugs.
  Empty limitation/issue arrays are valid only if no real current issue was found.
- Pi review evidence allowed in
  `data/reports/evaluator-runtime-separated-v2-provenance/`; Worker file outputs
  still empty. Main integration remains gated on a recorded ACCEPT.

## Final artifact verification decision

Task `2562b8ec-cf57-4ace-b348-9fbd6c994262`, revision 0: **ACCEPT**.
Acceptance scope **task_only**; deployment **NOT_AUTHORIZED**.
Decision: `agy-evidence-f04ab70a-955f-4b89-8da0-2a484c1a4740`.
Benchmark: `agy-evidence-fa512b06-98e4-4aad-a414-a7f1c1603bcc`, all 113 tests,
zero failures/errors/skips, 1.699 seconds; full output independently read.
Fresh inspection: `agy-evidence-b78e6266-927e-4bde-a68c-e231c9c2e16e`, empty
actual diff, zero monitored/Git changes, complete 593-file / 99,076,450-byte hashing.
Pi's exact snapshot file-set/per-file hash audit also passed. All five original
semantic/current-output criteria independently verified.

Fixture scratch was empty; five permitted infrastructure files / 463,680 bytes
were confined to v2 runtime and preserved without deleting originals. Runtime
archive SHA-256:
`77d02cd0fd385739cbb0d40c0241d24eeddfa70bc2877f791960a841db64a072`.
Current raw SQLite independently decoded: eight rows / 139,819 payload bytes,
one file view, one exact published approved shell body via native command tool,
and finish; no subtrajectories. Command payload SHA-256
`8698486e8b424333ab0bb032d134300d79780d97c3175b6b8f3944467f0b5b66`
matched immutable evidence. Pi's initial overly strict argv-wrapper comparison and
read-only debug syntax error are recorded in the audit; the corrected audit retains
all original tool arguments. No extra Worker command was found.

Evidence: `data/reports/evaluator-runtime-separated-v2-provenance/`.
Historical ESCALATEs remain unchanged. This ACCEPT verifies the immutable artifact,
not broad I1, learned models, main integration, profiling execution or deployment.

## Main mechanical integration gate (prospective)

The approved implementation sequence now permits Pi to mechanically integrate the
accepted agy-implemented artifact, without new code edits or implementation retuning.
Only these five files may be copied byte-for-byte from the accepted snapshot:
`scripts/recommendation_core.py`, `scripts/recommendation_evaluation.py`,
`tests/test_recommendation_evaluation.py`, `tests/test_evaluator_integration.py`,
`docs/evaluator-integration.md`. Verify existing target core is clean against HEAD
and other targets absent (or already identical); preserve target bytes and Git status
before copying. Abort rather than overwrite unrelated changes. Accepted snapshot,
sibling project and historical worktrees remain untouched. No Git/index/commit edits.

Pi evidence outputs: `data/reports/evaluator-main-integration-v1-provenance/`.
Fresh ephemeral fixture scratch:
`/home/sunf/FSB/Final/Amazon4U/.verification-temp/evaluator-main-integration-v1/fixtures`.
Before copying, independently run the original 63-test main baseline; after copying,
run the identical command expecting 113 tests and zero failures/errors/skips:

```sh
cd /home/sunf/FSB/Final/Amazon4U && /home/sunf/FSB/Final/Amazon4U/.venv/bin/python -B -c "import os,tempfile,unittest,sys; os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.environ['TMPDIR']='/home/sunf/FSB/Final/Amazon4U/.verification-temp/evaluator-main-integration-v1/fixtures'; tempfile.tempdir=None; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); r=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)"
```

Independently compare copied hashes and verify all non-target tracked bytes,
pre-existing scratch/historical evidence and Git HEAD/index unchanged. Source
preservation and main regression results must precede an integration-complete claim.
This is mechanical integration of already independently accepted Worker code,
not a new Worker implementation or a deployment approval.

### Main integration result

Mechanical integration completed after the recorded artifact ACCEPT. Main baseline
63 tests passed (1.213 seconds); integrated main 113 tests passed (1.334 seconds),
zero failures/errors/skips. Actual complete main benchmark output read and copied
hashes independently checked. Other pre-existing tracked Amazon4U bytes and
unrelated Git status records were unchanged at transfer; HEAD/index unchanged;
fixture scratch empty. Evidence and preserved original core are in the predeclared
`data/reports/evaluator-main-integration-v1-provenance/` directory.

Main `docs/evaluator-integration.md`, `docs/implementation-roadmap.md` and `AGENTS.md`
then received independent current-status documentation; this metadata amendment
is separate from the initially byte-identical five-file transfer and is not a
source-code change. Accepted isolated snapshot and historical decisions untouched.
Remaining-I1 verification is next; no learned training/profiling/deployment run.

### User-requested telemetry Git exclusion

User requested ignoring `datacloud_telemetry`. Main `Amazon4U/.gitignore` now
excludes `datacloud_telemetry/` directories at any depth within Amazon4U. This is
version-control housekeeping only: it does not authorize undeclared runtime writes,
remove monitoring requirements, delete historical evidence or change acceptance.
Root/shared and Movie4U ignore files remain untouched. Existing telemetry files are
preserved; ignored outputs still need prospective scope/protection declarations.


