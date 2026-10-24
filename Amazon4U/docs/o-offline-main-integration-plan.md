# Proposed O-OFFLINE main integration

Status is PROPOSED. Only inspection and planning are authorized so far.
No application file has been transferred. No test has run during this planning phase.

## Goal and scope

Make the accepted offline adapter available in Amazon4U main without changing its
accepted behavior or any model, evaluation, data or training semantics. Acceptance
requires exact seven-file transfer, complete main verification and preservation.
Deployment remains NOT_AUTHORIZED. This is not a commit, merge, release or remote job.

The source is the isolated candidate under
`Amazon4U/.worktrees/o-offline-collection-final-repair-v1/Amazon4U/`.
Its current technical acceptance is recorded in
`data/reports/o-offline-authorization-final-v7/operator-adjudication-v1/review.json`
relative to that candidate. The operator adjudication does not accept the old
native15 process. Its historical ESCALATE and unverified policy reads remain unchanged.

The destination is `/home/sunf/FSB/Final/Amazon4U`.
The enclosing Git root is `/home/sunf/FSB/Final`.

## Observed main state

The planning inventory verified these facts without importing project modules.

- All 31 inherited Python files match the accepted artifact byte for byte.
- All 28 H bundle members match the accepted artifact byte for byte.
- Main has exactly the original 412 statically enumerated test IDs.
- The existing accepted main H-REF receipt records 412 passing tests and 3,564
  expanded cases. Its complete stdout matches the accepted O reference hash.
- Only the existing Kaggle runner needs replacement. Six target files are absent.
- Root Git status shows only the pre-existing untracked Movie4U framework path.
  The plan must preserve it. Movie4U is not an integration target.

These are source and historical evidence checks, not a fresh runtime baseline.
`data/reports/o-offline-main-integration-planning-v1/main-static-baseline.json`
contains the observed hashes, IDs and prior evidence bindings.

## Exact transfer

`source-manifest.json` in the planning report directory pins all seven accepted hashes.
Transfer the bytes exactly. Do not merge, refactor, repair assertions or regenerate code.

1. Archive and replace `multiautoresearch/post-training/scripts/kaggle_job.py` only
   if its current SHA-256 remains
   `6c349367c74a7a39f870431dcdfb7874602c5ad38464def265676f7b6394b2bf`.
2. Exclusively create `multiautoresearch/adapters/__init__.py`.
3. Exclusively create `multiautoresearch/adapters/amazon4u.py`.
4. Exclusively create `multiautoresearch/adapters/amazon4u_kernel.py`.
5. Exclusively create `tests/test_kaggle_orchestration.py`.
6. Exclusively create `tests/test_amazon4u_bundle.py`.
7. Exclusively create `docs/o-offline-orchestration.md`.

The new adapters directory is expected to be absent. Refuse unexpected files,
symlinks, owners, targets or main drift. Create only the required directory.
Archive the existing runner and its identity before replacement. Preserve its old
bytes in fresh integration evidence. Do not adopt or erase any occupied output.
On partial transfer failure, preserve the partial state and stop. Do not roll back
automatically or overwrite concurrent user changes.

The reviewed runner diff changes explicit profile arguments and early dispatch
before credentials, CLI or mutable state. It also permits an explicit argv list.
The remaining legacy implementation is unchanged. See `runner-transfer.diff`.

## Verification sequence

Use the existing reviewed guards, result parser, scratch monitor and owned-cache
lease. Prepare fresh, versioned main verification files under
`data/reports/o-offline-main-integration-v1/` only after execution approval.
Retarget only main paths, the approved manifest and phase receipts. This is not
a new orchestration design. Review the exact infrastructure changes before use.

1. Recheck acceptance, source hashes, main preconditions, dependency provenance,
   current Git state and preservation inputs. Stop if stale or incomplete.
2. Review the versioned main verifier and run one infrastructure-only child with
   the existing 17 mocked cache and scratch-monitor checks. No target imports,
   real process experiments or H work in these mocks.
3. Archive the existing runner, transfer exactly the seven accepted files and
   verify the resulting hashes. All inherited 31 files and H28 members must stay exact.
4. After safety review, run one complete main O-only preflight with all 103 methods.
   Pin actual imports to main rather than the isolated source. Preserve the existing
   scientific-import and native-provider guards. Require zero H imports or updates.
5. Only after that same-source O-only PASS and fresh safety receipt, run one complete
   main regression with the original unittest discovery and all 515 expected IDs.
6. Read the complete results and adjudicate the original requirements and integration
   criteria below. Record a separate main integration decision. Do not infer acceptance
   from a green child alone.

The runtime commands use `/home/sunf/FSB/Final/Amazon4U/.venv/bin/python -B` and
run from main. The future main verifier commands are proposed as follows.
They are not executable approval because those versioned files are not prepared yet.

```text
python -B data/reports/o-offline-main-integration-v1/verify_infrastructure.py
python -B data/reports/o-offline-main-integration-v1/verify_o.py o-only
python -B data/reports/o-offline-main-integration-v1/verify_full.py candidate
```

Freeze their exact final bytes, arguments, outputs and receipts before the first child.
All child scratch and evidence paths must be new and private within that integration
report directory. Completed prior-phase outputs remain protected during later stages.

## Proposed allocations

| Child | Maximum starts | Wall limit | Memory limit | Each complete stream | Fresh scratch |
|---|---:|---:|---:|---:|---:|
| Existing infrastructure mocks, 17 methods | 1 | 60 seconds | 512 MiB | 8 MiB | 16 MiB |
| Complete main O-only, 103 methods | 1 | 60 seconds | 512 MiB | 8 MiB | 16 MiB |
| Complete main regression, 515 methods | 1 | 240 seconds | 2 GiB RSS | 8 MiB | 64 MiB |

The O-only address-space ceiling also stays 512 MiB. Do not silently impose the same
address-space ceiling on the inherited full suite, whose approved limit is child RSS.
The sampled monitor, complete-output limits and final resource receipt remain required.

The full suite includes the unchanged bounded synthetic CPU model and H fixtures.
No model changes or research training are proposed. Allocate at most 36 additional
H fixture steps under the existing source-inferred accounting. The prior O-associated
inferred total is 288. The proposed exception ceiling becomes 324. This is a new,
explicit allocation request, not inherited permission or a measured global counter.
The five consumed O-associated full starts stay consumed. This plan adds at most one
main full start, yielding six in the combined O-associated history.

No fresh 412-test baseline child is recommended. Reuse the accepted main 412-test
receipt only while all 31 Python hashes, H28 members, baseline IDs, dependencies and
receipt hashes remain valid. This avoids another full run and another H fixture
allocation. The tradeoff is that no fresh pre-transfer runtime result is claimed.
If any provenance check fails, stop and reassess the baseline rather than automatically
running an extra suite or changing the source.

## Preservation and safety

Retain all accepted and failed isolated artifacts, native outcomes, operator decisions,
old scratch residuals, source history, original criteria and consumed budgets.
Keep the isolated candidate read-only for this integration.

Preserve main outside the seven target paths. Monitor root and nested repository
sentinels, including the accepted candidate Git boundary. Use read-only Git metadata
commands with optional locks disabled. No staging or Git metadata writes.

Keep the approved raw hashing bounds of 1 GiB per snapshot, 128 MiB per file,
60 seconds per file and 300 seconds per snapshot. Include declared ignored evidence,
implementation paths and test-output paths. A stat-only planning estimate combines
784,832,649 bytes in the accepted snapshot with 39,430,974 bytes of root Git metadata,
for 824,263,623 bytes before overlap removal and later evidence growth. This estimate
is not a completed preservation snapshot. Recheck capacity before execution and stop
on any incomplete or unstable capture. Do not drop protected roots to force a pass.

Only the exact parent-created, identity-checked empty cache may be removed after its
child settles. Retain empty scratch. Unknown or pre-existing entries are never deleted.
No cleanup of old reports, worktrees, caches or user files is authorized.

These are reviewed cooperative controls, not an OS sandbox or hostile-sameUID guarantee.
No network, credentials, Kaggle CLI, pip, GPU, real-data work, native capability/race
experiments or real legacy training is proposed. The inherited full fixture behavior
remains bounded and separately reviewed.

## Main integration acceptance criteria

- MI1. All seven main target hashes equal the accepted manifest. The original runner
  is archived. No source, test assertion, dependency or scientific semantics change.
- MI2. All 31 inherited Python hashes and H28 members remain exact. All 412 baseline
  and prior 501 IDs remain among the 515 main methods. All 35 ledgers, 3,564 expanded
  cases and 34 D3 methods pass. Inherited stdout remains byte-identical.
- MI3. Main O-only passes all 103 methods before full release. Actual target imports
  resolve to main. No H import or update occurs in that preflight.
- MI4. Main full regression passes all 515 methods with zero failures, errors or skips.
  Complete results, resource limits, empty scratch, cache ownership and cleanup pass.
- MI5. Scope, protected artifacts, Git and sibling sentinels remain preserved.
  All unexpected drift, incomplete evidence or unsafe evaluation blocks acceptance.
- MI6. Reconcile original O criteria, OP rows, ordered limitations, operator amendment
  and actual main evidence. Preserve historical native15 ESCALATE and unverified reads.
  Do not revive the retired coordination hold as a current technical blocker.
- MI7. Record main integration acceptance separately from isolated acceptance.
  Deployment remains NOT_AUTHORIZED. Research execution and production readiness
  remain separate. No automatic commits, pushes, releases or remote work.

Any failed or interrupted child consumes its allocation. No automatic retry, source
correction or rollback. Preserve failure evidence and report the actual remaining gate.

## Approval needed

Approval must cover the seven-file main transfer, versioned verifier preparation,
one 17-method infrastructure check, one 103-method O-only preflight, one conditional
515-method main regression, their limits and the proposed 36-step H fixture allocation.
It does not include another baseline, Worker/SDK request, model change or deployment.

## Execution approval

The user approved this bounded plan with "lgtm, continue with it."
Execution evidence is recorded in `data/reports/o-offline-main-integration-v1/`.
This section supersedes the earlier planning-only authorization status. All file,
child, fixture, preservation and stop limits above remain unchanged. Integration
is not accepted until the separately recorded main result passes every gate.
