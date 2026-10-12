# Clean verification-only authorization

User approval: “oke please do so”, responding to the proposal for clean
verification-only review of preserved source with all outputs predeclared.
This permits a new scoped agy **read-only** task, not rewriting prior decisions
or implementation. Pi independently verifies/reviews and retains acceptance.
No main source integration, deployment, training/profiling, staging, commits,
pushes or rollback occurs as part of this task.

Historical recovery task `8aa6d9b3-2366-4bf5-943e-78e1d7ad60fd` remains
ESCALATE for unapproved compileall/bytecode writes. Historical integration task
`303fdf53-8c5b-4013-86be-4fec983c809a` and other prior ESCALATEs unchanged;
standalone evaluator ACCEPT retains its original narrow scope. Evidence is in
`evaluator-integration-recovery.md` and its preserved source/execution archives.

## Immutable snapshot and clean repository

Pi prepared a full local clone of existing HEAD
`ed60be08f3d1f22a9d76eecbee282415f95a6a5c` at
`.worktrees/evaluator-verification-only/`, without new commits/index updates.
The clone's original tracked checkout remains untouched, including outside-project
sentinels. Source to verify is a byte-identical snapshot inside the clone:
`Amazon4U/.verification-snapshot/` (233 files, 6,543,219 bytes).
It contains tracked project content plus the four untracked recovery source/test/doc
files, but no old caches or generated outputs. No original files were removed.

Pi configured Git's local exclude for this snapshot before baseline capture to
keep existing-HEAD Git state clean without a prohibited baseline commit. **Ignore
status does not exclude it from review monitoring**: the entire snapshot is an
explicit `protected_ignored_paths` entry and project_root for read-only delegation,
with authorized recursive raw-byte hashing (128 MiB/file, 512 MiB total,
60 seconds/file, 180 seconds/snapshot). All monitored changes block acceptance.
This is a bounded immutable-source verification layout, not an OS sandbox.

Snapshot per-file hashes:
`Amazon4U/.verification-snapshot/verification-source-manifest.json`.
Manifest SHA-256:
`65bf760f1532692295c5985badf702bc62434a9d4d5f27edb28846a90bca3916`.
Key source hashes match preserved recovery:
- core: `623c56864c621c59289f80213673d12d00c6395568f35b52a63a56cd610fcc8a`
- shared: `90553f08df868c5eb4a854dce4e01f03fd6613b6aab678a0fee8c5af9f114226`
- shared tests: `5d522eddc5c16bfd0b487a6dda84d900b5a99aec092471ed95f570d6fff837b1`
- integration tests: `354f5a1245ee5ac6159bda49d6e09d0d473dcd0262c644f120f6f77602b2b756`
- integration doc: `546ea32d9ac10d714ce4e3b4354db2ce2bce02ea44e842478ecc8304bf6d1346`

## Predeclared writes and command policy

- Worker allowed_paths: **empty**; read_only: **true**.
- Test generated_paths: **empty**. No source/tree output, result JSON, compileall,
  pycache creation, documentation edits, additional commands or Git metadata edits.
- Only test-local scratch is allowed, explicitly declared outside the pinned clone
  at `/home/sunf/FSB/Final/Amazon4U/.verification-temp/evaluator-verification-only/`.
  The fresh scratch directory was created empty by Pi before submission. Test
  fixture temporary files are ephemeral; fixtures must clean up. No existing files
  may be overwritten. This does not authorize writes elsewhere in main project.
- Durable stdout/stderr and verification results are captured by the delegation
  harness rather than written to the snapshot. Read-only Worker reports are returned
  through harness finish, not files.
- Exact single approved command uses main `.venv/bin/python -B`, sets
  PYTHONDONTWRITEBYTECODE and TMPDIR to declared scratch, discovers all `test_*.py`
  under snapshot tests, and fails on failures/errors/skips. No optimizer or real-data
  model execution in fixtures. Compileall is expressly forbidden even though the
  general AGENTS development section lists it: this narrower task command policy
  governs the current verification-only task.
- No deployment/generalization policy: deployment **NOT_AUTHORIZED**.

## Criteria and current status

Independent acceptance must establish the preserved one-engine/public API/import
identity, both default core fallback reasons and strict standalone defaults,
authoritative candidates/history/ranking/metrics, guarded KG fallback/adversarial
validation, all 113 existing regressions and exact hand-calculated oracles, and
complete immutable-snapshot/output-policy evidence. No weaker code criteria are
introduced by calling this verification-only. Historical execution acceptance is
explicitly excluded: source verification cannot erase old violations.

## Independent decision

Task `07318909-7b11-4a46-810a-c11d6c2c8b3a`, revision 0: **ESCALATE**.
Acceptance scope: **task_only**; deployment decision: **NOT_AUTHORIZED**.
Decision evidence: `agy-evidence-dd88fe97-5124-4ba3-abd1-d5f729a27133`.
No ACCEPT, main integration, deployment, training or profiling occurred.

- Criteria 1–4 verified by Pi's actual source/dependency/test review and original
  approved benchmark: **113 tests, zero failures/errors/skips**, 1.316 seconds.
  Full benchmark: `agy-evidence-45c75dd2-4e51-4220-a25b-4938d2f40750`.
- Fresh inspection `agy-evidence-830460f1-94ac-48f9-82ce-9d9373e04c9c`:
  empty diff, no monitored mutations or Git metadata changes; complete hashing of
  593 files / 99,076,450 bytes. Independent source manifest/file-set audit also
  confirms the 234-file protected snapshot is unchanged, including its manifest.
- Pi independently preserved and decoded all 26 current Worker SQLite rows
  (412,596 payload bytes): ten views, one approved command, one finish; no
  subtrajectories. No extra Worker command or compileall was observed. This is
  current invocation evidence, not a global OS activity or historical audit claim.
- **Criterion 5 remains unverified**: the initially empty, test-only external
  scratch retains 14 non-fixture runtime/log/telemetry/config files (463,797 bytes).
  These include `dak-setup.log`, telemetry locks and
  `unleash-repo-schema-v1-codeium-language-server.json`. Fixture temporary
  directories were cleaned, but the broader output-policy requirement cannot be
  established from that fact. Filenames alone do not establish who wrote the
  residual files. Pi does not attribute an unauthorized command to Worker.
- Worker claims that scratch remained clean and tests used only standard-library
  dependencies were rejected: scratch is not empty, and graph/candidate fixtures
  use DuckDB/PyArrow. Worker statements are not acceptance evidence.

Evidence under `data/reports/evaluator-verification-only-provenance/` includes
`worker-read-only.db`, `independent-action-and-snapshot-audit.json`,
`scratch-runtime-preservation.json` (per-file hashes) and
`scratch-runtime-preserved.tar.gz`. The archive was independently read back and
verified; SHA-256:
`e2cae2c25d37cb11eeb35823e40e4a9b342740de1d54a23d523219d0563ffefc`.
Original scratch files and historical artifacts remain in place, without deletion
or retroactive authorization. The unresolved output evidence does not indicate a
new scientific source defect or negate the passing benchmark.

Stopped at revision 0, not because the report-only revision limit was exhausted.
No automatic revision, replacement or output-policy expansion. Any fresh task
requires prospective authorization addressing infrastructure output isolation
separately from ephemeral fixture scratch; old decisions must remain unchanged.

### Subsequent user authorization

After discussing this noncritical blocker, the user granted standing authorization
for Pi to handle in-scope noncritical recoveries without asking each time, while
following the agreed loop. Recorded in
`delegation-and-execution.md#standing-authorization-noncritical-recovery-without-repeated-questions`
and `AGENTS.md`. This covers prospective infrastructure/test-scratch separation
and fresh verification for this issue. Pi must still predeclare its scope, outputs,
commands and limits; the historical decision above is not changed to ACCEPT.
No new recovery execution or main integration is claimed by this authorization note.
