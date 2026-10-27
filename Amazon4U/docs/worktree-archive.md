# Archived worktrees

Status: implemented 2026-10-24 on explicit user approval of Git-based archiving.
Decision: finished isolated checkouts under `Amazon4U/.worktrees/` are archived into the
main Git repository instead of being kept as live folders. Reason: 43 folders (2.9 GB)
carried stale `AGENTS.md` copies that cost tokens and contradicted the current rules,
while their content still had to stay available for audit and lookback.

`o-offline-collection-final-repair-v1` stays live: it is the source of the accepted
O-OFFLINE main integration. Every other worktree is listed below.

## Where the content is

- **Files Git can track** (tracked and untracked, non-ignored): a commit at
  `refs/archive/worktrees/<name>`. Its parent is the worktree's original `HEAD`.
  Clones also keep their own refs under `refs/archive/worktrees/<name>-refs/`.
  These refs live only in the local repository and are not pushed by `git push`,
  `--all` or `--tags`.
- **Ignored files, nested repositories and empty directories**: a tarball in
  `~/FSB-archive/amazon4u-worktrees/<name>.ignored.tar.zst`.
- **Manifest** for each worktree, with SHA-256 hashes of every tarball file:
  `~/FSB-archive/amazon4u-worktrees/<name>.manifest.json`.

Created with `scripts/archive_worktree.py`. Before an archive was accepted, every file
on disk was checked against the archive commit or the tarball manifest. Two archives,
`o-offline-round2-v1` and `i1-review`, were also fully restored to scratch and compared
byte for byte with the originals.

## Looking things up

```sh
git show refs/archive/worktrees/<name>:Amazon4U/scripts/<file>.py   # read one file
git grep '<pattern>' refs/archive/worktrees/<name>                   # search one worktree
git diff refs/archive/worktrees/<a> refs/archive/worktrees/<b>        # compare two attempts
git log refs/archive/worktrees/<name>                                # history it was based on
```

## Restoring a folder at its old path

Old references such as `.worktrees/<name>/...` work again after restoring:

```sh
cd /home/sunf/FSB/Final
git worktree add Amazon4U/.worktrees/<name> refs/archive/worktrees/<name>
tar --zstd -xf ~/FSB-archive/amazon4u-worktrees/<name>.ignored.tar.zst \
    -C Amazon4U/.worktrees/<name>    # only if the tarball exists
```

A restored folder is a linked worktree, not the original standalone clone. Its file
content is identical. Only the executable bit is kept for file modes.

## Archived worktrees

| Name | Kind | Archive commit | Archived files in Git | Files in tarball | Tarball |
|---|---|---|---:|---:|---|
| `bpr-mf-removal-review` | clone | `9d684b21a551` | 613 | 0 | none |
| `checkpoint-contracts-review` | clone | `a74bdd1a1083` | 403 | 52 | checkpoint-contracts-review.ignored.tar.zst |
| `checkpoint-identity-closure-review` | clone | `a49afd0e8e8a` | 402 | 12 | checkpoint-identity-closure-review.ignored.tar.zst |
| `checkpoint-identity-continuation-review` | clone | `aace5b0b5e07` | 402 | 7 | checkpoint-identity-continuation-review.ignored.tar.zst |
| `checkpoint-identity-correction-review` | clone | `2943cc878ec8` | 402 | 10 | checkpoint-identity-correction-review.ignored.tar.zst |
| `checkpoint-identity-finalization-review` | clone | `6c2b07cfb190` | 402 | 12 | checkpoint-identity-finalization-review.ignored.tar.zst |
| `checkpoint-identity-repair-review` | clone | `41eedd9bacfb` | 402 | 9 | checkpoint-identity-repair-review.ignored.tar.zst |
| `checkpoint-payload-a04-d01-correction` | copy | `80a8779a65e3` | 284 | 0 | checkpoint-payload-a04-d01-correction.ignored.tar.zst |
| `checkpoint-payload-authorized-correction` | clone | `f5ad0c4fb286` | 414 | 27 | checkpoint-payload-authorized-correction.ignored.tar.zst |
| `checkpoint-payload-matrix-correction` | clone | `b3999319e4fe` | 414 | 59 | checkpoint-payload-matrix-correction.ignored.tar.zst |
| `checkpoint-payload-oracle-review` | clone | `66915ee19f2b` | 413 | 19 | checkpoint-payload-oracle-review.ignored.tar.zst |
| `checkpoint-payload-recovery-review` | clone | `516b2de66934` | 414 | 34 | checkpoint-payload-recovery-review.ignored.tar.zst |
| `checkpoint-payload-review` | clone | `2e8ffae3fb26` | 414 | 8 | checkpoint-payload-review.ignored.tar.zst |
| `checkpoint-payload-text-correction` | copy | `d6e637093e80` | 284 | 0 | checkpoint-payload-text-correction.ignored.tar.zst |
| `checkpoint-seed-diagnostic-correction-review` | clone | `5a86d0b289ba` | 402 | 19 | checkpoint-seed-diagnostic-correction-review.ignored.tar.zst |
| `checkpoint-storage-bounded-correction-v1` | copy | `7c6eefb616c6` | 298 | 1 | checkpoint-storage-bounded-correction-v1.ignored.tar.zst |
| `checkpoint-storage-implementation` | copy | `fc70dd3b345a` | 298 | 1 | checkpoint-storage-implementation.ignored.tar.zst |
| `evaluator-hardening-clean` | clone | `396e47816947` | 228 | 11 | evaluator-hardening-clean.ignored.tar.zst |
| `evaluator-integration-recovery` | clone | `184edfb26f82` | 365 | 17 | evaluator-integration-recovery.ignored.tar.zst |
| `evaluator-integration-review` | clone | `7502366e8b30` | 365 | 17 | evaluator-integration-review.ignored.tar.zst |
| `evaluator-verification-only` | clone | `525f1153cc62` | 593 | 0 | none |
| `harness-evidence-review` | clone | `86f9adfd4a09` | 393 | 88 | harness-evidence-review.ignored.tar.zst |
| `harness-evidence-temp` | copy | `a925815f466e` | 25 | 0 | none |
| `harness-model-recovery` | clone | `8288d08c124f` | 394 | 0 | none |
| `i1-review` | linked worktree | `e57b0960ece4` | 358 | 3 | i1-review.ignored.tar.zst |
| `o-offline-clean-v1` | clone | `786aad65186e` | 169 | 25 | o-offline-clean-v1.ignored.tar.zst |
| `o-offline-collection-three-gaps-v1` | clone | `6a3a02bc8f05` | 169 | 45 | o-offline-collection-three-gaps-v1.ignored.tar.zst |
| `o-offline-compat-correction-v1` | clone | `9a332e3db426` | 169 | 39 | o-offline-compat-correction-v1.ignored.tar.zst |
| `o-offline-contract-repair-v1` | clone | `73b1e84d5c15` | 169 | 32 | o-offline-contract-repair-v1.ignored.tar.zst |
| `o-offline-corrective-60m-v1` | clone | `11c1d44ef0fb` | 169 | 37 | o-offline-corrective-60m-v1.ignored.tar.zst |
| `o-offline-corrective-task-v1` | clone | `9815d04c5f8e` | 169 | 61 | o-offline-corrective-task-v1.ignored.tar.zst |
| `o-offline-four-fix-v1` | clone | `a693ca49fac7` | 169 | 38 | o-offline-four-fix-v1.ignored.tar.zst |
| `o-offline-implementation-v1` | copy | `d09e277ef435` | 185 | 0 | o-offline-implementation-v1.ignored.tar.zst |
| `o-offline-native-h-metadata-v1` | clone | `bd0fcc2100b2` | 169 | 45 | o-offline-native-h-metadata-v1.ignored.tar.zst |
| `o-offline-pi-handoff-integration-v1` | clone | `0e5126a35b2e` | 169 | 30 | o-offline-pi-handoff-integration-v1.ignored.tar.zst |
| `o-offline-round2-v1` | clone | `e7df856d7486` | 169 | 32 | o-offline-round2-v1.ignored.tar.zst |
| `o-offline-round3-v1` | clone | `88dc961a55d8` | 169 | 67 | o-offline-round3-v1.ignored.tar.zst |
| `o-offline-sdk-correction-v1` | clone | `db971aaa713d` | 169 | 36 | o-offline-sdk-correction-v1.ignored.tar.zst |
| `o-offline-targeted-runtime-recovery-v1` | clone | `6873d6983caf` | 169 | 43 | o-offline-targeted-runtime-recovery-v1.ignored.tar.zst |
| `trace-audit` | linked worktree | `95b8c3cdb5c3` | 356 | 0 | trace-audit.ignored.tar.zst |
| `training-harness-implementation-v1` | copy | `88205c0fc2bb` | 321 | 1 | training-harness-implementation-v1.ignored.tar.zst |
| `training-harness-review` | clone | `b007f547b137` | 394 | 2 | training-harness-review.ignored.tar.zst |
| `training-mi` | linked worktree | `2fa6b139b254` | 567 | 96 | training-mi.ignored.tar.zst |

## Removal status

All 42 archives were verified. Deleting the live folders needs the user to run it,
because the agent's safety check blocks irreversible deletion:

```sh
cd /home/sunf/FSB/Final/Amazon4U
.venv/bin/python scripts/archive_worktree.py --remove $(ls .worktrees | grep -vx o-offline-collection-final-repair-v1)
```

The command re-archives and re-verifies each folder immediately before deleting it.
