# Kaggle execution and provenance (phase 1, deliverable 4: G4, G5, GR08)

Status: **G4 AGREED (user, 2026-10-24):** "you can check access using kaggle-cli and
create kernel-metadata.json to start training if need." Findings so far:

- Kaggle CLI 2.2.4 is installed.
- `~/.kaggle/kaggle.json` exists with mode 600. Its content was not read.
- The agent's read-only access check was refused by the local permission classifier.
  The user then ran `kaggle kernels list --mine --page-size 3` on 2026-10-24.
  - It listed three existing kernels under the account `dungsunf`, so access is
    **verified**.
  - The CLI warned that 2.2.5 is available. Upgrading is not needed for this work and
    was not done.
- Weekly GPU quota (user-reported, 2026-10-24): the full 30 hours are available.

`kernel-metadata.json` is generated with each run bundle once the GPU-profile code exists.
It is not written ahead of that code. The external `/home/sunf/my-project/kaggle-cli/` stays a
read-only reference.

## G4: remote execution

What the user would approve:

- Use of the user's Kaggle account for GPU-READY runs within the phase-2 budget.
- Credential handling: the Kaggle CLI reads `~/.kaggle/kaggle.json` or
  `KAGGLE_USERNAME`/`KAGGLE_KEY` itself. The agent never prints, copies, logs or stores
  the key in reports, bundles or Git. Reports record only the username and kernel slug.
- Kernels are private. Datasets are not uploaded; GPU-READY fixtures are generated inside
  the bundle from owned synthetic code.
- A quota check before the first run: the user confirms remaining weekly GPU hours,
  because `compute-plan.json` records `quota_verified: false`.

## GR08: real transport

O-OFFLINE calls a transport through five methods: `push`, `status`, `logs`,
`open_log_reader` and `list_output_page` (`FakeKaggleTransport`,
`multiautoresearch/adapters/amazon4u.py` line 1926). A real transport implements the same
methods with the Kaggle CLI:

| Method | Kaggle CLI | Behavior kept from O-OFFLINE |
|---|---|---|
| `push` | `kaggle kernels push -p <bundle>` | Exact acknowledged version required; ambiguous output becomes `submission_unknown`, never success |
| `status` | `kaggle kernels status <user/slug>` | Exact status tokens, no substring match, bounded polling with a deadline |
| `logs`, `open_log_reader` | Log file from `kaggle kernels output` | Size-bounded reads |
| `list_output_page` | `kaggle kernels output <user/slug> -p <fresh dir>` | Fresh directory per attempt, no overwrite, artifact completeness checks |

Placement: a new module `multiautoresearch/adapters/amazon4u_kaggle.py`, used only with an
approved G4 context. The OP18 refusal stays as the default. The legacy unprofiled paths in
`kaggle_job.py` (with the L1 to L9 defects) are not called. Every subprocess has a
timeout; failure to parse output is an error, not a retry.

Tests for the real transport run against a fake CLI executable on `PATH`, so they need no
network or credentials.

## G5: run provenance

Each real run binds:

- source: the complete bundle manifest with SHA-256 for every file, the Git commit and a
  clean or exactly recorded dirty state;
- data: fixture generator version and seed (GPU-READY), or input data hashes (profiling);
- configuration: resolved spec, authorization and plan hashes;
- environment: Python, NumPy and PyTorch versions, CUDA and driver versions, GPU model and
  count as observed inside the kernel;
- checkpoints: payload profile ID and hashes.

**Gap found:** `requirements.txt` pins only `duckdb` and `pyarrow`. PyTorch and NumPy are
installed in the local virtual environment but are not declared. The Kaggle image supplies
its own versions, so a GPU-READY run must record the observed versions, and the CPU oracle
must run **inside the same Kaggle session** rather than being compared with this machine.
Declaring the project's own versions in `requirements.txt` is proposed as a separate
minor fix.

## Decisions needed

1. Approve G4 as described, and confirm the remaining Kaggle GPU quota.
2. Approve the real-transport placement and the fake-CLI test approach.
3. Approve the G5 binding list.
