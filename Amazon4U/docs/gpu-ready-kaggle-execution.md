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

Status: **implemented and tested locally against a fake CLI (2026-10-24). No Kaggle call
has been made.** Runs 1 to 3 each need the user's per-run approval.

Placement (user, 2026-10-24, option A):

- `multiautoresearch/adapters/amazon4u_kaggle.py`: the Kaggle CLI transport.
- `multiautoresearch/adapters/amazon4u_gpu.py`: bundle, approval, launch, collection and
  record.
- The accepted O-OFFLINE adapter is unchanged. The new modules reuse only the
  device-independent parts listed in the audit below, and no refusal is bypassed.
- The legacy unprofiled paths in `kaggle_job.py` (L1 to L9 defects) are not called.

### Transport (`KaggleCliTransport`)

Each call is one subprocess with an argv list, no shell and a timeout. Output that does
not parse is an error, never a retry. The CLI reads its own credentials.

| Call | Kaggle CLI 2.2.5 | Rule |
|---|---|---|
| `require_cli_version` | `kaggle --version` | Must print exactly `Kaggle CLI 2.2.5` |
| `push` | `kernels push -p <bundle> -t <seconds> --accelerator NvidiaTeslaT4` | Exactly one `Kernel version <n> successfully pushed.` line. A `Kernel push error` or invalid-source line is a failed receipt. Anything else, or a timeout, is `submission_unknown` |
| `status` | `kernels status <owner/slug>/<version>` | Exactly one `<owner/slug>/<version> has status "KernelWorkerStatus.<NAME>"` line. Only QUEUED, RUNNING, COMPLETE, ERROR and CANCEL_ACKNOWLEDGED map to O-OFFLINE tokens. CANCEL_REQUESTED, NEW_SCRIPT and any other text are errors |
| `download_outputs` | `kernels output <owner/slug>/<version> -p <new private dir> -q` | One download per kernel, into a new directory created with mode 700 |
| `open_log_reader` | Reads `<slug>.log` from that download | No symlinks, regular file only, at most 4 MiB |

**Kernel versions.** CLI 2.2.4 ignored a `/<version>` suffix in `kernels status` and
`kernels output` and reported the latest session. The user upgraded to 2.2.5 on
2026-10-24, which sends the version to the API. The transport now pins 2.2.5 and always
queries `<owner>/<slug>/<version>`, using the version its own push acknowledged. It answers
only for kernels it pushed itself. Each run still pushes to a new slug,
`amazon4u-gpu-ready-run-<n>-<first 8 hex of the bundle digest>`, so the receipt must
acknowledge version 1.

The receipt's `accelerator` is the one requested, because the CLI does not echo it. GR05
evidence comes from the fingerprint observed inside the kernel.

### Bundle

`amazon4u_gpu bundle --run <n> --out <new dir>` writes three files:

- `kernel.py`: every `scripts/*.py` and `tests/*.py` file plus the run plan, each embedded
  gzip and base64 with its SHA-256. Kaggle uploads only this one script.
  - The kernel checks every hash and writes the sources outside `/kaggle/working`.
  - It runs `scripts.gpu_ready_parity` on `cuda:0` with `CUDA_VISIBLE_DEVICES=0` and
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  - The CPU oracle runs in the same session, inside that runner.
  - It exits 0 whenever a result was written, so a failed parity check is still
    collected as evidence.
- `kernel-metadata.json`: private, GPU on, internet off, no attached datasets,
  competitions, kernels or models.
- `bundle.json`: run number and purpose, kernel reference, accelerator, timeout (the
  plan's 30 minutes), budget limits, tolerances, the expected run binding, the kernel and
  metadata hashes, every embedded file's hash and size, and the Git commit with the
  dirty paths (G5).

The expected run binding is computed with the runner's own `bundle_digest` and
`plan_digest` over the embedded files. A local CPU execution of the generated kernel
reported the same binding, in 17 seconds. The script is about 500 KB. The user reports
Kaggle limits of 1 MB for a `.py` script and 20 MB for a notebook; run 1 confirms the
script is accepted.

### Approval, launch and collection

`amazon4u_gpu launch --bundle <dir> --approval <file> --kaggle <cli path>` makes these
checks, in order:

1. **Approval.** The bundle files must match `bundle.json`, and the metadata must be
   private, GPU-enabled and offline. The approval file (`amazon4u-gpu-ready-run-approval-v1`)
   must name this run, this kernel reference and this manifest's SHA-256, be `approved_by:
   user` and quote the user's statement. Otherwise nothing is called. The `bundle` command
   prints the template.
2. **Reservation.** The CLI version is checked. `run-<n>.reserved.json` is then created
   exclusively, so a run number is launched at most once. There are no retries.
3. **Push.** The push is checked with O-OFFLINE's `validate_submission_receipt`, which
   requires version 1. `submission_unknown` is recorded and never re-pushed.
4. **Status.** O-OFFLINE's `poll_kernel_status` waits for the result. It polls every 30
   seconds, with a deadline of the run timeout plus 15 minutes for the queue.
5. **Outputs and log.** On `complete` or `error`, the outputs are downloaded, and the log
   is read with O-OFFLINE's `consume_kernel_logs` (redacted, at most 1 MiB). An `error`
   status is recorded and not collected.
6. **Collection.** The download may hold only `result.json` and the kernel log, as
   regular files within the plan's 64 MiB. The result must:
   - pass `validate_result`;
   - carry the bundle's run binding;
   - show `cuda:0` with exactly one GPU used;
   - carry the fixed determinism record;
   - carry the plan's tolerances.

   The raw log is replaced by the redacted one.
7. **Record.** `record.json` (`amazon4u-O-record-cuda-v1`) is written. It holds the
   receipt, the terminal status, the hashes, the environment and the verdict. Acceptance
   is `requires_user_review`, deployment is `NOT_AUTHORIZED`, and the record lists the
   GPU-READY gates. The directory then moves to `data/reports/gpu-ready-v1/run-<n>/` with
   O-OFFLINE's `atomic_move_no_replace`.

A failure at any step writes `run-<n>.failure.json` once and keeps the download for
review.

Tests: `tests/test_amazon4u_gpu.py` (18) runs against a fake `kaggle` executable that
prints the 2.2.5 lines and records every call. Mutation check: disabling any one of
nine refusals makes a test fail. The nine are the approval hash, the run binding, the
output file set, determinism, the reservation, the single push version, exact status
tokens, the log symlink refusal and the receipt check.

**O-OFFLINE redaction fix (user-approved, 2026-10-24).** In `consume_kernel_logs`, the
two redaction replacements were written `r"\\1..."`. They emitted a literal `\1` instead
of the matched key name or `Bearer ` prefix. The secret was always removed. Only those
two replacement strings changed in `multiautoresearch/adapters/amazon4u.py`. No recorded
hash pins that file. `tests/test_amazon4u_gpu.py` asserts `api token=[REDACTED]` and
`Bearer [REDACTED]`. It failed on the old code and passes on the fix.

### Reuse audit of the O-OFFLINE adapter (2026-10-24)

Read line by line in `multiautoresearch/adapters/amazon4u.py` before any GR08 code.

Device-independent, reusable unchanged:

- `validate_submission_receipt`: owner, slug, exact version, version reference, timeout,
  accelerator and plan digest. The accelerator is a parameter, so the GPU path passes
  its own (`NvidiaTeslaT4`).
- `poll_kernel_status`: exact status tokens, version and reference checks, bounded polls
  and a deadline. The defaults (5 polls, 300 seconds) are fake-transport budgets, so the
  GPU path passes its own bounds within the 30-minute run limit.
- `consume_kernel_logs` with `BoundedLogReader`: byte bounds, deadline, version check and
  credential redaction.
- `validate_raw_relative_path`, `_rename_no_replace_linux` and `encode_bytes`.

Not device-independent, so not reusable:

- `collect_outputs`. Before it reads any output, it requires the canonical H-REF spec,
  the H-REF authorization and the H run envelope. It then calls
  `validate_collected_artifacts`, which requires the seven H-REF artifact roles and an
  `amazon4u-H-result-v1` result, the authorization scope
  `tiny_synthetic_cpu_backward_and_private_restore` and `device_capability` device `cpu`
  with accelerator `none`. A cuda-v1 result cannot pass without forged H-REF records.
- `render_plan`, `build_amazon4u_kernel_script` and `write_bundle`. They require an
  `ApprovedOfflineContext`, embed the H spec and authorization, hydrate the 28 H-REF
  paths and fix the accelerator to `none`.
- `dispatch_amazon4u`. It holds the OP18 live refusal and pushes with accelerator `none`.

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

1. G4: agreed, and the quota was confirmed (user, 2026-10-24).
2. Real-transport placement and fake-CLI tests: agreed as option A, and the two
   replacement designs were approved (user, 2026-10-24).
3. G5 binding list as implemented in `bundle.json` and the run binding: approved
   (user, 2026-10-24).
4. Run 1: approved (user, 2026-10-24). Runs 2 and 3 are decided after run 1.
