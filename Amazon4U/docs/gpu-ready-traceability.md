# GPU-READY traceability: run 1 against GR01 to GR10

Status: **accepted by the user (2026-10-25): "LGTM!! Accept GR01 - 10."** The accepted
caveats are the GR01 refusal after launch and the GR10 output-size check after download.

**Update after the user's review of run 1 (2026-10-24).** The user asked for three
changes:

- a fresh-process restart test for GR06;
- controlled GR04 negative tests that run in separate processes;
- O-OFFLINE hash pins.

The user left the GR10 ledger to the agent, and the agent added it. Building the GR06
test found that fresh-process resume was impossible. See the GR06 finding in
[`gpu-ready-design.md`](gpu-ready-design.md#resume-scope). Run 2 provides the GPU
evidence. The rows below show the evidence from run 1. Run 2 updates GR04, GR06 and GR10
in the section after the table. Under
the agreed review mode, the user reviews this evidence with the agent and no other model
is invoked. The agent's own assessment below is not acceptance.

Requirements are from [`gpu-ready-preparation.md`](gpu-ready-preparation.md). Run 1
evidence is in `data/reports/gpu-ready-v1/run-1/` (`result.json`, `record.json`,
`kernel-log.txt`). The bundle is `data/reports/gpu-ready-v1/run-1-bundle/`.

Run 1 facts:

- Kernel `dungsunf/amazon4u-gpu-ready-run-1-c9c8a316`. The receipt acknowledged
  version 1, and the terminal status was `complete`.
- Result `parity_complete`, `all_checks_passed: true`.
- Environment observed in the kernel: one Tesla T4 (7.5) visible and used, driver
  580.178.04, CUDA 12.8, cuDNN 91900, PyTorch 2.11.0+cu128, Python 3.13.15, NumPy 2.1.3.
- Run binding: bundle `c9c8a316…`, plan `7fe67feb…`. It equals `bundle.json`.
- The runner finished at about 25 s of the kernel session. Launch to collection took
  about 2 minutes.

Status key:

- **Evidence complete**: the named evidence exists and the agent found no gap. It still
  needs the user's review.
- **Partial**: some of the named evidence is missing or weaker than stated.
- **Missing**: no evidence yet.

| ID | Requirement | Evidence required | Evidence available | Status |
|---|---|---|---|---|
| GR01 | Explicit device seam, selected by profile; unsupported requests fail before launch | Unit tests with fakes on CPU; CPU path byte-identical | `require_device` accepts only `cpu`/`none` and `cuda:0`/`gpu`, with no fallback (`tests/test_gpu_ready_parity.py`). The H28 files are identical to HEAD. Run 1 ran on `cuda:0`. The design deviation (a separate trainer instead of `CUDABackend` in `Harness`) is recorded in `gpu-ready-design.md` | Evidence complete. Caveat: the bundle can only request `cuda:0`/`gpu`. If Kaggle gave no GPU, the refusal would happen inside the kernel, after launch |
| GR02 | Device-aware tensor, index and buffer construction for both models; no scattered `.cpu()` | Source review; CPU suite unchanged | EX01 places parameters and edge indices once at construction (`vectorized_models.py`). `vector_trainer.py` moves state only at the checkpoint boundary. Run 1 confirmed the parameters stayed on the device after restore. The CPU suite passes (593 tests) | Evidence complete. Needs the user's source review |
| GR03 | CPU/GPU parity on tiny fixtures: scores, loss, gradients and Adam updates, both models | Kaggle run report with recorded tolerances | 12 of 12 cases passed (2 models × 3 seeds × 2 conditions), oracle in the same session. Largest differences: scores 1.8e-7, loss 1.2e-7, gradients 1.7e-8, Adam 2.0e-5 (limit 1e-4), top-10 identical. The excluded product bias had a reference gradient of at most 5.6e-9 | Evidence complete |
| GR04 | Deterministic algorithms on; unsupported CUDA ops fail loudly | Run log showing settings and any refusal | `result.json` records exactly the fixed policy (`warn_only: false`, CUBLAS `:4096:8`, TF32 off), and collection checks it. No refusal occurred | Partial. The settings are proven. That an unsupported op raises was not exercised on the GPU; it rests on PyTorch behavior with `warn_only=False` |
| GR05 | Recorded environment: GPU, driver, CUDA, libraries and the allocated accelerator | Collected run record; a metadata label is not evidence | A fingerprint observed inside the kernel, including `nvidia-smi` driver output, is in `result.json` and `record.json` | Evidence complete |
| GR06 | GPU checkpoint profile with CUDA RNG/runtime state; D3 bounds kept; GPU restart versus uninterrupted run in the same Kaggle environment | CPU schema tests; GPU restart comparison | `tests/test_checkpoint_cuda.py`. In run 1, both models matched bit for bit after a cuda-v1 publish, retrieve and restore at epoch 2 of 4, with real D3 IO on Kaggle | **Partial. The restart is in the same process**, not a fresh process. See finding 1 |
| GR07 | Versioned GPU result and record schemas with their own gate list; v1 and `READINESS_GATES` untouched | Collection tests for both versions; v1 fixtures unchanged | `amazon4u-H-result-cuda-v1` and `amazon4u-O-record-cuda-v1`. The record carries `GPU_READINESS_GATES`. v1 results are refused (`tests/test_amazon4u_gpu.py`, `tests/test_gpu_ready_parity.py`). The O-OFFLINE suites pass | Evidence complete |
| GR08 | Real Kaggle transport: exact receipt, bounded polling, bounded logs and collection, no overwrite | Fake-transport tests pass; one live receipt reviewed | 18 fake-CLI tests plus the O-OFFLINE fake-transport suites pass. The live receipt is in `record.json`: `…/v1`, `NvidiaTeslaT4`, timeout 1800 | Evidence complete. The receipt needs the user's review |
| GR09 | Packaging, imports, native IO and collection work in the real environment | Phase-2 run report | The embedded sources hydrated and their hashes were checked. Imports worked on Kaggle's PyTorch 2.11. D3 `O_TMPFILE` and `linkat` IO worked (restart passed). Collection and the no-replace move worked. This document is the run report | Evidence complete |
| GR10 | Budgets enforced: run count, wall time, GPU-hours, output size; no automatic retry | Limits recorded before launch; receipts within them | Run count: a run number is reserved once and must be at most 3. Wall time: `-t 1800` and a runner timeout of 1680 s. Output: at most 64 MiB, checked. No retry (tested). `bundle.json` held the limits before launch. Run 1 used about 30 s of GPU time | Partial. See finding 2 |
| Gate | GPU-READY acceptance by independent review of the GR01 to GR10 evidence | User review | This document, runs 1 and 2 | **Accepted by the user, 2026-10-25** |

## Run 2 (2026-10-24)

Kernel `dungsunf/amazon4u-gpu-ready-run-2-485e06d6`, version 1, terminal status
`complete`. The environment is identical to run 1. The result is `parity_failed`,
because one probe failed.

| ID | Run 2 evidence | Status |
|---|---|---|
| GR03 | 12 of 12 parity cases passed again (largest Adam difference 1.98e-5) | Evidence complete |
| GR04 | Four of five probes passed: `MaxUnpool1d`, CUDA `histc`, CUDA adaptive-avg-pool backward and CUDA `put_` with accumulate. Each raised PyTorch's "does not have a deterministic implementation" error with the policy on, and its control succeeded. **The cuBLAS probe failed.** With the policy on and `CUBLAS_WORKSPACE_CONFIG` unset, a 16×16 CUDA matmul did not raise | **Complete (the user decided on 2026-10-25).** The cuBLAS probe tested a check PyTorch removed before 2.10; see below. The `CUBLAS_WORKSPACE_CONFIG` requirement is enforced by our own refusal |
| GR06 | Fresh-process restart passed for both models: three distinct processes sharing only the checkpoint bytes, logs, state and validation identical, parameters on `cuda:0`. Same-process restart passed again | Evidence complete |
| GR10 | The ledger booked 1800 s for each run, 3600 s of 5400 s in total. Collection still worked on a failed verdict | Evidence complete. The remaining caveat is that the output size is checked after download |

**The cuBLAS probe.** Its expectation came from the PyTorch documentation (CUDA 10.2 and
later need `CUBLAS_WORKSPACE_CONFIG` for deterministic cuBLAS). It is not changed after the
result.

**Cause (established 2026-10-25, without a GPU run).** PyTorch removed the check.
[pytorch/pytorch#161749](https://github.com/pytorch/pytorch/pull/161749) ("remove
workspace requirement checks", merged 2025-10-03 as `f7082e9`) deleted
`checkCuBLASConfigDeterministic` and its alert. The reason given is that since CUDA 11.x
PyTorch allocates a cuBLAS workspace per handle and stream (#85447), which NVIDIA documents
as enough for reproducible results. Two observations support it:

- `aten/src/ATen/Context.cpp` on PyTorch `main` no longer mentions
  `CUBLAS_WORKSPACE_CONFIG` or the check.
- In the local PyTorch 2.10.0+cu128 binaries, `libtorch_cpu.so` still holds the generic
  "does not have a deterministic implementation" alert. No library holds the old cuBLAS
  alert or the `:4096:8`/`:16:8` values. The only remaining use of the variable is
  workspace sizing.

So the probe's expectation was out of date for PyTorch 2.10 and later. The probe tested a
check that Kaggle's PyTorch 2.11 does not have. Run 2's result stays `parity_failed` as
recorded.

**Decision (2026-10-25).** The user: "I agree skip run 3 and mark GR04 complete." Run 3 is
not run. GR04 rests on the four passing probes and on `configure_determinism`, which refuses
to start without `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Our own protection does not depend on PyTorch's check. `configure_determinism` refuses to
start without `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which is tested, and every run so far set
it. What run 2 shows is that PyTorch itself would not catch a missing setting in this
environment.

## Findings

1. **GR06 restart is in-process.** `restart_case()` in `scripts/gpu_ready_parity.py`
   builds three trainers in one Python process: straight, first and resumed. The
   checkpoint goes through real D3 storage as bytes on disk. The resumed trainer is new
   and is rebuilt from the fixture. But the CUDA context, cuBLAS handles, allocator
   caches and global generators are shared with the run that wrote the checkpoint. So
   run 1 proves resume from the serialized state within one process. It does not prove
   resume in a new process, which is the case a real interruption produces. Closing this
   needs a fresh-process variant: one process trains and publishes, a new process
   retrieves, restores and finishes, both in the same Kaggle session. That is a code
   change plus one approved run.
2. **GR10 limits are partly indirect.**
   - GPU-hours have no cumulative ledger. The bound of 1.5 hours follows from at most 3
     runs × 30 minutes.
   - The output size is checked after `kernels output` has downloaded the files, so an
     oversized output would be downloaded before it is refused.
   - Actual GPU time is read from the log, not from a Kaggle usage record.
3. **Provenance records two expected dirty paths.**
   - `multiautoresearch/adapters/amazon4u.py` carries the user-approved redaction fix.
     It is not in the kernel bundle, which holds only `scripts/`, `tests/` and the plan.
     It ran only on the local launcher side.
   - `.venv` is the temporary symlink the agent creates in this worktree for running
     tests.
4. **The O-OFFLINE closure has no hash pin.** No test or record pins
   `multiautoresearch/adapters/amazon4u.py`, `amazon4u_kernel.py` or `__init__.py`. The
   H28 files, the protocol baselines and the D1 to D3 files are pinned. On 2026-10-24 the
   agent compared all modified tracked files with the frozen sets: H28 (28 paths, none
   changed), protocol baselines (19 paths, none changed) and O closure (4 paths, only
   `amazon4u.py`, the two approved strings).
