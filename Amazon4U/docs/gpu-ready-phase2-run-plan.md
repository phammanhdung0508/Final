# Phase-2 run plan (phase 1, deliverable 5)

Status: **AGREED (user, 2026-10-24), not started.** The budget and G4 are agreed.
Runs start after the local CPU work passes and Kaggle access is verified. Nothing here
runs now.
Machine-readable: [`../configs/gpu-ready-phase2-run-plan.json`](../configs/gpu-ready-phase2-run-plan.json).

## Order of work

1. **Local, CPU only (no Kaggle):** implement the approved GPU-profile code beside the
   accepted code. Run the full suite, which must stay at 0 failures, errors and skips,
   plus the new tests on CPU with fakes. Limits: 240 s and 2 GiB per full run.
2. **Kaggle run 1, smoke:** tiny fixture, one epoch, both models. Proves packaging,
   imports, GPU allocation, receipt, status, logs and collection.
3. **Kaggle run 2, parity:** both models, tiny fixtures, CPU oracle and GPU executor
   in the same session. Covers GR03, GR04 and GR05.
4. **Kaggle run 3, restart:** uninterrupted versus checkpoint-and-restore on the same GPU
   in the same session. Covers GR06.

Each run starts only after the previous run's report is reviewed. A failed run is
recorded and diagnosed; any further run needs a recorded reason and stays inside the
budget.

## Fixtures

Owned synthetic fixtures generated inside the bundle: a few users, items, category
paths and brands, as in the existing H-REF fixtures. No Amazon data, no held-out data.
Seeds: 42, 2026, 3407.

## Parity checks and tolerances (proposal)

| Quantity | Comparison | Proposed tolerance |
|---|---|---|
| Forward scores | GPU vs CPU oracle, float32 | max abs 1e-5 and rel 1e-4 |
| BPR loss and endpoint penalty | Same | rel 1e-5 |
| Gradients | Same, per parameter | max abs 1e-5 and rel 1e-4 |
| Parameters after N Adam steps (N = 5) | Same | max abs 1e-4 |
| Restart vs uninterrupted, same GPU | Bitwise expected under deterministic settings | Exact; any difference is reported, not tolerated |
| Top-10 ranking on fixtures | Identical item lists | Exact |

Tolerances are justified in the run report from float32 error growth and are fixed
before the run. They are never adjusted after seeing results.

**Agreed clarification (user, 2026-10-24, before any GPU run).** The
heterogeneous `self_biases_l1.Product` parameter has an exactly zero BPR gradient, so
Adam moves it by rounding noise. It is excluded from the parameter tolerance and checked
by ranking invariance instead. Its reference gradient is reported in every run. See
`gpu-ready-design.md`, implementation status.

The run command is `CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.gpu_ready_parity
--device cuda:0 --accelerator gpu --plan configs/gpu-ready-phase2-run-plan.json --out
<fresh path>`. It computes all three checks in one session: parity, restart and
environment.

## Determinism settings

`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. An
operation without a deterministic CUDA implementation must raise an error. Exceptions need
explicit approval and are logged. Sparse matrix products and scatter or index-add
backward passes are the expected risk points, to be verified in run 2.

## Limits

| Limit | Value |
|---|---|
| Kaggle GPU runs | 3 maximum |
| Wall time per run | 30 minutes |
| Total GPU time | 1.5 hours |
| GPU | One T4 of the Kaggle T4 x2 pair (`CUDA_VISIBLE_DEVICES=0`) |
| Collected output per run | 64 MiB |
| Automatic retries | None |

## Outputs

`data/reports/gpu-ready-v1/run-<n>/`: the plan and authorization hashes, receipt,
status history, bounded logs, collected artifacts, the observed environment and parity
tables. Each run directory is new; nothing is overwritten. The final review goes in
`data/reports/gpu-ready-v1/review.json`.

## Acceptance

GPU-READY is accepted only when GR01 to GR10 each have the evidence named in
`gpu-ready-preparation.md`. Acceptance is `task_only`. It does not establish full-data
feasibility, which needs EX01 and the separate Musical_Instruments profiling plan.
