# Musical_Instruments profiling report

Status: **accepted by the user (2026-10-25)** as the Musical_Instruments profiling
evidence ("Profiling report. Agree."). It records measurements and an agent
projection. It sets no budget and authorizes no training. Plan:
[`musical-instruments-profiling-plan.md`](musical-instruments-profiling-plan.md).

## Evidence

| Item | Value |
|---|---|
| Run | Profiling run 1, approved by the user: "Approve profiling run 1" |
| Kernel | `dungsunf/amazon4u-profile-mi-run-1-cee20ba8`, code commit `1d7f739` (branch `profiling-mi`) |
| Dataset | Private `dungsunf/amazon4u-mi-profiling`, manifest SHA-256 `1066146b…a7bc`; no test split |
| Hardware | One Tesla T4, torch 2.11.0+cu128, determinism policy on |
| Result | `profile_complete`; `data/reports/profiling-mi-v1/runs/run-1/result.json`, SHA-256 `ceb3045b…9576` |
| Derived numbers | `data/reports/profiling-mi-v1/report-numbers.py` and `.json` (rerunnable from the result) |

Collection checks passed: result matches its hash-chained journal, the bundle's run
binding, one GPU, the fixed determinism policy and the agreed limits and cases.

## Answers to the three profiling questions

**1. Training epoch time.** Mean of epochs 2 and 3. Epoch 1 (`warm_up`) was within 4 % of
them in every case, so a separate warm-up allowance is not needed.

| Case | Sampling | Training steps | Validation | Epoch | Build |
|---|---|---|---|---|---|
| LightGCN 2048, all ratings | 2.2 s | 14.8 s | 3.9 s | **20.8 s** | 3.4 s |
| LightGCN 4096, all ratings | 2.2 s | 7.4 s | 3.7 s | **13.2 s** | 3.3 s |
| LightGCN 2048, ≥4 | 1.8 s | 11.3 s | 3.3 s | **16.5 s** | 3.6 s |
| LightGCN 4096, ≥4 | 1.8 s | 5.8 s | 3.3 s | **10.8 s** | 3.3 s |
| KG + GNN 1024, all ratings | 2.4 s | 53.7 s | 3.7 s | **59.8 s** | 8.4 s |
| KG + GNN 1024, ≥4 | 1.8 s | 44.0 s | 3.4 s | **49.2 s** | 8.5 s |

**2. Full-ranking validation, end to end:** 3.3 to 3.9 s per pass. Inference is about
0.1 s or less; ranking about 2.3 to 2.7 s; Python-side metrics about 1 s.

**3. Peak memory:** GPU allocated 1.16 GiB (reserved at most 2.08 GiB) of the T4's 15 GiB;
host RSS 2.14 GiB. Memory is not a constraint on Musical_Instruments.

Also: no empty-pool skips; every validation pass had all metrics finite. Metric values
were not read (decision 2).

## What the numbers mean

- **LightGCN step time does not depend on batch size** (0.076 s at 2048 and 4096, all
  ratings). Each step propagates the full graph, so the step count sets the epoch time,
  and batch 4096 halves it. Measured, not a guess.
- **KG + GNN is about 3× LightGCN 2048 per epoch**, and 90 % of it is training steps
  (0.138 s per step, 388 steps). Its cost is the full-neighborhood masked forward per batch.
- **Validation is cheap on the GPU** (about 4 s, against about 70 s on the local CPU), so
  validating every epoch, as the plan says, costs 6 to 31 % of an epoch.
- **The preflight was accurate.** Its gate projection was 655 s; the Kaggle log ended at
  about 698 s, including boot and the notebook conversion.

## Projection for comparative training (agent estimate)

Assumptions, each open to the user:

- Musical_Instruments only; both conditions; the two learned methods. Popularity and
  KG-only need no GPU training and are left out.
- **Worst case: every run reaches the 100-epoch maximum.** Early stopping (patience 10
  checks) will usually stop earlier, but by how much is unknown before training (guess).
- Final runs: 3 seeds (42, 2026, 3407) × 2 methods × 2 conditions = 12 runs.
- Tuning: the frozen trial budget is still null. The compute plan's planning range is 2
  to 3 configurations; counted here as one seed per configuration.
- **No checkpoint cost.** It is not measured yet (see below).

| LightGCN batch | Tuning configs | Final runs (h) | Tuning (h) | Total GPU-hours |
|---|---|---|---|---|
| 2048 | 2 | 12.2 | 8.1 | **20.3** |
| 2048 | 3 | 12.2 | 12.2 | **24.4** |
| 4096 | 2 | 11.1 | 7.4 | **18.5** |
| 4096 | 3 | 11.1 | 11.1 | **22.2** |

KG + GNN is 74 to 82 % of every total. The longest single run is KG + GNN, all ratings, at
about 1.7 hours for 100 epochs, so each run fits one Kaggle session.

**Against the verified quota.** Only 30 GPU-hours a week are verified (one account), and
the compute plan keeps 20 % in reserve, leaving 24 hours. The worst cases above are 18.5 to
24.4 hours, so Musical_Instruments alone fits one week only with little or no margin in
the worst case. Early stopping should help, but that is not yet measured.

## Limits of this report

- **One run, one session.** Run-to-run and machine-to-machine variation are not measured.
  Within the run, step times varied by under 2 % (p90 against mean).
- **Checkpoint cost is missing.** The current D2/D3 code took 437 s per publish locally
  (plan doc, "Finding"). Comparative training needs durable checkpoints, so the agreed
  speed-up task must be done and its cost re-measured before any budget is frozen.
- **Ledger booking.** The profiling ledger books the full 2-hour session timeout per run;
  run 1 used about 12 minutes. A training ledger that books the full timeout would
  overcount heavily; its booking rule is a decision for the training budget.
- **Other categories.** Toys_and_Games and Electronics are not covered. Electronics has
  about 27× the edges (KG manifest), so its cost needs its own profiling.

## Decisions for the user

1. **Accept this report** as the Musical_Instruments profiling evidence.
2. **Checkpoint speed-up task** (frozen D1/D2, behavior-preserving): approve its scope, so
   the checkpoint cost can be measured and added to the projection.
3. **Later, when freezing settings:** LightGCN batch size (2048 or 4096; profiling shows
   the cost, not which is better), the tuning trial budget, and how a training ledger books
   GPU time. No recommendation is made here on settings that affect results.

## User decisions (2026-10-25)

| Decision | User's words |
|---|---|
| Report accepted as the profiling evidence | "Profiling report. Agree." |
| Write the plan for real-data checkpoint readiness | "yes please write the plan for fixing checkpoints." |
| Checkpoints keep the full copy of the training data for now; try the speed fix first; come back if loading still fails | "Agree, keep the full copy for now and try the speed fix first. Then come back if loading still fails." |
| Commit the plan and report documents; merge `profiling-mi` into `master` | "Saving the work. Agree." |

The plan: [`checkpoint-real-data-readiness-plan.md`](checkpoint-real-data-readiness-plan.md).
