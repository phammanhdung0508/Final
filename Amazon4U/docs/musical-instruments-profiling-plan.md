# Musical_Instruments profiling plan

Status: **AGREED (user, 2026-10-25); implementation started.** Decisions 1 to 7, the
5.1 and 5.2 additions, the threshold and the preflight limits are agreed. Each live
profiling run still needs its own approval, bound to a built bundle, as GPU-READY runs
did. Deployment remains `NOT_AUTHORIZED`.
Machine-readable companion:
[`../configs/musical-instruments-profiling-plan.json`](../configs/musical-instruments-profiling-plan.json).

## User decisions recorded on 2026-10-25

The threshold, the mean estimator and the preflight limits were agreed later the same
day: "The threshold. ACCEPT, I would like to add the `mean` ... Preflight limits. ACCEPT
... You may start".

| # | Decision | User's words |
|---|---|---|
| 1 | Profile both feedback conditions | "ACCEPT" |
| 2 | Compute validation metrics in full; record only checks and a SHA-256 | "ACCEPT" |
| 3 | Discard trained weights; keep digests only | "ACCEPT" |
| 4 | Profiling seed 7 | "ACCEPT" |
| 5 | Budget as proposed below | "ACCEPT" |
| 5.1 | Bounded preflight inside the Kaggle run (added by the user) | "incorporate a bounded preflight phase directly into the Kaggle profiling run ... If this estimate exceeds the approved threshold, the process should be halted, evidence should be logged, and the budget should not be automatically increased." |
| 5.2 | Separate warm-up, steady-state, validation and checkpoint durations; every epoch recorded (added by the user) | "the time taken for each individual epoch should be recorded, rather than just an average value." |
| 6 | Local CPU smoke test on real data | "ACCEPT" |
| 7 | Private Kaggle dataset: Musical_Instruments graph files, train and validation splits | "It ok too upload ... As long as it stay private." The dataset must be private; the test split is never uploaded |

## What profiling is for

Profiling measures cost on the full Musical_Instruments data, so that comparative
training can get a frozen, measured budget. It answers three questions from
[`compute-and-tuning-plan.md`](compute-and-tuning-plan.md):

1. How long does one training epoch take for each learned model?
2. How long does full-ranking validation take end to end (inference, candidate and
   history filtering, ranking and metrics)?
3. What is the peak GPU and host memory?

It does not select hyperparameters, compare methods, or touch the test split. The
frozen rule applies: "No test model evaluation or performance-driven tuning during
profiling" (`AGENTS.md`).

## Inputs already agreed

| Item | Value | Source |
|---|---|---|
| Category | Musical_Instruments, full 5-core `last_out` split | `dataset-scope.json` |
| Models | LightGCN (Option B, exact propagation) and Heterogeneous KG + GNN | Block 1, `lightgcn-protocol.md` |
| LightGCN initialization | Normal(0, std=0.1) (G2, option A) | `gpu-ready-g2-g3-options.md` |
| LightGCN batch sizes | 2048 and 4096 BPR triples | G3 |
| Heterogeneous batch size | 1024 | G3 |
| Heterogeneous aggregation | Full neighborhood (option A); no fanout | G3, agreed for Musical_Instruments |
| Batch policy | `filter_before_batch` | G3 |
| Validation | Full ranking every epoch, EX01 vectorized evaluator | `compute-plan.json`, G3 |
| Training recipe | BPR + endpoint penalty 1e-5, Adam 0.001, one uniform eligible negative per positive | Block 2 |
| Execution | Kaggle, one T4, GPU-READY launcher, approval and ledger | G4, GPU-READY |

Data size (KG-v1 manifest): 57,439 users, 24,556 products, 633 category paths, 3,316
brands, 396,958 training ratings, of which 340,862 are 4 or higher. The validation
split has 57,439 rows.

## Proposed scope

- **Cases.** Six measurement cases:
  - LightGCN at batch 2048 and 4096, under each feedback condition (four cases).
  - Heterogeneous KG + GNN at batch 1024, under each feedback condition (two cases).
- **Epochs.** 3 epochs per case, with full validation after every epoch. Three gives a
  first epoch that includes warm-up and two steady epochs to time.
- **No checkpoint step** (user, 2026-10-25; supersedes the first proposal of one
  checkpoint boundary per case). See "Decision: profile without the checkpoint step"
  below.
- **Seed.** One profiling seed, **7**, outside the frozen final seeds 42, 2026 and 3407,
  so that a profiling run can never be mistaken for or reused as a final fit.

What each case records:

- wall time for negative sampling, training steps, validation inference, ranking and
  metrics, each with CUDA synchronization at the phase boundary;
- steps per epoch, empty-pool skips and examples per second;
- peak GPU memory (`torch.cuda.max_memory_allocated` and reserved) and peak host RSS;
- the environment fingerprint, determinism record and run binding, as GPU-READY records.

## Decisions for the user

1. **Feedback conditions.** Recommendation: profile **both**. The ≥4 condition has a
   smaller graph and a different warm pool, and its code path (separate positive
   history, negatives that exclude all rated items) must work end to end before
   comparative training anyway. On Musical_Instruments the extra cost is small (guess).
   The alternative is the all-rating condition only, projecting ≥4 as cheaper.
2. **Validation metric values.** Recommendation: **compute them in full but do not
   report them.** The kernel computes every metric, because skipping them would not time
   validation end to end. The record keeps only checks (all finite, eligible-user and
   exclusion counts) and a SHA-256 of the metric values. Nobody reads NDCG during
   profiling, so no setting can be chosen from it. The alternative is to report them as
   labeled profiling-only values, which risks performance-driven choices.
3. **Trained weights.** Recommendation: **discard them.** The kernel keeps no model,
   embedding or checkpoint after the run, only digests. The project then still has no
   learned embeddings, and nothing from profiling can be reused as a starting point.
4. **Profiling seed.** Recommendation: 7, as above. Any value outside the final seeds
   works; the point is the separation.
5. **Budget.** Recommendation:
   - one planned Kaggle run with all six cases; at most 2 more runs, each only with a
     recorded reason and its own approval;
   - session timeout 2 hours, runner timeout 1 h 50 min;
   - a ledger cap of 6 GPU-hours for profiling, from the verified 30 GPU-hours a week;
   - output at most 64 MiB; no automatic retry.

   Estimate (guess, from graph sizes): each epoch takes well under a minute for both
   models on a T4, so the six cases fit in one session of under an hour. The 2-hour
   timeout is the margin. If the estimate is wrong, the timeout stops the run and the
   partial log is kept.
6. **Local real-data smoke test.** Recommendation: allow one local CPU run of the data
   loader and **one training step and one validation pass per model** on the real
   Musical_Instruments data, before the Kaggle run. It catches loader and memory bugs
   without GPU time. It is real-data work, so it is listed here for approval. Limits:
   10 minutes, 3 GiB host memory, no weights kept.
7. **Data upload.** The kernel runs with internet disabled. Kaggle allows internet,
   but the user prefers it off because downloads at start-up slow the run (2026-10-25).
   So the data has to be uploaded as
   a **private Kaggle dataset**. This is an outward action and needs approval. Contents:
   the Musical_Instruments KG-v1 parquet files (7.3 MB), `Musical_Instruments.train.csv`
   and `Musical_Instruments.valid.csv`, and a hash manifest. **The test split is never
   uploaded.** The kernel checks every file against the manifest and refuses to start if
   any file is missing, extra or changed, or if any file name contains `test`. The user
   approved the upload on condition that the dataset stays private (2026-10-25). The
   agent has not checked the Amazon Reviews 2023 terms; that check is the user's. The
   upload step creates the dataset as private and verifies it is private before the
   first run.

## Preflight and timing records (5.1 and 5.2, agreed design)

**Preflight phase (5.1).** The kernel runs it before any profiling case. The user's
review on 2026-10-25 added three requirements: a conservative step-time estimate, limits
of the preflight's own, and evidence that survives a timeout.

1. Load and hash-check the data; time it.
2. For each of the six cases, in the planned order:
   - build the graph and model;
   - sample one full epoch of negatives (Python-side sampling is part of every epoch);
   - run 5 warm-up steps, excluded from the estimate, then **50 timed steps**, each
     ended by `torch.cuda.synchronize()`;
   - run one full validation pass.
3. Record for the timed steps the **mean**, median, **p90 and maximum** step time. The
   mean is the direct estimator of total epoch time (user, 2026-10-25). With 50 steps,
   p90 is the 45th fastest; with the 20 first proposed it would be the 18th, which is
   noisy.
4. **Project with each estimator**: data load plus, for each case, build +
   3 × (sampling + steps per epoch × step time + validation), plus the
   time already used, for the mean, median and p90 step times. **The gate uses the larger
   of the mean and p90 projections** (agent choice: p90 is usually larger, but one very
   slow step can lift the mean above it).
5. **Stop early.** After each case's preflight, the kernel projects the remaining cases
   with a lower bound (the measured cases' costs, and for unmeasured cases the cheapest
   measured case of the same model). It stops as soon as that lower bound is over the
   threshold, instead of first measuring all six.
6. If the final projection is over the threshold, the kernel stops before the first case
   with status `preflight_exceeded`. Nothing raises the budget or starts a smaller run.

**Preflight limits of its own** (agreed, user, 2026-10-25):

- time: **900 s** for the whole preflight;
- GPU memory: peak allocated at most **12 GiB** of the T4's 15 GiB;
- host memory: peak RSS at most **12 GiB** (Kaggle sessions have about 30 GiB; guess).

Deadlines are checked between training steps and between validation user chunks
(1,024 users each), and memory after each phase. One full validation pass therefore
cannot overrun a limit by more than one chunk. Going over any limit stops the run with
status `preflight_limit_exceeded` and the measurements so far.

**Evidence that survives a timeout.** `result.json` is written once at the end, so it
cannot be the only record. The kernel also keeps:

- `progress.jsonl` in `/kaggle/working`: one JSON line per finished measurement, with a
  sequence number and the SHA-256 of the previous line. Each line is flushed and
  `fsync`ed before the next phase starts. A cut-off last line is detected and reported,
  never repaired. Collection checks the chain.
- the same lines on stdout, so the Kaggle log carries them even if no output file
  survives.
- a runner deadline at 6,600 s inside the 7,200 s session. At that deadline the runner
  writes a final journal line and a `result.json` with status `runner_timeout`, then
  exits by itself. Kaggle's own kill is the last resort, not the plan.

Verified in the current launcher: on an `error` terminal status it still downloads the
outputs and the log into a private `.download-*` directory and keeps them. It records a
failure but does not promote them. **Not verified:** whether Kaggle keeps
`/kaggle/working` files when it kills a session for exceeding its timeout. The stdout
copy covers that case if the log is kept. The first run will show which of the two
survives.

Threshold for the projection (agreed, user, 2026-10-25): the gate projection must be at
most **80 % of the runner timeout**, so 5,280 s of 6,600 s.

**Timing records (5.2).** Every case records each epoch on its own, never only an
average:

- `build`: graph and model construction.
- `epochs[i]` for i = 1 to 3, each with `sampling`, `training_steps`, `validation`
  (split into inference, ranking and metrics) and the step count.
- Epoch 1 is labeled `warm_up` (CUDA context, kernel loading and allocator growth).
  Epochs 2 and 3 are labeled `steady_state`.
- `checkpoint`: not measured in this run (see the decision below).
- Peak GPU memory per epoch, and peak host RSS.

The names are `warm_up` and `steady_state`, not "cold start". In this project
"cold start" means users or items without training history, and reusing the word for
timing would be confusing.

## Work needed before the run

The GPU-READY path runs synthetic fixtures only. H-REF's input adapter refuses
anything but a bounded synthetic policy, and that file is frozen. Profiling needs new
code beside it, in a worktree, each piece with CPU tests on fixtures:

| ID | Work | Evidence |
|---|---|---|
| P1 | Real-data loader: KG-v1 parquet and validation CSV into the existing `TrainingContext`, `LightGCNGraph` and `HeteroTrainingGraph`, checked against the KG-v1 manifest hashes and counts. A new real-data resource policy; H-REF's synthetic policy is not changed | Loader output equals the accepted builders on fixtures; hash mismatch, a test file and extra columns are refused |
| P2 | ≥4 condition built in memory from training ratings: positives ≥4, negatives exclude all rated items, condition-specific warm pool. No new on-disk graph | Equals the accepted `TrainingContext` semantics on fixtures, including a low-rated warm item that is never a negative |
| P3 | Profiling runner: preflight with p90 projection, early stop, its own limits and threshold stop (5.1), hash-chained progress journal and runner deadline, the six cases, per-epoch phase timers (5.2), memory peaks, sealed metric digest, weights discarded. Result schema `amazon4u-profile-result-v1` with its own gate list | CPU run on fixtures validates; tampered results refused |
| P4 | Data bundle and launch: dataset manifest builder, kernel hash check, `dataset_sources` in kernel metadata, a profiling approval schema bound to the bundle and dataset hashes, a separate profiling ledger | Fake-CLI tests as in GPU-READY; launch refused without a matching approval or over the ledger |
| P5 | Scale check: the trainer's Python-side sampling and the evaluator's chunking on 397K ratings and 57K users | Measured in the local smoke test (decision 6) |

Frozen files (H28, protocol baselines, D1 to D3, O-OFFLINE pins) are not changed. If P1
to P4 turn out to need a change to one of them, the agent stops and asks.

## After profiling

The agent writes a profiling report with the measured times and memory, and a
projection of GPU-hours for comparative training on Musical_Instruments, with its
assumptions stated. The user then decides on the frozen execution settings and the
tuning budget, before any comparative training. Toys_and_Games and Electronics are
rechecked separately; this plan does not establish their feasibility.

## Approval steps

| Step | Approval |
|---|---|
| This plan, with decisions 1 to 7 | User: agreed 2026-10-25 |
| Preflight threshold and limits (5.1) | User: agreed 2026-10-25 |
| P1 to P4 implementation in a worktree | Covered by approving this plan |
| Local smoke test (decision 6) | Agreed 2026-10-25 |
| Private dataset upload (decision 7) | Agreed 2026-10-25, private only |
| Each Kaggle profiling run | Its own approval, bound to the built bundle and dataset hashes |
| Comparative training | Separate, after the profiling report and the settings freeze |

## Implementation progress (2026-10-25)

Worktree `profiling-mi` (branch `profiling-mi`, from `master` at `a654f64`).

- **P1 and P2 done.** `scripts/profiling_data.py` stages, verifies and loads the dataset.
  `tests/test_profiling_data.py` has 5 tests. Each of three deliberate breaks fails them:
  the rated-edge cross-check, the test-file refusal and the condition. Both conditions use
  the accepted `TrainingContext` unchanged.
- **Local staging.** `data/reports/profiling-mi-v1/dataset/` (33 MB, git-ignored), manifest
  SHA-256 `46ebf34ad2481e8aa1a981ca0a052d546f98df320d92d48c2b879b0ca4429159`. No test file.
  Nothing is uploaded yet.
- **Smoke test (decision 6) passed:** 189 s and 2.0 GB peak RSS (limits 600 s, 3 GiB), on
  a machine with about 1.2 GB of free RAM. Weights were not kept and metric values were not
  read; the report keeps only timings, checks and a report SHA-256
  (`data/reports/profiling-mi-v1/smoke-local.json`). A first attempt stopped at import
  (the script was outside the package path) before reading any data.

  | CPU, 4 threads, all-rating | LightGCN, batch 2048 | KG + GNN, batch 1024 |
  |---|---|---|
  | Load (shared) | 9.7 s | |
  | Build | 5.4 s | 9.7 s |
  | One epoch of sampling | 2.8 s | 2.7 s |
  | Steps per epoch | 194 | 388 |
  | One step (first, includes warm-up) | 6.5 s | 1.3 s |
  | One validation pass | 68.7 s | 71.1 s |

  No empty-pool skips. Validation dominates on CPU. Whether it is the per-user Python
  ranking loop is a guess; the GPU run will show how much moves to the device.
- **P3 done (worktree, not committed).** `scripts/profiling_runner.py` runs the load, the
  preflight with mean, median and p90 projections, the lower-bound early stop, the
  preflight's own time and memory limits, the runner deadline, and the six cases with
  every epoch timed (sampling, steps, validation inference, ranking and metrics, memory).
  It writes only `progress.jsonl` (hash-chained, fsynced, echoed to stdout with prefix
  `A4U-PROFILE `) and `result.json` (`amazon4u-profile-result-v1`). `validate_result`
  refuses a result that differs from its own journal. No checkpoint step.
  `tests/test_profiling_runner.py` has 8 tests on a synthetic dataset on CPU. Four
  deliberate breaks each failed a test: the journal cross-check, the host-memory limit,
  the hash chain and the early stop. Full suite: 611 tests, 167 s, 0.95 GB peak.
- **Small change to an accepted file.** `vectorized_evaluation.evaluate_validation` takes an
  optional `tick` callback, called after each chunk's ranking and metrics. Without it the
  behavior is unchanged; a test checks the report is identical with and without it.
- **Agent choices in P3, open to the user's review:** the runner deadline counts from
  process start, not from session start, so Kaggle's boot time is outside it; the
  projection adds the data-load time again on top of the time already used, as the plan
  text says (conservative, because the data is loaded once); profiling uses the fixed-seed
  `fixture_rng(7)` stream codecs, since it builds no D1 identity.
- **Not run:** the runner on the real data. The approved local smoke test covered one step
  and one validation pass per model; a full local run is outside it.
- **P4 done (worktree, not committed).** `multiautoresearch/adapters/amazon4u_profile.py`
  sits beside the GPU-READY adapter, which is unchanged:
  - `upload-dir` copies a verified staged dataset and adds Kaggle's
    `dataset-metadata.json` with `isPrivate: true` and license `unknown` (the data terms
    are the user's to check). `create-dataset` runs `kaggle datasets create` without
    `--public`, and refuses metadata that is not private.
  - Before any push, `launch` checks the dataset is `ready` and that Kaggle's own metadata
    says `isPrivate: true`. The CLI leaves the key out when it is false, so a missing key
    refuses the launch.
  - The kernel names only that dataset in `dataset_sources`, has internet off, and runs the
    runner on the one mounted directory whose manifest has the pinned SHA-256. Where Kaggle
    mounts it is not assumed.
  - Approval schema `amazon4u-profiling-run-approval-v1` binds the run, kernel, bundle
    manifest, dataset reference and dataset manifest SHA-256.
  - Ledger in `data/reports/profiling-mi-v1/runs/`, separate from GPU-READY's. Each run
    books 7,200 s; cap 6 GPU-hours, at most 3 runs, never relaunched.
  - Collection keeps `result.json`, `progress.jsonl` and the redacted log. It promotes a
    run only if the result matches its journal, the bundle's run binding, one GPU, the fixed
    determinism policy, and the agreed limits and cases.
  - `tests/test_amazon4u_profile.py` has 13 tests with a fake CLI. They include a CPU run of
    the generated kernel. Each of 7 deliberate breaks failed a test. Full suite: 625 tests,
    156 s, 0.92 GB peak.
- **Dataset layout changed to flat** (agent fix): `kaggle datasets create` skips
  subdirectories by default, so the KG files are now `kg.<table>.parquet`. Restaged to
  `data/reports/profiling-mi-v1/dataset-flat/`, manifest SHA-256
  `1066146b398cad130f1f27c420f705495cb3a6f3213a4ad708b447e92163a7bc`. Every file hash equals
  the first staging; the first `dataset/` is kept unchanged.
- **The runner now accepts an existing output directory** (on Kaggle it is
  `/kaggle/working`). Files are still opened with `x`, so nothing is overwritten.
- **User answers, 2026-10-25:** "1. just note dataset we use from
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main 2. ACCEPT
  3. ACCETP". So: the dataset description names that source; the worktree was committed;
  the run-1 bundle and approval template were built.
- **Committed** on branch `profiling-mi` as `1d7f739` (not merged to `master`, not pushed).
- **Private dataset created**: `dungsunf/amazon4u-mi-profiling`, license `unknown`, from
  `data/reports/profiling-mi-v1/upload-v1/`. `check-dataset` returned `ready` and
  `isPrivate: true`. Kaggle lists the 11 data files with sizes equal to the local
  copies; no test file.
- **Run-1 bundle** `data/reports/profiling-mi-v1/bundle-run-1/`: kernel
  `dungsunf/amazon4u-profile-mi-run-1-cee20ba8`, bundle manifest SHA-256
  `506ce58ef64c60d5b7e9d17449a2c3d82c82b3ab5fb4d2655db1c5070b2388df`, git commit `1d7f739`,
  no dirty paths. Approval template: `approval-run-1.template.json` beside it.
  **Not launched**; run 1 needs the user's signed approval.
- **GPU-READY evidence copied** from the gr06 worktree to `data/reports/gpu-ready-v1/` in
  the main checkout, so it survives archiving that worktree. The original is unchanged.

**Checkpoint phase (closed, see the decision below).** A D3 publish and retrieve of a cuda-v1
checkpoint validates it against a D1 identity, provenance, model and runtime expectations.
For fixtures these come from the D2 test module. For the real data no builder exists yet,
and writing one is the largest piece of work so far. See the reply of 2026-10-25 for the
options.

## Finding: the real D1 identity works, D2/D3 checkpoints do not scale (2026-10-25)

The user chose to build the real identity (option 1). A local CPU probe on the full
Musical_Instruments all-rating data, LightGCN, epoch 0
(`data/reports/profiling-mi-v1/checkpoint-scale-probe.py` and `.txt`):

| Step | Result |
|---|---|
| D1 `build_canonical_identity` over 396,958 ratings and 1,041,782 graph edges | Accepted, 30 to 35 s, 2.4 GB peak |
| Payload build | 19 to 25 s |
| D2 validation alone (under `cProfile`, which slows it) | 1,180 s |
| D3 publish (validation, encoding, write), unprofiled | 437 s, 3.8 GB peak |
| D3 retrieve | Not measured: the system stopped the process for low memory. Not restarted |

Causes, from the profile:

- D2 rebuilds the full D1 identity 4 times and the graph fingerprint 8 times per
  validation (624 s of the 1,180 s).
- `_validate_clean_text` checks every character of every ID in Python: 62.7 million
  calls, about 1 billion character checks (648 s).
- The identity carries the full training records, histories and graph edges, so every
  checkpoint holds them.

Also found: D3 bounds a sparse tensor by its logical shape (57,439 × 24,556 = 1.4 billion
cells), so the IO policy needs `max_tensor_elements` of at least 2^31 for LightGCN's
adjacency buffer.

Extrapolation (guess): Electronics has 27 times more edges, so a publish would take hours.
The D1 and D2 code is frozen. Making it fast needs an approved change to frozen files.

## Decision: profile without the checkpoint step (user, 2026-10-25)

Agent recommendation, accepted by the user ("Ok, please proceed with your recommend"):

- The profiling run measures training epochs and validation only. It builds no D1
  identity and publishes no checkpoint. Training and validation code does not call the
  checkpoint code, so a later checkpoint speed-up does not change the profiling numbers.
- Profiling discards its weights, so a resume checkpoint would protect nothing. Resume
  correctness is already proven by GPU-READY.
- A separate task speeds up D1 and D2 without changing behavior (cache the identity in a
  process, replace the per-character text check), using the existing tests as the oracle.
  It changes frozen files and so needs its own scoped approval. It must be done, and the
  checkpoint cost re-measured, before the comparative-training budget is frozen.
- Status: agreed. The probe evidence stays in `data/reports/profiling-mi-v1/`.

## Run 1 result (2026-10-25)

Approved by the user: "Approve profiling run 1". Status **`profile_complete`**, collected
and checked into `data/reports/profiling-mi-v1/runs/run-1/` (result, journal, redacted log,
record). Tesla T4, torch 2.11.0+cu128. The dataset was re-checked private before the push.
Booked 7,200 s in the profiling ledger; the Kaggle log ends at about 698 s.

Preflight gate projection 655 s (mean 653, p90 655), threshold 5,280 s. No limit was hit.
Peak GPU allocated 1.16 GiB, peak host RSS 2.14 GiB. No empty-pool skips. Every validation
pass had all metrics finite; metric values were not read.

Steady-state epochs (epochs 2 and 3; epoch 1 differed by under 3 %):

| Case | Sampling | Training steps | Step mean | Validation | Epoch total |
|---|---|---|---|---|---|
| LightGCN 2048, all ratings | 2.2 s | 14.8 s (194 steps) | 0.076 s | 3.9 s | ~21 s |
| LightGCN 4096, all ratings | 2.2 s | 7.5 s (97) | 0.077 s | 3.7 s | ~13 s |
| LightGCN 2048, ≥4 | 1.8 s | 11.3 s (167) | 0.068 s | 3.4 s | ~17 s |
| LightGCN 4096, ≥4 | 1.8 s | 5.8 s (84) | 0.069 s | 3.3 s | ~11 s |
| KG + GNN 1024, all ratings | 2.4 s | 53.7 s (388) | 0.138 s | 3.8 s | ~60 s |
| KG + GNN 1024, ≥4 | 1.8 s | 44.0 s (333) | 0.132 s | 3.4 s | ~49 s |

Observations (inferred from these numbers): LightGCN step time does not change with
batch size, so each step's cost is the full-graph propagation, and batch 4096 halves the
epoch. Validation on the GPU is about 4 s, against about 70 s on the local CPU; most of it
is ranking and Python-side metrics, not inference. Report:
[`musical-instruments-profiling-report.md`](musical-instruments-profiling-report.md)
(for user review).
