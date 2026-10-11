# Block 3 — reproducibility and uncertainty

**Status:** user-approved protocol and clarifications before comparative training.
Implementation/tests pending; no models have been run. This resolves the earlier
proposed seed values and uncertainty framework. The exact bootstrap contrast/metric
list is now user-approved and frozen below; it may not be changed based on test
exposure. Block 3's protocol decisions are complete; implementation/tests remain
pending. The baseline protocol is now frozen as v1.0.0; execution/tuning settings
and scoped permissions remain pending. See [protocol lifecycle](protocol-lifecycle.md).

## Fixed seeds and dataset membership

Use training seeds **42, 2026, 3407** for every learned method in every category ×
feedback-condition group. There is no random split seed: preserve published
train/validation/test assignments. RNG choices must not change dataset membership,
warm pools, candidate definitions, eligibility or held-out assignments.

Seed Python, NumPy, PyTorch CPU/CUDA where applicable, initialization, epoch
shuffling, negative/neighbor sampling, DataLoader generators and workers. Derive
worker seeds deterministically with a seeded generator and worker initialization.
Where practical, use separately recorded initialization/shuffling/negative/neighbor
RNG streams derived from the run seed; record derivation and RNG algorithms/state
so streams can be reconstructed. Do not rely on process-randomized Python hash
values for reproducible derivation.

## Determinism

Enable deterministic algorithms where supported. Fail first on an unsupported
operation; do not automatically disable determinism. Any required nondeterministic
exception must be explicitly documented/approved, with operation, hardware and
software details recorded. GPU scatter/index-add, parallel reductions and kernels
may remain nondeterministic depending on their implementation/environment.

Claim reproducibility under recorded seeds, configuration, data and execution
environment **to the extent supported by framework and hardware**. Do not claim
bitwise equality across GPU models, CUDA/framework/driver versions or environments.

## Run provenance and completed artifacts

Record method, category, condition, seed, resolved model/training configuration,
protocol/configuration versions, Git commit and dirty-tree status. A dirty flag
alone is insufficient: preserve relevant tracked source/config diffs and hashes
plus reproducible snapshots of relevant untracked source files. Do not capture
credentials, unrelated projects or huge generated datasets in a code snapshot.

Record data/split fingerprints, condition warm-pool identity, KG fingerprint where
applicable, Python/library/framework versions, OS, CPU/GPU, CUDA/driver versions,
runtime, failure status and known nondeterministic exceptions. Use separate run
directories. Completed runs are immutable except explicitly versioned post-processing
artifacts linked to their source predictions; never overwrite original results.

## Validation-only selection and resumability

Select hyperparameters/checkpoints exclusively with validation NDCG@10 under the
frozen selection rules: Block 2 now sets patience 10 validation checks with strict
score improvement, earliest exact-score tie, and a maximum of 100 epochs. The
profiled validation frequency remains unresolved.
Neither test predictions nor bootstrap intervals select models, seeds, checkpoints,
search configurations or which results to report.

Resumable checkpoints include model/optimizer state, epoch/step, best validation
score/selected epoch, early-stopping counters, relevant RNG and sampler state,
and scheduler/mixed-precision scaler state if applicable. Scientific settings may
not silently change on resumption, including batch size, sampling and population.

Epoch-boundary reproducible resumption is the minimum intended guarantee.
Claim exact mid-epoch continuation only when sampler order/cursor, batch position,
worker/prefetch behavior and relevant RNG states are captured and independently
tested. Otherwise explicitly label support as epoch-boundary only. No resumption
capability has been implemented or verified by this documentation.

## Fixed evaluation population and fallback routes

Freeze the eligible-user population per category × condition × split before
comparing methods. Every method and training seed uses the same users. Failures,
missing learned representations and skipped training updates do not silently change
that population. Full ranking uses no sampled evaluation negatives.

Otherwise eligible users with no positive training history remain included via
the agreed **evaluation/inference-only** training-positive popularity fallback.
No synthetic training examples/model input injection are allowed. Retain route
labels and report counts/subgroup metrics separately from personalized predictions,
including KG-only's additional missing-channel fallback reasons. Scoring failures
remain failures, not a new fallback route.

## Per-seed and per-user results

Report **all registered metrics** per seed: NDCG@10, HitRate@10,
CatalogCoverage@10, NDCG@5/@20 and HitRate@5/@20. For learned methods summarize
with mean ± **sample** standard deviation across the three seeds (denominator
n-1). Three seeds provide a descriptive stability measure, not a precise training
population distribution estimate. Truly deterministic Popularity/KG-only methods
need one execution, not three artificial identical runs.

Store per-user test ranking metrics in stable user-ID order with eligibility,
route labels and prediction/checkpoint provenance. For learned methods, average
each user's metric over the three fixed seeds before paired bootstrap. Average
metrics, not scores/ranks to form a newly evaluated recommendation ensemble.
For deterministic methods use the single fixed per-user metric.

## Frozen bootstrap contrast and metric list

Within **each** of the three categories × two feedback conditions, report:

| Direction (positive favors KG + GNN) | Bootstrap metrics |
|---|---|
| Heterogeneous KG + GNN minus Popularity | NDCG@10, HitRate@10 |
| Heterogeneous KG + GNN minus BPR-MF | NDCG@10, HitRate@10 |
| Heterogeneous KG + GNN minus LightGCN | NDCG@10, HitRate@10 |
| Heterogeneous KG + GNN minus KG-only | NDCG@10, HitRate@10 |

Four contrasts × two metrics × six separate experiment groups gives **48 planned
descriptive intervals**, provided required runs and eligible populations are
available; failure/empty-population rules still apply. Share each replicate's user
resample across all compared methods, contrasts and both metrics within a group.
Do not pool category/condition groups into a primary inference result.

NDCG/HitRate at 5 and 20 retain their registered per-seed summaries, **without
bootstrap intervals**. CatalogCoverage@10 also remains per-seed reporting only.
Optional methods are outside this contrast plan unless explicitly amended before
experiments. Intervals are unadjusted for multiplicity and must not be presented
as controlled significance claims or used to select models.

**Status:** user-approved list, frozen before experiments. This completes the
previously pending Block 3 decision; it does not authorize training or implement
bootstrap. The point-estimate, resampling and uncertainty rules below are unchanged.

## Paired user bootstrap

For each predeclared method/metric contrast within a category × condition:

1. Complete validation-only selection; freeze fitted models/configurations.
2. Generate fixed test predictions under the registered protocol.
3. Use the common eligible test-user population and seed-averaged per-user metrics.
4. Compute the point estimate on the **complete fixed test population**, e.g.
   mean_user(NDCG@10_KG+GNN - NDCG@10_LightGCN).
5. Draw users with replacement; use the same resample for every method in a replicate.
6. Run **10,000 replicates** with separate bootstrap seed **12345**.
7. Report the bootstrap difference distribution's **2.5th/97.5th percentiles** as
   the 95% percentile interval. Do not replace the point estimate with a bootstrap
   replicate mean.

Bootstrap is post-processing only: no retraining, re-ranking, checkpoint selection,
metric selection or population changes. Record RNG algorithm, percentile method
and implementation version. Generate indices in streaming/chunked form; never
require a full 10,000 × n_users matrix. Chunking must retain the identical user
resample across compared methods within each replicate.

The interval measures **user-sampling variability conditional on fitted models
and fixed predictions**. It does not resample training seeds or incorporate
training-seed uncertainty. Report these separately, not as total uncertainty.
Multiple unadjusted intervals are descriptive, not multiplicity-controlled
significance claims. Use only the frozen contrast/metric list above;
no post-test selection of favorable intervals is permitted.

If no users are eligible, report relevant metrics/intervals as **N/A**, not zero.
Do not fabricate intervals for missing seed results.

## Catalog coverage

CatalogCoverage@10 is a group-level unique-item-union statistic with the agreed
warm-pool denominator, not an additive per-user metric. Do not apply this per-user
bootstrap procedure to coverage. Report it per training seed and mean ± sample
standard deviation across seeds, or once for deterministic methods.

## Failures and recovery

Log OOMs, interruptions, unsupported deterministic operations, non-finite outputs
and failed seeds. Do not replace seeds, change scientific settings/populations,
or rerun until favorable results appear. Logged same-configuration recovery from
an interruption is allowed; protocol-changing reruns require an explicit amendment.
Incomplete runs stay explicit: a two-seed summary must not be presented as the
planned three-seed result. Preserve failure/restart history and distinguish
partial diagnostics from complete results. If scientific changes follow exposure
to test results, the changed experiment is not a fresh unseen-test evaluation.

## Required verification before implementation acceptance

Test RNG/worker derivation and data-membership invariance; checkpoint/resumption
claims and dirty-source provenance; fixed user ordering/population and fallback
labels; n-1 seed standard deviation; shared paired resamples and full-population
point estimates; streaming bootstrap reproducibility; N/A empty populations;
coverage exclusion from per-user bootstrap; and explicit failed/missing seed handling.
Tests must not train models without separately approved test/training scope.

Configuration: `configs/reproducibility.json`. Related:
[evaluation protocol](evaluation-protocol.md),
[compute plan](compute-and-tuning-plan.md),
[training semantics](training-semantics.md).
