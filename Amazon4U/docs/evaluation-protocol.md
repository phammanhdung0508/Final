# Pre-training evaluation protocol

Status: **DRAFT — training blocked until all decisions below are finalized.**
No models have been trained for Amazon4U. This document must be finalized and
version-controlled before training or inspecting comparative model results.

The structural [KG-v1 schema](kg-schema.md) and `configs/kg-v1.json` are now
implemented: separate per-category training graphs with category-path and explicit
Brand relations, training-product catalog only, no node features or embeddings.
See [audit findings](kg-input-audit.md). This graph construction policy does not
finalize the model protocol. The task is top-K ranking; all ratings are the
primary condition and >=4 feedback is the planned sensitivity condition. The
[candidate and cold-start protocol](candidates-and-cold-start.md) now specifies
warm-only full ranking and frozen-model train+validation history filtering at test.
The metric/ranking/reporting section is now frozen in
`configs/evaluation-metrics.json`; remaining training decisions are not yet frozen.

## Data and splits

- Categories: Electronics, Toys_and_Games, Musical_Instruments.
- Source: Amazon Reviews 2023; revision and downloaded files recorded in
  `data/raw/download_manifest_all.json`.
- Interactions: published `benchmark/5core/last_out` train/valid/test CSVs.
- Product metadata: complete `raw_meta_<Category>` Parquet shards.
- Join key: `parent_asin`; user key: `user_id`.
- Retain published split assignments; do not randomly re-split.
- Before finalizing, audit duplicate user-item pairs, split overlap, per-user
  chronology, timestamp ties, and the publisher's precise last-out convention.
- Published per-category last-out splits are not necessarily a global time split.
- 5-core selection uses the overall collection; document its selection bias.
- Any development subset needs a reproducible selection rule, seed, counts,
  degree checks and preserved split assignments, fixed before training.

## Dataset scope — agreed before model training

**Decision:** use the complete downloaded benchmark splits for Electronics,
Toys_and_Games and Musical_Instruments, evaluated independently per category.
Do not randomly subsample interactions or evaluation users for reported experiments.
Full metadata remains available, but KG-v1 includes training products and approved
relations only; full download scope does not authorize extra graph nodes/features.

**Execution order:** develop and measure runtime/memory on Musical_Instruments
first (396,958 training interactions; 24,556 warm items), then apply the same
protocol to Toys_and_Games and Electronics. Tiny synthetic fixtures are for
correctness tests only; they are not research datasets or reported model results.
Both all-rating and >=4 conditions apply to every category, preserving publisher
split assignments and condition-specific eligibility/candidates.

**Rationale:** avoid an additional sampling choice and fragmented user histories;
test robustness across domains with different metadata/interaction coverage; start
with the smallest category to assess computational feasibility. Full downloaded
5-core collections do not imply training splits are themselves 5-core.

**Status:** user-approved scope; full source data and all-rating KGs already exist.
Model training/evaluation under this scope has not started. Configuration:
`configs/dataset-scope.json`. If runtime or memory requires a smaller scope, propose
and document a reproducible amendment before using a subset or observing comparative
model results; do not silently sample users, targets or interactions. Resource
measurements do not authorize selecting categories based on model performance.

## Leakage boundaries

- Build interaction graphs and train graph embeddings from training edges only.
- Validation selects hyperparameters/checkpoints; test is reserved for final
  evaluation, not feature selection, early stopping or repeated tuning.
- Fit learned preprocessing on training data only. Declare whether static item
  metadata and unseen item nodes are allowed under a transductive setting.
- Exclude held-out ratings/reviews from user/item features and text embeddings.
- Never use a target review as an input to predict its own interaction or rating.
- Stored ratings remain observations; interaction encoder edges are binary and
  rating values are not encoder features. Training target-edge shortcut/leakage
  is controlled by masking batch-supervised positive interaction pairs and their
  direct forward/reverse/duplicate equivalents before or during neighbor sampling.
  Preserve unrelated edges and valid remaining multi-hop metadata/KG paths. For
  BPR if adopted, both scores use the same masked graph view; negative target
  edges normally do not exist. This is a training shortcut safeguard, distinct
  from validation/test leakage. See [training semantics](training-semantics.md)
  for the agreed design and required regression checks.
- Cross-category history is a separate experiment. Include only allowed training
  events before the target timestamp across categories. Restrict embedding
  construction too; a filtered feature vector cannot fix future graph messages.
- Record that static product metadata snapshots lack guaranteed historical
  availability. Do not claim strictly historical evaluation without resolving it.
- Test that changing held-out labels/reviews leaves training graphs/features
  unchanged. Record all feature sources, availability times and fitting scope.

## Excluded fields (initial policy)

Exclude from model inputs:

- Product `average_rating` and `rating_number`.
- Full-collection popularity, rating/sentiment aggregates and review counts.
- Validation/test interaction ratings and review-derived representations.
- Target review text, title, helpfulness and any other post-interaction fields.
- `bought_together` until its provenance and temporal availability are established.
- IDs as numeric/ordered attributes; they are identity keys, not ordinal features.

Do not assume `store` is brand. Audit metadata missingness, malformed values,
category hierarchies, duplicates and conflicts before approving attributes.
For KG-v1 only, `categories` and explicit string `details.Brand` are approved
structural relation sources. All remaining metadata fields are excluded, including
`Best Sellers Rank`, full details dictionaries, price and text. No node features
or embeddings are approved. Other fields require a new registered feature policy
and review-derived-content/timing audits before use.

## Decisions requiring finalization BEFORE training

1. **Task finalized: top-K recommendation (implicit-feedback ranking).**
   Rank candidate products for each user; explicit rating prediction is not the
   primary task.
2. **Feedback conditions finalized:** all ratings as positive recorded behavior
   (primary); ratings >=4 as positive preference feedback (sensitivity).
   Preserve the same raw split assignments and do not replace low-rated held-out
   targets. Lower ratings and unobserved items are not verified negatives.
   Build the condition's interaction view using its training positives only.
   Existing KG-v1 is the all-rating structural graph, not a >=4 training artifact.
3. **Dataset scope finalized:** full downloaded splits for all three categories,
   independent graphs/experiments; Musical_Instruments first. No research subset
   or evaluation-user sampling. See the scope section and configuration above.
4. **Metrics/ranking/reporting finalized:** NDCG@10 primary; HitRate@10 and
   CatalogCoverage@10 secondary; NDCG/HitRate at 5 and 20 supplementary. Macro-average
   user ranking metrics, but compute coverage once over recommendations from all
   eligible users. Report separately by category × feedback condition. See below.
5. **Primary candidates/filtering finalized:** full ranking of training-warm
   items; validation filters all training items; test filters all training plus
   validation items, without supplying validation edges to the model. Exclude
   filtered-history and cold targets from warm metrics and report them separately.
   In >=4 sensitivity, warmth is derived from positive training edges; report the
   resulting pool change rather than calling it a fixed-candidate ablation.
   Exact score ties use parent_asin ascending; reject non-finite scores and report
   users with fewer than K candidates. **>=4 training-negative eligibility is now
   agreed:** sample only from that condition's warm pool, excluding all of the
   user's training-rated items, not just positives; never inspect held-out data.
   Distribution, number of negatives, empty-pool action and empty-profile fallback
   remain pending. See [training semantics](training-semantics.md), its configuration
   and the linked candidate protocol; the sampler is not implemented yet.
6. Final approved feature/relation list and per-field coverage.
7. Category-only and optional cross-category experiments, including a concrete
   time-safe graph/embedding construction strategy.
8. Baselines, comparable graph/splits/candidates, validation search budget,
   checkpoint selection rule, training seeds and reporting uncertainty.
   **Core membership finalized:** Popularity, BPR-MF, LightGCN, KG-only and
   Heterogeneous KG + GNN (`configs/model-suite.json`). BPR-MF + metadata is an
   optional feature-augmentation comparison, not a clean KG + GNN ablation;
   a matched encoder without metadata relations is an optional recommended control.
   See [models and baselines](models-and-baselines.md). Learned-ID GraphSAGE-style
   implementations are transductive, with no cold-start claim. **Block 1 is now
   agreed:** 64d BPR-MF/dot product; 64d LightGCN, 3 layers, mean 0–3/dot product;
   2-layer 64d learned-ID heterogeneous GraphSAGE-style/dot product; KG-only
   training-warm IDF/cosine without rating weighting or popularity bonus.
   KG-only uses ordered mixtures 1/0, 0.75/0.25, 0.5/0.5 selected by validation
   NDCG@10 per category/condition, with first-entry tie-breaking. Exact profile/
   missing-channel formulas, aggregation and initialization distribution remain
   pending. Profile LightGCN exact propagation and GraphSAGE batch size/fanout;
   do not waive masking or silently substitute sampled LightGCN. Remaining
   training settings and exact budgets are not approved by this block.
   Three final training seeds are agreed; exact values/uncertainty remain pending.
   The [compute and tuning plan](compute-and-tuning-plan.md) records conditional
   Kaggle capacity, a 20% reserve and a 2–3-configuration planning range. Profile
   Musical_Instruments before freezing trial counts/search spaces; no comparative
   results may be used to choose the budget. No profiling or model runs yet.
9. **Cold-item policy finalized:** exclude from primary warm metrics with
   explicit counts/proportions. Supplementary cold ranking is allowed only after
   registering and testing a metadata-inductive architecture, learned warm
   metadata vocabulary, training-only preprocessing and no cold ID embeddings.
   No such model/runner is currently implemented, so no cold performance is claimed.
   Audit unknown users (verified zero here) separately from empty positive-history
   users (nonzero under >=4). Freeze their training-only fallback before evaluation.
10. Split-boundary timestamp ties (44/17/6 users across the three categories):
    retain publisher assignments and document tie semantics, or register a
    different protocol. No cross-category fitted representations are currently
    allowed because category-specific train timestamps can exceed other targets.

## Metrics, ranking and reporting — frozen before training

**Status:** user-approved decision, documented but not yet implemented in an
Amazon4U evaluator. Configuration: `configs/evaluation-metrics.json`. This replaces
the earlier provisional metric candidates; no Amazon4U model results were inspected.

| Role | Metrics |
|---|---|
| Primary | **NDCG@10** |
| Secondary | HitRate@10, CatalogCoverage@10 |
| Supplementary | NDCG@5, NDCG@20, HitRate@5, HitRate@20 |
| Checkpoint/hyperparameter selection | **Validation NDCG@10 only** |

Relevance is binary according to the feedback condition, not graded by rating.
For the published one-target-per-user holdouts, if the eligible relevant target
has rank r, NDCG@K is `1/log2(r+1)` when r<=K and zero otherwise; HitRate@K is
one when r<=K and zero otherwise. Macro-average these metrics over eligible users.
Recall@K equals HitRate@K with a single relevant target, so it is not an additional
registered metric. Precision and rating-prediction RMSE/MAE are not registered here.

CatalogCoverage@10 is the number of unique items in eligible users' Top-10 lists
**divided by the entire condition-specific warm pool size**, before per-user
history filtering. It is a group-level statistic, not a per-user macro-average.
Use identical eligible users across methods within each category/condition/split.
Coverage is secondary; breadth alone is not evidence of better relevance.

### Deterministic ranking

Apply the documented history/candidate filter before selecting Top-K. Sort by
score descending, then **parent_asin ascending for exactly equal scores**. Do not
use held-out feedback, popularity, randomized order or an unregistered tolerance
to break ties. Apply the same ordering to every method. Reject NaN/infinite
scores rather than silently ranking or discarding them. If fewer than K candidates
remain, return all available candidates without padding and report affected-user
counts separately at each cutoff. Do not change the eligibility population by
method to hide scoring failures.

### Reporting format

Report six independent groups: each of the three categories × each of the two
feedback conditions. Keep validation-selection results separate from final test
results. Do not pool categories into a single primary score.

For each group:

1. Main method-comparison table: NDCG@10, HitRate@10, CatalogCoverage@10 for
   popularity, collaborative, KG-only and KG + GNN methods once implemented.
2. Supplementary table: NDCG/HitRate at 5 and 20; do not select the preferred K
   or checkpoint using these test outcomes.
3. Population table: warm pool size; eligible validation/test users; excluded
   low-rated, cold, history-overlapping and unknown-user targets; empty-positive
   histories and fallback usage; users with fewer than K candidates. If exclusion
   reasons overlap, label reason counts as overlapping and report the distinct
   excluded total separately (or register an explicit exclusion order).
4. Multi-seed results and uncertainty once their procedure is finalized. Do not
   invent intervals for single runs or treat paired user bootstraps as substitutes
   for training-seed variability.

A sensitivity summary compares KG + GNN against each baseline **within each
group**, noting whether improvements occur in both feedback conditions. Absolute
scores/coverage across conditions are not threshold-only comparisons because warm
pools, relevance and eligible populations differ. Do not suppress an unfavorable
condition or assume KG + GNN will win.

**Rationale:** NDCG rewards placing the held-out target near the top; HitRate gives
an interpretable retrieval-success rate; coverage measures recommendation breadth.
Fixed supplementary cutoffs test ranking robustness without choosing K after results.
The remaining fallback, seed/uncertainty, tuning-budget and model decisions still
block training despite this section being frozen.

## Freeze and amendments

When finalized, replace DRAFT with FROZEN, fill every unresolved decision, and
record the Git commit identifier in experiment artifacts. Store the protocol,
source revision, configurations, seeds and dataset counts with each run.

Changes after results are observed must be logged with rationale and labeled as
a new/exploratory protocol. Do not silently alter thresholds, features or metrics
based on test outcomes; evaluation under a changed protocol is not the original
pre-specified experiment.
