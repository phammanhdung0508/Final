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
Metrics and remaining training decisions are not yet frozen.

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

## Leakage boundaries

- Build interaction graphs and train graph embeddings from training edges only.
- Validation selects hyperparameters/checkpoints; test is reserved for final
  evaluation, not feature selection, early stopping or repeated tuning.
- Fit learned preprocessing on training data only. Declare whether static item
  metadata and unseen item nodes are allowed under a transductive setting.
- Exclude held-out ratings/reviews from user/item features and text embeddings.
- Never use a target review as an input to predict its own interaction or rating.
- Stored training ratings are observations/possible targets, not automatically
  encoder inputs. For supervised rating prediction, do not feed an example's
  target rating back through its own edge. Before GNN training, specify target
  edge masking or a disjoint message-passing/supervision design, including inverse
  edges; document any link-reconstruction objective separately.
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
3. Subset policy and seed, or explicit decision to use full downloaded splits.
4. Evaluation: primary metric, secondary metrics, cutoff K, macro/micro
   aggregation, eligible users/items and handling of empty targets.
5. **Primary candidates/filtering finalized:** full ranking of training-warm
   items; validation filters all training items; test filters all training plus
   validation items, without supplying validation edges to the model. Exclude
   filtered-history and cold targets from warm metrics and report them separately.
   In >=4 sensitivity, warmth is derived from positive training edges; report the
   resulting pool change rather than calling it a fixed-candidate ablation.
   Training negative sampling, score ties and empty-profile fallback remain to
   be frozen. See the linked candidate protocol for exact counts and cold rules.
6. Final approved feature/relation list and per-field coverage.
7. Category-only and optional cross-category experiments, including a concrete
   time-safe graph/embedding construction strategy.
8. Baselines, comparable graph/splits/candidates, validation search budget,
   checkpoint selection rule, training seeds and reporting uncertainty.
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

Metric candidates (NOT finalized): NDCG@K and Recall@K for top-K ranking.
RMSE/MAE are not task metrics for this experiment. Freeze the primary metric and
cutoffs before experiments.

## Freeze and amendments

When finalized, replace DRAFT with FROZEN, fill every unresolved decision, and
record the Git commit identifier in experiment artifacts. Store the protocol,
source revision, configurations, seeds and dataset counts with each run.

Changes after results are observed must be logged with rationale and labeled as
a new/exploratory protocol. Do not silently alter thresholds, features or metrics
based on test outcomes; evaluation under a changed protocol is not the original
pre-specified experiment.
