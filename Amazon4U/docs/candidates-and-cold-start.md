# Candidate universe and cold-start protocol

**Primary all-rating configuration finalized before model training.** This is
Amazon4U's protocol, not Movie4U's different split/catalog protocol. The overall
[evaluation protocol](evaluation-protocol.md) remains a draft until metrics,
training budgets and other remaining choices are frozen. No model has been run.

## Warmth and feedback conditions

The primary condition uses all recorded training ratings as interactions. A warm
item has at least one training interaction; a cold target appears in a holdout
but has no training interaction. The primary candidate universe contains only
warm items, derived solely from the training split. Do not build it from held-out
IDs or the full metadata catalog.

For the planned >=4 sensitivity condition, this document uses **condition-specific
warmth**: at least one >=4 training interaction. Items with only lower training
ratings have no positive training edge in that condition. This policy yields a
slightly different warm pool and must be reported explicitly; it is not a
fixed-candidate, threshold-only ablation. If a fixed all-rating warm pool is desired
instead, amend and freeze that alternative before training, including treatment
of items with no positive training representation. Raw-training coldness and
positive-training coldness must not be conflated.

Within each condition, every method uses the same warm pool and target eligibility.
Across conditions, report results separately; differing relevance, populations,
and warm pools prevent attributing absolute score changes solely to the threshold.

## Frozen model, history-aware filtering

- Full ranking over the warm pool; no sampled evaluation negatives. Training
  negative sampling, if any, is a separate configuration, not candidate sampling.
- Select hyperparameters/checkpoints with validation metrics, then freeze the
  selected checkpoint for test evaluation. No validation/test interactions,
  ratings or reviews enter the graph, feature fitting or embeddings. No refit on
  train+validation is authorized by this protocol.
- Validation candidates: warm pool minus **all** of that user's training items.
- Test candidates: warm pool minus **all** of that user's training and validation
  items. This applies to low ratings too. Validation IDs are used only for this
  set subtraction, never message passing, features, embeddings or user profiles.
- Filter before ranking/top-K extraction. Test filtering intentionally differs
  from the earlier train-only history discussion. Call this a frozen-model,
  train+validation-filtered (hybrid) protocol in reports.
- Exclude targets already in the filtered history from primary metrics and report
  their count. Do not selectively restore a target to the candidate pool.
- Exclude cold targets from primary warm metrics, but report all cold counts and
  proportions. Never silently discard them or claim full-population coverage.
- Preserve original split assignments. In the >=4 condition, low-rated held-out
  targets are not relevant; do not replace them with earlier positive targets.
- Report eligible rows/users after all exclusions. Unknown-user behavior and
  users with no positive training history are distinct cases. There are no unknown
  users in the current files, but the >=4 condition has empty positive histories;
  a common training-only fallback remains to be frozen before that runner is used.

## Supplementary cold-item evaluation: conditional, not currently implemented

Do not report cold scores unless the architecture and inference procedure are
explicitly registered and tested:

1. Represent items from metadata neighbors and approved features, rather than
   primarily from fitted item-ID embeddings. Pure ID models are unsupported.
2. Force metadata use during training, for example warm-item ID-embedding dropout;
   omit an ID embedding entirely for a cold item at inference. An initialized but
   untrained ID embedding is not an inductive cold-start representation.
3. Metadata neighbor identities must already belong to the learned warm vocabulary.
   Categories/brands unique to cold items are unlearned. Report unsupported items
   and unseen-node counts; do not silently invent trained representations for them.
4. Attach cold items through metadata edges only, without any held-out interaction,
   rating or review edge/feature. Never update user representations from cold target
   interactions. Record whether inference is local or recomputes message passing
   over metadata; no learned parameters or preprocessing may be refitted.
5. Exclude average_rating, rating_number, bestseller ranks, popularity and other
   dataset-wide/review-derived aggregates. Fit vocabulary, normalization, PCA and
   trainable text encoders on permitted training data only. Any externally frozen
   encoder requires provenance and a separately approved feature policy.

Split holdout reporting into warm/cold groups, with denominators and coverage.
Pure interaction MF, LightGCN and ID-only KG models are **not applicable** to the
cold group. Valid comparisons can include approved content-only or metadata-only
methods; category popularity must be computed from training interactions only.
Do not present cold-start scores as a general KG advantage when the actual model
is ID-based.

KG-v1 has no fitted item encoder, embeddings or cold-item attachment runner. Thus
cold performance is **not evaluated**, not zero. Before any supplementary run,
freeze its candidate universe, supported metadata policy, metrics and eligibility;
ranking only a selected list of cold ground truths is not a valid substitute.

## Actual pre-training counts

Source: `data/reports/candidate-audit.json`; reproduce with
`scripts/audit_candidates.py`. Cold proportions below use all holdout rows.
Metadata-only-node counts refer to the union of cold items in validation and test
and exactly KG-v1 CategoryPath/Brand identities, not every raw metadata field.

### All ratings (primary)

| Category | Warm items | Test cold rows (%) | Validation cold rows (%) | Test targets in filtered history | Unknown users: valid / test | Cold-only metadata nodes |
|---|---:|---:|---:|---:|---:|---:|
| Electronics | 367,052 | 5,101 (0.3108%) | 2,344 (0.1428%) | 0 | 0 / 0 | 177 |
| Toys_and_Games | 161,656 | 1,441 (0.3334%) | 720 (0.1666%) | 0 | 0 / 0 | 61 |
| Musical_Instruments | 24,556 | 106 (0.1845%) | 62 (0.1079%) | 0 | 0 / 0 | 9 |

Primary warm eligible test rows: **1,635,925 / 430,823 / 57,333**, respectively.
No targets overlap filtered history in either validation or test.

### >=4 sensitivity (condition-specific warmth)

| Category | Warm items | Relevant test targets | Relevant cold test targets | Warm eligible test targets | Users with no >=4 training history | Cold-only metadata nodes |
|---|---:|---:|---:|---:|---:|---:|
| Electronics | 364,473 | 1,243,692 | 6,914 | 1,236,778 | 26,780 | 508 |
| Toys_and_Games | 160,879 | 350,428 | 2,117 | 348,311 | 4,430 | 144 |
| Musical_Instruments | 24,475 | 47,878 | 177 | 47,701 | 395 | 18 |

Empty-positive-history counts are over all holdout users, not necessarily just
warm eligible users. Eligible counts above do not silently exclude empty-profile
users; their fallback must be specified before evaluation. Relevant cold ratios,
validation counts, and cold counts over all rows are in the JSON report.

## Limits and split verification

The local audit directly checked split files: each user has one validation and
one test row, no split pair overlap, and nondecreasing train→valid→test timestamps.
This is consistent with leave-last-two-out. Timestamp ties occur at some split
boundaries. The official tie-breaking implementation has not been verified.

Per-user splitting is not a global time split: another user's training event may
occur after the target time. Category-specific splits also cannot be safely merged
without a new cross-category temporal policy. Metadata is a snapshot with unknown
historical availability. These limitations remain even if the candidate filtering
and graph leakage checks pass.
