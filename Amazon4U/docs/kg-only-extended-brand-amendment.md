# Amendment: KG-only extended brand weights (labeled variant)

Status: **AGREED by the user (2026-10-27)**: "2. Add only idea A". It adds a separately
labeled variant. The core KG-only method, its 3-mixture grid and its selection are
unchanged and stay the reported core baseline. This record authorizes no test run.

Machine-readable companion:
[`../configs/kg-only-extended-brand-amendment.json`](../configs/kg-only-extended-brand-amendment.json).
Core method: [`kg-only-scoring.md`](kg-only-scoring.md).

## Timing and exposure

Unlike earlier amendments, this one was proposed **after** validation results were seen.
On Musical_Instruments validation, the core grid scored NDCG@10 0.0011, 0.0046 and
0.0052 for category/brand mixtures 1/0, 0.75/0.25 and 0.5/0.5 (all ratings), and
0.0011, 0.0049 and 0.0057 (ratings ≥ 4). The score rose with each step toward brand, and
the highest brand weight tried won. The proposal extends that direction only. No test
result has been seen: the test split has never been loaded. Report the variant as
post-validation, never as a pre-registered method.

## What it adds

| | Core KG-only (unchanged) | Variant `KG-only (extended brand grid)` |
|---|---|---|
| Mixture grid, in order | 1/0, 0.75/0.25, 0.5/0.5 | 1/0, 0.75/0.25, 0.5/0.5, **0.25/0.75, 0/1** |
| Selection | Best validation NDCG@10, ties keep the earliest | The same rule over the 5 entries |
| Profiles, IDF, cosine, missing channels, fallback, candidates | Frozen | Identical to core |

With mixture 0/1, users with no brand profile get the existing whole-user
`unusable_metadata_profile` popularity fallback, under the unchanged core rule.

## Reporting

- The core KG-only result is reported in every table, as before.
- The variant is reported beside it with its label and this post-validation note.
- The frozen bootstrap contrast list (KG + GNN minus Popularity, LightGCN and KG-only)
  uses **core** KG-only. The variant gets no confidence intervals unless a further
  amendment adds them before the test run.

## Implementation

`scripts/kg_only_fast.py` (`EXTENDED_GRID`) and `scripts/baseline_validation.py`, in the
`training-mi` worktree, merged into master on 2026-10-28 (commit `8844951`). The reference `KGOnlyScorer` accepts only the core grid. Spot
checks for the two new mixtures use the reference's own profiles and its score formula
`alpha * category + beta * brand`, with alpha and beta set to the new values.

## Status

- Recorded: 2026-10-27.
- Validation run completed: see Results.
- Protocol snapshot: to be frozen together with the research questions before the test
  run.

## Results (validation, 2026-10-27)

Validation NDCG@10 per mixture, category/brand:

| Condition | 1/0 | 0.75/0.25 | 0.5/0.5 | 0.25/0.75 | 0/1 |
|---|---|---|---|---|---|
| All ratings | 0.0011 | 0.0046 | **0.0052** | 0.0052 | 0.0030 |
| Ratings ≥ 4 | 0.0011 | 0.0049 | **0.0057** | 0.0056 | 0.0035 |

The variant selects **0.5/0.5 in both conditions**, the same mixture as core KG-only, so
its results equal core KG-only. 0.25/0.75 is slightly below 0.5/0.5 in both conditions
(all ratings 0.00521 against 0.00523). Brand alone (0/1) is worse: 746 users (all
ratings) and 1,265 users (ratings ≥ 4) have no brand profile and fall back to popularity.
The brand-heavier direction gave no gain. Core KG-only's 0.5/0.5 stands.

Popularity and the three core mixtures reproduced the first run exactly. Every mixture
passed the 1-in-200 per-user spot check against the reference scorer. Outputs:
`data/reports/training-mi-v1/baselines/validation-extended.json` (first run:
`validation.json`). Runtime 21 minutes, peak memory 2.0 GB, on CPU. The test split was
not loaded.
