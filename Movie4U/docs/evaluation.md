# Evaluation protocol

## Task

The task is **top-K recommendation**. The currently implemented experiment recommends unseen movies likely to receive a rating >= 4. This differs from the book's Chapter 12 example, which treats every rating as an interaction and predicts rating-link existence.

The next experiment will compare all-rating behavior prediction with >=4 preference prediction, as specified below. This documentation update does not change the existing pipeline or establish results for the all-rating condition.

## Splits

For each user, sort by timestamp then movieId. Reserve the latest floor(10%) interactions for test, the preceding floor(10%) for validation, and use the rest for training. Small fixtures reserve at least one row per holdout; users need at least three rows. This is a per-user temporal split, not a global-time deployment simulation; different users' timelines overlap.

All ratings participate in splitting. Positive held-out ratings are relevant items. Users without positive ratings in a given holdout are excluded from that split's metrics and counted explicitly. Their profiles still exist for the demo.

Both recommenders see the same fixed training history for validation and test. Candidate sets contain the entire catalog minus *all training-rated movies*, not just positive ratings. Validation items are not consumed as extra test history. Hidden low ratings remain candidates but are not counted as relevant. No test edge or reverse test edge is used for message passing, scoring, or explanations.

## Metrics

Macro-average Precision@10, Recall@10, NDCG@10 and HitRate@10 over eligible users. Precision uses denominator 10. Coverage is unique recommended movies divided by catalog size. There are no sampled evaluation negatives. Scores are not calibrated probabilities.

## Models

- **KG-only:** weighted genre vectors (IDF), aggregate genres of positively rated training movies into a user profile, cosine similarity, plus a fixed 0.05 normalized log-popularity term learned from positive training ratings.
- **KG + GNN:** two heterogeneous GraphSAGE layers over User/Movie/Genre, learned node embeddings plus genre features, dot-product decoder, binary cross-entropy, one sampled unknown movie per supervised positive. Training unknowns may contain future positives because the trainer does not inspect held-out labels; absence is not confirmed dislike.

The GNN checkpoint is selected by validation NDCG@10 only. The test split is evaluated after selection. Epoch limit and configuration are recorded; the training data fingerprint guards against mismatched checkpoint reuse.

## Planned rating-policy sensitivity analysis

Fix both conditions before running the new experiments:

| Condition | Training positives | Relevant held-out items | Interpretation |
|---|---|---|---|
| **All ratings (primary planned condition)** | Every training-rated movie | Every held-out rated movie | Which unseen movie will the user interact with next? |
| **Rating >= 4 (sensitivity condition)** | Training ratings >= 4 | Held-out ratings >= 4 | Which unseen movie will receive positive feedback? |

MovieLens ratings represent recorded rating interactions, not a complete record of viewing behavior. The >=4 condition measures relevance among observed positively rated movies; it does not establish preferences for unobserved movies. With multiple held-out items, the protocol evaluates future-interaction ranking rather than strictly one next-item prediction.

### Fixed rules across conditions

- Preserve the published-in-this-project temporal split assignments: split **all ratings first**, then apply each condition's relevance policy. Do not re-split filtered data or substitute earlier positive ratings for low-rated holdout items.
- Keep the same catalog and candidate rule in both conditions: exclude **all training-rated movies**, including low-rated ones. Do not consume validation interactions as test history.
- Construct each condition's scoring/message-passing interaction graph from its training positives only. Both methods use the same permitted interaction graph within a condition and the same movie/genre metadata. No held-out rating, review, target edge or inverse target edge may enter training features, embeddings, scoring or explanations.
- Under >=4, lower training ratings remain known history for candidate exclusion but are not positive graph edges. Neither low ratings nor unobserved movies automatically become verified negative examples. Record the training negative-sampling policy; do not consult held-out labels to choose negatives.
- Evaluate every user with relevant held-out items in the all-rating condition. Under >=4, exclude users with no positive held-out target and report their counts separately for validation and test. Do not invent replacement targets or silently drop users whose filtered training profile is empty; define a training-only fallback or explicit cold-start handling before running.
- Report user counts, positive interactions, graph sizes, recommendation coverage, and empty-profile/cold-start counts for each condition. Their eligible-user populations can differ, so absolute metric differences across conditions are not solely effects of the rating threshold.

### Selection and reporting

Use **NDCG@10 as the primary ranking metric** in each condition, with Precision@10, Recall@10, HitRate@10 and coverage as secondary metrics, retaining the definitions above. Retrain and select checkpoints separately using each condition's validation NDCG@10; never select the threshold or checkpoint using test results. Record the rating policy in configurations, graph fingerprints and checkpoint metadata so artifacts cannot be reused across conditions accidentally.

Use identical splits, candidates, relevance rules and evaluation code across methods **within each condition**. Freeze training seeds, tuning budgets, fallback behavior and uncertainty reporting before execution. Report per-condition results over multiple seeds and, where appropriate, paired uncertainty estimates over users.

Improvement by KG + GNN over KG-only under both conditions would strengthen robustness to the feedback definition. Improvement under only one condition is a condition-specific finding, not evidence to discard the other condition. Neither outcome establishes that the GNN must outperform the baseline.

**Implementation status:** the current pipeline implements the >=4 condition only. The all-rating condition, condition-specific graph construction checks and multi-seed sensitivity runner require implementation before this plan can be executed. Existing results remain results of the original >=4 experiment; this prospective plan must not be described as having been registered before those results were observed.

## Interpretation and limitations

This is an MVP, not a SOTA claim. Initial GNN performance may be worse than KG-only. Do not tune against test results; refine using validation data, then report a fresh, clearly documented experiment. Genre-only metadata, unobserved relevance, sparse histories, ID-based embeddings, and MovieLens's development-dataset status limit conclusions. User cold start and true item-inductive inference are not implemented.

Before research claims, add multiple seeds, stronger baselines (e.g. popularity and matrix factorization), validation-only hyperparameter tuning, uncertainty estimates, and a separate cold-start protocol.

## Book reference

Alessandro Negro, *Knowledge Graphs and LLMs in Action*, Chapter 12, section 12.2, printed pages 317–334. Data preparation and heterogeneous graph handling are in 12.2.2–12.2.3; the encoder/decoder is in 12.2.4. Movie4U extends the book's two-node-type graph with explicit Genre nodes and evaluates full-catalog ranking rather than only sampled binary classification.
