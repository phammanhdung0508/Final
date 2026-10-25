# KG-only scoring — agreed mathematical specification

**Status:** user-approved before implementation. No scorer or ranking tests have
been implemented/run by this documentation update. This completes the previously
unresolved IDF, profile and missing-channel decisions in Block 1. The overall
baseline protocol is frozen as v1.0.0; execution readiness and authorization
remain separate gates. See [protocol lifecycle](protocol-lifecycle.md).

## Scope and attribute identity

Work separately within each category × feedback condition. Let W be the
condition's training-warm product pool and P(u) the set of distinct products in
user u's condition-positive training history. The primary condition treats all
training ratings as positive interactions; sensitivity uses ratings >=4.

Use KG-v1 identities: complete observed category-path prefix IDs for the category
channel and explicit approved Brand IDs for the brand channel. Deduplicate product
attributes. Do not substitute category leaf strings, store or Manufacturer.
Neither held-out interactions nor their ratings/reviews contribute to profiles,
IDF, vocabulary or popularity. Validation item IDs are used only by the separately
registered test candidate filter, never to update a user profile.

## 1. IDF

For each channel and attribute a, let df(a) be the number of **distinct products
in W** containing a. Use natural logarithm:

`IDF(a) = 1 + ln((|W| + 1) / (df(a) + 1))`

The denominator universe is all products in W, including those missing that
channel, not only products with metadata. Compute vocabulary/df separately for
category-path and brand identities. Each product contributes at most once to each
attribute's df. No held-out or full-metadata-catalog counts are used.

## 2. Product channel vectors

For channel q in {category, brand}, the unnormalized vector for product i is:

`x_q(i)[a] = IDF(a)` if i contains a, otherwise `0`.

L2-normalize each channel independently to obtain v_q(i). If its norm is zero,
keep the zero vector. Categories use every observed KG-v1 path prefix; brands use
only explicit approved Brand identities. Do not fabricate an unknown attribute.

## 3. User channel vectors

For each channel, sum normalized product vectors over distinct positive-history
products:

`z_q(u) = sum(v_q(i) for i in P(u))`

L2-normalize z_q(u) to obtain v_q(u), keeping zero if the norm is zero. A product
without a channel contributes zero to that sum. There are no rating weights,
recency weights, held-out history or popularity terms in these profiles.

## 4. Similarity, missing channels and mixture

Define c_q(u,i) as cosine similarity between the user and product channel vectors,
or **zero if either vector is zero**. For the normalized nonzero vectors, cosine
is their dot product. The final ordinary KG-only score is:

`s(u,i) = alpha * c_category(u,i) + beta * c_brand(u,i)`

Use the agreed ordered grid `(alpha,beta)`:
`(1,0)`, `(0.75,0.25)`, `(0.5,0.5)`.

**No per-item weight renormalization.** A missing channel contributes zero; the
other weight does not increase. This explicitly supersedes the earlier unapproved
suggestion of renormalizing over available channels. Metadata-poor products may
therefore receive lower scores; report the limitation and channel coverage.
A product missing both channels scores zero under ordinary KG-only scoring.
Do not replace its score with popularity or switch to an item-specific fallback.

Select weights separately per category/condition by validation NDCG@10 only;
exact metric ties select the first grid entry. All trials use the same IDF,
profile, missing-channel and candidate rules. Freeze the chosen mixture before
final test evaluation. Never use test results/bootstrap intervals to select it.

## 5. Whole-user fallback and failures

For the mixture being evaluated, a user has a usable channel only when that
channel's weight is strictly positive and the user channel vector is nonzero.
If **no positively weighted user channel is usable**, use whole-user popularity
scores: condition-positive training interaction counts for each candidate item.

This includes no positive training history, no usable metadata profile, or no
usable channel under the selected mixture (for example a brand-only user under
category-only weights). Apply the same warm pool, validation/test history filters,
finite-score requirement and parent_asin tie-breaking to fallback recommendations.
Changing grid weights may change fallback usage; record that usage per trial and
for the selected configuration. Do not change eligible-user definitions.

Report fallback-user counts/reasons and the registered ranking metrics for that
subgroup separately from ordinary KG-only users and overall results. Distinguish
empty positive history from absent/unusable metadata channels. The fallback is
not a popularity bonus blended into ordinary scores. All methods now share an
approved evaluation/inference-only empty-positive-history fallback; no synthetic
training positives or model input injection are allowed. KG-only's additional
unusable-metadata/weighted-channel fallback remains model-specific. Report routes
as non-personalized fallback predictions, not personalized KG scores.

Scoring exceptions, non-finite outputs and corrupted profiles are failures, not
fallback triggers. Valid zero vectors and absent metadata are expected cases;
NaN vectors are not valid missing metadata. Never silently drop failed users.

## 6. Required checks before scorer acceptance

Add small hand-calculated tests for:

- Smoothed IDF and distinct-product df; for |W|=3 and df=2, IDF is
  `1 + ln(4/3)`. Duplicate attributes must not inflate df.
- Independent product-channel normalization and normalized sum user profiles,
  with each positive-history product counted once.
- Missing-channel cosine zero and fixed-weight combination: category cosine 1
  with absent brand scores 0.75 under `(0.75,0.25)`, not 1.
- Two absent item channels yield an ordinary score of zero, not item popularity.
- Whole-user fallback for zero positively weighted profile channels, including
  brand-only users under `(1,0)`, with unchanged candidate/history filters.
- Condition-positive popularity counts and fallback-subgroup reporting.
- Exact grid-score tie order, deterministic item tie order and no padding.
- Held-out mutations do not change profiles/IDF/popularity with training fixed.
- Below-4 training items do not enter >=4 profiles, but remain in history filters.
- Non-finite/corrupt scores fail rather than becoming fallback or exclusions.

**Rationale:** explicit normalization and missing-channel handling prevent
implementation-dependent ranking differences. Zero contributions preserve mixture
semantics; an explicit user-level fallback retains evaluable users without
quietly inserting popularity into ordinary metadata scores.

Configuration: `configs/model-suite.json`. Related documents:
[models and baselines](models-and-baselines.md),
[candidate rules](candidates-and-cold-start.md),
[evaluation protocol](evaluation-protocol.md).
