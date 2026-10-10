# Training semantics — approved recipe, execution and implementation pending

Training remains blocked until the remaining choices in
[evaluation-protocol.md](evaluation-protocol.md) are finalized. This document
separates user-approved rules from recommendations. No training sampler, model
or embeddings have been implemented by this decision-recording update.

## Agreed: >=4 positive and negative eligibility

Maintain two distinct, training-only histories for each user u:

- Positive history P(u): products rated >=4. Only these interactions produce
  positive training examples and positive interaction edges (including reverse
  edges if the eventual encoder uses them).
- Observed history O(u): **all** products rated in training, regardless of rating.
  This is the negative-sampling exclusion set, not an additional positive graph.

Let W4 be the category's condition-specific warm pool, derived exclusively from
>=4 training interactions. Eligible negatives are:

`N(u) = W4 minus O(u)`

An item can belong to W4 because another user rated it >=4 while being forbidden
as a negative for u because u rated it below 4. This is a user-specific exclusion,
not global removal of products receiving any low rating.

Lower-rated training interactions therefore produce neither positive examples
nor positive/reverse-positive edges and cannot be sampled as negatives for that
user. The excluded-item lookup must survive positive-graph filtering; never
reconstruct O(u) from P(u) or from the filtered graph.

The sampler must not consult validation/test item IDs, ratings, timestamps or
reviews to exclude future positives. Consequently, an unobserved training-time
item sampled as negative may later be positive; it is not a verified dislike.
Test-time validation-history filtering is a separate evaluation rule and must
not be reused inside the training sampler.

If N(u) is empty, handle it explicitly and record it. Never relax exclusions,
insert a low-rated observed item, or inspect held-out data to find a negative.
Block 2 now fixes the action: skip affected positive updates and report counts;
fail if no trainable examples remain. No synthetic or regularization-only updates.

**Rationale:** rating filtering changes positive feedback, not whether an item
was observed. Keeping those concepts separate avoids converting low-rated
observations into implicit negatives against the agreed experiment definition.

**Status:** user-approved eligibility/exclusion rules; not yet implemented.
Configuration: `configs/training-semantics.json`.

### Required regression checks before running a sampler

1. Every >=4 positive example originates from a training rating >=4.
2. Every sampled negative belongs to W4 and is absent from O(u).
3. Lower-rated training pairs appear in neither positive examples nor forward/
   reverse-positive graph edges.
4. A globally warm product rated below 4 by u is never sampled as u's negative.
5. Mutating held-out data does not change sampler inputs or output under the
   same training data, configuration and seed.
6. Empty negative pools are detected without silently weakening the rules.

## Agreed: empty-positive-history evaluation/inference fallback

This policy applies across methods to otherwise eligible users with no positive
training history under the feedback condition (notably users with only <4 training
ratings under >=4). It is **not a training-data augmentation policy**.

- Training: generate no personalized positive examples or positive interaction
  edges for that user. Do not fabricate positives, copy popular items into their
  history, or promote low-rated observations into positives.
- Evaluation/inference: route that user to condition-specific training-positive
  popularity scores, then apply the standard warm candidate pool and all normal
  history filters. Do not inspect held-out ratings to build popularity or profiles.
- This is a non-personalized fallback route, not a popularity vector injected into
  the model, a trained user representation or a personalized model prediction.
- Keep eligible users in overall evaluation, but report subgroup counts and
  registered metrics separately for fallback versus personalized routes. Include
  the feedback condition and distinguish empty history from KG-only's additional
  missing-metadata/unusable-weighted-channel fallback reasons.
- Fallback eligibility is determined from training history, not from scoring
  failures. Exceptions, corrupt profiles and non-finite model scores still fail.

**Status:** user-approved across-method empty-positive-history policy; not yet
implemented. The KG-only metadata-profile fallback remains a separate approved
case. Unknown-user/cold-target handling is governed by the existing candidate
protocol, not silently overridden here. This decision does not approve the
remaining loss, optimizer, negative-pool action or seed settings.

Required regression checks: no synthetic positives or encoder edges for these
users; popularity uses condition-positive training data only; identical candidate/
history filtering; no model input injection; route labels and subgroup reporting;
no fallback on scoring failures; held-out mutations leave fallback inputs unchanged.

## Agreed: training target-edge masking for heterogeneous KG + GNN

**Scope amendment:** this section applies to the heterogeneous encoder. Core
LightGCN now uses approved Option B: fixed-graph training-positive edge reuse,
without batch target masks. See [LightGCN methodology](lightgcn-protocol.md).
Held-out-edge exclusion and binary/rating-free inputs apply to both models.

**Issue:** target-edge shortcut / target-edge leakage during training. Encoding
an interaction while supervising its existence can provide a shortcut. This is
not itself validation/test leakage; exclusion of held-out edges is a separate,
mandatory boundary.

**Decision:** for each training batch, mask the union of its supervised positive
interaction pairs from the encoder graph view, including each direct equivalent:

- `u -> i+`;
- `i+ -> u`, if reverse interaction edges exist;
- duplicate or aliased interaction edges representing the same supervised pair.

Apply this exclusion **before or during neighbor sampling**, at every sampling
hop, and to any direct adjacency access used by the encoder. Removing edges only
after retrieving neighborhoods is insufficient. Do not reuse representations
computed from an unmasked graph for that training step.

Do **not** delete unrelated interactions, metadata relations, Brand/Category
nodes, or valid remaining multi-hop KG paths merely because they connect the
supervised user and product. The exclusion targets the direct supervised
interaction, not all evidence from which a recommendation could legitimately
be inferred.

If BPR is adopted, only the positive target interaction requires masking:
eligible negative items have no observed training interaction with that user.
Compute both `s(u,i+)` and `s(u,i-)` from the **same batch-masked graph view**;
do not score one using the original graph or perform asymmetric masking.
This does not finalize BPR loss or its remaining hyperparameters.

Interaction edges supplied to the encoder are **binary**, under the condition's
positive policy. Rating values are not encoder features, edge weights or inputs.
Stored KG-v1 ratings remain source observations for relevance selection/auditing;
this decision does not remove them from immutable sources or structural artifacts.
Validation/test target interaction edges and their direct/reverse equivalents
never enter the encoder graph. Evaluation uses the permitted training graph,
without training-batch target masks or adding held-out interactions.

**Status:** user-approved design; not yet implemented. This supersedes the earlier
unresolved choice between masking and disjoint supervision/message-passing.
A disjoint partition is not the current plan. The heterogeneous architecture,
learned-ID inputs, relation operator, activations and initialization are approved in
[heterogeneous encoder](heterogeneous-encoder.md). Its implementation/execution,
neighbor-sampling strategy and fanouts remain pending profiling/specification.

### Required masking regression checks

1. Every supervised positive pair is absent in both interaction directions and
   all direct duplicate equivalents from the effective adjacency and sampled
   neighborhoods at every hop during its training step.
2. Valid remaining metadata edges and multi-hop KG paths survive masking;
   unrelated interaction edges are not removed.
3. Positive and negative scores use the same masked view; no cached unmasked
   representation bypasses the exclusion.
4. Encoder inputs contain no rating-valued interaction features and no held-out
   interaction edges. Rating changes preserving positive membership must not
   change the binary encoder adjacency.
5. Training masks are batch-scoped: they do not permanently erase permitted
   training history used by subsequent batches or evaluation.

## Agreed: endpoint-only base-ID embedding regularization

For BPR-MF, core LightGCN and heterogeneous KG + GNN, define sets from the valid
supervised batch actually used in the update:

- U_B: unique supervised user IDs.
- I_B: the union of unique positive and negative item IDs.

Use the same explicit penalty for all three learned methods:

`L_reg = (1e-5 / (|U_B| + |I_B|))`
`        * (sum(||e_u^(0)||_2^2 for u in U_B)`
`           + sum(||e_i^(0)||_2^2 for i in I_B))`

Here e^(0) is the **base learned ID row**, not a propagated representation. Count
each typed ID once; user/item namespaces are distinct. Deduplicate items across
positive/negative roles and across examples. Do not separately average each role,
divide by embedding dimension, or insert an extra factor of one-half. This is a
mean squared row norm, not a mean over individual scalar embedding entries.

- Include only supervised endpoints, regardless of full-graph or sampled execution.
- Exclude neighbor-only IDs and all Brand/Category rows. A neighbor that also is
  a supervised user/item endpoint is included only because of that endpoint role.
- Do not regularize the full LightGCN embedding table or GraphSAGE computation
  neighborhood merely because those rows participate in propagation.
- Transformation matrices and biases receive **zero explicit L2 and zero optimizer
  weight decay**. Global optimizer weight decay must also be zero; the explicit
  endpoint penalty must not be replaced by whole-table or decoupled weight decay.
- Neighbor-only parameters may still receive ranking-loss gradients. This penalty
  exclusion does not detach their representations or alter message passing.
- No regularization-only update is created from a batch with no valid supervised
  examples. Block 2 now specifies empty-negative-pool handling: skip/report
  affected positives and fail if no trainable examples remain.

**Status:** user-approved coefficient, scope and normalization; not implemented.
This supersedes the ambiguous proposed “unique participating rows” definition.
Its scope is independent of fanout/graph degree, although ranking-loss gradients
and regularization effects can still vary with batching, data and sampling.
Future tuning of transformation regularization requires an explicit registered
search-space amendment. Optimizer type, ranking loss and other training settings
are not finalized by agreeing its weight-decay constraint.

Required tests before implementation acceptance:

1. Hand-calculate the penalty for distinct user/item endpoints.
2. Repeated endpoints and items appearing in both roles are counted once.
3. Adding neighbor-only nodes or changing fanout does not change this penalty
   when base endpoint rows are fixed.
4. LightGCN uses layer-0/base IDs, not layer-averaged representations; the same
   endpoint formula is applied across learned methods.
5. Brand/Category/neighbor-only rows and transformation parameters receive no
   gradient from this explicit penalty alone, while ranking gradients are allowed.
6. Optimizer groups have zero weight decay and no hidden additional penalty;
   empty invalid batches cannot produce a division-by-zero or synthetic update.

**Rationale:** an endpoint-only definition gives 1e-5 a reproducible meaning
without making regularization scope depend on sampled neighborhoods or the full
propagation graph. Configuration: `configs/training-semantics.json`.

## Block 2 — approved learned-model training recipe

**Status:** user-approved before training, documented but not implemented. This
supersedes the earlier proposed numerical recipe and empty-negative-pool action.
Applies to BPR-MF, core LightGCN and heterogeneous KG + GNN, with their respective
agreed graph semantics and the separate endpoint-only regularization formula.

| Setting | Approved value |
|---|---|
| Ranking loss | Mean BPR, computed stably as `softplus(s_negative - s_positive)` over valid supervised examples |
| Negative distribution | Uniform over the agreed user-specific eligible warm pool |
| Negatives per positive | 1 |
| Optimizer | Adam; betas (0.9, 0.999), epsilon 1e-8, weight decay 0 |
| Learning rate | Constant 0.001 |
| Maximum epochs | 100 |
| Early-stopping metric | Validation NDCG@10 only |
| Patience | 10 validation checks without strictly improved score |
| Exact checkpoint-score ties | Retain earliest checkpoint |
| Empty negative pool | Skip affected positive updates, report counts; fail if no trainable examples remain |

The optimized loss is the mean ranking term plus the already approved explicit
endpoint-only penalty. No hidden optimizer weight decay or neighbor/table-wide
penalty is introduced. Strict improvement means `score > best_score` with no
additional minimum-delta threshold. Exact ties do not reset patience. Validation
failures/non-finite metrics remain failures, not successful patience checks.

Skipping an unsampleable positive is not inventing a replacement example or
relaxing exclusions. Report affected users/positive counts. A batch with no valid
supervised examples produces no optimizer or regularization-only update. Detect
and fail a dataset/condition with no trainable examples rather than reporting a
fitted model. The empty-positive-history inference fallback is separate and does
not manufacture training examples.

**Rationale:** a modest common ranking recipe provides transparent baseline
settings; fixed sampling exclusions, exact stopping semantics and explicit skip
handling avoid implementation-dependent changes to the scientific protocol.
This approval does not authorize an unregistered hyperparameter search.

## Remaining profile-dependent and execution choices

Batch size, GraphSAGE fanout/sampling, exact LightGCN execution and the final
validation frequency are not frozen. Values 1,024 positives and fanout 10 remain
profiling starting recommendations. The [compute plan](compute-and-tuning-plan.md)
requires profiling full-ranking validation every epoch on Musical_Instruments
first, retaining it if practical and registering any necessary schedule amendment
before comparative experiments. Patience is now fixed at 10 **validation checks**,
not epochs, regardless of the subsequently approved validation frequency.
BPR-MF/LightGCN ID initialization distributions remain to specify separately;
the heterogeneous encoder's initialization is already frozen.

Batch target-edge masking for heterogeneous KG + GNN and binary, rating-free
interaction encoder inputs are agreed above. Core LightGCN's fixed-graph
training-edge reuse is an explicit approved exception, not an unrecorded shortcut. Empty-positive-history fallback is now agreed above as evaluation/inference only.
Sampling execution, fanouts and numerical resource/tuning budgets still require
finalization; encoder specifications are in the model docs. Block 3 now freezes
training seeds 42/2026/3407 and the reproducibility/uncertainty framework in
[reproducibility](reproducibility-and-uncertainty.md), including the frozen bootstrap
contrast list: KG + GNN minus each core baseline, for NDCG@10/HitRate@10 in each
category/condition. No test-based contrast/metric selection is allowed. The conditional Kaggle
capacity plan and 20% contingency reserve are documented in the compute plan;
2–3 tuning configurations is a planning range, not a frozen trial budget.
