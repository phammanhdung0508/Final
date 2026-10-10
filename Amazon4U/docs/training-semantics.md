# Training semantics — partially agreed, not implemented

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
The precise skip/fail policy remains to be agreed before execution.

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

## Agreed: training target-edge masking

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
A disjoint partition is not the current plan. Node features, architecture and
neighbor fanouts remain unresolved.

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

## Remaining choices — proposed, not approved

The initial recommendation was BPR loss, uniform eligible-negative sampling,
one negative per positive, Adam with constant learning rate 0.001, explicit
embedding L2 coefficient 1e-5, maximum 100 epochs, validation every epoch,
early-stopping patience 10, and 1,024 positives per batch. These numbers and
policies are **proposals only**, not a runnable frozen training configuration.
The [compute and tuning plan](compute-and-tuning-plan.md) now records the agreed
approach: profile full-ranking validation every epoch on Musical_Instruments first,
keep it if practical, and register any necessary frequency amendment before
comparative experiments. Patience is measured in validation checks; its numerical
value and the final measured schedule are not frozen.

Batch target-edge masking and binary, rating-free interaction encoder inputs
are now agreed above. Neighbor-sampling details, architecture, fanouts, other
node features, seeds, uncertainty, empty-profile fallback and resource/tuning
budgets still require finalization. Three final training seeds are now agreed;
exact values and uncertainty procedures remain pending. The conditional Kaggle
capacity plan and 20% contingency reserve are documented in the compute plan;
2–3 tuning configurations is a planning range, not a frozen trial budget.
