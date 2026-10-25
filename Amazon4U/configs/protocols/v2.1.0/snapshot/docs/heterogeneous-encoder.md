# Heterogeneous KG + GNN — agreed encoder specification

**Status:** user-approved mathematical architecture and initialization;
implementation/profiling/tests pending. No model or embeddings have been trained.
The baseline protocol is frozen as v1.0.0; execution and authorization remain
separate gates in [protocol lifecycle](protocol-lifecycle.md). This resolves Block 1's previously generic
“relation-aware aggregation”; it does not finalize training hyperparameters.

## Inputs and dimensions

Four node types: User, Product, CategoryPath, Brand. Each node has its own learned
64-dimensional ID embedding, h_v^(0). There are two 64-to-64 message-passing layers,
indexed l=0,1. This is a **transductive heterogeneous GraphSAGE-style recommender**,
not an inductive cold-start model. No text/numeric metadata node features are added.

Interaction encoder edges are binary under the feedback condition's positive
policy. Ratings are neither encoder features nor edge weights. Metadata edges
come from approved KG-v1 relations for training products only.

## Eight directed relations

| Source type | Relation | Destination type | Inverse |
|---|---|---|---|
| User | rated | Product | rated_by |
| Product | rated_by | User | rated |
| Product | belongs_to | CategoryPath | contains_product |
| CategoryPath | contains_product | Product | belongs_to |
| Product | has_brand | Brand | brand_of |
| Brand | brand_of | Product | has_brand |
| CategoryPath | category_child_of | CategoryPath | category_parent_of |
| CategoryPath | category_parent_of | CategoryPath | category_child_of |

Inverse relations are generated from permitted forward edges at encoder-view
construction, not from held-out data. This does not modify immutable KG-v1 files.
Deduplicate direct equivalents; category hierarchy directions remain distinct.
The interaction inverse is generated from condition-positive training edges only.

## Exact operator

For target v, let N_r(v) be its permitted incoming neighbors under directed
relation r in the **current batch-masked encoder graph**. Define R_v to contain
incoming relation types with at least one permitted neighbor.

Within each active relation:

`m_(v,r)^(l) = W_r^(l) * mean(h_w^(l) for w in N_r(v))`

Across relations and with the self term:

`z_v^(l+1) = W_self,t(v)^(l) * h_v^(l) + b_t(v)^(l)`
`             + mean(m_(v,r)^(l) for r in R_v)`

- Each directed relation has its **own 64x64 W_r in each layer**. No sharing across
  inverse directions, relations or layers.
- Each node type has its own 64x64 self matrix and length-64 bias per layer.
  Apply this self term **once per node**, not separately for every relation.
- Mean within a relation, then an unweighted mean across active relation types.
  No learned relation mixing weights, attention or additional message biases.
- If R_v is empty, the neighbor contribution is zero; the self term remains.
  Empty relations do not dilute the cross-relation mean.
- Do not add explicit self-loop messages on top of the specified self term.

`h_v^(1) = ReLU(z_v^(1))`
`h_v^(2) = z_v^(2)`

Use no dropout, layer normalization or output L2 normalization in this initial
architecture. The decoder is the raw dot product:

`s(u,i) = dot(h_u^(2), h_i^(2))`

The dot product has no separate trainable decoder parameters. Biases belong to
self terms, not decoder/item-score offsets.

## Initialization

- Every learned node-ID embedding entry is independently Normal(mean=0, std=0.1),
  equivalently variance 0.01.
- Every relation/self transformation matrix uses Xavier uniform with **gain 1**.
- Every self bias starts at zero.
- Use final training seeds 42/2026/3407 under the subsequently agreed
  [Block 3 protocol](reproducibility-and-uncertainty.md). Record initialization RNG
  stream derivation/state; the architecture decision alone did not set seeds.

## Masking and sampling integration

Use the agreed [target-edge masking](training-semantics.md) before or during every
neighbor-sampling hop. Exclude all batch-supervised direct positive interaction
pairs, their inverse/duplicate equivalents, while preserving valid remaining KG
paths and unrelated interactions. Positive and negative scores use the same
masked graph view; cached unmasked representations must not bypass the exclusion.

Active relation membership derives from the permitted masked adjacency, not an
unmasked degree cache. If sampled propagation is adopted, a sampled neighbor mean
approximates the specified within-relation mean. Do not silently drop a relation
that has permitted neighbors to satisfy an execution shortcut. Sampling strategy,
fanout, batch size and validation/inference execution remain profile-dependent
and must be specified before comparative runs. This approval does not silently
choose replacement sampling, fanouts or an evaluation sampler.

During validation/test, use the permitted training encoder graph with no held-out
interactions. Training-batch masks do not permanently delete training history.
Candidate filtering is separate and does not alter learned user representations.

## Required tests before encoder acceptance

- Compare a tiny hand-calculated two-relation example with within/cross-relation
  means; increasing neighbor count alone must not change relation mixture weights.
- Verify all eight relation schemas and independent per-layer/direction parameters.
- Verify one self term, no relation bias, correct empty-relation handling and
  self-only output when all incoming relations are empty.
- Verify ReLU first layer, identity second, raw dot product, and absence of
  implicit dropout, normalization, extra self loops or score biases.
- Verify initialization settings and reproducibility under a fixed recorded seed;
  distribution checks must not rely on exact empirical moments in tiny samples.
- Verify batch-masked adjacency/neighborhoods, inverse exclusion and unchanged
  valid remaining metadata paths; both BPR scores use the same masked view if
  BPR is adopted.
- Verify active-relation membership changes correctly after masking and does not
  reuse unmasked graph statistics.
- Verify held-out mutations and ratings preserving positive membership cannot
  change encoder inputs with permitted training/configuration fixed.

**Rationale:** relation-specific transforms retain relation semantics; separate
within/cross-relation means avoid edge-count-driven relation dominance. One self
term prevents high-relation-degree nodes from receiving repeated root transforms.
Explicit inverses, activation, initialization and empty cases eliminate implicit
framework defaults. The full architecture remains subject to the warm-only claim
boundary and resource profiling, not automatic training authorization.

Configuration: `configs/model-suite.json`.
Related: [model suite](models-and-baselines.md),
[evaluation protocol](evaluation-protocol.md),
[compute plan](compute-and-tuning-plan.md).
