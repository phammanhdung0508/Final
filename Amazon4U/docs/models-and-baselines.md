# Models and baselines — design discussion

No Amazon4U model or learned embeddings have been implemented/trained. This
record distinguishes agreed interpretation from proposed implementation choices.
The overall [evaluation protocol](evaluation-protocol.md) remains a draft.

## Agreed: GraphSAGE-style model interpretation

KG-v1 stores graph structure and IDs, not node features or learned embeddings.
GraphSAGE aggregates input representations and neighborhoods; learned node-ID
embeddings can serve as those inputs, but then the complete recommender is
transductive. An unseen product lacks a trained ID embedding.

Block 1 now adopts learned node-ID inputs for the heterogeneous model. Describe
that implementation as:

> A transductive heterogeneous GraphSAGE-style recommender using learned node
> embeddings and KG relations.

Do not describe this implementation as an inductive cold-start model merely
because GraphSAGE's aggregation mechanism can be inductive. Conversely, lacking
stored features in KG-v1 does not force every future implementation to be
transductive: shared transferable metadata features and a suitable encoder could
support unseen products under a separately registered and validated pathway.

**Decision status:** user-approved interpretation, claim boundary and Block 1
architecture below. Learned IDs, two layers and 64 dimensions are agreed; the
initialization distribution and exact aggregation operator are now agreed in
[heterogeneous encoder](heterogeneous-encoder.md). Execution/sampling details
remain unresolved. The current warm-item protocol
makes no cold-start performance claims. See
[cold-start requirements](candidates-and-cold-start.md).

**Rationale:** distinguish the inductive capacity of an aggregation architecture
from the actual recommender's inputs, training scope and inference capability.

## Core comparison suite — user-approved membership

1. **Popularity:** non-personalized reference using permitted training feedback.
2. **BPR-MF:** interaction-only collaborative baseline.
3. **LightGCN:** core interaction-graph baseline. It tests embedding propagation
   without Brand/Category KG relations.
4. **KG-only:** metadata/graph-based personalization without learned GNN propagation.
5. **Heterogeneous KG + GNN:** the main graph-aware model; GraphSAGE-style
   implementation details remain to be finalized.

**Status:** core membership finalized by the user, before training. Configuration:
`configs/model-suite.json`. This supersedes the initial four-method recommendation
that treated LightGCN as optional. LightGCN is core because an interaction-graph
baseline is needed before attributing improvements to KGs or metadata.

## Block 1 — frozen architecture and scoring, implementation pending

**Status:** user-approved before training; documented, not implemented. This
supersedes the earlier status in which dimensions, layers and learned-ID inputs
were proposals. The whole evaluation/training protocol remains draft.

| Core method | Agreed specification |
|---|---|
| Popularity | Count permitted positive training interactions per item; all ratings in the primary condition, >=4 in the sensitivity condition |
| BPR-MF | 64-dimensional learned user/item embeddings; dot-product score |
| LightGCN | 64-dimensional embeddings, 3 propagation layers, arithmetic mean of layers 0–3; dot-product score |
| KG-only | Training-warm IDF-weighted category-path/brand profiles; cosine similarity; no rating weighting or popularity bonus |
| Heterogeneous KG + GNN | Two-layer heterogeneous GraphSAGE-style encoder, 64-dimensional learned node-ID embeddings, relation-aware aggregation; dot-product score |

Dot products themselves introduce no trainable decoder parameters; the learned
representations are trainable. Interaction encoder edges are binary and do not
carry rating values. Existing KG-v1 still stores source observations, not learned
embeddings. Warm-only, transductive claim boundaries remain unchanged.

### Agreed KG-only validation grid

Use the following category/brand mixtures in this fixed order:

1. **1.0 / 0.0**
2. **0.75 / 0.25**
3. **0.5 / 0.5**

Select separately for each category × feedback condition using **validation
NDCG@10 only**. Exact validation-score ties select the first entry in the order
above. Freeze the selected mixture before evaluating test; no test metric or
bootstrap interval may select weights.

Keep IDF calculation, profile construction, missing-channel handling and candidate
rules identical across the three trials. The exact formulas are now agreed in
[KG-only scoring](kg-only-scoring.md): smoothed training-warm IDF, independently
normalized product channels, normalized sums for user profiles, missing-channel
cosine zero and no per-item weight renormalization. A whole-user popularity
fallback applies only when no positively weighted user profile channel is usable;
report its counts and subgroup metrics, never use it to hide scoring failures.
This is **limited validation tuning**, not exhaustive optimization. It supersedes the proposed fixed 0.75/0.25 heuristic.

**Rationale:** the small grid is inexpensive and makes the mixture choice more
defensible than an arbitrary fixed split. Training-derived IDF and absence of
rating weights/popularity bonus keep the metadata scoring interpretation explicit.

### Profile-dependent execution and remaining operator choice

- **LightGCN methodology amended to Option B:** the core baseline uses a fixed
  graph of condition-positive training interactions, including supervised positive
  pairs, with held-out edges excluded. No batch target masks apply to this core
  method. See [LightGCN protocol](lightgcn-protocol.md). Exact propagation execution
  remains pending profiling, not methodological selection. Masked Option A is a
  target-edge-masked LightGCN variant; sampled propagation also requires explicit
  amendment and sampled-variant labeling.
- GraphSAGE batch size and neighbor fanout remain profile-dependent. Earlier
  suggestions of 1,024 positives and fanout 10 per relation/hop are starting
  recommendations, not frozen universal/scientific constants. Finalize execution
  settings after resource profiling and before comparative runs.
- The [heterogeneous encoder](heterogeneous-encoder.md) now freezes dedicated
  relation/layer transforms, one node-type self term, within-relation means and
  an unweighted mean across active relations. Eight explicit directed relations,
  ReLU then identity, Normal(0,0.1²) ID initialization, Xavier-uniform gain-1
  transformations and zero biases are agreed. No dropout or layer/output
  normalization. This supersedes the previously generic aggregation proposal;
  framework defaults may not substitute a different operator.
- Block 1 did not itself approve the training recipe. Subsequent Block 2 now
  approves BPR/uniform-one-negative/Adam/constant-0.001/max-100-epoch settings,
  endpoint-only regularization and patience 10 validation checks in
  [training semantics](training-semantics.md). Numerical tuning budgets and
  profile-dependent execution settings remain pending.

## Optional comparisons and attribution limits

- **BPR-MF + metadata:** optional feature-augmentation comparison. It is not a
  clean ablation of KG + GNN because architecture and metadata handling differ.
- **Matched KG + GNN without metadata relations:** optional, recommended ablation
  for isolating the contribution of metadata relations. Keep the encoder design,
  decoder and training budget comparable; exact construction is not frozen.

Neither optional comparison is required by the core suite, and neither has an
approved runnable configuration. Comparison with LightGCN also changes architecture
as well as metadata and now also target-edge handling (fixed training-edge reuse
versus heterogeneous batch masking); beating it alone does not isolate metadata's
causal contribution.
Report comparisons accurately and do not promote a feature-augmentation baseline
as an architecture-matched ablation.

## Remaining decisions

Core membership and Block 1 architecture/scoring are finalized as above, including
the KG-only mixture grid and exact scoring/fallback specification. KG-only scorer
implementation and hand-calculated ranking tests remain pending. The exact GNN
operator and initialization are also agreed; implementation/tests and sampling
execution remain pending. Endpoint-only base-ID L2 (coefficient 1e-5) and zero
transformation/global weight decay are now agreed. Model-specific training
objectives, optimizer, negative distribution/count, learning rate, epoch cap,
stopping and empty-pool handling are now approved in Block 2. Numerical tuning
budgets, profile-dependent execution and BPR-MF/LightGCN ID initialization still
require explicit agreement. Final seeds 42/2026/3407 and the uncertainty framework
are agreed in [Block 3](reproducibility-and-uncertainty.md), now including the frozen
four KG + GNN-minus-baseline contrasts for NDCG@10/HitRate@10 per experiment group. Profile-dependent execution must also be finalized. Learned graph methods
follow their explicitly agreed training graph semantics: heterogeneous batch
masking versus core LightGCN fixed-graph training-edge reuse. Both exclude held-out
edges. See [training semantics](training-semantics.md) and the LightGCN amendment.
Within each category/condition, share candidate, relevance, history filtering and
eligibility rules across methods. No automatic cold-start claims apply to any
model here. Training remains blocked by the overall protocol.
