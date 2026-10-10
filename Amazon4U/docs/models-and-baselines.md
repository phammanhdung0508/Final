# Models and baselines — design discussion

No Amazon4U model or learned embeddings have been implemented/trained. This
record distinguishes agreed interpretation from proposed implementation choices.
The overall [evaluation protocol](evaluation-protocol.md) remains a draft.

## Agreed: GraphSAGE-style model interpretation

KG-v1 stores graph structure and IDs, not node features or learned embeddings.
GraphSAGE aggregates input representations and neighborhoods; learned node-ID
embeddings can serve as those inputs, but then the complete recommender is
transductive. An unseen product lacks a trained ID embedding.

If the proposed learned-ID implementation is adopted, describe it as:

> A transductive heterogeneous GraphSAGE-style recommender using learned node
> embeddings and KG relations.

Do not describe this implementation as an inductive cold-start model merely
because GraphSAGE's aggregation mechanism can be inductive. Conversely, lacking
stored features in KG-v1 does not force every future implementation to be
transductive: shared transferable metadata features and a suitable encoder could
support unseen products under a separately registered and validated pathway.

**Decision status:** user-approved interpretation and claim boundary. Learned-ID
initialization itself, layer count, dimensions, node features and encoder details
remain proposals, not approved model settings. The current warm-item protocol
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

## Optional comparisons and attribution limits

- **BPR-MF + metadata:** optional feature-augmentation comparison. It is not a
  clean ablation of KG + GNN because architecture and metadata handling differ.
- **Matched KG + GNN without metadata relations:** optional, recommended ablation
  for isolating the contribution of metadata relations. Keep the encoder design,
  decoder and training budget comparable; exact construction is not frozen.

Neither optional comparison is required by the core suite, and neither has an
approved runnable configuration. Comparison with LightGCN also changes architecture
as well as metadata; beating it alone does not isolate metadata's causal contribution.
Report comparisons accurately and do not promote a feature-augmentation baseline
as an architecture-matched ablation.

## Remaining decisions

Core membership is finalized; architectures, dimensions, initialization, KG-only
weighting, model-specific objectives, regularization, optimization, sampling,
tuning budgets and seeds still require explicit agreement. Learned graph methods
must follow the agreed [training target-edge masking rules](training-semantics.md).
Within each category/condition, share candidate, relevance, history filtering and
eligibility rules across methods. No automatic cold-start claims apply to any
model here. Training remains blocked by the overall protocol.
