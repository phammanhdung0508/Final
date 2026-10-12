# Core LightGCN — explicit Option B amendment

**Status:** user-approved methodology before implementation/model results.
Implementation, resource profiling and training hyperparameters remain pending.
No model has been trained. This supersedes the earlier blanket requirement for
batch target-edge masking on all learned graph methods.

## Core baseline: conventional training-edge reuse

Use **standard LightGCN semantics (Option B)** for the core interaction-only
baseline: propagate over a fixed graph containing every permitted condition-positive
training interaction and its reverse. A training-positive pair may remain in that
graph while the same pair is used as a supervised positive. Do **not** remove
batch targets from the core LightGCN graph.

This is conventional training-edge reuse, not validation/test leakage. No held-out
interaction, rating, review or reverse edge enters this graph. Under >=4, only
>=4 training interactions are encoder edges; all raw training-rated items still
belong to the negative-sampling exclusion/history-filter sets.

## Retained architecture

- 64-dimensional learned user/item ID embeddings.
- Three propagation layers over the user–item bipartite training graph.
- Standard symmetric degree-normalized interaction propagation, with binary edges:
  `H^(l+1) = D^(-1/2) A D^(-1/2) H^(l)`.
- Final representation is the arithmetic mean of layers 0–3.
- Raw user/item dot-product scoring; no Brand/Category nodes, feature transforms,
  nonlinear layer activations or added self-loop edges. Layer 0 supplies the ID
  representation in the final mean.

A and its degree normalization are derived from the permitted training graph,
not a batch-masked or held-out-augmented graph. Fixed graph does not mean fixed
embeddings: learned representations change with parameter updates. Computational
caching/propagation scheduling must preserve the agreed optimizer and valid
gradients; those execution details remain to specify after profiling.

## Method names and alternative experiments

- Core label: **LightGCN** (fixed-graph, exact full-graph propagation).
- Option A: **target-edge-masked LightGCN variant**, even if propagation is exact.
  It changes the graph according to supervised batches and is not the core method.
- Sampled propagation requires an explicit protocol amendment and **sampled
  LightGCN variant** labeling. If combined with masking, disclose both changes.

Option A is an optional possible additional experiment, not an approved runnable
configuration. Runtime profiling cannot silently replace the core method, waive
its chosen methodology or introduce sampled propagation. If exact propagation is
infeasible, document resource findings and request an amendment before comparative
runs rather than silently changing the baseline.

## Relationship to heterogeneous KG + GNN

The heterogeneous encoder keeps its agreed batch target-edge masking. Positive
and negative scores there use the same masked graph view. This amendment does
not weaken that model's masking rule or either model's held-out-data exclusion.

The comparison now differs in interaction graph use **as well as architecture
and metadata**. Do not attribute gains solely to metadata. The optional matched
heterogeneous encoder without metadata relations is the cleaner metadata control,
provided its masking and other settings remain matched.

## Required validation before implementation acceptance

- The core LightGCN graph is identical across batches and contains all permitted
  training-positive pairs in both directions, including supervised pairs.
- Validation/test interaction pairs and their reverse equivalents are absent;
  mutation of held-out data cannot change propagation inputs.
- Condition-positive edge selection, binary adjacency and degree normalization
  are correct; low-rated training pairs are absent in >=4 propagation.
- A tiny hand-calculated graph reproduces exact three-layer normalized propagation,
  layer 0–3 averaging and dot-product decoding.
- No hidden relation/feature transformations, target masks, extra self loops,
  sampled propagation or stale-gradient caching change the declared method.
- Shared candidate/filtering/negative-exclusion rules remain unchanged.

**Rationale:** the core baseline should represent established interaction-only
collaborative filtering. Batch masking is a scientifically different variant,
not merely a speed optimization. Methodology must be explicitly chosen separately
from computational feasibility. Configuration: `configs/model-suite.json` and
`configs/training-semantics.json`.
