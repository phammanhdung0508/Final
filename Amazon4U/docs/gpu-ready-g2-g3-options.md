# G2 and G3 options (phase 1, deliverable 3)

Status: **G2 AGREED, G3 AGREED as provisional starting values (user, 2026-10-24).**
Heterogeneous aggregation is agreed for Musical_Instruments (below). G3 settings are for profiling only and
are not comparative-run settings.

User decisions on 2026-10-24:

- G2: "I support using Normal(0, 0.1²) as the official initialization for LightGCN
  (option A)."
- G3: "I approve to use the suggested values as a starting point."
- Heterogeneous aggregation: "using full neighborhoods for Musical_Instruments, with
  Electronics decided after profiling." The analysis is in the last section of this note.

G2 changes frozen science. It is recorded here as agreed; the formal minor protocol
amendment (`amazon4u-protocol-v2.1.0`) is written once, together with the other settings
finalized before comparative training, so that the protocol version changes once.

## G2: LightGCN ID initialization

Pending in `../configs/training-semantics.json` as `LightGCN_ID_initialization_distribution`.
The heterogeneous model is already frozen at Normal(0, std=0.1) for ID embeddings.

| Option | Basis | Effect |
|---|---|---|
| A. Normal(0, std=0.1) | Matches the widely used PyTorch LightGCN reference and Amazon4U's heterogeneous ID initialization | Same ID starting scale for both learned models |
| B. Xavier-uniform | Used by the original TensorFlow LightGCN release | Scale depends on the number of rows and the dimension |

**Agreed: A** (was the recommendation). It keeps the two learned models comparable at initialization and
matches the agreed heterogeneous choice. Seeds stay 42, 2026 and 3407 through the
`initialization` RNG role. The choice is made before any results and is not tuned.
Once approved, record it as a minor protocol amendment (new methodological addition) with
rationale and zero held-out exposure, following `protocol-lifecycle.md`.

## G3: provisional profiling settings

Profiling measures cost. It does not select hyperparameters and never touches the test split.

| Setting | Provisional value to profile | Note |
|---|---|---|
| Validation frequency | Full-ranking validation every epoch | Already agreed for profiling (`compute-plan.json`) |
| LightGCN batch size | 2048 and 4096 BPR triples | Two points to measure; neither is a default |
| Heterogeneous batch size | 1024 | The existing starting point; not a frozen constant |
| Heterogeneous fanout | 10 per relation per hop, 2 hops | Used only if sampling is adopted; see the open decision below |
| LightGCN propagation | Exact full-graph sparse propagation once per step (Option B, no sampling) | Sampled propagation would need an amendment |
| Batch policy | `filter_before_batch` | The only policy implemented in H-REF |
| Evaluation execution | EX01 vectorized path, proved identical to the accepted evaluator on CPU fixtures | See `gpu-ready-design.md` |

## Heterogeneous aggregation (agreed)

The frozen encoder (`heterogeneous-encoder.md`) uses the mean over all permitted
neighbors. `model-suite.json` lists sampled aggregation only "if adopted" and calls it
approximate. Both choices keep batch positive-edge masking before aggregation.

| Option | What it is | Gains | Costs |
|---|---|---|---|
| A. Full neighborhood | Exact masked means over all neighbors, computed as sparse matrix products. Small graphs use the whole graph per step; large graphs use only the nodes the batch needs (its exact 2-hop field) | Matches the frozen encoder and the CPU oracle exactly. No neighbor RNG to save or restore. Validation uses the same code | Costs more per step on Electronics |
| B. Neighbor sampling | Up to 10 neighbors per relation per hop | Fewest edges per step, so cheaper on Electronics | Approximate. Needs its own RNG stream, checkpoint state and a recorded amendment. Inference still needs full neighborhoods to keep validation stable, so option A code is needed anyway |

Graph sizes (KG-v1 manifest, both edge directions):

| Category | Nodes | Edges | Largest category node | Estimated GPU memory, option A whole graph |
|---|---|---|---|---|
| Musical_Instruments | 86 K | 1.0 M | 23,709 products | under 1 GB |
| Toys_and_Games | 610 K | 7.3 M | 154,607 products | about 3 GB |
| Electronics | 2.05 M | 28.1 M | 348,005 products | about 9 GB |

Memory covers parameters, gradients, Adam moments, two-layer activations and the sparse
adjacency in float32. One Kaggle T4 has about 15 GB usable, so option A fits for every
category. These are **estimates from graph sizes, not measurements.**

Time is the limit, not memory. The rough cost per training step is proportional to
the edges touched:

- whole graph: 1.0 M edges for Musical_Instruments, 28.1 M for Electronics;
- option A over the batch's 2-hop field: an estimated 3 M to 5 M edges on Electronics
  at batch 1024;
- option B with fanout 10: about 1.3 M edges.

On Electronics this makes sampling perhaps three times cheaper than option A over the
2-hop field. It makes no material difference on Musical_Instruments.

**Recommendation:** use option A for Musical_Instruments profiling. Decide for
Electronics from the measured step time, before comparative training. Adopt sampling
only through a recorded amendment if option A cannot fit the budget there.

### T4 x2 on Kaggle

- Kaggle offers T4 GPUs as a pair. GPU-READY uses one of them (`CUDA_VISIBLE_DEVICES=0`)
  and records both the visible and the used device count.
- Musical_Instruments and Toys_and_Games: one T4 is estimated to be enough.
- Electronics: memory fits on one T4; total GPU time is the risk. With 100 maximum
  epochs, 3 seeds and 2 feedback conditions per learned model, Electronics alone may need
  more than one week of quota. A resource-driven scope change for Electronics would need
  a recorded, approved amendment before any comparative result is seen.
- Recommended use of the second GPU: one independent run per GPU, such as two seeds in
  parallel. This doubles throughput per session without the gradient synchronization
  and determinism risks of multi-GPU data parallel training.
- Whether Kaggle counts a T4 x2 session the same as a single-GPU session against the
  weekly quota is **unknown** and must be checked on the account.
- Verified quota (user, 2026-10-24): 30 GPU-hours per week on one account, not the 60
  assumed in `compute-plan.json`. This makes the Electronics time risk larger.

## Decisions needed

1. G2: agreed, option A.
2. G3: agreed as provisional starting values.
3. EX01: agreed as a profiling prerequisite.
4. Agreed: option A (full neighborhood) for Musical_Instruments; Electronics decided from
   measured step time before comparative training.
