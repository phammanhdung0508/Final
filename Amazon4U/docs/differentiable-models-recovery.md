# Differentiable PyTorch models recovery, subsystem 1

- Status: `frozen_verification_accepted_main_integration_regression_verified`
- Frozen verification task: `f709d062-ae0e-4eb6-960c-f94728180d1b`, **ACCEPT / task_only / NOT_AUTHORIZED**.
- Main integration: Pi-local import compatibility repair and regression verification;
  **204 tests passed**, including a fresh-process import-order regression.
  Evidence: `data/reports/differentiable-models-main-integration/`.
- The historical interrupted source task remains ESCALATE; the full training harness is not accepted.
- Governed by: `docs/evaluation-protocol.md`, `docs/heterogeneous-encoder.md`, `docs/lightgcn-protocol.md`, `docs/models-and-baselines.md`
- Scope: First subsystem of authorized smaller-scale recovery for the rejected bundled harness.
- Deliverables:
  - Source: [`scripts/differentiable_models.py`](../scripts/differentiable_models.py)
  - Unit tests: [`tests/test_differentiable_models.py`](../tests/test_differentiable_models.py)
  - Documentation: this file.

---

## 1. context and architectural boundaries

Following user approval of a smaller-scale recovery plan (addressing the rejected bundled harness), this subsystem implements exclusively the differentiable CPU PyTorch model architectures, training-only graphs, mathematical operators, and gradient propagation tests.

### Strict scope limits
- **NO optimizer execution**: No `optimizer.step()`, parameter updates, or gradient clipping (the BPR loss helper is provided for tensor loss calculation only).
- **NO training loops**: No epoch iteration, validation cadence, or early stopping checks.
- **NO checkpoints or run manifests**: Checkpoint serialization and provenance logging remain in future subtasks.
- **NO dataset downloads or real-data evaluation**: Verification runs exclusively on tiny synthetic fixtures (<=8 users, <=12 products, <=8 metadata nodes per type, dim 64, CPU).
- **Deployment remains NOT_AUTHORIZED**.

---

## 2. component specifications and contracts

### 2.1 Metadata validation and canonical identities
Per KG-v1 schema (`scripts/build_kg.py` and `docs/kg-schema.md`):
- **CategoryPath IDs**: Canonical `c:` prefix followed by compact UTF-8 JSON array of path components (e.g. `c:["Electronics","Audio"]`).
  - Delimiter splitting (such as ` > `) is strictly forbidden; labels may contain ` > ` characters as valid text.
  - JSON formatting requires `ensure_ascii=False` matching DuckDB's `to_json` literal UTF-8 output.
  - Labels and IDs with leading or trailing whitespace are rejected.
- **Immediate Parent Prefix Derivation**: `derive_parent_category_id(cat_id)` computes the immediate canonical parent by trimming the last path element (depth > 1) or returns `None` for root categories (depth = 1).
- **Brand IDs**: Canonical `b:` prefix followed by non-empty, non-padded brand string (e.g. `b:Sony`).
- **Product Categories Prefix-Membership Contract**:
  - The caller must supply ALL approved KG-v1 memberships in `product_categories` (as sequences of canonical `c:[...]` path prefixes).
  - The graph builder traces immediate and higher-order ancestors up to root for the `category_child_of` and `category_parent_of` hierarchy relations, but does **NOT** fabricate additional `belongs_to` or `contains_product` product membership edges beyond those explicitly supplied by the caller in `product_categories`.
- **Category Hierarchy Contract**: When `category_hierarchy` is supplied, every entry `child: parent` must be validated such that `parent == derive_parent_category_id(child)`. Self-loops, cycles, skipped levels (grandchild -> grandparent), and unrelated parents are strictly rejected. Unreachable hierarchy entries for categories not connected to warm products are excluded from the graph.

### 2.2 HeteroTrainingGraph
Self-contained, training-only bipartite/heterogeneous graph:
- **4 Node Types**: `User`, `Product`, `CategoryPath`, `Brand`.
- **8 Directed Relations and Exact Inverses**:
  - `rated` (User -> Product) & `rated_by` (Product -> User)
  - `belongs_to` (Product -> CategoryPath) & `contains_product` (CategoryPath -> Product)
  - `has_brand` (Product -> Brand) & `brand_of` (Brand -> Product)
  - `category_child_of` (CategoryPath -> CategoryPath) & `category_parent_of` (CategoryPath -> CategoryPath)
- **Condition-Positive Invariants**:
  - In `all_ratings`, all recorded training interactions are edges.
  - In `rating_ge_4`, only training interactions with rating >= 4.0 become edges.
  - Low-rated observations (< 4.0) remain recorded in `all_rated_history` for negative-sampling exclusion, but never enter the graph.
  - Empty trainable graph check: constructing a graph with zero condition-positive interactions explicitly raises `ValueError("Empty trainable graph: no condition-positive training interactions exist in TrainingContext.")`.
- **Warm-Product Reachability**: Only warm products in `context.sorted_warm_items` and metadata reachable from those warm products enter `self.categories` and `self.brands`. Cold products and their associated metadata are strictly excluded.
- **Rating Invariance**: Rating mutations that preserve condition-positive membership produce identical graphs.

### 2.3 DifferentiableLightGCN (option B)
Option B fixed-graph training-edge reuse baseline:
- **Configuration**: Strictly 64-dimensional base embeddings, 3 propagation layers, symmetric degree normalization $\tilde{A} = D^{-1/2} A D^{-1/2}$. Revalidates `config == graph.config` on initialization and re-checks `self.config == self.graph.config` in `forward()`.
- **Caller-Supplied Embeddings**: Base embeddings must be explicitly supplied by the caller as a float32 CPU tensor or mapping. Dictionary rows are strictly validated via `validate_embedding_row` (non-boolean, numeric, length 64, finite). Supplied tensors are detached and cloned (`detach().clone()`) to prevent external storage aliasing.
- **Adjacency Buffer and Validation**: Precomputed symmetric normalized adjacency matrix is registered as buffer `adj_user_item` and explicitly validated in `forward()` (sparse COO layout, CPU float32, expected `(users, warm_items)` shape, valid in-bounds indices, and finite values without NaN/Inf).
- **Exact Parity**: Fully matches the reference `propagate_lightgcn` implementation.
- **Propagation**: Layer averaging $H = \frac{1}{4} \sum_{l=0}^3 H^{(l)}$. Isolated users receive exact $1/4$ base embedding scaling.
- **Decoder**: Raw dot product $s(u, i) = \langle h_u, h_i \rangle$. No extra transforms, biases, or masks.
- **Evaluator Route**: Empty-positive-history users fallback to condition-positive training popularity.

### 2.4 DifferentiableHeteroGNN
Two-layer 64d transductive heterogeneous GraphSAGE-style recommender:
- **Initialization**:
  - Requires explicit integer `init_seed`.
  - Uses local deterministic CPU `torch.Generator(device="cpu")` without modifying global PyTorch RNG.
  - Base ID embeddings: $\text{Normal}(0, 0.1)$.
  - Relation transformation matrices ($8 \times 2 = 16$ matrices of shape $64 \times 64$): Xavier uniform with gain 1.0.
  - Node-type self transformation matrices ($4 \times 2 = 8$ matrices of shape $64 \times 64$): Xavier uniform with gain 1.0.
  - Node-type self biases ($4 \times 2 = 8$ vectors of shape $64$): Zeros.
  - **Parameter Tensors**: Exactly 32 transformation and bias tensors PLUS 4 base-ID embedding tables (User, Product, CategoryPath, Brand), totaling 36 independent parameter tensors, all with distinct data pointers.
- **Message Passing Operator**:
  - Within each active relation: $m_{v, r}^{(l)} = W_r^{(l)} \cdot \text{mean}_{w \in \mathcal{N}_r(v)}(h_w^{(l)})$.
  - Across active relations:
    $$z_v^{(l+1)} = W_{\text{self}, t(v)}^{(l)} h_v^{(l)} + b_{t(v)}^{(l)} + \frac{1}{|\mathcal{R}_v|} \sum_{r \in \mathcal{R}_v} m_{v, r}^{(l)}$$
    (If $\mathcal{R}_v = \emptyset$, neighbor contribution is zero).
  - Activations: Layer 0 uses ReLU ($h^{(1)} = \text{ReLU}(z^{(1)})$); Layer 1 uses Identity ($h^{(2)} = z^{(2)}$).
- **On-the-Fly Batch Target Masking**:
  - Masking takes a set of supervised condition-positive pairs $(u, i)$ and masks both forward and reverse edges $(u, i)$ and $(i, u)$ before both layers.
  - Effective adjacency and active relations $|\mathcal{R}_v|$ are recomputed dynamically.
  - The underlying graph structure is not mutated.
  - Metadata edges (such as `contains_product` and `belongs_to`) are strictly preserved under the mask.
- **Shared Forward Batch Scoring**:
  - `compute_batch_scores(user_ids, pos_items, neg_items)` executes a single fresh forward pass with the union of batch positive masks.
  - Validates positive eligibility ($i \in \text{pos\_history}(u)$) and negative eligibility ($j \notin \text{all\_rated\_history}(u)$).
  - Strictly rejects low-rated training items as negatives even when globally warm.
- **Loss and Endpoint Regularization (Optional Helper, No Optimizers)**:
  - Mean stable BPR ranking loss: $\frac{1}{B} \sum \text{softplus}(s_{\text{neg}} - s_{\text{pos}})$.
  - Explicit base-ID endpoint regularization:
    $$\mathcal{L}_{\text{reg}} = \frac{10^{-5}}{|U_B| + |I_B|} \left( \sum_{u \in U_B} \|e_u^{(0)}\|^2 + \sum_{i \in I_B} \|e_i^{(0)}\|^2 \right)$$
    over deduplicated supervised user and item base rows only. Metadata nodes, neighbor-only user/product rows, and transformation weights receive zero penalty and strictly zero regularization gradient.

---

## 3. verification and regression parity

- **Command**:
  ```bash
  PYTHONDONTWRITEBYTECODE=1 /home/sunf/FSB/Final/Amazon4U/.venv/bin/python -m unittest discover -s tests -v
  ```
- **Results**:
  - Total tests executed: 180 (140 baseline regressions + 40 new subsystem tests).
  - Pass rate: 100% (180 passed, 0 failures, 0 errors, 0 skips).
  - Total suite runtime: ~1.65 seconds (well within the declared 120s budget).
- **Tests Added**:
  1. `MetadataValidationTests`: Canonical CategoryPath JSON identities, literal UTF-8 matching DuckDB, trimmed labels/brands, parent derivation, delimiter preservation, and invalid input rejection.
  2. `HeteroTrainingGraphTests`: All-ratings vs rating_ge_4 edges, warm metadata restriction, cold metadata isolation, rating mutation invariance, category hierarchy cycle and invalid ancestry rejection, unreachable hierarchy exclusion, 8-relation schema and exact inverses, metadata deduplication, and empty positive context explicit failure.
  3. `DifferentiableLightGCNTests`: Reference propagation parity across 3 layers, mandatory caller-supplied embeddings, non-boolean strictly numeric length-64 finite row validation, missing/extra ID rejection, tensor detachment/cloning, config freeze and revalidation, registered buffer adjacency with shape/layout/device/dtype/index/finite validation, parameter corruption rejection, finite float32 overflow rejection, ranking-only backward with neighbor propagation, regularization-only backward with neighbor-only user/item row isolation (strictly zero grad), low-rated observed negative rejection, empty-positive fallback, and exact hand-calculated literal numeric propagation with isolated $1/4$ layer scaling.
  4. `DifferentiableHeteroGNNTests`: Local deterministic RNG reproducibility without global RNG mutation, all-parameters byte-for-byte reproducibility across all 36 parameter tensors, independent matrix memory pointers, cold metadata/held-out parameter isolation, literal 2-layer hand-calculated numeric oracle with changing neighbor counts, fixture cloning with identical mean output, exact 2-layer masked oracle verifying active relation recomputation and metadata preservation, empty-relation self-only output, ReLU and identity activation verification, batch target masking, shared forward scoring with mock/spy verifying single execution, exact hand-calculated endpoint regularization, ranking-backward metadata and neighbor gradients, regularization-only base-row gradient isolation (neighbor-only user and product rows strictly zero grad), parameter corruption rejection, finite float32 overflow rejection, unrelated interaction and metadata preservation under union masks, low-rated globally-warm negative rejection, and input/tensor shape/dtype/finiteness/meta rejection.

---

## 4. pending lifecycle gates

This subsystem satisfies the requirements for Subsystem 1 only.
The following remains pending and NOT authorized:
- Main integration gate is now regression-verified: 163 prior tests + 40 model/graph tests
  + one integration import-order test = 204 passing tests. The main source differs from
  the frozen candidate only by package-attribute backfilling for import compatibility;
  the main test file adds fresh-process package-first/top-level-first coverage. The
  frozen candidate and manifest remain unchanged.
- Milestone I2 training harness integration (full training loops, Adam optimizer execution with frozen constant learning rate 0.001).
- Checkpointing, model artifact serialization, and provenance recording.
- Kaggle profiling execution and tuning plan verification.
- Comparative benchmark training runs.
- Model deployment (status remains `NOT_AUTHORIZED`).
