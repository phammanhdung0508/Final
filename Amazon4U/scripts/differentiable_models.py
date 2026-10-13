"""Amazon4U Differentiable PyTorch Recommender Models and Training-Only Graphs.

Subsystem 1 of the model harness recovery:
- HeteroTrainingGraph: validated 4-node-type, 8-directed-relation training-only graph
  strictly constructed from condition-positive training interactions and warm-product metadata.
- DifferentiableHeteroGNN: 2-layer, 64-dimensional transductive heterogeneous GraphSAGE-style
  model with relation-specific transformations, node-type self terms, within/cross-relation means,
  ReLU then identity activations, dot-product scoring, and on-the-fly batch target masking.
- DifferentiableLightGCN: PyTorch Option B fixed-graph LightGCN implementation with exact
  degree-normalized propagation, layer 0..3 averaging, and dot-product scoring.
- Exact BPR ranking loss and endpoint-only base-ID regularization tensor computation.
"""

from collections import defaultdict
import json
import math
import sys
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonicalize module in sys.modules so scripts.differentiable_models and
# differentiable_models share the exact same module object and class identities.
if __name__ == "scripts.differentiable_models":
    sys.modules.setdefault("differentiable_models", sys.modules[__name__])
elif __name__ == "differentiable_models":
    sys.modules.setdefault("scripts.differentiable_models", sys.modules[__name__])
    if "scripts" in sys.modules:
        setattr(sys.modules["scripts"], "differentiable_models", sys.modules[__name__])

if __package__:
    from .learned_models import (
        LightGCNConfig,
        LightGCNGraph,
        MissingIdError,
        UnknownIdError,
        build_lightgcn_graph,
        propagate_lightgcn,
    )
    from .recommendation_core import (
        ScoringFailureError,
        TrainingContext,
    )
else:
    from learned_models import (
        LightGCNConfig,
        LightGCNGraph,
        MissingIdError,
        UnknownIdError,
        build_lightgcn_graph,
        propagate_lightgcn,
    )
    from recommendation_core import (
        ScoringFailureError,
        TrainingContext,
    )

# Backfill package attributes for modules canonicalized before the scripts package
# object existed. This preserves import identity across top-level and package imports
# when this module is imported early in full-suite discovery.
try:
    import scripts as _scripts_pkg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - scripts-directory import mode
    _scripts_pkg = None
if _scripts_pkg is not None:
    setattr(_scripts_pkg, "differentiable_models", sys.modules[__name__])
    if "scripts.learned_models" in sys.modules:
        setattr(_scripts_pkg, "learned_models", sys.modules["scripts.learned_models"])
    if "scripts.recommendation_core" in sys.modules:
        setattr(_scripts_pkg, "recommendation_core", sys.modules["scripts.recommendation_core"])


# -----------------------------------------------------------------------------
# Metadata Validation and Canonical Identities
# -----------------------------------------------------------------------------

NODE_TYPES: Tuple[str, ...] = ("User", "Product", "CategoryPath", "Brand")

RELATION_SCHEMAS: Dict[str, Tuple[str, str]] = {
    "rated": ("User", "Product"),
    "rated_by": ("Product", "User"),
    "belongs_to": ("Product", "CategoryPath"),
    "contains_product": ("CategoryPath", "Product"),
    "has_brand": ("Product", "Brand"),
    "brand_of": ("Brand", "Product"),
    "category_child_of": ("CategoryPath", "CategoryPath"),
    "category_parent_of": ("CategoryPath", "CategoryPath"),
}

INVERSE_RELATIONS: Dict[str, str] = {
    "rated": "rated_by",
    "rated_by": "rated",
    "belongs_to": "contains_product",
    "contains_product": "belongs_to",
    "has_brand": "brand_of",
    "brand_of": "has_brand",
    "category_child_of": "category_parent_of",
    "category_parent_of": "category_child_of",
}


def validate_category_id(cat_id: str) -> List[str]:
    """Validate canonical KG-v1 CategoryPath identity: 'c:' followed by compact JSON array.

    Rejects non-strings, leading/trailing whitespace, missing 'c:' prefix, invalid JSON,
    non-list JSON, empty paths, blank/whitespace-only/padded labels, and non-canonical
    formatting. Delimiter splitting (such as ' > ') is forbidden: labels may contain
    ' > ' characters as valid label text. Literal UTF-8 is required matching DuckDB to_json.
    """
    if not isinstance(cat_id, str):
        raise ValueError(f"CategoryPath ID must be a string, got {type(cat_id).__name__}: {cat_id!r}")
    if cat_id.strip() != cat_id:
        raise ValueError(f"CategoryPath ID must not contain leading or trailing whitespace: {cat_id!r}")
    if not cat_id.startswith("c:"):
        raise ValueError(f"CategoryPath ID must start with 'c:', got {cat_id!r}")

    payload = cat_id[2:]
    if not payload:
        raise ValueError(f"CategoryPath ID has empty JSON payload: {cat_id!r}")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"CategoryPath ID contains invalid JSON {payload!r}: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"CategoryPath JSON payload must be a list, got {type(parsed).__name__}: {parsed!r}")
    if len(parsed) == 0:
        raise ValueError(f"CategoryPath JSON list must have at least one element: {cat_id!r}")

    for idx, elem in enumerate(parsed):
        if not isinstance(elem, str):
            raise ValueError(f"CategoryPath label at index {idx} must be a string, got {type(elem).__name__}: {elem!r}")
        if not elem.strip():
            raise ValueError(f"CategoryPath label at index {idx} cannot be blank or whitespace-only: {elem!r}")
        if elem.strip() != elem:
            raise ValueError(f"CategoryPath label at index {idx} cannot contain leading or trailing whitespace: {elem!r}")

    # Canonical identity requires compact UTF-8 JSON without trailing or extra whitespaces
    canonical_payload = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if payload != canonical_payload:
        raise ValueError(
            f"CategoryPath ID must use compact canonical UTF-8 JSON formatting: expected 'c:{canonical_payload}', got {cat_id!r}"
        )

    return parsed


def derive_parent_category_id(cat_id: str) -> Optional[str]:
    """Derive immediate parent CategoryPath ID from canonical JSON prefix.

    Returns canonical parent 'c:[...]' if depth > 1, or None if root.
    """
    path = validate_category_id(cat_id)
    if len(path) <= 1:
        return None
    parent_path = path[:-1]
    return "c:" + json.dumps(parent_path, ensure_ascii=False, separators=(",", ":"))


def validate_brand_id(brand_id: str) -> str:
    """Validate canonical KG-v1 Brand identity: 'b:' followed by non-empty brand string.

    Rejects non-strings, leading/trailing whitespace, missing 'b:' prefix, and empty/padded brand strings.
    """
    if not isinstance(brand_id, str):
        raise ValueError(f"Brand ID must be a string, got {type(brand_id).__name__}: {brand_id!r}")
    if brand_id.strip() != brand_id:
        raise ValueError(f"Brand ID must not contain leading or trailing whitespace: {brand_id!r}")
    if not brand_id.startswith("b:"):
        raise ValueError(f"Brand ID must start with 'b:', got {brand_id!r}")

    label = brand_id[2:]
    if not label or not label.strip():
        raise ValueError(f"Brand ID must contain a non-blank label after 'b:', got {brand_id!r}")
    if label.strip() != label:
        raise ValueError(f"Brand label must not contain leading or trailing whitespace: {brand_id!r}")

    return label


def check_cpu_tensor(
    tensor: Any,
    name: str,
    expected_dim: Optional[int] = 64,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Validate that tensor is a finite float32 CPU tensor with expected dimensions.

    Rejects CUDA, MPS, meta, non-float32, non-finite, and shape-mismatched tensors.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.is_meta:
        raise ValueError(f"{name} must not be a meta tensor")
    if tensor.is_cuda:
        raise ValueError(f"{name} must not be a CUDA tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU device, got device {tensor.device}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

    if expected_shape is not None:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
    elif expected_dim is not None:
        if tensor.ndim < 1 or tensor.shape[-1] != expected_dim:
            raise ValueError(f"{name} last dimension mismatch: expected {expected_dim}, got {tuple(tensor.shape)}")

    return tensor


def validate_embedding_row(
    row: Any,
    entity_id: str,
    expected_len: int = 64,
) -> List[float]:
    """Validate that an embedding row is strictly a numeric non-bool sequence of length expected_len with finite floats.

    Rejects booleans, non-numeric elements, wrong length, strings, None, and NaN/Inf.
    """
    if not isinstance(row, (list, tuple)):
        raise ValueError(f"Embedding row for {entity_id!r} must be a list or tuple, got {type(row).__name__}")
    if len(row) != expected_len:
        raise ValueError(f"Embedding row for {entity_id!r} must have length {expected_len}, got {len(row)}")
    row_floats: List[float] = []
    for idx, v in enumerate(row):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"Embedding element for {entity_id!r} at index {idx} must be numeric non-bool, got {type(v).__name__}: {v!r}"
            )
        f_val = float(v)
        if math.isnan(f_val) or math.isinf(f_val):
            raise ValueError(f"Embedding element for {entity_id!r} at index {idx} must be finite, got {f_val}")
        row_floats.append(f_val)
    return row_floats


# -----------------------------------------------------------------------------
# HeteroTrainingGraph
# -----------------------------------------------------------------------------

class HeteroTrainingGraph:
    """Self-contained training-only heterogeneous graph for Amazon4U.

    Nodes:
    - User (from TrainingContext.users)
    - Product (from TrainingContext.sorted_warm_items)
    - CategoryPath (canonical 'c:[...]' JSON prefixes reachable from warm products)
    - Brand (canonical 'b:...' brands reachable from warm products)

    Relations (8 documented directions):
    - User - rated -> Product
    - Product - rated_by -> User
    - Product - belongs_to -> CategoryPath
    - CategoryPath - contains_product -> Product
    - Product - has_brand -> Brand
    - Brand - brand_of -> Product
    - CategoryPath - category_child_of -> CategoryPath
    - CategoryPath - category_parent_of -> CategoryPath

    Invariants:
    - Strictly condition-positive interactions; low-rated observations absent as edges in >=4.
    - Only warm products and metadata reachable from warm products enter the graph.
    - Cold products or metadata for cold products are excluded.
    - Category memberships contract: preserves provided KG-v1 prefix edges in belongs_to;
      derived ancestors and child_of/parent_of edges trace up to root.
    - All edges deduplicated; inverses derived exactly.
    - Delimiter splitting forbidden; JSON prefix derivation preserves labels with ' > '.
    """

    def __init__(
        self,
        context: TrainingContext,
        product_categories: Optional[Dict[str, Sequence[str]]] = None,
        product_brands: Optional[Dict[str, str]] = None,
        category_hierarchy: Optional[Dict[str, str]] = None,
    ) -> None:
        if not isinstance(context, TrainingContext):
            raise ValueError(f"context must be an instance of TrainingContext, got {type(context).__name__}")

        total_pos_edges = sum(len(context.positive_history.get(u, set())) for u in context.users)
        if total_pos_edges == 0:
            raise ValueError("Empty trainable graph: no condition-positive training interactions exist in TrainingContext.")

        self.context = context
        self.node_types = NODE_TYPES

        # Validate category_hierarchy if provided
        if category_hierarchy is not None:
            if not isinstance(category_hierarchy, dict):
                raise ValueError(f"category_hierarchy must be a dict, got {type(category_hierarchy).__name__}")
            for child, parent in category_hierarchy.items():
                if not isinstance(child, str) or not isinstance(parent, str):
                    raise ValueError(
                        f"category_hierarchy keys and values must be strings, got ({type(child).__name__}, {type(parent).__name__})"
                    )
                validate_category_id(child)
                validate_category_id(parent)
                expected_parent = derive_parent_category_id(child)
                if expected_parent is None:
                    raise ValueError(f"Root CategoryPath {child!r} cannot have a parent in category_hierarchy")
                if parent != expected_parent:
                    raise ValueError(
                        f"Invalid category_hierarchy mapping for {child!r}: parent must equal exact immediate canonical prefix {expected_parent!r}, got {parent!r}"
                    )

        # Node lists
        self.users: List[str] = sorted(context.users)
        self.products: List[str] = list(context.sorted_warm_items)

        warm_products_set: Set[str] = set(self.products)

        # 1. Condition-positive interaction edges
        rated_edges: Set[Tuple[str, str]] = set()
        rated_by_edges: Set[Tuple[str, str]] = set()

        for u in self.users:
            pos_items = context.positive_history.get(u, set())
            for item in sorted(pos_items):
                if item in warm_products_set:
                    rated_edges.add((u, item))
                    rated_by_edges.add((item, u))

        # 2. Metadata edges (belongs_to, contains_product, category hierarchy)
        belongs_to_edges: Set[Tuple[str, str]] = set()
        contains_product_edges: Set[Tuple[str, str]] = set()
        child_of_edges: Set[Tuple[str, str]] = set()
        parent_of_edges: Set[Tuple[str, str]] = set()

        categories_set: Set[str] = set()

        if product_categories is not None:
            if not isinstance(product_categories, dict):
                raise ValueError(f"product_categories must be a dict, got {type(product_categories).__name__}")

            # Validate and restrict to warm products only
            for p, cat_seq in product_categories.items():
                if not isinstance(p, str):
                    raise ValueError(f"Product ID in product_categories must be string, got {type(p).__name__}")
                if p not in warm_products_set:
                    # Exclude cold products
                    continue
                if not isinstance(cat_seq, (list, tuple, set)):
                    raise ValueError(f"Categories for product {p!r} must be a sequence, got {type(cat_seq).__name__}")

                for cat_id in cat_seq:
                    validate_category_id(cat_id)
                    belongs_to_edges.add((p, cat_id))
                    contains_product_edges.add((cat_id, p))
                    categories_set.add(cat_id)

                    # Trace hierarchy ancestors for warm-product-reachable categories
                    curr_cat = cat_id
                    visited_ancestors: Set[str] = {curr_cat}
                    while True:
                        parent_cat = derive_parent_category_id(curr_cat)
                        if parent_cat is None:
                            break
                        if parent_cat in visited_ancestors:
                            raise ValueError(f"Cycle detected in category hierarchy for {curr_cat!r}")
                        visited_ancestors.add(parent_cat)

                        child_of_edges.add((curr_cat, parent_cat))
                        parent_of_edges.add((parent_cat, curr_cat))
                        categories_set.add(parent_cat)
                        curr_cat = parent_cat

        # 3. Metadata edges (has_brand, brand_of)
        has_brand_edges: Set[Tuple[str, str]] = set()
        brand_of_edges: Set[Tuple[str, str]] = set()

        brands_set: Set[str] = set()

        if product_brands is not None:
            if not isinstance(product_brands, dict):
                raise ValueError(f"product_brands must be a dict, got {type(product_brands).__name__}")

            for p, brand_id in product_brands.items():
                if not isinstance(p, str):
                    raise ValueError(f"Product ID in product_brands must be string, got {type(p).__name__}")
                if p not in warm_products_set:
                    # Exclude cold products
                    continue
                validate_brand_id(brand_id)
                has_brand_edges.add((p, brand_id))
                brand_of_edges.add((brand_id, p))
                brands_set.add(brand_id)

        self.categories: List[str] = sorted(categories_set)
        self.brands: List[str] = sorted(brands_set)

        self.node_ids: Dict[str, List[str]] = {
            "User": self.users,
            "Product": self.products,
            "CategoryPath": self.categories,
            "Brand": self.brands,
        }

        self.node_to_idx: Dict[str, Dict[str, int]] = {
            ntype: {nid: i for i, nid in enumerate(nlist)}
            for ntype, nlist in self.node_ids.items()
        }

        # Store edges
        self.edges: Dict[str, Set[Tuple[str, str]]] = {
            "rated": rated_edges,
            "rated_by": rated_by_edges,
            "belongs_to": belongs_to_edges,
            "contains_product": contains_product_edges,
            "has_brand": has_brand_edges,
            "brand_of": brand_of_edges,
            "category_child_of": child_of_edges,
            "category_parent_of": parent_of_edges,
        }

        # Precompute sorted incoming neighbors per destination node:
        # incoming_neighbors[rel][dst_id] = [src_id_1, src_id_2, ...]
        self.incoming_neighbors: Dict[str, Dict[str, List[str]]] = {}
        for rel, edge_set in self.edges.items():
            incoming_map: Dict[str, List[str]] = defaultdict(list)
            for src, dst in edge_set:
                incoming_map[dst].append(src)
            # Sort for determinism
            self.incoming_neighbors[rel] = {
                dst: sorted(srcs) for dst, srcs in incoming_map.items()
            }


def build_hetero_graph(
    context: TrainingContext,
    product_categories: Optional[Dict[str, Sequence[str]]] = None,
    product_brands: Optional[Dict[str, str]] = None,
    category_hierarchy: Optional[Dict[str, str]] = None,
) -> HeteroTrainingGraph:
    """Build a validated HeteroTrainingGraph from TrainingContext and warm metadata."""
    return HeteroTrainingGraph(
        context,
        product_categories=product_categories,
        product_brands=product_brands,
        category_hierarchy=category_hierarchy,
    )


# -----------------------------------------------------------------------------
# Differentiable Heterogeneous KG + GNN Model
# -----------------------------------------------------------------------------

class DifferentiableHeteroGNN(nn.Module):
    """Two-layer 64d transductive heterogeneous GraphSAGE-style recommender.

    Specification (docs/heterogeneous-encoder.md):
    - 4 node types: User, Product, CategoryPath, Brand.
    - Each node type has 64d learned ID embeddings initialized with Normal(0, std=0.1)
      using an explicit integer seed and local deterministic torch.Generator.
    - 2 message-passing layers (l=0, 1).
    - 8 directed relations, each with independent 64x64 W_r matrix per layer (Xavier uniform gain 1).
    - 4 node types, each with independent 64x64 W_self matrix per layer (Xavier uniform gain 1)
      and length-64 bias b_self per layer (zeros).
    - Message passing operator:
      Within each active relation: m_(v,r)^(l) = W_r^(l) * mean(h_w^(l) for w in N_r(v))
      Across relations and self:
      z_v^(l+1) = W_self,t(v)^(l) * h_v^(l) + b_t(v)^(l) + mean(m_(v,r)^(l) for r in R_v)
      (If R_v is empty, neighbor contribution is zero).
    - Activations: h^(1) = ReLU(z^(1)), h^(2) = z^(2) (identity).
    - Raw dot product decoder: s(u, i) = dot(h_u^(2), h_i^(2)).
    - Batch target masking: on-the-fly masking of supervised positive edges in both
      forward and reverse directions for both layers, with active relations recomputed.
    """

    def __init__(
        self,
        graph: HeteroTrainingGraph,
        init_seed: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if not isinstance(graph, HeteroTrainingGraph):
            raise ValueError(f"graph must be a HeteroTrainingGraph, got {type(graph).__name__}")
        if isinstance(init_seed, bool) or not isinstance(init_seed, int):
            raise ValueError(f"init_seed must be an explicit integer, got {init_seed!r}")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim != 64:
            raise ValueError(f"embedding_dim must be exactly 64, got {embedding_dim!r}")

        self.graph = graph
        self.init_seed = init_seed
        self.embedding_dim = embedding_dim

        # Local deterministic generator (CPU only, does not alter global RNG)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(init_seed)

        # 1. Base ID embeddings for each node type
        self.user_embeddings = nn.Parameter(
            torch.normal(0.0, 0.1, size=(len(graph.users), embedding_dim), generator=gen)
        )
        self.product_embeddings = nn.Parameter(
            torch.normal(0.0, 0.1, size=(len(graph.products), embedding_dim), generator=gen)
        )
        self.category_embeddings = nn.Parameter(
            torch.normal(0.0, 0.1, size=(len(graph.categories), embedding_dim), generator=gen)
        )
        self.brand_embeddings = nn.Parameter(
            torch.normal(0.0, 0.1, size=(len(graph.brands), embedding_dim), generator=gen)
        )

        # 2. Relation transformation matrices (8 relations x 2 layers)
        self.relation_weights_l0 = nn.ParameterDict()
        self.relation_weights_l1 = nn.ParameterDict()

        for rel in RELATION_SCHEMAS:
            w0 = torch.empty(embedding_dim, embedding_dim, dtype=torch.float32)
            torch.nn.init.xavier_uniform_(w0, gain=1.0, generator=gen)
            self.relation_weights_l0[rel] = nn.Parameter(w0)

            w1 = torch.empty(embedding_dim, embedding_dim, dtype=torch.float32)
            torch.nn.init.xavier_uniform_(w1, gain=1.0, generator=gen)
            self.relation_weights_l1[rel] = nn.Parameter(w1)

        # 3. Node-type self transformation matrices and biases (4 node types x 2 layers)
        self.self_weights_l0 = nn.ParameterDict()
        self.self_biases_l0 = nn.ParameterDict()
        self.self_weights_l1 = nn.ParameterDict()
        self.self_biases_l1 = nn.ParameterDict()

        for ntype in NODE_TYPES:
            sw0 = torch.empty(embedding_dim, embedding_dim, dtype=torch.float32)
            torch.nn.init.xavier_uniform_(sw0, gain=1.0, generator=gen)
            self.self_weights_l0[ntype] = nn.Parameter(sw0)
            self.self_biases_l0[ntype] = nn.Parameter(torch.zeros(embedding_dim, dtype=torch.float32))

            sw1 = torch.empty(embedding_dim, embedding_dim, dtype=torch.float32)
            torch.nn.init.xavier_uniform_(sw1, gain=1.0, generator=gen)
            self.self_weights_l1[ntype] = nn.Parameter(sw1)
            self.self_biases_l1[ntype] = nn.Parameter(torch.zeros(embedding_dim, dtype=torch.float32))

    def _validate_all_parameters(self) -> None:
        """Validate expected full shapes, CPU device, float32 dtype, and finiteness of EVERY parameter."""
        check_cpu_tensor(
            self.user_embeddings,
            "user_embeddings",
            expected_shape=(len(self.graph.users), self.embedding_dim),
        )
        check_cpu_tensor(
            self.product_embeddings,
            "product_embeddings",
            expected_shape=(len(self.graph.products), self.embedding_dim),
        )
        check_cpu_tensor(
            self.category_embeddings,
            "category_embeddings",
            expected_shape=(len(self.graph.categories), self.embedding_dim),
        )
        check_cpu_tensor(
            self.brand_embeddings,
            "brand_embeddings",
            expected_shape=(len(self.graph.brands), self.embedding_dim),
        )

        for rel in RELATION_SCHEMAS:
            if rel not in self.relation_weights_l0 or rel not in self.relation_weights_l1:
                raise ValueError(f"Missing relation weight for {rel}")
            check_cpu_tensor(
                self.relation_weights_l0[rel],
                f"relation_weights_l0[{rel}]",
                expected_shape=(self.embedding_dim, self.embedding_dim),
            )
            check_cpu_tensor(
                self.relation_weights_l1[rel],
                f"relation_weights_l1[{rel}]",
                expected_shape=(self.embedding_dim, self.embedding_dim),
            )

        for ntype in NODE_TYPES:
            if (
                ntype not in self.self_weights_l0
                or ntype not in self.self_biases_l0
                or ntype not in self.self_weights_l1
                or ntype not in self.self_biases_l1
            ):
                raise ValueError(f"Missing self weight or bias for {ntype}")
            check_cpu_tensor(
                self.self_weights_l0[ntype],
                f"self_weights_l0[{ntype}]",
                expected_shape=(self.embedding_dim, self.embedding_dim),
            )
            check_cpu_tensor(
                self.self_biases_l0[ntype],
                f"self_biases_l0[{ntype}]",
                expected_shape=(self.embedding_dim,),
            )
            check_cpu_tensor(
                self.self_weights_l1[ntype],
                f"self_weights_l1[{ntype}]",
                expected_shape=(self.embedding_dim, self.embedding_dim),
            )
            check_cpu_tensor(
                self.self_biases_l1[ntype],
                f"self_biases_l1[{ntype}]",
                expected_shape=(self.embedding_dim,),
            )

    def _validate_mask_pairs(self, mask_pairs: Optional[Set[Tuple[str, str]]]) -> Optional[Set[Tuple[str, str]]]:
        if mask_pairs is None:
            return None
        if isinstance(mask_pairs, (str, bytes)):
            raise ValueError("mask_pairs must be a set or collection of pairs, not a string")
        if not isinstance(mask_pairs, (set, frozenset, list, tuple)):
            raise ValueError(f"mask_pairs must be a set or sequence of (user_id, item_id) tuples, got {type(mask_pairs).__name__}")
        if len(mask_pairs) == 0:
            raise ValueError("mask_pairs cannot be empty; pass None for unmasked forward pass")

        validated: Set[Tuple[str, str]] = set()
        for pair in mask_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"Each element in mask_pairs must be a 2-tuple (user_id, item_id), got {pair!r}")
            u, i = pair
            if not isinstance(u, str) or not isinstance(i, str):
                raise ValueError(f"Mask pair entries must be strings, got ({type(u).__name__}, {type(i).__name__})")
            if u not in self.graph.node_to_idx["User"]:
                raise ValueError(f"Unknown user ID in mask_pairs: {u!r}")
            if i not in self.graph.node_to_idx["Product"]:
                raise ValueError(f"Unknown product ID in mask_pairs: {i!r}")

            pos_history = self.graph.context.positive_history.get(u, set())
            if i not in pos_history:
                raise ValueError(
                    f"Mask pair ({u!r}, {i!r}) is not a genuine condition-positive interaction in TrainingContext"
                )
            validated.add((u, i))

        return validated

    def forward(
        self,
        mask_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute two-layer forward propagation with optional batch target-edge masking.

        Returns a dictionary mapping node type to propagated (N_T, 64) tensor representations.
        All operations are CPU float32 and preserve autograd gradients.
        """
        self._validate_all_parameters()
        valid_masks = self._validate_mask_pairs(mask_pairs)

        current_h: Dict[str, torch.Tensor] = {
            "User": self.user_embeddings,
            "Product": self.product_embeddings,
            "CategoryPath": self.category_embeddings,
            "Brand": self.brand_embeddings,
        }

        layer_configs = [
            (self.relation_weights_l0, self.self_weights_l0, self.self_biases_l0, True),  # Layer 0 -> ReLU
            (self.relation_weights_l1, self.self_weights_l1, self.self_biases_l1, False),  # Layer 1 -> Identity
        ]

        for layer_idx, (rel_weights, self_weights, self_biases, apply_relu) in enumerate(layer_configs):
            next_h: Dict[str, torch.Tensor] = {}

            for ntype in NODE_TYPES:
                nlist = self.graph.node_ids[ntype]
                num_nodes = len(nlist)
                if num_nodes == 0:
                    next_h[ntype] = torch.empty((0, self.embedding_dim), dtype=torch.float32)
                    continue

                w_self = self_weights[ntype]
                b_self = self_biases[ntype]
                h_ntype = current_h[ntype]

                # Self term for all nodes of this type: W_self * h + b
                z_self_all = F.linear(h_ntype, w_self, b_self)

                # Incoming relations with destination type ntype
                incoming_rels = [
                    rel for rel, (src_t, dst_t) in RELATION_SCHEMAS.items()
                    if dst_t == ntype
                ]

                node_z_list: List[torch.Tensor] = []

                for j, node_id in enumerate(nlist):
                    active_messages: List[torch.Tensor] = []

                    for rel in incoming_rels:
                        src_type = RELATION_SCHEMAS[rel][0]
                        src_nbrs = self.graph.incoming_neighbors[rel].get(node_id, [])

                        if valid_masks:
                            if rel == "rated":
                                # src is User, dst is Product node_id
                                src_nbrs = [u for u in src_nbrs if (u, node_id) not in valid_masks]
                            elif rel == "rated_by":
                                # src is Product, dst is User node_id
                                src_nbrs = [p for p in src_nbrs if (node_id, p) not in valid_masks]

                        if src_nbrs:
                            src_indices = [self.graph.node_to_idx[src_type][s] for s in src_nbrs]
                            h_src_nbrs = current_h[src_type][src_indices]
                            mean_h = h_src_nbrs.mean(dim=0)
                            m_rel = F.linear(mean_h, rel_weights[rel])
                            active_messages.append(m_rel)

                    if active_messages:
                        neighbor_term = torch.stack(active_messages).mean(dim=0)
                    else:
                        neighbor_term = torch.zeros(self.embedding_dim, dtype=torch.float32)

                    z_total = z_self_all[j] + neighbor_term
                    node_z_list.append(z_total)

                z_stacked = torch.stack(node_z_list)
                if apply_relu:
                    next_h[ntype] = F.relu(z_stacked)
                else:
                    next_h[ntype] = z_stacked

            current_h = next_h

        # Validate outputs
        for ntype in NODE_TYPES:
            check_cpu_tensor(
                current_h[ntype],
                f"forward representation for {ntype}",
                expected_shape=(len(self.graph.node_ids[ntype]), self.embedding_dim),
            )

        return current_h

    def compute_batch_scores(
        self,
        user_ids: Sequence[str],
        positive_item_ids: Sequence[str],
        negative_item_ids: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positive and negative scores from the same batch-masked forward pass.

        Validates all IDs, positive eligibility, negative eligibility, and non-empty batch.
        """
        if isinstance(user_ids, (str, bytes)) or isinstance(positive_item_ids, (str, bytes)) or isinstance(negative_item_ids, (str, bytes)):
            raise ValueError("Batch inputs must be sequences of ID strings, not a single string")
        if len(user_ids) != len(positive_item_ids) or len(user_ids) != len(negative_item_ids):
            raise ValueError(
                f"Batch lengths mismatch: {len(user_ids)} users, {len(positive_item_ids)} pos items, "
                f"{len(negative_item_ids)} neg items"
            )
        if len(user_ids) == 0:
            raise ValueError("Cannot score an empty batch: no supervised examples provided")

        mask_pairs: Set[Tuple[str, str]] = set()

        for u, pos_item, neg_item in zip(user_ids, positive_item_ids, negative_item_ids):
            if not isinstance(u, str) or not isinstance(pos_item, str) or not isinstance(neg_item, str):
                raise ValueError("Batch IDs must be strings")
            if u not in self.graph.node_to_idx["User"]:
                raise ValueError(f"Unknown user {u!r} not in graph")
            if pos_item not in self.graph.node_to_idx["Product"]:
                raise ValueError(f"Unknown positive item {pos_item!r} not in warm products")
            if neg_item not in self.graph.node_to_idx["Product"]:
                raise ValueError(f"Unknown negative item {neg_item!r} not in warm products")

            # Positive eligibility: must be in user's condition-positive training history
            pos_history = self.graph.context.positive_history.get(u, set())
            if pos_item not in pos_history:
                raise ValueError(
                    f"Item {pos_item!r} is not a condition-positive training interaction for user {u!r}"
                )

            # Negative eligibility: N(u) = W_condition minus O(u)
            obs_history = self.graph.context.all_rated_history.get(u, set())
            if neg_item in obs_history:
                raise ValueError(
                    f"Negative item {neg_item!r} is in user {u!r}'s training observed history (forbidden negative)"
                )

            mask_pairs.add((u, pos_item))

        # Single fresh forward pass with masked target edges
        embs = self.forward(mask_pairs=mask_pairs)

        u_indices = [self.graph.node_to_idx["User"][u] for u in user_ids]
        pos_indices = [self.graph.node_to_idx["Product"][i] for i in positive_item_ids]
        neg_indices = [self.graph.node_to_idx["Product"][i] for i in negative_item_ids]

        u_tensor = embs["User"][u_indices]
        pos_tensor = embs["Product"][pos_indices]
        neg_tensor = embs["Product"][neg_indices]

        # Raw dot product decoding
        s_pos = (u_tensor * pos_tensor).sum(dim=-1)
        s_neg = (u_tensor * neg_tensor).sum(dim=-1)

        check_cpu_tensor(s_pos, "s_pos", expected_shape=(len(user_ids),))
        check_cpu_tensor(s_neg, "s_neg", expected_shape=(len(user_ids),))

        return s_pos, s_neg

    def compute_batch_loss(
        self,
        user_ids: Sequence[str],
        positive_item_ids: Sequence[str],
        negative_item_ids: Sequence[str],
        reg_coefficient: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        """Compute BPR ranking loss and explicit endpoint-only base-ID regularization.

        Formula:
          L_ranking = mean(softplus(s_neg - s_pos))
          L_reg = (1e-5 / (|U_B| + |I_B|)) * (sum(||e_u^(0)||^2) + sum(||e_i^(0)||^2))
        """
        if isinstance(reg_coefficient, bool) or not isinstance(reg_coefficient, (int, float)) or reg_coefficient != 1e-5:
            raise ValueError(f"Endpoint regularization coefficient must be exactly 1e-5, got {reg_coefficient!r}")

        s_pos, s_neg = self.compute_batch_scores(user_ids, positive_item_ids, negative_item_ids)

        # 1. Ranking loss via softplus
        diff = s_neg - s_pos
        if not torch.isfinite(diff).all():
            raise ValueError("Score differences s_neg - s_pos contain non-finite values (overflow)")
        ranking_loss = F.softplus(diff).mean()

        # 2. Endpoint regularization on unique base ID rows
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(positive_item_ids) | set(negative_item_ids))
        n_endpoints = len(unique_users) + len(unique_items)

        u_base_indices = [self.graph.node_to_idx["User"][u] for u in unique_users]
        i_base_indices = [self.graph.node_to_idx["Product"][i] for i in unique_items]

        u_base_rows = self.user_embeddings[u_base_indices]
        i_base_rows = self.product_embeddings[i_base_indices]

        sq_u_norms = (u_base_rows ** 2).sum()
        sq_i_norms = (i_base_rows ** 2).sum()
        if not torch.isfinite(sq_u_norms) or not torch.isfinite(sq_i_norms):
            raise ValueError("Squared base norm overflow in endpoint regularization")

        reg_loss = (float(reg_coefficient) / float(n_endpoints)) * (sq_u_norms + sq_i_norms)
        total_loss = ranking_loss + reg_loss

        check_cpu_tensor(ranking_loss, "ranking_loss", expected_shape=())
        check_cpu_tensor(reg_loss, "reg_loss", expected_shape=())
        check_cpu_tensor(total_loss, "total_loss", expected_shape=())

        return {
            "total_loss": total_loss,
            "ranking_loss": ranking_loss,
            "reg_loss": reg_loss,
        }

    def score_candidates(
        self,
        user_id: str,
        candidate_items: Sequence[str],
    ) -> Tuple[Dict[str, float], str, Optional[str]]:
        """Evaluator-compatible candidate scoring using unmasked full-graph representation.

        Empty-positive-history users route to condition-positive training popularity.
        """
        if user_id not in self.graph.node_to_idx["User"]:
            raise ScoringFailureError(f"User {user_id!r} not in training graph")

        pos_history = self.graph.context.positive_history.get(user_id, set())
        if not pos_history:
            scores = {item: float(self.graph.context.popularity.get(item, 0)) for item in candidate_items}
            return scores, "fallback", "empty_positive_history"

        # Propagate unmasked graph for evaluation
        embs = self.forward(mask_pairs=None)

        u_idx = self.graph.node_to_idx["User"][user_id]
        u_vec = embs["User"][u_idx]

        scores: Dict[str, float] = {}
        for item in candidate_items:
            if item not in self.graph.node_to_idx["Product"]:
                raise ScoringFailureError(f"Candidate item {item!r} not in warm product graph")
            i_idx = self.graph.node_to_idx["Product"][item]
            i_vec = embs["Product"][i_idx]
            val = float(torch.dot(u_vec, i_vec).item())
            if math.isnan(val) or math.isinf(val):
                raise ScoringFailureError(f"Non-finite dot-product score for user {user_id!r}, item {item!r}: {val}")
            scores[item] = val

        return scores, "personalized", None


# -----------------------------------------------------------------------------
# Differentiable LightGCN Model (Option B)
# -----------------------------------------------------------------------------

class DifferentiableLightGCN(nn.Module):
    """PyTorch implementation of core LightGCN with Option B fixed-graph training-edge reuse.

    Specification (docs/lightgcn-protocol.md):
    - 64d base user and item embeddings supplied explicitly by caller.
    - 3 propagation layers over the bipartite user-item condition-positive training graph.
    - Standard symmetric degree normalization: D^(-1/2) A D^(-1/2).
    - Arithmetic mean of layers 0..3: H = (1/4) * sum_{l=0}^3 H^(l).
    - Raw dot-product scoring; no transforms, biases, extra self-loops, or batch masks.
    """

    def __init__(
        self,
        graph: LightGCNGraph,
        user_base_embeddings: Union[torch.Tensor, Dict[str, Sequence[float]]],
        item_base_embeddings: Union[torch.Tensor, Dict[str, Sequence[float]]],
        config: Optional[LightGCNConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(graph, LightGCNGraph):
            raise ValueError(f"graph must be an instance of LightGCNGraph, got {type(graph).__name__}")

        self.graph = graph
        if config is not None:
            if config != graph.config:
                raise ValueError(
                    f"Provided config {config} must match graph.config {graph.config}"
                )
        self.config = graph.config
        if self.config.embedding_dimension != 64:
            raise ValueError(f"LightGCN config embedding_dimension must be 64, got {self.config.embedding_dimension}")
        if self.config.propagation_layers != 3:
            raise ValueError(f"LightGCN config propagation_layers must be 3, got {self.config.propagation_layers}")

        self.embedding_dim = self.config.embedding_dimension

        # Caller MUST explicitly supply user and item embeddings
        if user_base_embeddings is None or item_base_embeddings is None:
            raise ValueError("user_base_embeddings and item_base_embeddings must be explicitly supplied by caller")

        # Process user base embeddings (detach and clone to prevent external alias modification)
        if isinstance(user_base_embeddings, dict):
            expected_users = set(graph.users)
            provided_users = set(user_base_embeddings.keys())
            if expected_users != provided_users:
                missing = expected_users - provided_users
                extra = provided_users - expected_users
                if missing:
                    raise MissingIdError(f"Missing base embedding for graph user(s): {sorted(missing)!r}")
                if extra:
                    raise UnknownIdError(f"Unknown or extra user ID(s) in user_base_embeddings: {sorted(extra)!r}")

            u_list = [validate_embedding_row(user_base_embeddings[u], u, self.embedding_dim) for u in graph.users]
            u_tensor = torch.tensor(u_list, dtype=torch.float32)
        elif isinstance(user_base_embeddings, torch.Tensor):
            u_tensor = user_base_embeddings.detach().clone()
        else:
            raise ValueError(f"user_base_embeddings must be a torch.Tensor or dict, got {type(user_base_embeddings).__name__}")

        check_cpu_tensor(u_tensor, "user_base_embeddings", expected_shape=(len(graph.users), self.embedding_dim))
        self.user_embeddings = nn.Parameter(u_tensor)

        # Process item base embeddings (detach and clone to prevent external alias modification)
        if isinstance(item_base_embeddings, dict):
            expected_items = set(graph.warm_items)
            provided_items = set(item_base_embeddings.keys())
            if expected_items != provided_items:
                missing = expected_items - provided_items
                extra = provided_items - expected_items
                if missing:
                    raise MissingIdError(f"Missing base embedding for graph item(s): {sorted(missing)!r}")
                if extra:
                    raise UnknownIdError(f"Unknown or extra item ID(s) in item_base_embeddings: {sorted(extra)!r}")

            i_list = [validate_embedding_row(item_base_embeddings[item], item, self.embedding_dim) for item in graph.warm_items]
            i_tensor = torch.tensor(i_list, dtype=torch.float32)
        elif isinstance(item_base_embeddings, torch.Tensor):
            i_tensor = item_base_embeddings.detach().clone()
        else:
            raise ValueError(f"item_base_embeddings must be a torch.Tensor or dict, got {type(item_base_embeddings).__name__}")

        check_cpu_tensor(i_tensor, "item_base_embeddings", expected_shape=(len(graph.warm_items), self.embedding_dim))
        self.item_embeddings = nn.Parameter(i_tensor)

        # Precompute symmetric normalized adjacency matrix A (users x items)
        u_map = {u: idx for idx, u in enumerate(graph.users)}
        i_map = {item: idx for idx, item in enumerate(graph.warm_items)}

        row_indices: List[int] = []
        col_indices: List[int] = []
        values: List[float] = []

        for u, nbrs in graph.user_neighbors.items():
            u_idx = u_map[u]
            for item, weight in nbrs:
                row_indices.append(u_idx)
                col_indices.append(i_map[item])
                values.append(float(weight))

        if values:
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
            vals = torch.tensor(values, dtype=torch.float32)
            check_cpu_tensor(vals, "sparse adjacency values", expected_dim=None)
            adj_coo = torch.sparse_coo_tensor(
                indices, vals, (len(graph.users), len(graph.warm_items))
            ).coalesce()
        else:
            adj_coo = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                (len(graph.users), len(graph.warm_items)),
            ).coalesce()

        self.register_buffer("adj_user_item", adj_coo)

    def _validate_adj_user_item(self) -> torch.Tensor:
        """Validate layout, device, dtype, shape, indices, and finiteness of adj_user_item."""
        adj = getattr(self, "adj_user_item", None)
        if not isinstance(adj, torch.Tensor):
            raise ValueError(f"adj_user_item must be a torch.Tensor, got {type(adj).__name__}")
        if adj.is_meta:
            raise ValueError("adj_user_item must not be a meta tensor")
        if adj.is_cuda:
            raise ValueError("adj_user_item must not be a CUDA tensor")
        if adj.device.type != "cpu":
            raise ValueError(f"adj_user_item must be on CPU device, got {adj.device}")
        if adj.dtype != torch.float32:
            raise ValueError(f"adj_user_item must have dtype torch.float32, got {adj.dtype}")
        if adj.layout != torch.sparse_coo:
            raise ValueError(f"adj_user_item must have layout torch.sparse_coo, got {adj.layout}")
        expected_shape = (len(self.graph.users), len(self.graph.warm_items))
        if tuple(adj.shape) != expected_shape:
            raise ValueError(f"adj_user_item shape mismatch: expected {expected_shape}, got {tuple(adj.shape)}")

        adj_coalesced = adj.coalesce()
        values = adj_coalesced.values()
        if not torch.isfinite(values).all():
            raise ValueError("adj_user_item contains non-finite values (NaN or Inf)")

        indices = adj_coalesced.indices()
        if indices.numel() > 0:
            if (indices[0] < 0).any() or (indices[0] >= len(self.graph.users)).any():
                raise ValueError("adj_user_item row indices out of bounds for users")
            if (indices[1] < 0).any() or (indices[1] >= len(self.graph.warm_items)).any():
                raise ValueError("adj_user_item col indices out of bounds for warm items")

        return adj_coalesced

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact full-graph LightGCN propagation over 3 layers with layer 0..3 averaging.

        Returns (propagated_users, propagated_items).
        """
        if self.config != self.graph.config:
            raise ValueError(f"LightGCN config {self.config} does not match graph.config {self.graph.config}")
        if self.config.embedding_dimension != 64:
            raise ValueError(f"LightGCN config embedding_dimension must be 64, got {self.config.embedding_dimension}")
        if self.config.propagation_layers != 3:
            raise ValueError(f"LightGCN config propagation_layers must be 3, got {self.config.propagation_layers}")

        check_cpu_tensor(
            self.user_embeddings,
            "user_embeddings",
            expected_shape=(len(self.graph.users), self.embedding_dim),
        )
        check_cpu_tensor(
            self.item_embeddings,
            "item_embeddings",
            expected_shape=(len(self.graph.warm_items), self.embedding_dim),
        )

        adj = self._validate_adj_user_item()
        adj_t = adj.t()

        u_layers: List[torch.Tensor] = [self.user_embeddings]
        i_layers: List[torch.Tensor] = [self.item_embeddings]

        for _ in range(self.config.propagation_layers):
            next_u = torch.sparse.mm(adj, i_layers[-1])
            next_i = torch.sparse.mm(adj_t, u_layers[-1])
            u_layers.append(next_u)
            i_layers.append(next_i)

        prop_u = torch.stack(u_layers).mean(dim=0)
        prop_i = torch.stack(i_layers).mean(dim=0)

        check_cpu_tensor(prop_u, "propagated_users", expected_shape=(len(self.graph.users), self.embedding_dim))
        check_cpu_tensor(prop_i, "propagated_items", expected_shape=(len(self.graph.warm_items), self.embedding_dim))

        return prop_u, prop_i

    def compute_batch_scores(
        self,
        user_ids: Sequence[str],
        positive_item_ids: Sequence[str],
        negative_item_ids: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute batch scores using Option B fixed-graph propagated embeddings."""
        if isinstance(user_ids, (str, bytes)) or isinstance(positive_item_ids, (str, bytes)) or isinstance(negative_item_ids, (str, bytes)):
            raise ValueError("Batch inputs must be sequences of ID strings, not a single string")
        if len(user_ids) != len(positive_item_ids) or len(user_ids) != len(negative_item_ids):
            raise ValueError("Batch sequences must have identical lengths")
        if len(user_ids) == 0:
            raise ValueError("Cannot score an empty batch")

        u_map = {u: idx for idx, u in enumerate(self.graph.users)}
        i_map = {item: idx for idx, item in enumerate(self.graph.warm_items)}

        for u, pos_item, neg_item in zip(user_ids, positive_item_ids, negative_item_ids):
            if not isinstance(u, str) or not isinstance(pos_item, str) or not isinstance(neg_item, str):
                raise ValueError("Batch IDs must be strings")
            if u not in u_map:
                raise ValueError(f"Unknown user {u!r} not in LightGCN graph")
            if pos_item not in i_map:
                raise ValueError(f"Unknown positive item {pos_item!r} not in LightGCN warm items")
            if neg_item not in i_map:
                raise ValueError(f"Unknown negative item {neg_item!r} not in LightGCN warm items")

            pos_history = self.graph.context.positive_history.get(u, set())
            if pos_item not in pos_history:
                raise ValueError(f"Positive item {pos_item!r} is not in positive history for user {u!r}")

            obs_history = self.graph.context.all_rated_history.get(u, set())
            if neg_item in obs_history:
                raise ValueError(f"Negative item {neg_item!r} is in observed history for user {u!r}")

        prop_u, prop_i = self.forward()

        u_indices = [u_map[u] for u in user_ids]
        pos_indices = [i_map[i] for i in positive_item_ids]
        neg_indices = [i_map[i] for i in negative_item_ids]

        u_tensor = prop_u[u_indices]
        pos_tensor = prop_i[pos_indices]
        neg_tensor = prop_i[neg_indices]

        s_pos = (u_tensor * pos_tensor).sum(dim=-1)
        s_neg = (u_tensor * neg_tensor).sum(dim=-1)

        check_cpu_tensor(s_pos, "s_pos", expected_shape=(len(user_ids),))
        check_cpu_tensor(s_neg, "s_neg", expected_shape=(len(user_ids),))

        return s_pos, s_neg

    def compute_batch_loss(
        self,
        user_ids: Sequence[str],
        positive_item_ids: Sequence[str],
        negative_item_ids: Sequence[str],
        reg_coefficient: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        """Compute BPR ranking loss and explicit endpoint-only base-ID regularization."""
        if isinstance(reg_coefficient, bool) or not isinstance(reg_coefficient, (int, float)) or reg_coefficient != 1e-5:
            raise ValueError(f"Endpoint regularization coefficient must be exactly 1e-5, got {reg_coefficient!r}")

        s_pos, s_neg = self.compute_batch_scores(user_ids, positive_item_ids, negative_item_ids)

        diff = s_neg - s_pos
        if not torch.isfinite(diff).all():
            raise ValueError("Score differences s_neg - s_pos contain non-finite values (overflow)")
        ranking_loss = F.softplus(diff).mean()

        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(positive_item_ids) | set(negative_item_ids))
        n_endpoints = len(unique_users) + len(unique_items)

        u_map = {u: idx for idx, u in enumerate(self.graph.users)}
        i_map = {item: idx for idx, item in enumerate(self.graph.warm_items)}

        u_base_indices = [u_map[u] for u in unique_users]
        i_base_indices = [i_map[i] for i in unique_items]

        u_base_rows = self.user_embeddings[u_base_indices]
        i_base_rows = self.item_embeddings[i_base_indices]

        sq_u_norms = (u_base_rows ** 2).sum()
        sq_i_norms = (i_base_rows ** 2).sum()
        if not torch.isfinite(sq_u_norms) or not torch.isfinite(sq_i_norms):
            raise ValueError("Squared base norm overflow in endpoint regularization")

        reg_loss = (float(reg_coefficient) / float(n_endpoints)) * (sq_u_norms + sq_i_norms)
        total_loss = ranking_loss + reg_loss

        check_cpu_tensor(ranking_loss, "ranking_loss", expected_shape=())
        check_cpu_tensor(reg_loss, "reg_loss", expected_shape=())
        check_cpu_tensor(total_loss, "total_loss", expected_shape=())

        return {
            "total_loss": total_loss,
            "ranking_loss": ranking_loss,
            "reg_loss": reg_loss,
        }

    def score_candidates(
        self,
        user_id: str,
        candidate_items: Sequence[str],
    ) -> Tuple[Dict[str, float], str, Optional[str]]:
        """Evaluator-compatible candidate scoring using propagated LightGCN representations."""
        u_map = {u: idx for idx, u in enumerate(self.graph.users)}
        if user_id not in u_map:
            raise ScoringFailureError(f"User {user_id!r} not in LightGCN graph")

        pos_history = self.graph.context.positive_history.get(user_id, set())
        if not pos_history:
            scores = {item: float(self.graph.context.popularity.get(item, 0)) for item in candidate_items}
            return scores, "fallback", "empty_positive_history"

        prop_u, prop_i = self.forward()
        u_vec = prop_u[u_map[user_id]]

        i_map = {item: idx for idx, item in enumerate(self.graph.warm_items)}
        scores: Dict[str, float] = {}

        for item in candidate_items:
            if item not in i_map:
                raise ScoringFailureError(f"Candidate item {item!r} not in LightGCN warm items")
            i_vec = prop_i[i_map[item]]
            val = float(torch.dot(u_vec, i_vec).item())
            if math.isnan(val) or math.isinf(val):
                raise ScoringFailureError(f"Non-finite dot-product score for user {user_id!r}, item {item!r}: {val}")
            scores[item] = val

        return scores, "personalized", None
