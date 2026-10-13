"""Amazon4U Step 3: learned-model training harness foundation, LightGCN, and Heterogeneous KG+GNN reference implementations.

Python standard library only.
Provides:
- LightGCNConfig: validated configuration for LightGCN architecture and Option B semantics.
- LightGCNGraph: fixed condition-positive training graph construction from TrainingContext,
  with symmetric degree normalization, reverse edges, and strict held-out exclusion.
- propagate_lightgcn: deterministic exact propagation over L layers with layer 0..L averaging.
- LightGCNScorer: deterministic candidate dot-product scoring with Evaluator compatibility
  and empty-positive-history popularity fallback.
- HeterogeneousKGNNConfig: validated configuration for Heterogeneous KG+GNN architecture.
- HeterogeneousKGNNGraph: typed heterogeneous graph with 4 node types, 8 directed relations,
  strict held-out exclusion, and validation against corrupt/duplicate structures.
- build_heterogeneous_graph: constructor from TrainingContext and metadata.
- mask_heterogeneous_batch_edges: batch positive-edge masking helper excluding direct forward/reverse edges.
- HeterogeneousParameters & initialize_heterogeneous_parameters: typed layer matrices and self biases with Xavier uniform init.
- initialize_node_embeddings: Normal(0, 0.1) node ID base embeddings.
- propagate_heterogeneous_kgnn: deterministic exact 2-layer forward pass with relation transforms, self terms, ReLU then identity.
- HeterogeneousKGNNScorer: candidate scoring via dot product with Evaluator compatibility.
- compute_heterogeneous_bpr_batch_loss: composed BPR ranking loss with batch-masked view and endpoint regularization.
- compute_bpr_batch_loss: composed BPR ranking loss and explicit endpoint-only base-ID regularization.
- TrainingHarnessGuard: guards preventing no-valid-batch optimizer or regularization-only updates.
- create_checkpoint_manifest & validate_checkpoint_manifest: serializable checkpoint and provenance contracts.
"""

from dataclasses import asdict, dataclass
import json
import math
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Canonicalize module in sys.modules so scripts.learned_models and learned_models
# share the exact same module object and class identities.
if __name__ == "scripts.learned_models":
    sys.modules.setdefault("learned_models", sys.modules[__name__])
elif __name__ == "learned_models":
    sys.modules.setdefault("scripts.learned_models", sys.modules[__name__])
    if "scripts" in sys.modules:
        setattr(sys.modules["scripts"], "learned_models", sys.modules[__name__])

if __package__:
    from .recommendation_evaluation import (
        ScoringFailureError,
        TrainingContext,
    )
else:
    from recommendation_evaluation import (
        ScoringFailureError,
        TrainingContext,
    )


class MissingIdError(KeyError, ValueError):
    """Raised when required entity IDs are missing from base embeddings."""


class UnknownIdError(KeyError, ValueError):
    """Raised when unknown or extra entity IDs are present in base embeddings."""


@dataclass(frozen=True)
class LightGCNConfig:
    """Validated LightGCN configuration consistent with configs/model-suite.json and docs/lightgcn-protocol.md."""

    embedding_dimension: int = 64
    propagation_layers: int = 3
    layer_combination: str = "arithmetic_mean_layers_0_through_3"
    decoder: str = "dot_product"
    normalization: str = "standard_symmetric_training_degree_normalization"
    target_masking: bool = False
    methodology: str = "user_approved_Option_B_standard_fixed_graph"

    def __post_init__(self) -> None:
        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int) or self.embedding_dimension <= 0:
            raise ValueError(f"embedding_dimension must be a positive integer, got {self.embedding_dimension!r}.")

        if isinstance(self.propagation_layers, bool) or not isinstance(self.propagation_layers, int) or self.propagation_layers <= 0:
            raise ValueError(f"propagation_layers must be a positive integer, got {self.propagation_layers!r}.")

        if self.propagation_layers != 3:
            raise ValueError(
                f"Core LightGCN requires exactly 3 propagation layers (matching layer_combination "
                f"{self.layer_combination!r}), got {self.propagation_layers!r}."
            )

        if self.layer_combination != "arithmetic_mean_layers_0_through_3":
            raise ValueError(f"Invalid layer_combination {self.layer_combination!r}. Must be 'arithmetic_mean_layers_0_through_3'.")

        if self.decoder != "dot_product":
            raise ValueError(f"Invalid decoder {self.decoder!r}. Must be 'dot_product'.")

        if self.normalization != "standard_symmetric_training_degree_normalization":
            raise ValueError(f"Invalid normalization {self.normalization!r}. Must be 'standard_symmetric_training_degree_normalization'.")

        if self.target_masking is not False:
            raise ValueError("Core LightGCN uses Option B (target_masking=False). Batch masking is an optional variant, not core.")

        if self.methodology != "user_approved_Option_B_standard_fixed_graph":
            raise ValueError(f"Invalid methodology {self.methodology!r}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LightGCNConfig":
        if not isinstance(data, dict):
            raise ValueError(f"LightGCNConfig data must be a dictionary, got {type(data)}.")
        return cls(**data)


class LightGCNGraph:
    """Bipartite user-item training graph with Option B fixed-graph training-edge reuse.

    Constructed strictly from TrainingContext condition-positive training interactions.
    Held-out edges are excluded by construction.
    Computes standard symmetric degree normalization: D^(-1/2) A D^(-1/2).
    """

    def __init__(self, context: TrainingContext, config: Optional[LightGCNConfig] = None) -> None:
        if not isinstance(context, TrainingContext):
            raise ValueError(f"context must be an instance of TrainingContext, got {type(context)}.")

        self.context = context
        self.config = config or LightGCNConfig()

        total_pos_edges = sum(len(context.positive_history.get(u, set())) for u in context.users)
        if total_pos_edges == 0:
            raise ValueError("Empty trainable graph: no condition-positive training interactions exist in TrainingContext.")

        self.users: List[str] = sorted(context.users)
        self.positive_users: List[str] = sorted(context.positive_users)
        self.warm_items: List[str] = list(context.sorted_warm_items)

        # Degree calculation:
        # User degree = number of condition-positive training items for user
        # Item degree = condition-positive training popularity for item
        self.user_degrees: Dict[str, int] = {u: len(context.positive_history.get(u, set())) for u in self.users}
        self.item_degrees: Dict[str, int] = {i: context.popularity.get(i, 0) for i in self.warm_items}

        # Build symmetric normalized adjacency lists:
        # weight(u, i) = 1.0 / sqrt(d_u * d_i)
        self.user_neighbors: Dict[str, List[Tuple[str, float]]] = {u: [] for u in self.users}
        self.item_neighbors: Dict[str, List[Tuple[str, float]]] = {i: [] for i in self.warm_items}

        edge_count = 0
        for u in self.users:
            pos_items = sorted(context.positive_history.get(u, set()))
            if not pos_items:
                continue
            d_u = self.user_degrees[u]
            if d_u <= 0:
                raise ValueError(f"User {u} has positive items but non-positive degree {d_u}.")

            for item in pos_items:
                d_i = self.item_degrees.get(item, 0)
                if d_i <= 0:
                    raise ValueError(f"Item {item} appears in positive history for user {u} but has non-positive popularity {d_i}.")

                weight = 1.0 / math.sqrt(float(d_u * d_i))
                if math.isnan(weight) or math.isinf(weight) or weight <= 0.0:
                    raise ValueError(f"Invalid non-finite normalized edge weight for ({u}, {item}): {weight}")

                self.user_neighbors[u].append((item, weight))
                self.item_neighbors[item].append((u, weight))
                edge_count += 1

        # Sort item neighbors by user ID for determinism
        for item in self.warm_items:
            self.item_neighbors[item].sort(key=lambda pair: pair[0])

        self.num_edges: int = edge_count


def build_lightgcn_graph(context: TrainingContext, config: Optional[LightGCNConfig] = None) -> LightGCNGraph:
    """Build a validated LightGCNGraph from TrainingContext."""
    return LightGCNGraph(context, config=config)


def propagate_lightgcn(
    graph: LightGCNGraph,
    user_base_embeddings: Dict[str, Sequence[float]],
    item_base_embeddings: Dict[str, Sequence[float]],
    layers: int = 3,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Exact deterministic full-batch LightGCN propagation over L layers.

    Layer l+1:
      e_u^(l+1) = sum_{i in N(u)} (1 / sqrt(d_u * d_i)) * e_i^(l)
      e_i^(l+1) = sum_{u in N(i)} (1 / sqrt(d_u * d_i)) * e_u^(l)
    Final representation:
      e_u = (1 / (L + 1)) * sum_{l=0}^L e_u^(l)
      e_i = (1 / (L + 1)) * sum_{l=0}^L e_i^(l)

    Validation requirements:
    - user_base_embeddings keys must exactly match graph.users (all context training users).
    - item_base_embeddings keys must exactly match graph.warm_items (all condition-warm items).
    - Missing or extra/unknown IDs raise MissingIdError or UnknownIdError (subclasses of KeyError and ValueError).
    """
    if not isinstance(graph, LightGCNGraph):
        raise ValueError(f"graph must be an instance of LightGCNGraph, got {type(graph)}.")

    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise ValueError(f"layers must be a positive integer, got {layers!r}.")

    if not isinstance(user_base_embeddings, dict):
        raise ValueError("user_base_embeddings must be a dictionary.")
    if not isinstance(item_base_embeddings, dict):
        raise ValueError("item_base_embeddings must be a dictionary.")

    # Validate that user base embedding keys exactly match graph.users
    expected_users = set(graph.users)
    provided_users = set(user_base_embeddings.keys())
    missing_users = expected_users - provided_users
    extra_users = provided_users - expected_users

    if missing_users:
        raise MissingIdError(f"Missing base embedding for graph user(s): {sorted(missing_users)!r}.")
    if extra_users:
        raise UnknownIdError(
            f"Unknown or extra user ID(s) in user_base_embeddings not present in graph.users: {sorted(extra_users)!r}."
        )

    # Validate that item base embedding keys exactly match graph.warm_items
    expected_items = set(graph.warm_items)
    provided_items = set(item_base_embeddings.keys())
    missing_items = expected_items - provided_items
    extra_items = provided_items - expected_items

    if missing_items:
        raise MissingIdError(f"Missing base embedding for graph item(s): {sorted(missing_items)!r}.")
    if extra_items:
        raise UnknownIdError(
            f"Unknown or extra item ID(s) in item_base_embeddings not present in graph.warm_items: {sorted(extra_items)!r}."
        )

    # Validate dimension consistency and finiteness
    target_dim: Optional[int] = None

    for u in graph.users:
        vec = user_base_embeddings[u]
        if not isinstance(vec, (list, tuple)):
            raise ValueError(f"Base embedding for user {u!r} must be a sequence.")
        if len(vec) == 0:
            raise ValueError(f"Base embedding for user {u!r} has zero dimension.")
        if target_dim is None:
            target_dim = len(vec)
        elif len(vec) != target_dim:
            raise ValueError(f"Inconsistent user base embedding dimension for {u!r}: expected {target_dim}, got {len(vec)}.")
        for val in vec:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Non-finite value in base embedding for user {u!r}: {val!r}")

    for item in graph.warm_items:
        vec = item_base_embeddings[item]
        if not isinstance(vec, (list, tuple)):
            raise ValueError(f"Base embedding for item {item!r} must be a sequence.")
        if len(vec) == 0:
            raise ValueError(f"Base embedding for item {item!r} has zero dimension.")
        if target_dim is None:
            target_dim = len(vec)
        elif len(vec) != target_dim:
            raise ValueError(f"Inconsistent item base embedding dimension for {item!r}: expected {target_dim}, got {len(vec)}.")
        for val in vec:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Non-finite value in base embedding for item {item!r}: {val!r}")

    if target_dim is None:
        raise ValueError("No embeddings provided.")

    dim = target_dim

    # Layer 0 embeddings
    user_layers: List[Dict[str, List[float]]] = [
        {u: [float(v) for v in user_base_embeddings[u]] for u in graph.users}
    ]
    item_layers: List[Dict[str, List[float]]] = [
        {i: [float(v) for v in item_base_embeddings[i]] for i in graph.warm_items}
    ]

    # Propagation through layers 1..layers
    for l in range(layers):
        prev_user = user_layers[l]
        prev_item = item_layers[l]

        next_user: Dict[str, List[float]] = {}
        for u in graph.users:
            neighbors = graph.user_neighbors.get(u, [])
            if not neighbors:
                next_user[u] = [0.0] * dim
            else:
                next_user[u] = [
                    math.fsum(w * prev_item[item][k] for item, w in neighbors)
                    for k in range(dim)
                ]

        next_item: Dict[str, List[float]] = {}
        for item in graph.warm_items:
            neighbors = graph.item_neighbors.get(item, [])
            if not neighbors:
                next_item[item] = [0.0] * dim
            else:
                next_item[item] = [
                    math.fsum(w * prev_user[u][k] for u, w in neighbors)
                    for k in range(dim)
                ]

        user_layers.append(next_user)
        item_layers.append(next_item)

    # Layer averaging over layers 0..layers
    total_layer_count = layers + 1
    propagated_users: Dict[str, List[float]] = {}
    for u in graph.users:
        propagated_users[u] = [
            math.fsum(user_layers[l][u][k] for l in range(total_layer_count)) / float(total_layer_count)
            for k in range(dim)
        ]

    propagated_items: Dict[str, List[float]] = {}
    for item in graph.warm_items:
        propagated_items[item] = [
            math.fsum(item_layers[l][item][k] for l in range(total_layer_count)) / float(total_layer_count)
            for k in range(dim)
        ]

    return propagated_users, propagated_items


class LightGCNScorer:
    """Candidate item scorer using propagated LightGCN user and item embeddings.

    Implements raw dot-product decoding.
    Integrates with Evaluator API:
    - Empty-positive-history users route to condition-positive training popularity fallback.
    - Personalization errors or missing embeddings raise ScoringFailureError.
    """

    def __init__(
        self,
        context: TrainingContext,
        user_embeddings: Dict[str, Sequence[float]],
        item_embeddings: Dict[str, Sequence[float]],
        config: Optional[LightGCNConfig] = None,
    ) -> None:
        if not isinstance(context, TrainingContext):
            raise ValueError(f"context must be an instance of TrainingContext, got {type(context)}.")
        if not isinstance(user_embeddings, dict):
            raise ValueError("user_embeddings must be a dictionary.")
        if not isinstance(item_embeddings, dict):
            raise ValueError("item_embeddings must be a dictionary.")

        self.context = context
        self.config = config or LightGCNConfig()
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

        # Validate finiteness and dimension consistency
        target_dim: Optional[int] = None
        for u in context.positive_users:
            if u not in user_embeddings:
                raise ValueError(f"Propagated embedding missing for positive user {u!r}.")
            vec = user_embeddings[u]
            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                raise ValueError(f"Invalid embedding vector for user {u!r}.")
            if target_dim is None:
                target_dim = len(vec)
            elif len(vec) != target_dim:
                raise ValueError(f"Inconsistent user embedding dimension for {u!r}: expected {target_dim}, got {len(vec)}.")
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Non-finite value in user embedding for {u!r}: {v!r}")

        for item in context.sorted_warm_items:
            if item not in item_embeddings:
                raise ValueError(f"Propagated embedding missing for warm item {item!r}.")
            vec = item_embeddings[item]
            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                raise ValueError(f"Invalid embedding vector for item {item!r}.")
            if target_dim is None:
                target_dim = len(vec)
            elif len(vec) != target_dim:
                raise ValueError(f"Inconsistent item embedding dimension for {item!r}: expected {target_dim}, got {len(vec)}.")
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Non-finite value in item embedding for {item!r}: {v!r}")

        self.embedding_dim = target_dim or self.config.embedding_dimension

    def score_candidates(
        self, user_id: str, candidate_items: Sequence[str]
    ) -> Tuple[Dict[str, float], str, Optional[str]]:
        if user_id not in self.context.users:
            raise ScoringFailureError(f"Unknown user {user_id!r} not in training context.")

        pos_history = self.context.positive_history.get(user_id, set())
        if not pos_history:
            scores = {item: float(self.context.popularity.get(item, 0)) for item in candidate_items}
            return scores, "fallback", "empty_positive_history"

        if user_id not in self.user_embeddings:
            raise ScoringFailureError(f"User {user_id!r} missing from LightGCN user embeddings.")

        u_vec = self.user_embeddings[user_id]
        scores: Dict[str, float] = {}

        for item in candidate_items:
            if item not in self.item_embeddings:
                raise ScoringFailureError(f"Candidate item {item!r} missing from LightGCN item embeddings.")
            i_vec = self.item_embeddings[item]

            try:
                score = math.fsum(u_vec[k] * i_vec[k] for k in range(len(u_vec)))
            except OverflowError as exc:
                raise ScoringFailureError(f"Dot-product score overflow for user {user_id}, item {item}: {exc}") from exc

            if math.isnan(score) or math.isinf(score):
                raise ScoringFailureError(f"Non-finite dot-product score computed for user {user_id}, item {item}: {score}")

            scores[item] = score

        return scores, "personalized", None


def compute_bpr_batch_loss(
    user_ids: Sequence[str],
    positive_item_ids: Sequence[str],
    negative_item_ids: Sequence[str],
    user_propagated_embeddings: Dict[str, Sequence[float]],
    item_propagated_embeddings: Dict[str, Sequence[float]],
    user_base_embeddings: Dict[str, Sequence[float]],
    item_base_embeddings: Dict[str, Sequence[float]],
    reg_coefficient: float = 1e-5,
) -> Dict[str, float]:
    """Composed BPR loss with explicit endpoint-only base-ID regularization.

    Formula:
      L_total = bpr_loss(s_pos, s_neg) + endpoint_regularization(...)
    Rejects empty batches with no supervised examples.
    """
    if len(user_ids) != len(positive_item_ids) or len(user_ids) != len(negative_item_ids):
        raise ValueError("User, positive item, and negative item sequences must have identical lengths.")

    if not user_ids:
        raise ValueError("Cannot compute BPR batch loss on empty batch: no supervised examples remain.")

    s_pos: List[float] = []
    s_neg: List[float] = []

    for u, pos_item, neg_item in zip(user_ids, positive_item_ids, negative_item_ids):
        if u not in user_propagated_embeddings:
            raise KeyError(f"Propagated embedding missing for user {u!r}.")
        if pos_item not in item_propagated_embeddings:
            raise KeyError(f"Propagated embedding missing for positive item {pos_item!r}.")
        if neg_item not in item_propagated_embeddings:
            raise KeyError(f"Propagated embedding missing for negative item {neg_item!r}.")

        u_vec = user_propagated_embeddings[u]
        p_vec = item_propagated_embeddings[pos_item]
        n_vec = item_propagated_embeddings[neg_item]

        if len(u_vec) != len(p_vec) or len(u_vec) != len(n_vec):
            raise ValueError(f"Inconsistent embedding dimensions in batch scoring for user {u!r}.")

        score_pos = math.fsum(u_vec[k] * p_vec[k] for k in range(len(u_vec)))
        score_neg = math.fsum(u_vec[k] * n_vec[k] for k in range(len(u_vec)))

        if math.isnan(score_pos) or math.isinf(score_pos):
            raise ValueError(f"Non-finite positive score for user {u}, item {pos_item}: {score_pos}")
        if math.isnan(score_neg) or math.isinf(score_neg):
            raise ValueError(f"Non-finite negative score for user {u}, item {neg_item}: {score_neg}")

        s_pos.append(score_pos)
        s_neg.append(score_neg)

    if __package__:
        from .recommendation_core import bpr_loss, endpoint_regularization
    else:
        from recommendation_core import bpr_loss, endpoint_regularization

    ranking_loss = bpr_loss(s_pos, s_neg)
    reg_loss = endpoint_regularization(
        user_ids=user_ids,
        positive_item_ids=positive_item_ids,
        negative_item_ids=negative_item_ids,
        user_base_embeddings=user_base_embeddings,
        item_base_embeddings=item_base_embeddings,
        coefficient=reg_coefficient,
    )

    total_loss = ranking_loss + reg_loss
    return {
        "total_loss": total_loss,
        "bpr_loss": ranking_loss,
        "reg_loss": reg_loss,
        "batch_size": float(len(user_ids)),
    }


class TrainingHarnessGuard:
    """Enforces scientific protocol guards on training and optimizer execution.

    - Prevents optimizer updates when no valid supervised examples remain.
    - Prohibits regularization-only updates and synthetic positive updates.
    - Guards against unauthorized real-data training or profiling (fails closed
      whenever is_real_data=True, regardless of allow_synthetic_updates).
    - Permits synthetic non-real-data guard validation only when allow_synthetic_updates=True
      for reference-fixture validation.
    """

    def __init__(self, allow_synthetic_updates: bool = False, is_real_data: bool = True) -> None:
        self.allow_synthetic_updates = bool(allow_synthetic_updates)
        self.is_real_data = bool(is_real_data)

    def validate_batch_trainable(self, valid_examples_count: int, skipped_empty_pool: int = 0) -> int:
        if (
            isinstance(valid_examples_count, bool)
            or not isinstance(valid_examples_count, int)
            or valid_examples_count <= 0
        ):
            raise ValueError(
                f"No valid supervised examples remain in batch (skipped empty pool: {skipped_empty_pool}). "
                "Optimizer update forbidden; regularization-only or synthetic positive updates are prohibited."
            )

        if self.is_real_data:
            raise RuntimeError(
                "Training execution on real data is NOT_AUTHORIZED under current protocol lifecycle state. "
                "Step 3a permits reference utilities and synthetic fixtures only."
            )

        if not self.allow_synthetic_updates:
            raise RuntimeError(
                "Synthetic batch validation requires allow_synthetic_updates=True. "
                "Optimizer update forbidden; reference utilities and synthetic fixtures only."
            )

        return valid_examples_count


def create_checkpoint_manifest(
    method: str,
    category: str,
    feedback_condition: str,
    protocol_version: str = "amazon4u-protocol-v2.0.0",
    config_hashes: Optional[Dict[str, str]] = None,
    seed: int = 42,
    source_revision: Optional[str] = None,
    dirty_flag: bool = False,
    failure_status: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
    best_validation_ndcg10: Optional[float] = None,
    selected_epoch: Optional[int] = None,
    early_stopping_counter: int = 0,
) -> Dict[str, Any]:
    """Create a serializable checkpoint and run-provenance manifest.

    Verifies required provenance fields and enforces that no deployment or real training
    authorization is implied.
    """
    valid_methods = {"Popularity", "LightGCN", "KG-only", "Heterogeneous KG+GNN"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method {method!r}. Must be one of {sorted(valid_methods)}.")

    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"category must be a non-empty string, got {category!r}.")

    valid_conditions = {"all_ratings", "rating_ge_4"}
    if feedback_condition not in valid_conditions:
        raise ValueError(f"Invalid feedback_condition {feedback_condition!r}. Must be one of {sorted(valid_conditions)}.")

    if not re.match(r"^amazon4u-protocol-v\d+\.\d+\.\d+$", protocol_version):
        raise ValueError(f"Invalid protocol_version {protocol_version!r}.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}.")

    if config_hashes is not None and not isinstance(config_hashes, dict):
        raise ValueError("config_hashes must be a dictionary of string hashes.")

    if source_revision is not None and not isinstance(source_revision, str):
        raise ValueError("source_revision must be a string or None.")

    if not isinstance(dirty_flag, bool):
        raise ValueError("dirty_flag must be a boolean.")

    if failure_status is not None and not isinstance(failure_status, str):
        raise ValueError("failure_status must be a string or None.")

    manifest: Dict[str, Any] = {
        "manifest_schema_version": "checkpoint-provenance-v1",
        "method": method,
        "category": category,
        "feedback_condition": feedback_condition,
        "protocol_version": protocol_version,
        "config_hashes": config_hashes or {},
        "seed": seed,
        "source_revision": source_revision,
        "dirty_flag": dirty_flag,
        "failure_status": failure_status,
        "training_authorized": False,
        "profiling_authorized": False,
        "deployment_decision": "NOT_AUTHORIZED",
        "model_config": model_config or {},
        "best_validation_ndcg10": best_validation_ndcg10,
        "selected_epoch": selected_epoch,
        "early_stopping_counter": early_stopping_counter,
    }

    # Verify JSON serializability
    json.dumps(manifest)
    return manifest


def validate_checkpoint_manifest(manifest: Dict[str, Any]) -> bool:
    """Validate that a checkpoint manifest adheres to the provenance contract."""
    if not isinstance(manifest, dict):
        return False

    required_keys = {
        "manifest_schema_version",
        "method",
        "category",
        "feedback_condition",
        "protocol_version",
        "config_hashes",
        "seed",
        "source_revision",
        "dirty_flag",
        "failure_status",
        "training_authorized",
        "profiling_authorized",
        "deployment_decision",
    }
    if not required_keys.issubset(manifest.keys()):
        return False

    if manifest.get("training_authorized") is not False:
        return False
    if manifest.get("profiling_authorized") is not False:
        return False
    if manifest.get("deployment_decision") != "NOT_AUTHORIZED":
        return False

    return True


NODE_TYPES: Tuple[str, ...] = ("User", "Product", "CategoryPath", "Brand")

DIRECTED_RELATIONS: Tuple[Tuple[str, str, str], ...] = (
    ("User", "rated", "Product"),
    ("Product", "rated_by", "User"),
    ("Product", "belongs_to", "CategoryPath"),
    ("CategoryPath", "contains_product", "Product"),
    ("Product", "has_brand", "Brand"),
    ("Brand", "brand_of", "Product"),
    ("CategoryPath", "category_child_of", "CategoryPath"),
    ("CategoryPath", "category_parent_of", "CategoryPath"),
)

RELATION_NAMES: Tuple[str, ...] = tuple(rel[1] for rel in DIRECTED_RELATIONS)

INVERSE_RELATION_MAP: Dict[str, str] = {
    "rated": "rated_by",
    "rated_by": "rated",
    "belongs_to": "contains_product",
    "contains_product": "belongs_to",
    "has_brand": "brand_of",
    "brand_of": "has_brand",
    "category_child_of": "category_parent_of",
    "category_parent_of": "category_child_of",
}

RELATION_SCHEMA_MAP: Dict[str, Tuple[str, str]] = {
    rel[1]: (rel[0], rel[2]) for rel in DIRECTED_RELATIONS
}


@dataclass(frozen=True)
class HeterogeneousKGNNConfig:
    """Validated Heterogeneous KG+GNN configuration consistent with configs/model-suite.json and docs/heterogeneous-encoder.md."""

    embedding_dimension: int = 64
    layers: int = 2
    decoder: str = "dot_product"
    activations: Tuple[str, ...] = ("ReLU", "identity")
    dropout: float = 0.0
    layer_normalization: bool = False
    output_l2_normalization: bool = False
    encoder: str = "heterogeneous_GraphSAGE_style"
    inputs: str = "learned_node_ID_embeddings"

    def __post_init__(self) -> None:
        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int) or self.embedding_dimension <= 0:
            raise ValueError(f"embedding_dimension must be a positive integer, got {self.embedding_dimension!r}.")
        if self.embedding_dimension != 64:
            raise ValueError(f"Frozen protocol requires embedding_dimension=64, got {self.embedding_dimension!r}.")

        if isinstance(self.layers, bool) or not isinstance(self.layers, int) or self.layers <= 0:
            raise ValueError(f"layers must be a positive integer, got {self.layers!r}.")
        if self.layers != 2:
            raise ValueError(f"Frozen protocol requires layers=2, got {self.layers!r}.")

        if self.decoder != "dot_product":
            raise ValueError(f"Invalid decoder {self.decoder!r}. Must be 'dot_product'.")

        act_tuple = tuple(self.activations) if isinstance(self.activations, (list, tuple)) else None
        if act_tuple != ("ReLU", "identity"):
            raise ValueError(f"Invalid activations {self.activations!r}. Must be ('ReLU', 'identity').")

        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or float(self.dropout) != 0.0:
            raise ValueError(f"Frozen protocol requires dropout=0, got {self.dropout!r}.")

        if self.layer_normalization is not False:
            raise ValueError("Frozen protocol requires layer_normalization=False.")

        if self.output_l2_normalization is not False:
            raise ValueError("Frozen protocol requires output_l2_normalization=False.")

        if self.encoder != "heterogeneous_GraphSAGE_style":
            raise ValueError(f"Invalid encoder {self.encoder!r}. Must be 'heterogeneous_GraphSAGE_style'.")

        if self.inputs != "learned_node_ID_embeddings":
            raise ValueError(f"Invalid inputs {self.inputs!r}. Must be 'learned_node_ID_embeddings'.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["activations"] = list(self.activations)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeterogeneousKGNNConfig":
        if not isinstance(data, dict):
            raise ValueError(f"HeterogeneousKGNNConfig data must be a dictionary, got {type(data)}.")
        d = dict(data)
        if "activations" in d and isinstance(d["activations"], (list, tuple)):
            d["activations"] = tuple(d["activations"])
        return cls(**d)


class HeterogeneousKGNNGraph:
    """Typed heterogeneous graph with 4 node types, 8 directed relations,
    strict held-out exclusion, and validation against corrupt/duplicate structures.
    """

    def __init__(
        self,
        nodes: Dict[str, Sequence[str]],
        edges: Optional[Dict[str, Sequence[Tuple[str, str]]]] = None,
        auto_inverse: bool = True,
        config: Optional[HeterogeneousKGNNConfig] = None,
    ) -> None:
        if not isinstance(nodes, dict):
            raise ValueError("nodes must be a dictionary mapping node_type to a sequence of node IDs.")

        self.config = config or HeterogeneousKGNNConfig()
        self.nodes: Dict[str, List[str]] = {}
        self.node_sets: Dict[str, Set[str]] = {}

        total_nodes = 0
        for ntype, id_seq in nodes.items():
            if ntype not in NODE_TYPES:
                raise ValueError(f"Unknown node type {ntype!r}. Must be one of {NODE_TYPES}.")
            if not isinstance(id_seq, (list, tuple, set)):
                raise ValueError(f"Node sequence for type {ntype!r} must be a sequence, got {type(id_seq)}.")

            seen: Set[str] = set()
            id_list: List[str] = []
            for nid in id_seq:
                if not isinstance(nid, str) or not nid.strip():
                    raise ValueError(f"Invalid node ID for type {ntype!r}: {nid!r}. Must be non-empty string.")
                if nid in seen:
                    raise ValueError(f"Duplicate node ID {nid!r} for type {ntype!r}.")
                seen.add(nid)
                id_list.append(nid)

            self.nodes[ntype] = sorted(id_list)
            self.node_sets[ntype] = set(id_list)
            total_nodes += len(id_list)

        # Initialize any missing node types as empty lists
        for ntype in NODE_TYPES:
            if ntype not in self.nodes:
                self.nodes[ntype] = []
                self.node_sets[ntype] = set()

        if total_nodes == 0:
            raise ValueError("Empty graph: no nodes provided.")

        raw_edges: Dict[str, List[Tuple[str, str]]] = {r: [] for r in RELATION_NAMES}
        if edges is not None:
            if not isinstance(edges, dict):
                raise ValueError("edges must be a dictionary mapping relation_name to sequence of (src, dst) pairs.")
            for r_name, edge_seq in edges.items():
                if r_name not in RELATION_NAMES:
                    raise ValueError(f"Unknown relation name {r_name!r}. Must be one of {RELATION_NAMES}.")
                if not isinstance(edge_seq, (list, tuple, set)):
                    raise ValueError(f"Edges for relation {r_name!r} must be a sequence of (src, dst) pairs.")

                src_type, dst_type = RELATION_SCHEMA_MAP[r_name]
                seen_pairs: Set[Tuple[str, str]] = set()

                for edge in edge_seq:
                    if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                        raise ValueError(f"Edge in relation {r_name!r} must be a (src, dst) pair, got {edge!r}.")
                    src, dst = edge[0], edge[1]
                    if not isinstance(src, str) or not isinstance(dst, str):
                        raise ValueError(f"Edge endpoints must be strings, got ({src!r}, {dst!r}).")

                    if src not in self.node_sets[src_type]:
                        raise UnknownIdError(f"Source node ID {src!r} not found in node type {src_type!r} for relation {r_name!r}.")
                    if dst not in self.node_sets[dst_type]:
                        raise UnknownIdError(f"Destination node ID {dst!r} not found in node type {dst_type!r} for relation {r_name!r}.")

                    pair = (src, dst)
                    if pair in seen_pairs:
                        raise ValueError(f"Duplicate edge {pair!r} in relation {r_name!r}.")
                    seen_pairs.add(pair)
                    raw_edges[r_name].append(pair)

        # Handle inverse edges
        edge_sets: Dict[str, Set[Tuple[str, str]]] = {r: set(raw_edges[r]) for r in RELATION_NAMES}
        if auto_inverse:
            for r_name in RELATION_NAMES:
                inv_name = INVERSE_RELATION_MAP[r_name]
                for src, dst in raw_edges[r_name]:
                    inv_pair = (dst, src)
                    edge_sets[inv_name].add(inv_pair)

        self.edges: Dict[str, List[Tuple[str, str]]] = {}
        self.incoming_neighbors: Dict[str, Dict[str, List[str]]] = {}
        total_edge_count = 0

        for r_name in RELATION_NAMES:
            sorted_pairs = sorted(edge_sets[r_name])
            self.edges[r_name] = sorted_pairs
            total_edge_count += len(sorted_pairs)

            inc_dict: Dict[str, List[str]] = {}
            for src, dst in sorted_pairs:
                if dst not in inc_dict:
                    inc_dict[dst] = []
                inc_dict[dst].append(src)
            self.incoming_neighbors[r_name] = inc_dict

        self.num_edges: int = total_edge_count


def build_heterogeneous_graph(
    context: TrainingContext,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[HeterogeneousKGNNConfig] = None,
) -> HeterogeneousKGNNGraph:
    """Build a validated HeterogeneousKGNNGraph from TrainingContext and optional metadata.

    Strictly uses condition-positive training interactions.
    Held-out interactions are excluded by construction.
    Rating values are not used as encoder features or weights.
    Metadata attributes are filtered to warm products only.
    """
    if not isinstance(context, TrainingContext):
        raise ValueError(f"context must be an instance of TrainingContext, got {type(context)}.")

    users = sorted(context.users)
    products = list(context.sorted_warm_items)

    total_pos_edges = sum(len(context.positive_history.get(u, set())) for u in context.users)
    if total_pos_edges == 0:
        raise ValueError("Empty usable heterogeneous graph: no condition-positive training interactions exist in TrainingContext.")

    nodes: Dict[str, List[str]] = {
        "User": users,
        "Product": products,
        "CategoryPath": [],
        "Brand": [],
    }
    edges: Dict[str, List[Tuple[str, str]]] = {r: [] for r in RELATION_NAMES}

    # Training interactions
    for u in users:
        pos_items = sorted(context.positive_history.get(u, set()))
        for item in pos_items:
            if item in context.warm_items:
                edges["rated"].append((u, item))

    # Parse metadata if provided
    categories_set: Set[str] = set()
    brands_set: Set[str] = set()

    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a dictionary, got {type(metadata)}.")

        if any(k in metadata for k in ("product_categories", "product_brands", "category_hierarchy", "belongs_to", "has_brand", "category_child_of")):
            if "belongs_to" in metadata:
                for src, dst in metadata["belongs_to"]:
                    if src in context.warm_items:
                        categories_set.add(dst)
                        edges["belongs_to"].append((src, dst))
            elif "product_categories" in metadata:
                for p, cat_list in metadata["product_categories"].items():
                    if p in context.warm_items:
                        for c in cat_list:
                            categories_set.add(c)
                            edges["belongs_to"].append((p, c))

            if "has_brand" in metadata:
                for src, dst in metadata["has_brand"]:
                    if src in context.warm_items:
                        brands_set.add(dst)
                        edges["has_brand"].append((src, dst))
            elif "product_brands" in metadata:
                for p, b in metadata["product_brands"].items():
                    if p in context.warm_items and b:
                        brands_set.add(b)
                        edges["has_brand"].append((p, b))

            if "category_child_of" in metadata:
                for child, parent in metadata["category_child_of"]:
                    categories_set.add(child)
                    categories_set.add(parent)
                    edges["category_child_of"].append((child, parent))
            elif "category_hierarchy" in metadata:
                for child, parent in metadata["category_hierarchy"]:
                    categories_set.add(child)
                    categories_set.add(parent)
                    edges["category_child_of"].append((child, parent))
        else:
            for p, pmeta in metadata.items():
                if p not in context.warm_items:
                    continue
                if isinstance(pmeta, dict):
                    cats = pmeta.get("categories") or pmeta.get("category_paths") or []
                    for c in cats:
                        if isinstance(c, str) and c.strip():
                            categories_set.add(c)
                            edges["belongs_to"].append((p, c))
                    brand = pmeta.get("brand") or pmeta.get("has_brand")
                    if isinstance(brand, str) and brand.strip():
                        brands_set.add(brand)
                        edges["has_brand"].append((p, brand))

    nodes["CategoryPath"] = sorted(categories_set)
    nodes["Brand"] = sorted(brands_set)

    return HeterogeneousKGNNGraph(nodes=nodes, edges=edges, auto_inverse=True, config=config)


def mask_heterogeneous_batch_edges(
    graph: HeterogeneousKGNNGraph,
    user_ids: Sequence[str],
    positive_item_ids: Sequence[str],
) -> HeterogeneousKGNNGraph:
    """Batch positive-edge masking helper for Heterogeneous KG+GNN.

    Given supervised positive User-Product pairs, removes direct forward 'rated' edges,
    reverse 'rated_by' edges, and duplicate direct equivalents from the effective adjacency.
    Preserves all unrelated interaction edges and all metadata edges/valid multi-hop paths.
    Active relation membership is computed strictly from the resulting masked edges.
    """
    if not isinstance(graph, HeterogeneousKGNNGraph):
        raise ValueError(f"graph must be an instance of HeterogeneousKGNNGraph, got {type(graph)}.")

    if len(user_ids) != len(positive_item_ids):
        raise ValueError(
            f"user_ids (len {len(user_ids)}) and positive_item_ids (len {len(positive_item_ids)}) must have identical lengths."
        )

    masked_pairs = set(zip(user_ids, positive_item_ids))

    new_edges: Dict[str, List[Tuple[str, str]]] = {}
    for r_name in RELATION_NAMES:
        if r_name == "rated":
            new_edges[r_name] = [
                (src, dst) for (src, dst) in graph.edges["rated"]
                if (src, dst) not in masked_pairs
            ]
        elif r_name == "rated_by":
            new_edges[r_name] = [
                (src, dst) for (src, dst) in graph.edges["rated_by"]
                if (dst, src) not in masked_pairs
            ]
        else:
            new_edges[r_name] = list(graph.edges[r_name])

    return HeterogeneousKGNNGraph(
        nodes=graph.nodes,
        edges=new_edges,
        auto_inverse=False,
        config=graph.config,
    )


@dataclass
class HeterogeneousParameters:
    """Parameters for Heterogeneous KG+GNN:
    - relation_matrices[layer][rel_name]: 64x64 W_r matrix per directed relation per layer (no bias).
    - self_matrices[layer][node_type]: 64x64 W_self matrix per node type per layer.
    - self_biases[layer][node_type]: 64-dim bias vector per node type per layer.
    """

    relation_matrices: Dict[int, Dict[str, List[List[float]]]]
    self_matrices: Dict[int, Dict[str, List[List[float]]]]
    self_biases: Dict[int, Dict[str, List[float]]]
    embedding_dimension: int = 64
    layers: int = 2

    def validate(self) -> None:
        d = self.embedding_dimension
        for l in range(self.layers):
            if l not in self.relation_matrices:
                raise ValueError(f"Missing relation_matrices for layer {l}.")
            for r in RELATION_NAMES:
                if r not in self.relation_matrices[l]:
                    raise ValueError(f"Missing relation matrix for relation {r!r} in layer {l}.")
                mat = self.relation_matrices[l][r]
                if len(mat) != d or any(len(row) != d for row in mat):
                    raise ValueError(f"Relation matrix for {r!r} layer {l} must be {d}x{d}.")
                for row in mat:
                    for v in row:
                        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                            raise ValueError(f"Non-finite value in relation matrix {r!r} layer {l}: {v!r}")

            if l not in self.self_matrices:
                raise ValueError(f"Missing self_matrices for layer {l}.")
            for t in NODE_TYPES:
                if t not in self.self_matrices[l]:
                    raise ValueError(f"Missing self matrix for node type {t!r} in layer {l}.")
                mat = self.self_matrices[l][t]
                if len(mat) != d or any(len(row) != d for row in mat):
                    raise ValueError(f"Self matrix for {t!r} layer {l} must be {d}x{d}.")
                for row in mat:
                    for v in row:
                        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                            raise ValueError(f"Non-finite value in self matrix {t!r} layer {l}: {v!r}")

            if l not in self.self_biases:
                raise ValueError(f"Missing self_biases for layer {l}.")
            for t in NODE_TYPES:
                if t not in self.self_biases[l]:
                    raise ValueError(f"Missing self bias for node type {t!r} in layer {l}.")
                b = self.self_biases[l][t]
                if len(b) != d:
                    raise ValueError(f"Self bias for {t!r} layer {l} must be length {d}.")
                for v in b:
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                        raise ValueError(f"Non-finite value in self bias {t!r} layer {l}: {v!r}")


def initialize_heterogeneous_parameters(
    config: Optional[HeterogeneousKGNNConfig] = None,
    seed: Optional[int] = None,
) -> HeterogeneousParameters:
    """Initialize Heterogeneous KG+GNN parameters:
    - Relation and self matrices: Xavier uniform with gain 1.0 (limit = sqrt(6 / (fan_in + fan_out))).
    - Self biases: initialized to 0.0.
    """
    cfg = config or HeterogeneousKGNNConfig()
    d = cfg.embedding_dimension
    layers = cfg.layers
    rng = random.Random(seed) if seed is not None else random.Random()

    limit = 1.0 * math.sqrt(6.0 / float(d + d))

    relation_matrices: Dict[int, Dict[str, List[List[float]]]] = {}
    self_matrices: Dict[int, Dict[str, List[List[float]]]] = {}
    self_biases: Dict[int, Dict[str, List[float]]] = {}

    for l in range(layers):
        relation_matrices[l] = {}
        for r in RELATION_NAMES:
            relation_matrices[l][r] = [
                [rng.uniform(-limit, limit) for _ in range(d)]
                for _ in range(d)
            ]

        self_matrices[l] = {}
        self_biases[l] = {}
        for t in NODE_TYPES:
            self_matrices[l][t] = [
                [rng.uniform(-limit, limit) for _ in range(d)]
                for _ in range(d)
            ]
            self_biases[l][t] = [0.0] * d

    params = HeterogeneousParameters(
        relation_matrices=relation_matrices,
        self_matrices=self_matrices,
        self_biases=self_biases,
        embedding_dimension=d,
        layers=layers,
    )
    params.validate()
    return params


def initialize_node_embeddings(
    graph: HeterogeneousKGNNGraph,
    config: Optional[HeterogeneousKGNNConfig] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Initialize learned node-ID base embeddings:
    Each entry is drawn independently from Normal(mean=0, std=0.1).
    """
    if not isinstance(graph, HeterogeneousKGNNGraph):
        raise ValueError(f"graph must be an instance of HeterogeneousKGNNGraph, got {type(graph)}.")

    cfg = config or graph.config
    d = cfg.embedding_dimension
    rng = random.Random(seed) if seed is not None else random.Random()

    embeddings: Dict[str, Dict[str, List[float]]] = {}
    for ntype in NODE_TYPES:
        embeddings[ntype] = {}
        for nid in graph.nodes[ntype]:
            embeddings[ntype][nid] = [rng.gauss(0.0, 0.1) for _ in range(d)]

    return embeddings


def propagate_heterogeneous_kgnn(
    graph: HeterogeneousKGNNGraph,
    node_base_embeddings: Dict[str, Dict[str, Sequence[float]]],
    parameters: HeterogeneousParameters,
    config: Optional[HeterogeneousKGNNConfig] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Deterministic exact 2-layer forward pass for Heterogeneous KG+GNN.

    Operator per layer l:
      For target v of type t_v:
        Self term: W_self,t_v^(l) * h_v^(l) + b_t_v^(l)
        For each incoming active relation r in R_v:
          m_(v,r)^(l) = W_r^(l) * mean(h_w^(l) for w in N_r(v))
        Neighbor contribution: mean(m_(v,r)^(l) for r in R_v) if R_v nonempty else 0
        z_v^(l+1) = self_term + neighbor_contribution
      Activation:
        Layer 1 (l=0): h_v^(1) = ReLU(z_v^(1))
        Layer 2 (l=1): h_v^(2) = z_v^(2) (identity)
    """
    if not isinstance(graph, HeterogeneousKGNNGraph):
        raise ValueError(f"graph must be an instance of HeterogeneousKGNNGraph, got {type(graph)}.")
    if not isinstance(node_base_embeddings, dict):
        raise ValueError("node_base_embeddings must be a dictionary.")

    cfg = config or graph.config
    d = cfg.embedding_dimension
    parameters.validate()
    if parameters.embedding_dimension != d:
        raise ValueError(
            f"Parameter embedding_dimension {parameters.embedding_dimension!r} does not match config embedding_dimension {d!r}."
        )
    if parameters.layers != cfg.layers:
        raise ValueError(f"Parameter layers {parameters.layers!r} does not match config layers {cfg.layers!r}.")

    for ntype in NODE_TYPES:
        expected_ids = set(graph.nodes[ntype])
        if ntype not in node_base_embeddings:
            if expected_ids:
                raise MissingIdError(f"Missing base embeddings for node type {ntype!r}.")
            type_emb: Dict[str, Sequence[float]] = {}
        else:
            type_emb = node_base_embeddings[ntype]
            if not isinstance(type_emb, dict):
                raise ValueError(f"node_base_embeddings[{ntype!r}] must be a dictionary.")

        provided_ids = set(type_emb.keys())
        missing_ids = expected_ids - provided_ids
        extra_ids = provided_ids - expected_ids

        if missing_ids:
            raise MissingIdError(f"Missing base embeddings for {ntype!r} nodes: {sorted(missing_ids)!r}.")
        if extra_ids:
            raise UnknownIdError(f"Unknown or extra base embedding IDs for {ntype!r}: {sorted(extra_ids)!r}.")

        for nid, vec in type_emb.items():
            if not isinstance(vec, (list, tuple)):
                raise ValueError(f"Embedding vector for ({ntype!r}, {nid!r}) must be a sequence.")
            if len(vec) != d:
                raise ValueError(f"Inconsistent embedding dimension for ({ntype!r}, {nid!r}): expected {d}, got {len(vec)}.")
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Non-finite value in base embedding for ({ntype!r}, {nid!r}): {v!r}")

    for provided_type in node_base_embeddings.keys():
        if provided_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type {provided_type!r} in node_base_embeddings.")

    current_h: Dict[str, Dict[str, List[float]]] = {
        ntype: {nid: [float(val) for val in node_base_embeddings[ntype][nid]] for nid in graph.nodes[ntype]}
        for ntype in NODE_TYPES
    }

    for l in range(cfg.layers):
        next_h: Dict[str, Dict[str, List[float]]] = {ntype: {} for ntype in NODE_TYPES}
        w_self_layer = parameters.self_matrices[l]
        b_self_layer = parameters.self_biases[l]
        w_rel_layer = parameters.relation_matrices[l]

        for ntype in NODE_TYPES:
            for v in graph.nodes[ntype]:
                h_v = current_h[ntype][v]
                w_self = w_self_layer[ntype]
                b_self = b_self_layer[ntype]

                # 1. Self term
                self_term = [
                    math.fsum(w_self[i][j] * h_v[j] for j in range(d)) + b_self[i]
                    for i in range(d)
                ]

                # 2. Active incoming relations
                relation_messages: List[List[float]] = []
                for rel in DIRECTED_RELATIONS:
                    src_t, r_name, dst_t = rel
                    if dst_t != ntype:
                        continue
                    incoming = graph.incoming_neighbors.get(r_name, {}).get(v, [])
                    if not incoming:
                        continue
                    k_neighbors = len(incoming)
                    mean_neighbor = [
                        math.fsum(current_h[src_t][w][j] for w in incoming) / float(k_neighbors)
                        for j in range(d)
                    ]
                    w_r = w_rel_layer[r_name]
                    m_vr = [
                        math.fsum(w_r[i][j] * mean_neighbor[j] for j in range(d))
                        for i in range(d)
                    ]
                    relation_messages.append(m_vr)

                # 3. Cross-relation unweighted mean
                if not relation_messages:
                    neighbor_term = [0.0] * d
                else:
                    n_active = len(relation_messages)
                    neighbor_term = [
                        math.fsum(msg[i] for msg in relation_messages) / float(n_active)
                        for i in range(d)
                    ]

                # 4. Pre-activation
                z_v = [self_term[i] + neighbor_term[i] for i in range(d)]

                # 5. Activation: Layer 0 -> 1 is ReLU; Layer 1 -> 2 is identity
                if l == 0:
                    h_next = [max(0.0, z_v[i]) for i in range(d)]
                else:
                    h_next = z_v

                next_h[ntype][v] = h_next

        current_h = next_h

    return current_h


class HeterogeneousKGNNScorer:
    """Candidate item scorer using computed Heterogeneous KG+GNN user and product embeddings.

    Implements raw dot-product decoding.
    Integrates with Evaluator API:
    - Empty-positive-history users route to condition-positive training popularity fallback.
    - Personalization errors or missing/corrupt embeddings raise ScoringFailureError.
    """

    def __init__(
        self,
        context: TrainingContext,
        user_embeddings: Optional[Dict[str, Sequence[float]]] = None,
        product_embeddings: Optional[Dict[str, Sequence[float]]] = None,
        node_embeddings: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
        item_embeddings: Optional[Dict[str, Sequence[float]]] = None,
        config: Optional[HeterogeneousKGNNConfig] = None,
    ) -> None:
        if not isinstance(context, TrainingContext):
            raise ValueError(f"context must be an instance of TrainingContext, got {type(context)}.")

        self.context = context
        self.config = config or HeterogeneousKGNNConfig()

        if node_embeddings is not None:
            if not isinstance(node_embeddings, dict):
                raise ValueError("node_embeddings must be a dictionary.")
            user_embeddings = user_embeddings or node_embeddings.get("User")
            product_embeddings = product_embeddings or node_embeddings.get("Product")

        if product_embeddings is None and item_embeddings is not None:
            product_embeddings = item_embeddings

        if user_embeddings is None or not isinstance(user_embeddings, dict):
            raise ValueError("user_embeddings must be provided as a dictionary.")
        if product_embeddings is None or not isinstance(product_embeddings, dict):
            raise ValueError("product_embeddings (or item_embeddings) must be provided as a dictionary.")

        self.user_embeddings = user_embeddings
        self.product_embeddings = product_embeddings

        # Validate finiteness and dimension consistency
        target_dim: Optional[int] = None
        for u in context.positive_users:
            if u not in user_embeddings:
                raise ValueError(f"Propagated embedding missing for positive user {u!r}.")
            vec = user_embeddings[u]
            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                raise ValueError(f"Invalid embedding vector for user {u!r}.")
            if target_dim is None:
                target_dim = len(vec)
            elif len(vec) != target_dim:
                raise ValueError(f"Inconsistent user embedding dimension for {u!r}: expected {target_dim}, got {len(vec)}.")
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Non-finite value in user embedding for {u!r}: {v!r}")

        for item in context.sorted_warm_items:
            if item not in product_embeddings:
                raise ValueError(f"Propagated embedding missing for warm product {item!r}.")
            vec = product_embeddings[item]
            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                raise ValueError(f"Invalid embedding vector for product {item!r}.")
            if target_dim is None:
                target_dim = len(vec)
            elif len(vec) != target_dim:
                raise ValueError(f"Inconsistent product embedding dimension for {item!r}: expected {target_dim}, got {len(vec)}.")
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Non-finite value in product embedding for {item!r}: {v!r}")

        self.embedding_dim = target_dim or self.config.embedding_dimension

    def score_candidates(
        self, user_id: str, candidate_items: Sequence[str]
    ) -> Tuple[Dict[str, float], str, Optional[str]]:
        if user_id not in self.context.users:
            raise ScoringFailureError(f"Unknown user {user_id!r} not in training context.")

        pos_history = self.context.positive_history.get(user_id, set())
        if not pos_history:
            scores = {item: float(self.context.popularity.get(item, 0)) for item in candidate_items}
            return scores, "fallback", "empty_positive_history"

        if user_id not in self.user_embeddings:
            raise ScoringFailureError(f"User {user_id!r} missing from Heterogeneous KG+GNN user embeddings.")

        u_vec = self.user_embeddings[user_id]
        scores: Dict[str, float] = {}

        for item in candidate_items:
            if item not in self.product_embeddings:
                raise ScoringFailureError(f"Candidate product {item!r} missing from Heterogeneous KG+GNN product embeddings.")
            i_vec = self.product_embeddings[item]

            try:
                score = math.fsum(u_vec[k] * i_vec[k] for k in range(len(u_vec)))
            except OverflowError as exc:
                raise ScoringFailureError(f"Dot-product score overflow for user {user_id}, product {item}: {exc}") from exc

            if math.isnan(score) or math.isinf(score):
                raise ScoringFailureError(f"Non-finite dot-product score computed for user {user_id}, product {item}: {score}")

            scores[item] = score

        return scores, "personalized", None


def compute_heterogeneous_bpr_batch_loss(
    graph: HeterogeneousKGNNGraph,
    parameters: HeterogeneousParameters,
    base_node_embeddings: Dict[str, Dict[str, Sequence[float]]],
    batch_user_ids: Sequence[str],
    batch_positive_item_ids: Sequence[str],
    batch_negative_item_ids: Sequence[str],
    reg_coefficient: float = 1e-5,
    config: Optional[HeterogeneousKGNNConfig] = None,
) -> Dict[str, float]:
    """Composed BPR loss for Heterogeneous KG+GNN using the same batch-masked graph view.

    Steps:
    1. Removes direct forward 'rated' and reverse 'rated_by' edges for all batch positive pairs.
    2. Propagates representations through the masked graph view.
    3. Scores both positive and negative items using the exact same masked representations.
    4. Applies standard BPR ranking loss.
    5. Applies explicit endpoint regularization on unique supervised base-ID endpoints (User and Product).
    No optimizer updates and no regularization-only updates.
    """
    if len(batch_user_ids) != len(batch_positive_item_ids) or len(batch_user_ids) != len(batch_negative_item_ids):
        raise ValueError("User, positive item, and negative item sequences must have identical lengths.")

    if not batch_user_ids:
        raise ValueError("Cannot compute BPR batch loss on empty batch: no supervised examples remain.")

    # 1. Mask graph
    masked_graph = mask_heterogeneous_batch_edges(
        graph=graph,
        user_ids=batch_user_ids,
        positive_item_ids=batch_positive_item_ids,
    )

    # 2. Propagate on masked graph
    masked_embeddings = propagate_heterogeneous_kgnn(
        graph=masked_graph,
        node_base_embeddings=base_node_embeddings,
        parameters=parameters,
        config=config,
    )

    user_rep = masked_embeddings["User"]
    product_rep = masked_embeddings["Product"]

    s_pos: List[float] = []
    s_neg: List[float] = []

    for u, pos_item, neg_item in zip(batch_user_ids, batch_positive_item_ids, batch_negative_item_ids):
        if u not in user_rep:
            raise KeyError(f"Propagated embedding missing for user {u!r}.")
        if pos_item not in product_rep:
            raise KeyError(f"Propagated embedding missing for positive product {pos_item!r}.")
        if neg_item not in product_rep:
            raise KeyError(f"Propagated embedding missing for negative product {neg_item!r}.")

        u_vec = user_rep[u]
        p_vec = product_rep[pos_item]
        n_vec = product_rep[neg_item]

        if len(u_vec) != len(p_vec) or len(u_vec) != len(n_vec):
            raise ValueError(f"Inconsistent embedding dimensions in batch scoring for user {u!r}.")

        score_pos = math.fsum(u_vec[k] * p_vec[k] for k in range(len(u_vec)))
        score_neg = math.fsum(u_vec[k] * n_vec[k] for k in range(len(u_vec)))

        if math.isnan(score_pos) or math.isinf(score_pos):
            raise ValueError(f"Non-finite positive score for user {u}, product {pos_item}: {score_pos}")
        if math.isnan(score_neg) or math.isinf(score_neg):
            raise ValueError(f"Non-finite negative score for user {u}, product {neg_item}: {score_neg}")

        s_pos.append(score_pos)
        s_neg.append(score_neg)

    if __package__:
        from .recommendation_core import bpr_loss, endpoint_regularization
    else:
        from recommendation_core import bpr_loss, endpoint_regularization

    ranking_loss = bpr_loss(s_pos, s_neg)

    user_base = base_node_embeddings.get("User", {})
    product_base = base_node_embeddings.get("Product", {})

    reg_loss = endpoint_regularization(
        user_ids=batch_user_ids,
        positive_item_ids=batch_positive_item_ids,
        negative_item_ids=batch_negative_item_ids,
        user_base_embeddings=user_base,
        item_base_embeddings=product_base,
        coefficient=reg_coefficient,
    )

    total_loss = ranking_loss + reg_loss
    return {
        "total_loss": total_loss,
        "bpr_loss": ranking_loss,
        "reg_loss": reg_loss,
        "batch_size": float(len(batch_user_ids)),
    }

