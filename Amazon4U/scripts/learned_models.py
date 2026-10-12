"""Amazon4U Step 3a: learned-model training harness foundation and LightGCN reference implementation.

Python standard library only.
Provides:
- LightGCNConfig: validated configuration for LightGCN architecture and Option B semantics.
- LightGCNGraph: fixed condition-positive training graph construction from TrainingContext,
  with symmetric degree normalization, reverse edges, and strict held-out exclusion.
- propagate_lightgcn: deterministic exact propagation over L layers with layer 0..L averaging.
- LightGCNScorer: deterministic candidate dot-product scoring with Evaluator compatibility
  and empty-positive-history popularity fallback.
- compute_bpr_batch_loss: composed BPR ranking loss and explicit endpoint-only base-ID regularization.
- TrainingHarnessGuard: guards preventing no-valid-batch optimizer or regularization-only updates.
- create_checkpoint_manifest & validate_checkpoint_manifest: serializable checkpoint and provenance contracts.
"""

from dataclasses import asdict, dataclass
import json
import math
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
