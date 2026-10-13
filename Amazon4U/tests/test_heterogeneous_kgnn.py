"""Focused unit and regression tests for Heterogeneous KG+GNN reference encoder Step 3b.

Python standard library only.
Tests:
- HeterogeneousKGNNConfig validation and frozen defaults matching model-suite.json and docs/heterogeneous-encoder.md.
- HeterogeneousKGNNGraph construction, typed nodes/relations, duplicate/corrupt validation, and auto-inverses.
- build_heterogeneous_graph from TrainingContext with strict held-out exclusion and rating-free edges.
- Batch positive-edge masking helper: removes direct forward/reverse edges, preserves unrelated/metadata/multi-hop edges.
- Parameter and base embedding initialization: Xavier uniform gain 1, zero self biases, Normal(0, 0.1) IDs, fixed seeds.
- Exact hand calculations for 2-relation fixture: within-relation mean, relation transform, cross-relation mean, self term.
- Proof that increasing neighbor count alone does not change relation mixture weights.
- Zero-neighbor behavior: zero neighbor contribution, self-term retained.
- Activation order: Layer 1 ReLU, Layer 2 identity.
- Raw dot-product candidate decoder, Evaluator integration, empty-positive popularity fallback, and failure modes.
- Composed BPR batch loss using same batch-masked graph view for positive and negative items.
- Endpoint regularization targeting User and Product base-IDs only, excluding Brand/CategoryPath.
- TrainingHarnessGuard compatibility and checkpoint manifest contracts.
- Module re-exports and import compatibility across scripts.learned_models, learned_models, and recommendation_core.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
import unittest
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import learned_models
import scripts.learned_models
import recommendation_core
import scripts.recommendation_core

from learned_models import (
    DIRECTED_RELATIONS,
    HeterogeneousKGNNConfig,
    HeterogeneousKGNNGraph,
    HeterogeneousKGNNScorer,
    HeterogeneousParameters,
    MissingIdError,
    NODE_TYPES,
    RELATION_NAMES,
    TrainingHarnessGuard,
    UnknownIdError,
    build_heterogeneous_graph,
    compute_heterogeneous_bpr_batch_loss,
    create_checkpoint_manifest,
    initialize_heterogeneous_parameters,
    initialize_node_embeddings,
    mask_heterogeneous_batch_edges,
    propagate_heterogeneous_kgnn,
    validate_checkpoint_manifest,
)
from recommendation_core import (
    Evaluator,
    ScoringFailureError,
    TrainingContext,
    bpr_loss,
    endpoint_regularization,
)


class HeterogeneousKGNNConfigTests(unittest.TestCase):
    """Test Heterogeneous KG+GNN configuration validation and frozen protocol constraints."""

    def test_default_config_matches_frozen_protocol(self):
        cfg = HeterogeneousKGNNConfig()
        self.assertEqual(cfg.embedding_dimension, 64)
        self.assertEqual(cfg.layers, 2)
        self.assertEqual(cfg.decoder, "dot_product")
        self.assertEqual(cfg.activations, ("ReLU", "identity"))
        self.assertEqual(cfg.dropout, 0.0)
        self.assertFalse(cfg.layer_normalization)
        self.assertFalse(cfg.output_l2_normalization)
        self.assertEqual(cfg.encoder, "heterogeneous_GraphSAGE_style")
        self.assertEqual(cfg.inputs, "learned_node_ID_embeddings")

        as_dict = cfg.to_dict()
        self.assertIsInstance(as_dict, dict)
        self.assertEqual(as_dict["activations"], ["ReLU", "identity"])
        reconstructed = HeterogeneousKGNNConfig.from_dict(as_dict)
        self.assertEqual(cfg, reconstructed)

    def test_invalid_configurations_rejected(self):
        # Invalid embedding dimensions (must be 64)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(embedding_dimension=0)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(embedding_dimension=32)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(embedding_dimension=128)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(embedding_dimension=True)  # type: ignore
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(embedding_dimension=64.0)  # type: ignore

        # Invalid layers (must be 2)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(layers=0)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(layers=1)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(layers=3)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(layers=False)  # type: ignore

        # Invalid decoder
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(decoder="cosine")
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(decoder="mlp")

        # Invalid activations
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(activations=("identity", "ReLU"))  # type: ignore
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(activations=("ReLU", "ReLU"))  # type: ignore
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(activations=("ReLU",))  # type: ignore

        # Invalid dropout
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(dropout=0.1)
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(dropout=True)  # type: ignore

        # Invalid layer normalization
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(layer_normalization=True)

        # Invalid output L2 normalization
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(output_l2_normalization=True)

        # Invalid encoder
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(encoder="lightgcn")

        # Invalid inputs
        with self.assertRaises(ValueError):
            HeterogeneousKGNNConfig(inputs="pretrained_features")


class HeterogeneousKGNNGraphValidationTests(unittest.TestCase):
    """Test graph structure, typed endpoints, duplicate checks, auto-inverses, and held-out isolation."""

    def test_valid_graph_construction_and_auto_inverses(self):
        nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2"],
            "CategoryPath": ["c1", "c2"],
            "Brand": ["b1"],
        }
        forward_edges = {
            "rated": [("u1", "p1"), ("u2", "p2")],
            "belongs_to": [("p1", "c1"), ("p2", "c2")],
            "has_brand": [("p1", "b1")],
            "category_child_of": [("c2", "c1")],
        }
        graph = HeterogeneousKGNNGraph(nodes=nodes, edges=forward_edges, auto_inverse=True)

        # Verify all 4 node types are stored
        self.assertEqual(graph.nodes["User"], ["u1", "u2"])
        self.assertEqual(graph.nodes["Product"], ["p1", "p2"])
        self.assertEqual(graph.nodes["CategoryPath"], ["c1", "c2"])
        self.assertEqual(graph.nodes["Brand"], ["b1"])

        # Verify auto-inverse generation
        self.assertIn(("p1", "u1"), graph.edges["rated_by"])
        self.assertIn(("p2", "u2"), graph.edges["rated_by"])
        self.assertIn(("c1", "p1"), graph.edges["contains_product"])
        self.assertIn(("c2", "p2"), graph.edges["contains_product"])
        self.assertIn(("b1", "p1"), graph.edges["brand_of"])
        self.assertIn(("c1", "c2"), graph.edges["category_parent_of"])

        # Verify incoming neighbors
        self.assertEqual(graph.incoming_neighbors["rated_by"]["u1"], ["p1"])
        self.assertEqual(graph.incoming_neighbors["rated"]["p1"], ["u1"])
        self.assertEqual(graph.incoming_neighbors["contains_product"]["p1"], ["c1"])
        self.assertEqual(graph.incoming_neighbors["brand_of"]["p1"], ["b1"])
        self.assertEqual(graph.incoming_neighbors["category_parent_of"]["c2"], ["c1"])

    def test_invalid_node_types_and_duplicates_rejected(self):
        # Unknown node type
        with self.assertRaises(ValueError):
            HeterogeneousKGNNGraph(nodes={"User": ["u1"], "Store": ["s1"]})

        # Empty graph
        with self.assertRaises(ValueError):
            HeterogeneousKGNNGraph(nodes={"User": [], "Product": []})

        # Duplicate node ID within type
        with self.assertRaises(ValueError):
            HeterogeneousKGNNGraph(nodes={"User": ["u1", "u1"], "Product": ["p1"]})

        # Non-string node ID
        with self.assertRaises(ValueError):
            HeterogeneousKGNNGraph(nodes={"User": [123], "Product": ["p1"]})  # type: ignore

        # Empty string node ID
        with self.assertRaises(ValueError):
            HeterogeneousKGNNGraph(nodes={"User": [""], "Product": ["p1"]})

    def test_invalid_relations_and_unknown_endpoints_rejected(self):
        nodes = {"User": ["u1"], "Product": ["p1"]}

        # Unknown relation
        with self.assertRaises(ValueError):
            HeterogeneousKGNNGraph(nodes=nodes, edges={"bought_together": [("p1", "p1")]})

        # Unknown source node
        with self.assertRaises((UnknownIdError, ValueError)):
            HeterogeneousKGNNGraph(nodes=nodes, edges={"rated": [("u_unknown", "p1")]})

        # Unknown destination node
        with self.assertRaises((UnknownIdError, ValueError)):
            HeterogeneousKGNNGraph(nodes=nodes, edges={"rated": [("u1", "p_unknown")]})

        # Duplicate edge in explicit relation
        with self.assertRaises(ValueError):
            HeterogeneousKGNNGraph(nodes=nodes, edges={"rated": [("u1", "p1"), ("u1", "p1")]})

        # Mismatched node type endpoints (e.g. User in belongs_to)
        with self.assertRaises((UnknownIdError, ValueError)):
            HeterogeneousKGNNGraph(
                nodes={"User": ["u1"], "Product": ["p1"], "CategoryPath": ["c1"]},
                edges={"belongs_to": [("u1", "c1")]},
            )

    def test_build_heterogeneous_graph_rejects_empty_positive_training_context(self):
        records = [("u_low", "p_low", 2.0), ("u_other", "p_other", 3.0)]
        context = TrainingContext(records, condition="rating_ge_4")
        with self.assertRaises(ValueError):
            build_heterogeneous_graph(context=context, metadata={})

    def test_build_heterogeneous_graph_from_context_and_heldout_isolation(self):
        # Training interactions: u1 -> p1, u2 -> p2
        # Held-out interaction: u1 -> p3 (must NOT appear in graph)
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        context = TrainingContext(records, condition="all_ratings")
        metadata = {
            "p1": {"categories": ["c1", "c2"], "brand": "b1"},
            "p2": {"categories": ["c2"], "brand": "b2"},
            "p3": {"categories": ["c3"], "brand": "b3"},  # Cold/heldout product metadata must be filtered
        }
        graph = build_heterogeneous_graph(context=context, metadata=metadata)

        self.assertEqual(graph.nodes["User"], ["u1", "u2"])
        self.assertEqual(graph.nodes["Product"], ["p1", "p2"])
        self.assertNotIn("p3", graph.nodes["Product"])

        # Cold item metadata nodes not added
        self.assertNotIn("c3", graph.nodes["CategoryPath"])
        self.assertNotIn("b3", graph.nodes["Brand"])

        # Interaction edges are binary: rated and rated_by exist, held-out p3 does not exist
        self.assertEqual(set(graph.edges["rated"]), {("u1", "p1"), ("u2", "p2")})
        self.assertEqual(set(graph.edges["rated_by"]), {("p1", "u1"), ("p2", "u2")})


class HeterogeneousKGNNBatchMaskingTests(unittest.TestCase):
    """Test batch positive-edge masking, preservation of unrelated/metadata/multi-hop edges, and unmasked cache bypass."""

    def setUp(self):
        self.nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": ["c1"],
            "Brand": ["b1"],
        }
        self.edges = {
            "rated": [
                ("u1", "p1"),  # To be masked
                ("u1", "p2"),  # Unrelated interaction for u1
                ("u2", "p1"),  # Unrelated interaction for p1
                ("u2", "p3"),  # Unrelated interaction
            ],
            "belongs_to": [("p1", "c1"), ("p2", "c1")],
            "has_brand": [("p1", "b1")],
        }
        self.graph = HeterogeneousKGNNGraph(nodes=self.nodes, edges=self.edges, auto_inverse=True)

    def test_batch_masking_removes_forward_and_reverse_edges(self):
        masked_graph = mask_heterogeneous_batch_edges(
            graph=self.graph,
            user_ids=["u1"],
            positive_item_ids=["p1"],
        )

        # Direct forward and reverse edges removed
        self.assertNotIn(("u1", "p1"), masked_graph.edges["rated"])
        self.assertNotIn(("p1", "u1"), masked_graph.edges["rated_by"])

        # Unrelated interactions preserved
        self.assertIn(("u1", "p2"), masked_graph.edges["rated"])
        self.assertIn(("p2", "u1"), masked_graph.edges["rated_by"])
        self.assertIn(("u2", "p1"), masked_graph.edges["rated"])
        self.assertIn(("p1", "u2"), masked_graph.edges["rated_by"])
        self.assertIn(("u2", "p3"), masked_graph.edges["rated"])
        self.assertIn(("p3", "u2"), masked_graph.edges["rated_by"])

        # Metadata edges strictly preserved
        self.assertIn(("p1", "c1"), masked_graph.edges["belongs_to"])
        self.assertIn(("c1", "p1"), masked_graph.edges["contains_product"])
        self.assertIn(("p2", "c1"), masked_graph.edges["belongs_to"])
        self.assertIn(("c1", "p2"), masked_graph.edges["contains_product"])
        self.assertIn(("p1", "b1"), masked_graph.edges["has_brand"])
        self.assertIn(("b1", "p1"), masked_graph.edges["brand_of"])

        # Multi-hop path u1 -> p2 -> c1 -> p1 is preserved
        self.assertIn("p2", masked_graph.incoming_neighbors["rated_by"]["u1"])
        self.assertIn("c1", masked_graph.incoming_neighbors["contains_product"]["p2"])
        self.assertIn("p2", masked_graph.incoming_neighbors["belongs_to"]["c1"])
        self.assertIn("c1", masked_graph.incoming_neighbors["contains_product"]["p1"])

    def test_duplicate_supervised_pairs_handled_and_active_relations_updated(self):
        # Mask with duplicate batch pairs
        masked_graph = mask_heterogeneous_batch_edges(
            graph=self.graph,
            user_ids=["u1", "u1"],
            positive_item_ids=["p1", "p1"],
        )
        self.assertNotIn(("u1", "p1"), masked_graph.edges["rated"])
        self.assertNotIn(("p1", "u1"), masked_graph.edges["rated_by"])

        # If we mask both p1 and p2 for u1:
        masked_all_u1 = mask_heterogeneous_batch_edges(
            graph=self.graph,
            user_ids=["u1", "u1"],
            positive_item_ids=["p1", "p2"],
        )
        # u1 has zero incoming rated_by edges left
        self.assertEqual(masked_all_u1.incoming_neighbors["rated_by"].get("u1", []), [])
        # Node sets remain unchanged
        self.assertEqual(masked_all_u1.nodes["User"], self.graph.nodes["User"])


class HeterogeneousOperatorHandCalculationTests(unittest.TestCase):
    """Test exact hand-calculated forward pass, mixture weights, zero-neighbors, and activations."""

    def test_tiny_hand_calculated_two_relation_fixture(self):
        """Verify hand-calculated 64-dim fixture with 2 active relations:
        Target product p1 has:
        - 1 user incoming neighbor along 'rated': u1 with h_u1 = [1.0, 0, ..., 0]
        - 2 category incoming neighbors along 'contains_product': c1 with [2.0, 0, ...], c2 with [4.0, 0, ...]
        Within-relation means:
          rated: [1.0, 0, ...]
          contains_product: [(2.0 + 4.0)/2, 0, ...] = [3.0, 0, ...]
        Relation transforms:
          W_rated = 2.0 * I  => m_rated = [2.0, 0, ...]
          W_contains_product = 0.5 * I => m_c = [1.5, 0, ...]
        Cross-relation unweighted mean:
          mean_messages = (m_rated + m_c) / 2 = [1.75, 0, ...]
        Self term:
          h_p1 = [0.5, 0, ...]
          W_self = 1.0 * I
          b_self = [0.25, 0, ...]
          self_term = [0.5 + 0.25, 0, ...] = [0.75, 0, ...]
        Pre-activation:
          z_p1^(1) = self_term + mean_messages = [0.75 + 1.75, 0, ...] = [2.5, 0, ...]
        Layer 1 activation:
          ReLU(z_p1^(1)) = [2.5, 0, ...]
        """
        dim = 64
        nodes = {
            "User": ["u1"],
            "Product": ["p1"],
            "CategoryPath": ["c1", "c2"],
            "Brand": [],
        }
        edges = {
            "rated": [("u1", "p1")],
            "contains_product": [("c1", "p1"), ("c2", "p1")],
        }
        graph = HeterogeneousKGNNGraph(nodes=nodes, edges=edges, auto_inverse=False)

        # Base embeddings
        base_embeddings: Dict[str, Dict[str, List[float]]] = {
            "User": {"u1": [1.0 if k == 0 else 0.0 for k in range(dim)]},
            "Product": {"p1": [0.5 if k == 0 else 0.0 for k in range(dim)]},
            "CategoryPath": {
                "c1": [2.0 if k == 0 else 0.0 for k in range(dim)],
                "c2": [4.0 if k == 0 else 0.0 for k in range(dim)],
            },
            "Brand": {},
        }

        # Parameters
        relation_matrices: Dict[int, Dict[str, List[List[float]]]] = {0: {}, 1: {}}
        self_matrices: Dict[int, Dict[str, List[List[float]]]] = {0: {}, 1: {}}
        self_biases: Dict[int, Dict[str, List[float]]] = {0: {}, 1: {}}

        for l in (0, 1):
            for r in RELATION_NAMES:
                scale = 2.0 if r == "rated" else (0.5 if r == "contains_product" else 1.0)
                relation_matrices[l][r] = [
                    [scale if i == j else 0.0 for j in range(dim)]
                    for i in range(dim)
                ]
            for t in NODE_TYPES:
                self_matrices[l][t] = [
                    [1.0 if i == j else 0.0 for j in range(dim)]
                    for i in range(dim)
                ]
                self_biases[l][t] = [0.25 if (k == 0 and t == "Product") else 0.0 for k in range(dim)]

        params = HeterogeneousParameters(
            relation_matrices=relation_matrices,
            self_matrices=self_matrices,
            self_biases=self_biases,
            embedding_dimension=dim,
            layers=2,
        )

        propagated = propagate_heterogeneous_kgnn(
            graph=graph,
            node_base_embeddings=base_embeddings,
            parameters=params,
        )

        # Verify Layer 2 output (and Layer 1 intermediate logic)
        # Note: in layer 1, p1 had z_p1^(1) = 2.5, which is positive so ReLU keeps it at 2.5.
        # Then in layer 2:
        # User u1 has no incoming neighbors in graph, so for u1:
        # layer 1: self_term = 1.0 * 1.0 + 0 = 1.0; ReLU(1.0) = 1.0
        # Categories c1, c2 have no incoming neighbors:
        # layer 1: c1 = 2.0, c2 = 4.0
        # Then in layer 2, incoming to p1 are u1 (1.0) and c1 (2.0), c2 (4.0).
        # So layer 2 pre-activation for p1:
        # m_rated^(1) = 2.0 * 1.0 = 2.0
        # m_c^(1) = 0.5 * 3.0 = 1.5
        # mean_msg = 1.75
        # self_term = 1.0 * 2.5 + 0.25 = 2.75
        # z_p1^(2) = 2.75 + 1.75 = 4.5
        # Layer 2 identity: h_p1^(2) = 4.5!
        self.assertAlmostEqual(propagated["Product"]["p1"][0], 4.5, places=9)
        for k in range(1, dim):
            self.assertAlmostEqual(propagated["Product"]["p1"][k], 0.0, places=9)

    def test_increasing_neighbor_count_does_not_change_relation_mixture_weights(self):
        """Increasing neighbor count in one relation from 2 to 4 while preserving its mean
        does not alter the cross-relation mean or mixture weight.
        """
        dim = 64
        # Case A: 2 category neighbors c1=2.0, c2=4.0 (mean = 3.0)
        # Case B: 4 category neighbors c1=2.0, c2=4.0, c3=3.0, c4=3.0 (mean = 3.0)
        nodes_a = {"User": ["u1"], "Product": ["p1"], "CategoryPath": ["c1", "c2"], "Brand": []}
        edges_a = {"rated": [("u1", "p1")], "contains_product": [("c1", "p1"), ("c2", "p1")]}
        graph_a = HeterogeneousKGNNGraph(nodes=nodes_a, edges=edges_a, auto_inverse=False)

        nodes_b = {"User": ["u1"], "Product": ["p1"], "CategoryPath": ["c1", "c2", "c3", "c4"], "Brand": []}
        edges_b = {
            "rated": [("u1", "p1")],
            "contains_product": [("c1", "p1"), ("c2", "p1"), ("c3", "p1"), ("c4", "p1")],
        }
        graph_b = HeterogeneousKGNNGraph(nodes=nodes_b, edges=edges_b, auto_inverse=False)

        base_emb_a: Dict[str, Dict[str, List[float]]] = {
            "User": {"u1": [1.0 if k == 0 else 0.0 for k in range(dim)]},
            "Product": {"p1": [0.5 if k == 0 else 0.0 for k in range(dim)]},
            "CategoryPath": {
                "c1": [2.0 if k == 0 else 0.0 for k in range(dim)],
                "c2": [4.0 if k == 0 else 0.0 for k in range(dim)],
            },
            "Brand": {},
        }
        base_emb_b: Dict[str, Dict[str, List[float]]] = {
            "User": {"u1": [1.0 if k == 0 else 0.0 for k in range(dim)]},
            "Product": {"p1": [0.5 if k == 0 else 0.0 for k in range(dim)]},
            "CategoryPath": {
                "c1": [2.0 if k == 0 else 0.0 for k in range(dim)],
                "c2": [4.0 if k == 0 else 0.0 for k in range(dim)],
                "c3": [3.0 if k == 0 else 0.0 for k in range(dim)],
                "c4": [3.0 if k == 0 else 0.0 for k in range(dim)],
            },
            "Brand": {},
        }

        params = initialize_heterogeneous_parameters(seed=123)

        out_a = propagate_heterogeneous_kgnn(graph=graph_a, node_base_embeddings=base_emb_a, parameters=params)
        out_b = propagate_heterogeneous_kgnn(graph=graph_b, node_base_embeddings=base_emb_b, parameters=params)

        for k in range(dim):
            self.assertAlmostEqual(out_a["Product"]["p1"][k], out_b["Product"]["p1"][k], places=9)

    def test_zero_neighbor_node_receives_self_term_only(self):
        dim = 64
        nodes = {"User": ["u_iso"], "Product": [], "CategoryPath": [], "Brand": []}
        graph = HeterogeneousKGNNGraph(nodes=nodes, edges={}, auto_inverse=False)

        base_emb = {"User": {"u_iso": [2.0 if k == 0 else 0.0 for k in range(dim)]}, "Product": {}, "CategoryPath": {}, "Brand": {}}

        params = initialize_heterogeneous_parameters(seed=42)
        # Set self biases to non-zero values for test
        for l in (0, 1):
            params.self_biases[l]["User"] = [1.0 for _ in range(dim)]

        propagated = propagate_heterogeneous_kgnn(graph=graph, node_base_embeddings=base_emb, parameters=params)

        # Hand calculate:
        # Layer 0 -> 1:
        # z_1 = W_self^(0) * [2, 0, ...] + [1, 1, ...]
        w_self_0 = params.self_matrices[0]["User"]
        expected_z1 = [2.0 * w_self_0[i][0] + 1.0 for i in range(dim)]
        expected_h1 = [max(0.0, val) for val in expected_z1]

        # Layer 1 -> 2:
        # z_2 = W_self^(1) * h1 + [1, 1, ...]
        w_self_1 = params.self_matrices[1]["User"]
        expected_z2 = [
            math.fsum(w_self_1[i][j] * expected_h1[j] for j in range(dim)) + 1.0
            for i in range(dim)
        ]
        # Layer 2 identity: h2 = z2
        for k in range(dim):
            self.assertAlmostEqual(propagated["User"]["u_iso"][k], expected_z2[k], places=9)

    def test_activation_order_layer1_relu_layer2_identity(self):
        dim = 64
        nodes = {"User": ["u1"], "Product": [], "CategoryPath": [], "Brand": []}
        graph = HeterogeneousKGNNGraph(nodes=nodes, edges={}, auto_inverse=False)

        base_emb = {"User": {"u1": [1.0 for _ in range(dim)]}, "Product": {}, "CategoryPath": {}, "Brand": {}}

        # Construct parameters where:
        # Layer 0 produces negative values: W_self^(0) = 0, b_self^(0) = -5.0
        # Since Layer 0 uses ReLU, h^(1) must become max(0, -5.0) = 0.0!
        # Layer 1 produces negative values: W_self^(1) = 0, b_self^(1) = -3.0
        # Since Layer 1 uses identity, h^(2) must be -3.0 (NOT clamped)!
        relation_matrices: Dict[int, Dict[str, List[List[float]]]] = {0: {}, 1: {}}
        self_matrices: Dict[int, Dict[str, List[List[float]]]] = {0: {}, 1: {}}
        self_biases: Dict[int, Dict[str, List[float]]] = {0: {}, 1: {}}

        for l in (0, 1):
            for r in RELATION_NAMES:
                relation_matrices[l][r] = [[0.0 for _ in range(dim)] for _ in range(dim)]
            for t in NODE_TYPES:
                self_matrices[l][t] = [[0.0 for _ in range(dim)] for _ in range(dim)]
                self_biases[l][t] = [-5.0 if l == 0 else -3.0 for _ in range(dim)]

        params = HeterogeneousParameters(
            relation_matrices=relation_matrices,
            self_matrices=self_matrices,
            self_biases=self_biases,
            embedding_dimension=dim,
            layers=2,
        )

        propagated = propagate_heterogeneous_kgnn(graph=graph, node_base_embeddings=base_emb, parameters=params)

        for k in range(dim):
            # Proves Layer 2 is identity, preserving negative values:
            self.assertAlmostEqual(propagated["User"]["u1"][k], -3.0, places=9)


class HeterogeneousInitializationTests(unittest.TestCase):
    """Test parameter and embedding initialization distributions, gain, and reproducibility."""

    def test_parameter_initialization_bounds_and_biases(self):
        params = initialize_heterogeneous_parameters(seed=2026)
        d = 64
        limit = math.sqrt(3.0 / d)  # ~0.216506

        for l in range(2):
            for r in RELATION_NAMES:
                mat = params.relation_matrices[l][r]
                self.assertEqual(len(mat), d)
                for row in mat:
                    self.assertEqual(len(row), d)
                    for val in row:
                        self.assertGreaterEqual(val, -limit)
                        self.assertLessEqual(val, limit)

            for t in NODE_TYPES:
                mat = params.self_matrices[l][t]
                self.assertEqual(len(mat), d)
                for row in mat:
                    self.assertEqual(len(row), d)
                    for val in row:
                        self.assertGreaterEqual(val, -limit)
                        self.assertLessEqual(val, limit)

                bias = params.self_biases[l][t]
                self.assertEqual(len(bias), d)
                for val in bias:
                    self.assertEqual(val, 0.0)

        # Deterministic with seed
        params_2 = initialize_heterogeneous_parameters(seed=2026)
        self.assertEqual(
            params.relation_matrices[0]["rated"][0][0],
            params_2.relation_matrices[0]["rated"][0][0],
        )

    def test_node_embedding_initialization_reproducibility(self):
        nodes = {"User": ["u1", "u2"], "Product": ["p1"], "CategoryPath": [], "Brand": []}
        graph = HeterogeneousKGNNGraph(nodes=nodes, edges={}, auto_inverse=False)

        emb1 = initialize_node_embeddings(graph=graph, seed=42)
        emb2 = initialize_node_embeddings(graph=graph, seed=42)

        self.assertEqual(emb1["User"]["u1"], emb2["User"]["u1"])
        self.assertEqual(len(emb1["User"]["u1"]), 64)
        for val in emb1["User"]["u1"]:
            self.assertTrue(math.isfinite(val))


class HeterogeneousKGNNScorerTests(unittest.TestCase):
    """Test dot-product decoding, empty-history popularity fallback, Evaluator integration, and errors."""

    def setUp(self):
        records = [
            ("u_warm", "p1", 5.0),
            ("u_empty", "p2", 2.0),
            ("u_other", "p2", 5.0),
            ("u_other", "p3", 5.0),
        ]
        self.context = TrainingContext(records, condition="rating_ge_4")
        self.user_emb = {
            "u_warm": [1.0 if k == 0 else 0.0 for k in range(64)],
            "u_other": [0.5 if k == 0 else 0.0 for k in range(64)],
        }
        self.prod_emb = {
            "p1": [2.0 if k == 0 else 0.0 for k in range(64)],
            "p2": [0.5 if k == 0 else 0.0 for k in range(64)],
            "p3": [-1.0 if k == 0 else 0.0 for k in range(64)],
        }

    def test_score_candidates_dot_product_and_fallback(self):
        scorer = HeterogeneousKGNNScorer(
            context=self.context,
            user_embeddings=self.user_emb,
            product_embeddings=self.prod_emb,
        )

        # Personalized user
        scores, status, reason = scorer.score_candidates("u_warm", ["p1", "p2", "p3"])
        self.assertEqual(status, "personalized")
        self.assertIsNone(reason)
        self.assertAlmostEqual(scores["p1"], 2.0)
        self.assertAlmostEqual(scores["p2"], 0.5)
        self.assertAlmostEqual(scores["p3"], -1.0)

        # Empty positive history user routes to popularity fallback
        fb_scores, fb_status, fb_reason = scorer.score_candidates("u_empty", ["p1", "p2", "p3"])
        self.assertEqual(fb_status, "fallback")
        self.assertEqual(fb_reason, "empty_positive_history")
        self.assertAlmostEqual(fb_scores["p1"], float(self.context.popularity.get("p1", 0)))
        self.assertAlmostEqual(fb_scores["p2"], float(self.context.popularity.get("p2", 0)))
        self.assertAlmostEqual(fb_scores["p3"], float(self.context.popularity.get("p3", 0)))

    def test_scorer_failure_conditions(self):
        scorer = HeterogeneousKGNNScorer(
            context=self.context,
            user_embeddings=self.user_emb,
            product_embeddings=self.prod_emb,
        )

        # Unknown user
        with self.assertRaises(ScoringFailureError):
            scorer.score_candidates("u_unknown", ["p1"])

        # Missing candidate item embedding
        with self.assertRaises(ScoringFailureError):
            scorer.score_candidates("u_warm", ["p_missing"])

        # Corrupt / non-finite embedding on init
        with self.assertRaises(ValueError):
            HeterogeneousKGNNScorer(
                context=self.context,
                user_embeddings={"u_warm": [float("nan")] * 64, "u_other": [0.0] * 64},
                product_embeddings=self.prod_emb,
            )

    def test_full_evaluator_integration(self):
        scorer = HeterogeneousKGNNScorer(
            context=self.context,
            user_embeddings=self.user_emb,
            product_embeddings=self.prod_emb,
        )
        evaluator = Evaluator(self.context)
        validation_targets = [
            ("u_warm", "p2", 5.0),
            ("u_empty", "p1", 5.0),
        ]
        report = evaluator.evaluate(validation_targets, scorer, split="validation")
        self.assertEqual(report["audit"]["eligible_users"], 2)
        self.assertEqual(report["subgroups"]["personalized"]["count"], 1)
        self.assertEqual(report["subgroups"]["fallback"]["count"], 1)
        self.assertEqual(report["subgroups"]["fallback"]["by_reason"]["empty_positive_history"]["count"], 1)
        self.assertIn("NDCG@10", report["overall_metrics"])
        self.assertIn("HitRate@10", report["overall_metrics"])
        self.assertIn("CatalogCoverage@10", report["overall_metrics"])
        self.assertTrue(math.isfinite(report["overall_metrics"]["NDCG@10"]))


class HeterogeneousBPRBatchLossTests(unittest.TestCase):
    """Test composed BPR loss on batch-masked graph, endpoint regularization, and harness guards."""

    def setUp(self):
        self.dim = 64
        self.nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": ["c1"],
            "Brand": ["b1"],
        }
        self.edges = {
            "rated": [("u1", "p1"), ("u2", "p2")],
            "belongs_to": [("p1", "c1")],
            "has_brand": [("p1", "b1")],
        }
        self.graph = HeterogeneousKGNNGraph(nodes=self.nodes, edges=self.edges, auto_inverse=True)
        self.params = initialize_heterogeneous_parameters(seed=42)
        self.base_emb = initialize_node_embeddings(self.graph, seed=42)

    def test_compute_heterogeneous_bpr_batch_loss_success(self):
        loss_dict = compute_heterogeneous_bpr_batch_loss(
            graph=self.graph,
            parameters=self.params,
            base_node_embeddings=self.base_emb,
            batch_user_ids=["u1"],
            batch_positive_item_ids=["p1"],
            batch_negative_item_ids=["p3"],
            reg_coefficient=1e-5,
        )
        self.assertIn("total_loss", loss_dict)
        self.assertIn("bpr_loss", loss_dict)
        self.assertIn("reg_loss", loss_dict)
        self.assertEqual(loss_dict["batch_size"], 1.0)
        self.assertAlmostEqual(loss_dict["total_loss"], loss_dict["bpr_loss"] + loss_dict["reg_loss"], places=9)
        self.assertTrue(math.isfinite(loss_dict["total_loss"]))
        self.assertGreater(loss_dict["bpr_loss"], 0.0)

    def test_endpoint_regularization_targets_user_and_product_only(self):
        # Base embeddings for u1, p1, p3
        loss_dict = compute_heterogeneous_bpr_batch_loss(
            graph=self.graph,
            parameters=self.params,
            base_node_embeddings=self.base_emb,
            batch_user_ids=["u1"],
            batch_positive_item_ids=["p1"],
            batch_negative_item_ids=["p3"],
            reg_coefficient=1e-5,
        )
        # Expected endpoint reg: 1e-5 * mean(norm_sq(u1), norm_sq(p1), norm_sq(p3))
        norm_u1 = sum(x * x for x in self.base_emb["User"]["u1"])
        norm_p1 = sum(x * x for x in self.base_emb["Product"]["p1"])
        norm_p3 = sum(x * x for x in self.base_emb["Product"]["p3"])
        expected_reg = 1e-5 * ((norm_u1 + norm_p1 + norm_p3) / 3.0)
        self.assertAlmostEqual(loss_dict["reg_loss"], expected_reg, places=9)

    def test_empty_batch_and_guard_rejections(self):
        # Empty batch
        with self.assertRaises(ValueError):
            compute_heterogeneous_bpr_batch_loss(
                graph=self.graph,
                parameters=self.params,
                base_node_embeddings=self.base_emb,
                batch_user_ids=[],
                batch_positive_item_ids=[],
                batch_negative_item_ids=[],
            )

        # TrainingHarnessGuard prevents real-data training
        guard = TrainingHarnessGuard(is_real_data=True)
        with self.assertRaises(RuntimeError):
            guard.validate_batch_trainable(valid_examples_count=10)

        # Guard requires allow_synthetic_updates for synthetic validation
        guard_synth = TrainingHarnessGuard(allow_synthetic_updates=False, is_real_data=False)
        with self.assertRaises(RuntimeError):
            guard_synth.validate_batch_trainable(valid_examples_count=10)


class CheckpointAndImportCompatibilityTests(unittest.TestCase):
    """Test checkpoint contracts, provenance validation, and re-export compatibility."""

    def test_checkpoint_manifest_heterogeneous_kgnn(self):
        manifest = create_checkpoint_manifest(
            method="Heterogeneous KG+GNN",
            category="Musical_Instruments",
            feedback_condition="all_ratings",
            seed=42,
        )
        self.assertEqual(manifest["method"], "Heterogeneous KG+GNN")
        self.assertFalse(manifest["training_authorized"])
        self.assertFalse(manifest["profiling_authorized"])
        self.assertEqual(manifest["deployment_decision"], "NOT_AUTHORIZED")
        self.assertTrue(validate_checkpoint_manifest(manifest))

    def test_import_compatibility_across_modules(self):
        # Ensure classes and functions are identical across scripts.learned_models,
        # learned_models, recommendation_core, and scripts.recommendation_core
        self.assertIs(learned_models.HeterogeneousKGNNConfig, scripts.learned_models.HeterogeneousKGNNConfig)
        self.assertIs(learned_models.HeterogeneousKGNNGraph, scripts.learned_models.HeterogeneousKGNNGraph)
        self.assertIs(learned_models.HeterogeneousKGNNScorer, scripts.learned_models.HeterogeneousKGNNScorer)
        self.assertIs(recommendation_core.HeterogeneousKGNNConfig, learned_models.HeterogeneousKGNNConfig)
        self.assertIs(recommendation_core.HeterogeneousKGNNGraph, learned_models.HeterogeneousKGNNGraph)
        self.assertIs(recommendation_core.HeterogeneousKGNNScorer, learned_models.HeterogeneousKGNNScorer)
        self.assertIs(recommendation_core.build_heterogeneous_graph, learned_models.build_heterogeneous_graph)
        self.assertIs(recommendation_core.mask_heterogeneous_batch_edges, learned_models.mask_heterogeneous_batch_edges)
        self.assertIs(recommendation_core.propagate_heterogeneous_kgnn, learned_models.propagate_heterogeneous_kgnn)
        self.assertIs(recommendation_core.compute_heterogeneous_bpr_batch_loss, learned_models.compute_heterogeneous_bpr_batch_loss)


if __name__ == "__main__":
    unittest.main()
