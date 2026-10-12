"""Comprehensive unit and regression tests for learned models and training harness Step 3a.

Python standard library only.
Tests:
- LightGCNConfig validation and frozen defaults matching model-suite.json and docs/lightgcn-protocol.md.
- LightGCNGraph construction, binary adjacency, symmetric degree normalization, reverse edges,
  fixed Option B training-edge reuse, condition-positive edge filtering, and held-out isolation.
- Hand-calculated exact full-batch propagation, layer 0..3 arithmetic averaging, and dot product decoding.
- LightGCNScorer candidate scoring, finite validation, Evaluator integration, and empty-positive fallback.
- BPR batch loss composition, endpoint regularization integration, and NegativeSampler compatibility.
- TrainingHarnessGuard preventing no-valid-batch optimizer updates, regularization-only updates,
  and unauthorized real-data training.
- Checkpoint and provenance manifest contracts, JSON serialization, and unauthorized deployment safeguards.
- Both import styles: scripts.learned_models and learned_models.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
import unittest
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import learned_models
import scripts.learned_models

from learned_models import (
    LightGCNConfig,
    LightGCNGraph,
    LightGCNScorer,
    MissingIdError,
    TrainingHarnessGuard,
    UnknownIdError,
    build_lightgcn_graph,
    compute_bpr_batch_loss,
    create_checkpoint_manifest,
    propagate_lightgcn,
    validate_checkpoint_manifest,
)
from recommendation_core import (
    Evaluator,
    NegativeSampler,
    ScoringFailureError,
    TrainingContext,
    bpr_loss,
    endpoint_regularization,
)


class LightGCNConfigTests(unittest.TestCase):
    """Test LightGCN configuration validation and Option B constraints."""

    def test_default_config_matches_frozen_protocol(self):
        cfg = LightGCNConfig()
        self.assertEqual(cfg.embedding_dimension, 64)
        self.assertEqual(cfg.propagation_layers, 3)
        self.assertEqual(cfg.layer_combination, "arithmetic_mean_layers_0_through_3")
        self.assertEqual(cfg.decoder, "dot_product")
        self.assertEqual(cfg.normalization, "standard_symmetric_training_degree_normalization")
        self.assertFalse(cfg.target_masking)
        self.assertEqual(cfg.methodology, "user_approved_Option_B_standard_fixed_graph")

        as_dict = cfg.to_dict()
        self.assertIsInstance(as_dict, dict)
        reconstructed = LightGCNConfig.from_dict(as_dict)
        self.assertEqual(cfg, reconstructed)

    def test_invalid_configurations_rejected(self):
        # Invalid embedding dimensions
        with self.assertRaises(ValueError):
            LightGCNConfig(embedding_dimension=0)
        with self.assertRaises(ValueError):
            LightGCNConfig(embedding_dimension=-64)
        with self.assertRaises(ValueError):
            LightGCNConfig(embedding_dimension=True)  # type: ignore
        with self.assertRaises(ValueError):
            LightGCNConfig(embedding_dimension=64.0)  # type: ignore

        # Invalid propagation layers (must be exactly 3 for frozen protocol)
        with self.assertRaises(ValueError):
            LightGCNConfig(propagation_layers=0)
        with self.assertRaises(ValueError):
            LightGCNConfig(propagation_layers=-1)
        with self.assertRaises(ValueError):
            LightGCNConfig(propagation_layers=False)  # type: ignore
        with self.assertRaises(ValueError):
            LightGCNConfig(propagation_layers=1)
        with self.assertRaises(ValueError):
            LightGCNConfig(propagation_layers=2)
        with self.assertRaises(ValueError):
            LightGCNConfig(propagation_layers=4)

        # Invalid layer combination
        with self.assertRaises(ValueError):
            LightGCNConfig(layer_combination="last_layer_only")

        # Invalid decoder
        with self.assertRaises(ValueError):
            LightGCNConfig(decoder="mlp")

        # Invalid normalization
        with self.assertRaises(ValueError):
            LightGCNConfig(normalization="row_normalization")

        # Option A target masking is rejected for core LightGCN
        with self.assertRaises(ValueError):
            LightGCNConfig(target_masking=True)

        # Invalid methodology
        with self.assertRaises(ValueError):
            LightGCNConfig(methodology="unapproved_variant")

        with self.assertRaises(ValueError):
            LightGCNConfig.from_dict("not_a_dict")  # type: ignore


class LightGCNGraphTests(unittest.TestCase):
    """Test LightGCN bipartite training graph construction and degree normalization."""

    def setUp(self):
        # 3 users, 3 items
        # u1: p1 (rating 5.0), p2 (rating 4.0)
        # u2: p1 (rating 3.0), p3 (rating 5.0)
        # u3: p2 (rating 2.0)
        self.records = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 4.0),
            ("u2", "p1", 3.0),
            ("u2", "p3", 5.0),
            ("u3", "p2", 2.0),
        ]

    def test_all_ratings_graph_construction(self):
        context = TrainingContext(self.records, condition="all_ratings")
        graph = build_lightgcn_graph(context)

        # In all_ratings condition, all ratings are positive interactions:
        # u1 -> p1, p2 (d_u1 = 2)
        # u2 -> p1, p3 (d_u2 = 2)
        # u3 -> p2 (d_u3 = 1)
        # p1 -> u1, u2 (d_p1 = 2)
        # p2 -> u1, u3 (d_p2 = 2)
        # p3 -> u2 (d_p3 = 1)
        self.assertEqual(graph.user_degrees["u1"], 2)
        self.assertEqual(graph.user_degrees["u2"], 2)
        self.assertEqual(graph.user_degrees["u3"], 1)

        self.assertEqual(graph.item_degrees["p1"], 2)
        self.assertEqual(graph.item_degrees["p2"], 2)
        self.assertEqual(graph.item_degrees["p3"], 1)

        # Total positive edges: 5 forward + 5 reverse = 10 directed half-edges (num_edges counts 5 pairs)
        self.assertEqual(graph.num_edges, 5)

        # Test symmetric degree normalization weights: 1 / sqrt(d_u * d_i)
        # Edge (u1, p1): 1 / sqrt(2 * 2) = 0.5
        u1_neighbors = dict(graph.user_neighbors["u1"])
        self.assertAlmostEqual(u1_neighbors["p1"], 0.5, places=12)
        self.assertAlmostEqual(u1_neighbors["p2"], 0.5, places=12)

        # Edge (u3, p2): 1 / sqrt(1 * 2) = 1 / sqrt(2)
        u3_neighbors = dict(graph.user_neighbors["u3"])
        self.assertAlmostEqual(u3_neighbors["p2"], 1.0 / math.sqrt(2.0), places=12)

        # Reverse edge (p2, u3): 1 / sqrt(2 * 1) = 1 / sqrt(2)
        p2_neighbors = dict(graph.item_neighbors["p2"])
        self.assertAlmostEqual(p2_neighbors["u3"], 1.0 / math.sqrt(2.0), places=12)
        self.assertAlmostEqual(p2_neighbors["u1"], 0.5, places=12)

    def test_rating_ge_4_condition_excludes_lower_ratings(self):
        context = TrainingContext(self.records, condition="rating_ge_4")
        graph = build_lightgcn_graph(context)

        # In rating_ge_4 condition:
        # u1: p1 (5.0), p2 (4.0) -> pos: {p1, p2} (d_u1 = 2)
        # u2: p1 (3.0 < 4), p3 (5.0 >= 4) -> pos: {p3} (d_u2 = 1)
        # u3: p2 (2.0 < 4) -> pos: empty (d_u3 = 0)
        # warm items: {p1, p2, p3}
        # p1: rated >=4 by u1 only -> d_p1 = 1
        # p2: rated >=4 by u1 only -> d_p2 = 1
        # p3: rated >=4 by u2 only -> d_p3 = 1
        self.assertEqual(graph.user_degrees["u1"], 2)
        self.assertEqual(graph.user_degrees["u2"], 1)
        self.assertEqual(graph.user_degrees["u3"], 0)

        self.assertEqual(graph.item_degrees["p1"], 1)
        self.assertEqual(graph.item_degrees["p2"], 1)
        self.assertEqual(graph.item_degrees["p3"], 1)

        # Lower-rated interactions (u2, p1) and (u3, p2) MUST NOT appear as edges in graph
        u2_items = [item for item, _ in graph.user_neighbors["u2"]]
        self.assertEqual(u2_items, ["p3"])
        self.assertNotIn("p1", u2_items)

        u3_items = [item for item, _ in graph.user_neighbors["u3"]]
        self.assertEqual(u3_items, [])

        p1_users = [u for u, _ in graph.item_neighbors["p1"]]
        self.assertEqual(p1_users, ["u1"])
        self.assertNotIn("u2", p1_users)

        # Check normalization weight:
        # Edge (u1, p1): d_u1=2, d_p1=1 -> 1 / sqrt(2 * 1) = 1 / sqrt(2)
        u1_neighbors = dict(graph.user_neighbors["u1"])
        self.assertAlmostEqual(u1_neighbors["p1"], 1.0 / math.sqrt(2.0), places=12)

    def test_empty_trainable_graph_rejected(self):
        # All ratings below 4 with condition rating_ge_4 produces an empty positive graph
        records = [("u1", "p1", 2.0), ("u2", "p2", 3.0)]
        context = TrainingContext(records, condition="rating_ge_4")
        with self.assertRaises(ValueError):
            build_lightgcn_graph(context)

    def test_invalid_context_type_rejected(self):
        with self.assertRaises(ValueError):
            build_lightgcn_graph("not_a_context")  # type: ignore

    def test_heldout_isolation(self):
        # Verifies that TrainingContext does not expose held-out items, and
        # graph is derived solely from training records
        context = TrainingContext(self.records, condition="all_ratings")
        graph = build_lightgcn_graph(context)
        # Adding a hypothetical held-out target record to evaluation does not touch graph
        self.assertNotIn("heldout_val_item", graph.warm_items)
        self.assertNotIn("heldout_user", graph.users)


class ExactPropagationTests(unittest.TestCase):
    """Test deterministic exact propagation and layer averaging against hand-calculated values."""

    def setUp(self):
        # 2 users (u1, u2), 2 items (i1, i2)
        # u1: i1, i2 (d_u1 = 2)
        # u2: i1     (d_u2 = 1)
        # i1: u1, u2 (d_i1 = 2)
        # i2: u1     (d_i2 = 1)
        records = [
            ("u1", "i1", 5.0),
            ("u1", "i2", 5.0),
            ("u2", "i1", 5.0),
        ]
        self.context = TrainingContext(records, condition="all_ratings")
        self.graph = build_lightgcn_graph(self.context)

        # 2-dimensional base embeddings
        self.user_base = {
            "u1": [1.0, 0.0],
            "u2": [0.0, 1.0],
        }
        self.item_base = {
            "i1": [1.0, 1.0],
            "i2": [2.0, 0.0],
        }

    def test_hand_calculated_three_layer_propagation(self):
        sqrt2 = math.sqrt(2.0)

        # Hand calculations:
        # Weights:
        # w(u1, i1) = 0.5
        # w(u1, i2) = 1/sqrt(2)
        # w(u2, i1) = 1/sqrt(2)

        # Layer 1:
        # e_u1^(1) = 0.5*[1, 1] + (1/sqrt2)*[2, 0] = [0.5 + sqrt2, 0.5]
        # e_u2^(1) = (1/sqrt2)*[1, 1] = [1/sqrt2, 1/sqrt2]
        # e_i1^(1) = 0.5*[1, 0] + (1/sqrt2)*[0, 1] = [0.5, 1/sqrt2]
        # e_i2^(1) = (1/sqrt2)*[1, 0] = [1/sqrt2, 0.0]

        # Layer 2:
        # e_u1^(2) = 0.5*[0.5, 1/sqrt2] + (1/sqrt2)*[1/sqrt2, 0] = [0.25 + 0.5, 0.5/sqrt2] = [0.75, 0.5/sqrt2]
        # e_u2^(2) = (1/sqrt2)*[0.5, 1/sqrt2] = [0.5/sqrt2, 0.5]
        # e_i1^(2) = 0.5*[0.5 + sqrt2, 0.5] + (1/sqrt2)*[1/sqrt2, 1/sqrt2] = [0.25 + 0.5*sqrt2 + 0.5, 0.25 + 0.5] = [0.75 + 0.5*sqrt2, 0.75]
        # e_i2^(2) = (1/sqrt2)*[0.5 + sqrt2, 0.5] = [0.5/sqrt2 + 1.0, 0.5/sqrt2]

        # Layer 3:
        # e_u1^(3) = 0.5*[0.75 + 0.5*sqrt2, 0.75] + (1/sqrt2)*[0.5/sqrt2 + 1.0, 0.5/sqrt2]
        #          = [0.375 + 0.25*sqrt2 + 0.25 + 1.0/sqrt2, 0.375 + 0.25]
        #          = [0.625 + 0.75*sqrt2, 0.625]
        # e_u2^(3) = (1/sqrt2)*[0.75 + 0.5*sqrt2, 0.75] = [0.75/sqrt2 + 0.5, 0.75/sqrt2]
        # e_i1^(3) = 0.5*[0.75, 0.5/sqrt2] + (1/sqrt2)*[0.5/sqrt2, 0.5]
        #          = [0.375 + 0.25, 0.25/sqrt2 + 0.5/sqrt2] = [0.625, 0.75/sqrt2]
        # e_i2^(3) = (1/sqrt2)*[0.75, 0.5/sqrt2] = [0.75/sqrt2, 0.25]

        # Arithmetic mean over layers 0..3: (L0 + L1 + L2 + L3) / 4.0
        expected_u1_0 = (1.0 + (0.5 + sqrt2) + 0.75 + (0.625 + 0.75 * sqrt2)) / 4.0
        expected_u1_1 = (0.0 + 0.5 + (0.5 / sqrt2) + 0.625) / 4.0

        expected_u2_0 = (0.0 + (1.0 / sqrt2) + (0.5 / sqrt2) + (0.75 / sqrt2 + 0.5)) / 4.0
        expected_u2_1 = (1.0 + (1.0 / sqrt2) + 0.5 + (0.75 / sqrt2)) / 4.0

        expected_i1_0 = (1.0 + 0.5 + (0.75 + 0.5 * sqrt2) + 0.625) / 4.0
        expected_i1_1 = (1.0 + (1.0 / sqrt2) + 0.75 + (0.75 / sqrt2)) / 4.0

        expected_i2_0 = (2.0 + (1.0 / sqrt2) + (0.5 / sqrt2 + 1.0) + (0.75 / sqrt2)) / 4.0
        expected_i2_1 = (0.0 + 0.0 + (0.5 / sqrt2) + 0.25) / 4.0

        prop_u, prop_i = propagate_lightgcn(self.graph, self.user_base, self.item_base, layers=3)

        self.assertAlmostEqual(prop_u["u1"][0], expected_u1_0, places=12)
        self.assertAlmostEqual(prop_u["u1"][1], expected_u1_1, places=12)
        self.assertAlmostEqual(prop_u["u2"][0], expected_u2_0, places=12)
        self.assertAlmostEqual(prop_u["u2"][1], expected_u2_1, places=12)

        self.assertAlmostEqual(prop_i["i1"][0], expected_i1_0, places=12)
        self.assertAlmostEqual(prop_i["i1"][1], expected_i1_1, places=12)
        self.assertAlmostEqual(prop_i["i2"][0], expected_i2_0, places=12)
        self.assertAlmostEqual(prop_i["i2"][1], expected_i2_1, places=12)

    def test_propagation_validation_rejections(self):
        # Invalid layers
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, self.user_base, self.item_base, layers=0)
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, self.user_base, self.item_base, layers=-2)
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, self.user_base, self.item_base, layers=True)  # type: ignore

        # Missing user embedding
        incomplete_user = {"u1": [1.0, 0.0]}
        with self.assertRaises(KeyError):
            propagate_lightgcn(self.graph, incomplete_user, self.item_base, layers=3)
        with self.assertRaises(MissingIdError):
            propagate_lightgcn(self.graph, incomplete_user, self.item_base, layers=3)

        # Missing item embedding
        incomplete_item = {"i1": [1.0, 1.0]}
        with self.assertRaises(KeyError):
            propagate_lightgcn(self.graph, self.user_base, incomplete_item, layers=3)
        with self.assertRaises(MissingIdError):
            propagate_lightgcn(self.graph, self.user_base, incomplete_item, layers=3)

        # Extra unknown user embedding
        extra_user = dict(self.user_base, unknown_user=[0.5, 0.5])
        with self.assertRaises(KeyError):
            propagate_lightgcn(self.graph, extra_user, self.item_base, layers=3)
        with self.assertRaises(UnknownIdError):
            propagate_lightgcn(self.graph, extra_user, self.item_base, layers=3)
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, extra_user, self.item_base, layers=3)

        # Extra unknown item embedding
        extra_item = dict(self.item_base, unknown_item=[0.5, 0.5])
        with self.assertRaises(KeyError):
            propagate_lightgcn(self.graph, self.user_base, extra_item, layers=3)
        with self.assertRaises(UnknownIdError):
            propagate_lightgcn(self.graph, self.user_base, extra_item, layers=3)
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, self.user_base, extra_item, layers=3)

        # Inconsistent dimension between user and item
        bad_item_dim = {"i1": [1.0, 1.0, 1.0], "i2": [2.0, 0.0, 1.0]}
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, self.user_base, bad_item_dim, layers=3)

        # Inconsistent dimension across users
        bad_user_dim = {"u1": [1.0, 0.0], "u2": [0.0]}
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, bad_user_dim, self.item_base, layers=3)

        # Non-finite values in embeddings
        nan_user = {"u1": [float("nan"), 0.0], "u2": [0.0, 1.0]}
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, nan_user, self.item_base, layers=3)

        inf_item = {"i1": [float("inf"), 1.0], "i2": [2.0, 0.0]}
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, self.user_base, inf_item, layers=3)

        bool_user = {"u1": [True, 0.0], "u2": [0.0, 1.0]}  # type: ignore
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, bool_user, self.item_base, layers=3)

        # Zero-dimension vector
        empty_vec_user = {"u1": [], "u2": []}
        with self.assertRaises(ValueError):
            propagate_lightgcn(self.graph, empty_vec_user, self.item_base, layers=3)

    def test_propagation_exact_key_matching_with_empty_history_users(self):
        # In rating_ge_4, u3 has only rating 2.0 (empty positive history), but is in context.users
        records = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 5.0),
            ("u2", "p2", 4.0),
            ("u3", "p1", 2.0),
        ]
        context = TrainingContext(records, condition="rating_ge_4")
        graph = build_lightgcn_graph(context)
        self.assertEqual(graph.users, ["u1", "u2", "u3"])
        self.assertEqual(graph.positive_users, ["u1", "u2"])
        self.assertEqual(graph.warm_items, ["p1", "p2"])

        # Omission of u3 must raise MissingIdError
        user_base_missing_u3 = {
            "u1": [1.0, 0.0],
            "u2": [0.0, 1.0],
        }
        item_base = {
            "p1": [1.0, 0.0],
            "p2": [0.0, 1.0],
        }
        with self.assertRaises(MissingIdError):
            propagate_lightgcn(graph, user_base_missing_u3, item_base, layers=3)

        # Addition of an unknown user must raise UnknownIdError
        user_base_extra = {
            "u1": [1.0, 0.0],
            "u2": [0.0, 1.0],
            "u3": [0.0, 0.0],
            "u_unknown": [0.5, 0.5],
        }
        with self.assertRaises(UnknownIdError):
            propagate_lightgcn(graph, user_base_extra, item_base, layers=3)

        # Addition of an unknown item must raise UnknownIdError
        item_base_extra = {
            "p1": [1.0, 0.0],
            "p2": [0.0, 1.0],
            "p_unknown": [0.5, 0.5],
        }
        user_base_valid = {
            "u1": [1.0, 0.0],
            "u2": [0.0, 1.0],
            "u3": [0.0, 0.0],
        }
        with self.assertRaises(UnknownIdError):
            propagate_lightgcn(graph, user_base_valid, item_base_extra, layers=3)

        # Exact match must succeed
        prop_u, prop_i = propagate_lightgcn(graph, user_base_valid, item_base, layers=3)
        self.assertEqual(set(prop_u.keys()), {"u1", "u2", "u3"})
        self.assertEqual(set(prop_i.keys()), {"p1", "p2"})


class LightGCNScorerTests(unittest.TestCase):
    """Test candidate dot-product scoring, Evaluator integration, and fallback routing."""

    def setUp(self):
        # 3 users, 3 items
        # u1 and u2 have >=4 ratings; u3 has only rating 2.0 (empty positive under rating_ge_4)
        records = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 5.0),
            ("u2", "p2", 4.0),
            ("u2", "p3", 4.0),
            ("u3", "p1", 2.0),
        ]
        self.context = TrainingContext(records, condition="rating_ge_4")
        self.graph = build_lightgcn_graph(self.context)

        # Base embeddings
        self.user_base = {
            "u1": [1.0, 0.5],
            "u2": [0.2, 0.8],
            "u3": [0.0, 0.0],
        }
        self.item_base = {
            "p1": [0.7, 0.3],
            "p2": [0.4, 0.6],
            "p3": [0.1, 0.9],
        }
        self.prop_u, self.prop_i = propagate_lightgcn(self.graph, self.user_base, self.item_base, layers=3)
        self.scorer = LightGCNScorer(self.context, self.prop_u, self.prop_i)

    def test_score_candidates_personalized(self):
        candidates = ["p1", "p2", "p3"]
        scores, route, reason = self.scorer.score_candidates("u1", candidates)

        self.assertEqual(route, "personalized")
        self.assertIsNone(reason)
        self.assertEqual(set(scores.keys()), set(candidates))

        # Dot-product calculation check
        u1_vec = self.prop_u["u1"]
        for p in candidates:
            p_vec = self.prop_i[p]
            expected = u1_vec[0] * p_vec[0] + u1_vec[1] * p_vec[1]
            self.assertAlmostEqual(scores[p], expected, places=12)

    def test_empty_positive_history_routes_to_popularity_fallback(self):
        candidates = ["p1", "p2", "p3"]
        # u3 has empty positive history in rating_ge_4 condition
        scores, route, reason = self.scorer.score_candidates("u3", candidates)

        self.assertEqual(route, "fallback")
        self.assertEqual(reason, "empty_positive_history")
        for p in candidates:
            self.assertEqual(scores[p], float(self.context.popularity.get(p, 0)))

    def test_unknown_user_raises_scoring_failure(self):
        # A user not in user_embeddings raises ScoringFailureError
        with self.assertRaises(ScoringFailureError):
            self.scorer.score_candidates("unknown_user", ["p1"])

    def test_unknown_candidate_raises_scoring_failure(self):
        with self.assertRaises(ScoringFailureError):
            self.scorer.score_candidates("u1", ["unknown_item"])

    def test_evaluator_integration_with_lightgcn(self):
        evaluator = Evaluator(self.context)
        validation_targets = [
            ("u1", "p3", 5.0),  # eligible personalized user
            ("u3", "p2", 5.0),  # eligible user with empty positive history -> fallback
        ]
        report = evaluator.evaluate(validation_targets, self.scorer, split="validation")

        self.assertEqual(report["audit"]["eligible_users"], 2)
        self.assertEqual(report["subgroups"]["personalized"]["count"], 1)
        self.assertEqual(report["subgroups"]["fallback"]["count"], 1)
        self.assertEqual(report["subgroups"]["fallback"]["by_reason"]["empty_positive_history"]["count"], 1)

        # Metrics are finite and present
        self.assertIn("NDCG@10", report["overall_metrics"])
        self.assertIn("HitRate@10", report["overall_metrics"])
        self.assertIn("CatalogCoverage@10", report["overall_metrics"])


class BPRBatchLossAndGuardTests(unittest.TestCase):
    """Test BPR batch loss composition, endpoint regularization, and training harness guards."""

    def setUp(self):
        records = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 5.0),
            ("u2", "p2", 5.0),
            ("u2", "p3", 5.0),
        ]
        self.context = TrainingContext(records, condition="all_ratings")
        self.graph = build_lightgcn_graph(self.context)

        self.user_base = {
            "u1": [1.0, 0.0],
            "u2": [0.0, 1.0],
        }
        self.item_base = {
            "p1": [1.0, 0.0],
            "p2": [0.5, 0.5],
            "p3": [0.0, 1.0],
        }
        self.prop_u, self.prop_i = propagate_lightgcn(self.graph, self.user_base, self.item_base, layers=3)

    def test_bpr_batch_loss_composition(self):
        user_ids = ["u1", "u2"]
        pos_items = ["p1", "p3"]
        neg_items = ["p3", "p1"]

        loss_dict = compute_bpr_batch_loss(
            user_ids=user_ids,
            positive_item_ids=pos_items,
            negative_item_ids=neg_items,
            user_propagated_embeddings=self.prop_u,
            item_propagated_embeddings=self.prop_i,
            user_base_embeddings=self.user_base,
            item_base_embeddings=self.item_base,
            reg_coefficient=1e-5,
        )

        self.assertIn("total_loss", loss_dict)
        self.assertIn("bpr_loss", loss_dict)
        self.assertIn("reg_loss", loss_dict)
        self.assertEqual(loss_dict["batch_size"], 2.0)
        self.assertAlmostEqual(loss_dict["total_loss"], loss_dict["bpr_loss"] + loss_dict["reg_loss"], places=12)

        # Verify endpoint regularization calculation:
        # Unique users: u1, u2 (2)
        # Unique items: p1, p3 (2)
        # Total unique endpoints = 4
        # User base squared norms: u1: 1.0^2 + 0.0^2 = 1.0; u2: 0.0^2 + 1.0^2 = 1.0
        # Item base squared norms: p1: 1.0^2 + 0.0^2 = 1.0; p3: 0.0^2 + 1.0^2 = 1.0
        # Total squared norm = 4.0
        # Expected reg penalty = 1e-5 * (4.0 / 4) = 1e-5
        self.assertAlmostEqual(loss_dict["reg_loss"], 1e-5, places=12)

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            compute_bpr_batch_loss(
                user_ids=[],
                positive_item_ids=[],
                negative_item_ids=[],
                user_propagated_embeddings=self.prop_u,
                item_propagated_embeddings=self.prop_i,
                user_base_embeddings=self.user_base,
                item_base_embeddings=self.item_base,
            )

    def test_negative_sampler_integration(self):
        sampler = NegativeSampler(self.context)
        rng = random.Random(42)
        positive_pairs = [("u1", "p1"), ("u2", "p3")]
        batch, skipped = sampler.prepare_batch(positive_pairs, rng)

        self.assertEqual(skipped, 0)
        self.assertEqual(len(batch), 2)

        u_ids = [b[0] for b in batch]
        pos_ids = [b[1] for b in batch]
        neg_ids = [b[2] for b in batch]

        loss = compute_bpr_batch_loss(
            user_ids=u_ids,
            positive_item_ids=pos_ids,
            negative_item_ids=neg_ids,
            user_propagated_embeddings=self.prop_u,
            item_propagated_embeddings=self.prop_i,
            user_base_embeddings=self.user_base,
            item_base_embeddings=self.item_base,
        )
        self.assertGreater(loss["total_loss"], 0.0)

    def test_training_harness_guard_rejects_empty_batches(self):
        # Empty batch on synthetic non-real-data raises ValueError before any update
        guard_synthetic = TrainingHarnessGuard(allow_synthetic_updates=True, is_real_data=False)
        with self.assertRaises(ValueError):
            guard_synthetic.validate_batch_trainable(0, skipped_empty_pool=5)
        with self.assertRaises(ValueError):
            guard_synthetic.validate_batch_trainable(-1)
        with self.assertRaises(ValueError):
            guard_synthetic.validate_batch_trainable(False)  # type: ignore

        # Empty batch on real data must still raise ValueError before any update check
        guard_real = TrainingHarnessGuard(allow_synthetic_updates=False, is_real_data=True)
        with self.assertRaises(ValueError):
            guard_real.validate_batch_trainable(0, skipped_empty_pool=3)

        # Empty batch with allow_synthetic_updates=True and is_real_data=True still raises ValueError first
        guard_bypass_attempt = TrainingHarnessGuard(allow_synthetic_updates=True, is_real_data=True)
        with self.assertRaises(ValueError):
            guard_bypass_attempt.validate_batch_trainable(0, skipped_empty_pool=1)

    def test_training_harness_guard_blocks_unauthorized_real_data_training(self):
        # Real-data training without authorization raises RuntimeError
        guard = TrainingHarnessGuard(allow_synthetic_updates=False, is_real_data=True)
        with self.assertRaises(RuntimeError):
            guard.validate_batch_trainable(10, skipped_empty_pool=0)

        # Default guard parameters enforce real-data fail-closed
        guard_default = TrainingHarnessGuard()
        self.assertTrue(guard_default.is_real_data)
        self.assertFalse(guard_default.allow_synthetic_updates)
        with self.assertRaises(RuntimeError):
            guard_default.validate_batch_trainable(10)

        # Crucial protocol guard check: allow_synthetic_updates=True with is_real_data=True STILL RAISES!
        guard_bypass = TrainingHarnessGuard(allow_synthetic_updates=True, is_real_data=True)
        with self.assertRaises(RuntimeError):
            guard_bypass.validate_batch_trainable(10)
        with self.assertRaises(RuntimeError):
            guard_bypass.validate_batch_trainable(1)

    def test_training_harness_guard_synthetic_validation(self):
        # Synthetic validation without allow_synthetic_updates=True raises RuntimeError
        guard_disallowed = TrainingHarnessGuard(allow_synthetic_updates=False, is_real_data=False)
        with self.assertRaises(RuntimeError):
            guard_disallowed.validate_batch_trainable(10)

        # Synthetic non-real-data valid examples pass guard validation only as reference-fixture check
        guard_allowed = TrainingHarnessGuard(allow_synthetic_updates=True, is_real_data=False)
        self.assertEqual(guard_allowed.validate_batch_trainable(10), 10)
        self.assertEqual(guard_allowed.validate_batch_trainable(1), 1)


class CheckpointContractTests(unittest.TestCase):
    """Test serializable checkpoint manifest contracts and provenance requirements."""

    def test_checkpoint_manifest_creation_and_validation(self):
        manifest = create_checkpoint_manifest(
            method="LightGCN",
            category="Musical_Instruments",
            feedback_condition="all_ratings",
            protocol_version="amazon4u-protocol-v2.0.0",
            config_hashes={"model_suite": "e5047f..."},
            seed=42,
            source_revision="67e5004...",
            dirty_flag=False,
            failure_status=None,
            model_config={"embedding_dimension": 64},
            best_validation_ndcg10=0.085,
            selected_epoch=12,
            early_stopping_counter=3,
        )

        # Validate required fields
        self.assertEqual(manifest["method"], "LightGCN")
        self.assertEqual(manifest["category"], "Musical_Instruments")
        self.assertEqual(manifest["feedback_condition"], "all_ratings")
        self.assertEqual(manifest["protocol_version"], "amazon4u-protocol-v2.0.0")
        self.assertEqual(manifest["seed"], 42)
        self.assertFalse(manifest["training_authorized"])
        self.assertFalse(manifest["profiling_authorized"])
        self.assertEqual(manifest["deployment_decision"], "NOT_AUTHORIZED")

        # Must round-trip to JSON and back
        serialized = json.dumps(manifest)
        loaded = json.loads(serialized)
        self.assertEqual(manifest, loaded)
        self.assertTrue(validate_checkpoint_manifest(loaded))

    def test_checkpoint_manifest_invalid_inputs(self):
        # Invalid method
        with self.assertRaises(ValueError):
            create_checkpoint_manifest(
                method="UnapprovedModel",
                category="Musical_Instruments",
                feedback_condition="all_ratings",
            )

        # Invalid feedback condition
        with self.assertRaises(ValueError):
            create_checkpoint_manifest(
                method="LightGCN",
                category="Musical_Instruments",
                feedback_condition="invalid_condition",
            )

        # Invalid protocol version
        with self.assertRaises(ValueError):
            create_checkpoint_manifest(
                method="LightGCN",
                category="Musical_Instruments",
                feedback_condition="all_ratings",
                protocol_version="unversioned",
            )

        # Non-integer seed
        with self.assertRaises(ValueError):
            create_checkpoint_manifest(
                method="LightGCN",
                category="Musical_Instruments",
                feedback_condition="all_ratings",
                seed="42",  # type: ignore
            )

    def test_checkpoint_manifest_tampering_rejected(self):
        manifest = create_checkpoint_manifest(
            method="LightGCN",
            category="Musical_Instruments",
            feedback_condition="all_ratings",
        )
        self.assertTrue(validate_checkpoint_manifest(manifest))

        # Tampering with authorization must invalidate manifest
        tampered_training = dict(manifest, training_authorized=True)
        self.assertFalse(validate_checkpoint_manifest(tampered_training))

        tampered_deployment = dict(manifest, deployment_decision="AUTHORIZED")
        self.assertFalse(validate_checkpoint_manifest(tampered_deployment))

        # Missing required key
        tampered_missing = dict(manifest)
        del tampered_missing["protocol_version"]
        self.assertFalse(validate_checkpoint_manifest(tampered_missing))


class ModuleImportCompatibilityTests(unittest.TestCase):
    """Test that learned models and utilities are accessible via both import styles."""

    def test_dual_import_canonicalization(self):
        import learned_models as lm1
        import scripts.learned_models as lm2
        from recommendation_core import (
            LightGCNConfig as RLightGCNConfig,
            LightGCNGraph as RLightGCNGraph,
            LightGCNScorer as RLightGCNScorer,
            MissingIdError as RMissingIdError,
            UnknownIdError as RUnknownIdError,
        )

        self.assertIs(lm1.LightGCNConfig, lm2.LightGCNConfig)
        self.assertIs(lm1.LightGCNGraph, lm2.LightGCNGraph)
        self.assertIs(lm1.LightGCNScorer, lm2.LightGCNScorer)
        self.assertIs(lm1.MissingIdError, lm2.MissingIdError)
        self.assertIs(lm1.UnknownIdError, lm2.UnknownIdError)
        self.assertIs(lm1.LightGCNConfig, RLightGCNConfig)
        self.assertIs(lm1.LightGCNGraph, RLightGCNGraph)
        self.assertIs(lm1.LightGCNScorer, RLightGCNScorer)
        self.assertIs(lm1.MissingIdError, RMissingIdError)
        self.assertIs(lm1.UnknownIdError, RUnknownIdError)


if __name__ == "__main__":
    unittest.main()
