"""Unit and regression tests for Differentiable PyTorch Models (Subsystem 1 Recovery).

Tests:
- Canonical metadata validation: CategoryPath JSON prefix identities (preserving ' > ' chars),
  literal UTF-8 matching DuckDB to_json, trimmed identities, Brand IDs, parent derivation,
  and rejection of delimiters / malformed inputs / cycles.
- HeteroTrainingGraph: 4 node types, 8 directed relations and exact inverses, strictly
  condition-positive training interactions, warm-product-only metadata restriction,
  cold metadata exclusion, edge deduplication, held-out isolation, and category hierarchy validation.
- DifferentiableLightGCN: Option B fixed-graph training-edge reuse, exact 3-layer degree-normalized
  propagation parity with reference propagate_lightgcn, empty-positive user handling,
  caller-supplied finite 64d base embeddings (detached and cloned), config validation,
  and literal numeric graph propagation with isolated 1/4 layer scaling.
- DifferentiableHeteroGNN:
  - Local deterministic RNG generator (Normal(0, 0.1) IDs, Xavier gain-1 matrices, zero biases)
    without mutating global PyTorch RNG state; all-parameters byte-for-byte reproducibility.
  - Independent 64x64 matrix storage for all 16 relations and 8 self matrices/biases.
  - Literal 2-layer multi-relation numeric oracle with changing neighbor counts,
    within-relation and cross-relation unweighted means, self-only empty-relation output,
    ReLU (layer 0) and identity (layer 1) activations, and exact raw dot-product decoding.
  - Fixture cloned with extra same-valued category neighbor showing unchanged within/cross mean output.
  - Exact two-layer masked oracle verifying active relations recomputed and metadata preserved.
  - On-the-fly batch target masking (forward and reverse edges) across both layers.
  - Shared forward positive/negative batch scoring with mock/spy verifying single call with union masks.
  - Differentiability: autograd backward propagates finite non-zero gradients to supervised base rows,
    participating metadata neighbor base rows, relation matrices, self matrices, and biases.
  - Explicit endpoint-only base-ID regularization (1e-5 / (|U_B| + |I_B|)) with exact reg-only gradient isolation.
  - Strict exclusion of low-rated training items from negative sampling even when globally warm.
  - Rejection of CUDA/meta tensors, non-float32 dtypes, non-finite values, empty batches,
    string batch sequences, empty explicit masks, unknown IDs, and forbidden observed-history negatives.
  - Evaluator candidate scoring compatibility and empty-positive-history popularity fallback.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F

import differentiable_models
import scripts.differentiable_models

from differentiable_models import (
    INVERSE_RELATIONS,
    NODE_TYPES,
    RELATION_SCHEMAS,
    DifferentiableHeteroGNN,
    DifferentiableLightGCN,
    HeteroTrainingGraph,
    build_hetero_graph,
    check_cpu_tensor,
    derive_parent_category_id,
    validate_brand_id,
    validate_category_id,
)
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


class MetadataValidationTests(unittest.TestCase):
    """Test canonical KG-v1 CategoryPath and Brand identity validation."""

    def test_valid_canonical_category_ids(self):
        # Single element
        path = validate_category_id('c:["Electronics"]')
        self.assertEqual(path, ["Electronics"])

        # Multiple elements
        path = validate_category_id('c:["Electronics","Audio","Headphones"]')
        self.assertEqual(path, ["Electronics", "Audio", "Headphones"])

        # Label containing ' > ' chars must NOT be split by delimiters
        complex_id = 'c:["Musical Instruments","Guitars > Electric > Solid Body"]'
        path = validate_category_id(complex_id)
        self.assertEqual(path, ["Musical Instruments", "Guitars > Electric > Solid Body"])

    def test_unicode_utf8_json_and_trimmed_identities(self):
        # Literal UTF-8 characters matching DuckDB to_json output
        utf8_id = 'c:["Café","Música","Électronique"]'
        path = validate_category_id(utf8_id)
        self.assertEqual(path, ["Café", "Música", "Électronique"])

        # Escaped unicode payload is rejected as non-canonical
        escaped_id = 'c:["Caf\\u00e9"]'
        with self.assertRaises(ValueError):
            validate_category_id(escaped_id)

        # Padded category labels are rejected
        with self.assertRaises(ValueError):
            validate_category_id('c:[" Electronics "]')
        with self.assertRaises(ValueError):
            validate_category_id('c:["Electronics"," Audio"]')

        # Padded category ID itself is rejected
        with self.assertRaises(ValueError):
            validate_category_id(' c:["Electronics"] ')

        # Padded brand ID and label are rejected
        with self.assertRaises(ValueError):
            validate_brand_id(" b:Sony ")
        with self.assertRaises(ValueError):
            validate_brand_id("b: Sony ")
        with self.assertRaises(ValueError):
            validate_brand_id("b:Sony ")

    def test_derive_parent_category_id(self):
        # Root path has no parent
        self.assertIsNone(derive_parent_category_id('c:["Electronics"]'))

        # Multi-level path derives immediate parent JSON prefix
        parent = derive_parent_category_id('c:["Electronics","Audio","Headphones"]')
        self.assertEqual(parent, 'c:["Electronics","Audio"]')

        # Preserves ' > ' characters in parent derivation
        child_id = 'c:["Musical Instruments","Guitars > Basses","Accessories"]'
        parent = derive_parent_category_id(child_id)
        self.assertEqual(parent, 'c:["Musical Instruments","Guitars > Basses"]')

        # Preserves literal UTF-8 in parent derivation
        utf8_child = 'c:["Café","Música","Électronique"]'
        utf8_parent = derive_parent_category_id(utf8_child)
        self.assertEqual(utf8_parent, 'c:["Café","Música"]')

    def test_invalid_category_ids_rejected(self):
        # Non-string
        with self.assertRaises(ValueError):
            validate_category_id(123)  # type: ignore

        # Missing c: prefix
        with self.assertRaises(ValueError):
            validate_category_id('["Electronics"]')
        with self.assertRaises(ValueError):
            validate_category_id('Electronics > Audio')

        # Invalid JSON
        with self.assertRaises(ValueError):
            validate_category_id('c:[Electronics]')

        # Non-list JSON
        with self.assertRaises(ValueError):
            validate_category_id('c:{"category":"Electronics"}')

        # Empty list
        with self.assertRaises(ValueError):
            validate_category_id('c:[]')

        # Non-string element
        with self.assertRaises(ValueError):
            validate_category_id('c:["Electronics",123]')

        # Blank or whitespace-only element
        with self.assertRaises(ValueError):
            validate_category_id('c:["Electronics",""]')
        with self.assertRaises(ValueError):
            validate_category_id('c:["Electronics","   "]')

        # Non-compact formatting (whitespace after comma)
        with self.assertRaises(ValueError):
            validate_category_id('c:["Electronics", "Audio"]')

    def test_brand_id_validation(self):
        self.assertEqual(validate_brand_id("b:Sony"), "Sony")
        self.assertEqual(validate_brand_id("b:Fender Musical Instruments"), "Fender Musical Instruments")

        with self.assertRaises(ValueError):
            validate_brand_id("Sony")
        with self.assertRaises(ValueError):
            validate_brand_id("b:")
        with self.assertRaises(ValueError):
            validate_brand_id("b:   ")
        with self.assertRaises(ValueError):
            validate_brand_id(123)  # type: ignore


class HeteroTrainingGraphTests(unittest.TestCase):
    """Test HeteroTrainingGraph construction, metadata filtering, and isolation."""

    def setUp(self):
        self.training_records = [
            ("u1", "p1", 5.0, 100),
            ("u1", "p2", 4.0, 101),
            ("u2", "p2", 5.0, 102),
            ("u2", "p3", 2.0, 103),  # low rating (< 4)
            ("u3", "p4", 5.0, 104),
        ]

    def test_all_ratings_vs_rating_ge_4_graph_edges(self):
        # In all_ratings condition, p3 is condition-positive for u2
        ctx_all = TrainingContext(self.training_records, condition="all_ratings")
        g_all = build_hetero_graph(ctx_all)
        self.assertIn(("u2", "p3"), g_all.edges["rated"])
        self.assertIn(("p3", "u2"), g_all.edges["rated_by"])

        # In rating_ge_4 condition, p3 (rating 2.0) is absent as an interaction edge
        ctx_ge4 = TrainingContext(self.training_records, condition="rating_ge_4")
        g_ge4 = build_hetero_graph(ctx_ge4)
        self.assertNotIn(("u2", "p3"), g_ge4.edges["rated"])
        self.assertNotIn(("p3", "u2"), g_ge4.edges["rated_by"])

        # However, p3 was rated in training, so it is in u2's all_rated_history
        self.assertIn("p3", ctx_ge4.all_rated_history["u2"])

    def test_metadata_restriction_to_warm_products_and_cold_isolation(self):
        ctx = TrainingContext(self.training_records, condition="rating_ge_4")
        # Warm products are p1, p2, p4 (p3 had rating 2.0, so in rating_ge_4 warm items are {p1, p2, p4})
        warm_items = set(ctx.sorted_warm_items)
        self.assertEqual(warm_items, {"p1", "p2", "p4"})

        # Metadata provided for warm products and a cold product p_cold
        product_categories = {
            "p1": ['c:["Musical Instruments","Guitars > Basses"]'],
            "p2": ['c:["Musical Instruments","Drums"]'],
            "p_cold": ['c:["Unrelated","ColdCategory"]'],
        }
        product_brands = {
            "p1": "b:Fender",
            "p2": "b:Yamaha",
            "p_cold": "b:ColdBrand",
        }

        graph = build_hetero_graph(
            ctx,
            product_categories=product_categories,
            product_brands=product_brands,
        )

        # Cold product and its metadata must NOT be in the graph
        self.assertNotIn("p_cold", graph.products)
        self.assertNotIn('c:["Unrelated","ColdCategory"]', graph.categories)
        self.assertNotIn('c:["Unrelated"]', graph.categories)
        self.assertNotIn("b:ColdBrand", graph.brands)

        # Warm metadata must be present along with ancestor CategoryPaths
        self.assertIn('c:["Musical Instruments","Guitars > Basses"]', graph.categories)
        self.assertIn('c:["Musical Instruments"]', graph.categories)  # Derived parent
        self.assertIn('c:["Musical Instruments","Drums"]', graph.categories)
        self.assertIn("b:Fender", graph.brands)
        self.assertIn("b:Yamaha", graph.brands)

        # Check edge schemas and inverses
        self.assertIn(("p1", 'c:["Musical Instruments","Guitars > Basses"]'), graph.edges["belongs_to"])
        self.assertIn(('c:["Musical Instruments","Guitars > Basses"]', "p1"), graph.edges["contains_product"])

        # Category hierarchy edge
        child_id = 'c:["Musical Instruments","Guitars > Basses"]'
        parent_id = 'c:["Musical Instruments"]'
        self.assertIn((child_id, parent_id), graph.edges["category_child_of"])
        self.assertIn((parent_id, child_id), graph.edges["category_parent_of"])

        # Brand edges
        self.assertIn(("p1", "b:Fender"), graph.edges["has_brand"])
        self.assertIn(("b:Fender", "p1"), graph.edges["brand_of"])

    def test_invalid_hierarchy_cycles_and_unreachable_isolation(self):
        ctx = TrainingContext(self.training_records, condition="all_ratings")
        warm_p = "p1"
        valid_cats = {warm_p: ['c:["Electronics","Audio","Headphones"]']}

        # Non-dict hierarchy rejected
        with self.assertRaises(ValueError):
            build_hetero_graph(ctx, product_categories=valid_cats, category_hierarchy=["not", "a", "dict"])  # type: ignore

        # Self-loop rejected
        with self.assertRaises(ValueError):
            build_hetero_graph(
                ctx,
                product_categories=valid_cats,
                category_hierarchy={'c:["Electronics","Audio"]': 'c:["Electronics","Audio"]'},
            )

        # Root category having a parent rejected
        with self.assertRaises(ValueError):
            build_hetero_graph(
                ctx,
                product_categories=valid_cats,
                category_hierarchy={'c:["Electronics"]': 'c:["Other"]'},
            )

        # Skipped ancestor (grandchild mapped directly to grandparent) rejected
        with self.assertRaises(ValueError):
            build_hetero_graph(
                ctx,
                product_categories=valid_cats,
                category_hierarchy={'c:["Electronics","Audio","Headphones"]': 'c:["Electronics"]'},
            )

        # Unrelated parent rejected
        with self.assertRaises(ValueError):
            build_hetero_graph(
                ctx,
                product_categories=valid_cats,
                category_hierarchy={'c:["Electronics","Audio"]': 'c:["Books"]'},
            )

        # Valid unreachable hierarchy entry must NOT enter the graph
        valid_unreachable_hierarchy = {
            'c:["Electronics","Audio","Headphones"]': 'c:["Electronics","Audio"]',
            'c:["Unreachable","Sub"]': 'c:["Unreachable"]',  # No warm product belongs to this
        }
        graph = build_hetero_graph(
            ctx,
            product_categories=valid_cats,
            category_hierarchy=valid_unreachable_hierarchy,
        )
        self.assertNotIn('c:["Unreachable","Sub"]', graph.categories)
        self.assertNotIn('c:["Unreachable"]', graph.categories)
        self.assertNotIn(
            ('c:["Unreachable","Sub"]', 'c:["Unreachable"]'),
            graph.edges["category_child_of"],
        )

    def test_rating_mutation_preserving_membership_cannot_alter_graph(self):
        records_a = [("u1", "p1", 4.0, 100), ("u1", "p2", 5.0, 101)]
        records_b = [("u1", "p1", 5.0, 100), ("u1", "p2", 4.5, 101)]

        ctx_a = TrainingContext(records_a, condition="rating_ge_4")
        ctx_b = TrainingContext(records_b, condition="rating_ge_4")

        g_a = build_hetero_graph(ctx_a)
        g_b = build_hetero_graph(ctx_b)

        self.assertEqual(g_a.edges["rated"], g_b.edges["rated"])
        self.assertEqual(g_a.edges["rated_by"], g_b.edges["rated_by"])

    def test_eight_relation_schema_inverses_and_deduplication(self):
        records = [
            ("u1", "p1", 5.0, 100),
            ("u2", "p2", 4.0, 101),
        ]
        ctx = TrainingContext(records, condition="rating_ge_4")
        p_cats = {
            "p1": ['c:["Electronics","Audio"]', 'c:["Electronics","Audio"]'],  # Duplicate category
            "p2": ['c:["Electronics","Audio","Headphones"]'],
        }
        p_brands = {
            "p1": "b:Sony",
            "p2": "b:Sony",
        }
        graph = build_hetero_graph(ctx, product_categories=p_cats, product_brands=p_brands)

        expected_relations = {
            "rated", "rated_by", "belongs_to", "contains_product",
            "has_brand", "brand_of", "category_child_of", "category_parent_of",
        }
        self.assertEqual(set(graph.edges.keys()), expected_relations)

        # Check deduplication of redundant category memberships
        self.assertEqual(len([e for e in graph.edges["rated"] if e == ("u1", "p1")]), 1)
        self.assertEqual(len([e for e in graph.edges["belongs_to"] if e == ("p1", 'c:["Electronics","Audio"]')]), 1)
        self.assertEqual(len([e for e in graph.edges["contains_product"] if e == ('c:["Electronics","Audio"]', "p1")]), 1)

        # Check all 8 relations, type schemas, and exact inverses
        for rel in expected_relations:
            src_type, dst_type = RELATION_SCHEMAS[rel]
            inv_rel = INVERSE_RELATIONS[rel]
            self.assertEqual(INVERSE_RELATIONS[inv_rel], rel)

            for src, dst in graph.edges[rel]:
                self.assertIn(src, graph.node_ids[src_type], f"Source {src} not in {src_type}")
                self.assertIn(dst, graph.node_ids[dst_type], f"Destination {dst} not in {dst_type}")
                self.assertIn((dst, src), graph.edges[inv_rel], f"Inverse edge ({dst}, {src}) not in {inv_rel}")

    def test_empty_positive_context_fails_graph_construction(self):
        # All ratings below 4.0 under rating_ge_4 yields 0 positive interactions
        records = [("u1", "p1", 2.0, 100), ("u2", "p2", 1.0, 101)]
        ctx = TrainingContext(records, condition="rating_ge_4")
        with self.assertRaises(ValueError) as cm:
            build_hetero_graph(ctx)
        self.assertIn("Empty trainable graph", str(cm.exception))

        with self.assertRaises(ValueError) as cm2:
            LightGCNGraph(ctx)
        self.assertIn("Empty trainable graph", str(cm2.exception))


class DifferentiableLightGCNTests(unittest.TestCase):
    """Test PyTorch LightGCN Option B fixed-graph implementation and reference parity."""

    def setUp(self):
        self.records = [
            ("u1", "p1", 5.0, 100),
            ("u1", "p2", 4.0, 101),
            ("u2", "p2", 5.0, 102),
            ("u3", "p3", 1.0, 103),  # u3 has empty positive history in rating_ge_4
        ]
        self.ctx = TrainingContext(self.records, condition="rating_ge_4")
        self.ref_graph = LightGCNGraph(self.ctx)

        # Deterministic 64d initial embeddings
        self.u_init = {
            u: [0.05 * (idx + 1) + 0.001 * k for k in range(64)]
            for idx, u in enumerate(self.ref_graph.users)
        }
        self.i_init = {
            item: [0.08 * (idx + 1) + 0.001 * k for k in range(64)]
            for idx, item in enumerate(self.ref_graph.warm_items)
        }

    def test_parity_with_reference_propagation(self):
        ref_u, ref_i = propagate_lightgcn(self.ref_graph, self.u_init, self.i_init, layers=3)

        model = DifferentiableLightGCN(
            self.ref_graph,
            user_base_embeddings=self.u_init,
            item_base_embeddings=self.i_init,
        )
        prop_u, prop_i = model.forward()

        # Check user parity
        u_map = {u: idx for idx, u in enumerate(self.ref_graph.users)}
        for u in self.ref_graph.users:
            pyt_vec = prop_u[u_map[u]].detach().tolist()
            ref_vec = ref_u[u]
            for val_pyt, val_ref in zip(pyt_vec, ref_vec):
                self.assertAlmostEqual(val_pyt, val_ref, places=5)

        # Check item parity
        i_map = {item: idx for idx, item in enumerate(self.ref_graph.warm_items)}
        for item in self.ref_graph.warm_items:
            pyt_vec = prop_i[i_map[item]].detach().tolist()
            ref_vec = ref_i[item]
            for val_pyt, val_ref in zip(pyt_vec, ref_vec):
                self.assertAlmostEqual(val_pyt, val_ref, places=5)

    def test_caller_supplied_embeddings_mandatory(self):
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=None, item_base_embeddings=self.i_init)  # type: ignore

    def test_caller_supplied_embeddings_dict_missing_or_extra_ids(self):
        # Missing user
        bad_u = dict(self.u_init)
        del bad_u[self.ref_graph.users[0]]
        with self.assertRaises(MissingIdError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u, item_base_embeddings=self.i_init)

        # Extra user
        bad_u_extra = dict(self.u_init)
        bad_u_extra["extra_user"] = [0.0] * 64
        with self.assertRaises(UnknownIdError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_extra, item_base_embeddings=self.i_init)

        # Missing item
        bad_i = dict(self.i_init)
        del bad_i[self.ref_graph.warm_items[0]]
        with self.assertRaises(MissingIdError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=self.u_init, item_base_embeddings=bad_i)

        # Extra item
        bad_i_extra = dict(self.i_init)
        bad_i_extra["extra_item"] = [0.0] * 64
        with self.assertRaises(UnknownIdError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=self.u_init, item_base_embeddings=bad_i_extra)

    def test_caller_supplied_tensors_detached_and_cloned(self):
        # Modifying original tensor after passing to model must NOT alter model parameters
        u_t = torch.zeros((len(self.ref_graph.users), 64), dtype=torch.float32)
        i_t = torch.zeros((len(self.ref_graph.warm_items), 64), dtype=torch.float32)

        model = DifferentiableLightGCN(self.ref_graph, user_base_embeddings=u_t, item_base_embeddings=i_t)

        u_t.fill_(999.0)
        i_t.fill_(999.0)

        self.assertTrue((model.user_embeddings == 0.0).all())
        self.assertTrue((model.item_embeddings == 0.0).all())

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(
                self.ref_graph,
                self.u_init,
                self.i_init,
                config=LightGCNConfig(embedding_dimension=32),
            )
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(
                self.ref_graph,
                self.u_init,
                self.i_init,
                config=LightGCNConfig(propagation_layers=2),
            )

    def test_empty_positive_history_fallback_in_candidate_scoring(self):
        model = DifferentiableLightGCN(
            self.ref_graph,
            user_base_embeddings=self.u_init,
            item_base_embeddings=self.i_init,
        )
        # u3 has empty positive history
        scores, route, reason = model.score_candidates("u3", ["p1", "p2"])
        self.assertEqual(route, "fallback")
        self.assertEqual(reason, "empty_positive_history")
        self.assertEqual(scores["p1"], float(self.ctx.popularity.get("p1", 0)))

        # u1 has positive history
        scores_u1, route_u1, reason_u1 = model.score_candidates("u1", ["p1", "p2"])
        self.assertEqual(route_u1, "personalized")
        self.assertIsNone(reason_u1)
        self.assertTrue(all(math.isfinite(s) for s in scores_u1.values()))

    def test_lightgcn_literal_numeric_graph_dot_and_isolated_layer_scaling(self):
        """Verify exact hand-calculated 3-layer LightGCN propagation on a tiny graph."""
        # u1 connected to p1 (degrees 1 and 1); u_iso has degree 0
        records = [
            ("u1", "p1", 5.0, 100),
            ("u_iso", "p1", 1.0, 101),  # in rating_ge_4, u_iso has 0 positive edges
            ("u2", "p2", 5.0, 102),
        ]
        ctx = TrainingContext(records, condition="rating_ge_4")
        g = LightGCNGraph(ctx)

        dim = 64
        # Supply literal base embeddings:
        # e_u1^(0) = 1.0 * 1, e_iso^(0) = 4.0 * 1, e_u2^(0) = 0.5 * 1
        # e_p1^(0) = 2.0 * 1, e_p2^(0) = 3.0 * 1
        u_init = {
            "u1": [1.0] * dim,
            "u_iso": [4.0] * dim,
            "u2": [0.5] * dim,
        }
        i_init = {
            "p1": [2.0] * dim,
            "p2": [3.0] * dim,
        }

        model = DifferentiableLightGCN(g, user_base_embeddings=u_init, item_base_embeddings=i_init)
        prop_u, prop_i = model.forward()

        # Degree-normalized weight between u1 and p1 is 1 / (sqrt(1)*sqrt(1)) = 1.0.
        # Propagation trace:
        # Layer 0: e_u1 = 1.0, e_iso = 4.0, e_p1 = 2.0
        # Layer 1: e_u1 = 2.0, e_iso = 0.0, e_p1 = 1.0
        # Layer 2: e_u1 = 1.0, e_iso = 0.0, e_p1 = 2.0
        # Layer 3: e_u1 = 2.0, e_iso = 0.0, e_p1 = 1.0
        #
        # Layer 0..3 average:
        # e_u1 = (1.0 + 2.0 + 1.0 + 2.0) / 4 = 6.0 / 4 = 1.5
        # e_p1 = (2.0 + 1.0 + 2.0 + 1.0) / 4 = 6.0 / 4 = 1.5
        # e_iso = (4.0 + 0.0 + 0.0 + 0.0) / 4 = 1.0 (exact 1/4 factor!)
        u_map = {u: idx for idx, u in enumerate(g.users)}
        i_map = {item: idx for idx, item in enumerate(g.warm_items)}

        u1_out = prop_u[u_map["u1"]]
        u_iso_out = prop_u[u_map["u_iso"]]
        p1_out = prop_i[i_map["p1"]]

        self.assertTrue(torch.allclose(u1_out, torch.full((dim,), 1.5), atol=1e-5))
        self.assertTrue(torch.allclose(p1_out, torch.full((dim,), 1.5), atol=1e-5))
        self.assertTrue(torch.allclose(u_iso_out, torch.full((dim,), 1.0), atol=1e-5))

        # Raw dot product between u1 and p1: 64 * (1.5 * 1.5) = 64 * 2.25 = 144.0
        # p2 is unrated by u1, hence an eligible negative
        s_pos, s_neg = model.compute_batch_scores(["u1"], ["p1"], ["p2"])
        self.assertAlmostEqual(s_pos.item(), 144.0, places=4)

    def test_dictionary_embedding_rows_strictly_numeric_nonbool_length64_finite(self):
        u0 = self.ref_graph.users[0]
        # Boolean element in list (torch.tensor would otherwise cast bool to float)
        bad_u_bool = dict(self.u_init)
        bad_u_bool[u0] = [True] * 64
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_bool, item_base_embeddings=self.i_init)

        # Non-numeric element (string)
        bad_u_str = dict(self.u_init)
        bad_u_str[u0] = ["0.5"] * 64
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_str, item_base_embeddings=self.i_init)

        # None element
        bad_u_none = dict(self.u_init)
        bad_u_none[u0] = [None] * 64
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_none, item_base_embeddings=self.i_init)

        # Wrong length (63 and 65)
        bad_u_short = dict(self.u_init)
        bad_u_short[u0] = [0.1] * 63
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_short, item_base_embeddings=self.i_init)

        bad_u_long = dict(self.u_init)
        bad_u_long[u0] = [0.1] * 65
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_long, item_base_embeddings=self.i_init)

        # Non-finite (NaN, Inf)
        bad_u_nan = dict(self.u_init)
        bad_u_nan[u0] = [float("nan")] * 64
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_nan, item_base_embeddings=self.i_init)

        bad_u_inf = dict(self.u_init)
        bad_u_inf[u0] = [float("inf")] * 64
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=bad_u_inf, item_base_embeddings=self.i_init)

        # Same checks for item embeddings
        i0 = self.ref_graph.warm_items[0]
        bad_i_bool = dict(self.i_init)
        bad_i_bool[i0] = [False] * 64
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(self.ref_graph, user_base_embeddings=self.u_init, item_base_embeddings=bad_i_bool)

    def test_lightgcn_config_revalidation_and_freeze(self):
        # Provided config must match graph.config
        with self.assertRaises(ValueError):
            DifferentiableLightGCN(
                self.ref_graph,
                self.u_init,
                self.i_init,
                config=LightGCNConfig(embedding_dimension=32),
            )

        model = DifferentiableLightGCN(self.ref_graph, self.u_init, self.i_init)
        # Mutating model.config after init must be caught in forward
        object.__setattr__(model, "config", LightGCNConfig(embedding_dimension=32))
        with self.assertRaises(ValueError):
            model.forward()

    def test_lightgcn_adjacency_validation_and_corruption_rejection(self):
        model = DifferentiableLightGCN(self.ref_graph, self.u_init, self.i_init)
        # Verify registered buffer
        self.assertIn("adj_user_item", dict(model.named_buffers()))

        # Non-finite values in adj (NaN)
        indices = torch.tensor([[0], [0]], dtype=torch.long)
        bad_adj = torch.sparse_coo_tensor(
            indices, torch.tensor([float("nan")]), (len(self.ref_graph.users), len(self.ref_graph.warm_items))
        )
        model.adj_user_item = bad_adj
        with self.assertRaises(ValueError):
            model.forward()

        # Inf values in adj
        bad_adj_inf = torch.sparse_coo_tensor(
            indices, torch.tensor([float("inf")]), (len(self.ref_graph.users), len(self.ref_graph.warm_items))
        )
        model.adj_user_item = bad_adj_inf
        with self.assertRaises(ValueError):
            model.forward()

        # Dense layout rejection
        model.adj_user_item = torch.zeros((len(self.ref_graph.users), len(self.ref_graph.warm_items)))
        with self.assertRaises(ValueError):
            model.forward()

        # Wrong shape rejection
        model.adj_user_item = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0),
            (len(self.ref_graph.users) + 1, len(self.ref_graph.warm_items)),
        )
        with self.assertRaises(ValueError):
            model.forward()

        # Out-of-bounds indices rejection
        oob_indices = torch.tensor([[len(self.ref_graph.users) + 5], [0]], dtype=torch.long)
        model.adj_user_item = torch.sparse_coo_tensor(
            oob_indices, torch.tensor([1.0]), (len(self.ref_graph.users), len(self.ref_graph.warm_items))
        )
        with self.assertRaises(ValueError):
            model.forward()

        # Meta tensor rejection
        model.adj_user_item = torch.empty((len(self.ref_graph.users), len(self.ref_graph.warm_items)), device="meta")
        with self.assertRaises(ValueError):
            model.forward()

    def test_lightgcn_parameter_corruption_and_overflow(self):
        model = DifferentiableLightGCN(self.ref_graph, self.u_init, self.i_init)

        # Corrupt user_embeddings with NaN
        with torch.no_grad():
            model.user_embeddings[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt with Inf
        model = DifferentiableLightGCN(self.ref_graph, self.u_init, self.i_init)
        with torch.no_grad():
            model.user_embeddings[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt item_embeddings with wrong shape
        model = DifferentiableLightGCN(self.ref_graph, self.u_init, self.i_init)
        model.item_embeddings = nn.Parameter(torch.zeros((len(self.ref_graph.warm_items), 32)))
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt with wrong dtype (float64)
        model = DifferentiableLightGCN(self.ref_graph, self.u_init, self.i_init)
        model.item_embeddings = nn.Parameter(torch.zeros((len(self.ref_graph.warm_items), 64), dtype=torch.float64))
        with self.assertRaises(ValueError):
            model.forward()

        # Finite float32 overflow: extreme finite values cause overflow to Inf in loss
        model = DifferentiableLightGCN(self.ref_graph, self.u_init, self.i_init)
        with torch.no_grad():
            model.user_embeddings.fill_(1e30)
            model.item_embeddings.fill_(1e30)
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["u1"], ["p1"], ["p2"])

    def test_lightgcn_ranking_and_regularization_backward_gradient_scopes(self):
        records = [
            ("u1", "p1", 5.0, 100),
            ("u2", "p1", 5.0, 101),  # u2 is neighbor of p1
            ("u1", "p2", 5.0, 102),  # p2 is neighbor of u1
            ("u3", "p3", 5.0, 103),  # p3 is eligible negative for u1
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        g = LightGCNGraph(ctx)

        u_init = {u: [0.1] * 64 for u in g.users}
        i_init = {i: [0.2] * 64 for i in g.warm_items}
        model = DifferentiableLightGCN(g, u_init, i_init)

        loss_dict = model.compute_batch_loss(["u1"], ["p1"], ["p3"])

        # 1. Ranking backward: propagation sends gradient to neighbor u2 and neighbor p2
        loss_dict["ranking_loss"].backward()

        u_map = {u: idx for idx, u in enumerate(g.users)}
        i_map = {i: idx for idx, i in enumerate(g.warm_items)}

        u1_idx = u_map["u1"]
        u2_idx = u_map["u2"]
        p1_idx = i_map["p1"]
        p2_idx = i_map["p2"]
        p3_idx = i_map["p3"]

        self.assertTrue((model.user_embeddings.grad[u1_idx] != 0).any())
        self.assertTrue((model.item_embeddings.grad[p1_idx] != 0).any())
        self.assertTrue((model.item_embeddings.grad[p3_idx] != 0).any())

        # Neighbor u2 and neighbor p2 receive non-zero ranking gradient through LightGCN propagation
        self.assertTrue((model.user_embeddings.grad[u2_idx] != 0).any())
        self.assertTrue((model.item_embeddings.grad[p2_idx] != 0).any())

        # 2. Regularization backward: strictly endpoint-only base rows
        model.zero_grad()
        loss_dict_reg = model.compute_batch_loss(["u1"], ["p1"], ["p3"])
        loss_dict_reg["reg_loss"].backward()

        self.assertTrue((model.user_embeddings.grad[u1_idx] != 0).any())
        self.assertTrue((model.item_embeddings.grad[p1_idx] != 0).any())
        self.assertTrue((model.item_embeddings.grad[p3_idx] != 0).any())

        # Neighbor-only u2 and p2 have STRICTLY ZERO regularization gradient
        self.assertTrue((model.user_embeddings.grad[u2_idx] == 0).all())
        self.assertTrue((model.item_embeddings.grad[p2_idx] == 0).all())

    def test_lightgcn_low_rated_observed_negative_rejected(self):
        records = [
            ("u1", "p1", 5.0, 100),
            ("u1", "p_low", 2.0, 101),  # Low training rating (< 4)
            ("u2", "p_low", 5.0, 102),  # p_low is globally warm
            ("u2", "p3", 5.0, 103),
        ]
        ctx = TrainingContext(records, condition="rating_ge_4")
        g = LightGCNGraph(ctx)
        u_init = {u: [0.1] * 64 for u in g.users}
        i_init = {i: [0.2] * 64 for i in g.warm_items}
        model = DifferentiableLightGCN(g, u_init, i_init)

        with self.assertRaises(ValueError):
            model.compute_batch_scores(["u1"], ["p1"], ["p_low"])


class DifferentiableHeteroGNNTests(unittest.TestCase):
    """Test Heterogeneous KG+GNN model initialization, operations, masking, and gradients."""

    def setUp(self):
        self.records = [
            ("u1", "p1", 5.0, 100),
            ("u1", "p2", 5.0, 101),
            ("u2", "p2", 5.0, 102),
            ("u2", "p3", 5.0, 103),
            ("u3", "p4", 1.0, 104),  # u3 has empty positive history under rating_ge_4
        ]
        self.ctx = TrainingContext(self.records, condition="rating_ge_4")
        self.product_categories = {
            "p1": ['c:["Electronics","Audio"]'],
            "p2": ['c:["Electronics","Audio"]'],
            "p3": ['c:["Electronics","Accessories"]'],
        }
        self.product_brands = {
            "p1": "b:Sony",
            "p2": "b:Sony",
            "p3": "b:Bose",
        }
        self.graph = build_hetero_graph(
            self.ctx,
            product_categories=self.product_categories,
            product_brands=self.product_brands,
        )

    def test_local_rng_reproducibility_and_global_isolation(self):
        state_before = torch.get_rng_state()

        m1 = DifferentiableHeteroGNN(self.graph, init_seed=42)
        m2 = DifferentiableHeteroGNN(self.graph, init_seed=42)
        m3 = DifferentiableHeteroGNN(self.graph, init_seed=2026)

        state_after = torch.get_rng_state()
        # Global PyTorch RNG must remain untouched byte-for-byte
        self.assertTrue(torch.equal(state_before, state_after))

        # Same seed produces exact identical parameters across all tensors
        self.assertTrue(torch.equal(m1.user_embeddings, m2.user_embeddings))
        self.assertTrue(torch.equal(m1.product_embeddings, m2.product_embeddings))
        self.assertTrue(torch.equal(m1.relation_weights_l0["rated"], m2.relation_weights_l0["rated"]))
        self.assertTrue(torch.equal(m1.self_weights_l0["Product"], m2.self_weights_l0["Product"]))

        # Different seed produces different parameters
        self.assertFalse(torch.equal(m1.user_embeddings, m3.user_embeddings))
        self.assertFalse(torch.equal(m1.relation_weights_l0["rated"], m3.relation_weights_l0["rated"]))

    def test_all_parameters_reproducibility(self):
        m1 = DifferentiableHeteroGNN(self.graph, init_seed=42)
        m2 = DifferentiableHeteroGNN(self.graph, init_seed=42)

        # Check EVERY parameter for byte-for-byte identity
        params1 = dict(m1.named_parameters())
        params2 = dict(m2.named_parameters())
        self.assertEqual(set(params1.keys()), set(params2.keys()))

        for name, p1 in params1.items():
            p2 = params2[name]
            self.assertTrue(
                torch.equal(p1, p2),
                f"Parameter {name} differed between identical seed initializations",
            )

    def test_independent_matrix_storage(self):
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)

        # 16 relation matrices: 8 relations x 2 layers
        # 8 self matrices: 4 node types x 2 layers
        # 8 self biases: 4 node types x 2 layers
        all_param_tensors = []
        for rel in RELATION_SCHEMAS:
            all_param_tensors.append(model.relation_weights_l0[rel])
            all_param_tensors.append(model.relation_weights_l1[rel])
        for ntype in NODE_TYPES:
            all_param_tensors.append(model.self_weights_l0[ntype])
            all_param_tensors.append(model.self_biases_l0[ntype])
            all_param_tensors.append(model.self_weights_l1[ntype])
            all_param_tensors.append(model.self_biases_l1[ntype])

        # All data pointers must be distinct
        ptrs = {t.data_ptr() for t in all_param_tensors}
        self.assertEqual(len(ptrs), len(all_param_tensors))

    def test_cold_metadata_and_heldout_isolation_same_seeded_params(self):
        # Build clean graph A with warm-only data
        g_warm = build_hetero_graph(
            self.ctx,
            product_categories={"p1": ['c:["Electronics","Audio"]']},
            product_brands={"p1": "b:Sony"},
        )

        # Build graph B with extra cold product and cold metadata
        g_with_cold = build_hetero_graph(
            self.ctx,
            product_categories={
                "p1": ['c:["Electronics","Audio"]'],
                "p_cold": ['c:["ColdCat","Sub"]'],
            },
            product_brands={
                "p1": "b:Sony",
                "p_cold": "b:ColdBrand",
            },
        )

        # Nodes and edges must be identical because cold data is isolated
        self.assertEqual(g_warm.node_ids, g_with_cold.node_ids)
        self.assertEqual(g_warm.edges, g_with_cold.edges)

        # Models initialized with the same seed have identical parameters
        m_warm = DifferentiableHeteroGNN(g_warm, init_seed=42)
        m_cold = DifferentiableHeteroGNN(g_with_cold, init_seed=42)

        for (k1, v1), (k2, v2) in zip(m_warm.named_parameters(), m_cold.named_parameters()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.equal(v1, v2))

    def test_literal_two_layer_hand_calculated_numeric_oracle(self):
        """Verify complete step-by-step literal hand calculation, cloning, and exact mask oracle."""
        records = [("u1", "p1", 5.0, 100)]
        ctx = TrainingContext(records, condition="all_ratings")
        p_cats = {"p1": ['c:["CatA"]', 'c:["CatB"]']}
        g = build_hetero_graph(ctx, product_categories=p_cats)

        model = DifferentiableHeteroGNN(g, init_seed=42)
        dim = 64

        with torch.no_grad():
            # Base ID embeddings:
            # u1: 0.5, p1: 0.2, c1: 0.4, c2: 0.8
            u1_emb = torch.full((1, dim), 0.5, dtype=torch.float32)
            p1_emb = torch.full((1, dim), 0.2, dtype=torch.float32)
            c1_emb = torch.full((1, dim), 0.4, dtype=torch.float32)
            c2_emb = torch.full((1, dim), 0.8, dtype=torch.float32)

            model.user_embeddings.copy_(u1_emb)
            model.product_embeddings.copy_(p1_emb)
            model.category_embeddings.copy_(torch.cat([c1_emb, c2_emb], dim=0))

            # Layer 0 parameters:
            # Product: W_self = 0.5 * I, b_self = 0.1
            model.self_weights_l0["Product"].copy_(torch.eye(dim) * 0.5)
            model.self_biases_l0["Product"].copy_(torch.full((dim,), 0.1))
            model.relation_weights_l0["rated"].copy_(torch.eye(dim) * 1.0)
            model.relation_weights_l0["contains_product"].copy_(torch.eye(dim) * 0.5)

            # User: W_self = 0.6 * I, b_self = 0.0
            model.self_weights_l0["User"].copy_(torch.eye(dim) * 0.6)
            model.self_biases_l0["User"].zero_()
            model.relation_weights_l0["rated_by"].copy_(torch.eye(dim) * 0.4)

            # Category: W_self = 1.0 * I, b_self = 0.0, W_belongs_to = 0.5 * I
            for cat in ["CategoryPath"]:
                model.self_weights_l0[cat].copy_(torch.eye(dim))
                model.self_biases_l0[cat].zero_()
                model.relation_weights_l0["belongs_to"].copy_(torch.eye(dim) * 0.5)

            # Layer 1 parameters:
            # Product: W_self = 0.8 * I, b_self = -0.05
            model.self_weights_l1["Product"].copy_(torch.eye(dim) * 0.8)
            model.self_biases_l1["Product"].copy_(torch.full((dim,), -0.05))
            model.relation_weights_l1["rated"].copy_(torch.eye(dim) * 0.2)
            model.relation_weights_l1["contains_product"].copy_(torch.eye(dim) * 0.1)

            # User: W_self = 1.0 * I, b_self = 0.0
            model.self_weights_l1["User"].copy_(torch.eye(dim) * 1.0)
            model.self_biases_l1["User"].zero_()
            model.relation_weights_l1["rated_by"].copy_(torch.eye(dim) * 0.5)

            # Category: W_self = 1.0 * I, b_self = 0.0, W_belongs_to = 0.2 * I
            for cat in ["CategoryPath"]:
                model.self_weights_l1[cat].copy_(torch.eye(dim))
                model.self_biases_l1[cat].zero_()
                model.relation_weights_l1["belongs_to"].copy_(torch.eye(dim) * 0.2)

        # UNMASKED FORWARD LITERAL ORACLE:
        # Layer 0:
        # For User u1:
        #   z_u1^(1) = 0.6 * 0.5 + 0.4 * 0.2 = 0.30 + 0.08 = 0.38 -> ReLU(0.38) = 0.38
        # For Category c1:
        #   z_c1^(1) = 1.0 * 0.4 + 0.5 * 0.2 = 0.40 + 0.10 = 0.50 -> ReLU(0.50) = 0.50
        # For Category c2:
        #   z_c2^(1) = 1.0 * 0.8 + 0.5 * 0.2 = 0.80 + 0.10 = 0.90 -> ReLU(0.90) = 0.90
        # For Product p1:
        #   Self: 0.5 * 0.2 + 0.1 = 0.20
        #   Relation rated: 1.0 * 0.5 = 0.50
        #   Relation contains_product: 0.5 * ((0.4 + 0.8) / 2) = 0.5 * 0.60 = 0.30
        #   Active relations: 2 -> mean neighbor = (0.50 + 0.30) / 2 = 0.40
        #   z_p1^(1) = 0.20 + 0.40 = 0.60 -> ReLU(0.60) = 0.60
        #
        # Layer 1:
        # For User u1:
        #   z_u1^(2) = 1.0 * 0.38 + 0.5 * 0.60 = 0.38 + 0.30 = 0.68 -> Identity: 0.68
        # For Product p1:
        #   Self: 0.8 * 0.60 - 0.05 = 0.48 - 0.05 = 0.43
        #   Relation rated: 0.2 * 0.38 = 0.076
        #   Relation contains_product: 0.1 * ((0.50 + 0.90) / 2) = 0.1 * 0.70 = 0.070
        #   Active relations: 2 -> mean neighbor = (0.076 + 0.070) / 2 = 0.073
        #   z_p1^(2) = 0.43 + 0.073 = 0.503 -> Identity: 0.503
        #
        # Raw dot product: 64 * (0.68 * 0.503) = 64 * 0.34204 = 21.89056

        embs = model.forward(mask_pairs=None)
        u1_out = embs["User"][0]
        p1_out = embs["Product"][0]

        self.assertTrue(torch.allclose(u1_out, torch.full((dim,), 0.68), atol=1e-5))
        self.assertTrue(torch.allclose(p1_out, torch.full((dim,), 0.503), atol=1e-5))

        raw_dot = torch.dot(u1_out, p1_out).item()
        self.assertAlmostEqual(raw_dot, 21.89056, places=4)

        # CLONED FIXTURE WITH EXTRA SAME-VALUED CATEGORY NEIGHBOR:
        # If we add c3 with embedding 0.6 (the mean of 0.4 and 0.8):
        # Within-relation mean of c1 (0.4), c2 (0.8), c3 (0.6) is (0.4 + 0.8 + 0.6) / 3 = 0.60.
        # Layer 0 output for c3: 1.0 * 0.6 + 0.5 * 0.2 = 0.60 + 0.10 = 0.70 -> ReLU = 0.70.
        # Layer 1 within-relation mean: (0.50 + 0.90 + 0.70) / 3 = 2.10 / 3 = 0.70.
        # Product p1's incoming message is IDENTICAL!
        p_cats_cloned = {"p1": ['c:["CatA"]', 'c:["CatB"]', 'c:["CatC"]']}
        g_cloned = build_hetero_graph(ctx, product_categories=p_cats_cloned)
        model_cloned = DifferentiableHeteroGNN(g_cloned, init_seed=42)

        with torch.no_grad():
            c3_emb = torch.full((1, dim), 0.6, dtype=torch.float32)
            model_cloned.user_embeddings.copy_(u1_emb)
            model_cloned.product_embeddings.copy_(p1_emb)
            model_cloned.category_embeddings.copy_(torch.cat([c1_emb, c2_emb, c3_emb], dim=0))

            # Copy all transformation parameters exactly
            for rel in RELATION_SCHEMAS:
                model_cloned.relation_weights_l0[rel].copy_(model.relation_weights_l0[rel])
                model_cloned.relation_weights_l1[rel].copy_(model.relation_weights_l1[rel])
            for ntype in NODE_TYPES:
                model_cloned.self_weights_l0[ntype].copy_(model.self_weights_l0[ntype])
                model_cloned.self_biases_l0[ntype].copy_(model.self_biases_l0[ntype])
                model_cloned.self_weights_l1[ntype].copy_(model.self_weights_l1[ntype])
                model_cloned.self_biases_l1[ntype].copy_(model.self_biases_l1[ntype])

        embs_cloned = model_cloned.forward(mask_pairs=None)
        self.assertTrue(torch.allclose(embs_cloned["Product"][0], torch.full((dim,), 0.503), atol=1e-5))

        # EXACT TWO-LAYER MASK ORACLE AND METADATA PRESERVATION:
        # Mask edge (u1, p1):
        # User u1:
        #   Layer 0: rated_by has 0 neighbors -> self-only: 0.6 * 0.5 = 0.30 -> ReLU(0.30) = 0.30
        #   Layer 1: rated_by has 0 neighbors -> self-only: 1.0 * 0.30 = 0.30 -> Identity: 0.30
        # Product p1:
        #   Layer 0: rated has 0 neighbors!
        #   Active relations: ONLY contains_product (1 active relation, recomputed!).
        #   Message contains_product = 0.5 * 0.60 = 0.30.
        #   Across active relations: mean neighbor = 0.30 / 1 = 0.30.
        #   z_p1^(1) = 0.20 + 0.30 = 0.50 -> ReLU(0.50) = 0.50.
        # Category c1 and c2:
        #   Incoming neighbor p1 (belongs_to is metadata, NOT masked!):
        #   z_c1^(1) = 0.40 + 0.5 * 0.20 = 0.50 -> ReLU = 0.50.
        #   z_c2^(1) = 0.80 + 0.5 * 0.20 = 0.90 -> ReLU = 0.90.
        # Product p1 at Layer 1:
        #   Self: 0.8 * 0.50 - 0.05 = 0.40 - 0.05 = 0.35.
        #   rated has 0 neighbors under mask.
        #   contains_product message: 0.1 * ((0.50 + 0.90) / 2) = 0.070.
        #   Active relations: 1 (only contains_product).
        #   z_p1^(2) = 0.35 + 0.070 = 0.420 -> Identity: 0.420.
        masked_embs = model.forward(mask_pairs={("u1", "p1")})
        self.assertTrue(torch.allclose(masked_embs["User"][0], torch.full((dim,), 0.30), atol=1e-5))
        self.assertTrue(torch.allclose(masked_embs["Product"][0], torch.full((dim,), 0.420), atol=1e-5))

    def test_empty_relation_self_only_output(self):
        records = [("u1", "p1", 5.0, 100), ("u_isolated", "p1", 1.0, 101)]
        ctx = TrainingContext(records, condition="rating_ge_4")
        g = build_hetero_graph(ctx)
        model = DifferentiableHeteroGNN(g, init_seed=42)

        embs = model.forward()
        u_iso_idx = g.node_to_idx["User"]["u_isolated"]
        h2_iso = embs["User"][u_iso_idx]

        h0 = model.user_embeddings[u_iso_idx]
        z1 = F.linear(h0, model.self_weights_l0["User"], model.self_biases_l0["User"])
        h1 = F.relu(z1)
        z2 = F.linear(h1, model.self_weights_l1["User"], model.self_biases_l1["User"])

        self.assertTrue(torch.allclose(h2_iso, z2, atol=1e-5))

    def test_relu_identity_activations(self):
        records = [("u1", "p1", 5.0, 100)]
        ctx = TrainingContext(records, condition="all_ratings")
        g = build_hetero_graph(ctx)
        model = DifferentiableHeteroGNN(g, init_seed=42)

        with torch.no_grad():
            model.self_biases_l0["User"].fill_(-100.0)
            model.self_biases_l1["User"].fill_(-50.0)

        embs = model.forward()
        u_vec = embs["User"][0]
        self.assertTrue((u_vec < 0.0).any())

    def test_batch_masking_effective_adjacency_and_no_cached_counts(self):
        records = [("u1", "p1", 5.0, 100), ("u1", "p2", 5.0, 101)]
        ctx = TrainingContext(records, condition="all_ratings")
        g = build_hetero_graph(ctx)
        model = DifferentiableHeteroGNN(g, init_seed=42)

        unmasked_embs = model.forward(mask_pairs=None)
        masked_embs = model.forward(mask_pairs={("u1", "p1")})

        u1_idx = g.node_to_idx["User"]["u1"]
        p1_idx = g.node_to_idx["Product"]["p1"]

        self.assertFalse(torch.equal(unmasked_embs["User"][u1_idx], masked_embs["User"][u1_idx]))
        self.assertFalse(torch.equal(unmasked_embs["Product"][p1_idx], masked_embs["Product"][p1_idx]))

        unmasked_again = model.forward(mask_pairs=None)
        self.assertTrue(torch.equal(unmasked_embs["User"], unmasked_again["User"]))

    def test_shared_forward_positive_negative_scoring_and_mock_spy(self):
        # u1 rated p1 (pos); p3 is warm and unrated by u1 (eligible neg)
        # u2 rated p2 (pos); p1 is warm and unrated by u2 (eligible neg)
        u_batch = ["u1", "u2"]
        pos_batch = ["p1", "p2"]
        neg_batch = ["p3", "p1"]

        model = DifferentiableHeteroGNN(self.graph, init_seed=42)

        # Mock / Spy on model.forward to verify it is called exactly once with union masks
        call_records: List[Any] = []
        real_forward = model.forward

        def forward_spy(mask_pairs=None):
            call_records.append(mask_pairs)
            return real_forward(mask_pairs=mask_pairs)

        model.forward = forward_spy  # type: ignore

        s_pos, s_neg = model.compute_batch_scores(u_batch, pos_batch, neg_batch)

        # Verify forward called exactly once with the union of positive mask pairs
        self.assertEqual(len(call_records), 1)
        expected_union_masks = {("u1", "p1"), ("u2", "p2")}
        self.assertEqual(call_records[0], expected_union_masks)

        self.assertEqual(s_pos.shape, (2,))
        self.assertEqual(s_neg.shape, (2,))
        self.assertTrue(torch.isfinite(s_pos).all())
        self.assertTrue(torch.isfinite(s_neg).all())

    def test_endpoint_regularization_exact_value(self):
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)

        # Batch: u1 -> pos p1, neg p3; u2 -> pos p2, neg p1
        user_ids = ["u1", "u2"]
        pos_items = ["p1", "p2"]
        neg_items = ["p3", "p1"]

        # Unique endpoints: U_B = {u1, u2} (2), I_B = {p1, p2, p3} (3) -> total 5 endpoints
        loss_dict = model.compute_batch_loss(user_ids, pos_items, neg_items, reg_coefficient=1e-5)
        reg_loss = loss_dict["reg_loss"].item()

        # Manual calculation:
        u1_row = model.user_embeddings[self.graph.node_to_idx["User"]["u1"]]
        u2_row = model.user_embeddings[self.graph.node_to_idx["User"]["u2"]]
        p1_row = model.product_embeddings[self.graph.node_to_idx["Product"]["p1"]]
        p2_row = model.product_embeddings[self.graph.node_to_idx["Product"]["p2"]]
        p3_row = model.product_embeddings[self.graph.node_to_idx["Product"]["p3"]]

        expected_sum_sq = (
            (u1_row ** 2).sum()
            + (u2_row ** 2).sum()
            + (p1_row ** 2).sum()
            + (p2_row ** 2).sum()
            + (p3_row ** 2).sum()
        )
        expected_reg = (1e-5 / 5.0) * expected_sum_sq.item()

        self.assertAlmostEqual(reg_loss, expected_reg, places=6)

    def test_ranking_backward_metadata_gradients_and_reg_isolation(self):
        """Autograd backward propagates non-zero gradients to metadata neighbors; reg loss isolates base rows."""
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)

        # Batch: u1 -> p1 (positive), p3 (negative)
        loss_dict = model.compute_batch_loss(
            user_ids=["u1"],
            positive_item_ids=["p1"],
            negative_item_ids=["p3"],
            reg_coefficient=1e-5,
        )

        # 1. Ranking-only backward:
        loss_dict["ranking_loss"].backward()

        u1_idx = self.graph.node_to_idx["User"]["u1"]
        p1_idx = self.graph.node_to_idx["Product"]["p1"]
        p3_idx = self.graph.node_to_idx["Product"]["p3"]

        self.assertIsNotNone(model.user_embeddings.grad)
        self.assertTrue((model.user_embeddings.grad[u1_idx] != 0).any())

        self.assertIsNotNone(model.product_embeddings.grad)
        self.assertTrue((model.product_embeddings.grad[p1_idx] != 0).any())
        self.assertTrue((model.product_embeddings.grad[p3_idx] != 0).any())

        # Neighbor user u2 and neighbor product p2 receive ranking gradients through graph propagation
        u2_idx = self.graph.node_to_idx["User"]["u2"]
        p2_idx = self.graph.node_to_idx["Product"]["p2"]
        self.assertTrue((model.user_embeddings.grad[u2_idx] != 0).any())
        self.assertTrue((model.product_embeddings.grad[p2_idx] != 0).any())

        # Metadata neighbor base rows receive gradients through message passing
        cat_idx = self.graph.node_to_idx["CategoryPath"]['c:["Electronics","Audio"]']
        self.assertIsNotNone(model.category_embeddings.grad)
        self.assertTrue((model.category_embeddings.grad[cat_idx] != 0).any())

        brand_idx = self.graph.node_to_idx["Brand"]["b:Sony"]
        self.assertIsNotNone(model.brand_embeddings.grad)
        self.assertTrue((model.brand_embeddings.grad[brand_idx] != 0).any())

        # Transformation matrices and biases receive gradients
        self.assertIsNotNone(model.relation_weights_l0["contains_product"].grad)
        self.assertTrue((model.relation_weights_l0["contains_product"].grad != 0).any())
        self.assertIsNotNone(model.self_weights_l0["Product"].grad)
        self.assertTrue((model.self_weights_l0["Product"].grad != 0).any())

        # 2. Regularization-only backward gradient isolation:
        model.zero_grad()
        loss_dict_reg = model.compute_batch_loss(
            user_ids=["u1"],
            positive_item_ids=["p1"],
            negative_item_ids=["p3"],
            reg_coefficient=1e-5,
        )
        loss_dict_reg["reg_loss"].backward()

        # Base user and product rows have non-zero gradients for endpoints only
        self.assertTrue((model.user_embeddings.grad[u1_idx] != 0).any())
        self.assertTrue((model.product_embeddings.grad[p1_idx] != 0).any())
        self.assertTrue((model.product_embeddings.grad[p3_idx] != 0).any())

        # Neighbor-only user and product rows have STRICTLY ZERO regularization gradient
        self.assertTrue((model.user_embeddings.grad[u2_idx] == 0).all())
        self.assertTrue((model.product_embeddings.grad[p2_idx] == 0).all())

        # Category and Brand embeddings have STRICTLY ZERO gradient
        if model.category_embeddings.grad is not None:
            self.assertTrue((model.category_embeddings.grad == 0).all())
        if model.brand_embeddings.grad is not None:
            self.assertTrue((model.brand_embeddings.grad == 0).all())

        # Relation transformation weights have STRICTLY ZERO gradient
        for rel in RELATION_SCHEMAS:
            if model.relation_weights_l0[rel].grad is not None:
                self.assertTrue((model.relation_weights_l0[rel].grad == 0).all())
            if model.relation_weights_l1[rel].grad is not None:
                self.assertTrue((model.relation_weights_l1[rel].grad == 0).all())

        # Self weights and biases have STRICTLY ZERO gradient
        for ntype in NODE_TYPES:
            if model.self_weights_l0[ntype].grad is not None:
                self.assertTrue((model.self_weights_l0[ntype].grad == 0).all())
            if model.self_biases_l0[ntype].grad is not None:
                self.assertTrue((model.self_biases_l0[ntype].grad == 0).all())

    def test_low_rated_globally_warm_negative_rejected(self):
        """A globally warm item rated below 4 by user u1 is strictly forbidden as a negative for u1."""
        records = [
            ("u1", "p1", 5.0, 100),
            ("u1", "p_low", 2.0, 101),  # Low training rating (< 4)
            ("u2", "p_low", 5.0, 102),  # p_low is GLOBALLY WARM under rating_ge_4
            ("u2", "p3", 5.0, 103),
        ]
        ctx = TrainingContext(records, condition="rating_ge_4")
        self.assertIn("p_low", ctx.sorted_warm_items)  # Globally warm

        g = build_hetero_graph(ctx)
        model = DifferentiableHeteroGNN(g, init_seed=42)

        # Attempting to use p_low as negative for u1 must fail because it's in u1's all_rated_history
        with self.assertRaises(ValueError):
            model.compute_batch_scores(["u1"], ["p1"], ["p_low"])

    def test_invalid_batch_and_tensor_rejections(self):
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)

        # Empty batch
        with self.assertRaises(ValueError):
            model.compute_batch_loss([], [], [])

        # Single string passed instead of sequence
        with self.assertRaises(ValueError):
            model.compute_batch_loss("u1", ["p1"], ["p3"])  # type: ignore
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["u1"], "p1", ["p3"])  # type: ignore

        # Length mismatch
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["u1"], ["p1"], [])

        # Unknown user
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["unknown_user"], ["p1"], ["p3"])

        # Unknown positive item
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["u1"], ["unknown_item"], ["p3"])

        # Positive item not in user's positive history (p3 was not rated by u1)
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["u1"], ["p3"], ["p2"])

        # Negative item in user's observed history (p2 was rated by u1, so forbidden as negative)
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["u1"], ["p1"], ["p2"])

        # Invalid mask pair
        with self.assertRaises(ValueError):
            model.forward(mask_pairs={("u1", "p3")})

        # Empty explicit mask rejected
        with self.assertRaises(ValueError):
            model.forward(mask_pairs=set())

        # Check CPU tensor validation helper rejects CUDA, meta, non-float32, non-finite
        with self.assertRaises(ValueError):
            check_cpu_tensor(torch.tensor([1, 2, 3], dtype=torch.long), "test")
        with self.assertRaises(ValueError):
            check_cpu_tensor(torch.tensor([float("nan")]), "test")
        with self.assertRaises(ValueError):
            check_cpu_tensor(torch.tensor([float("inf")]), "test")
        with self.assertRaises(ValueError):
            check_cpu_tensor(torch.empty(64, device="meta"), "test")

    def test_hetero_parameter_corruption_calls_forward(self):
        """Test calling HeteroGNN forward after parameter corruption."""
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)

        # Corrupt user_embeddings with NaN
        with torch.no_grad():
            model.user_embeddings[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt relation weight with Inf
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        with torch.no_grad():
            model.relation_weights_l0["rated"][0, 0] = float("inf")
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt self weight with wrong shape
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        model.self_weights_l0["Product"] = nn.Parameter(torch.zeros((32, 64)))
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt with empty-type matrix (shape (0, 64))
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        model.relation_weights_l1["rated"] = nn.Parameter(torch.empty((0, 64)))
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt self bias with wrong shape
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        model.self_biases_l0["User"] = nn.Parameter(torch.zeros(32))
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt with wrong dtype (float64)
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        model.user_embeddings = nn.Parameter(torch.zeros((len(self.graph.users), 64), dtype=torch.float64))
        with self.assertRaises(ValueError):
            model.forward()

        # Corrupt with meta tensor
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        model.user_embeddings = nn.Parameter(torch.empty((len(self.graph.users), 64), device="meta"))
        with self.assertRaises(ValueError):
            model.forward()

        # Missing relation weight
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        del model.relation_weights_l0["rated"]
        with self.assertRaises(ValueError):
            model.forward()

    def test_hetero_finite_float32_overflow(self):
        """Extreme finite values cause float32 overflow to Inf in forward or loss."""
        model = DifferentiableHeteroGNN(self.graph, init_seed=42)
        with torch.no_grad():
            model.user_embeddings.fill_(1e30)
            model.product_embeddings.fill_(1e30)
        with self.assertRaises(ValueError):
            model.compute_batch_loss(["u1"], ["p1"], ["p3"])

    def test_unrelated_interaction_and_metadata_preservation_under_union_masks(self):
        """Union batch masks recompute effective adjacency and preserve unrelated edges and metadata."""
        records = [
            ("u1", "p1", 5.0, 100),
            ("u2", "p2", 5.0, 101),
            ("u3", "p3", 5.0, 102),  # unrelated interaction
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        p_cats = {
            "p1": ['c:["Electronics","Audio"]'],
            "p2": ['c:["Electronics","Accessories"]'],
            "p3": ['c:["Electronics","Audio","Headphones"]'],
        }
        p_brands = {
            "p1": "b:Sony",
            "p2": "b:Bose",
            "p3": "b:Sennheiser",
        }
        g = build_hetero_graph(ctx, product_categories=p_cats, product_brands=p_brands)

        # Snapshot public graph state
        edges_before = {k: set(v) for k, v in g.edges.items()}
        incoming_before = {k: {dst: list(srcs) for dst, srcs in v.items()} for k, v in g.incoming_neighbors.items()}

        model = DifferentiableHeteroGNN(g, init_seed=42)

        # Apply union mask to (u1, p1) and (u2, p2)
        union_mask = {("u1", "p1"), ("u2", "p2")}
        masked_embs = model.forward(mask_pairs=union_mask)

        # Assert public graph inspection: edges and incoming_neighbors are NOT mutated
        edges_after = {k: set(v) for k, v in g.edges.items()}
        incoming_after = {k: {dst: list(srcs) for dst, srcs in v.items()} for k, v in g.incoming_neighbors.items()}
        self.assertEqual(edges_before, edges_after)
        self.assertEqual(incoming_before, incoming_after)

        # Metadata edges: belongs_to, contains_product, has_brand, brand_of, child_of, parent_of
        # are fully preserved
        self.assertIn(("p1", 'c:["Electronics","Audio"]'), g.edges["belongs_to"])
        self.assertIn(('c:["Electronics","Audio"]', "p1"), g.edges["contains_product"])
        self.assertIn(("p1", "b:Sony"), g.edges["has_brand"])
        self.assertIn(("b:Sony", "p1"), g.edges["brand_of"])
        self.assertIn(('c:["Electronics","Audio"]', 'c:["Electronics"]'), g.edges["category_child_of"])
        self.assertIn(('c:["Electronics"]', 'c:["Electronics","Audio"]'), g.edges["category_parent_of"])

        self.assertTrue(torch.isfinite(masked_embs["User"]).all())
        self.assertTrue(torch.isfinite(masked_embs["Product"]).all())
        self.assertTrue(torch.isfinite(masked_embs["CategoryPath"]).all())
        self.assertTrue(torch.isfinite(masked_embs["Brand"]).all())


class IntegrationImportTests(unittest.TestCase):
    """Verify main-checkout import order without relying on discovery state."""

    def test_fresh_process_package_and_top_level_import_orders(self):
        for first in ("differentiable_models", "scripts.differentiable_models"):
            with self.subTest(first=first):
                code = f'''
import importlib, sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
importlib.import_module({first!r})
import differentiable_models, scripts.differentiable_models
import learned_models, scripts.learned_models
import recommendation_core, scripts.recommendation_core
import scripts
assert scripts.differentiable_models is differentiable_models
assert scripts.learned_models is learned_models
assert scripts.recommendation_core is recommendation_core
assert differentiable_models.TrainingContext is recommendation_core.TrainingContext
assert learned_models.HeterogeneousKGNNConfig is recommendation_core.HeterogeneousKGNNConfig
'''
                result = subprocess.run(
                    [sys.executable, "-B", "-c", code],
                    cwd=ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
