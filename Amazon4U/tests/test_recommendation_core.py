"""Comprehensive regression fixtures for Amazon4U recommendation core milestone I1.

Python standard library only.
Verifies hand-calculated metrics, IDF, profiles, mixtures, fallbacks, negative sampling,
BPR loss, endpoint regularization, and strict held-out isolation.
"""

from collections import defaultdict
import math
from pathlib import Path
import random
import sys
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recommendation_core import (
    Evaluator,
    KGOnlyScorer,
    NegativeSampler,
    PopularityScorer,
    ScoringFailureError,
    TrainingContext,
    bpr_loss,
    endpoint_regularization,
    select_kg_only_mixture,
    stable_softplus,
)


class TrainingContextTests(unittest.TestCase):
    def test_duplicate_training_pair_rejected(self):
        records = [
            ("u1", "p1", 5.0),
            ("u1", "p1", 4.0),
        ]
        with self.assertRaises(ValueError):
            TrainingContext(records, condition="all_ratings")

    def test_invalid_identifiers_and_ratings_rejected(self):
        with self.assertRaises(ValueError):
            TrainingContext([("", "p1", 5.0)], condition="all_ratings")
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "  ", 5.0)], condition="all_ratings")
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "p1", float("nan"))], condition="all_ratings")
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "p1", float("inf"))], condition="all_ratings")
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "p1", True)], condition="all_ratings")
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "p1", False)], condition="all_ratings")
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "p1", 5.0)], condition="invalid_condition")

    def test_all_ratings_vs_rating_ge_4_conditions(self):
        records = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 2.0),
            ("u2", "p2", 4.0),
            ("u3", "p3", 1.0),
        ]
        # all_ratings condition
        ctx_all = TrainingContext(records, condition="all_ratings")
        self.assertEqual(ctx_all.warm_items, {"p1", "p2", "p3"})
        self.assertEqual(ctx_all.popularity, {"p1": 1, "p2": 2, "p3": 1})
        self.assertEqual(ctx_all.positive_history["u1"], {"p1", "p2"})
        self.assertEqual(ctx_all.all_rated_history["u1"], {"p1", "p2"})
        self.assertEqual(ctx_all.positive_users, {"u1", "u2", "u3"})

        # rating_ge_4 condition
        ctx_ge4 = TrainingContext(records, condition="rating_ge_4")
        self.assertEqual(ctx_ge4.warm_items, {"p1", "p2"})  # p3 only had rating 1.0, not warm!
        self.assertEqual(ctx_ge4.popularity, {"p1": 1, "p2": 1})  # p2 has only one >= 4 rating (u2)
        self.assertEqual(ctx_ge4.positive_history["u1"], {"p1"})  # p2 was rating 2.0
        self.assertEqual(ctx_ge4.all_rated_history["u1"], {"p1", "p2"})  # preserves all rated
        self.assertEqual(ctx_ge4.positive_history["u3"], set())  # empty positive history
        self.assertEqual(ctx_ge4.all_rated_history["u3"], {"p3"})
        self.assertEqual(ctx_ge4.positive_users, {"u1", "u2"})


class PopularityScorerTests(unittest.TestCase):
    def test_popularity_scorer_returns_non_personalized_route(self):
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        scorer = PopularityScorer(ctx)
        scores, route, reason = scorer.score_candidates("u1", ["p1", "p2"])
        self.assertEqual(route, "non_personalized")
        self.assertIsNone(reason)
        self.assertEqual(scores, {"p1": 2.0, "p2": 1.0})


class KGOnlyScorerTests(unittest.TestCase):
    def test_smoothed_idf_and_distinct_product_df(self):
        # 3 warm items: p1, p2, p3
        # cat A in p1 and p2; p1 has duplicate 'cat_A'
        # cat B in p3 only
        # brand X in p1 only
        # p2 has no brand (missing channel)
        # p3 has brand Y
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {
            "p1": ["cat_A", "cat_A", "cat_B"],
            "p2": ["cat_A"],
            "p3": ["cat_B"],
        }
        item_brands = {
            "p1": "brand_X",
            "p2": None,
            "p3": "brand_Y",
        }
        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(1.0, 0.0))

        # |W| = 3
        # cat_A: df=2 (p1, p2). Formula: 1 + ln((3+1)/(2+1)) = 1 + ln(4/3)
        expected_idf_cat_A = 1.0 + math.log(4.0 / 3.0)
        self.assertAlmostEqual(scorer.idf_category["cat_A"], expected_idf_cat_A, places=9)

        # cat_B: df=2 (p1, p3). Formula: 1 + ln((3+1)/(2+1)) = 1 + ln(4/3)
        self.assertAlmostEqual(scorer.idf_category["cat_B"], expected_idf_cat_A, places=9)

        # brand_X: df=1 (p1). Formula: 1 + ln((3+1)/(1+1)) = 1 + ln(2)
        expected_idf_brand_X = 1.0 + math.log(2.0)
        self.assertAlmostEqual(scorer.idf_brand["brand_X"], expected_idf_brand_X, places=9)

    def test_independent_channel_normalization_and_user_profiles(self):
        records = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 5.0),
            ("u2", "p3", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {
            "p1": ["c1"],
            "p2": ["c2"],
            "p3": ["c1", "c2"],
        }
        item_brands = {
            "p1": "b1",
            "p2": "b1",
            "p3": None,
        }
        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(0.75, 0.25))

        for p in ("p1", "p2", "p3"):
            v_cat = scorer.product_cat_vec[p]
            norm_cat = math.sqrt(sum(v * v for v in v_cat.values()))
            self.assertAlmostEqual(norm_cat, 1.0, places=9)

        self.assertEqual(scorer.product_brand_vec["p1"], {"b1": 1.0})
        self.assertEqual(scorer.product_brand_vec["p2"], {"b1": 1.0})
        self.assertEqual(scorer.product_brand_vec["p3"], {})

        v_u1_cat = scorer.user_cat_vec["u1"]
        norm_u1_cat = math.sqrt(sum(v * v for v in v_u1_cat.values()))
        self.assertAlmostEqual(norm_u1_cat, 1.0, places=9)
        self.assertEqual(scorer.user_brand_vec["u1"], {"b1": 1.0})

    def test_hand_computed_asymmetric_idf_and_profiles(self):
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
            ("u4", "p4", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {
            "p1": ["c1", "c2"],
            "p2": ["c1"],
            "p3": ["c1"],
            "p4": ["c2", "c3"],
        }
        item_brands = {
            "p1": "b1",
            "p2": "b1",
            "p3": "b2",
            "p4": None,
        }
        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(0.75, 0.25))

        idf_c1 = 1.0 + math.log(5.0 / 4.0)
        idf_c2 = 1.0 + math.log(5.0 / 3.0)
        idf_c3 = 1.0 + math.log(5.0 / 2.0)

        self.assertAlmostEqual(scorer.idf_category["c1"], idf_c1, places=9)
        self.assertAlmostEqual(scorer.idf_category["c2"], idf_c2, places=9)
        self.assertAlmostEqual(scorer.idf_category["c3"], idf_c3, places=9)

        norm_p1 = math.sqrt(idf_c1 * idf_c1 + idf_c2 * idf_c2)
        self.assertAlmostEqual(scorer.product_cat_vec["p1"]["c1"], idf_c1 / norm_p1, places=9)
        self.assertAlmostEqual(scorer.product_cat_vec["p1"]["c2"], idf_c2 / norm_p1, places=9)

    def test_cold_metadata_invariance(self):
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        base_cats = {"p1": ["c1"], "p2": ["c2"]}
        base_brands = {"p1": "b1", "p2": "b2"}
        scorer_base = KGOnlyScorer(ctx, base_cats, base_brands, mixture=(0.5, 0.5))

        extra_cats = dict(base_cats)
        extra_cats["p_cold_999"] = ["c1", "c_unknown_999"]
        extra_brands = dict(base_brands)
        extra_brands["p_cold_999"] = "brand_unknown_999"
        scorer_extra = KGOnlyScorer(ctx, extra_cats, extra_brands, mixture=(0.5, 0.5))

        self.assertEqual(scorer_base.idf_category, scorer_extra.idf_category)
        self.assertEqual(scorer_base.idf_brand, scorer_extra.idf_brand)
        self.assertEqual(scorer_base.product_cat_vec, scorer_extra.product_cat_vec)
        self.assertEqual(scorer_base.product_brand_vec, scorer_extra.product_brand_vec)
        self.assertEqual(scorer_base.user_cat_vec, scorer_extra.user_cat_vec)

    def test_invalid_identity_mappings_rejected(self):
        records = [("u1", "p1", 5.0)]
        ctx = TrainingContext(records, condition="all_ratings")
        with self.assertRaises(ValueError):
            KGOnlyScorer(ctx, {"p1": "not_a_collection"}, {"p1": "b1"})
        with self.assertRaises(ValueError):
            KGOnlyScorer(ctx, {"p1": ["  "]}, {"p1": "b1"})
        with self.assertRaises(ValueError):
            KGOnlyScorer(ctx, {"p1": ["c1"]}, {"p1": "  "})

    def test_corrupt_nonfinite_profile_detected_as_failure_even_unused_channel(self):
        records = [("u1", "p1", 5.0)]
        ctx = TrainingContext(records, condition="all_ratings")
        scorer = KGOnlyScorer(ctx, {"p1": ["c1"]}, {"p1": "b1"}, mixture=(1.0, 0.0))
        scorer.user_brand_vec["u1"]["b1"] = float("nan")
        with self.assertRaises(ScoringFailureError):
            scorer.score_candidates("u1", ["p1"])

    def test_missing_channel_cosine_zero_and_fixed_weight_combination(self):
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {"p1": ["c1"], "p2": ["c1"]}
        item_brands = {"p1": "b1", "p2": None}

        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(0.75, 0.25))
        scores, route, reason = scorer.score_candidates("u1", ["p2"])

        self.assertEqual(route, "personalized")
        self.assertIsNone(reason)
        self.assertAlmostEqual(scores["p2"], 0.75, places=9)

    def test_absent_item_channels_ordinary_zero_score(self):
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {"p1": ["c1"], "p2": []}
        item_brands = {"p1": "b1", "p2": None}

        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(0.5, 0.5))
        scores, route, reason = scorer.score_candidates("u1", ["p2"])
        self.assertEqual(route, "personalized")
        self.assertIsNone(reason)
        self.assertEqual(scores["p2"], 0.0)

    def test_whole_user_fallback_for_zero_positively_weighted_channels(self):
        records = [
            ("u_empty", "p1", 2.0),
            ("u_brand_only", "p2", 5.0),
            ("u_normal", "p1", 5.0),
        ]
        ctx = TrainingContext(records, condition="rating_ge_4")
        item_categories = {"p1": ["c1"], "p2": []}
        item_brands = {"p1": None, "p2": "b1"}

        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(1.0, 0.0))

        scores_empty, route_empty, reason_empty = scorer.score_candidates("u_empty", ["p1", "p2"])
        self.assertEqual(route_empty, "fallback")
        self.assertEqual(reason_empty, "empty_positive_history")
        self.assertEqual(scores_empty["p1"], 1.0)
        self.assertEqual(scores_empty["p2"], 1.0)

        scores_brand, route_brand, reason_brand = scorer.score_candidates("u_brand_only", ["p1", "p2"])
        self.assertEqual(route_brand, "fallback")
        self.assertEqual(reason_brand, "unusable_metadata_profile")
        self.assertEqual(scores_brand["p1"], 1.0)
        self.assertEqual(scores_brand["p2"], 1.0)

    def test_mixture_selection_on_validation_ndcg10_and_exact_tie(self):
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {"p1": ["c1"], "p2": ["c1"], "p3": ["c1"]}
        item_brands = {"p1": "b1", "p2": "b1", "p3": "b1"}
        validation_targets = [
            ("u1", "p2", 5.0),
            ("u2", "p3", 5.0),
        ]
        best_mixture, trials = select_kg_only_mixture(ctx, item_categories, item_brands, validation_targets)
        self.assertEqual(best_mixture, (1.0, 0.0))
        self.assertEqual(len(trials), 3)

    def test_mixture_selection_fails_on_all_na_population(self):
        records = [
            ("u1", "p1", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {"p1": ["c1"]}
        item_brands = {"p1": "b1"}
        validation_targets = [("u1", "p_cold", 5.0)]
        with self.assertRaises(ValueError):
            select_kg_only_mixture(ctx, item_categories, item_brands, validation_targets)

    def test_direct_candidate_profile_corruption_fails(self):
        records = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        scorer = KGOnlyScorer(ctx, {"p1": ["c1"], "p2": ["c1"]}, {"p1": "b1", "p2": "b1"}, mixture=(1.0, 0.0))

        # Corrupt candidate category profile
        scorer.product_cat_vec["p2"]["c1"] = float("nan")
        with self.assertRaises(ScoringFailureError):
            scorer.score_candidates("u1", ["p2"])

        scorer.product_cat_vec["p2"]["c1"] = float("inf")
        with self.assertRaises(ScoringFailureError):
            scorer.score_candidates("u1", ["p2"])

        # Reset category profile and corrupt brand profile
        scorer.product_cat_vec["p2"]["c1"] = 1.0
        scorer.product_brand_vec["p2"]["b1"] = float("nan")
        with self.assertRaises(ScoringFailureError):
            scorer.score_candidates("u1", ["p2"])

    def test_multi_positive_asymmetric_user_profile_oracle(self):
        records = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 5.0),
            ("u2", "p3", 5.0),
        ]
        ctx = TrainingContext(records, condition="all_ratings")
        item_categories = {
            "p1": ["c1", "c2"],
            "p2": ["c2", "c3"],
            "p3": ["c1", "c3"],
        }
        item_brands = {
            "p1": "b1",
            "p2": None,
            "p3": "b1",
        }
        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(0.75, 0.25))

        # Hand calculations:
        # 3 warm items: p1, p2, p3
        # df(c1)=2, df(c2)=2, df(c3)=2 => idf = 1 + ln(4/3) for all cats.
        # df(b1)=2 => idf(b1) = 1 + ln(4/3).
        # Since IDF is constant across all categories:
        # p1 cat vec norm = sqrt(idf^2 + idf^2) = idf * sqrt(2)
        # p1 normalized cat vec: {c1: 1/sqrt(2), c2: 1/sqrt(2)}
        # p2 normalized cat vec: {c2: 1/sqrt(2), c3: 1/sqrt(2)}
        # p3 normalized cat vec: {c1: 1/sqrt(2), c3: 1/sqrt(2)}
        #
        # User u1 has distinct positive items p1, p2.
        # u1 cat unnormalized sum:
        #   c1: 1/sqrt(2)
        #   c2: 1/sqrt(2) + 1/sqrt(2) = 2/sqrt(2) = sqrt(2)
        #   c3: 1/sqrt(2)
        # u1 cat sum norm = sqrt((1/2) + 2 + (1/2)) = sqrt(3)
        # u1 normalized cat profile:
        #   c1: (1/sqrt(2)) / sqrt(3) = 1 / sqrt(6)
        #   c2: sqrt(2) / sqrt(3) = sqrt(2/3)
        #   c3: (1/sqrt(2)) / sqrt(3) = 1 / sqrt(6)
        #
        # u1 brand sum: {b1: 1.0} => norm = 1.0 => u1 brand profile: {b1: 1.0}
        #
        # Candidate p3:
        # p3 cat similarity with u1:
        #   common cats: c1, c3
        #   cat_cosine = (1/sqrt(6)) * (1/sqrt(2)) + (1/sqrt(6)) * (1/sqrt(2))
        #              = 2 / sqrt(12) = 2 / (2 * sqrt(3)) = 1 / sqrt(3)
        # p3 brand similarity with u1:
        #   common brand: b1 => brand_cosine = 1.0 * 1.0 = 1.0
        #
        # Combined score with mixture (0.75, 0.25):
        expected_score = 0.75 * (1.0 / math.sqrt(3.0)) + 0.25 * 1.0

        scores, route, reason = scorer.score_candidates("u1", ["p3"])
        self.assertEqual(route, "personalized")
        self.assertIsNone(reason)
        self.assertAlmostEqual(scores["p3"], expected_score, places=9)



class EvaluatorTests(unittest.TestCase):
    def test_hand_calculated_ndcg_and_hitrate(self):
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
            ("u4", "p4", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        class MockScorer:
            def score_candidates(self, u, cands):
                scores = {"p3": 3.0, "p2": 2.0, "p4": 1.0}
                return scores, "personalized", None

        val_targets = [("u1", "p2", 5.0)]
        res = evaluator.evaluate(val_targets, MockScorer(), split="validation")

        expected_ndcg = 1.0 / math.log2(3.0)
        self.assertAlmostEqual(res["overall_metrics"]["NDCG@5"], expected_ndcg, places=9)
        self.assertAlmostEqual(res["overall_metrics"]["NDCG@10"], expected_ndcg, places=9)
        self.assertAlmostEqual(res["overall_metrics"]["HitRate@5"], 1.0, places=9)
        u_res = res["per_user_results"][0]
        self.assertIn("top_5", u_res)
        self.assertIn("top_10", u_res)
        self.assertIn("top_20", u_res)

    def test_exact_score_tie_parent_asin_ascending(self):
        train = [
            ("u1", "p_train", 5.0),
            ("u2", "b_item", 5.0),
            ("u3", "a_item", 5.0),
            ("u4", "c_item", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        class TieScorer:
            def score_candidates(self, u, cands):
                return {item: 1.0 for item in cands}, "personalized", None

        val_targets = [("u1", "c_item", 5.0)]
        res = evaluator.evaluate(val_targets, TieScorer(), split="validation")
        user_res = res["per_user_results"][0]
        self.assertEqual(user_res["top_10"][:3], ["a_item", "b_item", "c_item"])
        expected_ndcg = 1.0 / math.log2(3.0 + 1.0)
        self.assertAlmostEqual(user_res["metrics"]["NDCG@5"], 0.5, places=9)

    def test_validation_vs_test_history_filters_and_validation_history_requirements(self):
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)
        scorer = PopularityScorer(ctx)

        val_targets = [("u1", "p2", 5.0)]
        val_res = evaluator.evaluate(val_targets, scorer, split="validation")
        self.assertEqual(val_res["audit"]["eligible_users"], 1)

        with self.assertRaises(ValueError):
            evaluator.evaluate(val_targets, scorer, split="test", validation_history=None)

        with self.assertRaises(ValueError):
            evaluator.evaluate(val_targets, scorer, split="test", validation_history={})

        with self.assertRaises(ValueError):
            evaluator.evaluate(val_targets, scorer, split="test", validation_history={"u1": "p2"})

        test_targets = [("u1", "p2", 5.0)]
        val_history = {"u1": ["p2"]}
        test_res = evaluator.evaluate(test_targets, scorer, split="test", validation_history=val_history)
        self.assertEqual(test_res["audit"]["eligible_users"], 0)
        self.assertEqual(test_res["audit"]["history_overlaps"], 1)

        val_history_low = {"u1": ["p3"]}
        test_targets_low = [("u1", "p2", 5.0)]
        test_res_low = evaluator.evaluate(test_targets_low, scorer, split="test", validation_history=val_history_low)
        self.assertEqual(test_res_low["audit"]["eligible_users"], 1)
        self.assertEqual(test_res_low["per_user_results"][0]["candidate_count"], 1)

    def test_eligibility_and_overlapping_exclusion_audit(self):
        train = [
            ("u_low", "p_warm2", 2.0),
            ("u_overlap", "p_warm", 5.0),
            ("u_cold", "p_warm", 5.0),
            ("u_irr_cold", "p_warm", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        evaluator = Evaluator(ctx)
        scorer = PopularityScorer(ctx)

        targets = [
            ("u_low", "p_warm", 5.0),
            ("u_unknown", "p_warm", 5.0),
            ("u_irr_cold", "p_warm2", 2.0),
            ("u_cold", "p_cold", 5.0),
            ("u_overlap", "p_warm", 5.0),
        ]
        res = evaluator.evaluate(targets, scorer, split="validation")
        audit = res["audit"]
        self.assertEqual(audit["total_targets"], 5)
        self.assertEqual(audit["eligible_users"], 1)
        self.assertEqual(audit["unknown_users"], 1)
        self.assertEqual(audit["cold_targets"], 2)
        self.assertEqual(audit["irrelevant_targets"], 1)
        self.assertEqual(audit["history_overlaps"], 1)
        self.assertIn("cold_target+irrelevant_target", audit["overlapping_reasons"])

    def test_empty_positive_history_user_does_not_call_scorer(self):
        train = [
            ("u_normal", "p1", 5.0),
            ("u_empty", "p1", 2.0),
            ("u_other", "p2", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        evaluator = Evaluator(ctx)

        class BombScorer:
            def score_candidates(self, u, cands):
                raise RuntimeError("Personalized scorer should NOT be called for empty positive history user!")

        targets = [("u_empty", "p2", 5.0)]
        res = evaluator.evaluate(targets, BombScorer(), split="validation")
        self.assertEqual(res["audit"]["eligible_users"], 1)
        user_res = res["per_user_results"][0]
        self.assertEqual(user_res["route"], "fallback")
        self.assertEqual(user_res["fallback_reason"], "empty_positive_history")
        self.assertEqual(res["subgroups"]["fallback"]["count"], 1)
        self.assertEqual(res["subgroups"]["personalized"]["count"], 0)
        self.assertIsNone(res["subgroups"]["personalized"]["metrics"]["NDCG@10"])

    def test_candidate_shortage_and_no_padding(self):
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)
        scorer = PopularityScorer(ctx)

        targets = [("u1", "p2", 5.0)]
        res = evaluator.evaluate(targets, scorer, split="validation")
        self.assertEqual(res["candidate_shortages"][5], 1)
        self.assertEqual(res["candidate_shortages"][10], 1)
        self.assertEqual(res["candidate_shortages"][20], 1)
        self.assertEqual(len(res["per_user_results"][0]["top_10"]), 1)

    def test_nonfinite_scoring_failure_raises_error(self):
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        class NanScorer:
            def score_candidates(self, u, cands):
                return {c: float("nan") for c in cands}, "personalized", None

        targets = [("u1", "p2", 5.0)]
        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate(targets, NanScorer(), split="validation")

    def test_evaluator_validates_inputs_and_bool_ratings(self):
        train = [("u1", "p1", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)
        scorer = PopularityScorer(ctx)

        with self.assertRaises(ValueError):
            evaluator.evaluate([("u1", "p1", True)], scorer, split="validation")
        with self.assertRaises(ValueError):
            evaluator.evaluate([("u1", "p1", 5.0), ("u1", "p2", 5.0)], scorer, split="validation")

    def test_scorer_failure_and_missing_candidates(self):
        train = [("u1", "p1", 5.0), ("u2", "p2", 5.0), ("u3", "p3", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        class IncompleteScorer:
            def score_candidates(self, u, cands):
                return {"p2": 1.0}, "personalized", None

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], IncompleteScorer(), split="validation")

        class BuggyScorer:
            def score_candidates(self, u, cands):
                raise ZeroDivisionError("division by zero")

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], BuggyScorer(), split="validation")

    def test_filter_before_scoring(self):
        train = [("u1", "p1", 5.0), ("u2", "p2", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        received_candidates = []

        class InspectScorer:
            def score_candidates(self, u, cands):
                received_candidates.extend(cands)
                return {c: 1.0 for c in cands}, "personalized", None

        evaluator.evaluate([("u1", "p2", 5.0)], InspectScorer(), split="validation")
        self.assertNotIn("p1", received_candidates)
        self.assertIn("p2", received_candidates)

    def test_catalog_coverage_calculation(self):
        train_exact = [("u1", "p_init", 5.0)]
        for i in range(19):
            train_exact.append((f"user_{i}", f"p_{i:02d}", 5.0))
        ctx_exact = TrainingContext(train_exact, condition="all_ratings")
        evaluator_exact = Evaluator(ctx_exact)

        class TopTenScorer:
            def score_candidates(self, u, cands):
                intended = {f"p_{i:02d}": 100.0 - i for i in range(10)}
                scores = {}
                for c in cands:
                    if c in intended:
                        scores[c] = intended[c]
                    else:
                        scores[c] = 1.0
                return scores, "personalized", None

        targets = [("user_10", "p_00", 5.0), ("user_11", "p_00", 5.0)]
        res = evaluator_exact.evaluate(targets, TopTenScorer(), split="validation")
        self.assertEqual(len(ctx_exact.warm_items), 20)
        self.assertAlmostEqual(res["overall_metrics"]["CatalogCoverage@10"], 0.5, places=9)
        self.assertAlmostEqual(res["subgroups"]["personalized"]["metrics"]["CatalogCoverage@10"], 0.5, places=9)

    def test_empty_eligible_population_metrics_are_none(self):
        train = [("u1", "p1", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)
        targets = [("u1", "p_cold", 5.0)]
        res = evaluator.evaluate(targets, PopularityScorer(ctx), split="validation")
        self.assertEqual(res["audit"]["eligible_users"], 0)
        self.assertIsNone(res["overall_metrics"]["NDCG@10"])
        self.assertIsNone(res["overall_metrics"]["HitRate@10"])
        self.assertIsNone(res["overall_metrics"]["CatalogCoverage@10"])

    def test_subgroup_metrics_and_denominators(self):
        train = [
            ("u_norm", "p1", 5.0),
            ("u_empty", "p1", 2.0),
            ("u_other", "p2", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        evaluator = Evaluator(ctx)

        class MockScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "personalized", None

        targets = [("u_norm", "p2", 5.0), ("u_empty", "p2", 5.0)]
        res = evaluator.evaluate(targets, MockScorer(), split="validation")
        audit = res["audit"]
        self.assertEqual(audit["total_targets"], 2)
        self.assertEqual(audit["eligible_users"], 2)
        self.assertEqual(audit["excluded_targets"], 0)
        self.assertEqual(audit["warm_pool_size"], 2)

        subgroups = res["subgroups"]
        self.assertEqual(subgroups["personalized"]["count"], 1)
        self.assertEqual(subgroups["fallback"]["count"], 1)
        self.assertIsNotNone(subgroups["personalized"]["metrics"]["NDCG@10"])
        self.assertIsNotNone(subgroups["fallback"]["metrics"]["NDCG@10"])
        self.assertIn("CatalogCoverage@10", subgroups["fallback"]["metrics"])
        self.assertIn("empty_positive_history", subgroups["fallback"]["by_reason"])

    def test_evaluator_malformed_route_and_reason_rejections(self):
        train = [("u1", "p1", 5.0), ("u2", "p2", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        # 1. Invalid route name
        class InvalidRouteScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "invalid_route", None

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], InvalidRouteScorer(), split="validation")

        # 2. Non-fallback route with non-None fallback_reason
        class RouteReasonMismatchScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "personalized", "empty_positive_history"

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], RouteReasonMismatchScorer(), split="validation")

        # 3. Unhashable fallback reason (list or dict)
        class UnhashableReasonScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "fallback", ["empty_positive_history"]

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], UnhashableReasonScorer(), split="validation")

        # 4. Invalid scoring_error fallback reason
        class ScoringErrorReasonScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "fallback", "scoring_error"

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], ScoringErrorReasonScorer(), split="validation")

        # 5. Empty-positive fallback reported by scorer for user with positive history
        class BogusEmptyHistoryScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "fallback", "empty_positive_history"

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], BogusEmptyHistoryScorer(), split="validation")

        # 6. Metadata fallback reported by non-KGOnly scorer
        class BogusMetadataFallbackScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "fallback", "unusable_metadata_profile"

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], BogusMetadataFallbackScorer(), split="validation")

    def test_test_split_filters_low_rated_validation_records_in_rating_ge_4(self):
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        evaluator = Evaluator(ctx)
        scorer = PopularityScorer(ctx)

        # Genuine low-rated validation records (rating 1.0 < 4.0)
        val_records_tuples = [("u1", "p2", 1.0)]
        test_targets = [("u1", "p3", 5.0)]
        res_tuple = evaluator.evaluate(test_targets, scorer, split="test", validation_history=val_records_tuples)
        self.assertEqual(res_tuple["audit"]["eligible_users"], 1)
        self.assertEqual(res_tuple["per_user_results"][0]["candidate_count"], 1)
        self.assertEqual(res_tuple["per_user_results"][0]["top_10"], ["p3"])

        # Dict format of records
        val_records_dicts = [{"user_id": "u1", "parent_asin": "p2", "rating": 1.0}]
        res_dict = evaluator.evaluate(test_targets, scorer, split="test", validation_history=val_records_dicts)
        self.assertEqual(res_dict["audit"]["eligible_users"], 1)
        self.assertEqual(res_dict["per_user_results"][0]["candidate_count"], 1)

        # If test target is the low-rated validation item, it is detected as history overlap
        res_overlap = evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="test", validation_history=val_records_tuples)
        self.assertEqual(res_overlap["audit"]["eligible_users"], 0)
        self.assertEqual(res_overlap["audit"]["history_overlaps"], 1)

    def test_macro_metrics_multi_user_and_k_boundaries(self):
        # 3 users, 25 warm items: p_01 to p_25
        # Each user rated p_01 in train (filtered out)
        # Candidate pool for each user has 24 items: p_02 to p_25
        train = [("u1", "p_01", 5.0), ("u2", "p_01", 5.0), ("u3", "p_01", 5.0)]
        for i in range(2, 26):
            train.append((f"user_{i}", f"p_{i:02d}", 5.0))
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        # Custom deterministic scorer:
        # u1 target is p_05. Scored such that p_05 is rank 4 (within top 5, 10, 20).
        # u2 target is p_08. Scored such that p_08 is rank 7 (outside top 5, inside top 10, 20).
        # u3 target is p_16. Scored such that p_16 is rank 15 (outside top 5 and 10, inside top 20).
        class BoundaryScorer:
            def score_candidates(self, u, cands):
                sorted_c = sorted(cands)
                scores = {}
                for idx, c in enumerate(sorted_c):
                    scores[c] = float(100 - idx)
                return scores, "personalized", None

        targets = [
            ("u1", "p_05", 5.0),
            ("u2", "p_08", 5.0),
            ("u3", "p_16", 5.0),
        ]
        res = evaluator.evaluate(targets, BoundaryScorer(), split="validation")
        self.assertEqual(res["audit"]["eligible_users"], 3)

        # Expected individual metrics:
        # u1: rank 4.
        #   Hit@5=1, Hit@10=1, Hit@20=1
        #   NDCG@5 = 1/log2(5), NDCG@10 = 1/log2(5), NDCG@20 = 1/log2(5)
        # u2: rank 7.
        #   Hit@5=0, Hit@10=1, Hit@20=1
        #   NDCG@5 = 0, NDCG@10 = 1/log2(8) = 1/3, NDCG@20 = 1/3
        # u3: rank 15.
        #   Hit@5=0, Hit@10=0, Hit@20=1
        #   NDCG@5 = 0, NDCG@10 = 0, NDCG@20 = 1/log2(16) = 1/4 = 0.25

        # Macro averages across 3 users:
        exp_hr_5 = (1.0 + 0.0 + 0.0) / 3.0
        exp_hr_10 = (1.0 + 1.0 + 0.0) / 3.0
        exp_hr_20 = (1.0 + 1.0 + 1.0) / 3.0

        exp_ndcg_5 = (1.0 / math.log2(5.0)) / 3.0
        exp_ndcg_10 = (1.0 / math.log2(5.0) + (1.0 / 3.0)) / 3.0
        exp_ndcg_20 = (1.0 / math.log2(5.0) + (1.0 / 3.0) + 0.25) / 3.0

        ov = res["overall_metrics"]
        self.assertAlmostEqual(ov["HitRate@5"], exp_hr_5, places=9)
        self.assertAlmostEqual(ov["HitRate@10"], exp_hr_10, places=9)
        self.assertAlmostEqual(ov["HitRate@20"], exp_hr_20, places=9)

        self.assertAlmostEqual(ov["NDCG@5"], exp_ndcg_5, places=9)
        self.assertAlmostEqual(ov["NDCG@10"], exp_ndcg_10, places=9)
        self.assertAlmostEqual(ov["NDCG@20"], exp_ndcg_20, places=9)


class NegativeSamplerTests(unittest.TestCase):
    def test_negative_pool_excludes_all_training_rated_items_and_below4_exclusion(self):
        train = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 2.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
            ("u4", "p4", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        sampler = NegativeSampler(ctx)
        self.assertEqual(sampler.get_negative_pool("u1"), ["p3", "p4"])

    def test_uniform_draw_and_skip_on_empty_pool(self):
        train = [
            ("u_all", "p1", 5.0),
            ("u_all", "p2", 5.0),
            ("u_other", "p1", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        sampler = NegativeSampler(ctx)
        self.assertEqual(sampler.get_negative_pool("u_all"), [])

        rng = random.Random(42)
        self.assertIsNone(sampler.sample_negative("u_all", rng))

        positive_pairs = [("u_all", "p1"), ("u_other", "p1")]
        trainable, skipped = sampler.prepare_batch(positive_pairs, rng)
        self.assertEqual(skipped, 1)
        self.assertEqual(len(trainable), 1)
        self.assertEqual(trainable[0][0], "u_other")
        self.assertEqual(trainable[0][2], "p2")

    def test_reject_positive_pair_not_in_user_positive_history(self):
        train = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 2.0),
            ("u2", "p3", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        sampler = NegativeSampler(ctx)
        rng = random.Random(42)

        with self.assertRaises(ValueError):
            sampler.prepare_batch([("u1", "p2")], rng)

        with self.assertRaises(ValueError):
            sampler.prepare_batch([("u_unknown", "p1")], rng)

    def test_globally_trainable_vs_locally_all_skipped_batch(self):
        train = [
            ("u_all", "p1", 5.0),
            ("u_all", "p2", 5.0),
            ("u_trainable", "p1", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        sampler = NegativeSampler(ctx)
        rng = random.Random(42)

        trainable, skipped = sampler.prepare_batch([("u_all", "p1")], rng)
        self.assertEqual(trainable, [])
        self.assertEqual(skipped, 1)

    def test_globally_exhausted_dataset_fails_preflight(self):
        train = [("u_only", "p1", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        with self.assertRaises(ValueError):
            NegativeSampler(ctx)

    def test_negative_sampler_rating_ge_4_preflight_cold_observations_regression(self):
        # Regression case:
        # u1 rates a=5, x/y/z=2; u2 rates a/b/c=5.
        # W={a,b,c}; only u1 has trainable positives and its negatives are [b,c], so constructor MUST succeed.
        train = [
            ("u1", "a", 5.0),
            ("u1", "x", 2.0),
            ("u1", "y", 2.0),
            ("u1", "z", 2.0),
            ("u2", "a", 5.0),
            ("u2", "b", 5.0),
            ("u2", "c", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        self.assertEqual(ctx.warm_items, {"a", "b", "c"})
        self.assertEqual(ctx.positive_history["u1"], {"a"})
        self.assertEqual(ctx.all_rated_history["u1"], {"a", "x", "y", "z"})
        self.assertEqual(ctx.positive_history["u2"], {"a", "b", "c"})
        self.assertEqual(ctx.all_rated_history["u2"], {"a", "b", "c"})

        sampler = NegativeSampler(ctx)
        pool_u1 = sampler.get_negative_pool("u1")
        self.assertEqual(pool_u1, ["b", "c"])

        pool_u2 = sampler.get_negative_pool("u2")
        self.assertEqual(pool_u2, [])

        rng = random.Random(42)
        trainable, skipped = sampler.prepare_batch([("u1", "a"), ("u2", "b")], rng)
        self.assertEqual(len(trainable), 1)
        self.assertEqual(trainable[0][0], "u1")
        self.assertEqual(trainable[0][1], "a")
        self.assertIn(trainable[0][2], ["b", "c"])
        self.assertEqual(skipped, 1)

    def test_negative_sampler_rejects_unknown_users(self):
        train = [("u1", "p1", 5.0), ("u2", "p2", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        sampler = NegativeSampler(ctx)
        rng = random.Random(42)

        with self.assertRaises(ValueError):
            sampler.get_negative_pool("unknown_user")

        with self.assertRaises(ValueError):
            sampler.sample_negative("unknown_user", rng)


class ReferenceLossAndRegularizationTests(unittest.TestCase):
    def test_stable_bpr_loss_hand_calculated(self):
        sp = [2.0, 1.0]
        sn = [1.0, 1.0]
        loss = bpr_loss(sp, sn)
        expected = (math.log(1.0 + math.exp(-1.0)) + math.log(2.0)) / 2.0
        self.assertAlmostEqual(loss, expected, places=9)

    def test_stable_softplus_extreme_values_and_nonfinite_failures(self):
        self.assertAlmostEqual(stable_softplus(0.0), math.log(2.0), places=9)
        self.assertAlmostEqual(stable_softplus(100.0), 100.0, places=9)
        self.assertAlmostEqual(stable_softplus(-100.0), 0.0, places=9)

        with self.assertRaises(ValueError):
            stable_softplus(float("nan"))
        with self.assertRaises(ValueError):
            stable_softplus(float("inf"))
        with self.assertRaises(ValueError):
            stable_softplus(True)

    def test_bpr_loss_validation(self):
        with self.assertRaises(ValueError):
            bpr_loss([], [])
        with self.assertRaises(ValueError):
            bpr_loss([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            bpr_loss([float("nan")], [1.0])

    def test_endpoint_regularization_hand_calculated(self):
        user_emb = {
            "u1": [1.0, 2.0],  # norm^2 = 5
            "u2": [3.0, 0.0],  # norm^2 = 9
        }
        item_emb = {
            "p1": [1.0, 1.0],  # norm^2 = 2
            "p2": [2.0, 2.0],  # norm^2 = 8
            "p3": [0.0, 1.0],  # norm^2 = 1
        }
        u_batch = ["u1", "u1", "u2"]
        pos_batch = ["p1", "p2", "p1"]
        neg_batch = ["p2", "p3", "p3"]

        reg = endpoint_regularization(
            u_batch, pos_batch, neg_batch,
            user_emb, item_emb,
            coefficient=1e-5
        )
        self.assertAlmostEqual(reg, 5e-5, places=9)

    def test_endpoint_regularization_namespaces_and_ignored_neighbors(self):
        user_emb = {
            "id_1": [1.0, 0.0],
            "huge_neighbor": [1000.0, 1000.0],
        }
        item_emb = {
            "id_1": [0.0, 2.0],
            "p_both": [3.0, 0.0],
            "huge_neighbor": [5000.0, 5000.0],
        }
        reg = endpoint_regularization(
            ["id_1", "id_1"],
            ["id_1", "p_both"],
            ["p_both", "p_both"],
            user_emb,
            item_emb,
            coefficient=1e-5,
        )
        expected = 1e-5 * (14.0 / 3.0)
        self.assertAlmostEqual(reg, expected, places=9)

    def test_endpoint_regularization_inconsistent_dimension_and_empty_failures(self):
        with self.assertRaises(ValueError):
            endpoint_regularization([], [], [], {}, {})

        user_emb = {"u1": [1.0, 2.0]}
        item_emb = {"p1": [1.0]}
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p1"], user_emb, item_emb)

        item_zero_dim = {"p1": []}
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p1"], {"u1": []}, item_zero_dim)

    def test_bpr_finite_difference_overflow_and_aggregate_overflow(self):
        # sp = -1e308, sn = 1e308 causes diff = 2e308 float OverflowError
        with self.assertRaises(ValueError):
            bpr_loss([-1e308], [1e308])

        with self.assertRaises(ValueError):
            bpr_loss([1e308], [-1e308])

    def test_endpoint_regularization_coefficient_fixed_to_1e5(self):
        user_emb = {"u1": [1.0, 1.0]}
        item_emb = {"p1": [1.0, 1.0], "p2": [1.0, 1.0]}
        # Exactly 1e-5 succeeds
        val = endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, item_emb, coefficient=1e-5)
        self.assertIsInstance(val, float)

        # Other coefficients must be rejected
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, item_emb, coefficient=1e-4)
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, item_emb, coefficient=0.0)
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, item_emb, coefficient=-1e-5)
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, item_emb, coefficient=True)
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, item_emb, coefficient=float("nan"))
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, item_emb, coefficient=float("inf"))

    def test_endpoint_regularization_squared_norm_overflow_and_nonfinite_rows(self):
        user_emb = {"u1": [1.0]}
        item_emb = {"p1": [1.0], "p2": [1.0]}

        # Squared-norm overflow: 1e200 * 1e200 overflows in float
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], {"u1": [1e200]}, item_emb, coefficient=1e-5)

        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, {"p1": [1e200], "p2": [1.0]}, coefficient=1e-5)

        # Non-finite values in rows
        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], {"u1": [float("nan")]}, item_emb, coefficient=1e-5)

        with self.assertRaises(ValueError):
            endpoint_regularization(["u1"], ["p1"], ["p2"], user_emb, {"p1": [float("inf")], "p2": [1.0]}, coefficient=1e-5)

    def test_stable_softplus_failures(self):
        with self.assertRaises(ValueError):
            stable_softplus(float("nan"))
        with self.assertRaises(ValueError):
            stable_softplus(float("inf"))
        with self.assertRaises(ValueError):
            stable_softplus("-1")
        with self.assertRaises(ValueError):
            stable_softplus(True)
        with self.assertRaises(ValueError):
            stable_softplus(False)



class HeldoutIsolationTests(unittest.TestCase):
    def test_heldout_mutations_do_not_alter_training_or_profiles(self):
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
        ]
        item_categories = {"p1": ["c1"], "p2": ["c2"]}
        item_brands = {"p1": "b1", "p2": "b2"}

        ctx = TrainingContext(train, condition="all_ratings")
        scorer = KGOnlyScorer(ctx, item_categories, item_brands, mixture=(1.0, 0.0))

        idf_c1_before = scorer.idf_category["c1"]
        pop_p1_before = ctx.popularity["p1"]
        u1_vec_before = dict(scorer.user_cat_vec["u1"])

        evaluator = Evaluator(ctx)
        evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")
        evaluator.evaluate([("u1", "p2", 1.0)], scorer, split="validation")
        evaluator.evaluate([("u1", "p_heldout_cold", 5.0)], scorer, split="validation")

        self.assertEqual(scorer.idf_category["c1"], idf_c1_before)
        self.assertEqual(ctx.popularity["p1"], pop_p1_before)
        self.assertEqual(scorer.user_cat_vec["u1"], u1_vec_before)


if __name__ == "__main__":
    unittest.main()
