"""Regression tests for shared recommendation evaluation subsystem.

Verifies:
- >1-user macro oracles and rank boundaries including literal cutoff boundaries K and K+1
  for K in (5, 10, 20) and beyond @20
- CatalogCoverage@10 union over entire condition-warm pool denominator
- Both feedback conditions (all_ratings and rating_ge_4)
- Low training ratings globally warm yet filtered from user candidates
- Low-rated validation record test filtering with tuple/dict/generator records and
  asserting exclusion happens before provider and cannot change training view
- Missing, malformed, and duplicate validation history rejection on test split
- Target validation: duplicate heldout target users, malformed IDs/ratings rejection
- Eligibility overlap accounting (unknown, cold, irrelevant, history-overlap)
- Empty targets (proportions N/A as None) and empty eligible populations returning None
- Exact tie-breaking (parent_asin ascending) and candidate shortages without padding
- Strengthened fallback oracle: catches all-rating-count popularity leakage, asserts ranking,
  full-warm coverage, registered fallback/by-reason metrics, and mixed macro means/subgroups
- Shared empty-positive-history fallback provider bypass
- Strict provider route validation: rejects arbitrary provider fallback claims,
  non-None fallback reasons, malformed mapping/tuple/route outputs, +/-Inf, bool,
  NaN, missing candidate scores, and provider exceptions as ScoringFailureError
- Held-out mutation isolation: deep context snapshot verification across altered targets
  and validation records
- Deterministic record-order invariance
- TrainingContext validation failures and duplicate detection
"""

from collections import defaultdict
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple
import unittest

from scripts.recommendation_evaluation import (
    Evaluator,
    ScoringFailureError,
    TrainingContext,
)


class TestRecommendationEvaluation(unittest.TestCase):

    def test_multi_user_macro_oracles_and_rank_boundaries_beyond_20(self):
        """Test macro oracles across multiple users with rank boundaries: 1, 5, 6, 10, 11, 20, 21, and 25 (>20)."""
        # Create 30 warm items: item_01 to item_30
        items = [f"item_{i:02d}" for i in range(1, 31)]
        # Training interactions: user_1..user_8 give positive rating to item_01
        # user_dummy interacts with all 30 items so all items are condition-warm
        interactions = [(f"user_{u}", items[0], 5.0) for u in range(1, 9)] + [
            ("user_dummy", it, 5.0) for it in items
        ]
        ctx = TrainingContext(interactions, condition="all_ratings")
        evaluator = Evaluator(ctx)

        # Candidates for user_u are items[1:] (items[0] is in training history)
        # items[1:] has 29 items: item_02 .. item_30
        # OrderedScorer will score candidate items such that alphabetical order is rank order:
        # rank 1: item_02
        # rank 5: item_06 (cutoff K=5)
        # rank 6: item_07 (cutoff K+1=6)
        # rank 10: item_11 (cutoff K=10)
        # rank 11: item_12 (cutoff K+1=11)
        # rank 20: item_21 (cutoff K=20)
        # rank 21: item_22 (cutoff K+1=21)
        # rank 25: item_26 (beyond @20)
        targets = [
            ("user_1", "item_02", 5.0),
            ("user_2", "item_06", 5.0),
            ("user_3", "item_07", 5.0),
            ("user_4", "item_11", 5.0),
            ("user_5", "item_12", 5.0),
            ("user_6", "item_21", 5.0),
            ("user_7", "item_22", 5.0),
            ("user_8", "item_26", 5.0),
        ]

        class OrderedScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                scores = {}
                for idx, it in enumerate(sorted(candidate_items)):
                    scores[it] = 100.0 - idx
                return scores, "personalized", None

        report = evaluator.evaluate(targets, OrderedScorer(), split="validation")

        self.assertEqual(report["audit"]["eligible_users"], 8)
        per_user = {r["user_id"]: r for r in report["per_user_results"]}

        # user_1: rank 1
        self.assertEqual(per_user["user_1"]["metrics"]["HitRate@5"], 1.0)
        self.assertAlmostEqual(per_user["user_1"]["metrics"]["NDCG@5"], 1.0)
        self.assertEqual(per_user["user_1"]["metrics"]["HitRate@10"], 1.0)
        self.assertAlmostEqual(per_user["user_1"]["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(per_user["user_1"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(per_user["user_1"]["metrics"]["NDCG@20"], 1.0)

        # user_2: rank 5 (exact cutoff K=5)
        ndcg_r5 = 1.0 / math.log2(5 + 1.0)
        self.assertEqual(per_user["user_2"]["metrics"]["HitRate@5"], 1.0)
        self.assertAlmostEqual(per_user["user_2"]["metrics"]["NDCG@5"], ndcg_r5)
        self.assertEqual(per_user["user_2"]["metrics"]["HitRate@10"], 1.0)
        self.assertAlmostEqual(per_user["user_2"]["metrics"]["NDCG@10"], ndcg_r5)
        self.assertEqual(per_user["user_2"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(per_user["user_2"]["metrics"]["NDCG@20"], ndcg_r5)

        # user_3: rank 6 (cutoff boundary K+1=6)
        ndcg_r6 = 1.0 / math.log2(6 + 1.0)
        self.assertEqual(per_user["user_3"]["metrics"]["HitRate@5"], 0.0)
        self.assertAlmostEqual(per_user["user_3"]["metrics"]["NDCG@5"], 0.0)
        self.assertEqual(per_user["user_3"]["metrics"]["HitRate@10"], 1.0)
        self.assertAlmostEqual(per_user["user_3"]["metrics"]["NDCG@10"], ndcg_r6)
        self.assertEqual(per_user["user_3"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(per_user["user_3"]["metrics"]["NDCG@20"], ndcg_r6)

        # user_4: rank 10 (exact cutoff K=10)
        ndcg_r10 = 1.0 / math.log2(10 + 1.0)
        self.assertEqual(per_user["user_4"]["metrics"]["HitRate@5"], 0.0)
        self.assertAlmostEqual(per_user["user_4"]["metrics"]["NDCG@5"], 0.0)
        self.assertEqual(per_user["user_4"]["metrics"]["HitRate@10"], 1.0)
        self.assertAlmostEqual(per_user["user_4"]["metrics"]["NDCG@10"], ndcg_r10)
        self.assertEqual(per_user["user_4"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(per_user["user_4"]["metrics"]["NDCG@20"], ndcg_r10)

        # user_5: rank 11 (cutoff boundary K+1=11)
        ndcg_r11 = 1.0 / math.log2(11 + 1.0)
        self.assertEqual(per_user["user_5"]["metrics"]["HitRate@5"], 0.0)
        self.assertAlmostEqual(per_user["user_5"]["metrics"]["NDCG@5"], 0.0)
        self.assertEqual(per_user["user_5"]["metrics"]["HitRate@10"], 0.0)
        self.assertAlmostEqual(per_user["user_5"]["metrics"]["NDCG@10"], 0.0)
        self.assertEqual(per_user["user_5"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(per_user["user_5"]["metrics"]["NDCG@20"], ndcg_r11)

        # user_6: rank 20 (exact cutoff K=20)
        ndcg_r20 = 1.0 / math.log2(20 + 1.0)
        self.assertEqual(per_user["user_6"]["metrics"]["HitRate@5"], 0.0)
        self.assertAlmostEqual(per_user["user_6"]["metrics"]["NDCG@5"], 0.0)
        self.assertEqual(per_user["user_6"]["metrics"]["HitRate@10"], 0.0)
        self.assertAlmostEqual(per_user["user_6"]["metrics"]["NDCG@10"], 0.0)
        self.assertEqual(per_user["user_6"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(per_user["user_6"]["metrics"]["NDCG@20"], ndcg_r20)

        # user_7: rank 21 (cutoff boundary K+1=21)
        self.assertEqual(per_user["user_7"]["metrics"]["HitRate@5"], 0.0)
        self.assertAlmostEqual(per_user["user_7"]["metrics"]["NDCG@5"], 0.0)
        self.assertEqual(per_user["user_7"]["metrics"]["HitRate@10"], 0.0)
        self.assertAlmostEqual(per_user["user_7"]["metrics"]["NDCG@10"], 0.0)
        self.assertEqual(per_user["user_7"]["metrics"]["HitRate@20"], 0.0)
        self.assertAlmostEqual(per_user["user_7"]["metrics"]["NDCG@20"], 0.0)

        # user_8: rank 25 (beyond @20)
        self.assertEqual(per_user["user_8"]["metrics"]["HitRate@5"], 0.0)
        self.assertAlmostEqual(per_user["user_8"]["metrics"]["NDCG@5"], 0.0)
        self.assertEqual(per_user["user_8"]["metrics"]["HitRate@10"], 0.0)
        self.assertAlmostEqual(per_user["user_8"]["metrics"]["NDCG@10"], 0.0)
        self.assertEqual(per_user["user_8"]["metrics"]["HitRate@20"], 0.0)
        self.assertAlmostEqual(per_user["user_8"]["metrics"]["NDCG@20"], 0.0)

        # Hand-calculated macro averages
        expected_hr5 = (1.0 + 1.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0) / 8.0
        expected_ndcg5 = (1.0 + ndcg_r5 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0) / 8.0
        expected_hr10 = (1.0 + 1.0 + 1.0 + 1.0 + 0.0 + 0.0 + 0.0 + 0.0) / 8.0
        expected_ndcg10 = (1.0 + ndcg_r5 + ndcg_r6 + ndcg_r10 + 0.0 + 0.0 + 0.0 + 0.0) / 8.0
        expected_hr20 = (1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 0.0 + 0.0) / 8.0
        expected_ndcg20 = (1.0 + ndcg_r5 + ndcg_r6 + ndcg_r10 + ndcg_r11 + ndcg_r20 + 0.0 + 0.0) / 8.0

        om = report["overall_metrics"]
        self.assertAlmostEqual(om["HitRate@5"], expected_hr5)
        self.assertAlmostEqual(om["NDCG@5"], expected_ndcg5)
        self.assertAlmostEqual(om["HitRate@10"], expected_hr10)
        self.assertAlmostEqual(om["NDCG@10"], expected_ndcg10)
        self.assertAlmostEqual(om["HitRate@20"], expected_hr20)
        self.assertAlmostEqual(om["NDCG@20"], expected_ndcg20)

    def test_coverage_union_and_full_warm_denominator(self):
        """Test CatalogCoverage@10 union across users divided by entire condition-warm pool."""
        interactions = [(f"u{i}", f"item_{i:02d}", 5.0) for i in range(1, 21)]
        ctx = TrainingContext(interactions, condition="all_ratings")
        self.assertEqual(len(ctx.warm_items), 20)
        evaluator = Evaluator(ctx)

        targets = [
            ("u1", "item_20", 5.0),
            ("u2", "item_19", 5.0),
        ]

        class DistinctScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                scores = {it: 0.0 for it in candidate_items}
                if user_id == "u1":
                    for idx, it in enumerate([f"item_{i:02d}" for i in range(2, 12)]):
                        scores[it] = 100.0 - idx
                else:
                    for idx, it in enumerate([f"item_{i:02d}" for i in range(7, 17)]):
                        scores[it] = 100.0 - idx
                return scores, "personalized", None

        report = evaluator.evaluate(targets, DistinctScorer(), split="validation")
        self.assertAlmostEqual(report["overall_metrics"]["CatalogCoverage@10"], 15.0 / 20.0)
        self.assertEqual(report["audit"]["warm_pool_size"], 20)

    def test_both_feedback_conditions(self):
        """Test all_ratings vs rating_ge_4 warmth, popularity, and relevance semantics."""
        interactions = [
            ("u1", "item_high", 5.0),
            ("u2", "item_low", 2.0),
        ]

        # all_ratings: both items warm
        ctx_all = TrainingContext(interactions, condition="all_ratings")
        self.assertIn("item_high", ctx_all.warm_items)
        self.assertIn("item_low", ctx_all.warm_items)
        self.assertEqual(ctx_all.popularity["item_high"], 1)
        self.assertEqual(ctx_all.popularity["item_low"], 1)

        # rating_ge_4: item_low is cold
        ctx_ge4 = TrainingContext(interactions, condition="rating_ge_4")
        self.assertIn("item_high", ctx_ge4.warm_items)
        self.assertNotIn("item_low", ctx_ge4.warm_items)
        self.assertEqual(ctx_ge4.popularity["item_high"], 1)
        self.assertNotIn("item_low", ctx_ge4.popularity)

        def mock_scorer(u, cands):
            return {it: 1.0 for it in cands}, "personalized", None

        # Target with rating 3.0:
        # In all_ratings: relevant
        ev_all = Evaluator(ctx_all)
        rep_all = ev_all.evaluate([("u1", "item_low", 3.0)], mock_scorer, split="validation")
        self.assertEqual(rep_all["audit"]["eligible_users"], 1)
        self.assertEqual(rep_all["audit"]["irrelevant_targets"], 0)

        # In rating_ge_4: irrelevant target
        ev_ge4 = Evaluator(ctx_ge4)
        rep_ge4 = ev_ge4.evaluate([("u1", "item_high", 3.0)], mock_scorer, split="validation")
        self.assertEqual(rep_ge4["audit"]["eligible_users"], 0)
        self.assertEqual(rep_ge4["audit"]["irrelevant_targets"], 1)

        # Invalid condition
        with self.assertRaises(ValueError):
            TrainingContext(interactions, condition="rating_ge_3")

    def test_low_training_ratings_globally_warm_yet_filtered(self):
        """Test that low training ratings are filtered from candidates even when globally warm."""
        interactions = [
            ("u1", "item_low_shared", 2.0),
            ("u1", "item_pos", 5.0),
            ("u2", "item_low_shared", 5.0),  # makes item_low_shared warm in rating_ge_4
            ("u2", "item_other", 5.0),
        ]
        ctx = TrainingContext(interactions, condition="rating_ge_4")
        self.assertIn("item_low_shared", ctx.warm_items)

        def mock_scorer(u, cands):
            return {it: 1.0 for it in cands}, "personalized", None

        ev = Evaluator(ctx)
        rep = ev.evaluate([("u1", "item_other", 5.0)], mock_scorer, split="validation")
        self.assertEqual(rep["audit"]["eligible_users"], 1)
        user_res = rep["per_user_results"][0]
        self.assertNotIn("item_low_shared", user_res["top_20"])
        self.assertEqual(user_res["candidate_count"], 1)

    def test_low_rated_validation_record_test_filtering(self):
        """Test validation record filtering carrying rating 2.0 via tuple/dict/generator before scoring."""
        interactions = [
            ("u1", "item_train_1", 5.0),
            ("u2", "item_train_2", 5.0),
            ("u3", "item_train_3", 5.0),
            ("u_other", "item_val_tuple", 5.0),  # globally warm in rating_ge_4
            ("u_other", "item_val_dict", 5.0),
            ("u_other", "item_val_gen", 5.0),
            ("u_other", "item_target_1", 5.0),
            ("u_other", "item_target_2", 5.0),
            ("u_other", "item_target_3", 5.0),
        ]
        ctx = TrainingContext(interactions, condition="rating_ge_4")

        # Snapshot training context to assert immutability
        snap_all = {u: set(s) for u, s in ctx.all_rated_history.items()}
        snap_pos = {u: set(s) for u, s in ctx.positive_history.items()}
        snap_warm = set(ctx.warm_items)
        snap_pop = dict(ctx.popularity)

        ev = Evaluator(ctx)

        # Validation history with actual rating 2.0 records across tuple, dict, and generator formats
        val_records = [
            ("u1", "item_val_tuple", 2.0),
            {"user_id": "u2", "parent_asin": "item_val_dict", "rating": 2.0},
            ("u3", "item_val_gen", 2.0),
        ]

        class ProviderAssertionScorer:
            def score_candidates(self, user_id: str, candidates: Sequence[str]):
                # Assert exclusion happens BEFORE provider is called
                if user_id == "u1":
                    assert "item_val_tuple" not in candidates, "item_val_tuple must be filtered before provider!"
                elif user_id == "u2":
                    assert "item_val_dict" not in candidates, "item_val_dict must be filtered before provider!"
                elif user_id == "u3":
                    assert "item_val_gen" not in candidates, "item_val_gen must be filtered before provider!"
                return {it: 1.0 for it in candidates}, "personalized", None

        # 1. Test generator of records
        def val_record_generator():
            for rec in val_records:
                yield rec

        targets = [
            ("u1", "item_target_1", 5.0),
            ("u2", "item_target_2", 5.0),
            ("u3", "item_target_3", 5.0),
        ]

        rep = ev.evaluate(targets, ProviderAssertionScorer(), split="test", validation_history=val_record_generator())
        self.assertEqual(report_count := rep["audit"]["eligible_users"], 3)
        per_user = {r["user_id"]: r for r in rep["per_user_results"]}
        self.assertNotIn("item_val_tuple", per_user["u1"]["top_20"])
        self.assertNotIn("item_val_dict", per_user["u2"]["top_20"])
        self.assertNotIn("item_val_gen", per_user["u3"]["top_20"])

        # 2. Test that changing validation rating from 2.0 to 5.0 cannot alter the training view
        val_records_high = [
            ("u1", "item_val_tuple", 5.0),
            {"user_id": "u2", "parent_asin": "item_val_dict", "rating": 5.0},
            ("u3", "item_val_gen", 5.0),
        ]
        rep2 = ev.evaluate(targets, ProviderAssertionScorer(), split="test", validation_history=val_records_high)
        self.assertEqual(rep2["audit"]["eligible_users"], 3)

        self.assertEqual(ctx.all_rated_history, snap_all)
        self.assertEqual(ctx.positive_history, snap_pos)
        self.assertEqual(ctx.warm_items, snap_warm)
        self.assertEqual(ctx.popularity, snap_pop)

    def test_missing_and_malformed_validation_history_in_test_split(self):
        """Test strict validation history requirements: rejects None, string, duplicate, missing/empty users, mapping-valued histories, conflicting user_id, and explicitly None ratings."""
        interactions = [("u1", "item_1", 5.0), ("u2", "item_2", 5.0)]
        ctx = TrainingContext(interactions, condition="all_ratings")
        ev = Evaluator(ctx)
        target = [("u1", "item_2", 5.0)]

        def dummy_scorer(u, cands):
            return {it: 1.0 for it in cands}, "personalized", None

        # Missing validation_history in test split (must never default to train-only)
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history=None)

        # String passed instead of collection/records
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history="item_1")

        # Dict with string value instead of collection
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": "item_1"})

        # Duplicate records in dict
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": ["item_2", "item_2"]})

        # Duplicate records in sequence
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history=[("u1", "item_2", 2.0), ("u1", "item_2", 2.0)])

        # Known target user missing from validation_history
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u2": ["item_1"]})

        # 1. Mapping-valued user history (silently failed in prior revision)
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": {"parent_asin": "item_2", "rating": 2.0}})

        # Record dict with conflicting user_id in user collection
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": [{"user_id": "u2", "parent_asin": "item_2", "rating": 2.0}]})

        # Record dict with matching user_id succeeds
        rep_match = ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": [{"user_id": "u1", "parent_asin": "item_1", "rating": 2.0}]})
        self.assertEqual(rep_match["audit"]["eligible_users"], 1)

        # 2. Empty per-user validation histories for known target users (must reject empty collections/generators)
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": []})
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": set()})
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": (x for x in [])})

        # Unknown target user does not raise missing validation history, audits as unknown_user
        unk_target = [("u_unknown", "item_2", 5.0)]
        rep_unk = ev.evaluate(unk_target, dummy_scorer, split="test", validation_history={"u1": ["item_1"]})
        self.assertEqual(rep_unk["audit"]["unknown_users"], 1)
        self.assertEqual(rep_unk["audit"]["eligible_users"], 0)

        # 3. Validation records with explicitly None ratings must be rejected as malformed
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": [{"parent_asin": "item_1", "rating": None}]})
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history=[{"user_id": "u1", "parent_asin": "item_1", "rating": None}])
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history=[("u1", "item_1", None)])
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": [("item_1", None)]})

        # Identifier-only validation inputs (rating omitted entirely) succeed
        rep_id_only_str = ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": ["item_1"]})
        self.assertEqual(rep_id_only_str["audit"]["eligible_users"], 1)
        rep_id_only_dict = ev.evaluate(target, dummy_scorer, split="test", validation_history={"u1": [{"parent_asin": "item_1"}]})
        self.assertEqual(rep_id_only_dict["audit"]["eligible_users"], 1)
        rep_id_only_tuple = ev.evaluate(target, dummy_scorer, split="test", validation_history=[("u1", "item_1")])
        self.assertEqual(rep_id_only_tuple["audit"]["eligible_users"], 1)

        # Malformed record: invalid rating
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history=[("u1", "item_2", True)])
        with self.assertRaises(ValueError):
            ev.evaluate(target, dummy_scorer, split="test", validation_history=[("u1", "item_2", float("nan"))])

    def test_target_validation_and_duplicate_rejection(self):
        """Test rejection of duplicate held-out target users, non-iterables, and invalid target fields."""
        interactions = [("u1", "item_1", 5.0), ("u2", "item_2", 5.0)]
        ctx = TrainingContext(interactions, condition="all_ratings")
        ev = Evaluator(ctx)

        def dummy_scorer(u, cands):
            return {it: 1.0 for it in cands}, "personalized", None

        # Duplicate held-out target for user u1
        with self.assertRaises(ValueError):
            ev.evaluate([("u1", "item_2", 5.0), ("u1", "item_1", 5.0)], dummy_scorer)

        # Targets non-iterable or string
        with self.assertRaises(ValueError):
            ev.evaluate("user_1", dummy_scorer)
        with self.assertRaises(ValueError):
            ev.evaluate(12345, dummy_scorer)

        # Invalid user_id
        with self.assertRaises(ValueError):
            ev.evaluate([("", "item_2", 5.0)], dummy_scorer)
        with self.assertRaises(ValueError):
            ev.evaluate([(None, "item_2", 5.0)], dummy_scorer)

        # Invalid parent_asin
        with self.assertRaises(ValueError):
            ev.evaluate([("u1", "", 5.0)], dummy_scorer)

        # Invalid rating
        with self.assertRaises(ValueError):
            ev.evaluate([("u1", "item_2", True)], dummy_scorer)
        with self.assertRaises(ValueError):
            ev.evaluate([("u1", "item_2", float("nan"))], dummy_scorer)
        with self.assertRaises(ValueError):
            ev.evaluate([("u1", "item_2", float("inf"))], dummy_scorer)
        with self.assertRaises(ValueError):
            ev.evaluate([("u1", "item_2", "5.0")], dummy_scorer)

    def test_eligibility_overlap_accounting(self):
        """Test accounting for disjoint overlapping exclusion reasons."""
        interactions = [
            ("u_known_1", "item_warm", 5.0),
            ("u_known_2", "item_warm", 5.0),
            ("u_dummy", "item_warm_2", 5.0),
        ]
        ctx = TrainingContext(interactions, condition="rating_ge_4")
        ev = Evaluator(ctx)

        targets = [
            # Target 1: eligible (known user, warm target, rating >= 4, not in history)
            ("u_known_1", "item_warm_2", 5.0),
            # Target 2: unknown_user + cold_target + irrelevant_target
            ("u_unknown", "item_cold", 2.0),
            # Target 3: history_overlap + irrelevant_target (known user, warm target in history, rating < 4)
            ("u_known_2", "item_warm", 2.0),
        ]

        def dummy_scorer(u, cands):
            return {it: 1.0 for it in cands}, "personalized", None

        rep = ev.evaluate(targets, dummy_scorer, split="validation")
        audit = rep["audit"]
        self.assertEqual(audit["total_targets"], 3)
        self.assertEqual(audit["eligible_users"], 1)
        self.assertEqual(audit["excluded_targets"], 2)

        # Check overlapping reasons
        self.assertEqual(audit["overlapping_reasons"]["cold_target+irrelevant_target+unknown_user"], 1)
        self.assertEqual(audit["overlapping_reasons"]["history_overlap+irrelevant_target"], 1)

        # Check marginal counts
        self.assertEqual(audit["unknown_users"], 1)
        self.assertEqual(audit["cold_targets"], 1)
        self.assertEqual(audit["irrelevant_targets"], 2)
        self.assertEqual(audit["history_overlaps"], 1)

        # Check proportions
        self.assertAlmostEqual(audit["proportions"]["eligible_users"], 1.0 / 3.0)
        self.assertAlmostEqual(audit["proportions"]["excluded_targets"], 2.0 / 3.0)

    def test_empty_targets_and_empty_populations_return_none(self):
        """Test empty targets list and empty eligible population returning None across metrics."""
        interactions = [("u1", "item_1", 5.0)]
        ctx = TrainingContext(interactions, condition="all_ratings")
        ev = Evaluator(ctx)

        def dummy_scorer(u, cands):
            return {it: 1.0 for it in cands}, "personalized", None

        # Case 1: Empty targets list -> proportions must be None (N/A), not 0.0
        rep_empty = ev.evaluate([], dummy_scorer, split="validation")
        audit_empty = rep_empty["audit"]
        self.assertEqual(audit_empty["total_targets"], 0)
        self.assertEqual(audit_empty["eligible_users"], 0)
        self.assertEqual(audit_empty["excluded_targets"], 0)
        self.assertIsNone(audit_empty["proportions"]["unknown_users"])
        self.assertIsNone(audit_empty["proportions"]["irrelevant_targets"])
        self.assertIsNone(audit_empty["proportions"]["cold_targets"])
        self.assertIsNone(audit_empty["proportions"]["history_overlaps"])
        self.assertIsNone(audit_empty["proportions"]["eligible_users"])
        self.assertIsNone(audit_empty["proportions"]["excluded_targets"])
        self.assertEqual(audit_empty["overlapping_proportions"], {})

        for m in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(rep_empty["overall_metrics"][m])
        self.assertEqual(rep_empty["per_user_results"], [])

        # Case 2: Targets present but 0 eligible users (cold item)
        rep_cold = ev.evaluate([("u1", "item_cold", 5.0)], dummy_scorer, split="validation")
        self.assertEqual(rep_cold["audit"]["total_targets"], 1)
        self.assertEqual(rep_cold["audit"]["eligible_users"], 0)
        self.assertAlmostEqual(rep_cold["audit"]["proportions"]["cold_targets"], 1.0)
        self.assertAlmostEqual(rep_cold["audit"]["proportions"]["eligible_users"], 0.0)

        for m in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(rep_cold["overall_metrics"][m])

        for route in ("personalized", "non_personalized", "fallback"):
            sub = rep_cold["subgroups"][route]
            self.assertEqual(sub["count"], 0)
            for m in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
                self.assertIsNone(sub["metrics"][m])

    def test_exact_ties_and_candidate_shortages(self):
        """Test exact score tie-breaking (parent_asin ascending) and candidate shortages without padding."""
        interactions = [
            ("u1", "item_x", 5.0),
            ("u2", "item_c", 5.0),
            ("u2", "item_a", 5.0),
            ("u2", "item_b", 5.0),
        ]
        ctx = TrainingContext(interactions, condition="all_ratings")
        ev = Evaluator(ctx)

        class TieScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return {it: 1.0 for it in candidate_items}, "personalized", None

        rep = ev.evaluate([("u1", "item_b", 5.0)], TieScorer(), split="validation")
        self.assertEqual(rep["candidate_shortages"][5], 1)
        self.assertEqual(rep["candidate_shortages"][10], 1)
        self.assertEqual(rep["candidate_shortages"][20], 1)

        user_res = rep["per_user_results"][0]
        self.assertEqual(user_res["top_5"], ["item_a", "item_b", "item_c"])
        self.assertEqual(user_res["top_10"], ["item_a", "item_b", "item_c"])
        self.assertEqual(user_res["top_20"], ["item_a", "item_b", "item_c"])
        self.assertEqual(len(user_res["top_5"]), 3)

    def test_strengthened_fallback_oracle_and_popularity_leakage(self):
        """Test fallback ranking with unequal condition-positive popularity and low-rated observations to catch count leakage."""
        interactions = [
            # item_pos_high: 3 positive ratings, 0 low ratings -> pos count = 3, all count = 3
            ("u_p1", "item_pos_high", 5.0),
            ("u_p2", "item_pos_high", 5.0),
            ("u_p3", "item_pos_high", 5.0),
            # item_pos_mid: 2 positive ratings, 1 low rating -> pos count = 2, all count = 3
            ("u_p1", "item_pos_mid", 5.0),
            ("u_p2", "item_pos_mid", 5.0),
            ("u_p3", "item_pos_mid", 1.0),
            # item_pos_low: 1 positive rating, 5 low ratings -> pos count = 1, all count = 6 (catches all-rating leakage!)
            ("u_p1", "item_pos_low", 5.0),
            ("u_l1", "item_pos_low", 1.0),
            ("u_l2", "item_pos_low", 1.0),
            ("u_l3", "item_pos_low", 1.0),
            ("u_l4", "item_pos_low", 1.0),
            ("u_l5", "item_pos_low", 1.0),
            # item_other: 1 positive rating -> pos count = 1, all count = 1
            ("u_p1", "item_other", 5.0),
            # u_fallback has only low training ratings -> empty positive history under rating_ge_4
            ("u_fallback", "dummy_train", 2.0),
        ]
        ctx = TrainingContext(interactions, condition="rating_ge_4")
        self.assertEqual(ctx.popularity["item_pos_high"], 3)
        self.assertEqual(ctx.popularity["item_pos_mid"], 2)
        self.assertEqual(ctx.popularity["item_pos_low"], 1)
        self.assertEqual(ctx.popularity["item_other"], 1)
        self.assertEqual(len(ctx.warm_items), 4)

        ev = Evaluator(ctx)

        # u_fallback target: item_pos_mid (will be ranked #2 under condition-positive popularity)
        # u_ordinary (u_p2): target item_pos_mid is in history, so target item_pos_low (ranked #1 by provider)
        targets = [
            ("u_fallback", "item_pos_mid", 5.0),
            ("u_p2", "item_pos_low", 5.0),
        ]

        class OrdinaryScorer:
            def score_candidates(self, user_id: str, candidates: Sequence[str]):
                # Score target item highest
                return {it: (10.0 if it == "item_pos_low" else 1.0) for it in candidates}, "personalized", None

        rep = ev.evaluate(targets, OrdinaryScorer(), split="validation")
        self.assertEqual(rep["audit"]["eligible_users"], 2)

        per_user = {r["user_id"]: r for r in rep["per_user_results"]}

        # u_fallback: empty-positive-history route
        uf = per_user["u_fallback"]
        self.assertEqual(uf["route"], "fallback")
        self.assertEqual(uf["fallback_reason"], "empty_positive_history")
        # Candidate rank order must be: item_pos_high (score 3), item_pos_mid (score 2), item_other (score 1, name asc), item_pos_low (score 1)
        self.assertEqual(uf["top_5"], ["item_pos_high", "item_pos_mid", "item_other", "item_pos_low"])
        # Target item_pos_mid is at rank 2!
        ndcg_fallback = 1.0 / math.log2(2 + 1.0)
        self.assertEqual(uf["metrics"]["HitRate@5"], 1.0)
        self.assertAlmostEqual(uf["metrics"]["NDCG@5"], ndcg_fallback)

        # u_p2: personalized route, target at rank 1
        uo = per_user["u_p2"]
        self.assertEqual(uo["route"], "personalized")
        self.assertEqual(uo["metrics"]["HitRate@5"], 1.0)
        self.assertAlmostEqual(uo["metrics"]["NDCG@5"], 1.0)

        # Macro averages
        om = rep["overall_metrics"]
        self.assertAlmostEqual(om["HitRate@5"], 1.0)
        self.assertAlmostEqual(om["NDCG@5"], (1.0 + ndcg_fallback) / 2.0)

        # Subgroups
        sub = rep["subgroups"]
        self.assertEqual(sub["personalized"]["count"], 1)
        self.assertAlmostEqual(sub["personalized"]["metrics"]["NDCG@5"], 1.0)
        self.assertEqual(sub["fallback"]["count"], 1)
        self.assertAlmostEqual(sub["fallback"]["metrics"]["NDCG@5"], ndcg_fallback)
        self.assertEqual(sub["fallback"]["by_reason"]["empty_positive_history"]["count"], 1)
        self.assertAlmostEqual(sub["fallback"]["by_reason"]["empty_positive_history"]["metrics"]["NDCG@5"], ndcg_fallback)

        # Full-warm coverage: recommended items union divided by warm pool size (4)
        # u_fallback top 10 has all 4 warm items
        self.assertAlmostEqual(om["CatalogCoverage@10"], 4.0 / 4.0)

    def test_shared_empty_positive_history_provider_bypass(self):
        """Test that users with empty positive history bypass the provider entirely."""
        interactions = [
            ("u1", "item_low", 2.0),
            ("u2", "item_low", 5.0),
            ("u2", "item_target", 5.0),
        ]
        ctx = TrainingContext(interactions, condition="rating_ge_4")
        self.assertEqual(len(ctx.positive_history["u1"]), 0)

        ev = Evaluator(ctx)

        class ExplodingScorer:
            called = False
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                self.called = True
                raise RuntimeError("Provider must not be called for empty positive history user!")

        scorer = ExplodingScorer()
        rep = ev.evaluate([("u1", "item_target", 5.0)], scorer, split="validation")

        self.assertFalse(scorer.called)
        user_res = rep["per_user_results"][0]
        self.assertEqual(user_res["route"], "fallback")
        self.assertEqual(user_res["fallback_reason"], "empty_positive_history")
        self.assertEqual(rep["subgroups"]["fallback"]["count"], 1)

    def test_provider_route_validation_and_rejection_of_arbitrary_fallback(self):
        """Test strict rejection of arbitrary provider fallback claims, non-None reasons, and invalid outputs."""
        interactions = [("u1", "item_1", 5.0), ("u2", "item_2", 5.0)]
        ctx = TrainingContext(interactions, condition="all_ratings")
        ev = Evaluator(ctx)
        target = [("u1", "item_2", 5.0)]

        # 1. Reject arbitrary fallback reasons from provider for positive-history user
        for unauthorized_reason in (
            "missing_metadata",
            "unusable_metadata_profile",
            "database_failure",
            "scoring_error",
            "empty_positive_history",
            None,
        ):
            class ProviderFallbackScorer:
                def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                    return {it: 1.0 for it in candidate_items}, "fallback", unauthorized_reason
            with self.assertRaises(ScoringFailureError):
                ev.evaluate(target, ProviderFallbackScorer(), split="validation")

        # 2. Reject non-fallback route with non-None fallback_reason
        class NonNoneReasonScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return {it: 1.0 for it in candidate_items}, "personalized", "some_reason"
        with self.assertRaises(ScoringFailureError):
            ev.evaluate(target, NonNoneReasonScorer(), split="validation")

        class NonNoneReasonScorer2:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return {it: 1.0 for it in candidate_items}, "non_personalized", "other_reason"
        with self.assertRaises(ScoringFailureError):
            ev.evaluate(target, NonNoneReasonScorer2(), split="validation")

        # 3. Reject invalid route name or route type
        for bad_route in ("magic_route", 123, None, []):
            class BadRouteScorer:
                def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                    return {it: 1.0 for it in candidate_items}, bad_route, None
            with self.assertRaises(ScoringFailureError):
                ev.evaluate(target, BadRouteScorer(), split="validation")

        # 4. Reject malformed mapping / tuple / list outputs
        class ListOutputScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return [1.0, 2.0]
        with self.assertRaises(ScoringFailureError):
            ev.evaluate(target, ListOutputScorer(), split="validation")

        class NonDictTupleScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return [1.0], "personalized", None
        with self.assertRaises(ScoringFailureError):
            ev.evaluate(target, NonDictTupleScorer(), split="validation")

        class BadTupleLenScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return ({candidate_items[0]: 1.0}, "personalized", None, "extra")
        with self.assertRaises(ScoringFailureError):
            ev.evaluate(target, BadTupleLenScorer(), split="validation")

        # 5. Reject +Inf, -Inf, NaN, bool, non-numeric, and missing candidate scores
        for bad_score in (float("inf"), float("-inf"), float("nan"), True, False, "1.0", None):
            class BadScoreScorer:
                def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                    return {candidate_items[0]: bad_score}, "personalized", None
            with self.assertRaises(ScoringFailureError):
                ev.evaluate(target, BadScoreScorer(), split="validation")

        class MissingCandidateScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return {}, "personalized", None
        with self.assertRaises(ScoringFailureError):
            ev.evaluate(target, MissingCandidateScorer(), split="validation")

        # 6. Reject provider exceptions wrapped as ScoringFailureError
        class CrashScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                raise ZeroDivisionError("Crash")
        with self.assertRaises(ScoringFailureError):
            ev.evaluate(target, CrashScorer(), split="validation")

    def test_heldout_isolation_deep_context_snapshot(self):
        """Test deep context snapshot verification across altered targets and validation records."""
        interactions = [("u1", "item_1", 5.0), ("u2", "item_2", 5.0)]
        ctx = TrainingContext(interactions, condition="all_ratings")
        ev = Evaluator(ctx)

        # Snapshot all fields of TrainingContext
        snap_all_rated = {u: set(items) for u, items in ctx.all_rated_history.items()}
        snap_pos = {u: set(items) for u, items in ctx.positive_history.items()}
        snap_warm = set(ctx.warm_items)
        snap_sorted_warm = list(ctx.sorted_warm_items)
        snap_pop = dict(ctx.popularity)
        snap_users = set(ctx.users)
        snap_items = set(ctx.all_training_items)

        def dummy_scorer(u, cands):
            return {it: 1.0 for it in cands}, "personalized", None

        # Run evaluation multiple times with altered targets (warm, cold, irrelevant)
        altered_targets = [
            ("u1", "item_2", 5.0),
            ("u2", "item_cold", 5.0),
            ("u_unknown", "item_1", 2.0),
        ]
        ev.evaluate(altered_targets[:1], dummy_scorer, split="validation")
        ev.evaluate(altered_targets, dummy_scorer, split="validation")

        # Run test split with altered validation records
        val_records = [
            ("u1", "item_val_low", 1.0),
            {"user_id": "u2", "parent_asin": "item_val_high", "rating": 5.0},
        ]
        ev.evaluate([("u1", "item_2", 5.0)], dummy_scorer, split="test", validation_history=val_records)

        # Mutate lists in place
        altered_targets.append(("u1", "item_leaked", 5.0))
        val_records.append(("u1", "item_leaked_val", 5.0))

        # Assert TrainingContext remained completely unchanged
        self.assertEqual(ctx.all_rated_history, snap_all_rated)
        self.assertEqual(ctx.positive_history, snap_pos)
        self.assertEqual(ctx.warm_items, snap_warm)
        self.assertEqual(ctx.sorted_warm_items, snap_sorted_warm)
        self.assertEqual(ctx.popularity, snap_pop)
        self.assertEqual(ctx.users, snap_users)
        self.assertEqual(ctx.all_training_items, snap_items)

    def test_deterministic_record_order_invariance(self):
        """Test that permuting input records does not alter deterministic evaluation results."""
        raw_interactions = [
            ("u1", "item_1", 5.0),
            ("u2", "item_2", 5.0),
            ("u3", "item_3", 5.0),
            ("u1", "item_4", 4.0),
            ("u2", "item_5", 5.0),
        ]
        raw_targets = [
            ("u3", "item_1", 5.0),
            ("u1", "item_2", 5.0),
            ("u2", "item_3", 5.0),
        ]

        class LocalDeterministicScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return {it: float(len(it)) for it in candidate_items}, "personalized", None

        # Run 1: original order
        ctx1 = TrainingContext(raw_interactions, condition="all_ratings")
        ev1 = Evaluator(ctx1)
        rep1 = ev1.evaluate(raw_targets, LocalDeterministicScorer(), split="validation")

        # Run 2: shuffled order
        shuffled_interactions = list(raw_interactions)
        random.Random(42).shuffle(shuffled_interactions)
        shuffled_targets = list(raw_targets)
        random.Random(42).shuffle(shuffled_targets)

        ctx2 = TrainingContext(shuffled_interactions, condition="all_ratings")
        ev2 = Evaluator(ctx2)
        rep2 = ev2.evaluate(shuffled_targets, LocalDeterministicScorer(), split="validation")

        self.assertEqual(rep1["overall_metrics"], rep2["overall_metrics"])
        self.assertEqual(rep1["audit"], rep2["audit"])
        self.assertEqual(rep1["candidate_shortages"], rep2["candidate_shortages"])
        self.assertEqual(
            [r["user_id"] for r in rep1["per_user_results"]],
            [r["user_id"] for r in rep2["per_user_results"]],
        )
        self.assertEqual(rep1["per_user_results"], rep2["per_user_results"])

    def test_training_context_validation_failures(self):
        """Test TrainingContext input validation and duplicate detection."""
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "i1", 5.0), ("u1", "i1", 4.0)])
        with self.assertRaises(ValueError):
            TrainingContext([("", "i1", 5.0)])
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "", 5.0)])
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "i1", True)])
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "i1", "high")])
        with self.assertRaises(ValueError):
            TrainingContext([("u1", "i1", float("nan"))])

    def test_adversarial_clear_and_remove_mutations_incomplete_maps_fail(self):
        """Test that providers mutating candidate input via clear/remove/pop with incomplete maps fail as ScoringFailureError.

        Covers:
        - Both provider types: score_candidates and callable
        - Both feedback conditions: all_ratings and rating_ge_4
        - Both splits: validation and test (with synthetic history filtering)
        - Asserting missing authoritative score raises ScoringFailureError, never fallback or partial results.
        """
        for condition in ("all_ratings", "rating_ge_4"):
            interactions = [
                ("u1", "item_train_1", 5.0),
                ("u1", "item_train_2", 2.0 if condition == "rating_ge_4" else 5.0),
                ("u2", "item_train_3", 5.0),
                ("u_dummy", "item_cand_1", 5.0),
                ("u_dummy", "item_cand_2", 5.0),
                ("u_dummy", "item_cand_3", 5.0),
                ("u_dummy", "item_cand_4", 5.0),
            ]
            ctx = TrainingContext(interactions, condition=condition)
            ev = Evaluator(ctx)

            for split in ("validation", "test"):
                if split == "validation":
                    targets = [("u1", "item_cand_1", 5.0), ("u2", "item_cand_2", 5.0)]
                    val_hist = None
                else:
                    targets = [("u1", "item_cand_1", 5.0), ("u2", "item_cand_2", 5.0)]
                    val_hist = [
                        ("u1", "item_cand_4", 2.0),
                        {"user_id": "u2", "parent_asin": "item_cand_4", "rating": 5.0},
                    ]

                # 1. Scorer that clears candidates and returns empty dict (score_candidates method)
                class ClearScorerMethod:
                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        candidate_items.clear()
                        return {}, "personalized", None

                with self.assertRaises(ScoringFailureError) as cm:
                    ev.evaluate(targets, ClearScorerMethod(), split=split, validation_history=val_hist)
                self.assertIn("Scorer missing candidate score", str(cm.exception))

                # 2. Scorer that clears candidates and returns empty dict (callable)
                class ClearScorerCallable:
                    def __call__(self, user_id: str, candidate_items: Sequence[str]):
                        candidate_items.clear()
                        return {}, "personalized", None

                with self.assertRaises(ScoringFailureError) as cm:
                    ev.evaluate(targets, ClearScorerCallable(), split=split, validation_history=val_hist)
                self.assertIn("Scorer missing candidate score", str(cm.exception))

                # 3. Scorer that removes / pops an authoritative candidate and only scores remaining (score_candidates)
                class PopScorerMethod:
                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        if len(candidate_items) > 0:
                            candidate_items.pop(0)
                        return {it: 1.0 for it in candidate_items}, "personalized", None

                with self.assertRaises(ScoringFailureError) as cm:
                    ev.evaluate(targets, PopScorerMethod(), split=split, validation_history=val_hist)
                self.assertIn("Scorer missing candidate score", str(cm.exception))

                # 4. Scorer that removes a specific authoritative candidate and only scores remaining (callable)
                class RemoveScorerCallable:
                    def __call__(self, user_id: str, candidate_items: Sequence[str]):
                        if "item_cand_2" in candidate_items:
                            candidate_items.remove("item_cand_2")
                        return {it: 1.0 for it in candidate_items}, "personalized", None

                with self.assertRaises(ScoringFailureError) as cm:
                    ev.evaluate(targets, RemoveScorerCallable(), split=split, validation_history=val_hist)
                self.assertIn("Scorer missing candidate score", str(cm.exception))

                # 5. Scorer that does candidates.clear() but provides a complete score map succeeds!
                class HonestClearBaseline:
                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        return {it: float(len(it)) for it in candidate_items}, "personalized", None

                class ClearScorerCompleteMap:
                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        # Snapshot scores for all candidates before clearing private input
                        scores = {it: float(len(it)) for it in candidate_items}
                        candidate_items.clear()
                        return scores, "personalized", None

                rep_honest_clear = ev.evaluate(targets, HonestClearBaseline(), split=split, validation_history=val_hist)
                rep_clear_complete = ev.evaluate(targets, ClearScorerCompleteMap(), split=split, validation_history=val_hist)
                self.assertEqual(rep_clear_complete["audit"]["eligible_users"], 2)
                self.assertEqual(rep_clear_complete["overall_metrics"], rep_honest_clear["overall_metrics"])
                self.assertEqual(rep_clear_complete["per_user_results"], rep_honest_clear["per_user_results"])
                for ur in rep_clear_complete["per_user_results"]:
                    self.assertGreater(ur["candidate_count"], 0)
                    self.assertEqual(len(ur["top_5"]), min(5, ur["candidate_count"]))

    def test_adversarial_append_and_replace_cold_and_history_ids_cannot_inject(self):
        """Test that appending or replacing candidates with cold/history IDs cannot inject ranked IDs, counts, or coverage.

        Covers:
        - Both provider interfaces: score_candidates and callable
        - Both feedback conditions: all_ratings and rating_ge_4
        - Both validation and test synthetic history filtering
        - Incomplete maps (replacement omitting authoritative candidate) fail as ScoringFailureError
        - Complete maps yield unchanged honest reports despite appended/replaced cold and history IDs with huge scores
        - Independent assertion of candidate counts, filtered IDs, coverage, and registered metrics
        """
        for condition in ("all_ratings", "rating_ge_4"):
            interactions = [
                ("u1", "item_train_1", 5.0),
                ("u1", "item_train_low", 2.0),
                ("u2", "item_train_2", 5.0),
                ("u2", "item_train_low", 2.0),
                ("u_dummy", "item_cand_01", 5.0),
                ("u_dummy", "item_cand_02", 5.0),
                ("u_dummy", "item_cand_03", 5.0),
                ("u_dummy", "item_cand_04", 5.0),
                ("u_dummy", "item_cand_05", 5.0),
                ("u_dummy", "item_cand_06", 5.0),
                ("u_dummy", "item_cand_07", 5.0),
                ("u_dummy", "item_cand_08", 5.0),
                ("u_dummy", "item_cand_09", 5.0),
                ("u_dummy", "item_cand_10", 5.0),
                ("u_dummy", "item_cand_11", 5.0),
                ("u_dummy", "item_cand_12", 5.0),
            ]
            if condition == "all_ratings":
                # item_train_low is warm; we add an explicitly cold item never in interactions
                cold_items = ["item_never_warm_1", "item_never_warm_2"]
            else:
                # In rating_ge_4, item_train_low has only rating 2.0, so it is cold!
                cold_items = ["item_train_low", "item_never_warm_1"]

            ctx = TrainingContext(interactions, condition=condition)
            ev = Evaluator(ctx)

            for split in ("validation", "test"):
                targets = [("u1", "item_cand_01", 5.0), ("u2", "item_cand_02", 5.0)]
                if split == "validation":
                    val_hist = None
                    val_hist_dict = {}
                else:
                    val_hist = [
                        ("u1", "item_cand_11", 2.0),
                        {"user_id": "u2", "parent_asin": "item_cand_12", "rating": 5.0},
                    ]
                    val_hist_dict = {"u1": {"item_cand_11"}, "u2": {"item_cand_12"}}

                # Baseline honest scorer
                class HonestScorer:
                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        return {it: 100.0 - idx for idx, it in enumerate(candidate_items)}, "personalized", None

                honest_report = ev.evaluate(targets, HonestScorer(), split=split, validation_history=val_hist)

                # Case 1: Incomplete map from replace (replacing authoritative candidate with cold ID)
                class ReplaceIncompleteScorer:
                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        candidate_items[0] = cold_items[0]
                        # Scores only mutated list -> missing the original candidate_items[0]
                        return {it: 1.0 for it in candidate_items}, "personalized", None

                with self.assertRaises(ScoringFailureError) as cm:
                    ev.evaluate(targets, ReplaceIncompleteScorer(), split=split, validation_history=val_hist)
                self.assertIn("Scorer missing candidate score", str(cm.exception))

                # Case 2: Complete map with appended cold and history IDs with massive scores (score_candidates)
                class AdversarialAppendMethodScorer:
                    def __init__(self, ctx_ref, vh_map):
                        self.ctx_ref = ctx_ref
                        self.vh_map = vh_map

                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        # Prepare honest scores for all authoritative candidates
                        scores = {it: 100.0 - idx for idx, it in enumerate(candidate_items)}
                        # Mutate private input by appending cold items
                        for ci in cold_items:
                            candidate_items.append(ci)
                            scores[ci] = 99999.0  # Massive score attempting to rank cold item #1!
                        # Mutate private input by appending this user's history items
                        user_history = set(self.ctx_ref.all_rated_history.get(user_id, set())) | set(self.vh_map.get(user_id, set()))
                        for hi in sorted(user_history):
                            candidate_items.append(hi)
                            scores[hi] = 99998.0  # Massive score attempting to rank history item!
                        return scores, "personalized", None

                adv_scorer_method = AdversarialAppendMethodScorer(ctx, val_hist_dict)
                adv_report_method = ev.evaluate(targets, adv_scorer_method, split=split, validation_history=val_hist)

                # Independent assertions on the adversarial report
                self.assertEqual(adv_report_method["audit"]["eligible_users"], honest_report["audit"]["eligible_users"])
                self.assertEqual(adv_report_method["overall_metrics"], honest_report["overall_metrics"])
                self.assertEqual(adv_report_method["candidate_shortages"], honest_report["candidate_shortages"])

                for u_idx, ur in enumerate(adv_report_method["per_user_results"]):
                    honest_ur = honest_report["per_user_results"][u_idx]
                    uid = ur["user_id"]
                    # Candidate count must strictly match authoritative candidate count
                    self.assertEqual(ur["candidate_count"], honest_ur["candidate_count"])
                    # Cold items and user's history items MUST NEVER appear in top_5, top_10, top_20, or top_k
                    for ci in cold_items:
                        self.assertNotIn(ci, ur["top_5"])
                        self.assertNotIn(ci, ur["top_10"])
                        self.assertNotIn(ci, ur["top_20"])
                        self.assertNotIn(ci, ur["top_k"][5])
                        self.assertNotIn(ci, ur["top_k"][10])
                        self.assertNotIn(ci, ur["top_k"][20])
                    user_history = set(ctx.all_rated_history.get(uid, set())) | set(val_hist_dict.get(uid, set()))
                    for hi in user_history:
                        self.assertNotIn(hi, ur["top_5"])
                        self.assertNotIn(hi, ur["top_10"])
                        self.assertNotIn(hi, ur["top_20"])
                        self.assertNotIn(hi, ur["top_k"][5])
                        self.assertNotIn(hi, ur["top_k"][10])
                        self.assertNotIn(hi, ur["top_k"][20])
                    # Top-k rankings and metrics must match honest report exactly
                    self.assertEqual(ur["top_5"], honest_ur["top_5"])
                    self.assertEqual(ur["top_10"], honest_ur["top_10"])
                    self.assertEqual(ur["top_20"], honest_ur["top_20"])
                    self.assertEqual(ur["metrics"], honest_ur["metrics"])

                # CatalogCoverage@10 must match honest report exactly
                self.assertEqual(
                    adv_report_method["overall_metrics"]["CatalogCoverage@10"],
                    honest_report["overall_metrics"]["CatalogCoverage@10"],
                )

                # Case 3: Complete map with replaced items and appended cold/history IDs (callable provider)
                class AdversarialReplaceCallableScorer:
                    def __init__(self, ctx_ref, vh_map):
                        self.ctx_ref = ctx_ref
                        self.vh_map = vh_map

                    def __call__(self, user_id: str, candidate_items: Sequence[str]):
                        # Prepare honest scores before mutation
                        scores = {it: 100.0 - idx for idx, it in enumerate(candidate_items)}
                        # Replace first candidate with cold item in private input
                        candidate_items[0] = cold_items[0]
                        scores[cold_items[0]] = 99999.0
                        # Append history items
                        user_history = set(self.ctx_ref.all_rated_history.get(user_id, set())) | set(self.vh_map.get(user_id, set()))
                        for hi in sorted(user_history):
                            candidate_items.append(hi)
                            scores[hi] = 88888.0
                        return scores, "personalized", None

                adv_scorer_callable = AdversarialReplaceCallableScorer(ctx, val_hist_dict)
                adv_report_callable = ev.evaluate(targets, adv_scorer_callable, split=split, validation_history=val_hist)
                self.assertEqual(adv_report_callable["overall_metrics"], honest_report["overall_metrics"])
                self.assertEqual(adv_report_callable["per_user_results"], honest_report["per_user_results"])

    def test_adversarial_reorder_and_shuffle_mutations(self):
        """Test that providers reordering/shuffling candidate inputs yield unchanged honest reports with complete maps."""
        for condition in ("all_ratings", "rating_ge_4"):
            interactions = [
                ("u1", "item_train_1", 5.0),
                ("u2", "item_train_2", 5.0),
            ] + [("u_dummy", f"item_cand_{i:02d}", 5.0) for i in range(1, 15)]
            ctx = TrainingContext(interactions, condition=condition)
            ev = Evaluator(ctx)

            targets = [("u1", "item_cand_01", 5.0), ("u2", "item_cand_02", 5.0)]
            val_hist = [("u1", "item_cand_13", 2.0), ("u2", "item_cand_14", 5.0)]

            # Honest scorer
            class HonestScorer:
                def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                    return {it: float(len(it) * 10 + int(it.split("_")[-1])) for it in candidate_items}, "personalized", None

            honest_report_val = ev.evaluate(targets, HonestScorer(), split="validation")
            honest_report_test = ev.evaluate(targets, HonestScorer(), split="test", validation_history=val_hist)

            # Adversarial reverse scorer (score_candidates)
            class ReverseScorerMethod:
                def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                    scores = {it: float(len(it) * 10 + int(it.split("_")[-1])) for it in candidate_items}
                    candidate_items.reverse()
                    return scores, "personalized", None

            adv_val = ev.evaluate(targets, ReverseScorerMethod(), split="validation")
            self.assertEqual(adv_val["overall_metrics"], honest_report_val["overall_metrics"])
            self.assertEqual(adv_val["per_user_results"], honest_report_val["per_user_results"])

            adv_test = ev.evaluate(targets, ReverseScorerMethod(), split="test", validation_history=val_hist)
            self.assertEqual(adv_test["overall_metrics"], honest_report_test["overall_metrics"])
            self.assertEqual(adv_test["per_user_results"], honest_report_test["per_user_results"])

            # Adversarial shuffle scorer (callable)
            class ShuffleScorerCallable:
                def __call__(self, user_id: str, candidate_items: Sequence[str]):
                    scores = {it: float(len(it) * 10 + int(it.split("_")[-1])) for it in candidate_items}
                    random.Random(123).shuffle(candidate_items)
                    return scores, "personalized", None

            adv_shuffle_val = ev.evaluate(targets, ShuffleScorerCallable(), split="validation")
            self.assertEqual(adv_shuffle_val["overall_metrics"], honest_report_val["overall_metrics"])
            self.assertEqual(adv_shuffle_val["per_user_results"], honest_report_val["per_user_results"])

            adv_shuffle_test = ev.evaluate(targets, ShuffleScorerCallable(), split="test", validation_history=val_hist)
            self.assertEqual(adv_shuffle_test["overall_metrics"], honest_report_test["overall_metrics"])
            self.assertEqual(adv_shuffle_test["per_user_results"], honest_report_test["per_user_results"])

            # Adversarial sort scorer (score_candidates)
            class SortScorerMethod:
                def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                    scores = {it: float(len(it) * 10 + int(it.split("_")[-1])) for it in candidate_items}
                    candidate_items.sort()
                    return scores, "personalized", None

            adv_sort_val = ev.evaluate(targets, SortScorerMethod(), split="validation")
            self.assertEqual(adv_sort_val["overall_metrics"], honest_report_val["overall_metrics"])
            self.assertEqual(adv_sort_val["per_user_results"], honest_report_val["per_user_results"])

            adv_sort_test = ev.evaluate(targets, SortScorerMethod(), split="test", validation_history=val_hist)
            self.assertEqual(adv_sort_test["overall_metrics"], honest_report_test["overall_metrics"])
            self.assertEqual(adv_sort_test["per_user_results"], honest_report_test["per_user_results"])

            # Adversarial sort reverse scorer (callable)
            class SortReverseScorerCallable:
                def __call__(self, user_id: str, candidate_items: Sequence[str]):
                    scores = {it: float(len(it) * 10 + int(it.split("_")[-1])) for it in candidate_items}
                    candidate_items.sort(reverse=True)
                    return scores, "personalized", None

            adv_sort_rev_val = ev.evaluate(targets, SortReverseScorerCallable(), split="validation")
            self.assertEqual(adv_sort_rev_val["overall_metrics"], honest_report_val["overall_metrics"])
            self.assertEqual(adv_sort_rev_val["per_user_results"], honest_report_val["per_user_results"])

            adv_sort_rev_test = ev.evaluate(targets, SortReverseScorerCallable(), split="test", validation_history=val_hist)
            self.assertEqual(adv_sort_rev_test["overall_metrics"], honest_report_test["overall_metrics"])
            self.assertEqual(adv_sort_rev_test["per_user_results"], honest_report_test["per_user_results"])

    def test_adversarial_multi_user_and_repeated_evaluation_isolation(self):
        """Test multi-user isolation across mutating providers and repeated evaluation runs without state corruption."""
        interactions = [
            ("u1", "item_train_1", 5.0),
            ("u2", "item_train_2", 5.0),
            ("u3_fallback", "item_train_low", 2.0),  # empty positive history under rating_ge_4
        ] + [("u_dummy", f"item_{i:02d}", 5.0) for i in range(1, 20)]
        ctx = TrainingContext(interactions, condition="rating_ge_4")
        ev = Evaluator(ctx)

        targets = [
            ("u1", "item_01", 5.0),
            ("u2", "item_02", 5.0),
            ("u3_fallback", "item_03", 5.0),
        ]
        val_hist = [
            ("u1", "item_18", 2.0),
            ("u2", "item_19", 2.0),
            ("u3_fallback", "item_17", 2.0),
        ]

        # Multi-user mutating scorer: mutates candidates differently per user and stores references
        class StatefulMutatingScorer:
            def __init__(self):
                self.saved_calls = []

            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                self.saved_calls.append(candidate_items)
                scores = {it: 10.0 for it in candidate_items}
                if user_id == "u1":
                    candidate_items.clear()  # clears private input
                elif user_id == "u2":
                    candidate_items.append("item_cold_fake")  # appends to private input
                    scores["item_cold_fake"] = 999.0
                    candidate_items.reverse()
                return scores, "personalized", None

        # Honest baseline
        class HonestScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                return {it: 10.0 for it in candidate_items}, "personalized", None

        honest_report = ev.evaluate(targets, HonestScorer(), split="test", validation_history=val_hist)

        # Repeated evaluation run 1: with mutating scorer
        scorer1 = StatefulMutatingScorer()
        rep1 = ev.evaluate(targets, scorer1, split="test", validation_history=val_hist)

        # Mutate saved calls after evaluation to verify no retrospective corruption
        for saved in scorer1.saved_calls:
            saved.append("corrupt_afterward")

        # Repeated evaluation run 2: with honest scorer
        rep2 = ev.evaluate(targets, HonestScorer(), split="test", validation_history=val_hist)

        # Reports must match honest report exactly
        self.assertEqual(rep1["overall_metrics"], honest_report["overall_metrics"])
        self.assertEqual(rep2["overall_metrics"], honest_report["overall_metrics"])
        self.assertEqual(rep1["per_user_results"], honest_report["per_user_results"])
        self.assertEqual(rep2["per_user_results"], honest_report["per_user_results"])

        # Check user routes and fallback isolation
        per_user_1 = {r["user_id"]: r for r in rep1["per_user_results"]}
        self.assertEqual(per_user_1["u1"]["route"], "personalized")
        self.assertEqual(per_user_1["u2"]["route"], "personalized")
        self.assertEqual(per_user_1["u3_fallback"]["route"], "fallback")
        self.assertEqual(per_user_1["u3_fallback"]["fallback_reason"], "empty_positive_history")

        # Repeated evaluation run 3: incomplete map failure does not corrupt evaluator for run 4
        class FailingScorer:
            def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                if user_id == "u2":
                    candidate_items.clear()
                    return {}, "personalized", None
                return {it: 10.0 for it in candidate_items}, "personalized", None

        with self.assertRaises(ScoringFailureError):
            ev.evaluate(targets, FailingScorer(), split="test", validation_history=val_hist)

        # Repeated evaluation run 4: honest evaluation immediately after failure
        rep4 = ev.evaluate(targets, HonestScorer(), split="test", validation_history=val_hist)
        self.assertEqual(rep4["overall_metrics"], honest_report["overall_metrics"])
        self.assertEqual(rep4["per_user_results"], honest_report["per_user_results"])

    def test_adversarial_hand_calculated_fixed_oracles_and_candidate_isolation(self):
        """Compact hand-calculated fixture with literal expected full candidate lists/counts,
        warm-pool denominator, union coverage, and NDCG/HitRate at all cutoffs for each user and
        macro across both feedback conditions and validation/test synthetic history filtering.

        Verifies:
        - Condition-differing warmth: item_low is warm in all_ratings (denom 10) but cold in rating_ge_4 (denom 9).
        - Low-rated validation record (rating 2.0) item_vlow is filtered before provider.
        - Provider receives list type and exact expected filtered IDs before mutation.
        - All complete-map actions (clear, remove, pop, append, replace, reverse, shuffle, sort)
          across both score_candidates and callable interfaces yield unchanged honest reports and
          match literal hand-calculated fixed oracles.
        - Added cold/history IDs in append/replace never enter ranked lists, counts, or coverage.
        - Missing-map actions fail as ScoringFailureError with no fallback or silent candidate shrinking.
        """
        interactions = [
            ("u1", "item_t1", 5.0),
            ("u2", "item_t2", 5.0),
            ("u_dummy", "item_low", 2.0),
            ("u_dummy", "item_01", 5.0),
            ("u_dummy", "item_02", 5.0),
            ("u_dummy", "item_03", 5.0),
            ("u_dummy", "item_04", 5.0),
            ("u_dummy", "item_05", 5.0),
            ("u_dummy", "item_06", 5.0),
            ("u_dummy", "item_vlow", 5.0),
        ]

        targets = [
            ("u1", "item_01", 5.0),
            ("u2", "item_06", 5.0),
        ]

        val_history = [
            ("u1", "item_vlow", 2.0),
            {"user_id": "u2", "parent_asin": "item_vlow", "rating": 2.0},
        ]

        base_scores = {
            "item_01": 100.0,
            "item_02": 90.0,
            "item_03": 80.0,
            "item_04": 70.0,
            "item_05": 60.0,
            "item_06": 50.0,
            "item_low": 30.0,
            "item_t1": 20.0,
            "item_t2": 10.0,
            "item_vlow": 5.0,
        }

        # Literal hand-calculated oracle values
        ndcg_r6 = 1.0 / math.log2(6.0 + 1.0)  # 1.0 / log2(7.0)
        expected_macro_ndcg10 = (1.0 + ndcg_r6) / 2.0
        expected_macro_ndcg20 = (1.0 + ndcg_r6) / 2.0

        oracle_configs = {
            ("all_ratings", "validation"): {
                "warm_pool_size": 10,
                "coverage_10": 1.0,  # 10 / 10
                "expected_candidates": {
                    "u1": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_low", "item_t2", "item_vlow"],
                    "u2": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_low", "item_t1", "item_vlow"],
                },
                "expected_counts": {"u1": 9, "u2": 9},
            },
            ("all_ratings", "test"): {
                "warm_pool_size": 10,
                "coverage_10": 0.9,  # 9 / 10 (item_vlow filtered by validation history)
                "expected_candidates": {
                    "u1": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_low", "item_t2"],
                    "u2": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_low", "item_t1"],
                },
                "expected_counts": {"u1": 8, "u2": 8},
            },
            ("rating_ge_4", "validation"): {
                "warm_pool_size": 9,
                "coverage_10": 1.0,  # 9 / 9 (item_low is cold)
                "expected_candidates": {
                    "u1": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_t2", "item_vlow"],
                    "u2": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_t1", "item_vlow"],
                },
                "expected_counts": {"u1": 8, "u2": 8},
            },
            ("rating_ge_4", "test"): {
                "warm_pool_size": 9,
                "coverage_10": 8.0 / 9.0,  # 8 / 9 (item_low cold, item_vlow val-filtered)
                "expected_candidates": {
                    "u1": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_t2"],
                    "u2": ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_t1"],
                },
                "expected_counts": {"u1": 7, "u2": 7},
            },
        }

        for (condition, split), conf in oracle_configs.items():
            ctx = TrainingContext(interactions, condition=condition)
            ev = Evaluator(ctx)
            current_vh = val_history if split == "test" else None

            exp_cands = conf["expected_candidates"]
            exp_counts = conf["expected_counts"]
            exp_warm_size = conf["warm_pool_size"]
            exp_cov = conf["coverage_10"]

            # 1. Honest baseline provider (asserts received candidates and type)
            class HonestBaseline:
                def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                    assert isinstance(candidate_items, list), f"Expected list, got {type(candidate_items)}"
                    assert candidate_items == exp_cands[user_id], f"Candidate mismatch for {user_id}: {candidate_items} != {exp_cands[user_id]}"
                    return {it: base_scores[it] for it in candidate_items}, "personalized", None

            honest_rep = ev.evaluate(targets, HonestBaseline(), split=split, validation_history=current_vh)

            # Assert honest report against literal fixed oracles
            self.assertEqual(honest_rep["audit"]["warm_pool_size"], exp_warm_size)
            self.assertEqual(honest_rep["audit"]["eligible_users"], 2)
            self.assertEqual(honest_rep["candidate_shortages"], {5: 0, 10: 2, 20: 2})
            self.assertAlmostEqual(honest_rep["overall_metrics"]["CatalogCoverage@10"], exp_cov)
            self.assertAlmostEqual(honest_rep["overall_metrics"]["HitRate@5"], 0.5)
            self.assertAlmostEqual(honest_rep["overall_metrics"]["NDCG@5"], 0.5)
            self.assertAlmostEqual(honest_rep["overall_metrics"]["HitRate@10"], 1.0)
            self.assertAlmostEqual(honest_rep["overall_metrics"]["NDCG@10"], expected_macro_ndcg10)
            self.assertAlmostEqual(honest_rep["overall_metrics"]["HitRate@20"], 1.0)
            self.assertAlmostEqual(honest_rep["overall_metrics"]["NDCG@20"], expected_macro_ndcg20)

            per_user_h = {r["user_id"]: r for r in honest_rep["per_user_results"]}
            # User 1 (target item_01 at rank 1)
            u1_h = per_user_h["u1"]
            self.assertEqual(u1_h["candidate_count"], exp_counts["u1"])
            self.assertEqual(u1_h["metrics"]["HitRate@5"], 1.0)
            self.assertAlmostEqual(u1_h["metrics"]["NDCG@5"], 1.0)
            self.assertEqual(u1_h["metrics"]["HitRate@10"], 1.0)
            self.assertAlmostEqual(u1_h["metrics"]["NDCG@10"], 1.0)
            self.assertEqual(u1_h["metrics"]["HitRate@20"], 1.0)
            self.assertAlmostEqual(u1_h["metrics"]["NDCG@20"], 1.0)
            self.assertEqual(u1_h["top_5"], ["item_01", "item_02", "item_03", "item_04", "item_05"])

            # User 2 (target item_06 at rank 6)
            u2_h = per_user_h["u2"]
            self.assertEqual(u2_h["candidate_count"], exp_counts["u2"])
            self.assertEqual(u2_h["metrics"]["HitRate@5"], 0.0)
            self.assertAlmostEqual(u2_h["metrics"]["NDCG@5"], 0.0)
            self.assertEqual(u2_h["metrics"]["HitRate@10"], 1.0)
            self.assertAlmostEqual(u2_h["metrics"]["NDCG@10"], ndcg_r6)
            self.assertEqual(u2_h["metrics"]["HitRate@20"], 1.0)
            self.assertAlmostEqual(u2_h["metrics"]["NDCG@20"], ndcg_r6)
            self.assertEqual(u2_h["top_5"], ["item_01", "item_02", "item_03", "item_04", "item_05"])

            # 2. Test all complete-map mutation actions through BOTH score_candidates and callable interfaces
            complete_actions = [
                ("clear", lambda uid, cands, sc: cands.clear()),
                ("remove", lambda uid, cands, sc: cands.remove(cands[0])),
                ("pop", lambda uid, cands, sc: cands.pop(0)),
                ("append", lambda uid, cands, sc: (
                    cands.append("item_cold_fake"),
                    cands.append("item_t1" if uid == "u1" else "item_t2"),
                    sc.__setitem__("item_cold_fake", 99999.0),
                    sc.__setitem__("item_t1" if uid == "u1" else "item_t2", 99998.0),
                )),
                ("replace", lambda uid, cands, sc: (
                    cands.__setitem__(0, "item_cold_fake"),
                    cands.__setitem__(1, "item_t1" if uid == "u1" else "item_t2"),
                    sc.__setitem__("item_cold_fake", 99999.0),
                    sc.__setitem__("item_t1" if uid == "u1" else "item_t2", 99998.0),
                )),
                ("reverse", lambda uid, cands, sc: cands.reverse()),
                ("shuffle", lambda uid, cands, sc: random.Random(42).shuffle(cands)),
                ("sort", lambda uid, cands, sc: cands.sort(reverse=True)),
            ]

            for action_name, action_fn in complete_actions:
                # 2a. score_candidates interface
                class CompleteMapMethodScorer:
                    def __init__(self, fn, expected_map):
                        self.fn = fn
                        self.expected_map = expected_map

                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        assert isinstance(candidate_items, list), f"Expected list, got {type(candidate_items)}"
                        assert candidate_items == self.expected_map[user_id], f"Mismatch for {user_id}: {candidate_items} != {self.expected_map[user_id]}"
                        scores = {it: base_scores[it] for it in self.expected_map[user_id]}
                        self.fn(user_id, candidate_items, scores)
                        return scores, "personalized", None

                scorer_method = CompleteMapMethodScorer(action_fn, exp_cands)
                rep_m = ev.evaluate(targets, scorer_method, split=split, validation_history=current_vh)
                self.assertEqual(rep_m["audit"]["warm_pool_size"], exp_warm_size)
                self.assertAlmostEqual(rep_m["overall_metrics"]["CatalogCoverage@10"], exp_cov)
                self.assertEqual(rep_m["overall_metrics"], honest_rep["overall_metrics"])
                self.assertEqual(rep_m["per_user_results"], honest_rep["per_user_results"])

                # 2b. callable interface
                class CompleteMapCallableScorer:
                    def __init__(self, fn, expected_map):
                        self.fn = fn
                        self.expected_map = expected_map

                    def __call__(self, user_id: str, candidate_items: Sequence[str]):
                        assert isinstance(candidate_items, list), f"Expected list, got {type(candidate_items)}"
                        assert candidate_items == self.expected_map[user_id], f"Mismatch for {user_id}: {candidate_items} != {self.expected_map[user_id]}"
                        scores = {it: base_scores[it] for it in self.expected_map[user_id]}
                        self.fn(user_id, candidate_items, scores)
                        return scores, "personalized", None

                scorer_callable = CompleteMapCallableScorer(action_fn, exp_cands)
                rep_c = ev.evaluate(targets, scorer_callable, split=split, validation_history=current_vh)
                self.assertEqual(rep_c["audit"]["warm_pool_size"], exp_warm_size)
                self.assertAlmostEqual(rep_c["overall_metrics"]["CatalogCoverage@10"], exp_cov)
                self.assertEqual(rep_c["overall_metrics"], honest_rep["overall_metrics"])
                self.assertEqual(rep_c["per_user_results"], honest_rep["per_user_results"])

            # 3. Test missing-map mutation actions through BOTH interfaces (must raise ScoringFailureError)
            missing_actions = [
                ("clear", lambda uid, cands: (cands.clear(), {})),
                ("remove", lambda uid, cands: (cands.remove(cands[0]), {it: base_scores[it] for it in cands})),
                ("pop", lambda uid, cands: (cands.pop(0), {it: base_scores[it] for it in cands})),
                ("replace", lambda uid, cands: (cands.__setitem__(0, "item_cold_fake"), {it: base_scores.get(it, 999.0) for it in cands})),
            ]

            for action_name, action_fn in missing_actions:
                # 3a. score_candidates interface
                class MissingMapMethodScorer:
                    def __init__(self, fn, expected_map):
                        self.fn = fn
                        self.expected_map = expected_map

                    def score_candidates(self, user_id: str, candidate_items: Sequence[str]):
                        assert isinstance(candidate_items, list), f"Expected list, got {type(candidate_items)}"
                        assert candidate_items == self.expected_map[user_id], f"Mismatch for {user_id}: {candidate_items} != {self.expected_map[user_id]}"
                        _, scores = self.fn(user_id, candidate_items)
                        return scores, "personalized", None

                with self.assertRaises(ScoringFailureError) as cm:
                    ev.evaluate(targets, MissingMapMethodScorer(action_fn, exp_cands), split=split, validation_history=current_vh)
                self.assertIn("Scorer missing candidate score", str(cm.exception))

                # 3b. callable interface
                class MissingMapCallableScorer:
                    def __init__(self, fn, expected_map):
                        self.fn = fn
                        self.expected_map = expected_map

                    def __call__(self, user_id: str, candidate_items: Sequence[str]):
                        assert isinstance(candidate_items, list), f"Expected list, got {type(candidate_items)}"
                        assert candidate_items == self.expected_map[user_id], f"Mismatch for {user_id}: {candidate_items} != {self.expected_map[user_id]}"
                        _, scores = self.fn(user_id, candidate_items)
                        return scores, "personalized", None

                with self.assertRaises(ScoringFailureError) as cm:
                    ev.evaluate(targets, MissingMapCallableScorer(action_fn, exp_cands), split=split, validation_history=current_vh)
                self.assertIn("Scorer missing candidate score", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
