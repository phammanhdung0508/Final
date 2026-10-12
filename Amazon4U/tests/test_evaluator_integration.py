"""Independent integration tests for Amazon4U recommendation evaluator subsystem.

Verifies:
- Public core Evaluator candidate mutation and strict test-history invariants
- Hand-written small KG fallback, subgroup metrics, and validation mixture selection tie oracle
- Guarded metadata fallback rejection (fake scorer, wrong context, usable profiles, corrupt profiles, spoofed scores/reasons)
- Canonical class and exception identity across package (scripts.*) and top-level imports in the same interpreter
"""

import math
import subprocess
import sys
import unittest
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.recommendation_core import (
    BaseEvaluator,
    Evaluator,
    KGOnlyScorer,
    PopularityScorer,
    ScoringFailureError,
    TrainingContext,
    select_kg_only_mixture,
)


class CandidateMutationAndStrictHistoryOraclesTests(unittest.TestCase):
    """Test candidate mutation isolation and strict test-history invariants through core Evaluator."""

    def test_core_evaluator_candidate_mutation_isolation(self):
        """Scorer mutating the candidate list passed to it must not corrupt ranking or context."""
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
            ("u4", "p4", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)

        class MutatingScorer:
            def score_candidates(self, user_id: str, candidates: Sequence[str]) -> Tuple[Dict[str, float], str, Optional[str]]:
                # Attempt in-place mutation of candidates if it is a list
                if isinstance(candidates, list):
                    candidates.append("injected_item")
                    if len(candidates) > 1:
                        candidates.pop(0)
                scores = {"p2": 10.0, "p3": 20.0, "p4": 5.0}
                return scores, "personalized", None

        val_targets = [("u1", "p3", 5.0)]
        res = evaluator.evaluate(val_targets, MutatingScorer(), split="validation")

        self.assertEqual(res["audit"]["eligible_users"], 1)
        u_res = res["per_user_results"][0]
        # Candidate count for u1 should be 3: p2, p3, p4 (p1 is in training history)
        self.assertEqual(u_res["candidate_count"], 3)
        # p3 has highest score (20.0), so rank 1
        self.assertEqual(u_res["top_5"][0], "p3")
        self.assertEqual(u_res["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(u_res["metrics"]["NDCG@5"], 1.0)
        # Warm items in context remain untouched
        self.assertEqual(ctx.warm_items, {"p1", "p2", "p3", "p4"})

    def test_core_evaluator_strict_test_history_invariants(self):
        """Strict test history filtering, low-rated validation exclusion, and error handling."""
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        evaluator = Evaluator(ctx)
        scorer = PopularityScorer(ctx)

        # 1. Missing validation history raises ValueError on test split
        with self.assertRaises(ValueError):
            evaluator.evaluate([("u1", "p3", 5.0)], scorer, split="test", validation_history=None)

        # 2. Known target user with empty validation history raises ValueError
        with self.assertRaises(ValueError):
            evaluator.evaluate([("u1", "p3", 5.0)], scorer, split="test", validation_history={"u1": []})

        # 3. Duplicate validation history record raises ValueError
        with self.assertRaises(ValueError):
            evaluator.evaluate(
                [("u1", "p3", 5.0)],
                scorer,
                split="test",
                validation_history=[("u1", "p2", 1.0), ("u1", "p2", 1.0)],
            )

        # 4. Low-rated validation record (rating 1.0 < 4.0) is strictly filtered from test candidates
        val_history = [("u1", "p2", 1.0)]
        res = evaluator.evaluate([("u1", "p3", 5.0)], scorer, split="test", validation_history=val_history)
        self.assertEqual(res["audit"]["eligible_users"], 1)
        # Candidates for u1: warm items are {p1, p2, p3}. p1 in train, p2 in val history -> only p3 is candidate
        self.assertEqual(res["per_user_results"][0]["candidate_count"], 1)
        self.assertEqual(res["per_user_results"][0]["top_10"], ["p3"])

        # 5. Heldout test target that matches validation item is counted as history overlap
        res_overlap = evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="test", validation_history=val_history)
        self.assertEqual(res_overlap["audit"]["eligible_users"], 0)
        self.assertEqual(res_overlap["audit"]["history_overlaps"], 1)


class KGOnlyFallbackSubgroupAndMixtureSelectionOraclesTests(unittest.TestCase):
    """Test hand-written small KG fallback, subgroup metrics, and validation mixture selection."""

    def test_kg_only_unusable_metadata_profile_subgroup_metrics_oracle(self):
        """User with empty positively weighted metadata profile falls back to exact condition-positive popularity."""
        train = [
            ("u_meta", "p1", 5.0),
            ("u_empty", "p1", 5.0),
            ("u_other", "p2", 5.0),
            ("u_other", "p3", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        # p1 has category and brand, but u_empty rated only p1 which we simulate as having missing metadata
        item_categories = {"p1": ["cat_A"], "p2": ["cat_A"], "p3": []}
        item_brands = {"p1": None, "p2": "brand_X", "p3": None}

        # Train with a user who rated an item with no category:
        train_custom = [
            ("u1", "p_nocat", 5.0),
            ("u2", "p_withcat", 5.0),
            ("u3", "p_withcat", 5.0),
            ("u4", "p_other", 5.0),
        ]
        ctx_custom = TrainingContext(train_custom, condition="all_ratings")
        # p_nocat has empty categories and no brand
        cats = {"p_nocat": [], "p_withcat": ["c1"], "p_other": ["c1"]}
        brands = {"p_nocat": None, "p_withcat": "b1", "p_other": None}

        # Mixture (1.0, 0.0): category weight is 1.0, brand weight is 0.0
        scorer = KGOnlyScorer(ctx_custom, cats, brands, mixture=(1.0, 0.0))
        # u1 rated only p_nocat, which has empty categories, so u1's user category profile is empty
        self.assertEqual(scorer.user_cat_vec["u1"], {})
        # u1 has positive history {p_nocat}, but its positively weighted category channel is unusable
        scores, route, reason = scorer.score_candidates("u1", ["p_withcat", "p_other"])
        self.assertEqual(route, "fallback")
        self.assertEqual(reason, "unusable_metadata_profile")
        # Scores must be exact condition-positive popularity
        self.assertEqual(scores["p_withcat"], float(ctx_custom.popularity["p_withcat"]))
        self.assertEqual(scores["p_other"], float(ctx_custom.popularity["p_other"]))

        evaluator = Evaluator(ctx_custom)
        targets = [
            ("u1", "p_withcat", 5.0),  # fallback user
            ("u2", "p_other", 5.0),    # personalized user
        ]
        report = evaluator.evaluate(targets, scorer, split="validation")

        self.assertEqual(report["audit"]["eligible_users"], 2)
        subgroups = report["subgroups"]

        # Personalized subgroup
        self.assertEqual(subgroups["personalized"]["count"], 1)
        self.assertIsNotNone(subgroups["personalized"]["metrics"]["NDCG@10"])

        # Fallback subgroup
        self.assertEqual(subgroups["fallback"]["count"], 1)
        self.assertIsNotNone(subgroups["fallback"]["metrics"]["NDCG@10"])
        self.assertIsNotNone(subgroups["fallback"]["metrics"]["HitRate@10"])
        self.assertIsNotNone(subgroups["fallback"]["metrics"]["CatalogCoverage@10"])

        # by_reason subgroup
        self.assertIn("unusable_metadata_profile", subgroups["fallback"]["by_reason"])
        reason_group = subgroups["fallback"]["by_reason"]["unusable_metadata_profile"]
        self.assertEqual(reason_group["count"], 1)
        # For u1: candidates are p_withcat (pop=2) and p_other (pop=1).
        # Target is p_withcat, which has pop=2 (rank 1).
        # NDCG@10 = 1.0, HitRate@10 = 1.0
        self.assertEqual(reason_group["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(reason_group["metrics"]["NDCG@10"], 1.0)

    def test_select_kg_only_mixture_validation_tie_oracle(self):
        """Validation metric ties must deterministically select the earliest ordered entry."""
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        # Symmetrical categories and brands such that all mixtures give identical rankings
        cats = {"p1": ["c1"], "p2": ["c1"], "p3": ["c1"]}
        brands = {"p1": "b1", "p2": "b1", "p3": "b1"}
        targets = [("u1", "p2", 5.0), ("u2", "p3", 5.0)]

        best_mixture, trials = select_kg_only_mixture(ctx, cats, brands, targets)
        # KGOnlyScorer.GRID is ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5))
        # Exact tie on NDCG@10 must retain earliest entry: (1.0, 0.0)
        self.assertEqual(best_mixture, (1.0, 0.0))
        self.assertEqual(len(trials), 3)
        for m in KGOnlyScorer.GRID:
            self.assertIn(m, trials)


class MetadataFallbackRejectionTests(unittest.TestCase):
    """Test strict rejection of invalid or spoofed metadata fallback claims in core Evaluator."""

    def setUp(self):
        self.train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        self.ctx = TrainingContext(self.train, condition="all_ratings")
        self.evaluator = Evaluator(self.ctx)

    def test_fake_metadata_fallback_from_non_kg_scorer_rejected(self):
        """Non-KGOnlyScorer cannot claim unusable_metadata_profile fallback."""
        class FakeScorer:
            def score_candidates(self, u, cands):
                scores = {c: 1.0 for c in cands}
                return scores, "fallback", "unusable_metadata_profile"

        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], FakeScorer(), split="validation")

    def test_metadata_fallback_with_wrong_context_rejected(self):
        """Legitimate KGOnlyScorer built with a different TrainingContext cannot be evaluated."""
        ctx_other = TrainingContext(self.train, condition="all_ratings")
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": None, "p2": None, "p3": None}
        scorer_other = KGOnlyScorer(ctx_other, cats, brands, mixture=(1.0, 0.0))

        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer_other, split="validation")

    def test_metadata_fallback_with_usable_profile_rejected(self):
        """User with usable positively weighted metadata channel cannot claim fallback."""
        cats = {"p1": ["c1"], "p2": ["c1"], "p3": ["c1"]}
        brands = {"p1": "b1", "p2": "b1", "p3": "b1"}
        scorer = KGOnlyScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))

        # Manually force scorer to return fallback for a user that has usable categories
        original_score = scorer.score_candidates
        def spoofed_score(u, cands):
            return {c: float(self.ctx.popularity.get(c, 0)) for c in cands}, "fallback", "unusable_metadata_profile"
        scorer.score_candidates = spoofed_score

        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")

    def test_metadata_fallback_with_corrupt_profile_rejected(self):
        """Corrupt non-finite value in profile (even in unused channel) fails fallback validation."""
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": "b1", "p2": "b2", "p3": "b3"}
        scorer = KGOnlyScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        # Corrupt unused brand channel
        scorer.user_brand_vec["u1"]["b1"] = float("nan")

        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")

    def test_metadata_fallback_with_spoofed_popularity_scores_rejected(self):
        """Fallback scores that do not equal exact condition-positive popularity are rejected."""
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": None, "p2": None, "p3": None}
        scorer = KGOnlyScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))

        def spoofed_scores(u, cands):
            # Alter popularity score for one candidate
            scores = {c: float(self.ctx.popularity.get(c, 0)) for c in cands}
            scores["p2"] = 999.0
            return scores, "fallback", "unusable_metadata_profile"
        scorer.score_candidates = spoofed_scores

        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")

    def test_metadata_fallback_with_spoofed_reason_rejected(self):
        """Arbitrary fallback reasons are rejected."""
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": None, "p2": None, "p3": None}
        scorer = KGOnlyScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))

        for bogus_reason in ("scoring_error", "arbitrary_reason", "model_choice"):
            def make_spoofed(reason):
                def spoofed(u, cands):
                    scores = {c: float(self.ctx.popularity.get(c, 0)) for c in cands}
                    return scores, "fallback", reason
                return spoofed
            scorer.score_candidates = make_spoofed(bogus_reason)
            with self.assertRaises(ScoringFailureError):
                self.evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")

    def test_base_evaluator_rejects_all_provider_fallback_claims(self):
        """BaseEvaluator must reject provider fallback claims unconditionally."""
        base_evaluator = BaseEvaluator(self.ctx)
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": None, "p2": None, "p3": None}
        scorer = KGOnlyScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))

        with self.assertRaises(ScoringFailureError):
            base_evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")


class ImportAliasesAndCanonicalIdentityTests(unittest.TestCase):
    """Test import alias canonicalization and identical class/exception objects."""

    def test_in_process_canonical_identity(self):
        import scripts.recommendation_core as rc_pkg
        import scripts.recommendation_evaluation as re_pkg
        import recommendation_core as rc_top
        import recommendation_evaluation as re_top

        self.assertIs(rc_pkg.TrainingContext, re_pkg.TrainingContext)
        self.assertIs(rc_top.TrainingContext, re_top.TrainingContext)
        self.assertIs(rc_pkg.TrainingContext, rc_top.TrainingContext)

        self.assertIs(rc_pkg.ScoringFailureError, re_pkg.ScoringFailureError)
        self.assertIs(rc_top.ScoringFailureError, re_top.ScoringFailureError)
        self.assertIs(rc_pkg.ScoringFailureError, rc_top.ScoringFailureError)

        self.assertIs(rc_pkg.Evaluator, rc_top.Evaluator)
        self.assertIs(rc_pkg.KGOnlyScorer, rc_top.KGOnlyScorer)
        self.assertIs(rc_pkg.PopularityScorer, rc_top.PopularityScorer)
        self.assertIs(rc_pkg.BaseEvaluator, re_pkg.Evaluator)

    def test_same_interpreter_import_order_invariance_in_subprocesses(self):
        """Test importing package first vs top-level first in fresh python interpreters."""
        # Order 1: package first, then top-level
        code_pkg_first = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
import scripts.recommendation_core as rc_pkg
import scripts.recommendation_evaluation as re_pkg
import recommendation_core as rc_top
import recommendation_evaluation as re_top

assert rc_pkg.TrainingContext is rc_top.TrainingContext
assert rc_pkg.ScoringFailureError is rc_top.ScoringFailureError
assert rc_pkg.Evaluator is rc_top.Evaluator
assert rc_pkg.KGOnlyScorer is rc_top.KGOnlyScorer
assert rc_pkg.BaseEvaluator is re_pkg.Evaluator
"""
        res1 = subprocess.run([sys.executable, "-B", "-c", code_pkg_first], capture_output=True, text=True)
        self.assertEqual(res1.returncode, 0, f"pkg_first failed: {res1.stderr}")

        # Order 2: top-level first, then package
        code_top_first = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
import recommendation_core as rc_top
import recommendation_evaluation as re_top
import scripts.recommendation_core as rc_pkg
import scripts.recommendation_evaluation as re_pkg

assert rc_pkg.TrainingContext is rc_top.TrainingContext
assert rc_pkg.ScoringFailureError is rc_top.ScoringFailureError
assert rc_pkg.Evaluator is rc_top.Evaluator
assert rc_pkg.KGOnlyScorer is rc_top.KGOnlyScorer
assert rc_pkg.BaseEvaluator is re_pkg.Evaluator
"""
        res2 = subprocess.run([sys.executable, "-B", "-c", code_top_first], capture_output=True, text=True)
        self.assertEqual(res2.returncode, 0, f"top_first failed: {res2.stderr}")


class FallbackReasonSchemaAndRejectionTests(unittest.TestCase):
    """Test core vs standalone fallback reason schema and rejection of invalid claims."""

    def test_core_evaluator_schema_empty_targets(self):
        """Core Evaluator with empty targets must emit both fallback reasons with count 0 and all-None metrics."""
        ctx = TrainingContext([("u1", "p1", 5.0)], condition="all_ratings")
        evaluator = Evaluator(ctx)
        rep = evaluator.evaluate([], PopularityScorer(ctx), split="validation")

        self.assertEqual(rep["audit"]["total_targets"], 0)
        self.assertEqual(rep["audit"]["eligible_users"], 0)
        by_reason = rep["subgroups"]["fallback"]["by_reason"]
        self.assertEqual(set(by_reason.keys()), {"empty_positive_history", "unusable_metadata_profile"})
        self.assertEqual(by_reason["empty_positive_history"]["count"], 0)
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 0)
        for metric_name in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(by_reason["empty_positive_history"]["metrics"][metric_name])
            self.assertIsNone(by_reason["unusable_metadata_profile"]["metrics"][metric_name])
            self.assertIsNone(rep["overall_metrics"][metric_name])

    def test_core_evaluator_schema_no_eligible_users(self):
        """Core Evaluator with targets but zero eligible users must emit both reasons with count 0 and all-None metrics."""
        ctx = TrainingContext([("u1", "p1", 5.0)], condition="all_ratings")
        evaluator = Evaluator(ctx)
        rep = evaluator.evaluate([("u_unknown", "p1", 5.0)], PopularityScorer(ctx), split="validation")

        self.assertEqual(rep["audit"]["total_targets"], 1)
        self.assertEqual(rep["audit"]["eligible_users"], 0)
        self.assertEqual(rep["audit"]["unknown_users"], 1)
        by_reason = rep["subgroups"]["fallback"]["by_reason"]
        self.assertEqual(set(by_reason.keys()), {"empty_positive_history", "unusable_metadata_profile"})
        self.assertEqual(by_reason["empty_positive_history"]["count"], 0)
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 0)
        for metric_name in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(by_reason["empty_positive_history"]["metrics"][metric_name])
            self.assertIsNone(by_reason["unusable_metadata_profile"]["metrics"][metric_name])

    def test_core_evaluator_schema_no_fallback_users(self):
        """Core Evaluator with eligible users but 0 fallback users must emit both reasons with count 0 and all-None metrics."""
        train = [("u1", "p1", 5.0), ("u2", "p2", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = Evaluator(ctx)
        rep = evaluator.evaluate([("u1", "p2", 5.0)], PopularityScorer(ctx), split="validation")

        self.assertEqual(rep["audit"]["eligible_users"], 1)
        self.assertEqual(rep["subgroups"]["non_personalized"]["count"], 1)
        self.assertEqual(rep["subgroups"]["fallback"]["count"], 0)
        by_reason = rep["subgroups"]["fallback"]["by_reason"]
        self.assertEqual(set(by_reason.keys()), {"empty_positive_history", "unusable_metadata_profile"})
        self.assertEqual(by_reason["empty_positive_history"]["count"], 0)
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 0)
        for metric_name in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(by_reason["empty_positive_history"]["metrics"][metric_name])
            self.assertIsNone(by_reason["unusable_metadata_profile"]["metrics"][metric_name])

    def test_core_evaluator_schema_empty_positive_only_fallback(self):
        """Core Evaluator with only empty-positive fallback must include empty count>0 and unusable_metadata_profile count=0 all-None."""
        train = [
            ("u_empty", "p1", 2.0),
            ("u_other", "p2", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        evaluator = Evaluator(ctx)
        rep = evaluator.evaluate([("u_empty", "p2", 5.0)], PopularityScorer(ctx), split="validation")

        self.assertEqual(rep["audit"]["eligible_users"], 1)
        self.assertEqual(rep["subgroups"]["fallback"]["count"], 1)
        by_reason = rep["subgroups"]["fallback"]["by_reason"]
        self.assertEqual(set(by_reason.keys()), {"empty_positive_history", "unusable_metadata_profile"})
        self.assertEqual(by_reason["empty_positive_history"]["count"], 1)
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 0)
        self.assertIsNotNone(by_reason["empty_positive_history"]["metrics"]["NDCG@10"])
        for metric_name in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(by_reason["unusable_metadata_profile"]["metrics"][metric_name])

    def test_standalone_evaluator_default_fallback_reasons_and_rejection(self):
        """Standalone BaseEvaluator keeps empty_positive_history ONLY and rejects provider fallback claims."""
        train = [("u1", "p1", 5.0), ("u2", "p2", 5.0)]
        ctx = TrainingContext(train, condition="all_ratings")
        evaluator = BaseEvaluator(ctx)

        # Standalone schema on empty targets
        rep_empty = evaluator.evaluate([], PopularityScorer(ctx), split="validation")
        self.assertEqual(set(rep_empty["subgroups"]["fallback"]["by_reason"].keys()), {"empty_positive_history"})
        self.assertNotIn("unusable_metadata_profile", rep_empty["subgroups"]["fallback"]["by_reason"])

        # Standalone schema on non-fallback evaluation
        rep_non_fb = evaluator.evaluate([("u1", "p2", 5.0)], PopularityScorer(ctx), split="validation")
        self.assertEqual(set(rep_non_fb["subgroups"]["fallback"]["by_reason"].keys()), {"empty_positive_history"})
        self.assertNotIn("unusable_metadata_profile", rep_non_fb["subgroups"]["fallback"]["by_reason"])

        # Rejection of provider returning fallback route
        class ProviderFallbackScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "fallback", None

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], ProviderFallbackScorer(), split="validation")

        # Rejection of provider returning fallback_reason
        class ProviderReasonScorer:
            def score_candidates(self, u, cands):
                return {c: 1.0 for c in cands}, "personalized", "any_reason"

        with self.assertRaises(ScoringFailureError):
            evaluator.evaluate([("u1", "p2", 5.0)], ProviderReasonScorer(), split="validation")


class AdversarialProfileValidationAndSuperDelegationTests(unittest.TestCase):
    """Test adversarial validator override bypass attempts and ordinary route super-delegation."""

    def setUp(self):
        train = [
            ("u1", "p1", 5.0),
            ("u2", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        self.ctx = TrainingContext(train, condition="all_ratings")
        self.evaluator = Evaluator(self.ctx)

    def test_adversarial_scorer_override_validate_profile_dict_rejected(self):
        """Scorer overriding _validate_profile_dict cannot bypass authoritative interface validation gate."""
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": "b1", "p2": "b2", "p3": "b3"}

        class AdversarialScorer(KGOnlyScorer):
            def _validate_profile_dict(self, *args, **kwargs):
                # Attempt to bypass validation by doing nothing
                return None

            def score_candidates(self, user_id, candidates):
                # Forged score method returning exact popularity
                scores = {item: float(self.context.popularity.get(item, 0)) for item in candidates}
                return scores, "fallback", "unusable_metadata_profile"

        scorer = AdversarialScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        # Corrupt unused brand channel with non-finite NaN
        scorer.user_brand_vec["u1"]["b1"] = float("nan")

        # Acceptance gate must call authoritative KGOnlyScorer._validate_profile_dict, catching NaN
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")

    def test_adversarial_candidate_profile_override_rejected(self):
        """Candidate product profile corruption in unused channel cannot bypass authoritative gate."""
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": "b1", "p2": "b2", "p3": "b3"}

        class AdversarialCandidateScorer(KGOnlyScorer):
            def _validate_profile_dict(self, *args, **kwargs):
                return None

            def score_candidates(self, user_id, candidates):
                scores = {item: float(self.context.popularity.get(item, 0)) for item in candidates}
                return scores, "fallback", "unusable_metadata_profile"

        scorer = AdversarialCandidateScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        # Corrupt candidate item brand profile with infinity
        scorer.product_brand_vec["p2"]["b2"] = float("inf")

        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer, split="validation")

    def test_malformed_outer_container_claims_rejected_as_scoring_failure(self):
        """Malformed outer profile containers (missing, None, non-dict) raise ScoringFailureError, not AttributeError/TypeError."""
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": None, "p2": None, "p3": None}

        class FallbackClaimScorer(KGOnlyScorer):
            def score_candidates(self, user_id, candidates):
                scores = {item: float(self.context.popularity.get(item, 0)) for item in candidates}
                return scores, "fallback", "unusable_metadata_profile"

        # 1. user_cat_vec missing or not a dict
        scorer1 = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        scorer1.user_cat_vec = None
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer1, split="validation")

        scorer1_del = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        del scorer1_del.user_cat_vec
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer1_del, split="validation")

        # 2. user_brand_vec missing or not a dict
        scorer2 = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        scorer2.user_brand_vec = ["not_a_dict"]
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer2, split="validation")

        # 3. product_cat_vec missing or not a dict
        scorer3 = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        scorer3.product_cat_vec = None
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer3, split="validation")

        # 4. product_brand_vec missing or not a dict
        scorer4 = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        scorer4.product_brand_vec = "not_a_dict"
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer4, split="validation")

    def test_missing_user_and_candidate_profile_state_rejected_as_scoring_failure(self):
        """Missing user or candidate entries in outer containers raise ScoringFailureError, not KeyError."""
        cats = {"p1": [], "p2": [], "p3": []}
        brands = {"p1": None, "p2": None, "p3": None}

        class FallbackClaimScorer(KGOnlyScorer):
            def score_candidates(self, user_id, candidates):
                scores = {item: float(self.context.popularity.get(item, 0)) for item in candidates}
                return scores, "fallback", "unusable_metadata_profile"

        # Missing user entry in user_cat_vec
        scorer_u_cat = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        del scorer_u_cat.user_cat_vec["u1"]
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer_u_cat, split="validation")

        # Missing user entry in user_brand_vec
        scorer_u_brand = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        del scorer_u_brand.user_brand_vec["u1"]
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer_u_brand, split="validation")

        # Missing candidate entry in product_cat_vec
        scorer_p_cat = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        del scorer_p_cat.product_cat_vec["p2"]
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer_p_cat, split="validation")

        # Missing candidate entry in product_brand_vec
        scorer_p_brand = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        del scorer_p_brand.product_brand_vec["p2"]
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer_p_brand, split="validation")

        # Missing or non-numeric mixture weights
        scorer_mix = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        del scorer_mix.alpha
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer_mix, split="validation")

        scorer_mix_nan = FallbackClaimScorer(self.ctx, cats, brands, mixture=(1.0, 0.0))
        scorer_mix_nan.alpha = float("nan")
        with self.assertRaises(ScoringFailureError):
            self.evaluator.evaluate([("u1", "p2", 5.0)], scorer_mix_nan, split="validation")

    def test_core_route_validation_super_delegation(self):
        """Ordinary routes ('personalized', 'non_personalized') delegate strictly to super()."""
        # Non-dict scores raises ScoringFailureError
        with self.assertRaises(ScoringFailureError):
            self.evaluator._validate_provider_result(None, "u1", ["p2"], [1.0], "personalized", None)

        # Invalid route raises ScoringFailureError
        with self.assertRaises(ScoringFailureError):
            self.evaluator._validate_provider_result(None, "u1", ["p2"], {"p2": 1.0}, "invalid_route", None)

        # Non-null reason on personalized route raises ScoringFailureError
        with self.assertRaises(ScoringFailureError):
            self.evaluator._validate_provider_result(None, "u1", ["p2"], {"p2": 1.0}, "personalized", "bad_reason")

        # Non-null reason on non_personalized route raises ScoringFailureError
        with self.assertRaises(ScoringFailureError):
            self.evaluator._validate_provider_result(None, "u1", ["p2"], {"p2": 1.0}, "non_personalized", "bad_reason")

        # Valid personalized route passes
        scores, route, reason = self.evaluator._validate_provider_result(None, "u1", ["p2"], {"p2": 1.0}, "personalized", None)
        self.assertEqual(route, "personalized")
        self.assertIsNone(reason)

        # Valid non_personalized route passes
        scores, route, reason = self.evaluator._validate_provider_result(None, "u1", ["p2"], {"p2": 1.0}, "non_personalized", None)
        self.assertEqual(route, "non_personalized")
        self.assertIsNone(reason)


class HandCalculatedMixedRoutesAndBothFallbackReasonsTests(unittest.TestCase):
    """Hand-calculated fixtures asserting all registered metrics across mixed routes, both fallback reasons, conditions, and splits."""

    def setUp(self):
        self.train_interactions = [
            ("u_pers", "p1", 5.0),
            ("u_pers", "p2", 5.0),
            ("u_meta", "p_nocat", 5.0),
            ("u_empty", "p_low", 2.0),
            ("u4", "p_cand1", 5.0),
            ("u4", "p_cand2", 5.0),
            ("u5", "p_cand2", 5.0),
            ("u6", "p_cand2", 5.0),
            ("u4", "p_cand3", 5.0),
            ("u5", "p_cand3", 5.0),
        ]
        self.cats = {
            "p1": ["c1"],
            "p2": ["c1"],
            "p_nocat": [],
            "p_cand1": ["c1"],
            "p_cand2": ["c2"],
            "p_cand3": ["c1"],
            "p_low": ["c1"],
        }
        self.brands = {
            "p1": None,
            "p2": None,
            "p_nocat": None,
            "p_cand1": None,
            "p_cand2": None,
            "p_cand3": None,
            "p_low": None,
        }

    def test_hand_calculated_mixed_routes_and_both_fallback_reasons_rating_ge_4_validation(self):
        """Validation split under rating_ge_4 with literal hand-calculated per-user/macro/subgroup/by-reason metrics and coverage."""
        ctx = TrainingContext(self.train_interactions, condition="rating_ge_4")
        self.assertEqual(ctx.warm_items, {"p1", "p2", "p_nocat", "p_cand1", "p_cand2", "p_cand3"})
        self.assertEqual(len(ctx.warm_items), 6)

        scorer = KGOnlyScorer(ctx, self.cats, self.brands, mixture=(1.0, 0.0))
        evaluator = Evaluator(ctx)

        targets = [
            ("u_pers", "p_cand3", 5.0),  # Personalized: target rank 2
            ("u_meta", "p_cand2", 5.0),  # Fallback unusable_metadata: target rank 1
            ("u_empty", "p1", 5.0),      # Fallback empty_positive: target rank 3
        ]
        report = evaluator.evaluate(targets, scorer, split="validation")

        audit = report["audit"]
        self.assertEqual(audit["total_targets"], 3)
        self.assertEqual(audit["eligible_users"], 3)
        self.assertEqual(audit["excluded_targets"], 0)
        self.assertEqual(audit["warm_pool_size"], 6)

        # Shortages: u_pers has 4 candidates (< 5, < 10, < 20); u_meta has 5 (< 10, < 20); u_empty has 6 (< 10, < 20)
        self.assertEqual(report["candidate_shortages"][5], 1)
        self.assertEqual(report["candidate_shortages"][10], 3)
        self.assertEqual(report["candidate_shortages"][20], 3)

        # Per-user results
        u_map = {r["user_id"]: r for r in report["per_user_results"]}

        # u_pers: rank 2 of 4 items ["p_cand1", "p_cand3", "p_cand2", "p_nocat"]
        r_pers = u_map["u_pers"]
        self.assertEqual(r_pers["route"], "personalized")
        self.assertIsNone(r_pers["fallback_reason"])
        self.assertEqual(r_pers["candidate_count"], 4)
        self.assertEqual(r_pers["top_5"], ["p_cand1", "p_cand3", "p_cand2", "p_nocat"])
        ndcg_rank2 = 1.0 / math.log2(3.0)
        self.assertAlmostEqual(r_pers["metrics"]["NDCG@5"], ndcg_rank2)
        self.assertAlmostEqual(r_pers["metrics"]["NDCG@10"], ndcg_rank2)
        self.assertAlmostEqual(r_pers["metrics"]["NDCG@20"], ndcg_rank2)
        self.assertEqual(r_pers["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(r_pers["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(r_pers["metrics"]["HitRate@20"], 1.0)

        # u_meta: rank 1 of 5 items ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"]
        r_meta = u_map["u_meta"]
        self.assertEqual(r_meta["route"], "fallback")
        self.assertEqual(r_meta["fallback_reason"], "unusable_metadata_profile")
        self.assertEqual(r_meta["candidate_count"], 5)
        self.assertEqual(r_meta["top_5"], ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"])
        self.assertEqual(r_meta["metrics"]["NDCG@5"], 1.0)
        self.assertEqual(r_meta["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(r_meta["metrics"]["NDCG@20"], 1.0)
        self.assertEqual(r_meta["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(r_meta["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(r_meta["metrics"]["HitRate@20"], 1.0)

        # u_empty: rank 3 of 6 items ["p_cand2", "p_cand3", "p1", "p2", "p_cand1", "p_nocat"]
        r_empty = u_map["u_empty"]
        self.assertEqual(r_empty["route"], "fallback")
        self.assertEqual(r_empty["fallback_reason"], "empty_positive_history")
        self.assertEqual(r_empty["candidate_count"], 6)
        self.assertEqual(r_empty["top_5"], ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"])
        self.assertEqual(r_empty["metrics"]["NDCG@5"], 0.5)
        self.assertEqual(r_empty["metrics"]["NDCG@10"], 0.5)
        self.assertEqual(r_empty["metrics"]["NDCG@20"], 0.5)
        self.assertEqual(r_empty["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(r_empty["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(r_empty["metrics"]["HitRate@20"], 1.0)

        # Macro overall metrics
        macro_ndcg = (ndcg_rank2 + 1.0 + 0.5) / 3.0
        overall = report["overall_metrics"]
        self.assertAlmostEqual(overall["NDCG@5"], macro_ndcg)
        self.assertAlmostEqual(overall["NDCG@10"], macro_ndcg)
        self.assertAlmostEqual(overall["NDCG@20"], macro_ndcg)
        self.assertEqual(overall["HitRate@5"], 1.0)
        self.assertEqual(overall["HitRate@10"], 1.0)
        self.assertEqual(overall["HitRate@20"], 1.0)
        self.assertEqual(overall["CatalogCoverage@10"], 1.0)

        # Subgroups
        sg = report["subgroups"]
        self.assertEqual(sg["personalized"]["count"], 1)
        self.assertAlmostEqual(sg["personalized"]["metrics"]["NDCG@10"], ndcg_rank2)
        self.assertEqual(sg["personalized"]["metrics"]["HitRate@10"], 1.0)
        self.assertAlmostEqual(sg["personalized"]["metrics"]["CatalogCoverage@10"], 4.0 / 6.0)

        self.assertEqual(sg["non_personalized"]["count"], 0)
        for m in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(sg["non_personalized"]["metrics"][m])

        self.assertEqual(sg["fallback"]["count"], 2)
        self.assertAlmostEqual(sg["fallback"]["metrics"]["NDCG@10"], (1.0 + 0.5) / 2.0)
        self.assertEqual(sg["fallback"]["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(sg["fallback"]["metrics"]["CatalogCoverage@10"], 1.0)

        # Fallback by reason
        by_reason = sg["fallback"]["by_reason"]
        self.assertEqual(set(by_reason.keys()), {"empty_positive_history", "unusable_metadata_profile"})

        # unusable_metadata_profile: u_meta (rank 1)
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 1)
        self.assertEqual(by_reason["unusable_metadata_profile"]["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(by_reason["unusable_metadata_profile"]["metrics"]["HitRate@10"], 1.0)
        self.assertAlmostEqual(by_reason["unusable_metadata_profile"]["metrics"]["CatalogCoverage@10"], 5.0 / 6.0)

        # empty_positive_history: u_empty (rank 3)
        self.assertEqual(by_reason["empty_positive_history"]["count"], 1)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["NDCG@10"], 0.5)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["CatalogCoverage@10"], 1.0)

    def test_hand_calculated_mixed_routes_and_both_fallback_reasons_rating_ge_4_test_split(self):
        """Test split under rating_ge_4 with validation_history filtering and literal metrics."""
        ctx = TrainingContext(self.train_interactions, condition="rating_ge_4")
        scorer = KGOnlyScorer(ctx, self.cats, self.brands, mixture=(1.0, 0.0))
        evaluator = Evaluator(ctx)

        val_history = [
            ("u_pers", "p_cand1", 5.0),
            ("u_meta", "p1", 5.0),
            ("u_empty", "p_cand1", 1.0),
        ]
        targets = [
            ("u_pers", "p_cand3", 5.0),
            ("u_meta", "p_cand2", 5.0),
            ("u_empty", "p2", 5.0),
        ]
        report = evaluator.evaluate(targets, scorer, split="test", validation_history=val_history)

        self.assertEqual(report["audit"]["eligible_users"], 3)
        u_map = {r["user_id"]: r for r in report["per_user_results"]}

        # u_pers: rank 1 of 3
        self.assertEqual(u_map["u_pers"]["candidate_count"], 3)
        self.assertEqual(u_map["u_pers"]["top_5"][0], "p_cand3")
        self.assertEqual(u_map["u_pers"]["metrics"]["NDCG@10"], 1.0)

        # u_meta: rank 1 of 4
        self.assertEqual(u_map["u_meta"]["candidate_count"], 4)
        self.assertEqual(u_map["u_meta"]["top_5"][0], "p_cand2")
        self.assertEqual(u_map["u_meta"]["metrics"]["NDCG@10"], 1.0)

        # u_empty: rank 4 of 5 -> NDCG = 1.0 / log2(5)
        self.assertEqual(u_map["u_empty"]["candidate_count"], 5)
        self.assertEqual(u_map["u_empty"]["top_5"][3], "p2")
        ndcg_rank4 = 1.0 / math.log2(5.0)
        self.assertAlmostEqual(u_map["u_empty"]["metrics"]["NDCG@10"], ndcg_rank4)

        # Macro NDCG@10
        expected_macro_ndcg = (1.0 + 1.0 + ndcg_rank4) / 3.0
        self.assertAlmostEqual(report["overall_metrics"]["NDCG@10"], expected_macro_ndcg)
        self.assertEqual(report["overall_metrics"]["HitRate@10"], 1.0)
        self.assertEqual(report["overall_metrics"]["CatalogCoverage@10"], 1.0)

        # by_reason check
        by_reason = report["subgroups"]["fallback"]["by_reason"]
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 1)
        self.assertEqual(by_reason["unusable_metadata_profile"]["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(by_reason["empty_positive_history"]["count"], 1)
        self.assertAlmostEqual(by_reason["empty_positive_history"]["metrics"]["NDCG@10"], ndcg_rank4)

    def test_hand_calculated_all_ratings_condition_with_empty_reason_schema(self):
        """all_ratings condition where empty_positive_history has count 0 and all-None metrics in by_reason."""
        train = [
            ("u_pers", "p1", 5.0),
            ("u_pers", "p2", 5.0),
            ("u_meta", "p_nocat", 5.0),
            ("u4", "p_cand1", 5.0),
            ("u4", "p_cand2", 5.0),
            ("u5", "p_cand2", 5.0),
        ]
        ctx = TrainingContext(train, condition="all_ratings")
        cats = {"p1": ["c1"], "p2": ["c1"], "p_nocat": [], "p_cand1": ["c1"], "p_cand2": ["c2"]}
        brands = {"p1": None, "p2": None, "p_nocat": None, "p_cand1": None, "p_cand2": None}
        scorer = KGOnlyScorer(ctx, cats, brands, mixture=(1.0, 0.0))
        evaluator = Evaluator(ctx)

        targets = [
            ("u_pers", "p_cand1", 5.0),
            ("u_meta", "p_cand2", 5.0),
        ]
        report = evaluator.evaluate(targets, scorer, split="validation")

        self.assertEqual(report["audit"]["eligible_users"], 2)
        by_reason = report["subgroups"]["fallback"]["by_reason"]
        self.assertEqual(set(by_reason.keys()), {"empty_positive_history", "unusable_metadata_profile"})

        # empty_positive_history must be count 0 with all-None metrics
        self.assertEqual(by_reason["empty_positive_history"]["count"], 0)
        for m in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(by_reason["empty_positive_history"]["metrics"][m])

        # unusable_metadata_profile must be count 1 with valid metrics
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 1)
        self.assertEqual(by_reason["unusable_metadata_profile"]["metrics"]["HitRate@10"], 1.0)
        self.assertIsNotNone(by_reason["unusable_metadata_profile"]["metrics"]["NDCG@10"])

    def test_hand_calculated_popularity_and_fallback_routes(self):
        """Mixed non_personalized route and empty_positive_history fallback route with exact subgroup metrics."""
        train = [
            ("u_norm", "p1", 5.0),
            ("u_empty", "p_low", 2.0),
            ("u3", "p2", 5.0),
            ("u3", "p3", 5.0),
        ]
        ctx = TrainingContext(train, condition="rating_ge_4")
        evaluator = Evaluator(ctx)
        scorer = PopularityScorer(ctx)

        targets = [
            ("u_norm", "p2", 5.0),   # PopularityScorer -> route non_personalized
            ("u_empty", "p3", 5.0),  # Empty positive history -> route fallback
        ]
        report = evaluator.evaluate(targets, scorer, split="validation")

        self.assertEqual(report["audit"]["eligible_users"], 2)
        sg = report["subgroups"]
        self.assertEqual(sg["non_personalized"]["count"], 1)
        self.assertIsNotNone(sg["non_personalized"]["metrics"]["NDCG@10"])
        self.assertEqual(sg["personalized"]["count"], 0)
        for m in ("NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20", "CatalogCoverage@10"):
            self.assertIsNone(sg["personalized"]["metrics"][m])
        self.assertEqual(sg["fallback"]["count"], 1)
        self.assertIsNotNone(sg["fallback"]["metrics"]["NDCG@10"])
        self.assertEqual(sg["fallback"]["by_reason"]["empty_positive_history"]["count"], 1)
        self.assertEqual(sg["fallback"]["by_reason"]["unusable_metadata_profile"]["count"], 0)

    def test_compact_table_driven_four_condition_split_combinations_literal_metrics(self):
        """Compact table-driven oracle asserting all registered metrics across both conditions and splits with literal expected values."""
        cases = [
            {
                "label": "rating_ge_4_validation",
                "condition": "rating_ge_4",
                "split": "validation",
                "targets": [
                    ("u_pers", "p_cand3", 5.0),
                    ("u_meta", "p_cand2", 5.0),
                    ("u_empty", "p1", 5.0),
                ],
                "val_history": None,
                "warm_pool_size": 6,
                "total_targets": 3,
                "eligible_users": 3,
                "excluded_targets": 0,
                "shortages": {5: 1, 10: 3, 20: 3},
                "users": {
                    "u_pers": {
                        "route": "personalized",
                        "fallback_reason": None,
                        "candidate_count": 4,
                        "top_5": ["p_cand1", "p_cand3", "p_cand2", "p_nocat"],
                        "top_10": ["p_cand1", "p_cand3", "p_cand2", "p_nocat"],
                        "top_20": ["p_cand1", "p_cand3", "p_cand2", "p_nocat"],
                        "metrics": {
                            "NDCG@5": 1.0 / math.log2(3.0),
                            "NDCG@10": 1.0 / math.log2(3.0),
                            "NDCG@20": 1.0 / math.log2(3.0),
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                    "u_meta": {
                        "route": "fallback",
                        "fallback_reason": "unusable_metadata_profile",
                        "candidate_count": 5,
                        "top_5": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"],
                        "top_10": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"],
                        "top_20": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"],
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                    "u_empty": {
                        "route": "fallback",
                        "fallback_reason": "empty_positive_history",
                        "candidate_count": 6,
                        "top_5": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"],
                        "top_10": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1", "p_nocat"],
                        "top_20": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1", "p_nocat"],
                        "metrics": {
                            "NDCG@5": 0.5,
                            "NDCG@10": 0.5,
                            "NDCG@20": 0.5,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                },
                "overall": {
                    "NDCG@5": (1.0 / math.log2(3.0) + 1.0 + 0.5) / 3.0,
                    "NDCG@10": (1.0 / math.log2(3.0) + 1.0 + 0.5) / 3.0,
                    "NDCG@20": (1.0 / math.log2(3.0) + 1.0 + 0.5) / 3.0,
                    "HitRate@5": 1.0,
                    "HitRate@10": 1.0,
                    "HitRate@20": 1.0,
                    "CatalogCoverage@10": 6.0 / 6.0,
                },
                "subgroups": {
                    "personalized": {
                        "count": 1,
                        "metrics": {
                            "NDCG@5": 1.0 / math.log2(3.0),
                            "NDCG@10": 1.0 / math.log2(3.0),
                            "NDCG@20": 1.0 / math.log2(3.0),
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 4.0 / 6.0,
                        },
                    },
                    "non_personalized": {
                        "count": 0,
                        "metrics": {
                            "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                            "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                            "CatalogCoverage@10": None,
                        },
                    },
                    "fallback": {
                        "count": 2,
                        "metrics": {
                            "NDCG@5": (1.0 + 0.5) / 2.0,
                            "NDCG@10": 0.75,
                            "NDCG@20": 0.75,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 6.0 / 6.0,
                        },
                        "by_reason": {
                            "unusable_metadata_profile": {
                                "count": 1,
                                "metrics": {
                                    "NDCG@5": 1.0,
                                    "NDCG@10": 1.0,
                                    "NDCG@20": 1.0,
                                    "HitRate@5": 1.0,
                                    "HitRate@10": 1.0,
                                    "HitRate@20": 1.0,
                                    "CatalogCoverage@10": 5.0 / 6.0,
                                },
                            },
                            "empty_positive_history": {
                                "count": 1,
                                "metrics": {
                                    "NDCG@5": 0.5,
                                    "NDCG@10": 0.5,
                                    "NDCG@20": 0.5,
                                    "HitRate@5": 1.0,
                                    "HitRate@10": 1.0,
                                    "HitRate@20": 1.0,
                                    "CatalogCoverage@10": 6.0 / 6.0,
                                },
                            },
                        },
                    },
                },
            },
            {
                "label": "rating_ge_4_test",
                "condition": "rating_ge_4",
                "split": "test",
                "targets": [
                    ("u_pers", "p_cand3", 5.0),
                    ("u_meta", "p_cand2", 5.0),
                    ("u_empty", "p2", 5.0),
                ],
                "val_history": [
                    ("u_pers", "p_cand1", 5.0),
                    ("u_meta", "p1", 5.0),
                    ("u_empty", "p_cand1", 1.0),
                ],
                "warm_pool_size": 6,
                "total_targets": 3,
                "eligible_users": 3,
                "excluded_targets": 0,
                "shortages": {5: 2, 10: 3, 20: 3},
                "users": {
                    "u_pers": {
                        "route": "personalized",
                        "fallback_reason": None,
                        "candidate_count": 3,
                        "top_5": ["p_cand3", "p_cand2", "p_nocat"],
                        "top_10": ["p_cand3", "p_cand2", "p_nocat"],
                        "top_20": ["p_cand3", "p_cand2", "p_nocat"],
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                    "u_meta": {
                        "route": "fallback",
                        "fallback_reason": "unusable_metadata_profile",
                        "candidate_count": 4,
                        "top_5": ["p_cand2", "p_cand3", "p2", "p_cand1"],
                        "top_10": ["p_cand2", "p_cand3", "p2", "p_cand1"],
                        "top_20": ["p_cand2", "p_cand3", "p2", "p_cand1"],
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                    "u_empty": {
                        "route": "fallback",
                        "fallback_reason": "empty_positive_history",
                        "candidate_count": 5,
                        "top_5": ["p_cand2", "p_cand3", "p1", "p2", "p_nocat"],
                        "top_10": ["p_cand2", "p_cand3", "p1", "p2", "p_nocat"],
                        "top_20": ["p_cand2", "p_cand3", "p1", "p2", "p_nocat"],
                        "metrics": {
                            "NDCG@5": 1.0 / math.log2(5.0),
                            "NDCG@10": 1.0 / math.log2(5.0),
                            "NDCG@20": 1.0 / math.log2(5.0),
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                },
                "overall": {
                    "NDCG@5": (1.0 + 1.0 + 1.0 / math.log2(5.0)) / 3.0,
                    "NDCG@10": (1.0 + 1.0 + 1.0 / math.log2(5.0)) / 3.0,
                    "NDCG@20": (1.0 + 1.0 + 1.0 / math.log2(5.0)) / 3.0,
                    "HitRate@5": 1.0,
                    "HitRate@10": 1.0,
                    "HitRate@20": 1.0,
                    "CatalogCoverage@10": 6.0 / 6.0,
                },
                "subgroups": {
                    "personalized": {
                        "count": 1,
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 3.0 / 6.0,
                        },
                    },
                    "non_personalized": {
                        "count": 0,
                        "metrics": {
                            "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                            "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                            "CatalogCoverage@10": None,
                        },
                    },
                    "fallback": {
                        "count": 2,
                        "metrics": {
                            "NDCG@5": (1.0 + 1.0 / math.log2(5.0)) / 2.0,
                            "NDCG@10": (1.0 + 1.0 / math.log2(5.0)) / 2.0,
                            "NDCG@20": (1.0 + 1.0 / math.log2(5.0)) / 2.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 6.0 / 6.0,
                        },
                        "by_reason": {
                            "unusable_metadata_profile": {
                                "count": 1,
                                "metrics": {
                                    "NDCG@5": 1.0,
                                    "NDCG@10": 1.0,
                                    "NDCG@20": 1.0,
                                    "HitRate@5": 1.0,
                                    "HitRate@10": 1.0,
                                    "HitRate@20": 1.0,
                                    "CatalogCoverage@10": 4.0 / 6.0,
                                },
                            },
                            "empty_positive_history": {
                                "count": 1,
                                "metrics": {
                                    "NDCG@5": 1.0 / math.log2(5.0),
                                    "NDCG@10": 1.0 / math.log2(5.0),
                                    "NDCG@20": 1.0 / math.log2(5.0),
                                    "HitRate@5": 1.0,
                                    "HitRate@10": 1.0,
                                    "HitRate@20": 1.0,
                                    "CatalogCoverage@10": 5.0 / 6.0,
                                },
                            },
                        },
                    },
                },
            },
            {
                "label": "all_ratings_validation",
                "condition": "all_ratings",
                "split": "validation",
                "targets": [
                    ("u_pers", "p_cand1", 5.0),
                    ("u_meta", "p_cand2", 5.0),
                ],
                "val_history": None,
                "warm_pool_size": 7,
                "total_targets": 2,
                "eligible_users": 2,
                "excluded_targets": 0,
                "shortages": {5: 0, 10: 2, 20: 2},
                "users": {
                    "u_pers": {
                        "route": "personalized",
                        "fallback_reason": None,
                        "candidate_count": 5,
                        "top_5": ["p_cand1", "p_cand3", "p_low", "p_cand2", "p_nocat"],
                        "top_10": ["p_cand1", "p_cand3", "p_low", "p_cand2", "p_nocat"],
                        "top_20": ["p_cand1", "p_cand3", "p_low", "p_cand2", "p_nocat"],
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                    "u_meta": {
                        "route": "fallback",
                        "fallback_reason": "unusable_metadata_profile",
                        "candidate_count": 6,
                        "top_5": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1"],
                        "top_10": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1", "p_low"],
                        "top_20": ["p_cand2", "p_cand3", "p1", "p2", "p_cand1", "p_low"],
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                },
                "overall": {
                    "NDCG@5": 1.0,
                    "NDCG@10": 1.0,
                    "NDCG@20": 1.0,
                    "HitRate@5": 1.0,
                    "HitRate@10": 1.0,
                    "HitRate@20": 1.0,
                    "CatalogCoverage@10": 7.0 / 7.0,
                },
                "subgroups": {
                    "personalized": {
                        "count": 1,
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 5.0 / 7.0,
                        },
                    },
                    "non_personalized": {
                        "count": 0,
                        "metrics": {
                            "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                            "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                            "CatalogCoverage@10": None,
                        },
                    },
                    "fallback": {
                        "count": 1,
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 6.0 / 7.0,
                        },
                        "by_reason": {
                            "unusable_metadata_profile": {
                                "count": 1,
                                "metrics": {
                                    "NDCG@5": 1.0,
                                    "NDCG@10": 1.0,
                                    "NDCG@20": 1.0,
                                    "HitRate@5": 1.0,
                                    "HitRate@10": 1.0,
                                    "HitRate@20": 1.0,
                                    "CatalogCoverage@10": 6.0 / 7.0,
                                },
                            },
                            "empty_positive_history": {
                                "count": 0,
                                "metrics": {
                                    "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                                    "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                                    "CatalogCoverage@10": None,
                                },
                            },
                        },
                    },
                },
            },
            {
                "label": "all_ratings_test",
                "condition": "all_ratings",
                "split": "test",
                "targets": [
                    ("u_pers", "p_cand3", 5.0),
                    ("u_meta", "p_cand2", 5.0),
                ],
                "val_history": [
                    ("u_pers", "p_cand1", 5.0),
                    ("u_meta", "p1", 5.0),
                ],
                "warm_pool_size": 7,
                "total_targets": 2,
                "eligible_users": 2,
                "excluded_targets": 0,
                "shortages": {5: 1, 10: 2, 20: 2},
                "users": {
                    "u_pers": {
                        "route": "personalized",
                        "fallback_reason": None,
                        "candidate_count": 4,
                        "top_5": ["p_cand3", "p_low", "p_cand2", "p_nocat"],
                        "top_10": ["p_cand3", "p_low", "p_cand2", "p_nocat"],
                        "top_20": ["p_cand3", "p_low", "p_cand2", "p_nocat"],
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                    "u_meta": {
                        "route": "fallback",
                        "fallback_reason": "unusable_metadata_profile",
                        "candidate_count": 5,
                        "top_5": ["p_cand2", "p_cand3", "p2", "p_cand1", "p_low"],
                        "top_10": ["p_cand2", "p_cand3", "p2", "p_cand1", "p_low"],
                        "top_20": ["p_cand2", "p_cand3", "p2", "p_cand1", "p_low"],
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                        },
                    },
                },
                "overall": {
                    "NDCG@5": 1.0,
                    "NDCG@10": 1.0,
                    "NDCG@20": 1.0,
                    "HitRate@5": 1.0,
                    "HitRate@10": 1.0,
                    "HitRate@20": 1.0,
                    "CatalogCoverage@10": 6.0 / 7.0,
                },
                "subgroups": {
                    "personalized": {
                        "count": 1,
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 4.0 / 7.0,
                        },
                    },
                    "non_personalized": {
                        "count": 0,
                        "metrics": {
                            "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                            "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                            "CatalogCoverage@10": None,
                        },
                    },
                    "fallback": {
                        "count": 1,
                        "metrics": {
                            "NDCG@5": 1.0,
                            "NDCG@10": 1.0,
                            "NDCG@20": 1.0,
                            "HitRate@5": 1.0,
                            "HitRate@10": 1.0,
                            "HitRate@20": 1.0,
                            "CatalogCoverage@10": 5.0 / 7.0,
                        },
                        "by_reason": {
                            "unusable_metadata_profile": {
                                "count": 1,
                                "metrics": {
                                    "NDCG@5": 1.0,
                                    "NDCG@10": 1.0,
                                    "NDCG@20": 1.0,
                                    "HitRate@5": 1.0,
                                    "HitRate@10": 1.0,
                                    "HitRate@20": 1.0,
                                    "CatalogCoverage@10": 5.0 / 7.0,
                                },
                            },
                            "empty_positive_history": {
                                "count": 0,
                                "metrics": {
                                    "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                                    "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                                    "CatalogCoverage@10": None,
                                },
                            },
                        },
                    },
                },
            },
        ]

        metric_names_k = ["NDCG@5", "NDCG@10", "NDCG@20", "HitRate@5", "HitRate@10", "HitRate@20"]
        metric_names_all = metric_names_k + ["CatalogCoverage@10"]

        for c in cases:
            with self.subTest(label=c["label"]):
                ctx = TrainingContext(self.train_interactions, condition=c["condition"])
                scorer = KGOnlyScorer(ctx, self.cats, self.brands, mixture=(1.0, 0.0))
                evaluator = Evaluator(ctx)

                report = evaluator.evaluate(
                    c["targets"],
                    scorer,
                    split=c["split"],
                    validation_history=c["val_history"],
                )

                # 1. Audit assertions
                audit = report["audit"]
                self.assertEqual(audit["total_targets"], c["total_targets"])
                self.assertEqual(audit["eligible_users"], c["eligible_users"])
                self.assertEqual(audit["excluded_targets"], c["excluded_targets"])
                self.assertEqual(audit["warm_pool_size"], c["warm_pool_size"])

                # 2. Shortage assertions
                for k, expected_shortage in c["shortages"].items():
                    self.assertEqual(report["candidate_shortages"][k], expected_shortage)

                # 3. Per-user assertions
                u_map = {r["user_id"]: r for r in report["per_user_results"]}
                self.assertEqual(set(u_map.keys()), set(c["users"].keys()))
                for u, u_exp in c["users"].items():
                    u_act = u_map[u]
                    self.assertEqual(u_act["route"], u_exp["route"])
                    self.assertEqual(u_act["fallback_reason"], u_exp["fallback_reason"])
                    self.assertEqual(u_act["candidate_count"], u_exp["candidate_count"])
                    self.assertEqual(u_act["top_5"], u_exp["top_5"])
                    self.assertEqual(u_act["top_10"], u_exp["top_10"])
                    self.assertEqual(u_act["top_20"], u_exp["top_20"])
                    for m in metric_names_k:
                        self.assertAlmostEqual(u_act["metrics"][m], u_exp["metrics"][m], places=7)

                # 4. Overall metric assertions
                for m in metric_names_all:
                    self.assertAlmostEqual(report["overall_metrics"][m], c["overall"][m], places=7)

                # 5. Subgroup assertions
                sg_act = report["subgroups"]
                for route_name, sg_exp in c["subgroups"].items():
                    self.assertEqual(sg_act[route_name]["count"], sg_exp["count"])
                    for m in metric_names_all:
                        exp_val = sg_exp["metrics"][m]
                        act_val = sg_act[route_name]["metrics"][m]
                        if exp_val is None:
                            self.assertIsNone(act_val)
                        else:
                            self.assertIsNotNone(act_val)
                            self.assertAlmostEqual(act_val, exp_val, places=7)

                # 6. Fallback by_reason assertions
                by_reason_act = sg_act["fallback"]["by_reason"]
                by_reason_exp = c["subgroups"]["fallback"]["by_reason"]
                self.assertEqual(set(by_reason_act.keys()), {"empty_positive_history", "unusable_metadata_profile"})
                for reason_name, r_exp in by_reason_exp.items():
                    self.assertEqual(by_reason_act[reason_name]["count"], r_exp["count"])
                    for m in metric_names_all:
                        exp_val = r_exp["metrics"][m]
                        act_val = by_reason_act[reason_name]["metrics"][m]
                        if exp_val is None:
                            self.assertIsNone(act_val)
                        else:
                            self.assertIsNotNone(act_val)
                            self.assertAlmostEqual(act_val, exp_val, places=7)

    def test_honest_kg_only_scorer_subtype_dispatch_all_three_routes_in_one_call(self):
        """Honest KGOnlyScorer subtype dispatch fixture exercising all THREE routes and both fallback reasons in one call."""
        train = [
            ("u_pop", "p_pop_train", 5.0),
            ("u_pers", "p1", 5.0),
            ("u_meta", "p_nocat", 5.0),
            ("u_empty", "p_low", 2.0),
            ("u4", "p_cand1", 5.0),
            ("u4", "p_cand2", 5.0),
            ("u5", "p_cand2", 5.0),
            ("u6", "p_cand2", 5.0),
            ("u4", "p_cand3", 5.0),
            ("u5", "p_cand3", 5.0),
        ]
        cats = {
            "p_pop_train": ["c1"],
            "p1": ["c1"],
            "p_nocat": [],
            "p_cand1": ["c1"],
            "p_cand2": ["c2"],
            "p_cand3": ["c1"],
            "p_low": ["c1"],
        }
        brands = {k: None for k in cats}

        ctx = TrainingContext(train, condition="rating_ge_4")
        self.assertEqual(ctx.warm_items, {"p_pop_train", "p1", "p_nocat", "p_cand1", "p_cand2", "p_cand3"})
        self.assertEqual(len(ctx.warm_items), 6)

        class DispatchingKGOnlyScorer(KGOnlyScorer):
            """Subtype of KGOnlyScorer dispatching u_pop to PopularityScorer and others to super."""
            def __init__(self, context, item_categories, item_brands, mixture=(1.0, 0.0)):
                super().__init__(context, item_categories, item_brands, mixture=mixture)
                self.popularity_scorer = PopularityScorer(context)
                self.scored_users = []

            def score_candidates(self, user_id, candidates):
                self.scored_users.append(user_id)
                if user_id == "u_pop":
                    # Honest dispatch to PopularityScorer returning non_personalized route
                    return self.popularity_scorer.score_candidates(user_id, candidates)
                return super().score_candidates(user_id, candidates)

        scorer = DispatchingKGOnlyScorer(ctx, cats, brands, mixture=(1.0, 0.0))
        evaluator = Evaluator(ctx)

        targets = [
            ("u_pop", "p_cand2", 5.0),   # Route non_personalized via PopularityScorer: rank 1 of 5
            ("u_pers", "p_cand1", 5.0),  # Route personalized via KGOnlyScorer: rank 1 of 5
            ("u_meta", "p_cand3", 5.0),  # Route fallback (unusable_metadata_profile) via KGOnlyScorer: rank 2 of 5
            ("u_empty", "p1", 5.0),      # Route fallback (empty_positive_history) owned by Evaluator: rank 3 of 6
        ]

        report = evaluator.evaluate(targets, scorer, split="validation")

        # Provider bypass assertion: u_empty had empty positive history, so Evaluator owned fallback
        # and bypassed provider score_candidates execution; u_meta, u_pers, u_pop called scorer in deterministic user_id order
        self.assertEqual(scorer.scored_users, ["u_meta", "u_pers", "u_pop"])

        # 1. Audit
        audit = report["audit"]
        self.assertEqual(audit["total_targets"], 4)
        self.assertEqual(audit["eligible_users"], 4)
        self.assertEqual(audit["excluded_targets"], 0)
        self.assertEqual(audit["warm_pool_size"], 6)

        # 2. Per-user results
        u_map = {r["user_id"]: r for r in report["per_user_results"]}

        # u_pop: non_personalized, target p_cand2 rank 1
        r_pop = u_map["u_pop"]
        self.assertEqual(r_pop["route"], "non_personalized")
        self.assertIsNone(r_pop["fallback_reason"])
        self.assertEqual(r_pop["candidate_count"], 5)
        self.assertEqual(r_pop["top_5"], ["p_cand2", "p_cand3", "p1", "p_cand1", "p_nocat"])
        self.assertEqual(r_pop["metrics"]["NDCG@5"], 1.0)
        self.assertEqual(r_pop["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(r_pop["metrics"]["NDCG@20"], 1.0)
        self.assertEqual(r_pop["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(r_pop["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(r_pop["metrics"]["HitRate@20"], 1.0)

        # u_pers: personalized, target p_cand1 rank 1
        r_pers = u_map["u_pers"]
        self.assertEqual(r_pers["route"], "personalized")
        self.assertIsNone(r_pers["fallback_reason"])
        self.assertEqual(r_pers["candidate_count"], 5)
        self.assertEqual(r_pers["top_5"], ["p_cand1", "p_cand3", "p_pop_train", "p_cand2", "p_nocat"])
        self.assertEqual(r_pers["metrics"]["NDCG@5"], 1.0)
        self.assertEqual(r_pers["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(r_pers["metrics"]["NDCG@20"], 1.0)
        self.assertEqual(r_pers["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(r_pers["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(r_pers["metrics"]["HitRate@20"], 1.0)

        # u_meta: fallback unusable_metadata_profile, target p_cand3 rank 2
        r_meta = u_map["u_meta"]
        self.assertEqual(r_meta["route"], "fallback")
        self.assertEqual(r_meta["fallback_reason"], "unusable_metadata_profile")
        self.assertEqual(r_meta["candidate_count"], 5)
        self.assertEqual(r_meta["top_5"], ["p_cand2", "p_cand3", "p1", "p_cand1", "p_pop_train"])
        ndcg_meta = 1.0 / math.log2(3.0)
        self.assertAlmostEqual(r_meta["metrics"]["NDCG@5"], ndcg_meta)
        self.assertAlmostEqual(r_meta["metrics"]["NDCG@10"], ndcg_meta)
        self.assertAlmostEqual(r_meta["metrics"]["NDCG@20"], ndcg_meta)
        self.assertEqual(r_meta["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(r_meta["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(r_meta["metrics"]["HitRate@20"], 1.0)

        # u_empty: fallback empty_positive_history, target p1 rank 3
        r_empty = u_map["u_empty"]
        self.assertEqual(r_empty["route"], "fallback")
        self.assertEqual(r_empty["fallback_reason"], "empty_positive_history")
        self.assertEqual(r_empty["candidate_count"], 6)
        self.assertEqual(r_empty["top_5"], ["p_cand2", "p_cand3", "p1", "p_cand1", "p_nocat"])
        self.assertEqual(r_empty["metrics"]["NDCG@5"], 0.5)
        self.assertEqual(r_empty["metrics"]["NDCG@10"], 0.5)
        self.assertEqual(r_empty["metrics"]["NDCG@20"], 0.5)
        self.assertEqual(r_empty["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(r_empty["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(r_empty["metrics"]["HitRate@20"], 1.0)

        # 3. Overall macro metrics
        macro_ndcg = (1.0 + 1.0 + ndcg_meta + 0.5) / 4.0
        self.assertAlmostEqual(report["overall_metrics"]["NDCG@5"], macro_ndcg)
        self.assertAlmostEqual(report["overall_metrics"]["NDCG@10"], macro_ndcg)
        self.assertAlmostEqual(report["overall_metrics"]["NDCG@20"], macro_ndcg)
        self.assertEqual(report["overall_metrics"]["HitRate@5"], 1.0)
        self.assertEqual(report["overall_metrics"]["HitRate@10"], 1.0)
        self.assertEqual(report["overall_metrics"]["HitRate@20"], 1.0)
        self.assertEqual(report["overall_metrics"]["CatalogCoverage@10"], 6.0 / 6.0)

        # 4. Subgroup metrics
        sg = report["subgroups"]
        # Personalized (u_pers)
        self.assertEqual(sg["personalized"]["count"], 1)
        self.assertEqual(sg["personalized"]["metrics"]["NDCG@5"], 1.0)
        self.assertEqual(sg["personalized"]["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(sg["personalized"]["metrics"]["NDCG@20"], 1.0)
        self.assertEqual(sg["personalized"]["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(sg["personalized"]["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(sg["personalized"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(sg["personalized"]["metrics"]["CatalogCoverage@10"], 5.0 / 6.0)

        # Non-personalized (u_pop)
        self.assertEqual(sg["non_personalized"]["count"], 1)
        self.assertEqual(sg["non_personalized"]["metrics"]["NDCG@5"], 1.0)
        self.assertEqual(sg["non_personalized"]["metrics"]["NDCG@10"], 1.0)
        self.assertEqual(sg["non_personalized"]["metrics"]["NDCG@20"], 1.0)
        self.assertEqual(sg["non_personalized"]["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(sg["non_personalized"]["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(sg["non_personalized"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(sg["non_personalized"]["metrics"]["CatalogCoverage@10"], 5.0 / 6.0)

        # Fallback (u_meta, u_empty)
        self.assertEqual(sg["fallback"]["count"], 2)
        fallback_ndcg = (ndcg_meta + 0.5) / 2.0
        self.assertAlmostEqual(sg["fallback"]["metrics"]["NDCG@5"], fallback_ndcg)
        self.assertAlmostEqual(sg["fallback"]["metrics"]["NDCG@10"], fallback_ndcg)
        self.assertAlmostEqual(sg["fallback"]["metrics"]["NDCG@20"], fallback_ndcg)
        self.assertEqual(sg["fallback"]["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(sg["fallback"]["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(sg["fallback"]["metrics"]["HitRate@20"], 1.0)
        self.assertEqual(sg["fallback"]["metrics"]["CatalogCoverage@10"], 6.0 / 6.0)

        # by_reason
        by_reason = sg["fallback"]["by_reason"]
        # unusable_metadata_profile (u_meta)
        self.assertEqual(by_reason["unusable_metadata_profile"]["count"], 1)
        self.assertAlmostEqual(by_reason["unusable_metadata_profile"]["metrics"]["NDCG@5"], ndcg_meta)
        self.assertAlmostEqual(by_reason["unusable_metadata_profile"]["metrics"]["NDCG@10"], ndcg_meta)
        self.assertAlmostEqual(by_reason["unusable_metadata_profile"]["metrics"]["NDCG@20"], ndcg_meta)
        self.assertEqual(by_reason["unusable_metadata_profile"]["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(by_reason["unusable_metadata_profile"]["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(by_reason["unusable_metadata_profile"]["metrics"]["HitRate@20"], 1.0)
        self.assertAlmostEqual(by_reason["unusable_metadata_profile"]["metrics"]["CatalogCoverage@10"], 5.0 / 6.0)

        # empty_positive_history (u_empty)
        self.assertEqual(by_reason["empty_positive_history"]["count"], 1)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["NDCG@5"], 0.5)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["NDCG@10"], 0.5)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["NDCG@20"], 0.5)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["HitRate@5"], 1.0)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["HitRate@10"], 1.0)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["HitRate@20"], 1.0)
        self.assertEqual(by_reason["empty_positive_history"]["metrics"]["CatalogCoverage@10"], 6.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
