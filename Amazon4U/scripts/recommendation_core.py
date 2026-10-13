"""Amazon4U milestone I1: recommendation core reference utilities and evaluation harness.

Python standard library only.
Provides:
- TrainingContext: validated training-only context, condition-specific warmth and popularity,
  distinct positive vs all-rated history, duplicate/invalid input failures, held-out-independent.
- PopularityScorer: deterministic condition-positive training popularity scoring with
  explicit non-personalized baseline route.
- KGOnlyScorer: KG-v1 attribute mapping, smoothed IDF over condition-warm items, channel L2
  normalization, normalized sum user profiles, mixture weights, whole-user fallback.
- Evaluator: target eligibility and exclusion accounting (unknown, cold, irrelevant, overlap),
  history filtering (validation vs test), deterministic ranking (score desc, parent_asin asc),
  shortage reporting, non-finite score rejection, no padding, registered metrics
  (NDCG@K, HitRate@K for K in (5, 10, 20), CatalogCoverage@10), route and fallback subgroup reporting.
- NegativeSampler: on-demand sorted condition-warm pool minus all user training-rated items,
  uniform 1-negative sampling with supplied RNG, skip accounting, empty-trainable dataset failure.
- bpr_loss: numerically stable mean softplus(s_neg - s_pos) with finite input/output validation.
- endpoint_regularization: 1e-5 * sum(squared L2 base row norms) / (unique typed endpoint rows).
"""

from collections import defaultdict
import math
import random
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Canonicalize module in sys.modules so scripts.recommendation_core and recommendation_core
# share the exact same module object and class identities.
if __name__ == "scripts.recommendation_core":
    sys.modules.setdefault("recommendation_core", sys.modules[__name__])
elif __name__ == "recommendation_core":
    sys.modules.setdefault("scripts.recommendation_core", sys.modules[__name__])
    if "scripts" in sys.modules:
        setattr(sys.modules["scripts"], "recommendation_core", sys.modules[__name__])

if __package__:
    from .recommendation_evaluation import (
        Evaluator as BaseEvaluator,
        ScoringFailureError,
        TrainingContext,
    )
else:
    from recommendation_evaluation import (
        Evaluator as BaseEvaluator,
        ScoringFailureError,
        TrainingContext,
    )


class PopularityScorer:
    """Deterministic condition-positive popularity scorer with non-personalized baseline route."""

    def __init__(self, context: TrainingContext) -> None:
        self.context = context

    def score_candidates(self, user_id: str, candidate_items: Sequence[str]) -> Tuple[Dict[str, float], str, Optional[str]]:
        scores = {item: float(self.context.popularity.get(item, 0)) for item in candidate_items}
        return scores, "non_personalized", None


class KGOnlyScorer:
    """KG-v1 metadata scorer using condition-warm IDF, normalized channels, and whole-user fallback."""

    GRID: Tuple[Tuple[float, float], ...] = (
        (1.0, 0.0),
        (0.75, 0.25),
        (0.5, 0.5),
    )

    def __init__(
        self,
        context: TrainingContext,
        item_categories: Dict[str, Sequence[str]],
        item_brands: Dict[str, Optional[str]],
        mixture: Tuple[float, float] = (1.0, 0.0),
    ) -> None:
        if mixture not in self.GRID:
            raise ValueError(f"Invalid mixture {mixture}. Must be one of {self.GRID}.")
        if not isinstance(item_categories, dict):
            raise ValueError("item_categories must be a dictionary.")
        if not isinstance(item_brands, dict):
            raise ValueError("item_brands must be a dictionary.")

        self.context = context
        self.mixture = mixture
        self.alpha, self.beta = mixture

        # Validate identity mappings over warm items
        for item in context.sorted_warm_items:
            cats = item_categories.get(item)
            if cats is not None:
                if isinstance(cats, (str, bytes)):
                    raise ValueError(f"Category for item {item} must be a collection of identities, not a string.")
                if not isinstance(cats, (list, tuple, set, frozenset)):
                    raise ValueError(f"Category for item {item} must be an identity collection.")
                for cat in cats:
                    if not isinstance(cat, str) or not cat.strip():
                        raise ValueError(f"Invalid blank/non-string category identity for item {item}: {cat!r}")

            brand = item_brands.get(item)
            if brand is not None:
                if not isinstance(brand, str) or not brand.strip():
                    raise ValueError(f"Invalid blank/non-string brand identity for item {item}: {brand!r}")

        n_warm = len(context.warm_items)

        # Compute document frequencies in deterministic sorted order
        df_cat: Dict[str, int] = defaultdict(int)
        for item in context.sorted_warm_items:
            cats = item_categories.get(item)
            if cats:
                for cat in sorted(set(cats)):
                    df_cat[cat] += 1

        self.idf_category: Dict[str, float] = {}
        for cat in sorted(df_cat.keys()):
            self.idf_category[cat] = 1.0 + math.log((n_warm + 1.0) / (df_cat[cat] + 1.0))

        df_brand: Dict[str, int] = defaultdict(int)
        for item in context.sorted_warm_items:
            brand = item_brands.get(item)
            if brand is not None and brand != "":
                df_brand[brand] += 1

        self.idf_brand: Dict[str, float] = {}
        for brand in sorted(df_brand.keys()):
            self.idf_brand[brand] = 1.0 + math.log((n_warm + 1.0) / (df_brand[brand] + 1.0))

        # Build product vectors
        self.product_cat_vec: Dict[str, Dict[str, float]] = {}
        for item in context.sorted_warm_items:
            cats = item_categories.get(item)
            if cats:
                unique_cats = sorted(set(cats))
                vec = {cat: self.idf_category[cat] for cat in unique_cats if cat in self.idf_category}
                norm = math.sqrt(math.fsum(v * v for _, v in sorted(vec.items())))
                if norm > 0.0:
                    self.product_cat_vec[item] = {k: v / norm for k, v in sorted(vec.items())}
                else:
                    self.product_cat_vec[item] = {}
            else:
                self.product_cat_vec[item] = {}

        self.product_brand_vec: Dict[str, Dict[str, float]] = {}
        for item in context.sorted_warm_items:
            brand = item_brands.get(item)
            if brand is not None and brand != "" and brand in self.idf_brand:
                self.product_brand_vec[item] = {brand: 1.0}
            else:
                self.product_brand_vec[item] = {}

        # Build user vectors
        self.user_cat_vec: Dict[str, Dict[str, float]] = {}
        self.user_brand_vec: Dict[str, Dict[str, float]] = {}

        for user_id in sorted(context.users):
            distinct_pos = sorted(context.positive_history.get(user_id, set()))
            if distinct_pos:
                cat_sum: Dict[str, float] = defaultdict(float)
                for item in distinct_pos:
                    for cat, val in sorted(self.product_cat_vec.get(item, {}).items()):
                        cat_sum[cat] += val
                norm_cat = math.sqrt(math.fsum(v * v for _, v in sorted(cat_sum.items())))
                if norm_cat > 0.0:
                    self.user_cat_vec[user_id] = {k: v / norm_cat for k, v in sorted(cat_sum.items())}
                else:
                    self.user_cat_vec[user_id] = {}

                brand_sum: Dict[str, float] = defaultdict(float)
                for item in distinct_pos:
                    for brand, val in sorted(self.product_brand_vec.get(item, {}).items()):
                        brand_sum[brand] += val
                norm_brand = math.sqrt(math.fsum(v * v for _, v in sorted(brand_sum.items())))
                if norm_brand > 0.0:
                    self.user_brand_vec[user_id] = {k: v / norm_brand for k, v in sorted(brand_sum.items())}
                else:
                    self.user_brand_vec[user_id] = {}
            else:
                self.user_cat_vec[user_id] = {}
                self.user_brand_vec[user_id] = {}

        # Validate that profiles do not contain non-finite values
        self._validate_profile_finiteness()

    @staticmethod
    def _validate_profile_dict(profile: Any, label: str = "profile") -> None:
        if not isinstance(profile, dict):
            raise ScoringFailureError(f"Corrupt profile dictionary for {label}: {profile!r}")
        for k, v in profile.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise ScoringFailureError(f"Corrupt non-finite value in {label}: {v!r}")

    @property
    def user_category_profiles(self) -> Dict[str, Dict[str, float]]:
        return self.user_cat_vec

    @property
    def user_brand_profiles(self) -> Dict[str, Dict[str, float]]:
        return self.user_brand_vec

    @property
    def item_category_profiles(self) -> Dict[str, Dict[str, float]]:
        return self.product_cat_vec

    @property
    def item_brand_profiles(self) -> Dict[str, Dict[str, float]]:
        return self.product_brand_vec

    def _validate_profile_finiteness(self) -> None:
        for p, vec in self.product_cat_vec.items():
            KGOnlyScorer._validate_profile_dict(vec, f"product category vector for {p}")
        for p, vec in self.product_brand_vec.items():
            KGOnlyScorer._validate_profile_dict(vec, f"product brand vector for {p}")
        for u, vec in self.user_cat_vec.items():
            KGOnlyScorer._validate_profile_dict(vec, f"user category vector for {u}")
        for u, vec in self.user_brand_vec.items():
            KGOnlyScorer._validate_profile_dict(vec, f"user brand vector for {u}")

    def score_candidates(self, user_id: str, candidate_items: Sequence[str]) -> Tuple[Dict[str, float], str, Optional[str]]:
        u_cat = self.user_cat_vec.get(user_id, {})
        u_brand = self.user_brand_vec.get(user_id, {})

        # Verify profile finiteness for this user across both channels regardless of mixture
        KGOnlyScorer._validate_profile_dict(u_cat, f"user category profile for {user_id}")
        KGOnlyScorer._validate_profile_dict(u_brand, f"user brand profile for {user_id}")

        pos_items = self.context.positive_history.get(user_id, set())
        if not pos_items:
            scores = {item: float(self.context.popularity.get(item, 0)) for item in candidate_items}
            return scores, "fallback", "empty_positive_history"

        cat_usable = (self.alpha > 0.0) and (len(u_cat) > 0)
        brand_usable = (self.beta > 0.0) and (len(u_brand) > 0)

        if not (cat_usable or brand_usable):
            scores = {item: float(self.context.popularity.get(item, 0)) for item in candidate_items}
            return scores, "fallback", "unusable_metadata_profile"

        scores: Dict[str, float] = {}
        for item in candidate_items:
            p_cat = self.product_cat_vec.get(item, {})
            p_brand = self.product_brand_vec.get(item, {})

            # Check candidate product profiles for non-finite corruption
            KGOnlyScorer._validate_profile_dict(p_cat, f"product category profile for candidate {item}")
            KGOnlyScorer._validate_profile_dict(p_brand, f"product brand profile for candidate {item}")

            c_cat = 0.0
            if cat_usable and p_cat:
                common_cats = sorted(set(u_cat.keys()) & set(p_cat.keys()))
                c_cat = math.fsum(u_cat[cat] * p_cat[cat] for cat in common_cats)

            c_brand = 0.0
            if brand_usable and p_brand:
                common_brands = sorted(set(u_brand.keys()) & set(p_brand.keys()))
                c_brand = math.fsum(u_brand[b] * p_brand[b] for b in common_brands)

            score = self.alpha * c_cat + self.beta * c_brand
            if math.isnan(score) or math.isinf(score):
                raise ScoringFailureError(f"Non-finite candidate score computed for user {user_id}, item {item}: {score}")
            scores[item] = score

        return scores, "personalized", None


class Evaluator(BaseEvaluator):
    """Core evaluator extending shared BaseEvaluator with guarded KG metadata fallback."""

    DEFAULT_FALLBACK_REASONS: Tuple[str, ...] = (
        "empty_positive_history",
        "unusable_metadata_profile",
    )

    def _get_default_fallback_reasons(self) -> Set[str]:
        return set(self.DEFAULT_FALLBACK_REASONS)

    def _validate_provider_result(
        self,
        scorer: Any,
        user_id: str,
        authoritative_candidates: Sequence[str],
        scores: Any,
        route: Any,
        reason: Any,
    ) -> Tuple[Dict[str, float], str, Optional[str]]:
        """Validate provider scores, route, and fallback reason.

        Permits strictly validated whole-user unusable_metadata_profile fallback
        only for legitimate same-context KGOnlyScorer instances with unusable positively
        weighted profiles and exact condition-positive popularity scores.
        All other provider fallback claims, wrong contexts, corrupt profiles, usable profiles,
        spoofed reasons/scores, and scoring failures are rejected.
        Ordinary routes ('personalized', 'non_personalized') are delegated to super().
        """
        if route != "fallback":
            return super()._validate_provider_result(
                scorer, user_id, authoritative_candidates, scores, route, reason
            )

        # route == "fallback"
        if not isinstance(scores, dict):
            raise ScoringFailureError(f"Scorer must return a dictionary of candidate scores for user {user_id}.")

        if not isinstance(reason, str) or reason != "unusable_metadata_profile":
            raise ScoringFailureError(f"Invalid fallback reason {reason!r} for user {user_id}. Only 'unusable_metadata_profile' may be reported by scorer.")

        if not isinstance(scorer, KGOnlyScorer):
            raise ScoringFailureError(f"Metadata fallback 'unusable_metadata_profile' can only be reported by KGOnlyScorer, got {type(scorer).__name__} for user {user_id}.")

        if scorer.context is not self.context:
            raise ScoringFailureError(f"Scorer context is not identical to Evaluator context for user {user_id}.")

        pos_items = self.context.positive_history.get(user_id, set())
        if not pos_items:
            raise ScoringFailureError(f"Empty-positive fallback is evaluator-owned; cannot be reported by scorer for user {user_id}.")

        # Validate outer profile containers exist and are dictionaries
        for attr in ("user_cat_vec", "user_brand_vec", "product_cat_vec", "product_brand_vec"):
            if not hasattr(scorer, attr) or not isinstance(getattr(scorer, attr), dict):
                raise ScoringFailureError(f"Scorer profile container {attr!r} is missing or not a dictionary for user {user_id}.")

        # Validate user channel profile existence and finiteness via authoritative base validator
        if user_id not in scorer.user_cat_vec:
            raise ScoringFailureError(f"Scorer user category profile container missing entry for user {user_id}.")
        if user_id not in scorer.user_brand_vec:
            raise ScoringFailureError(f"Scorer user brand profile container missing entry for user {user_id}.")

        u_cat = scorer.user_cat_vec[user_id]
        u_brand = scorer.user_brand_vec[user_id]
        KGOnlyScorer._validate_profile_dict(u_cat, f"user category profile for {user_id}")
        KGOnlyScorer._validate_profile_dict(u_brand, f"user brand profile for {user_id}")

        # Validate candidate channel profile existence and finiteness via authoritative base validator
        for item in authoritative_candidates:
            if item not in scorer.product_cat_vec:
                raise ScoringFailureError(f"Scorer product category profile container missing entry for candidate {item}.")
            if item not in scorer.product_brand_vec:
                raise ScoringFailureError(f"Scorer product brand profile container missing entry for candidate {item}.")
            p_cat = scorer.product_cat_vec[item]
            p_brand = scorer.product_brand_vec[item]
            KGOnlyScorer._validate_profile_dict(p_cat, f"product category profile for candidate {item}")
            KGOnlyScorer._validate_profile_dict(p_brand, f"product brand profile for candidate {item}")

        # Validate mixture weights exist and are finite numbers
        if not hasattr(scorer, "alpha") or not hasattr(scorer, "beta"):
            raise ScoringFailureError(f"Scorer mixture weights missing for user {user_id}.")
        if isinstance(scorer.alpha, bool) or not isinstance(scorer.alpha, (int, float)) or math.isnan(scorer.alpha) or math.isinf(scorer.alpha):
            raise ScoringFailureError(f"Scorer alpha mixture weight must be a finite number for user {user_id}.")
        if isinstance(scorer.beta, bool) or not isinstance(scorer.beta, (int, float)) or math.isnan(scorer.beta) or math.isinf(scorer.beta):
            raise ScoringFailureError(f"Scorer beta mixture weight must be a finite number for user {user_id}.")

        # Enforce positively weighted channel usability: user cannot claim metadata fallback if any positively weighted channel is usable
        cat_usable = (scorer.alpha > 0.0) and (len(u_cat) > 0)
        brand_usable = (scorer.beta > 0.0) and (len(u_brand) > 0)
        if cat_usable or brand_usable:
            raise ScoringFailureError(f"User {user_id} has usable positively weighted metadata channels; metadata fallback is invalid.")

        # Enforce exact condition-positive popularity scores for all authoritative candidates
        for item in authoritative_candidates:
            if item not in scores:
                raise ScoringFailureError(f"Fallback scores missing candidate {item} for user {user_id}.")
            s = scores[item]
            if isinstance(s, bool) or not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s):
                raise ScoringFailureError(f"Non-finite candidate score {s!r} for user {user_id}, item {item}.")
            expected = float(self.context.popularity.get(item, 0))
            if s != expected:
                raise ScoringFailureError(f"Fallback score for candidate {item} ({s!r}) does not match condition-positive popularity {expected}.")

        return scores, route, reason


def select_kg_only_mixture(
    context: TrainingContext,
    item_categories: Dict[str, Sequence[str]],
    item_brands: Dict[str, Optional[str]],
    validation_targets: Iterable[Union[Tuple[str, str, float], Dict[str, Any]]],
) -> Tuple[Tuple[float, float], Dict[Tuple[float, float], Dict[str, Any]]]:
    """Select KG-only mixture from the ordered grid using validation NDCG@10 only.

    On exact metric ties, selects the earliest ordered entry.
    Fails explicitly if there are no eligible validation users (all-N/A population).
    """
    evaluator = Evaluator(context)
    trials: Dict[Tuple[float, float], Dict[str, Any]] = {}
    best_mixture: Optional[Tuple[float, float]] = None
    best_score: float = -1.0

    targets_list = list(validation_targets)

    for mixture in KGOnlyScorer.GRID:
        scorer = KGOnlyScorer(context, item_categories, item_brands, mixture=mixture)
        report = evaluator.evaluate(targets_list, scorer, split="validation")
        trials[mixture] = report

        if report["audit"]["eligible_users"] == 0:
            continue

        ndcg10 = report["overall_metrics"].get("NDCG@10")
        if ndcg10 is None or math.isnan(ndcg10) or math.isinf(ndcg10):
            continue

        score = float(ndcg10)
        if score > best_score:
            best_score = score
            best_mixture = mixture

    if best_mixture is None:
        raise ValueError("Cannot select KG-only mixture: validation population has no eligible users or all-N/A metric scores.")

    return best_mixture, trials


class NegativeSampler:
    """Uniform one-negative sampler over condition-warm pool minus all user training-rated items.

    Evaluates user negative pools on demand to avoid memory scaling issues.
    Validates that training pairs exist in user condition-positive history.
    """

    def __init__(self, context: TrainingContext) -> None:
        self.context = context
        self.sorted_warm_items = context.sorted_warm_items
        warm_set = context.warm_items

        # Dataset preflight: ensure at least one trainable positive example exists globally
        has_trainable = False
        for u in context.positive_users:
            if len(warm_set - self.context.all_rated_history[u]) > 0:
                has_trainable = True
                break

        if not has_trainable:
            raise ValueError("No trainable positive examples exist in the training context (all positive users have exhausted warm item pools).")

    def get_negative_pool(self, user_id: str) -> List[str]:
        if user_id not in self.context.users:
            raise ValueError(f"Unknown user {user_id!r} in NegativeSampler.")
        rated = self.context.all_rated_history[user_id]
        return [item for item in self.sorted_warm_items if item not in rated]

    def sample_negative(self, user_id: str, rng: random.Random) -> Optional[str]:
        pool = self.get_negative_pool(user_id)
        if not pool:
            return None
        return rng.choice(pool)

    def prepare_batch(
        self,
        positive_pairs: Sequence[Tuple[str, str]],
        rng: random.Random,
    ) -> Tuple[List[Tuple[str, str, str]], int]:
        trainable: List[Tuple[str, str, str]] = []
        skipped_empty_pool = 0

        for user_id, pos_item in positive_pairs:
            if user_id not in self.context.users:
                raise ValueError(f"Unknown user {user_id!r} in positive pairs.")
            pos_history = self.context.positive_history.get(user_id, set())
            if pos_item not in pos_history:
                raise ValueError(
                    f"Positive pair ({user_id!r}, {pos_item!r}) not in user's condition-positive training history."
                )

            neg_item = self.sample_negative(user_id, rng)
            if neg_item is None:
                skipped_empty_pool += 1
            else:
                trainable.append((user_id, pos_item, neg_item))

        return trainable, skipped_empty_pool


def stable_softplus(x: float) -> float:
    """Numerically stable softplus: ln(1 + e^x) with finite validation."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
        raise ValueError(f"Non-finite or invalid input to softplus: {x!r}")

    try:
        if x > 35.0:
            val = float(x)
        elif x < -35.0:
            val = math.exp(x)
        else:
            val = math.log1p(math.exp(x))
    except OverflowError as exc:
        raise ValueError(f"Softplus overflow for input {x}: {exc}") from exc

    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"Non-finite softplus output computed for input: {x}")
    return val


def bpr_loss(s_positive: Sequence[float], s_negative: Sequence[float]) -> float:
    """Mean softplus(s_negative - s_positive) over valid supervised examples."""
    if len(s_positive) != len(s_negative):
        raise ValueError("Positive and negative score sequences must have identical lengths.")
    if not s_positive:
        raise ValueError("Cannot compute BPR loss on empty score sequences.")

    losses: List[float] = []
    for sp, sn in zip(s_positive, s_negative):
        if isinstance(sp, bool) or not isinstance(sp, (int, float)) or math.isnan(sp) or math.isinf(sp):
            raise ValueError(f"Non-finite positive score in BPR loss: {sp!r}")
        if isinstance(sn, bool) or not isinstance(sn, (int, float)) or math.isnan(sn) or math.isinf(sn):
            raise ValueError(f"Non-finite negative score in BPR loss: {sn!r}")

        try:
            diff = sn - sp
        except OverflowError as exc:
            raise ValueError(f"Score difference overflow in BPR loss: sn={sn}, sp={sp}") from exc

        if math.isnan(diff) or math.isinf(diff):
            raise ValueError(f"Score difference overflow in BPR loss: sn={sn}, sp={sp}")

        losses.append(stable_softplus(diff))

    try:
        total = math.fsum(losses)
        res = total / len(s_positive)
    except OverflowError as exc:
        raise ValueError(f"Aggregate BPR loss overflow: {exc}") from exc

    if math.isnan(res) or math.isinf(res):
        raise ValueError("Non-finite aggregate BPR loss computed.")
    return res


def endpoint_regularization(
    user_ids: Sequence[str],
    positive_item_ids: Sequence[str],
    negative_item_ids: Sequence[str],
    user_base_embeddings: Dict[str, Sequence[float]],
    item_base_embeddings: Dict[str, Sequence[float]],
    coefficient: float = 1e-5,
) -> float:
    """Explicit endpoint-only base-ID regularization.

    Formula: 1e-5 * sum(squared L2 row norms of unique base endpoints) / (unique endpoint row count).
    Baseline endpoint coefficient must be fixed exactly 1e-5.
    Unique endpoints: unique batch user IDs + union of positive and negative item IDs.
    Excludes Brand, CategoryPath, and neighbor-only rows. No dimension division or half factor.
    """
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or math.isnan(coefficient) or math.isinf(coefficient) or coefficient != 1e-5:
        raise ValueError(f"Endpoint regularization coefficient must be exactly 1e-5, got {coefficient!r}.")

    if len(user_ids) != len(positive_item_ids) or len(user_ids) != len(negative_item_ids):
        raise ValueError("User, positive item, and negative item batches must have identical lengths.")
    if not user_ids:
        raise ValueError("Cannot compute endpoint regularization on empty batch.")

    unique_users = sorted(set(user_ids))
    unique_items = sorted(set(positive_item_ids) | set(negative_item_ids))

    n_unique_endpoints = len(unique_users) + len(unique_items)
    if n_unique_endpoints == 0:
        raise ValueError("No unique endpoints in batch.")

    target_dim: Optional[int] = None
    squared_norms: List[float] = []

    for u in unique_users:
        if u not in user_base_embeddings:
            raise KeyError(f"Base embedding missing for user {u}")
        vec = user_base_embeddings[u]
        if not isinstance(vec, (list, tuple)):
            raise ValueError(f"User embedding row for {u} must be a sequence.")
        if len(vec) == 0:
            raise ValueError(f"User embedding row for {u} has zero dimension.")
        if target_dim is None:
            target_dim = len(vec)
        elif len(vec) != target_dim:
            raise ValueError(f"Inconsistent base embedding dimension: expected {target_dim}, got {len(vec)} for user {u}.")

        for v in vec:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise ValueError(f"Non-finite value in base embedding for user {u}: {v!r}")

        try:
            sq_norm = math.fsum(v * v for v in vec)
        except OverflowError as exc:
            raise ValueError(f"Squared-norm overflow in base embedding for user {u}: {exc}") from exc

        if math.isnan(sq_norm) or math.isinf(sq_norm):
            raise ValueError(f"Non-finite squared norm in base embedding for user {u}: {sq_norm}")
        squared_norms.append(sq_norm)

    for item in unique_items:
        if item not in item_base_embeddings:
            raise KeyError(f"Base embedding missing for item {item}")
        vec = item_base_embeddings[item]
        if not isinstance(vec, (list, tuple)):
            raise ValueError(f"Item embedding row for {item} must be a sequence.")
        if len(vec) == 0:
            raise ValueError(f"Item embedding row for {item} has zero dimension.")
        if target_dim is None:
            target_dim = len(vec)
        elif len(vec) != target_dim:
            raise ValueError(f"Inconsistent base embedding dimension: expected {target_dim}, got {len(vec)} for item {item}.")

        for v in vec:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise ValueError(f"Non-finite value in base embedding for item {item}: {v!r}")

        try:
            sq_norm = math.fsum(v * v for v in vec)
        except OverflowError as exc:
            raise ValueError(f"Squared-norm overflow in base embedding for item {item}: {exc}") from exc

        if math.isnan(sq_norm) or math.isinf(sq_norm):
            raise ValueError(f"Non-finite squared norm in base embedding for item {item}: {sq_norm}")
        squared_norms.append(sq_norm)

    try:
        total_sq_norm = math.fsum(squared_norms)
        penalty = coefficient * (total_sq_norm / n_unique_endpoints)
    except OverflowError as exc:
        raise ValueError(f"Aggregate endpoint penalty overflow: {exc}") from exc

    if math.isnan(penalty) or math.isinf(penalty):
        raise ValueError("Non-finite endpoint penalty computed.")
    return penalty


if __package__:
    from .learned_models import (
        HeterogeneousKGNNConfig,
        HeterogeneousKGNNGraph,
        HeterogeneousKGNNScorer,
        HeterogeneousParameters,
        LightGCNConfig,
        LightGCNGraph,
        LightGCNScorer,
        MissingIdError,
        TrainingHarnessGuard,
        UnknownIdError,
        build_heterogeneous_graph,
        build_lightgcn_graph,
        compute_bpr_batch_loss,
        compute_heterogeneous_bpr_batch_loss,
        create_checkpoint_manifest,
        initialize_heterogeneous_parameters,
        initialize_node_embeddings,
        mask_heterogeneous_batch_edges,
        propagate_heterogeneous_kgnn,
        propagate_lightgcn,
        validate_checkpoint_manifest,
    )
else:
    from learned_models import (
        HeterogeneousKGNNConfig,
        HeterogeneousKGNNGraph,
        HeterogeneousKGNNScorer,
        HeterogeneousParameters,
        LightGCNConfig,
        LightGCNGraph,
        LightGCNScorer,
        MissingIdError,
        TrainingHarnessGuard,
        UnknownIdError,
        build_heterogeneous_graph,
        build_lightgcn_graph,
        compute_bpr_batch_loss,
        compute_heterogeneous_bpr_batch_loss,
        create_checkpoint_manifest,
        initialize_heterogeneous_parameters,
        initialize_node_embeddings,
        mask_heterogeneous_batch_edges,
        propagate_heterogeneous_kgnn,
        propagate_lightgcn,
        validate_checkpoint_manifest,
    )


