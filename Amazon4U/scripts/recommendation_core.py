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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


class ScoringFailureError(Exception):
    """Raised when candidate scoring fails, returns non-finite values, or produces corrupt outputs."""
    pass


class TrainingContext:
    """Validated training-only context for recommendation evaluation and modeling."""

    def __init__(
        self,
        interactions: Iterable[Union[Tuple[str, str, float], Dict[str, Any]]],
        condition: str = "all_ratings",
    ) -> None:
        if condition not in ("all_ratings", "rating_ge_4"):
            raise ValueError(f"Invalid feedback condition: {condition}. Must be 'all_ratings' or 'rating_ge_4'.")
        self.condition = condition

        self.all_rated_history: Dict[str, Set[str]] = defaultdict(set)
        self.positive_history: Dict[str, Set[str]] = defaultdict(set)
        self.users: Set[str] = set()
        self.all_training_items: Set[str] = set()
        self._positive_interaction_counts: Dict[str, int] = defaultdict(int)

        seen_pairs: Set[Tuple[str, str]] = set()

        for record in interactions:
            if isinstance(record, dict):
                user_id = record.get("user_id")
                parent_asin = record.get("parent_asin")
                rating = record.get("rating")
            else:
                user_id, parent_asin, rating = record

            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(f"Invalid user_id: {user_id!r}. Must be a non-empty string.")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"Invalid parent_asin: {parent_asin!r}. Must be a non-empty string.")
            if isinstance(rating, bool) or not isinstance(rating, (int, float)) or math.isnan(rating) or math.isinf(rating):
                raise ValueError(f"Invalid rating: {rating!r}. Must be a finite number and not boolean.")

            pair = (user_id, parent_asin)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate training user-item pair detected: {pair}")
            seen_pairs.add(pair)

            self.users.add(user_id)
            self.all_training_items.add(parent_asin)
            self.all_rated_history[user_id].add(parent_asin)

            is_positive = True if condition == "all_ratings" else (rating >= 4.0)
            if is_positive:
                self.positive_history[user_id].add(parent_asin)
                self._positive_interaction_counts[parent_asin] += 1

        if condition == "all_ratings":
            self.warm_items: Set[str] = set(self.all_training_items)
        else:
            self.warm_items = {item for item, count in self._positive_interaction_counts.items() if count > 0}

        self.sorted_warm_items: List[str] = sorted(self.warm_items)
        self.popularity: Dict[str, int] = {item: self._positive_interaction_counts[item] for item in self.warm_items}
        self.positive_users: Set[str] = {u for u, items in self.positive_history.items() if len(items) > 0}


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

    def _validate_profile_finiteness(self) -> None:
        for p, vec in self.product_cat_vec.items():
            for k, v in vec.items():
                if math.isnan(v) or math.isinf(v):
                    raise ScoringFailureError(f"Corrupt non-finite value in product category vector for {p}: {v}")
        for p, vec in self.product_brand_vec.items():
            for k, v in vec.items():
                if math.isnan(v) or math.isinf(v):
                    raise ScoringFailureError(f"Corrupt non-finite value in product brand vector for {p}: {v}")
        for u, vec in self.user_cat_vec.items():
            for k, v in vec.items():
                if math.isnan(v) or math.isinf(v):
                    raise ScoringFailureError(f"Corrupt non-finite value in user category vector for {u}: {v}")
        for u, vec in self.user_brand_vec.items():
            for k, v in vec.items():
                if math.isnan(v) or math.isinf(v):
                    raise ScoringFailureError(f"Corrupt non-finite value in user brand vector for {u}: {v}")

    def score_candidates(self, user_id: str, candidate_items: Sequence[str]) -> Tuple[Dict[str, float], str, Optional[str]]:
        u_cat = self.user_cat_vec.get(user_id, {})
        u_brand = self.user_brand_vec.get(user_id, {})

        # Verify profile finiteness for this user across both channels regardless of mixture
        for k, v in u_cat.items():
            if math.isnan(v) or math.isinf(v):
                raise ScoringFailureError(f"Corrupt non-finite user profile value for {user_id}: {v}")
        for k, v in u_brand.items():
            if math.isnan(v) or math.isinf(v):
                raise ScoringFailureError(f"Corrupt non-finite user profile value for {user_id}: {v}")

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
            for k, v in p_cat.items():
                if math.isnan(v) or math.isinf(v):
                    raise ScoringFailureError(f"Corrupt non-finite product profile for candidate {item}: {v}")
            for k, v in p_brand.items():
                if math.isnan(v) or math.isinf(v):
                    raise ScoringFailureError(f"Corrupt non-finite product profile for candidate {item}: {v}")

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


class Evaluator:
    """Reusable evaluator enforcing history filtering, target auditing, ranking, metrics, and subgroup reporting."""

    CUTOFFS = (5, 10, 20)

    def __init__(self, context: TrainingContext) -> None:
        self.context = context

    def evaluate(
        self,
        targets: Iterable[Union[Tuple[str, str, float], Dict[str, Any]]],
        scorer: Any,
        split: str = "validation",
        validation_history: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> Dict[str, Any]:
        if split not in ("validation", "test"):
            raise ValueError(f"Invalid split: {split}. Must be 'validation' or 'test'.")

        val_history_map: Dict[str, Set[str]] = defaultdict(set)
        if split == "test":
            if validation_history is None:
                raise ValueError("validation_history must be explicitly provided for test split.")
            if isinstance(validation_history, dict):
                for u, items in validation_history.items():
                    if not isinstance(u, str) or not u.strip():
                        raise ValueError(f"Invalid user_id in validation_history: {u!r}")
                    if isinstance(items, (str, bytes)):
                        raise ValueError(f"Validation history items for user {u} must be an item collection, not a string.")
                    if not isinstance(items, (list, tuple, set, frozenset)):
                        raise ValueError(f"Validation history items for user {u} must be a collection of item IDs.")
                    for item in items:
                        if not isinstance(item, str) or not item.strip():
                            raise ValueError(f"Invalid item ID {item!r} in validation_history for user {u}.")
                        val_history_map[u].add(item)
            elif isinstance(validation_history, (list, tuple, set)):
                for rec in validation_history:
                    if isinstance(rec, dict):
                        u = rec.get("user_id")
                        item = rec.get("parent_asin")
                    elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                        u, item = rec[0], rec[1]
                    else:
                        raise ValueError(f"Invalid validation_history record: {rec!r}")
                    if not isinstance(u, str) or not u.strip():
                        raise ValueError(f"Invalid user_id {u!r} in validation_history record.")
                    if not isinstance(item, str) or not item.strip():
                        raise ValueError(f"Invalid item ID {item!r} in validation_history record.")
                    val_history_map[u].add(item)
            else:
                raise ValueError("validation_history must be a dictionary or sequence of records.")

        target_records: List[Tuple[str, str, float]] = []
        seen_target_users: Set[str] = set()

        for rec in targets:
            if isinstance(rec, dict):
                u = rec.get("user_id")
                p = rec.get("parent_asin")
                r = rec.get("rating")
            elif isinstance(rec, (list, tuple)) and len(rec) >= 3:
                u, p, r = rec[0], rec[1], rec[2]
            else:
                raise ValueError(f"Invalid target record: {rec!r}")

            if not isinstance(u, str) or not u.strip():
                raise ValueError(f"Invalid target user_id: {u!r}. Must be a non-empty string.")
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"Invalid target parent_asin: {p!r}. Must be a non-empty string.")
            if isinstance(r, bool) or not isinstance(r, (int, float)) or math.isnan(r) or math.isinf(r):
                raise ValueError(f"Invalid target rating: {r!r} for user {u}. Must be a finite number and not boolean.")

            if u in seen_target_users:
                raise ValueError(f"Duplicate held-out target for user {u}. Exactly one target per user permitted.")
            seen_target_users.add(u)

            # In test split, require explicit validation history entry for known target users
            if split == "test" and (u in self.context.users) and (u not in val_history_map):
                raise ValueError(f"Missing validation history entry for known target user {u} in test split.")

            target_records.append((u, p, float(r)))

        target_records.sort(key=lambda x: x[0])

        warm_pool = self.context.warm_items
        warm_pool_size = len(warm_pool)

        unknown_users = 0
        irrelevant_targets = 0
        cold_targets = 0
        history_overlaps = 0
        overlapping_reasons: Dict[str, int] = defaultdict(int)

        eligible_records: List[Tuple[str, str, float, Set[str]]] = []

        for u, p, r in target_records:
            is_known = u in self.context.users
            is_relevant = True if self.context.condition == "all_ratings" else (r >= 4.0)
            is_warm = p in warm_pool

            if split == "validation":
                filtered_history = set(self.context.all_rated_history.get(u, set()))
            else:
                filtered_history = set(self.context.all_rated_history.get(u, set())) | set(val_history_map.get(u, set()))

            in_history = p in filtered_history

            reasons: List[str] = []
            if not is_known:
                unknown_users += 1
                reasons.append("unknown_user")
            if not is_relevant:
                irrelevant_targets += 1
                reasons.append("irrelevant_target")
            if not is_warm:
                cold_targets += 1
                reasons.append("cold_target")
            if in_history:
                history_overlaps += 1
                reasons.append("history_overlap")

            if reasons:
                reasons_key = "+".join(sorted(reasons))
                overlapping_reasons[reasons_key] += 1
            else:
                eligible_records.append((u, p, r, filtered_history))

        eligible_count = len(eligible_records)
        shortage_counts = {k: 0 for k in self.CUTOFFS}

        per_user_results: List[Dict[str, Any]] = []
        recommended_at_10: Set[str] = set()

        for u, target_item, rating, filtered_history in eligible_records:
            candidates = [item for item in self.context.sorted_warm_items if item not in filtered_history]

            for k in self.CUTOFFS:
                if len(candidates) < k:
                    shortage_counts[k] += 1

            has_positive_history = len(self.context.positive_history.get(u, set())) > 0

            if not has_positive_history:
                route = "fallback"
                fallback_reason = "empty_positive_history"
                scores = {item: float(self.context.popularity.get(item, 0)) for item in candidates}
            else:
                try:
                    if hasattr(scorer, "score_candidates"):
                        res = scorer.score_candidates(u, candidates)
                    elif callable(scorer):
                        res = scorer(u, candidates)
                    else:
                        raise ScoringFailureError(f"Scorer {scorer} has no score_candidates method and is not callable.")

                    if isinstance(res, tuple):
                        if len(res) == 3:
                            scores, route, fallback_reason = res
                        elif len(res) == 2:
                            scores, route = res
                            fallback_reason = None
                        else:
                            raise ScoringFailureError(f"Scorer returned unexpected tuple length: {len(res)}")
                    elif isinstance(res, dict):
                        scores = res
                        route = "personalized"
                        fallback_reason = None
                    else:
                        raise ScoringFailureError(f"Scorer returned invalid output type: {type(res)}")
                except ScoringFailureError:
                    raise
                except Exception as exc:
                    raise ScoringFailureError(f"Scorer failed during evaluation for user {u}: {exc}") from exc

                if not isinstance(scores, dict):
                    raise ScoringFailureError(f"Scorer must return a dictionary of candidate scores for user {u}.")

                if not isinstance(route, str) or route not in ("personalized", "non_personalized", "fallback"):
                    raise ScoringFailureError(f"Invalid route {route!r} for user {u}. Must be 'personalized', 'non_personalized', or 'fallback'.")

                if route in ("personalized", "non_personalized"):
                    if fallback_reason is not None:
                        raise ScoringFailureError(f"Non-fallback route {route!r} must have fallback_reason None, got {fallback_reason!r} for user {u}.")
                elif route == "fallback":
                    if not isinstance(fallback_reason, str) or fallback_reason not in ("empty_positive_history", "unusable_metadata_profile"):
                        raise ScoringFailureError(f"Invalid fallback reason {fallback_reason!r} for user {u}. Must be 'empty_positive_history' or 'unusable_metadata_profile'.")
                    if fallback_reason == "empty_positive_history":
                        raise ScoringFailureError(f"Empty-positive fallback is evaluator-owned; cannot be reported by scorer for user {u} with positive history.")
                    if fallback_reason == "unusable_metadata_profile" and not isinstance(scorer, KGOnlyScorer):
                        raise ScoringFailureError(f"Metadata fallback 'unusable_metadata_profile' can only be reported by KGOnlyScorer, got {type(scorer).__name__} for user {u}.")

            # Validate that every candidate is scored
            for item in candidates:
                if item not in scores:
                    raise ScoringFailureError(f"Scorer missing candidate score for item {item} for user {u}.")
                s = scores[item]
                if isinstance(s, bool) or not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s):
                    raise ScoringFailureError(f"Non-finite candidate score {s!r} for user {u}, item {item}.")

            ranked_items = sorted(candidates, key=lambda item: (-scores[item], item))

            user_metrics: Dict[str, float] = {}
            for k in self.CUTOFFS:
                top_k = ranked_items[:k]
                if target_item in top_k:
                    rank_1based = top_k.index(target_item) + 1
                    user_metrics[f"HitRate@{k}"] = 1.0
                    user_metrics[f"NDCG@{k}"] = 1.0 / math.log2(rank_1based + 1.0)
                else:
                    user_metrics[f"HitRate@{k}"] = 0.0
                    user_metrics[f"NDCG@{k}"] = 0.0

            top_5 = ranked_items[:5]
            top_10 = ranked_items[:10]
            top_20 = ranked_items[:20]
            recommended_at_10.update(top_10)

            per_user_results.append({
                "user_id": u,
                "target_item": target_item,
                "route": route,
                "fallback_reason": fallback_reason,
                "candidate_count": len(candidates),
                "metrics": user_metrics,
                "top_5": top_5,
                "top_10": top_10,
                "top_20": top_20,
                "top_k": {5: top_5, 10: top_10, 20: top_20},
            })

        if eligible_count == 0:
            overall_metrics: Dict[str, Optional[float]] = {
                "NDCG@5": None,
                "NDCG@10": None,
                "NDCG@20": None,
                "HitRate@5": None,
                "HitRate@10": None,
                "HitRate@20": None,
                "CatalogCoverage@10": None,
            }
        else:
            overall_metrics = {}
            for k in self.CUTOFFS:
                overall_metrics[f"NDCG@{k}"] = sum(r["metrics"][f"NDCG@{k}"] for r in per_user_results) / eligible_count
                overall_metrics[f"HitRate@{k}"] = sum(r["metrics"][f"HitRate@{k}"] for r in per_user_results) / eligible_count
            overall_metrics["CatalogCoverage@10"] = (len(recommended_at_10) / warm_pool_size) if warm_pool_size > 0 else 0.0

        # Subgroup aggregation across protocol routes
        all_routes = ["personalized", "non_personalized", "fallback"]
        subgroup_results: Dict[str, Any] = {}

        for route_name in all_routes:
            sub_users = [r for r in per_user_results if r["route"] == route_name]
            count = len(sub_users)
            if count == 0:
                sub_metrics: Dict[str, Optional[float]] = {
                    "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                    "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                    "CatalogCoverage@10": None,
                }
            else:
                sub_metrics = {}
                for k in self.CUTOFFS:
                    sub_metrics[f"NDCG@{k}"] = sum(r["metrics"][f"NDCG@{k}"] for r in sub_users) / count
                    sub_metrics[f"HitRate@{k}"] = sum(r["metrics"][f"HitRate@{k}"] for r in sub_users) / count
                sub_rec_10 = set().union(*(r["top_10"] for r in sub_users))
                sub_metrics["CatalogCoverage@10"] = (len(sub_rec_10) / warm_pool_size) if warm_pool_size > 0 else 0.0

            subgroup_results[route_name] = {
                "count": count,
                "metrics": sub_metrics,
            }

        # Fallback by reason aggregation
        fallback_users = [r for r in per_user_results if r["route"] == "fallback"]
        observed_reasons = ["empty_positive_history", "unusable_metadata_profile"]
        by_reason: Dict[str, Any] = {}

        for reason in observed_reasons:
            reason_users = [r for r in fallback_users if r["fallback_reason"] == reason]
            r_count = len(reason_users)
            if r_count == 0:
                r_metrics: Dict[str, Optional[float]] = {
                    "NDCG@5": None, "NDCG@10": None, "NDCG@20": None,
                    "HitRate@5": None, "HitRate@10": None, "HitRate@20": None,
                    "CatalogCoverage@10": None,
                }
            else:
                r_metrics = {}
                for k in self.CUTOFFS:
                    r_metrics[f"NDCG@{k}"] = sum(r["metrics"][f"NDCG@{k}"] for r in reason_users) / r_count
                    r_metrics[f"HitRate@{k}"] = sum(r["metrics"][f"HitRate@{k}"] for r in reason_users) / r_count
                r_rec_10 = set().union(*(r["top_10"] for r in reason_users))
                r_metrics["CatalogCoverage@10"] = (len(r_rec_10) / warm_pool_size) if warm_pool_size > 0 else 0.0

            by_reason[reason] = {"count": r_count, "metrics": r_metrics}

        subgroup_results["fallback"]["by_reason"] = by_reason

        return {
            "split": split,
            "condition": self.context.condition,
            "audit": {
                "total_targets": len(target_records),
                "eligible_users": eligible_count,
                "excluded_targets": len(target_records) - eligible_count,
                "warm_pool_size": warm_pool_size,
                "unknown_users": unknown_users,
                "irrelevant_targets": irrelevant_targets,
                "cold_targets": cold_targets,
                "history_overlaps": history_overlaps,
                "overlapping_reasons": dict(overlapping_reasons),
            },
            "candidate_shortages": shortage_counts,
            "overall_metrics": overall_metrics,
            "subgroups": subgroup_results,
            "per_user_results": per_user_results,
        }


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
