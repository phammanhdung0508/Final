"""Amazon4U shared recommendation evaluation subsystem.

Python standard library only.
Provides:
- ScoringFailureError: raised when candidate scoring fails, returns non-finite values,
  produces corrupt outputs, or makes invalid route/fallback claims.
- TrainingContext: validated training-only context, condition-specific warmth and popularity,
  distinct positive vs all-rated history, duplicate/invalid input failures, held-out-independent.
- Evaluator: target eligibility and exclusion accounting (unknown, cold, irrelevant, overlap),
  history filtering (validation vs test), deterministic ranking (score desc, parent_asin asc),
  shortage reporting, non-finite score rejection, no padding, registered metrics
  (NDCG@K, HitRate@K for K in (5, 10, 20), CatalogCoverage@10), route and fallback subgroup reporting.

Note: Baseline score providers (such as PopularityScorer), metadata-specific scoring,
KG-only mixtures, neighbor sampling, learned GNN models, and optimizer loss/regularization
are intentionally excluded from this module and remain in separate subsystems.
Evaluator-owned fallback for empty positive history uses condition-positive training popularity.
"""

from collections import defaultdict
from collections.abc import Mapping
import math
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Canonicalize module in sys.modules so scripts.recommendation_evaluation and recommendation_evaluation
# share the exact same module object and class identities.
if __name__ == "scripts.recommendation_evaluation":
    sys.modules.setdefault("recommendation_evaluation", sys.modules[__name__])
elif __name__ == "recommendation_evaluation":
    sys.modules.setdefault("scripts.recommendation_evaluation", sys.modules[__name__])
    if "scripts" in sys.modules:
        setattr(sys.modules["scripts"], "recommendation_evaluation", sys.modules[__name__])


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
            elif isinstance(record, (list, tuple)) and len(record) >= 3:
                user_id, parent_asin, rating = record[0], record[1], record[2]
            else:
                raise ValueError(f"Invalid interaction record: {record!r}")

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

            is_positive = True if condition == "all_ratings" else (float(rating) >= 4.0)
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

class Evaluator:
    """Reusable evaluator enforcing history filtering, target auditing, ranking, metrics, and subgroup reporting."""

    CUTOFFS = (5, 10, 20)
    DEFAULT_FALLBACK_REASONS: Tuple[str, ...] = ("empty_positive_history",)

    def __init__(self, context: TrainingContext) -> None:
        self.context = context

    def _get_default_fallback_reasons(self) -> Set[str]:
        """Return default fallback reasons to always include in report schema."""
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

        In this shared evaluator milestone, all provider fallback claims are rejected;
        evaluator owns fallback for empty positive history.
        Subclasses may override this hook to permit strictly validated provider fallback.
        """
        if not isinstance(scores, dict):
            raise ScoringFailureError(f"Scorer must return a dictionary of candidate scores for user {user_id}.")

        if not isinstance(route, str) or route not in ("personalized", "non_personalized"):
            raise ScoringFailureError(f"Invalid route {route!r} for user {user_id}. In this evaluator milestone, provider must return only 'personalized' or 'non_personalized'.")

        if reason is not None:
            raise ScoringFailureError(f"Provider returned fallback_reason {reason!r} for user {user_id}. Evaluator owns fallback for empty positive history; all provider fallback claims are rejected.")

        return scores, route, reason

    def evaluate(
        self,
        targets: Iterable[Union[Tuple[str, str, float], Dict[str, Any]]],
        scorer: Any,
        split: str = "validation",
        validation_history: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate recommendations for held-out targets.

        Accepted validation_history formats on test split:
        1. Mapping from user_id to collection (e.g. list, set, or iterable) of item IDs
           (strings), tuples (item_id[, rating]), or record dicts (parent_asin[, rating[, user_id]]).
           Mapping-valued user histories (e.g. {"u1": {"parent_asin": ...}}) are rejected.
           If user_id is present in a record dict, it must match the outer user key.
        2. Iterable of record tuples (user_id, item_id[, rating]) or record dicts
           (user_id, parent_asin[, rating]).
        In all formats, ratings are optional (identifier-only entries are valid). If a rating is
        explicitly provided, it cannot be None and must be a finite non-boolean number.
        Every known target user in test split requires at least one actual supplied validation ID;
        empty collections or exhausted generators for known targets are rejected.
        """
        if split not in ("validation", "test"):
            raise ValueError(f"Invalid split: {split}. Must be 'validation' or 'test'.")

        val_history_map: Dict[str, Set[str]] = defaultdict(set)
        seen_val_pairs: Set[Tuple[str, str]] = set()

        if split == "test":
            if validation_history is None:
                raise ValueError("validation_history must be explicitly provided for test split.")
            if isinstance(validation_history, (str, bytes)):
                raise ValueError("validation_history must be a dictionary or iterable collection of records, not a string.")

            if isinstance(validation_history, Mapping):
                for u, items in validation_history.items():
                    if not isinstance(u, str) or not u.strip():
                        raise ValueError(f"Invalid user_id in validation_history: {u!r}. Must be a non-empty string.")
                    if isinstance(items, Mapping):
                        raise ValueError(f"Validation history for user {u} cannot be a mapping/dict; records must be inside a list or collection: {items!r}")
                    if isinstance(items, (str, bytes)):
                        raise ValueError(f"Validation history items for user {u} must be an item collection, not a string.")
                    if not isinstance(items, Iterable):
                        raise ValueError(f"Validation history items for user {u} must be an iterable collection of item IDs or records.")

                    for item in items:
                        if isinstance(item, Mapping):
                            if "user_id" in item and item["user_id"] != u:
                                raise ValueError(f"Conflicting user_id {item['user_id']!r} in validation record for user {u}.")
                            item_id = item.get("parent_asin")
                            if "rating" in item:
                                r = item["rating"]
                                if r is None:
                                    raise ValueError(f"Validation record rating explicitly supplied as None for user {u}: {item!r}")
                                if isinstance(r, bool) or not isinstance(r, (int, float)) or math.isnan(r) or math.isinf(r):
                                    raise ValueError(f"Invalid rating {r!r} in validation_history for user {u}.")
                            else:
                                r = None
                        elif isinstance(item, (list, tuple)):
                            if len(item) == 0:
                                raise ValueError(f"Empty record tuple in validation_history for user {u}.")
                            item_id = item[0]
                            if len(item) >= 2:
                                r = item[1]
                                if r is None:
                                    raise ValueError(f"Validation record rating explicitly supplied as None for user {u}: {item!r}")
                                if isinstance(r, bool) or not isinstance(r, (int, float)) or math.isnan(r) or math.isinf(r):
                                    raise ValueError(f"Invalid rating {r!r} in validation_history for user {u}.")
                            else:
                                r = None
                        elif isinstance(item, str):
                            item_id = item
                            r = None
                        else:
                            raise ValueError(f"Invalid item entry in validation_history for user {u}: {item!r}")

                        if not isinstance(item_id, str) or not item_id.strip():
                            raise ValueError(f"Invalid item ID {item_id!r} in validation_history for user {u}.")
                        pair = (u, item_id)
                        if pair in seen_val_pairs:
                            raise ValueError(f"Duplicate validation record detected: {pair}")
                        seen_val_pairs.add(pair)
                        val_history_map[u].add(item_id)
            elif isinstance(validation_history, Iterable):
                for rec in validation_history:
                    if isinstance(rec, Mapping):
                        u = rec.get("user_id")
                        item_id = rec.get("parent_asin")
                        if "rating" in rec:
                            r = rec["rating"]
                            if r is None:
                                raise ValueError(f"Validation record rating explicitly supplied as None in record: {rec!r}")
                            if isinstance(r, bool) or not isinstance(r, (int, float)) or math.isnan(r) or math.isinf(r):
                                raise ValueError(f"Invalid rating {r!r} in validation record: {rec!r}")
                        else:
                            r = None
                    elif isinstance(rec, (list, tuple)):
                        if len(rec) < 2:
                            raise ValueError(f"Invalid validation_history record: {rec!r}. Must contain at least (user_id, item_id).")
                        u, item_id = rec[0], rec[1]
                        if len(rec) >= 3:
                            r = rec[2]
                            if r is None:
                                raise ValueError(f"Validation record rating explicitly supplied as None in record: {rec!r}")
                            if isinstance(r, bool) or not isinstance(r, (int, float)) or math.isnan(r) or math.isinf(r):
                                raise ValueError(f"Invalid rating {r!r} in validation record: {rec!r}")
                        else:
                            r = None
                    else:
                        raise ValueError(f"Invalid validation_history record: {rec!r}")

                    if not isinstance(u, str) or not u.strip():
                        raise ValueError(f"Invalid user_id {u!r} in validation_history record: {rec!r}")
                    if not isinstance(item_id, str) or not item_id.strip():
                        raise ValueError(f"Invalid item ID {item_id!r} in validation_history record: {rec!r}")

                    pair = (u, item_id)
                    if pair in seen_val_pairs:
                        raise ValueError(f"Duplicate validation record detected: {pair}")
                    seen_val_pairs.add(pair)
                    val_history_map[u].add(item_id)
            else:
                raise ValueError("validation_history must be a dictionary or iterable collection of records.")

        target_records: List[Tuple[str, str, float]] = []
        seen_target_users: Set[str] = set()

        if isinstance(targets, (str, bytes)) or not isinstance(targets, Iterable):
            raise ValueError("targets must be an iterable collection of target records.")

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
            if split == "test" and (u in self.context.users) and (len(val_history_map.get(u, set())) == 0):
                raise ValueError(f"Missing validation history for known target user {u} in test split (every known target user requires at least one supplied validation ID).")

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
        total_targets = len(target_records)
        excluded_targets = total_targets - eligible_count
        shortage_counts = {k: 0 for k in self.CUTOFFS}

        per_user_results: List[Dict[str, Any]] = []
        recommended_at_10: Set[str] = set()

        for u, target_item, rating, filtered_history in eligible_records:
            authoritative_candidates = tuple(
                item for item in self.context.sorted_warm_items if item not in filtered_history
            )

            for k in self.CUTOFFS:
                if len(authoritative_candidates) < k:
                    shortage_counts[k] += 1

            has_positive_history = len(self.context.positive_history.get(u, set())) > 0

            if not has_positive_history:
                # Shared evaluator-owned fallback for empty positive training history
                route = "fallback"
                fallback_reason = "empty_positive_history"
                scores = {item: float(self.context.popularity.get(item, 0)) for item in authoritative_candidates}
            else:
                provider_input = list(authoritative_candidates)
                try:
                    if hasattr(scorer, "score_candidates"):
                        res = scorer.score_candidates(u, provider_input)
                    elif callable(scorer):
                        res = scorer(u, provider_input)
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

                scores, route, fallback_reason = self._validate_provider_result(
                    scorer, u, authoritative_candidates, scores, route, fallback_reason
                )

            # Validate that every candidate has a finite numeric score
            for item in authoritative_candidates:
                if item not in scores:
                    raise ScoringFailureError(f"Scorer missing candidate score for item {item} for user {u}.")
                s = scores[item]
                if isinstance(s, bool) or not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s):
                    raise ScoringFailureError(f"Non-finite candidate score {s!r} for user {u}, item {item}.")

            ranked_items = sorted(authoritative_candidates, key=lambda item: (-scores[item], item))

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
                "candidate_count": len(authoritative_candidates),
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
        observed_reasons = sorted(
            set(r["fallback_reason"] for r in fallback_users if r["fallback_reason"] is not None)
            | self._get_default_fallback_reasons()
        )
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
                "total_targets": total_targets,
                "eligible_users": eligible_count,
                "excluded_targets": excluded_targets,
                "warm_pool_size": warm_pool_size,
                "unknown_users": unknown_users,
                "irrelevant_targets": irrelevant_targets,
                "cold_targets": cold_targets,
                "history_overlaps": history_overlaps,
                "proportions": {
                    "unknown_users": (unknown_users / total_targets) if total_targets > 0 else None,
                    "irrelevant_targets": (irrelevant_targets / total_targets) if total_targets > 0 else None,
                    "cold_targets": (cold_targets / total_targets) if total_targets > 0 else None,
                    "history_overlaps": (history_overlaps / total_targets) if total_targets > 0 else None,
                    "eligible_users": (eligible_count / total_targets) if total_targets > 0 else None,
                    "excluded_targets": (excluded_targets / total_targets) if total_targets > 0 else None,
                },
                "overlapping_reasons": dict(overlapping_reasons),
                "overlapping_proportions": {
                    k: (v / total_targets) for k, v in overlapping_reasons.items()
                } if total_targets > 0 else {},
            },
            "candidate_shortages": shortage_counts,
            "overall_metrics": overall_metrics,
            "subgroups": subgroup_results,
            "per_user_results": per_user_results,
        }
