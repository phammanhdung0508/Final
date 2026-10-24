"""EX01: batched validation ranking that reproduces the accepted evaluator report.

`recommendation_core.Evaluator` scores and sorts every candidate per user in Python.
This module computes the same report from one representation pass:

- target parsing, eligibility and exclusion accounting are the accepted rules;
- scores are user-by-item matrix chunks; users without positive history get
  condition-positive training popularity, as the evaluator's own fallback;
- training-rated items are removed before ranking; ranking is score descending, then
  `parent_asin` ascending, via a stable sort over columns in `parent_asin` order;
- non-finite candidate scores raise `ScoringFailureError`;
- metrics and subgroup sums run in the evaluator's user order, so equal scores give an
  identical report.

Only the validation split is supported. Final-test evaluation is not authorized.
"""

import math
from collections import defaultdict

import torch

from scripts.recommendation_core import Evaluator
from scripts.recommendation_evaluation import ScoringFailureError

CUTOFFS = Evaluator.CUTOFFS
METRIC_KEYS = ('NDCG@5', 'NDCG@10', 'NDCG@20', 'HitRate@5', 'HitRate@10', 'HitRate@20',
               'CatalogCoverage@10')


def _parse_targets(targets):
    if isinstance(targets, (str, bytes)):
        raise ValueError('targets must be an iterable collection of target records.')
    records, seen = [], set()
    for rec in targets:
        if isinstance(rec, dict):
            u, p, r = rec.get('user_id'), rec.get('parent_asin'), rec.get('rating')
        elif isinstance(rec, (list, tuple)) and len(rec) >= 3:
            u, p, r = rec[0], rec[1], rec[2]
        else:
            raise ValueError(f'Invalid target record: {rec!r}')
        if not isinstance(u, str) or not u.strip():
            raise ValueError(f'Invalid target user_id: {u!r}. Must be a non-empty string.')
        if not isinstance(p, str) or not p.strip():
            raise ValueError(f'Invalid target parent_asin: {p!r}. Must be a non-empty string.')
        if isinstance(r, bool) or not isinstance(r, (int, float)) or math.isnan(r) or math.isinf(r):
            raise ValueError(f'Invalid target rating: {r!r} for user {u}. Must be a finite number and not boolean.')
        if u in seen:
            raise ValueError(f'Duplicate held-out target for user {u}. Exactly one target per user permitted.')
        seen.add(u)
        records.append((u, p, float(r)))
    records.sort(key=lambda x: x[0])
    return records


def _empty_metrics():
    return {k: None for k in METRIC_KEYS}


def _group_metrics(results, warm_pool_size):
    count = len(results)
    if count == 0:
        return _empty_metrics()
    metrics = {}
    for k in CUTOFFS:
        metrics[f'NDCG@{k}'] = sum(r['metrics'][f'NDCG@{k}'] for r in results) / count
        metrics[f'HitRate@{k}'] = sum(r['metrics'][f'HitRate@{k}'] for r in results) / count
    covered = set().union(*(r['top_10'] for r in results))
    metrics['CatalogCoverage@10'] = (len(covered) / warm_pool_size) if warm_pool_size > 0 else 0.0
    return metrics


def evaluate_validation(context, users, items, user_table, item_table, targets, *,
                        chunk_users=1024):
    """Validation report equal to `Evaluator(context).evaluate(targets, model)`.

    `users` and `items` give the row order of `user_table` and `item_table`; `items` must
    equal `context.sorted_warm_items`.
    """
    if list(items) != list(context.sorted_warm_items):
        raise ValueError('item rows must follow context.sorted_warm_items')
    if user_table.shape[0] != len(users) or item_table.shape[0] != len(items):
        raise ValueError('representation tables do not match the supplied orders')
    device = item_table.device
    u_index = {u: i for i, u in enumerate(users)}
    i_index = {p: i for i, p in enumerate(items)}
    n_items = len(items)
    warm_pool = context.warm_items

    records = _parse_targets(targets)
    counters = defaultdict(int)
    overlapping = defaultdict(int)
    eligible = []
    for u, p, r in records:
        history = context.all_rated_history.get(u, set())
        reasons = []
        if u not in context.users:
            counters['unknown_users'] += 1
            reasons.append('unknown_user')
        if not (context.condition == 'all_ratings' or r >= 4.0):
            counters['irrelevant_targets'] += 1
            reasons.append('irrelevant_target')
        if p not in warm_pool:
            counters['cold_targets'] += 1
            reasons.append('cold_target')
        if p in history:
            counters['history_overlaps'] += 1
            reasons.append('history_overlap')
        if reasons:
            overlapping['+'.join(sorted(reasons))] += 1
        else:
            eligible.append((u, p))

    popularity = torch.tensor([float(context.popularity.get(p, 0)) for p in items],
                              dtype=torch.float32, device=device)
    shortage = {k: 0 for k in CUTOFFS}
    per_user = []
    recommended_at_10 = set()
    for start in range(0, len(eligible), chunk_users):
        chunk = eligible[start:start + chunk_users]
        personalized = [bool(context.positive_history.get(u)) for u, _ in chunk]
        rows = torch.tensor([u_index[u] for u, _ in chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            scores = user_table.index_select(0, rows) @ item_table.T
            fallback = torch.tensor([not x for x in personalized], device=device)
            scores = torch.where(fallback.unsqueeze(1), popularity.unsqueeze(0), scores)
            candidate = torch.ones(len(chunk), n_items, dtype=torch.bool, device=device)
            hist_r, hist_c = [], []
            for j, (u, _) in enumerate(chunk):
                for item in context.all_rated_history.get(u, ()):
                    if item in i_index:
                        hist_r.append(j)
                        hist_c.append(i_index[item])
            if hist_r:
                candidate[torch.tensor(hist_r, device=device), torch.tensor(hist_c, device=device)] = False
            bad = candidate & ~torch.isfinite(scores)
            if bad.any():
                j, col = (int(x) for x in bad.nonzero()[0])
                raise ScoringFailureError(
                    f'Non-finite candidate score {float(scores[j, col])!r} for user {chunk[j][0]}, item {items[col]}.')
            masked = torch.where(candidate, scores, torch.full_like(scores, -math.inf))
            order = torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :max(CUTOFFS)]
            n_candidates = candidate.sum(dim=1).tolist()
            order = order.tolist()
        for j, (u, target) in enumerate(chunk):
            n_cand = n_candidates[j]
            for k in CUTOFFS:
                if n_cand < k:
                    shortage[k] += 1
            ranked = [items[c] for c in order[j][:min(n_cand, max(CUTOFFS))]]
            metrics = {}
            for k in CUTOFFS:
                top = ranked[:k]
                if target in top:
                    metrics[f'HitRate@{k}'] = 1.0
                    metrics[f'NDCG@{k}'] = 1.0 / math.log2(top.index(target) + 2.0)
                else:
                    metrics[f'HitRate@{k}'] = 0.0
                    metrics[f'NDCG@{k}'] = 0.0
            top_5, top_10, top_20 = ranked[:5], ranked[:10], ranked[:20]
            recommended_at_10.update(top_10)
            route = 'personalized' if personalized[j] else 'fallback'
            per_user.append({
                'user_id': u, 'target_item': target, 'route': route,
                'fallback_reason': None if personalized[j] else 'empty_positive_history',
                'candidate_count': n_cand, 'metrics': metrics,
                'top_5': top_5, 'top_10': top_10, 'top_20': top_20,
                'top_k': {5: top_5, 10: top_10, 20: top_20},
            })

    warm_pool_size = len(warm_pool)
    total = len(records)
    n_eligible = len(eligible)
    excluded = total - n_eligible
    if n_eligible == 0:
        overall = _empty_metrics()
    else:
        overall = {}
        for k in CUTOFFS:
            overall[f'NDCG@{k}'] = sum(r['metrics'][f'NDCG@{k}'] for r in per_user) / n_eligible
            overall[f'HitRate@{k}'] = sum(r['metrics'][f'HitRate@{k}'] for r in per_user) / n_eligible
        overall['CatalogCoverage@10'] = (len(recommended_at_10) / warm_pool_size) if warm_pool_size > 0 else 0.0

    subgroups = {}
    for route in ('personalized', 'non_personalized', 'fallback'):
        members = [r for r in per_user if r['route'] == route]
        subgroups[route] = {'count': len(members), 'metrics': _group_metrics(members, warm_pool_size)}
    fallback_users = [r for r in per_user if r['route'] == 'fallback']
    reasons = sorted({r['fallback_reason'] for r in fallback_users if r['fallback_reason'] is not None}
                     | set(Evaluator.DEFAULT_FALLBACK_REASONS))
    subgroups['fallback']['by_reason'] = {
        reason: {'count': len(m), 'metrics': _group_metrics(m, warm_pool_size)}
        for reason in reasons
        for m in [[r for r in fallback_users if r['fallback_reason'] == reason]]
    }

    def share(n):
        return (n / total) if total > 0 else None
    audit_counts = {key: counters[key] for key in
                    ('unknown_users', 'irrelevant_targets', 'cold_targets', 'history_overlaps')}
    return {
        'split': 'validation',
        'condition': context.condition,
        'audit': {
            'total_targets': total, 'eligible_users': n_eligible, 'excluded_targets': excluded,
            'warm_pool_size': warm_pool_size, **audit_counts,
            'proportions': {**{k: share(v) for k, v in audit_counts.items()},
                            'eligible_users': share(n_eligible),
                            'excluded_targets': share(excluded)},
            'overlapping_reasons': dict(overlapping),
            'overlapping_proportions': {k: v / total for k, v in overlapping.items()} if total > 0 else {},
        },
        'candidate_shortages': shortage,
        'overall_metrics': overall,
        'subgroups': subgroups,
        'per_user_results': per_user,
    }


def evaluate_model(model, targets, *, chunk_users=1024):
    """Validation report for a vectorized model, using one unmasked representation pass."""
    graph = model.graph
    items = graph.products if hasattr(graph, 'products') else graph.warm_items
    with torch.no_grad():
        user_table, item_table = model.representations()
    return evaluate_validation(graph.context, graph.users, items, user_table, item_table,
                               targets, chunk_users=chunk_users)
