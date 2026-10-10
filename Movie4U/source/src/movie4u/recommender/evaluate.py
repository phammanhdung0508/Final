"""Full-catalog macro-averaged ranking metrics; no sampled test negatives."""

import numpy as np


def top_indices(scores, observed, k):
    values = np.asarray(scores, dtype=np.float64).copy()
    if observed:
        values[list(observed)] = -np.inf
    candidates = np.flatnonzero(np.isfinite(values))
    return candidates[np.lexsort((candidates, -values[candidates]))[:k]]


def ranking_metrics(scores, dataset, split="test", k=10):
    relevance = dataset.relevant(split)
    precision, recall, ndcg, hitrate, recommended = [], [], [], [], set()
    for user, truth in relevance.items():
        ranked = top_indices(scores[user], dataset.observed[user], k)
        hits = np.array([m in truth for m in ranked], dtype=float)
        precision.append(hits.sum() / k)
        recall.append(hits.sum() / len(truth))
        dcg = (hits / np.log2(np.arange(len(hits)) + 2)).sum()
        ideal = (1 / np.log2(np.arange(min(k, len(truth))) + 2)).sum()
        ndcg.append(dcg / ideal)
        hitrate.append(float(hits.sum() > 0))
        recommended.update(ranked.tolist())
    if not relevance:
        raise ValueError("No users with positive held-out interactions")
    return {
        f"precision@{k}": float(np.mean(precision)),
        f"recall@{k}": float(np.mean(recall)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"hit_rate@{k}": float(np.mean(hitrate)),
        "coverage": len(recommended) / len(dataset.movies),
        "evaluated_users": len(relevance),
        "excluded_users_without_positive_holdout": len(dataset.users) - len(relevance),
    }
