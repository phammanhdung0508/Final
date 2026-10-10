#!/usr/bin/env python3
"""Fixed benchmark evaluator for post-training recommendation models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.model_kgat import load_model
from prepare import load_dataset, GraphDataset


def top_indices(scores: np.ndarray, observed: set[int], k: int) -> np.ndarray:
    """Return top-k candidate movie indices excluding observed training movies."""
    values = np.asarray(scores, dtype=np.float64).copy()
    if observed:
        values[list(observed)] = -np.inf
    candidates = np.flatnonzero(np.isfinite(values))
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)
    return candidates[np.lexsort((candidates, -values[candidates]))[:k]]


def ranking_metrics(
    scores: np.ndarray,
    dataset: GraphDataset,
    split: str = "validation",
    k: int = 10,
) -> dict[str, float]:
    """Calculate macro-averaged full-catalog ranking metrics."""
    relevance = dataset.relevant(split)
    if not relevance:
        raise ValueError(f"No users with positive held-out interactions in split '{split}'")

    precision, recall, ndcg, hitrate = [], [], [], []
    recommended: set[int] = set()

    for user, truth in relevance.items():
        ranked = top_indices(scores[user], dataset.observed[user], k)
        hits = np.array([m in truth for m in ranked], dtype=float)
        precision.append(hits.sum() / k)
        recall.append(hits.sum() / len(truth))
        dcg = (hits / np.log2(np.arange(len(hits)) + 2)).sum()
        ideal = (1.0 / np.log2(np.arange(min(k, len(truth))) + 2)).sum()
        ndcg.append(dcg / ideal)
        hitrate.append(float(hits.sum() > 0))
        recommended.update(ranked.tolist())

    eval_score = float(np.mean(ndcg))
    return {
        "eval_score": eval_score,
        f"ndcg@{k}": eval_score,
        f"hit_rate@{k}": float(np.mean(hitrate)),
        f"recall@{k}": float(np.mean(recall)),
        f"precision@{k}": float(np.mean(precision)),
        "coverage": len(recommended) / len(dataset.movies),
        "evaluated_users": len(relevance),
        "excluded_users_without_positive_holdout": len(dataset.users) - len(relevance),
    }


def evaluate(
    model_path: Path | str = "final_model",
    split: str = "validation",
    k: int = 10,
    device: str = "cpu",
) -> dict[str, float]:
    """Load model and dataset, compute full-catalog ranking metrics, print summary."""
    dataset = load_dataset()
    torch_device = torch.device(device)
    model, _ = load_model(model_path, dataset, device=torch_device)
    model.eval()

    with torch.no_grad():
        graph = dataset.graph().to(torch_device)
        embeddings = model(graph)
        user_emb = embeddings["user"].detach().cpu()
        movie_emb = embeddings["movie"].detach().cpu()
        scores = (user_emb @ movie_emb.T).numpy()

    metrics = ranking_metrics(scores, dataset, split=split, k=k)

    print("--- Benchmark Evaluation Summary ---")
    print(f"eval_score: {metrics['eval_score']:.5f}")
    print(f"ndcg_{k}: {metrics[f'ndcg@{k}']:.5f}")
    print(f"hit_rate_{k}: {metrics[f'hit_rate@{k}']:.5f}")
    print(f"recall_{k}: {metrics[f'recall@{k}']:.5f}")
    print(f"precision_{k}: {metrics[f'precision@{k}']:.5f}")
    print(f"coverage: {metrics['coverage']:.5f}")
    print(f"num_correct: {int(round(metrics[f'hit_rate@{k}'] * metrics['evaluated_users']))}")
    print(f"num_examples: {metrics['evaluated_users']}")
    sys.stdout.flush()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommendation model on held-out split.")
    parser.add_argument("--model-path", type=Path, default=Path("final_model"), help="Path to model directory or gnn.pt")
    parser.add_argument("--split", choices=["validation", "test"], default="validation", help="Evaluation split")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff")
    parser.add_argument("--device", default="cpu", help="Computation device (cpu or cuda)")
    parser.add_argument("--json-output-file", type=Path, default=None, help="Save metrics to JSON file")
    args = parser.parse_args()

    metrics = evaluate(
        model_path=args.model_path,
        split=args.split,
        k=args.top_k,
        device=args.device,
    )

    if args.json_output_file:
        args.json_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_file.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
