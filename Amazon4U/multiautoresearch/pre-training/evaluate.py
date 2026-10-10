#!/usr/bin/env python3
"""Fixed benchmark evaluator for pre-training graph link prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import load_model
from prepare import load_dataset, GraphPretrainDataset


def evaluate_link_prediction(
    model,
    dataset: GraphPretrainDataset,
    device: torch.device,
    split: str = "validation",
    num_negatives_per_pos: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos_edges = dataset.val_genre_edges if split == "validation" else dataset.test_genre_edges

    with torch.no_grad():
        graph = dataset.graph().to(device)
        embeddings = model(graph)

        pos_pairs = torch.from_numpy(pos_edges).to(device)
        pos_logits = model.decode_link(embeddings, "movie", "genre", pos_pairs)

        # Sample negatives
        neg_pairs = torch.from_numpy(
            dataset.sample_negative_genre_edges(len(pos_edges) * num_negatives_per_pos, rng)
        ).to(device)
        neg_logits = model.decode_link(embeddings, "movie", "genre", neg_pairs)

        # BCE Loss
        all_logits = torch.cat([pos_logits, neg_logits])
        all_labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ])
        val_loss = float(F.binary_cross_entropy_with_logits(all_logits, all_labels).item())

        # Ranking MRR and Hits@10
        pos_scores = pos_logits.cpu().numpy()
        neg_scores = neg_logits.cpu().numpy().reshape(len(pos_edges), num_negatives_per_pos)

        ranks = []
        hits10 = []
        for p, n in zip(pos_scores, neg_scores):
            # Rank of positive among 1 pos + num_negatives
            rank = int(np.sum(n >= p)) + 1
            ranks.append(1.0 / rank)
            hits10.append(float(rank <= 10))

        mrr = float(np.mean(ranks))
        hits_10 = float(np.mean(hits10))

    metrics = {
        "val_bpb": val_loss,
        "val_loss": val_loss,
        "link_mrr": mrr,
        "link_hits@10": hits_10,
        "num_examples": len(pos_edges),
    }

    print("--- Pre-Training Evaluation Summary ---")
    print(f"val_bpb: {val_loss:.6f}")
    print(f"val_loss: {val_loss:.6f}")
    print(f"link_mrr: {mrr:.5f}")
    print(f"link_hits_10: {hits_10:.5f}")
    print(f"num_examples: {len(pos_edges)}")
    sys.stdout.flush()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-trained model on link prediction.")
    parser.add_argument("--model-path", type=Path, default=Path("final_model"), help="Path to checkpoint")
    parser.add_argument("--split", choices=["validation", "test"], default="validation", help="Evaluation split")
    parser.add_argument("--device", default="auto", help="Compute device (auto, cpu, cuda)")
    parser.add_argument("--json-output-file", type=Path, default=None, help="Save metrics to JSON file")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    dataset = load_dataset()
    model, _ = load_model(args.model_path, dataset, device=device)
    model.eval()

    metrics = evaluate_link_prediction(model, dataset, device, split=args.split)
    if args.json_output_file:
        args.json_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_file.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
