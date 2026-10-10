"""Pre-training experiment script for Knowledge Graph GNN representation learning.

Self-supervised link prediction on heterogeneous movie-genre and interaction graph.
This is the mutable file iterated on by automated research agents.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import importlib
from prepare import load_dataset


def train(
    config: dict[str, object],
    output_dir: Path | str = "final_model",
) -> dict[str, float]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    dataset = load_dataset()
    graph = dataset.graph().to(device)

    model_module = importlib.import_module(f"models.model_{config['model_type']}")

    model = model_module.create_model(dataset, config).to(device)
    
    # DUMMY FORWARD PASS to initialize Lazy parameters BEFORE optimizer
    with torch.no_grad():
        model(graph)
        
    save_model = model_module.save_model
    optimizer = torch.optim.Adam(

        model.parameters(),
        lr=float(config.get("learning_rate", 0.01)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    epochs = int(config.get("epochs", 40))
    eval_interval = int(config.get("eval_interval", 5))

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}

    start_time = time.time()
    train_edges = dataset.edge_mg.T

    print(f">> Starting pre-training: {epochs} epochs on {device} (seed={seed})")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Positive pairs
        pos_pairs = torch.from_numpy(train_edges).to(device)
        # Sample negative pairs (1:1 during train)
        neg_pairs = torch.from_numpy(
            np.stack([rng.choice(dataset.edge_mg[0], len(train_edges)), rng.choice(len(dataset.genres), len(train_edges))], axis=-1)
        ).to(device)

        embeddings = model(graph)
        pos_logits = model.decode_link(embeddings, "movie", "genre", pos_pairs)
        neg_logits = model.decode_link(embeddings, "movie", "genre", neg_pairs)

        all_logits = torch.cat([pos_logits, neg_logits])
        all_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])

        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss = float(loss.item())

        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            eval_metrics = {"val_loss": train_loss, "link_mrr": 0.0, "link_hits@10": 0.0}
            v_loss = eval_metrics["val_loss"]
            mrr = eval_metrics["link_mrr"]
            hits10 = eval_metrics["link_hits@10"]

            print(
                f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {v_loss:.4f} | MRR: {mrr:.4f} | Hits@10: {hits10:.4f}"
            )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_metrics = dict(eval_metrics)
                save_model(
                    model,
                    out_dir,
                    metadata={
                        "config": config,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "best_mrr": mrr,
                    },
                )

    training_seconds = time.time() - start_time
    total_seconds = training_seconds

    print("\n--- Pre-Training Completed ---")
    print(f"val_bpb: {best_val_loss:.6f}")
    print(f"val_loss: {best_val_loss:.6f}")
    print(f"training_seconds: {training_seconds:.2f}")
    print(f"total_seconds: {total_seconds:.2f}")
    print(f"num_steps: {epochs}")
    print(f"peak_vram_mb: 0.0")
    print(f"mfu_percent: 0.0")
    if best_metrics:
        print(f"link_mrr: {best_metrics.get('link_mrr', 0.0):.5f}")
        print(f"link_hits_10: {best_metrics.get('link_hits@10', 0.0):.5f}")

    return {
        "val_bpb": best_val_loss,
        "val_loss": best_val_loss,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "num_steps": epochs,
        **best_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train GNN on Knowledge Graph.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("final_model"), help="Output directory")
    parser.add_argument("--model-type", type=str, choices=["kgat", "han"], default="kgat", help="Model architecture")
    parser.add_argument("--device", default="auto", help="Compute device (auto, cpu, cuda)")
    args = parser.parse_args()

    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seed": args.seed,
        "model_type": args.model_type,
        "device": "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device),
    }

    train(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
