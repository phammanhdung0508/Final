"""Post-training experiment script for HeteroGraphSAGE recommendation.

This is the primary file modified by automated research agents.
Implements 4-fold disjoint cross-supervision, dynamic contrastive candidate sampling
(DNS with popularity weighting), hardest-negative BCE anchoring, and normalized
InfoNCE directional regularization to maximize held-out NDCG@10.
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
from prepare import load_dataset, GraphDataset
from evaluate import ranking_metrics


def sample_candidates(
    users: np.ndarray,
    num_movies: int,
    observed: dict[int, set[int]],
    pop_probs: np.ndarray,
    rng: np.random.Generator,
    num_candidates: int = 3,
) -> np.ndarray:
    """Sample K negative candidate movies per user (uniform + popularity-weighted)."""
    n_pairs = len(users)
    if num_candidates >= 3:
        n_pop = 1
        n_uni = num_candidates - n_pop
        c_uni = rng.integers(0, num_movies, size=(n_pairs, n_uni))
        c_pop = rng.choice(num_movies, size=(n_pairs, n_pop), p=pop_probs)
        candidates = np.hstack([c_uni, c_pop])
    else:
        candidates = rng.integers(0, num_movies, size=(n_pairs, num_candidates))

    # Fast collision resolution against observed training movies
    for i in range(n_pairs):
        u = int(users[i])
        obs = observed[u]
        for k in range(num_candidates):
            while candidates[i, k] in obs:
                candidates[i, k] = int(rng.integers(0, num_movies))

    return candidates.astype(np.int64)


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
    torch.set_num_threads(int(config.get("cpu_threads", 4)))

    dataset = load_dataset()
    positives = dataset.positive_train[rng.permutation(len(dataset.positive_train))]

    # Popularity distribution for candidate sampling
    movie_counts = np.bincount(dataset.positive_train[:, 1], minlength=len(dataset.movies))
    weights = np.maximum(movie_counts, 1).astype(np.float64) ** 0.75
    pop_probs = weights / weights.sum()

    k_folds = int(config.get("k_folds", 4))
    if k_folds > 1:
        num_pos = len(positives)
        fold_indices = np.array_split(np.arange(num_pos), k_folds)
        fold_graphs = []
        fold_supervisions = []
        for i in range(k_folds):
            sup_idx = fold_indices[i]
            msg_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
            fold_supervisions.append(positives[sup_idx])
            fold_graphs.append(dataset.graph(positives[msg_idx]).to(device))
    else:
        cut = max(1, int(len(positives) * float(config.get("supervision_ratio", 0.2))))
        fold_supervisions = [positives[:cut]]
        fold_graphs = [dataset.graph(positives[cut:]).to(device)]

    serving_graph = dataset.graph().to(device)

    epochs = int(config.get("epochs", 65))
    eval_interval = int(config.get("eval_interval", 2))
    top_k = int(config.get("top_k", 10))
    num_candidates = int(config.get("num_candidates", 3))
    tau = float(config.get("tau", 0.15))
    contrastive_alpha = float(config.get("contrastive_alpha", 0.35))

    pretrained_path = config.get("pretrained_model_path")
    if pretrained_path:
        # Fallback logic for Kaggle paths
        if not Path(pretrained_path).exists() and pretrained_path.startswith("/kaggle/input"):
            alt_path = pretrained_path.replace("-", "_")
            if Path(alt_path).exists():
                pretrained_path = alt_path
            else:
                import glob
                print("Looking for pt files in /kaggle/input:", glob.glob("/kaggle/input/**/*.pt", recursive=True))
                pts = glob.glob("/kaggle/input/**/*.pt", recursive=True)
                if pts:
                    pretrained_path = pts[0]
        
        print(f">> Loading pre-trained weights from {pretrained_path}")
        import importlib
        model_module = importlib.import_module(f"models.model_{config['model_type']}")
        model, _ = model_module.load_model(pretrained_path, dataset, device=device)
        save_model = model_module.save_model
    else:
        import importlib
        model_module = importlib.import_module(f"models.model_{config['model_type']}")
        model = model_module.create_model(dataset, config).to(device)
        save_model = model_module.save_model
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.003)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    milestones = list(config.get("milestones", [25, 45, 58]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )

    best_val_ndcg = -1.0
    best_metrics: dict[str, float] = {}
    history: list[dict[str, object]] = []

    start_time = time.time()
    last_loss = 0.0

    print(
        f">> Starting training: {epochs} epochs on {device} (seed={seed}, k_folds={k_folds}, "
        f"candidates={num_candidates}, tau={tau}, alpha={contrastive_alpha})"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        for fold_idx in range(len(fold_supervisions)):
            sup = fold_supervisions[fold_idx]
            tr_graph = fold_graphs[fold_idx]
            u_indices = sup[:, 0]
            p_indices = sup[:, 1]
            n_samples = len(sup)

            # Sample candidates (uniform + popularity-biased)
            candidates = sample_candidates(
                u_indices, len(dataset.movies), dataset.observed, pop_probs, rng, num_candidates=num_candidates
            )

            embeddings = model(tr_graph)
            u_emb = embeddings["user"][torch.from_numpy(u_indices).to(device)]
            p_emb = embeddings["movie"][torch.from_numpy(p_indices).to(device)]
            c_emb = embeddings["movie"][torch.from_numpy(candidates).to(device)]

            # 1. Raw dot-products for BCE
            s_pos = (u_emb * p_emb).sum(-1)
            s_cand = (u_emb.unsqueeze(1) * c_emb).sum(-1)

            # Hardest negative candidate mining
            hardest_idx = s_cand.argmax(dim=-1)
            s_neg_hard = s_cand[torch.arange(n_samples, device=device), hardest_idx]

            loss_bce_pos = F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))
            loss_bce_neg = F.binary_cross_entropy_with_logits(s_neg_hard, torch.zeros_like(s_neg_hard))
            loss_bce = 0.5 * (loss_bce_pos + loss_bce_neg)

            # 2. Normalized cosine similarity for contrastive InfoNCE
            if contrastive_alpha > 0.0:
                u_norm = F.normalize(u_emb, p=2, dim=-1)
                p_norm = F.normalize(p_emb, p=2, dim=-1)
                c_norm = F.normalize(c_emb, p=2, dim=-1)

                cos_pos = (u_norm * p_norm).sum(-1)
                cos_cand = (u_norm.unsqueeze(1) * c_norm).sum(-1)

                all_cos = torch.cat([cos_pos.unsqueeze(1), cos_cand], dim=1) / tau
                labels = torch.zeros(n_samples, dtype=torch.long, device=device)
                loss_infonce = F.cross_entropy(all_cos, labels)

                loss = loss_bce + contrastive_alpha * loss_infonce
            else:
                loss = loss_bce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_epoch_loss += float(loss.item())

        scheduler.step()
        last_loss = total_epoch_loss / len(fold_supervisions)

        # Validation check
        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                x = model(serving_graph)
                user_emb = x["user"].detach().cpu()
                movie_emb = x["movie"].detach().cpu()
                scores = (user_emb @ movie_emb.T).numpy()

            metrics = ranking_metrics(scores, dataset, split="validation", k=top_k)
            val_ndcg = metrics[f"ndcg@{top_k}"]
            record = {"epoch": epoch, "loss": last_loss, **metrics}
            history.append(record)

            print(
                f"Epoch {epoch:3d}/{epochs} | Loss: {last_loss:.4f} | "
                f"Val NDCG@{top_k}: {val_ndcg:.5f} | HitRate@{top_k}: {metrics[f'hit_rate@{top_k}']:.4f}"
            )

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_metrics = dict(metrics)
                save_model(
                    model,
                    out_dir,
                    metadata={
                        "config": config,
                        "epoch": epoch,
                        "best_validation_ndcg": best_val_ndcg,
                        "dataset_fingerprint": dataset.fingerprint,
                        "user_ids": dataset.users.userId.tolist(),
                        "movie_ids": dataset.movies.movieId.tolist(),
                    },
                )

    training_seconds = time.time() - start_time
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    print("\n--- Training Completed ---")
    print(f"training_seconds: {training_seconds:.2f}")
    print(f"train_loss: {last_loss:.5f}")
    print(f"eval_score: {best_val_ndcg:.5f}")
    print(f"ndcg_{top_k}: {best_val_ndcg:.5f}")
    if best_metrics:
        print(f"hit_rate_{top_k}: {best_metrics.get(f'hit_rate@{top_k}', 0.0):.5f}")
        print(f"recall_{top_k}: {best_metrics.get(f'recall@{top_k}', 0.0):.5f}")
        print(f"precision_{top_k}: {best_metrics.get(f'precision@{top_k}', 0.0):.5f}")

    return {
        "eval_score": best_val_ndcg,
        "train_loss": last_loss,
        "training_seconds": training_seconds,
        **best_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeteroGraphSAGE for recommendation post-training.")
    parser.add_argument("--epochs", type=int, default=65, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Hidden embedding dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN message passing layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--supervision-ratio", type=float, default=0.2, help="Fraction of positive edges held out for supervision (if k_folds=1)")
    parser.add_argument("--k-folds", type=int, default=4, help="Number of disjoint cross-supervision folds")
    parser.add_argument("--num-candidates", type=int, default=3, help="Number of negative candidates per interaction")
    parser.add_argument("--tau", type=float, default=0.15, help="InfoNCE temperature")
    parser.add_argument("--contrastive-alpha", type=float, default=0.35, help="InfoNCE loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("final_model"), help="Output directory for saved model")
    parser.add_argument("--model-type", type=str, choices=["kgat", "han"], default="kgat", help="Model architecture")
    parser.add_argument("--device", default="auto", help="Compute device (auto, cpu, cuda)")
    parser.add_argument("--pretrained-model-path", type=Path, default=None, help="Path to pre-trained weights")
    args = parser.parse_args()

    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "supervision_ratio": args.supervision_ratio,
        "k_folds": args.k_folds,
        "num_candidates": args.num_candidates,
        "tau": args.tau,
        "contrastive_alpha": args.contrastive_alpha,
        "milestones": [60, 120, 160] if args.epochs >= 160 else [25, 45, 58],
        "seed": args.seed,
        "device": "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device),
        "model_type": args.model_type,
        "pretrained_model_path": str(args.pretrained_model_path) if args.pretrained_model_path else None,
    }

    train(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
