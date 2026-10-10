"""Reproducible full-batch CPU training and validation checkpoint selection."""

import json
import random
from pathlib import Path
import numpy as np
import torch
from .model import create_model
from .evaluate import ranking_metrics


def sample_negatives(users, num_movies, observed, rng):
    result = []
    for user in users:
        if len(observed[int(user)]) >= num_movies:
            raise ValueError("User has no unobserved movies")
        movie = int(rng.integers(num_movies))
        while movie in observed[int(user)]:
            movie = int(rng.integers(num_movies))
        result.append([int(user), movie])
    return np.array(result, dtype=np.int64)


def train(dataset, config, output_dir):
    if config["epochs"] < 1:
        raise ValueError("epochs must be positive")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    positives = dataset.positive_train[rng.permutation(len(dataset.positive_train))]
    cut = max(1, int(len(positives) * config.get("supervision_ratio", 0.2)))
    supervision, message = positives[:cut], positives[cut:]
    if len(message) == 0:
        raise ValueError("Insufficient positive edges for disjoint training")
    graph, serving_graph = dataset.graph(message), dataset.graph()
    model = create_model(dataset, config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        negatives = sample_negatives(
            supervision[:, 0], len(dataset.movies), dataset.observed, rng
        )
        pairs = torch.from_numpy(np.concatenate([supervision, negatives]))
        labels = torch.cat([torch.ones(len(supervision)), torch.zeros(len(negatives))])
        optimizer.zero_grad()
        logits = model.decode(model(graph), pairs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == config["epochs"]:
            model.eval()
            with torch.no_grad():
                x = model(serving_graph)
                scores = (x["user"] @ x["movie"].T).numpy()
            metrics = ranking_metrics(scores, dataset, "validation", config["top_k"])
            record = {"epoch": epoch, "loss": float(loss.detach()), **metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            current = metrics[f"ndcg@{config['top_k']}"]
            if current > best:
                best = current
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "dataset_fingerprint": dataset.fingerprint,
                        "config": config,
                        "epoch": epoch,
                        "validation_ndcg": best,
                        "user_ids": dataset.users.userId.tolist(),
                        "movie_ids": dataset.movies.movieId.tolist(),
                    },
                    out / "gnn.pt",
                )
    (out / "training_history.json").write_text(json.dumps(history, indent=2))
    return history
