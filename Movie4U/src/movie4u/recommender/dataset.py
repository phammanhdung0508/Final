"""Shared chronological splits and a training-only heterogeneous graph."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


class GraphDataset:
    def __init__(self, graph_dir: str, split_dir: str, threshold: float = 4.0):
        self.graph_dir, self.split_dir = Path(graph_dir), Path(split_dir)
        self.movies = pd.read_csv(self.graph_dir / "movies.csv").sort_values("index")
        self.users = pd.read_csv(self.graph_dir / "users.csv").sort_values("index")
        self.genres = pd.read_csv(self.graph_dir / "genres.csv").sort_values("index")
        self.user_map = dict(zip(self.users.userId, self.users["index"]))
        self.movie_map = dict(zip(self.movies.movieId, self.movies["index"]))
        self.genre_map = dict(zip(self.genres.genre, self.genres["index"]))
        self.threshold = json.loads(
            (self.split_dir / "split_metadata.json").read_text()
        )["positive_threshold"]
        if self.threshold != threshold:
            raise ValueError("Configured threshold does not match prepared splits")
        digest = hashlib.sha256()
        for directory, names in [
            (self.graph_dir, ["movies", "users", "genres", "movie_genres"]),
            (self.split_dir, ["train"]),
        ]:
            for name in names:
                digest.update((directory / f"{name}.csv").read_bytes())
        digest.update(str(threshold).encode())
        self.fingerprint = digest.hexdigest()
        self.splits = {
            s: pd.read_csv(self.split_dir / f"{s}.csv")
            for s in ["train", "validation", "test"]
        }
        self.features = np.zeros((len(self.movies), len(self.genres)), dtype=np.float32)
        edges = pd.read_csv(self.graph_dir / "movie_genres.csv")
        self.genre_edges = np.array(
            [
                [self.movie_map[m], self.genre_map[g]]
                for m, g in zip(edges.movieId, edges.genre)
            ],
            dtype=np.int64,
        ).T
        self.features[self.genre_edges[0], self.genre_edges[1]] = 1
        self.positive_train = self.pairs(
            self.splits["train"][self.splits["train"].rating >= self.threshold]
        )
        self.observed = {i: set() for i in range(len(self.users))}
        for u, m in self.pairs(self.splits["train"]):
            self.observed[int(u)].add(int(m))

    def pairs(self, frame):
        return np.array(
            [
                [self.user_map[u], self.movie_map[m]]
                for u, m in zip(frame.userId, frame.movieId)
            ],
            dtype=np.int64,
        ).reshape(-1, 2)

    def graph(self, interactions=None):
        data = HeteroData()
        data["user"].num_nodes = len(self.users)
        data["movie"].num_nodes = len(self.movies)
        data["movie"].x = torch.from_numpy(self.features)
        data["genre"].num_nodes = len(self.genres)
        edges = torch.as_tensor(
            self.positive_train if interactions is None else interactions,
            dtype=torch.long,
        ).T.contiguous()
        ge = torch.as_tensor(self.genre_edges, dtype=torch.long)
        data["user", "likes", "movie"].edge_index = edges
        data["movie", "rev_likes", "user"].edge_index = edges.flip(0)
        data["movie", "has_genre", "genre"].edge_index = ge
        data["genre", "rev_has_genre", "movie"].edge_index = ge.flip(0)
        return data

    def relevant(self, split):
        frame = self.splits[split][self.splits[split].rating >= self.threshold]
        result = {}
        for u, m in self.pairs(frame):
            result.setdefault(int(u), set()).add(int(m))
        return result


def prepare_splits(graph_dir: str, output_dir: str, threshold=4.0):
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(Path(graph_dir) / "ratings.csv")
    chunks = {s: [] for s in ["train", "validation", "test"]}
    for _, rows in ratings.groupby("userId", sort=True):
        rows = rows.sort_values(["timestamp", "movieId"], kind="stable")
        n = len(rows)
        holdout = max(1, int(n * 0.1))
        if n < 3:
            raise ValueError("Each user needs at least three interactions")
        chunks["train"].append(rows.iloc[: n - 2 * holdout])
        chunks["validation"].append(rows.iloc[n - 2 * holdout : n - holdout])
        chunks["test"].append(rows.iloc[n - holdout :])
    result = {}
    for split, frames in chunks.items():
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(p / f"{split}.csv", index=False)
        result[split] = len(table)
    result.update(
        {
            "method": "per-user chronological 80/10/10 (rounded holdouts)",
            "positive_threshold": threshold,
            "tie_break": "movieId",
            "protocol": "fixed training history for validation and test; full unseen catalog",
        }
    )
    (p / "split_metadata.json").write_text(json.dumps(result, indent=2))
    return result


def load_graph_dataset(graph_dir: str, split_dir: str):
    return GraphDataset(graph_dir, split_dir)
