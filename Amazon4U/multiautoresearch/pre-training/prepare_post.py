"""Independent data preparation and graph dataset loader for post-training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


DEFAULT_DATASET = os.environ.get("MOVIELENS_DATASET", "ml-1m")
MOVIELENS_URL_SMALL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_URL_1M = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

DEFAULT_CACHE_DIR = Path(
    os.environ.get("POSTTRAIN_CACHE_DIR", f"~/.cache/autoresearch-posttraining-{DEFAULT_DATASET}")
).expanduser()


def find_or_download_raw_movielens(target_dir: Path, dataset_name: str = DEFAULT_DATASET) -> Path:
    """Find local MovieLens data or download and extract ml-1m or ml-latest-small."""
    if dataset_name == "ml-1m":
        url = MOVIELENS_URL_1M
        folder = "ml-1m"
    else:
        url = MOVIELENS_URL_SMALL
        folder = "ml-latest-small"

    # Check potential local candidate directories
    candidates = [
        Path(f"data/raw/movielens/{folder}"),
        Path(f"../Movie4U/data/raw/movielens/{folder}"),
        Path(f"../../Movie4U/data/raw/movielens/{folder}"),
        Path(f"/home/sunf/FSB/Final/data/raw/movielens/{folder}"),
        Path(f"/home/sunf/FSB/Final/Movie4U/data/raw/movielens/{folder}"),
        target_dir / "raw" / folder,
    ]
    for c in candidates:
        if (c / "ratings.csv").exists() or (c / "ratings.dat").exists():
            return c.resolve()

    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = raw_dir / folder
    if (extracted / "ratings.csv").exists() or (extracted / "ratings.dat").exists():
        return extracted

    zip_path = raw_dir / f"{folder}.zip"
    print(f">> Downloading MovieLens {dataset_name} from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f">> Extracting MovieLens {dataset_name} archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    return extracted


def build_knowledge_graph(source_dir: Path, output_dir: Path) -> dict:
    """Build canonical CSV graph tables without modifying source data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (source_dir / "movies.csv").exists():
        movies = pd.read_csv(source_dir / "movies.csv")
        ratings = pd.read_csv(source_dir / "ratings.csv")
        links_path = source_dir / "links.csv"
        if links_path.exists():
            links = pd.read_csv(links_path, dtype={"imdbId": str})
            movies = movies.merge(links, on="movieId", how="left", validate="one_to_one")
    elif (source_dir / "movies.dat").exists():
        movies = pd.read_csv(
            source_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        ratings = pd.read_csv(
            source_dir / "ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
    else:
        raise FileNotFoundError(f"No movies.csv or movies.dat found in {source_dir}")

    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["index"] = range(len(movies))

    users = pd.DataFrame({"userId": sorted(ratings.userId.unique())})
    users["index"] = range(len(users))

    genre_names = sorted(
        {
            g
            for text in movies.genres
            for g in str(text).split("|")
            if g != "(no genres listed)"
        }
    )
    genres = pd.DataFrame({"genre": genre_names, "index": range(len(genre_names))})

    genre_edges = pd.DataFrame(
        [
            {"movieId": int(row.movieId), "genre": genre}
            for row in movies.itertuples()
            for genre in str(row.genres).split("|")
            if genre != "(no genres listed)"
        ]
    )

    for name, table in [
        ("movies", movies),
        ("users", users),
        ("genres", genres),
        ("ratings", ratings),
        ("movie_genres", genre_edges),
    ]:
        table.to_csv(output_dir / f"{name}.csv", index=False)

    metadata = {
        "users": len(users),
        "movies": len(movies),
        "genres": len(genres),
        "ratings": len(ratings),
        "genre_edges": len(genre_edges),
        "source": str(source_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def prepare_splits(graph_dir: Path, output_dir: Path, threshold: float = 4.0) -> dict:
    """Prepare chronological 80/10/10 per-user train/val/test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(graph_dir / "ratings.csv")
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

    result: dict[str, object] = {}
    for split, frames in chunks.items():
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(output_dir / f"{split}.csv", index=False)
        result[split] = len(table)

    result.update(
        {
            "method": "per-user chronological 80/10/10 (rounded holdouts)",
            "positive_threshold": threshold,
            "tie_break": "movieId",
            "protocol": "fixed training history for validation and test; full unseen catalog",
        }
    )
    (output_dir / "split_metadata.json").write_text(json.dumps(result, indent=2))
    return result


class GraphDataset:
    """In-memory recommendation graph and chronological interaction view."""

    def __init__(self, graph_dir: str | Path, split_dir: str | Path, threshold: float = 4.0):
        self.graph_dir, self.split_dir = Path(graph_dir), Path(split_dir)
        self.movies = pd.read_csv(self.graph_dir / "movies.csv").sort_values("index")
        self.users = pd.read_csv(self.graph_dir / "users.csv").sort_values("index")
        self.genres = pd.read_csv(self.graph_dir / "genres.csv").sort_values("index")

        self.user_map = dict(zip(self.users.userId, self.users["index"]))
        self.movie_map = dict(zip(self.movies.movieId, self.movies["index"]))
        self.genre_map = dict(zip(self.genres.genre, self.genres["index"]))

        meta_path = self.split_dir / "split_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.threshold = float(meta["positive_threshold"])
        else:
            self.threshold = threshold

        # Fingerprint for dataset consistency
        digest = hashlib.sha256()
        for directory, names in [
            (self.graph_dir, ["movies", "users", "genres", "movie_genres"]),
            (self.split_dir, ["train"]),
        ]:
            for name in names:
                fpath = directory / f"{name}.csv"
                if fpath.exists():
                    digest.update(fpath.read_bytes())
        digest.update(str(self.threshold).encode())
        self.fingerprint = digest.hexdigest()

        self.splits = {
            s: pd.read_csv(self.split_dir / f"{s}.csv")
            for s in ["train", "validation", "test"]
            if (self.split_dir / f"{s}.csv").exists()
        }

        # Movie genre features
        self.features = np.zeros((len(self.movies), len(self.genres)), dtype=np.float32)
        edges = pd.read_csv(self.graph_dir / "movie_genres.csv")
        self.genre_edges = np.array(
            [
                [self.movie_map[m], self.genre_map[g]]
                for m, g in zip(edges.movieId, edges.genre)
            ],
            dtype=np.int64,
        ).T
        self.features[self.genre_edges[0], self.genre_edges[1]] = 1.0

        train_split = self.splits.get("train", pd.DataFrame())
        if not train_split.empty:
            self.positive_train = self.pairs(train_split[train_split.rating >= self.threshold])
        else:
            self.positive_train = np.empty((0, 2), dtype=np.int64)

        self.observed: dict[int, set[int]] = {i: set() for i in range(len(self.users))}
        if not train_split.empty:
            for u, m in self.pairs(train_split):
                self.observed[int(u)].add(int(m))

    def pairs(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                [self.user_map[u], self.movie_map[m]]
                for u, m in zip(frame.userId, frame.movieId)
            ],
            dtype=np.int64,
        ).reshape(-1, 2)

    def graph(self, interactions: np.ndarray | None = None) -> HeteroData:
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

    def relevant(self, split: str) -> dict[int, set[int]]:
        frame = self.splits[split][self.splits[split].rating >= self.threshold]
        result: dict[int, set[int]] = {}
        for u, m in self.pairs(frame):
            result.setdefault(int(u), set()).add(int(m))
        return result


def ensure_prepared(
    base_dir: Path | str | None = None,
    threshold: float = 4.0,
) -> tuple[Path, Path]:
    """Ensure raw data is extracted, KG is built, and splits are prepared."""
    root = Path(base_dir) if base_dir else (Path.cwd() / "data" if Path("data").is_dir() else DEFAULT_CACHE_DIR)
    kg_dir = root / "processed" / "kg"
    split_dir = root / "processed" / "splits"

    if (
        (kg_dir / "metadata.json").exists()
        and (split_dir / "split_metadata.json").exists()
        and (split_dir / "train.csv").exists()
    ):
        return kg_dir, split_dir

    raw_dir = find_or_download_raw_movielens(root)
    print(f">> Building Knowledge Graph in {kg_dir}...")
    build_knowledge_graph(raw_dir, kg_dir)
    print(f">> Preparing splits in {split_dir} (threshold={threshold})...")
    prepare_splits(kg_dir, split_dir, threshold=threshold)
    return kg_dir, split_dir


def load_dataset(base_dir: Path | str | None = None) -> GraphDataset:
    kg_dir, split_dir = ensure_prepared(base_dir)
    return GraphDataset(kg_dir, split_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MovieLens KG and chronological splits for post-training.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Base directory for processed data")
    parser.add_argument("--threshold", type=float, default=4.0, help="Positive rating threshold")
    args = parser.parse_args()

    kg_dir, split_dir = ensure_prepared(base_dir=args.output_dir, threshold=args.threshold)
    dataset = GraphDataset(kg_dir, split_dir)
    print(">> Preparation complete!")
    print(f"Users: {len(dataset.users)}, Movies: {len(dataset.movies)}, Genres: {len(dataset.genres)}")
    print(f"Train positive edges: {len(dataset.positive_train)}")
    for s in ["train", "validation", "test"]:
        rel = dataset.relevant(s)
        print(f"Split {s}: {len(dataset.splits[s])} total ratings, {len(rel)} users with positive holdout")


if __name__ == "__main__":
    main()
