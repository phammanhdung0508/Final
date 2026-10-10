from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autoresearch-posttraining-ml-1m"

def find_or_download_raw_movielens(base_dir: Path) -> Path:
    raw_dir = base_dir / "raw" / "ml-1m"
    if (raw_dir / "movies.dat").exists() or (raw_dir / "movies.csv").exists():
        return raw_dir

    if Path("data/raw/movielens/ml-1m/movies.dat").exists():
        return Path("data/raw/movielens/ml-1m")

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = base_dir / "raw" / "ml-1m.zip"
    print(f">> Downloading MovieLens 1M to {zip_path}...")
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-1m.zip", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(base_dir / "raw")
    return raw_dir

def build_knowledge_graph(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (source_dir / "movies.dat").exists():
        movies = pd.read_csv(source_dir / "movies.dat", sep="::", engine="python", names=["movieId", "title", "genres"], encoding="latin-1")
        ratings = pd.read_csv(source_dir / "ratings.dat", sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1")
        users = pd.read_csv(source_dir / "users.dat", sep="::", engine="python", names=["userId", "gender", "age", "occupation", "zip"], encoding="latin-1")
    else:
        raise FileNotFoundError(f"No movies.dat found in {source_dir}")

    # Movies
    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["index"] = range(len(movies))
    
    # Extract Year
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = movies['year'].fillna("Unknown")
    
    # Users
    users = users.sort_values("userId").reset_index(drop=True)
    users["index"] = range(len(users))

    # Genres
    genre_names = sorted({g for text in movies.genres for g in str(text).split("|") if g != "(no genres listed)"})
    genres = pd.DataFrame({"genre": genre_names, "index": range(len(genre_names))})
    
    # Years
    year_names = sorted({str(y) for y in movies.year})
    years = pd.DataFrame({"year": year_names, "index": range(len(year_names))})
    
    # Gender
    gender_names = sorted({str(g) for g in users.gender})
    genders = pd.DataFrame({"gender": gender_names, "index": range(len(gender_names))})
    
    # Age
    age_names = sorted({str(a) for a in users.age})
    ages = pd.DataFrame({"age": age_names, "index": range(len(age_names))})
    
    # Occupation
    occ_names = sorted({str(o) for o in users.occupation})
    occupations = pd.DataFrame({"occupation": occ_names, "index": range(len(occ_names))})

    # Edges
    genre_edges = pd.DataFrame([{"movieId": int(row.movieId), "genre": genre} for row in movies.itertuples() for genre in str(row.genres).split("|") if genre != "(no genres listed)"])
    year_edges = pd.DataFrame([{"movieId": int(row.movieId), "year": str(row.year)} for row in movies.itertuples()])
    gender_edges = pd.DataFrame([{"userId": int(row.userId), "gender": str(row.gender)} for row in users.itertuples()])
    age_edges = pd.DataFrame([{"userId": int(row.userId), "age": str(row.age)} for row in users.itertuples()])
    occ_edges = pd.DataFrame([{"userId": int(row.userId), "occupation": str(row.occupation)} for row in users.itertuples()])

    for name, table in [
        ("movies", movies), ("users", users), ("ratings", ratings), 
        ("genres", genres), ("years", years), ("genders", genders), ("ages", ages), ("occupations", occupations),
        ("movie_genres", genre_edges), ("movie_years", year_edges), 
        ("user_genders", gender_edges), ("user_ages", age_edges), ("user_occupations", occ_edges)
    ]:
        table.to_csv(output_dir / f"{name}.csv", index=False)

    metadata = {
        "users": len(users), "movies": len(movies), "genres": len(genres), 
        "years": len(years), "genders": len(genders), "ages": len(ages), "occupations": len(occupations),
        "source": str(source_dir)
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata

def prepare_splits(graph_dir: Path, output_dir: Path, threshold: float = 4.0) -> dict:
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

    result.update({
        "method": "per-user chronological 80/10/10",
        "positive_threshold": threshold,
    })
    (output_dir / "split_metadata.json").write_text(json.dumps(result, indent=2))
    return result

class GraphDataset:
    def __init__(self, graph_dir: str | Path, split_dir: str | Path, threshold: float = 4.0):
        self.graph_dir, self.split_dir = Path(graph_dir), Path(split_dir)
        self.movies = pd.read_csv(self.graph_dir / "movies.csv").sort_values("index")
        self.users = pd.read_csv(self.graph_dir / "users.csv").sort_values("index")
        
        self.genres = pd.read_csv(self.graph_dir / "genres.csv").sort_values("index")
        self.years = pd.read_csv(self.graph_dir / "years.csv").sort_values("index")
        self.genders = pd.read_csv(self.graph_dir / "genders.csv").sort_values("index")
        self.ages = pd.read_csv(self.graph_dir / "ages.csv").sort_values("index")
        self.occupations = pd.read_csv(self.graph_dir / "occupations.csv").sort_values("index")

        self.user_map = dict(zip(self.users.userId, self.users["index"]))
        self.movie_map = dict(zip(self.movies.movieId, self.movies["index"]))
        
        self.genre_map = dict(zip(self.genres.genre, self.genres["index"]))
        self.year_map = dict(zip(self.years.year.astype(str), self.years["index"]))
        self.gender_map = dict(zip(self.genders.gender, self.genders["index"]))
        self.age_map = dict(zip(self.ages.age.astype(str), self.ages["index"]))
        self.occ_map = dict(zip(self.occupations.occupation.astype(str), self.occupations["index"]))

        meta_path = self.split_dir / "split_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.threshold = float(meta["positive_threshold"])
        else:
            self.threshold = threshold

        digest = hashlib.sha256()
        digest.update(str(self.threshold).encode())
        self.fingerprint = digest.hexdigest()

        self.splits = {s: pd.read_csv(self.split_dir / f"{s}.csv") for s in ["train", "validation", "test"] if (self.split_dir / f"{s}.csv").exists()}

        # Read Edges
        df_mg = pd.read_csv(self.graph_dir / "movie_genres.csv")
        self.edge_mg = np.array([[self.movie_map[m], self.genre_map[g]] for m, g in zip(df_mg.movieId, df_mg.genre)], dtype=np.int64).T
        
        df_my = pd.read_csv(self.graph_dir / "movie_years.csv")
        self.edge_my = np.array([[self.movie_map[m], self.year_map[str(y)]] for m, y in zip(df_my.movieId, df_my.year)], dtype=np.int64).T
        
        df_ug = pd.read_csv(self.graph_dir / "user_genders.csv")
        self.edge_ug = np.array([[self.user_map[u], self.gender_map[g]] for u, g in zip(df_ug.userId, df_ug.gender)], dtype=np.int64).T
        
        df_ua = pd.read_csv(self.graph_dir / "user_ages.csv")
        self.edge_ua = np.array([[self.user_map[u], self.age_map[str(a)]] for u, a in zip(df_ua.userId, df_ua.age)], dtype=np.int64).T
        
        df_uo = pd.read_csv(self.graph_dir / "user_occupations.csv")
        self.edge_uo = np.array([[self.user_map[u], self.occ_map[str(o)]] for u, o in zip(df_uo.userId, df_uo.occupation)], dtype=np.int64).T

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
        return np.array([[self.user_map[u], self.movie_map[m]] for u, m in zip(frame.userId, frame.movieId)], dtype=np.int64).reshape(-1, 2)

    def graph(self, interactions: np.ndarray | None = None) -> HeteroData:
        data = HeteroData()
        data["user"].num_nodes = len(self.users)
        data["movie"].num_nodes = len(self.movies)
        data["genre"].num_nodes = len(self.genres)
        data["year"].num_nodes = len(self.years)
        data["gender"].num_nodes = len(self.genders)
        data["age"].num_nodes = len(self.ages)
        data["occupation"].num_nodes = len(self.occupations)

        edges = torch.as_tensor(self.positive_train if interactions is None else interactions, dtype=torch.long).T.contiguous()
        
        data["user", "likes", "movie"].edge_index = edges
        data["movie", "rev_likes", "user"].edge_index = edges.flip(0)
        
        # KG edges
        data["movie", "has_genre", "genre"].edge_index = torch.as_tensor(self.edge_mg, dtype=torch.long)
        data["genre", "rev_has_genre", "movie"].edge_index = torch.as_tensor(self.edge_mg, dtype=torch.long).flip(0)
        
        data["movie", "released_in", "year"].edge_index = torch.as_tensor(self.edge_my, dtype=torch.long)
        data["year", "rev_released_in", "movie"].edge_index = torch.as_tensor(self.edge_my, dtype=torch.long).flip(0)
        
        data["user", "has_gender", "gender"].edge_index = torch.as_tensor(self.edge_ug, dtype=torch.long)
        data["gender", "rev_has_gender", "user"].edge_index = torch.as_tensor(self.edge_ug, dtype=torch.long).flip(0)
        
        data["user", "has_age", "age"].edge_index = torch.as_tensor(self.edge_ua, dtype=torch.long)
        data["age", "rev_has_age", "user"].edge_index = torch.as_tensor(self.edge_ua, dtype=torch.long).flip(0)
        
        data["user", "has_occupation", "occupation"].edge_index = torch.as_tensor(self.edge_uo, dtype=torch.long)
        data["occupation", "rev_has_occupation", "user"].edge_index = torch.as_tensor(self.edge_uo, dtype=torch.long).flip(0)

        return data

    def relevant(self, split: str) -> dict[int, set[int]]:
        frame = self.splits[split][self.splits[split].rating >= self.threshold]
        result: dict[int, set[int]] = {}
        for u, m in self.pairs(frame):
            result.setdefault(int(u), set()).add(int(m))
        return result

def ensure_prepared(base_dir: Path | str | None = None, threshold: float = 4.0) -> tuple[Path, Path]:
    root = Path(base_dir) if base_dir else (Path.cwd() / "data" if Path("data").is_dir() else DEFAULT_CACHE_DIR)
    
    kg_dir = root / "processed_rich" / "kg"
    split_dir = root / "processed_rich" / "splits"

    if (kg_dir / "metadata.json").exists() and (split_dir / "split_metadata.json").exists():
        return kg_dir, split_dir

    raw_dir = find_or_download_raw_movielens(root)
    print(f">> Building Rich Knowledge Graph in {kg_dir}...")
    build_knowledge_graph(raw_dir, kg_dir)
    print(f">> Preparing splits in {split_dir} (threshold={threshold})...")
    prepare_splits(kg_dir, split_dir, threshold=threshold)
    return kg_dir, split_dir

def load_dataset(base_dir: Path | str | None = None) -> GraphDataset:
    kg_dir, split_dir = ensure_prepared(base_dir)
    return GraphDataset(kg_dir, split_dir)

def main() -> None:
    dataset = load_dataset()
    print(">> Rich KG Preparation complete!")
    print(f"Users: {len(dataset.users)}, Movies: {len(dataset.movies)}")
    print(f"Genres: {len(dataset.genres)}, Years: {len(dataset.years)}")
    print(f"Genders: {len(dataset.genders)}, Ages: {len(dataset.ages)}, Occupations: {len(dataset.occupations)}")
    print(f"Train positive edges: {len(dataset.positive_train)}")

if __name__ == "__main__":
    main()
