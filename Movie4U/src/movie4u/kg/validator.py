"""Validate canonical graph endpoints, identifiers, and edge attributes."""

from pathlib import Path
import pandas as pd


def validate_knowledge_graph(graph_dir: str) -> None:
    p = Path(graph_dir)
    movies, users, genres, ratings, edges = [
        pd.read_csv(p / f"{n}.csv")
        for n in ["movies", "users", "genres", "ratings", "movie_genres"]
    ]
    for table, key in [(movies, "movieId"), (users, "userId"), (genres, "genre")]:
        if table[key].isna().any() or table[key].duplicated().any():
            raise ValueError(f"Invalid or duplicate {key}")
        if table["index"].tolist() != list(range(len(table))):
            raise ValueError(f"Non-contiguous indices: {key}")
    if not set(ratings.movieId) <= set(movies.movieId) or not set(
        ratings.userId
    ) <= set(users.userId):
        raise ValueError("Rating references missing nodes")
    if not set(edges.movieId) <= set(movies.movieId) or not set(edges.genre) <= set(
        genres.genre
    ):
        raise ValueError("Genre edge references missing nodes")
    if ratings.duplicated(["userId", "movieId"]).any() or edges.duplicated().any():
        raise ValueError("Duplicate edges")
    if (
        ratings[["rating", "timestamp"]].isna().any().any()
        or not ratings.rating.between(0.5, 5).all()
    ):
        raise ValueError("Invalid rating attributes")
    if not ((ratings.rating * 2) % 1 == 0).all() or not (ratings.timestamp >= 0).all():
        raise ValueError("Invalid rating scale or timestamp")
