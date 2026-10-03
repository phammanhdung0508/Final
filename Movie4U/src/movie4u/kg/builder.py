"""Build canonical CSV graph tables without modifying source data."""

import json
from pathlib import Path
import pandas as pd


def build_knowledge_graph(source_dir: str, output_dir: str) -> dict:
    source, out = Path(source_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    movies = pd.read_csv(source / "movies.csv")
    ratings = pd.read_csv(source / "ratings.csv")
    links = pd.read_csv(source / "links.csv", dtype={"imdbId": str})
    movies = movies.merge(links, on="movieId", how="left", validate="one_to_one")
    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["index"] = range(len(movies))
    users = pd.DataFrame({"userId": sorted(ratings.userId.unique())})
    users["index"] = range(len(users))
    genre_names = sorted(
        {
            g
            for text in movies.genres
            for g in text.split("|")
            if g != "(no genres listed)"
        }
    )
    genres = pd.DataFrame({"genre": genre_names, "index": range(len(genre_names))})
    genre_edges = pd.DataFrame(
        [
            {"movieId": int(row.movieId), "genre": genre}
            for row in movies.itertuples()
            for genre in row.genres.split("|")
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
        table.to_csv(out / f"{name}.csv", index=False)
    metadata = {
        "users": len(users),
        "movies": len(movies),
        "genres": len(genres),
        "ratings": len(ratings),
        "genre_edges": len(genre_edges),
        "source": str(source),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
