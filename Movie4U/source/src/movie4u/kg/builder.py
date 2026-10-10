"""Build canonical CSV graph tables without modifying source data."""

import json
from pathlib import Path
import pandas as pd


def build_knowledge_graph(source_dir: str, output_dir: str) -> dict:
    source, out = Path(source_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # ml-1m parsing
    movies = pd.read_csv(source / "movies.dat", sep="::", engine="python", header=None, names=["movieId", "title", "genres"], encoding="latin-1")
    ratings = pd.read_csv(source / "ratings.dat", sep="::", engine="python", header=None, names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1")
    users = pd.read_csv(source / "users.dat", sep="::", engine="python", header=None, names=["userId", "gender", "age", "occupation", "zip"], encoding="latin-1")

    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["index"] = range(len(movies))
    
    users = users.sort_values("userId").reset_index(drop=True)
    users["index"] = range(len(users))
    
    # Genres
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
    
    # Years
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = movies['year'].fillna("Unknown")
    year_names = sorted({str(y) for y in movies.year})
    years = pd.DataFrame({"year": year_names, "index": range(len(year_names))})
    year_edges = pd.DataFrame([{"movieId": int(row.movieId), "year": str(row.year)} for row in movies.itertuples()])
    
    # Gender
    gender_names = sorted({str(g) for g in users.gender})
    genders = pd.DataFrame({"gender": gender_names, "index": range(len(gender_names))})
    gender_edges = pd.DataFrame([{"userId": int(row.userId), "gender": str(row.gender)} for row in users.itertuples()])
    
    # Occupations
    occupation_names = sorted(users.occupation.unique())
    occupations = pd.DataFrame({"occupation": occupation_names, "index": range(len(occupation_names))})
    user_occupations = pd.DataFrame([
        {"userId": row.userId, "occupation": row.occupation}
        for row in users.itertuples()
    ])
    
    # Ages
    age_names = sorted(users.age.unique())
    ages = pd.DataFrame({"age": age_names, "index": range(len(age_names))})
    user_ages = pd.DataFrame([
        {"userId": row.userId, "age": row.age}
        for row in users.itertuples()
    ])

    for name, table in [
        ("movies", movies),
        ("users", users),
        ("genres", genres),
        ("years", years),
        ("genders", genders),
        ("ratings", ratings),
        ("movie_genres", genre_edges),
        ("movie_years", year_edges),
        ("user_genders", gender_edges),
        ("occupations", occupations),
        ("user_occupations", user_occupations),
        ("ages", ages),
        ("user_ages", user_ages),
    ]:
        table.to_csv(out / f"{name}.csv", index=False)
        
    metadata = {
        "users": len(users),
        "movies": len(movies),
        "genres": len(genres),
        "years": len(years),
        "genders": len(genders),
        "occupations": len(occupations),
        "ages": len(ages),
        "ratings": len(ratings),
        "genre_edges": len(genre_edges),
        "movie_year_edges": len(year_edges),
        "user_gender_edges": len(gender_edges),
        "user_occupation_edges": len(user_occupations),
        "user_age_edges": len(user_ages),
        "source": str(source),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
