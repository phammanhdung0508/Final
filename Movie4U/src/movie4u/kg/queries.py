"""Training-history semantic evidence (not a causal explanation of GNN scores)."""


def explanation_paths(dataset, user_id: int, movie_id: int, limit=3):
    if user_id not in dataset.user_map or movie_id not in dataset.movie_map:
        raise KeyError("Unknown user or movie")
    u, m = dataset.user_map[user_id], dataset.movie_map[movie_id]
    target_genres = set(dataset.genres.genre[dataset.features[m] > 0])
    evidence = []
    movies = dataset.movies.set_index("index")
    for user, liked in dataset.positive_train:
        if user != u:
            continue
        shared = target_genres & set(dataset.genres.genre[dataset.features[liked] > 0])
        for genre in sorted(shared):
            evidence.append(
                {
                    "liked_movie_id": int(movies.loc[liked, "movieId"]),
                    "liked_title": str(movies.loc[liked, "title"]),
                    "genre": genre,
                    "path": f"User {user_id} -> liked -> {movies.loc[liked, 'title']} -> genre -> {genre} <- genre <- {movies.loc[m, 'title']}",
                }
            )
            if len(evidence) >= limit:
                return evidence
    return evidence
