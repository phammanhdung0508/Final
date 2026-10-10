# Knowledge Graph schema

## Canonical graph

The normalized CSV tables in `data/processed/kg/` form the canonical KG:

- `users.csv`: anonymized `userId` and contiguous `index`.
- `movies.csv`: `movieId`, title, original genre string, IMDb/TMDb IDs, and contiguous `index`.
- `genres.csv`: 19 actual genre labels and contiguous `index`. `(no genres listed)` is missing metadata, not a genre node.
- `ratings.csv`: User -> RATED -> Movie with numeric rating and Unix timestamp attributes.
- `movie_genres.csv`: Movie -> HAS_GENRE -> Genre.
- `metadata.json`: counts and source location.

No Neo4j service is required for the MVP. The typed node/edge tables are the KG, and PyG HeteroData is its model representation. IDs are mapped explicitly rather than interpreted as ordinal features. IMDb leading zeros are preserved.

## Training/serving graph

Only training ratings >= 4 are projected to `User -> likes -> Movie`. Genre relations cover the complete known catalog. Reverse edges exist for both relation types to enable bidirectional message passing. Titles and external IDs are not used as model features.

Training uses a seeded disjoint partition of positive training edges: 80% for message passing and 20% for supervision. Sampled negatives exclude *all* training-rated movies, including low ratings. No validation/test ratings enter this graph. After each checkpoint, inference uses all positive training edges for both validation and test, just as the KG-only baseline uses all positive training history.

Canonical ratings remain intact. Generated training views never overwrite raw data.

## Semantic evidence

The API retrieves `User -> liked Movie -> Genre <- candidate Movie` paths from training history. These are factual semantic connections, not guaranteed causal explanations of the GNN score. Only a few example paths are returned. No path means no shared-genre evidence, not an invalid recommendation.
