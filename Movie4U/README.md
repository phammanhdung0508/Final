# Movie4U

An end-to-end MVP comparing **KG-only** and **KG + heterogeneous GraphSAGE** movie recommendations, with a backend API and an Expo mobile demo.

## Required scope

1. Build one canonical Knowledge Graph from MovieLens Small.
2. Implement a KG-only recommender without a GNN.
3. Train a KG + GNN recommender using the same graph.
4. Compare both on identical splits, candidates, and ranking metrics.
5. Serve recommendations and KG evidence through an API.
6. Demonstrate both approaches in a mobile application.

Agents and richer external metadata are future extensions, not dependencies of this MVP.

## Quick start

Python 3.11+ is required. Run from this directory:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/run_pipeline.py
pytest -q
uvicorn movie4u.api.main:app --host 0.0.0.0 --port 8000
```

The dataset already lives in `data/raw/movielens/ml-latest-small/`. The pipeline creates the KG and chronological splits, trains a CPU-friendly GNN for 50 epochs, selects a checkpoint using validation NDCG@10, and writes `artifacts/metrics/comparison.json`. Use `--epochs 5` for a smoke run; its checkpoint is experimental, not the full training run.

Training is local and does not require external APIs, Neo4j, a GPU, or an LLM. Install PyTorch appropriate for your platform if the default package installation is unsuitable.

Open `http://localhost:8000/docs` for the API. See `mobile/README.md` to configure the backend address and run the Expo demo. The development API has no authentication; do not expose it to the public internet.

## Single-notebook walkthrough

Open `notebooks/Movie4U_walkthrough.ipynb` for dataset exploration, KG visualization, user recommendations, training curves, and evaluation comparison in one place.

```sh
source .venv/bin/activate
pip install -e '.[notebook]'
python -m ipykernel install --user --name movie4u --display-name 'Python (Movie4U)'
```

Select that kernel and Run All. The notebook loads existing artifacts without retraining or modifying source data. Change `USER_ID` to explore another profile. Notebook outputs are cleared in the committed version.

## Pipeline

```text
MovieLens -> validated KG -> chronological training view
                            |-> KG-only genre/path scoring
                            |-> heterogeneous GraphSAGE training
                            -> shared full-catalog evaluation
                            -> API -> mobile comparison UI
```

## Structure

- `src/movie4u/kg/`: schema, construction, validation, and queries
- `src/movie4u/recommender/`: baseline, graph dataset, GNN, training, metrics, inference
- `src/movie4u/api/`: read-only demonstration API
- `mobile/`: Expo / React Native / TypeScript application
- `scripts/`: pipeline entry point
- `configs/`: dataset paths, rating threshold, model settings
- `data/`: immutable raw data and reproducible processed outputs
- `artifacts/`: generated checkpoints, histories, and comparison metrics
- `tests/`: small-fixture unit and integration tests
- `docs/`: KG schema and evaluation protocol
- `notebooks/`: one end-to-end exploration and demonstration notebook

## API

- `GET /health`
- `GET /users`
- `GET /movies/{movie_id}`
- `GET /users/{user_id}/recommendations?method=kg_only&k=10`
- `GET /users/{user_id}/recommendations?method=kg_gnn&k=10`
- `GET /users/{user_id}/movies/{movie_id}/explanation`
- `GET /evaluation`

The demo selects existing MovieLens users; it does not create accounts, accept live feedback, or retrain online. Returned scores are not calibrated probabilities. KG paths are semantic evidence, not causal explanations of GNN behavior.

## Evaluation

See `docs/evaluation.md`, `docs/kg-schema.md`, and `docs/initial-results.md` for the first measured comparison. Both methods use the same training-only interaction graph and complete unseen catalog. Never interpret an unobserved rating as a verified dislike or assume the GNN must outperform the baseline. The first experiment is a measured starting point, not a SOTA result.

## Dataset attribution

MovieLens Small is provided by GroupLens. Cite F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, https://doi.org/10.1145/2827872. Follow the dataset's `README.txt` licensing terms, including the commercial-use restriction.

## Reference

Alessandro Negro, *Knowledge Graphs and LLMs in Action*, Chapter 12, section 12.2. The book motivates our heterogeneous encoder–decoder pipeline; this project adds explicit Genre nodes, a KG-only baseline, and ranking evaluation.
