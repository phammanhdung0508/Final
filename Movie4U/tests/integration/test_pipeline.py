import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from movie4u.kg.builder import build_knowledge_graph
from movie4u.kg.validator import validate_knowledge_graph
from movie4u.recommender.dataset import prepare_splits, GraphDataset
from movie4u.recommender.baseline import KGOnlyRecommender
from movie4u.recommender.model import create_model
from movie4u.recommender.train import train
from movie4u.recommender.inference import RecommendationService
from movie4u.api.main import create_app


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "movieId": list(range(1, 9)),
            "title": [f"Movie {i}" for i in range(1, 9)],
            "genres": ["Drama", "Comedy"] * 4,
        }
    ).to_csv(raw / "movies.csv", index=False)
    pd.DataFrame(
        {
            "movieId": list(range(1, 9)),
            "imdbId": ["0000001"] * 8,
            "tmdbId": list(range(1, 9)),
        }
    ).to_csv(raw / "links.csv", index=False)
    pd.DataFrame(
        [
            {"userId": u, "movieId": m, "rating": 4.0, "timestamp": m}
            for u in [1, 2]
            for m in range(1, 7)
        ]
    ).to_csv(raw / "ratings.csv", index=False)
    kg, splits = tmp_path / "data/processed/kg", tmp_path / "data/processed/gnn"
    build_knowledge_graph(raw, kg)
    validate_knowledge_graph(kg)
    prepare_splits(kg, splits)
    return tmp_path


def test_split_and_graph_do_not_include_holdout(project):
    ds = GraphDataset(project / "data/processed/kg", project / "data/processed/gnn")
    graph = ds.graph()
    train = set(map(tuple, ds.positive_train))
    for split in ["validation", "test"]:
        assert not train & set(map(tuple, ds.pairs(ds.splits[split])))
    forward = graph["user", "likes", "movie"].edge_index
    reverse = graph["movie", "rev_likes", "user"].edge_index
    assert torch.equal(forward.flip(0), reverse)
    assert set(forward[1].tolist()) == {0, 1, 2, 3}
    assert ds.features.shape == (8, 2)


def test_baseline_and_gnn_forward(project):
    ds = GraphDataset(project / "data/processed/kg", project / "data/processed/gnn")
    assert KGOnlyRecommender(ds).scores().shape == (2, 8)
    config = {"hidden_channels": 8, "num_layers": 2, "dropout": 0.0}
    model = create_model(ds, config)
    result = model(ds.graph())
    loss = model.decode(result, torch.tensor([[0, 4], [1, 5]])).sum()
    loss.backward()
    assert result["movie"].shape == (8, 8)


def test_api_and_unavailable_gnn(project):
    with TestClient(create_app(project)) as client:
        assert client.get("/health").json()["models"] == ["kg_only"]
        assert client.get("/users").status_code == 200
        result = client.get("/users/1/recommendations?k=2").json()["recommendations"]
        assert len(result) == 2
        assert not {r["movie_id"] for r in result} & {1, 2, 3, 4}
        assert client.get("/users/1/recommendations?method=kg_gnn").status_code == 503
        assert client.get("/users/999/recommendations").status_code == 404
        assert client.get("/users/1/recommendations?k=0").status_code == 422
        assert client.get("/movies/1").json()["title"] == "Movie 1"
        assert client.get("/users/1/movies/1/explanation").status_code == 200
        assert client.get("/evaluation").status_code == 503


def test_missing_data_returns_not_ready(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["status"] == "not_ready"
        assert client.get("/users").status_code == 503


def test_training_checkpoint_serves_gnn_and_rejects_stale_history(project):
    dataset = GraphDataset(
        project / "data/processed/kg", project / "data/processed/gnn"
    )
    config = {
        "seed": 42,
        "hidden_channels": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epochs": 1,
        "cpu_threads": 1,
        "top_k": 2,
    }
    train(dataset, config, project / "artifacts/models")
    with TestClient(create_app(project)) as client:
        assert client.get("/health").json()["models"] == ["kg_only", "kg_gnn"]
        response = client.get("/users/1/recommendations?method=kg_gnn&k=2")
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) == 2
    path = project / "data/processed/gnn/train.csv"
    rows = pd.read_csv(path)
    rows.loc[0, "rating"] = 2.0
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        RecommendationService(project)


def test_validator_rejects_missing_movie(project):
    p = project / "data/processed/kg/ratings.csv"
    rows = pd.read_csv(p)
    rows.loc[0, "movieId"] = 999
    rows.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing nodes"):
        validate_knowledge_graph(p.parent)
