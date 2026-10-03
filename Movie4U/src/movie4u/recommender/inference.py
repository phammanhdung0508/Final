"""Read-only inference for the comparison API."""

from pathlib import Path
import torch
from .baseline import KGOnlyRecommender
from .dataset import GraphDataset
from .model import create_model
from .evaluate import top_indices
from movie4u.kg.queries import explanation_paths


class RecommendationService:
    def __init__(self, project_dir):
        root = Path(project_dir)
        import json

        split_dir = root / "data/processed/gnn"
        threshold = json.loads((split_dir / "split_metadata.json").read_text())[
            "positive_threshold"
        ]
        self.dataset = GraphDataset(root / "data/processed/kg", split_dir, threshold)
        self.scores = {"kg_only": KGOnlyRecommender(self.dataset).scores()}
        checkpoint = root / "artifacts/models/gnn.pt"
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload.get("dataset_fingerprint") != self.dataset.fingerprint:
                raise ValueError(
                    "Checkpoint graph/history does not match current dataset; retrain"
                )
            if (
                payload["user_ids"] != self.dataset.users.userId.tolist()
                or payload["movie_ids"] != self.dataset.movies.movieId.tolist()
            ):
                raise ValueError("Checkpoint ID mappings do not match dataset")
            torch.set_num_threads(payload["config"].get("cpu_threads", 4))
            model = create_model(self.dataset, payload["config"])
            model.load_state_dict(payload["state_dict"])
            model.eval()
            with torch.no_grad():
                x = model(self.dataset.graph())
                self.scores["kg_gnn"] = (x["user"] @ x["movie"].T).numpy()

    def movie(self, movie_id):
        if movie_id not in self.dataset.movie_map:
            raise KeyError("Unknown movie")
        row = self.dataset.movies.iloc[self.dataset.movie_map[movie_id]]
        return {
            "movie_id": int(row.movieId),
            "title": str(row.title),
            "genres": []
            if row.genres == "(no genres listed)"
            else row.genres.split("|"),
        }

    def recommend(self, user_id, method="kg_only", top_k=10):
        if user_id not in self.dataset.user_map:
            raise KeyError("Unknown user")
        if method not in self.scores:
            raise ValueError("Requested model is unavailable")
        user = self.dataset.user_map[user_id]
        ranked = top_indices(
            self.scores[method][user], self.dataset.observed[user], top_k
        )
        return [
            {
                "rank": rank,
                **self.movie(int(self.dataset.movies.iloc[m].movieId)),
                "score": float(self.scores[method][user, m]),
                "evidence": explanation_paths(
                    self.dataset, user_id, int(self.dataset.movies.iloc[m].movieId)
                ),
            }
            for rank, m in enumerate(ranked, start=1)
        ]
