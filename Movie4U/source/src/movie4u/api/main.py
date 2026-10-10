"""Read-only demonstration API; processed data is loaded once at startup."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from movie4u.recommender.inference import RecommendationService

ROOT = Path(__file__).resolve().parents[3]


def create_app(project_dir=None):
    root = Path(project_dir or os.environ.get("MOVIE4U_ROOT", ROOT))

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.service = RecommendationService(root)
            app.state.error = None
        except (FileNotFoundError, ValueError) as exc:
            app.state.service = None
            app.state.error = str(exc)
        yield

    app = FastAPI(title="Movie4U API", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get(
            "MOVIE4U_CORS_ORIGINS", "http://localhost:8081,http://localhost:19006"
        ).split(","),
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def service(request):
        value = request.app.state.service
        if value is None:
            raise HTTPException(
                503, "Run the pipeline before requesting recommendations"
            )
        return value

    @app.get("/health")
    def health(request: Request):
        ready = request.app.state.service is not None
        return {
            "status": "ok" if ready else "not_ready",
            "models": list(request.app.state.service.scores) if ready else [],
        }

    @app.get("/users")
    def users(request: Request):
        dataset = service(request).dataset
        return [
            {
                "user_id": int(row.userId),
                "training_ratings": len(dataset.observed[int(row.index)]),
            }
            for row in dataset.users.itertuples(index=False)
        ]

    @app.get("/movies/{movie_id}")
    def movie(movie_id: int, request: Request):
        try:
            return service(request).movie(movie_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/users/{user_id}/recommendations")
    def recommendations(
        user_id: int,
        request: Request,
        method: Literal["kg_only", "kg_gnn"] = "kg_only",
        k: int = Query(10, ge=1, le=50),
    ):
        try:
            results = service(request).recommend(user_id, method, k)
            return {
                "user_id": user_id,
                "method": method,
                "score_note": "Ranking score, not a calibrated probability; not comparable across methods.",
                "recommendations": results,
            }
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/users/{user_id}/movies/{movie_id}/explanation")
    def explanation(user_id: int, movie_id: int, request: Request):
        from movie4u.kg.queries import explanation_paths

        try:
            return {
                "evidence": explanation_paths(
                    service(request).dataset, user_id, movie_id
                ),
                "note": "Shared-genre evidence from training history, not a causal explanation of the GNN score.",
            }
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/evaluation")
    def evaluation():
        path = root / "artifacts/metrics/comparison.json"
        if not path.exists():
            raise HTTPException(503, "Evaluation has not been run")
        return json.loads(path.read_text())

    return app


app = create_app()
