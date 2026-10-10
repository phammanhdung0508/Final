"""Heterogeneous GraphSAGE encoder and link predictor for pre-training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphSAGEPretrain(nn.Module):
    """Heterogeneous GNN encoder for pre-training representation learning."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        counts: dict[str, int],
        num_genres: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.embeddings = nn.ModuleDict(
            {
                node: nn.Embedding(count, hidden_channels)
                for node, count in counts.items()
            }
        )
        self.movie_features = nn.Linear(num_genres, hidden_channels)
        self.layers = nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge: SAGEConv(
                            (hidden_channels, hidden_channels), hidden_channels
                        )
                        for edge in metadata[1]
                    },
                    aggr="mean",
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, graph) -> dict[str, torch.Tensor]:
        x = {node: layer.weight for node, layer in self.embeddings.items()}
        x["movie"] = x["movie"] + self.movie_features(graph["movie"].x)
        for index, layer in enumerate(self.layers):
            x = layer(x, graph.edge_index_dict)
            if index < len(self.layers) - 1:
                x = {
                    node: nn.functional.dropout(
                        value.relu(), self.dropout, training=self.training
                    )
                    for node, value in x.items()
                }
        return x

    @staticmethod
    def decode_link(
        embeddings: dict[str, torch.Tensor],
        head_type: str,
        tail_type: str,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Dot-product decoder for link prediction pairs."""
        head_emb = embeddings[head_type][pairs[:, 0]]
        tail_emb = embeddings[tail_type][pairs[:, 1]]
        return (head_emb * tail_emb).sum(-1)


def create_model(dataset: Any, config: dict[str, Any]) -> HeteroGraphSAGEPretrain:
    return HeteroGraphSAGEPretrain(
        dataset.graph().metadata(),
        {
            "user": len(dataset.users),
            "movie": len(dataset.movies),
            "genre": len(dataset.genres),
        },
        len(dataset.genres),
        hidden_channels=int(config.get("hidden_channels", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    )


def save_model(
    model: HeteroGraphSAGEPretrain,
    output_path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(output_path)
    if target.suffix != ".pt":
        target.mkdir(parents=True, exist_ok=True)
        pt_file = target / "pretrain_gnn.pt"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        pt_file = target

    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "hidden_channels": model.hidden_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }
    if metadata:
        payload.update(metadata)

    torch.save(payload, pt_file)
    return pt_file


def load_model(
    model_path: Path | str,
    dataset: Any,
    device: torch.device | str = "cpu",
) -> tuple[HeteroGraphSAGEPretrain, dict[str, Any]]:
    target = Path(model_path)
    pt_file = target / "pretrain_gnn.pt" if target.is_dir() else target
    if not pt_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {pt_file}")

    payload = torch.load(pt_file, map_location=device, weights_only=False)
    config = payload.get("config", {})
    if not config:
        config = {
            "hidden_channels": payload.get("hidden_channels", 64),
            "num_layers": payload.get("num_layers", 2),
            "dropout": payload.get("dropout", 0.1),
        }

    model = create_model(dataset, config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload
