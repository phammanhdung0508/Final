import json
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GATConv

class HeteroKGAT(nn.Module):
    def __init__(
        self,
        metadata,
        counts,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.embeddings = nn.ModuleDict({
            node: nn.Embedding(count, hidden_channels)
            for node, count in counts.items()
        })
        
        # If node type has x (features), we need a linear layer. For rich KG, we only use embeddings.
        # So we skip self.movie_features since we moved everything to Graph edges.
        

        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GATConv(
                    (hidden_channels, hidden_channels), 
                    hidden_channels // heads, 
                    heads=heads, 
                    add_self_loops=False, 
                    dropout=dropout
                )
            self.layers.append(HeteroConv(conv_dict, aggr="mean"))
            
            # Skip connection
            self.skips.append(nn.ModuleDict({
                node: nn.Linear(hidden_channels, hidden_channels)
                for node in metadata[0]
            }))



    def forward(self, graph) -> dict[str, torch.Tensor]:
        x = {node: layer.weight for node, layer in self.embeddings.items()}
        for index, (layer, skip_dict) in enumerate(zip(self.layers, self.skips)):
            out = layer(x, graph.edge_index_dict)
            x_new = {}
            for node, value in x.items():
                if node in out:
                    x_new[node] = out[node] + skip_dict[node](value)
                else:
                    x_new[node] = skip_dict[node](value)
            
            x = x_new
            if index < len(self.layers) - 1:
                x = {
                    node: nn.functional.dropout(value.relu(), self.dropout, training=self.training)
                    for node, value in x.items()
                }
        return x



    @staticmethod
    def decode_link(embeddings: dict[str, torch.Tensor], src_type: str, dst_type: str, pairs: torch.Tensor) -> torch.Tensor:
        return (embeddings[src_type][pairs[:, 0]] * embeddings[dst_type][pairs[:, 1]]).sum(-1)

    @staticmethod
    def decode(embeddings: dict[str, torch.Tensor], pairs: torch.Tensor) -> torch.Tensor:
        return (embeddings["user"][pairs[:, 0]] * embeddings["movie"][pairs[:, 1]]).sum(-1)

def create_model(dataset: Any, config: dict[str, Any]) -> HeteroKGAT:
    metadata = dataset.graph().metadata()
    counts = {
        "user": len(dataset.users),
        "movie": len(dataset.movies),
        "genre": len(dataset.genres),
        "year": len(dataset.years),
        "gender": len(dataset.genders),
        "age": len(dataset.ages),
        "occupation": len(dataset.occupations),
    }
    return HeteroKGAT(
        metadata,
        counts,
        hidden_channels=int(config.get("hidden_channels", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
        heads=int(config.get("heads", 2)),
    )

def save_model(model: HeteroKGAT, output_path: Path | str, metadata: dict[str, Any] | None = None) -> Path:
    target = Path(output_path)
    if target.suffix != ".pt":
        target.mkdir(parents=True, exist_ok=True)
        pt_file = target / "gnn.pt"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        pt_file = target
    payload = {
        "state_dict": model.state_dict(),
        "hidden_channels": model.hidden_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, pt_file)
    meta_file = pt_file.parent / "model_config.json"
    clean_meta = {k: v for k, v in (metadata or {}).items() if isinstance(v, (str, int, float, bool, list, dict))}
    clean_meta["hidden_channels"] = model.hidden_channels
    clean_meta["num_layers"] = model.num_layers
    clean_meta["dropout"] = model.dropout
    meta_file.write_text(json.dumps(clean_meta, indent=2))
    return pt_file

def load_model(model_path: Path | str, dataset: Any, device: torch.device | str = "cpu"):
    target = Path(model_path)
    pt_file = target / "gnn.pt" if target.is_dir() else target
    payload = torch.load(pt_file, map_location=device, weights_only=False)
    config = payload.get("config", {
        "hidden_channels": payload.get("hidden_channels", 64),
        "num_layers": payload.get("num_layers", 2),
        "dropout": payload.get("dropout", 0.2),
    })
    model = create_model(dataset, config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload
