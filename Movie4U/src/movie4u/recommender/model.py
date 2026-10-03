"""Two-layer heterogeneous GraphSAGE with explicit Genre nodes."""

from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        metadata,
        counts,
        num_genres,
        hidden_channels=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
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
        self.dropout = dropout

    def forward(self, graph):
        x = {node: layer.weight for node, layer in self.embeddings.items()}
        x["movie"] = x["movie"] + self.movie_features(graph["movie"].x)
        for index, layer in enumerate(self.layers):
            x = layer(x, graph.edge_index_dict)
            # Keep final embeddings signed so the dot-product decoder can
            # produce negative logits for the binary cross-entropy objective.
            if index < len(self.layers) - 1:
                x = {
                    node: nn.functional.dropout(
                        value.relu(), self.dropout, training=self.training
                    )
                    for node, value in x.items()
                }
        return x

    @staticmethod
    def decode(embeddings, pairs):
        return (embeddings["user"][pairs[:, 0]] * embeddings["movie"][pairs[:, 1]]).sum(
            -1
        )


def create_model(dataset, config: dict):
    return HeteroGraphSAGE(
        dataset.graph().metadata(),
        {
            "user": len(dataset.users),
            "movie": len(dataset.movies),
            "genre": len(dataset.genres),
        },
        len(dataset.genres),
        config["hidden_channels"],
        config["num_layers"],
        config["dropout"],
    )
