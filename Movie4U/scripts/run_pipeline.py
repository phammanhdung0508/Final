"""Build, validate, split, train, and compare both recommenders."""

import argparse
import json
from pathlib import Path
import torch
import yaml
from movie4u.kg.builder import build_knowledge_graph
from movie4u.kg.validator import validate_knowledge_graph
from movie4u.recommender.dataset import prepare_splits, GraphDataset
from movie4u.recommender.train import train
from movie4u.recommender.baseline import KGOnlyRecommender
from movie4u.recommender.model import create_model
from movie4u.recommender.evaluate import ranking_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    data_config = yaml.safe_load((root / "configs/data.yaml").read_text())
    config = yaml.safe_load((root / "configs/model.yaml").read_text())
    threshold = yaml.safe_load((root / "configs/kg.yaml").read_text())[
        "positive_rating_threshold"
    ]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    kg = root / data_config["kg_output_dir"]
    split_dir = root / data_config["gnn_output_dir"]
    print(build_knowledge_graph(str(root / data_config["movielens_dir"]), str(kg)))
    validate_knowledge_graph(str(kg))
    print(prepare_splits(str(kg), str(split_dir), threshold))
    dataset = GraphDataset(kg, split_dir, threshold)
    if not args.skip_train:
        train(dataset, config, root / "artifacts/models")
    payload = torch.load(
        root / "artifacts/models/gnn.pt", weights_only=True, map_location="cpu"
    )
    if payload.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError(
            "Checkpoint does not match current graph/history; rerun without --skip-train"
        )
    torch.set_num_threads(payload["config"].get("cpu_threads", 4))
    model = create_model(dataset, payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        x = model(dataset.graph())
        scores = (x["user"] @ x["movie"].T).numpy()
    results = {
        "protocol": "full-catalog ranking, fixed training history, per-user chronological holdout; users without positive holdout excluded",
        "positive_threshold": threshold,
        "dataset_fingerprint": dataset.fingerprint,
        "gnn_best_epoch": payload["epoch"],
        "models": {},
    }
    for name, values in [
        ("kg_only", KGOnlyRecommender(dataset).scores()),
        ("kg_gnn", scores),
    ]:
        results["models"][name] = {
            split: ranking_metrics(values, dataset, split, config["top_k"])
            for split in ["validation", "test"]
        }
    out = root / "artifacts/metrics/comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
