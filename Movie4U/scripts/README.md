# Pipeline entry point

From the project root after installing the Python package:

```sh
python scripts/run_pipeline.py
```

This builds and validates the KG, creates chronological splits, trains GraphSAGE, selects a checkpoint using validation NDCG@10, and evaluates both approaches on the full unseen catalog.

Use `--epochs 5` for a smoke run, or `--skip-train` to evaluate an existing checkpoint. The default project path is inferred from the script location; `--root` overrides it.

Raw CSV files are never modified. Generated outputs are stored under `data/processed/` and `artifacts/`.
