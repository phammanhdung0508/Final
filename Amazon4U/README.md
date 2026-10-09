# Amazon4U dataset exploration

Agreed scope: **full downloaded benchmark splits** for Electronics,
Toys_and_Games and Musical_Instruments, evaluated separately under both feedback
conditions. Start development/resource checks on Musical_Instruments, then
Toys_and_Games and Electronics. No research subsampling or evaluation-user sampling.
See `configs/dataset-scope.json` and the scope decision in
`docs/evaluation-protocol.md`. Training still requires the remaining protocol freeze.
Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

## Reproduce

```sh
python scripts/download_5core.py --dry-run
python scripts/download_5core.py
# Set up analysis dependencies (Python 3.11+):
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/inspect_data.py
.venv/bin/python scripts/audit_kg_inputs.py
.venv/bin/python scripts/audit_candidates.py
.venv/bin/python scripts/build_kg.py
.venv/bin/python scripts/validate_kg_splits.py
.venv/bin/python -m unittest discover -s tests -v
```

Run from this directory; paths default to this project's data directory regardless of working directory.
Downloader requires curl, pins the source revision per run, handles paginated listings,
resumes partial downloads, and verifies file sizes (not content hashes). Use
`--kind benchmark` or `--kind metadata` to download just one collection.

## Files

- `data/raw/benchmark/5core/last_out/`: full train/valid/test interaction CSVs.
- `data/raw/raw_meta_<Category>/`: full product metadata Parquet shards.
- `data/raw/download_manifest_all.json`: source revision, file paths and sizes.
- `data/reports/inspection.json`: schemas, counts, examples, and metadata join coverage.
- `data/reports/kg-input-audit.json`: metadata quality, split integrity and cross-category timing.
- `data/reports/candidate-audit.json`: warm/cold, history-filtering and metadata vocabulary counts for both feedback conditions.
- `data/processed/kg-v1/`: structurally validated per-category training KGs and provenance manifest.
- `data/reports/kg-split-validation.json`: independent source/split validation.
- `data/*.log`: execution logs.

See [KG schema](docs/kg-schema.md), [audit findings](docs/kg-input-audit.md),
[graph configuration](configs/kg-v1.json) and the draft
[evaluation protocol](docs/evaluation-protocol.md) and
[candidate/cold-start rules](docs/candidates-and-cold-start.md).
Metrics/ranking/reporting are frozen in `configs/evaluation-metrics.json`:
NDCG@10 primary; HitRate@10 and CatalogCoverage@10 secondary; NDCG/HitRate
at 5 and 20 supplementary, reported per category × feedback condition.
Graphs are built; models are not trained. Other protocol decisions remain pending.
The builder refuses to overwrite existing output; use `--output` for a new build.

The agreed collections total approximately **4.31 GB** (26 files). Older Books and
Home_and_Kitchen benchmark downloads remain untouched; they are not included in this total.
Raw files, processed graphs, the local virtual environment and execution logs are ignored by Git.
Human-readable documentation, configuration, scripts, tests and JSON reports remain trackable.

## Visualize a training-KG neighborhood

Requires the system Graphviz executable `dot` in addition to Python dependencies.

```sh
.venv/bin/python scripts/visualize_kg.py --category all
# Or specify a category/user and bound the view:
.venv/bin/python scripts/visualize_kg.py --category Musical_Instruments --max-products 4
```

Open `data/visualizations/Musical_Instruments.html` in a browser for zoom controls
and hover details. Electronics and Toys_and_Games have equivalent views. SVG,
PNG, Graphviz DOT and exact sampled JSON are saved beside each HTML file.
Use `--user-id` for a particular user's training neighborhood. The default selects
one bounded-degree user deterministically and shows up to six earliest training
products. This is an illustrative, non-random view, not the full graph or an
evaluation sample. All pictured nodes/edges come from persisted KG-v1 files.

## First inspection

| Category | Metadata rows | Benchmark interactions (all splits) |
|---|---:|---:|
| Electronics | 1,610,012 | 15,473,536 |
| Toys_and_Games | 890,874 | 3,861,886 |
| Musical_Instruments | 213,593 | 511,836 |

All interactions in these downloaded splits matched a metadata `parent_asin`.
CSV columns: `user_id`, `parent_asin`, `rating`, `timestamp`.
Benchmark CSVs do **not** contain full review text.

Metadata includes titles, feature/description lists, categories, price, store,
images and other fields. `details` is a string, not a typed attribute dictionary;
inspect/parse its contents before constructing KG relations. Do not equate
`store` with brand without checking the records.

The approved research scope uses full downloaded splits, not a 1,000-interaction
sample. Tiny fixtures serve correctness tests only. Any future resource-driven
subset requires an approved, reproducible scope amendment before comparative
results. Published 5-core guarantees should not be assumed to hold on the
training split alone.

Use training interactions only when constructing the recommendation interaction
graph. Product-level `average_rating` and `rating_number` can incorporate future
reviews; do not use them as temporally safe features without addressing leakage.
Check the source dataset's usage terms and citation guidance before redistribution.
