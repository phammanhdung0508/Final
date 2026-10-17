# Amazon4U dataset exploration

Agreed scope: **full downloaded benchmark splits** for Electronics,
Toys_and_Games and Musical_Instruments, evaluated separately under both feedback
conditions. Start development/resource checks on Musical_Instruments, then
Toys_and_Games and Electronics. No research subsampling or evaluation-user sampling.
See `configs/dataset-scope.json` and the scope decision in
`docs/evaluation-protocol.md`. The baseline protocol is frozen as
**`amazon4u-protocol-v2.0.0`**, in lifecycle state **`IMPLEMENTATION_REGRESSION_VERIFICATION`**.
**Evaluator integration is complete in the main checkout**: `recommendation_core.py`
uses the shared engine in `scripts/recommendation_evaluation.py`, with guarded KG-only
fallback. The preserved evaluator artifact received **ACCEPT (task_only)**; the
integrated main suite initially passed **113 tests, zero failures/errors/skips**. See
[the integration record](docs/evaluator-integration.md) for acceptance and evidence.
Remaining I1 scorer/sampler/reference-loss verification is now also **ACCEPTED
(task_only)** under task `cc64f3cb-9a4f-40aa-abbf-5ced6c1bd000`; current v2 full
suite: **115 tests, zero failures/errors/skips**. Step 3a learned-model foundation
and core LightGCN reference implementation are also **ACCEPTED (task_only)** under
task `8153df65-e2e2-4571-a4ce-340d0ffeb1c2`; Step 3b Heterogeneous KG+GNN
reference encoder foundation is **ACCEPTED (task_only)** under task
`d35b3d7c-3cd6-4f8e-8ecf-adcb963f4b93`. Differentiable CPU models/training-only
graphs have frozen-verification **ACCEPT (task_only)** under
`f709d062-ae0e-4eb6-960c-f94728180d1b`, with separately reviewed main import
integration (**204 tests** at that milestone). Separately accepted D1 identity/provenance
is now integrated in main with **249 passing tests**.

**Latest D2 result:** explicitly bounded correction
`6efb0ae5-6a62-4376-91e9-7b358c51013f` is **ESCALATE at1/1**, despite its isolated
**283 passing tests**. Independent review found remaining strict-schema, CPU RNG
applicability, bounded-diagnostic, required coverage and disclosure issues. No rejected
payload source was transferred to main; D3 and real profiling/training remain blocked.
See [the current correction/result](docs/checkpoint-payload-authorized-correction.md).
Historical tasks/budgets remain preserved; no automatic replacement is authorized.

Full training harness remains unfinished. Checkpoint/provenance task
`e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a` is **ESCALATE / not accepted** after
the authorized final revision timed out; both revisions are used. Independent
benchmark: 168 tests, 6 errors. No checkpoint draft transferred to main; no
automatic retry or revision reset. See [final review](docs/checkpoint-final-review.md).
See [the harness review](docs/training-harness-review.md).
Next steps are in [the implementation roadmap](docs/implementation-roadmap.md).
Execution settings remain pending; profiling and comparative training are **not authorized**.
Deployment remains **NOT_AUTHORIZED**.

The active suite is **Popularity, LightGCN, KG-only and Heterogeneous KG+GNN**.
The user-approved [v2.0.0 amendment](docs/bpr-mf-removal-amendment.md) removes BPR-MF
and its metadata variant entirely, while retaining BPR loss for the two learned
models. The v1.0.0 frozen snapshot remains unchanged; the bootstrap plan now has
three baseline contrasts and 36 descriptive intervals.
See [version/lifecycle policy](docs/protocol-lifecycle.md) and
`configs/protocol-lifecycle.json` for separate version, state and permission records.
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

- `scripts/recommendation_core.py`: shared-evaluator subclass with guarded KG fallback, Popularity/KG-only scorers, negative sampler, reference loss/regularization, and learned-model re-exports.
- `scripts/recommendation_evaluation.py`: authoritative shared context/error and candidate/filtering/ranking/metrics engine.
- `scripts/learned_models.py`: Step 3a/3b LightGCN and Heterogeneous KG+GNN reference graph/propagation/scorer, loss-composition, masking, training guard and checkpoint manifest helpers.
- `tests/test_recommendation_core.py`: milestone I1 hand-calculated regression fixtures and isolation tests.
- `tests/test_recommendation_evaluation.py`: shared evaluator and adversarial candidate-isolation regressions.
- `tests/test_evaluator_integration.py`: core/shared API identity, guarded fallback and exact metric integration oracles.
- `tests/test_learned_models.py`: synthetic Step 3a LightGCN and training-harness foundation tests.
- `tests/test_heterogeneous_kgnn.py`: synthetic Step 3b Heterogeneous KG+GNN graph, masking, propagation, scorer and loss-composition tests.
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
[graph configuration](configs/kg-v1.json) and the baseline-frozen
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
