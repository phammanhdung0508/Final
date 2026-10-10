# Agent instructions — Amazon4U

Scope: this directory and all descendants. Amazon4U is independent of Movie4U;
do not copy MovieLens split rules, candidate rules or ID assumptions here, and do
not change sibling projects unless explicitly requested.

## Goal and current state

Compare KG-only and KG + GNN methods for **top-K Amazon product recommendation**,
with popularity and collaborative-filtering baselines under shared evaluation rules.
Agreed scope: **full downloaded splits** for Electronics, Toys_and_Games and
Musical_Instruments, evaluated independently. Develop/resource-check on
Musical_Instruments first, then Toys_and_Games and Electronics. No research
interaction subsampling or evaluation-user sampling; fixtures are correctness-only.
Record and obtain approval for any resource-driven scope amendment before using
it or observing comparative results. See `configs/dataset-scope.json`.

Full 5-core `last_out` benchmark splits and root-level Parquet product metadata
are downloaded. Separate, all-rating training KGs are built and validated.
Scripts cover downloading, inspection, audits, graph construction and visualization.
No model training or embeddings have been implemented/run. Training is blocked
until the remaining evaluation decisions are frozen; do not interpret graph
construction or a metadata audit as authorization to train.

## Read before changing the pipeline

- `README.md`: structure, dependencies and reproduction commands.
- `docs/evaluation-protocol.md`: pre-training protocol and unresolved decisions.
- `configs/dataset-scope.json`: approved full-data scope and execution order.
- `configs/evaluation-metrics.json`: frozen metrics, ranking and reporting rules.
- `docs/compute-and-tuning-plan.md` and `configs/compute-plan.json`: conditional
  Kaggle capacity, profiling-first validation and budget-freeze approach.
- `docs/training-semantics.md` and `configs/training-semantics.json`: agreed
  sampling eligibility, required regression checks and pending training choices.
- `docs/candidates-and-cold-start.md`: authoritative candidate/filtering policy.
- `docs/models-and-baselines.md` and `configs/model-suite.json`: agreed core
  membership, optional comparisons, attribution limits and model claim boundaries.
- `docs/kg-schema.md` and `configs/kg-v1.json`: implemented structural allowlist.
- `docs/kg-input-audit.md`: actual split/metadata findings and limitations.

Keep documentation, configuration, implementation and tests consistent. Do not
resolve protocol ambiguities silently. Record amendments before new experiments;
never choose thresholds, features or metrics based on test performance.

## Mandatory decision recording

- Always record project decisions made during discussions and implementation;
  do not leave agreed choices only in chat or rely on conversation memory.
- Update the relevant document in `docs/` and configuration, where applicable,
  before implementing or running work governed by that decision. This applies
  throughout the project, not only to evaluation and training.
- Record what was decided, why, its scope, and its status: proposed, agreed,
  implemented or superseded. Distinguish user-approved decisions from agent
  recommendations and unresolved questions; never present a proposal as agreed.
- Record changes to earlier decisions with their rationale and affected artifacts;
  preserve the previous decision's context instead of silently rewriting history.
- Keep implementation progress, verification results and remaining work current
  in the relevant documentation. Link supporting scripts, tests and reports.
- Before concluding a task, check that all new decisions and implemented changes
  are documented. Report unresolved choices explicitly. Documentation updates
  do not authorize staging, committing or pushing without a user request.

## Data and graph invariants

- Raw data in `data/raw/` is immutable. Preserve downloaded files, publisher split
  assignments and provenance. Never randomly re-split or move a held-out target.
- Join reviews/interactions and products using `parent_asin`; users use `user_id`.
  Benchmark CSVs contain rating/timestamp interactions, **not full review text**.
- Keep category graphs separate. Cross-category histories/embeddings are not
  approved: one category's training event can occur after another's target.
- KG-v1 nodes: User, Product, CategoryPath, Brand. Relations: `rated`, `belongs_to`,
  `category_child_of`, `has_brand`. Only training users/products become nodes.
- Metadata allowlist: `parent_asin`, observed `categories` path prefixes and
  explicit string `details.Brand`. Do not treat `store`/`Manufacturer` as brand.
  Category identities are complete JSON path prefixes, not ambiguous leaf names.
- Missing metadata does not justify fabricated edges or silently removing items.
  Fail on duplicate training pairs/metadata IDs and invalid structural inputs.
- KG-v1 stores all training ratings as observations, not binary likes. It has no
  approved node features/embeddings. The >=4 graph is a distinct future artifact.
- Use bounded-memory/chunked or disk-backed processing. Existing DuckDB audits
  cap memory at 512 MB and spill to temporary workspaces. Do not load full datasets
  into RAM or download additional large collections without explaining the cost.

## Leakage safeguards

- Training graphs, inverse edges, user profiles, statistics and graph embeddings
  derive only from permitted training data. Never expose held-out labels/reviews.
- A target review/rating must not become an input for predicting itself. The
  agreed training-target shortcut safeguard is batch positive-edge masking:
  exclude direct forward/reverse/duplicate equivalents before or during sampling
  at every hop, not after neighborhood retrieval. Preserve unrelated edges and
  valid remaining multi-hop KG paths. Both BPR scores, if BPR is adopted, use
  the same masked view; no cached unmasked representations may bypass it.
  Interaction encoder edges are binary, with no rating-value inputs. Masks are
  batch-scoped, not permanent history deletion. Add regression checks for effective
  adjacency, sampled neighborhoods and preservation of valid metadata paths.
- Exclude `average_rating`, `rating_number`, bestseller ranks, `bought_together`,
  full details dictionaries and dataset-wide/review-derived aggregates. Other
  metadata/features require an audited, explicitly registered allowlist change.
- Fit preprocessing, vocabularies, normalization, PCA and learned encoders only
  on permitted training data. Record provenance for external frozen encoders.
- Validation selects hyperparameters/checkpoints; test never selects them.
- Preserve regression checks showing held-out changes and excluded-field changes
  cannot alter graph/model-input tables.

## Feedback, candidates and cold start

- Primary: all-rating recorded behavior. Sensitivity: training/held-out ratings
  >=4 as positive feedback. Low ratings/unobserved items are not verified negatives.
- Preserve raw splits in both conditions; do not replace low-rated held-out items.
- For >=4 training, keep positive history separate from all-rating observed
  training history. Positives/forward/reverse-positive edges use >=4 only;
  negative eligibility is the condition warm pool minus ALL user training-rated
  items. A globally warm item rated below 4 by that user is forbidden as their
  negative. Never consult held-out data. Detect/report empty pools without
  relaxing exclusions; the skip/fail policy and sampling parameters are pending.
- Warm pool derives solely from permitted training interactions. Current >=4
  policy uses condition-specific warmth; explicitly report the resulting pool
  difference, not a fixed-candidate threshold-only ablation.
- Full candidate ranking, **no sampled evaluation negatives**. Use identical
  candidates and relevance rules across methods within each condition.
- Validation filters all user training items. Test filters all user training
  **and validation** items, including low ratings. Validation IDs are allowed only
  for candidate subtraction, not model updates, features or message passing.
- Freeze the selected model for test; no train+validation refit under this policy.
- Exclude cold/history-overlapping targets from primary warm metrics and report
  counts, proportions and eligible-user denominators. Never selectively restore
  a target or silently omit empty positive profiles; freeze a common fallback.
- A heterogeneous GraphSAGE-style recommender using learned node-ID embeddings
  is transductive, not automatically an inductive cold-start model. Learned-ID
  initialization is still a proposal; no features/embeddings exist in KG-v1.
  Report capability based on the implemented inputs/inference pathway, not the
  GraphSAGE name alone. See `docs/models-and-baselines.md`.
- Cold performance is not currently supported/evaluated. Only register it after
  verifying metadata-inductive inference, no cold-ID embeddings, learned warm
  metadata vocabulary and training-only preprocessing. ID-only models are N/A,
  not valid cold baselines. Report unsupported items/unseen metadata separately.
- Audit unknown users rather than assuming 5-core makes them impossible. Full
  5-core does not imply training alone is 5-core. Timestamp ties and static
  metadata availability remain documented limitations.

## Agreed core models

- Core: Popularity, BPR-MF, LightGCN, KG-only and Heterogeneous KG + GNN.
  LightGCN is not optional. Model-specific settings are not yet frozen.
- BPR-MF + metadata is an optional feature-augmentation comparison, not a clean
  KG + GNN ablation. A matched encoder without metadata relations is a recommended
  optional control; do not claim metadata attribution from architecture-changing
  comparisons alone. No optional experiment has a finalized runnable configuration.

## Frozen metrics and reporting

- Primary: NDCG@10. Secondary: HitRate@10 and CatalogCoverage@10.
  Supplementary: NDCG/HitRate at 5 and 20. Select checkpoints/hyperparameters with
  validation NDCG@10 only; do not choose K using test outcomes.
- Binary relevance follows the feedback condition. Macro-average NDCG/HitRate
  over eligible users; coverage is a unique-recommendation union divided by the
  condition's entire warm pool, not a per-user average.
- Filter history before Top-K. Sort score descending, then parent_asin ascending
  for exact ties. Reject non-finite scores. Do not pad lists shorter than K;
  report affected-user counts and never silently drop scoring failures.
- Report separately by category × feedback condition; no pooled primary score.
  Use the same eligible users across methods within a group. Separate validation
  from final test reporting and include eligibility/exclusion, empty-profile,
  fallback and candidate-shortage counts. Label overlapping exclusion reasons.
- Three final training seeds are agreed; exact values/uncertainty procedures and
  other training choices remain unresolved. Conditional 60 GPU-hours/week must
  be verified as permitted/available; reserve roughly 20%. Profile full-ranking
  validation every epoch on Musical_Instruments first, then recheck larger
  categories. Freeze trial budgets/search spaces before comparative results;
  2–3 configurations is only a planning range. Frequency amendments need approval,
  with patience counted in validation checks. Freezing metrics does not authorize
  training or silently reduce dataset/candidate scope.

## Development and verification

Save reusable tooling in `scripts/`, policies in `configs/`/`docs/`, tests in
`tests/`, generated graphs in `data/processed/`, reports in `data/reports/` and
visualizations in `data/visualizations/`. Visualizations are illustrative training
neighborhoods, not random or representative evaluation samples.

From `Amazon4U/`:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q scripts tests
```

When data/graph logic changes, also run the affected full-data audits as feasible:

```sh
.venv/bin/python scripts/audit_kg_inputs.py
.venv/bin/python scripts/audit_candidates.py
.venv/bin/python scripts/validate_kg_splits.py
```

Install dependencies with `.venv/bin/python -m pip install -r requirements.txt`
when necessary; `curl` is required for downloads and Graphviz `dot` for rendering.
Builders do not overwrite existing graphs: use a new `--output` for experimental
builds and validate that artifact explicitly. Preserve source hashes, configuration,
split/condition identity and eventually seeds/checkpoint protocol with each run.

Keep changes small and tested. Do not commit raw data, large generated graphs,
virtual environments or caches. Check root and local ignore rules with
`git check-ignore` before assuming reports are tracked. Do not stage, commit or
push changes unless requested; leave unrelated user changes untouched.
