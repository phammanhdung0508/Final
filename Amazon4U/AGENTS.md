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
No model training or embeddings have been implemented/run. The approved baseline
is now frozen as `amazon4u-protocol-v1.0.0`, state `BASELINE_PROTOCOL_FROZEN`.
Execution settings, tuning policy and independent implementation verification
remain pending. Profiling and comparative training are not authorized. Read
`configs/protocol-lifecycle.json` and `docs/protocol-lifecycle.md`; do not interpret
a state/version change, graph construction or an audit as permission to train.

## Version and authorization rules

- Protocol version, lifecycle state and execution revision are separate identifiers.
  Component/schema versions are not the global experiment version.
- Preserve frozen snapshots/manifests under `configs/protocols/`; append amendments
  with approval, rationale, changed rules and held-out exposure. Never overwrite
  earlier baselines to match later code/configs.
- Major bumps cover split/scope/relevance/candidate/core-comparison changes; minor
  bumps cover substantive methodological amendments/additions; patches are
  behavior-preserving clarifications. State changes alone do not bump versions.
- Bounded profiling needs explicit scope, limits, provisional settings, optimizer
  permission and fresh outputs, after prerequisite regression review. No test model
  evaluation or performance-driven tuning during profiling.
- Before comparative runs, freeze pending initialization/execution/tuning settings,
  verify implementation and record explicit training authorization. Run artifacts
  must include baseline/execution identifiers and hashes, authorization and complete
  code/data/config provenance. Deployment remains separately NOT_AUTHORIZED.

## Read before changing the pipeline

- `README.md`: structure, dependencies and reproduction commands.
- `docs/evaluation-protocol.md`: pre-training protocol and unresolved decisions.
- `configs/dataset-scope.json`: approved full-data scope and execution order.
- `configs/evaluation-metrics.json`: frozen metrics, ranking and reporting rules.
- `docs/reproducibility-and-uncertainty.md` and `configs/reproducibility.json`:
  agreed Block 3 seeds, provenance, resumption, failure and uncertainty rules.
- `docs/compute-and-tuning-plan.md` and `configs/compute-plan.json`: conditional
  Kaggle capacity, profiling-first validation and budget-freeze approach.
- `docs/training-semantics.md` and `configs/training-semantics.json`: agreed
  sampling eligibility, required regression checks and pending training choices.
- `docs/candidates-and-cold-start.md`: authoritative candidate/filtering policy.
- `docs/models-and-baselines.md` and `configs/model-suite.json`: agreed core
  membership, optional comparisons, attribution limits and model claim boundaries.
- `docs/kg-only-scoring.md`: frozen KG-only formulas, missing-channel policy,
  whole-user fallback and required hand-calculated tests.
- `docs/heterogeneous-encoder.md`: frozen relation operator, inverses, activations,
  initialization and required tests; execution/sampling settings remain pending.
- `docs/lightgcn-protocol.md`: explicit core Option B fixed-graph training-edge
  reuse, exception to heterogeneous masking, variant labels and comparison limits.
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
  heterogeneous model's agreed training-target shortcut safeguard is batch
  positive-edge masking:
  exclude direct forward/reverse/duplicate equivalents before or during sampling
  at every hop, not after neighborhood retrieval. Preserve unrelated edges and
  valid remaining multi-hop KG paths. Both BPR scores, if BPR is adopted, use
  the same masked view; no cached unmasked representations may bypass it.
  Interaction encoder edges are binary, with no rating-value inputs. Masks are
  batch-scoped, not permanent history deletion. Add regression checks for effective
  adjacency, sampled neighborhoods and preservation of valid metadata paths.
  Core LightGCN is an explicit approved exception: fixed condition-positive
  training graph with supervised-edge reuse and no batch masks. Held-out edges
  are still forbidden in both models; this is not validation/test leakage.
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
  relaxing exclusions. Block 2 uses uniform eligible sampling, one negative per
  positive; skip/report empty-pool positive updates and fail if no trainable
  examples remain. Never create synthetic or regularization-only updates.
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
  a target or silently omit empty positive profiles. Their agreed common fallback
  is evaluation/inference-only condition-positive training popularity with normal
  candidate/history filters. No synthetic personalized training positives/edges,
  no popularity-vector model input and no personalized-prediction claim for these
  users. Report non-personalized fallback counts and subgroup metrics separately;
  scoring failures must still fail.
- A heterogeneous GraphSAGE-style recommender using learned node-ID embeddings
  is transductive, not automatically an inductive cold-start model. Learned-ID
  inputs and encoder initialization are now agreed in Block 1; no learned
  features/embeddings have been generated in KG-v1.
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
  LightGCN is not optional. Block 1 freezes 64d BPR-MF/dot product, 64d LightGCN
  with 3 layers/mean 0–3/dot product, and 2-layer 64d learned-ID heterogeneous
  GraphSAGE-style/dot product. Sampling/profiling execution details remain pending.
  Exact heterogeneous aggregation/initialization
  are now frozen in `docs/heterogeneous-encoder.md`: relation-specific transforms,
  one node-type self term, within-relation means and mean over active relations;
  explicit inverses, ReLU then identity, Normal(0,std=0.1) IDs, Xavier-uniform
  gain-1 matrices, zero self biases, no dropout/layer/output normalization.
  Interaction inputs are binary and rating-free.
- KG-only uses training-warm IDF profiles and cosine, no rating weighting or
  popularity bonus. Tune category/brand mixtures in order 1/0, 0.75/0.25, 0.5/0.5
  by validation NDCG@10 per category/condition; exact ties select the first entry.
  Keep profile/IDF/missing-channel/candidate rules identical across trials.
  IDF is 1+ln((N+1)/(df+1)) over all condition-warm products, with distinct-product
  df. Normalize product channels separately; user channels are normalized sums
  over distinct positive-history products. Missing-channel cosine is zero, with
  no per-item mixture renormalization. Whole-user condition-positive popularity
  fallback applies only if no positively weighted user profile is usable, never
  per-item or for scoring failures. Report fallback counts/reasons/subgroup metrics.
  Empty-positive-history fallback is shared across methods; KG-only's additional
  missing-metadata fallback is distinct. Neither creates training examples or
  model input features. Do not use test outcomes to select weights.
- Core LightGCN uses approved Option B, exact propagation on the fixed training
  graph without batch target masks. Profile execution, not methodological identity.
  Masked Option A is a target-edge-masked variant, not the core baseline. Sampled
  propagation requires an explicit amendment and sampled-variant labeling.
  Heterogeneous KG + GNN retains masking; do not attribute comparison gains solely
  to metadata when architecture and training-edge use also differ.
  GraphSAGE batch size/fanout are profiling starting points, not frozen constants.
- BPR-MF + metadata is an optional feature-augmentation comparison, not a clean
  KG + GNN ablation. A matched encoder without metadata relations is a recommended
  optional control; do not claim metadata attribution from architecture-changing
  comparisons alone. No optional experiment has a finalized runnable configuration.

## Agreed Block 2 training recipe

- Learned methods: mean stable BPR via softplus(s_negative-s_positive), plus the
  explicit endpoint-only penalty below; uniform eligible negatives, one per positive.
- Adam, betas (0.9,0.999), epsilon 1e-8, zero weight decay; constant learning rate
  0.001, maximum 100 epochs. Stop after 10 validation checks without strictly better
  validation NDCG@10; exact ties retain the earliest checkpoint and do not reset
  patience. Failed validation is a failure, not a valid patience check.
- Empty negative pools skip/report affected positives; fail if no examples remain
  trainable. Invalid empty batches have no optimizer/regularization-only update.
- Model-specific batch sizes, fanouts, validation frequency, MF/LightGCN initialization
  and tuning budgets still require specification/profiling before training.
  Numerical approvals do not authorize training until the overall protocol is ready.

## Agreed endpoint regularization

- Learned methods use explicit 1e-5 times the mean squared L2 row norm of unique
  supervised base-ID endpoints: users plus union of positive/negative item IDs.
  Deduplicate per typed ID across roles/examples; no extra dimension or half factor.
- Do not select rows from the sampled computation graph or full propagation table.
  Brand/Category rows are always excluded; a neighbor user/product row is included
  only if it independently is a supervised endpoint. Use base rows, not propagated
  representations.
- Transformation matrices/biases and global optimizer weight decay are zero.
  No hidden table-wide or decoupled weight decay may replace the explicit penalty.
  Ranking-loss gradients through neighbors remain allowed; do not detach them.
- Test endpoint deduplication, fanout independence of the penalty, base-row selection
  and optimizer groups. No synthetic/regularization-only update for invalid empty
  batches. Other training settings remain pending; see training-semantics documents.

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
- Block 3 freezes seeds 42/2026/3407 across learned methods/groups. Record RNG
  streams/workers, environment and reconstructable source provenance (not just
  dirty flags). Fail first on unsupported determinism; explicitly approve/log
  exceptions. Claim resumption only at boundaries independently verified.
- Report all registered metrics per seed, mean ± sample SD, and fallback subgroups.
  Bootstrap fixed seed-averaged per-user test metrics after validation-only selection:
  10,000 shared paired-user resamples, seed 12345, 95% percentile intervals and
  full-test-population point estimates. Do not bootstrap coverage per user or
  conflate user/seed uncertainty. The contrast list is frozen: KG + GNN minus
  Popularity/BPR-MF/LightGCN/KG-only for NDCG@10/HitRate@10 per category/condition
  (48 planned intervals). Share resamples across all contrasts/metrics within a
  group; no bootstrap for @5/@20 or coverage and no optional-method contrasts
  without pre-experiment amendment. Unadjusted intervals are descriptive,
  empty populations N/A, failed seeds explicit.
  No hidden population changes, seed replacement or favorable-result retries.
- Other training choices remain unresolved. Conditional 60 GPU-hours/week must
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
