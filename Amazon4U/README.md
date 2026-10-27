# Amazon4U dataset exploration

## Research questions

Status: **agreed in discussion with the user (2026-10-27)**, recorded at the user's request
("in case i forgot can you note RQ we agree on to README.md"). Not yet frozen into a
protocol snapshot. That happens before the test run, which still needs its own approval.
All four are answered from one test run of the already-trained models, with no retraining.

- **Main question.** Does knowledge-graph information (brand, category) improve top-K
  recommendations over methods without it? This compares Popularity, LightGCN, KG-only and
  Heterogeneous KG + GNN.
- **RQ1. Similar products.** When users buy a product, how often do they later buy a similar
  one, and do the recommendation models suggest similar products as often as users really
  need them? "Similar" means the same sub-category with matching details such as brand.
  It is compared with the natural rate of similar next purchases in the training data,
  split into one-time products (such as guitars) and repeat products (such as strings).
- **RQ2. Complementary products.** Do the models recommend complementary products, of a
  different type that goes with what the user bought (such as strings or a capo after a
  guitar)? This is measured as correct recommendations from a different sub-category.
- **RQ3. Brand loyalty.** Do the recommenders capture brand loyalty, recommending products
  from brands the user bought before? This is compared with how often users actually return
  to a brand.

Notes:

- The original RQ3, re-buying the exact same product, cannot be measured. The training data
  has no repeat purchase of the same product by the same user, and the evaluation removes
  products a user already rated. It is future work.
- Training-data check for RQ3 (Musical_Instruments): about 13% of later branded purchases
  come from a brand the user bought before, against about 3% by chance. 86% of ratings
  are 4 or 5 stars, so "liked" adds little over "bought".
- Graph expansion (product details such as material) is not needed for RQ1. Details may be
  used to measure similarity, not as model input. An LLM explanation and an agent
  recommender are not needed for the research questions. Both are future work.
- Graph enrichment options (proposed, not approved), including LLM fact extraction from
  `title`, `features` and `description`: see
  [`docs/kg-metadata-enrichment-options.md`](docs/kg-metadata-enrichment-options.md).
- Validation so far (NDCG@10, all ratings): LightGCN 0.0313, KG + GNN 0.0216,
  Popularity 0.0157, KG-only 0.0052. Test results have not been seen. See
  [`docs/kg-only-extended-brand-amendment.md`](docs/kg-only-extended-brand-amendment.md).

**Main O-OFFLINE integration: ACCEPT / task_only (2026-10-24). Deployment: NOT_AUTHORIZED.**
Main now contains the accepted offline Kaggle adapter (`multiautoresearch/adapters/`,
updated `post-training/scripts/kaggle_job.py`, two O test modules). All seven files match
the accepted isolated artifact byte for byte; the previous runner is archived. Main
verification: 17 mock infrastructure checks, 103 O-only tests and the full 515-test
regression all passed with 0 failures/errors/skips (68.2s child, ~784 MiB peak RSS,
within 240s/2 GiB). All original 412 test IDs, 34 D3 methods and 3,564 expanded cases
still pass; the 31 inherited Python files and H28 bundle are unchanged. No unresolved
integration issue remains. Native15 stays historical ESCALATE.
Scope: offline packaging and fake-provider behavior only. Live Kaggle jobs, GPU-READY,
production LightGCN initialization, real-data training, profiling, comparative and
final-test runs each still need their own plan and approval.
Decision: `data/reports/o-offline-main-integration-v1/review.json`;
plan: `docs/o-offline-main-integration-plan.md`;
design: `docs/o-offline-orchestration.md`.

**Main H-REF integration: ACCEPT / task_only. Deployment: NOT_AUTHORIZED.**
[Current independent main acceptance](docs/training-harness-main-integration-recovery.md):
strictly scoped Pi-only scratch-monitor recovery,17mock checks PASS and full412PASS,
0failures/errors/skips;75.717s child/~758MiB RSS,within240s/2GiB. All389inherited/34D3/
23Hmethods/3564expanded cases,strict empty scratch/cache ownership/preservation pass.
The six H files and H8 are unchanged. No Worker/SDK or model correction. SDK4/native1,
7cumulative H-associated fullstarts;all current slots consumed. Old failed evidence/
residual scratch remain unchanged. CPU-reference task acceptance is not O/GPU/production/
profiling/experiment readiness;no further execution or Git actions automatically.

**Previous main H-REF integration attempt: ESCALATE / preserved history.**
[Current integration result and RCA](docs/training-harness-main-integration.md):
9mock checks and fresh389baseline passed;the exact six H files were transferred,but
post-transfer verification was interrupted by a Pi scratch-monitor deletion race.
No complete412result or H test execution;residual empty test directory preserved.
No fix/rerun/cleanup/rollback automatically. SDK4/native1;6H-associated full invocations,
0main slots remaining. Isolated H acceptance below is unchanged,not main readiness.
Deployment NOT_AUTHORIZED. Next proposal:separately approved Pi-only verifier recovery.

**H-REF: ACCEPT / task_only. Deployment: NOT_AUTHORIZED.**
[Final independent Pi acceptance](docs/h-ref-pi-final-acceptance.md):
`data/reports/training-harness-h-ref-v1/pi-final-acceptance-v1/review.json`.

Under explicit user authorization, Pi independently verified H1–H9 and every current
Worker limitation/issue using existing evidence. Assigned Pi duties are **fulfilled**,
not out of scope. The separate append-only Pi decision accepts the pinned isolated
bounded CPU-reference implementation and H-side interfaces—not main integration,
production/GPU readiness or experiments. All322isolated files,H8 and science unchanged.

Existing approved verification: **412PASS,0failures/errors/skips**,**81.938s child**,
~830.1MiB peakRSS within240s/2GiB;all389inherited/34D3methods/3564expanded cases pass.
Complete logs,owner-safe fresh-cache cleanup,empty fresh scratch and preservation PASS.
No new test or implementation change was made for final acceptance.

Native task `3606687c-beba-4d3f-bd24-03ae5238da37` remains historical **ESCALATE**,with
its failed original verification and unchanged reports/budgets. No native ACCEPT,
receipt rewrite,reopening,reassessment or extension change. Hierarchy:
**User authorization → Pi final acceptance authority → native agy execution evidence**.
Counts preserved:SDK4/native1(secondunused),**4actual H full runs**. Main integration,
O-OFFLINE,GPU-READY,profiling,comparative/final-test execution and deployment remain
separate gates. No new Worker,SDK,regression,experiment or main transfer authorized.

**Current implementation policy.** [Direct single-agent ownership is the default](docs/worker-delegation-policy.md).
Pi may implement and fix authorized code directly. Mandatory Pi–agy delegation and
the prospective **2 + 2 + 2** recovery process are retired. Delegation and second-model
review are optional. Original review requirements, execution limits, historical
reports, and consumed budgets remain unchanged. No new Worker or SDK calls follow.

**O-OFFLINE and later experiment gates remain separate.** The [historical preparation
proposal](docs/training-harness-preparation.md) and [34-part matrix](docs/training-harness-requirements.md)
remain evidence/scope references, not the latest H implementation state or launch authority.
O-OFFLINE is now main-integrated (see the top entry). Reuse `multiautoresearch` orchestration,not Movie/LM methodology.
GPU-READY,production initialization/environment/provenance/durability,main integration
and Musical_Instruments profiling still require their applicable approvals and evidence.
Accepted models/evaluator/D1–D3 and the frozen scientific protocol remain unchanged.

**Historical first two-SDK-request H result (not current):** both responses returned
zero files because of missing handoff context,including Pi's omitted loss-return key.
That two-request ESCALATE remains preserved at
`data/reports/training-harness-h-ref-v1/review.json`; later continuations did not erase
its decision or reset its consumed budget. None of those SDK requests was a native agy
invocation.

**Current checkpoint milestone: D3 main integration ACCEPT/task_only.**
[Independent main integration/retirement record](docs/checkpoint-storage-main-integration.md):
355baselineOK56.293s process;389integratedOK57.452s process,0failures/errors/skips
under240s each. All355baseline IDs/bytes,3564expanded cases/34D3methods preserved.
Three accepted D3files copied exactly;old main doc archived. Both isolated D3 trees
are retired from active work,retained unchanged for audit. No newWorker/rewrite/retry.
Review: `data/reports/checkpoint-storage-main-integration/review.json`.
**acceptance_scope: task_only; deployment_decision: NOT_AUTHORIZED.**
No training/profiling/heldout/remote work or stage/commit/push/release authorized.
The following milestone and failed-attempt entries retain their chronological history.

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

**D2: ACCEPT (task_only), now integrated in main.** Separate user-authorized
[main integration](docs/checkpoint-payload-main-integration.md) copied the exact
accepted source/tests/documentation. Main baseline249 passed; full integration
**355 tests, zero failures/errors/skips**,54.139s unittest / **56.643s process**,
under the unchanged120-second limit. All249 baseline tests and3564 expanded cases
pass. Independent integration decision:
`data/reports/checkpoint-payload-main-integration/review.json`.
No D3, profiling/training, deployment, commit or push is authorized by integration;
**deployment_decision: NOT_AUTHORIZED**.

**D2 is concluded. D3 implementation-first is now authorized within IO1–IO8**,
with Worker-proposed technical solutions and Pi's source/test verification. Only
reviewed safe synthetic IO may execute under the pinned bounded plan; no safety
claim without evidence and no unsafe testing. Preparation-only history is retained.
Historically D3 stopped before Worker after two120s baseline timeouts; that original
ESCALATE remains in `data/reports/checkpoint-storage-implementation/review.json`.
It is not the current implementation review and did not establish a D2 defect.
A later single user-authorized diagnostic found dense-matrix execution dominated
(73.97s); the optional-name test began at110.26s, not a demonstrated hang.
See `data/reports/checkpoint-storage-implementation/full-suite-diagnostic/findings.md`.
This data collection does not replace a passing original benchmark or authorize retuning.
The user has concluded that investigation. A [D3-only240s full-suite ceiling is
approved](docs/checkpoint-storage-execution-budget-amendment.md) by the user's
subsequent “Approve. please procced”, covering the fresh uninstrumented355 baseline
and later added D3 tests.
The approved fresh baseline passed **355 tests,55.645s process**. Initial D3 full
regression completed **379 tests,1failure/2errors,55.993s process**; all355 inherited
tests and3564 expanded cases pass. New source fixtures failed on permissions.
After the one consolidated revision, D3 is **ESCALATE/not accepted**: the assigned
source-parent-race guarantee remains unresolved and the returned source baseline hash
is malformed. The revision was inspected but **not applied or executed**. Both response
slots are consumed; no automatic third request or Pi production/hash repair. Decision:
`data/reports/checkpoint-storage-implementation/final-review-evidence/review.json`.
Scope **task_only**, deployment **NOT_AUTHORIZED**; no main code/test transfer.
Existing355 tests/assertions, D2 acceptance and original120s outcomes remain unchanged.
The user has subsequently approved [one bounded correction amendment](docs/checkpoint-storage-bounded-correction-amendment.md):
one additional response, not a reset; retain initial candidate/history and use Pi-verified
baseline hashes. Resolve original IO2 or report the precise blocker; preserve verified
logic, review allIO1–IO8 and execute only the one authorized safe full suite. No iterative
follow-up, race-research/controller project or security waiver.
That request ended with SDK `stop_reason:error`, HTTP200 and0response bytes; no patch
or feasibility report returned, no new code/test execution. Both candidates and prior
history are unchanged. Current **ESCALATE/task_only/not accepted**, deployment
**NOT_AUTHORIZED**: `data/reports/checkpoint-storage-implementation/bounded-correction-amendment-v1/review.json`.
IO2 remains unresolved; a failed completion is not evidence of infeasibility. No retry.
A subsequent [focused technical assessment](docs/checkpoint-storage-io2-technical-decision.md)
recommends direct-child physical acquisition from approved roots per existing D1 prefix,
not flattening D1 logical paths. Minimal SDK request/error-handling corrections and a
future bounded patch are proposed only; no further call, code change or test authorized.
IO3/IO5 remain unverified and IO6/IO7 unpassed; IO2 design alone cannot accept D3.
The user subsequently approved [one prefix-root bounded correction](docs/checkpoint-storage-prefix-root-correction.md):
independent directory FDs per D1 prefix,direct-child physical opens,unchanged logical IDs,
no continuous project-root ancestry claims. Existing candidate/fresh hashes;one fourth
request,no retry/revision,comprehensive IO1–IO8 review before at most one full240s suite.
SDK changes only error metadata and request/context bounds;no model/controller/routing change.
That phase completed **389tests,1failure/0errors/0skips,62.611s process**;all355 and3564
expanded cases pass. The remaining failure is a new global test-tripwire that blocks
a valid configs file before its scripts FIFO target,not observed product FIFO opening.
Root/decoder/publication/source pipeline controls otherwise pass. **ESCALATE/task_only/not
accepted**,deployment NOT_AUTHORIZED;no automatic fix/rerun/fifthWorker or budgetreset.
Review: `data/reports/checkpoint-storage-implementation/prefix-root-correction-v1/review.json`.
The user subsequently directed [one Pi-only test-oracle correction/full regression](docs/checkpoint-storage-oracle-only-correction.md),
not a production rewrite or new Worker response. Same-source-size singleton control
and FIFO target;one seventh full240s suite only,prior counts and failed389 preserved.
No further unrequested cycle;wholeIO1–IO8 and ordered limitation review still required.
**Previous isolated D3 result: ACCEPT/task_only in the isolated candidate**,389OK,0failures/errors/
skips,53.292s unittest/55.890s process under240s. All355 and3564expanded cases preserved.
Only the authorized Pi oracle method changed;production/API docs unchanged. FullIO1–IO8
and8exact ordered limitation reviews completed;oldfailed389/ESCALATEs stayhistory.
Decision: `data/reports/checkpoint-storage-implementation/oracle-only-correction-v1/review.json`.
**deployment_decision: NOT_AUTHORIZED**. No main D3 source/test transfer or research work;
main integration/regression requires separateapproval. No newWorker or automaticcycles.
See [D3 secure checkpoint storage/retrieval](docs/checkpoint-storage.md) and
`configs/checkpoint-storage-scope.json`. D2 validates already-decoded state; D3 owns
secure file/source IO, decoding-time bounds and integration with D1/D2. No additional
D2 correction, re-audit or security-controller work is planned absent a new error.

The preceding isolated source acceptance remains unchanged. The separately
approved [A04/D01 round](docs/checkpoint-payload-a04-d01-correction.md) closes the
original whole-D2 C1–C10 contract. Final full discovery: **355 tests, no failures,
errors or skips**, 111.539s unittest / **116.571s process**, within the unchanged
120-second limit. All337 inherited tests remain; all3564 expanded cases pass.
The independent review resolves both submission-time evidence questions and assesses
all six limitations in order. Failed attempts and Pi-authored verification corrections
remain disclosed. See `data/reports/checkpoint-payload-a04-d01-correction/review.json`.
Its original source-only no-transfer boundary is historical; the separate main
integration above fulfills the later authorization. D3, profiling/training and deployment remain unauthorized;
**deployment_decision: NOT_AUTHORIZED**.

**Prior text-only D2 result:** the separately authorized [text-only correction](docs/checkpoint-payload-text-correction.md)
is **ESCALATE at1/1**, not accepted. Two no-tool SDK responses produced localized fixes
and expanded evidence without Worker commands. After fixture screening/private-generator
corrections, final independent full discovery: **337tests,2failures,0errors,0skips,
46.429s**. All320inherited methods pass;3140expanded assertions succeed. Both failures
expose pre-existing Adam acceptance of step4.5with updates4. Whole-path diagnostic and
37-row evidence gaps remain. Useful isolated code/tests are preserved; no automatic
third response, source fix or main transfer. This was response/revision-count use, not
a Worker timeout. D3/training/deployment remain unauthorized.

**Prior matrix-guided D2 result:** correction
`eb5065fc-234e-4d6f-ac63-b788d55b1502` is **ESCALATE after its initial30minute timeout**.
Independent test-first run: **320cases,6failures,14errors**; all283old cases and verified
implementation bytes remain unchanged. New cases reproduce all6implementation blocker
groups, but no source fixes were made. Raw records prove prohibited probes/subsets and
external helper scripts, so Pi stopped with **one revision unused—not budget exhaustion**.
The detailed matrix preserved correct code and identified real defects, but compliant
Worker completion was not achieved. See [current review](docs/checkpoint-payload-matrix-correction.md).
The previous [283-green unaccepted correction](docs/checkpoint-payload-authorized-correction.md)
and all historical tasks/budgets remain preserved. No rejected payload source transferred
to main; D3 and real profiling/training remain blocked. No automatic replacement is authorized.

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
