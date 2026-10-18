# Amazon4U Implementation Control-Plane: Milestone I1

> Historical initial I1 implementation record. Current accepted main components and
> D1–D3 decisions are in `../../docs/implementation-roadmap.md`. Current next-stage
> authority is preparation only: `../../docs/training-harness-preparation.md` and
> `../../docs/training-harness-requirements.md`. Reuse existing Kaggle orchestration,
> not LM/Movie training methodology. No inherited commands below authorize execution;
> no new Worker,legacy replacement,training/profiling or Kaggle calls are approved.

## 1. Roles and Governance

- **Pi**: Planner, Orchestrator, and independent Reviewer. Responsible for design decisions, task authorization, verification review, and held-out evaluation.
- **Worker**: Implementation Worker only. Responsible for implementing code and regression tests within explicitly allowed paths, executing declared verification commands, and reporting factual results without claiming review acceptance or authorization.
- **Boundary enforcement**:
  - Implementation strictly confined to declared `allowed_paths` within `/home/sunf/FSB/Final/Amazon4U`.
  - No reuse of Movie4U assumptions, LM benchmark runners, or unrelated external runners.
  - No subagent delegation, no Git metadata changes (no commit, push, reset, or branch manipulation).
  - No real dataset exposure, model training, optimizer updates, or profiling.
  - Deployment decision is explicitly **NOT_AUTHORIZED**.

## 2. Scope and Rationale: Milestone I1

Milestone I1 implements the reusable shared evaluation harness, deterministic Popularity and KG-only baseline scorers, and non-training reference sampling/loss/regularization utilities using only the Python standard library.

### Implemented Components (`scripts/recommendation_core.py`)
1. **`TrainingContext`**:
   - Enforces the two approved feedback conditions (`all_ratings` and `rating_ge_4`).
   - Rejects duplicate training user-item pairs and invalid identifiers/ratings (including boolean rejection).
   - Preserves all rated history while computing distinct condition-positive history, condition-specific warm items, and popularity.
   - Strict isolation: zero held-out parameters or targets enter context initialization.
2. **`PopularityScorer`**:
   - Deterministic condition-positive training popularity baseline.
   - Returns explicit non-personalized baseline route (`"non_personalized"`).
3. **`KGOnlyScorer`**:
   - Consumes audited KG-v1 category path and brand identity mappings (rejects raw dicts, non-collections, blank strings).
   - Computes smoothed IDF over condition-warm items: `1.0 + ln((N + 1) / (df + 1))`.
   - Normalizes category and brand channels independently via L2 norm.
   - Derives user profiles as normalized sum over distinct condition-positive training products.
   - Implements mixture weighting over ordered grid `(1.0, 0.0)`, `(0.75, 0.25)`, `(0.5, 0.5)`. Missing channel cosine is 0.0 with no per-item renormalization.
   - Evaluates numeric operations in deterministic sorted order with `math.fsum` for cross-platform stability.
   - Whole-user fallback to condition-positive popularity only when no positively weighted channel is usable.
   - Detects corrupt non-finite profile values as failures (`ScoringFailureError`), never falling back.
4. **`select_kg_only_mixture`**:
   - Evaluates grid on validation targets and selects best mixture using validation `NDCG@10`.
   - Retains first exact finite tie in grid order.
   - Explicitly raises error when validation population has no eligible users (all-N/A population).
5. **`Evaluator`**:
   - Evaluates validation and test splits with exact single held-out target per user.
   - Validates target identifiers, finite non-boolean ratings, and rejects duplicate target users.
   - Validates test split validation history (rejects missing validation history, requires known target user entries).
   - Audits unknown users, cold targets, irrelevant targets, and history overlaps with overlapping reason breakdowns and denominators.
   - Ranks full candidate set (memory-bounded by warm pool, no user-item matrix).
   - Filter-before-scoring: filters history before passing candidates to scorer.
   - Rejects non-finite scores, missing candidates, or scorer exceptions as `ScoringFailureError` (no fallback on failure).
   - Deterministic tie-breaking: score descending, then `parent_asin` ascending.
   - No list padding: computes exact NDCG@K, HitRate@K for K in (5, 10, 20), and `CatalogCoverage@10` (union divided by entire condition warm pool).
   - Bypasses personalized scorers for empty-positive-history users, assigning condition-positive popularity fallback.
   - Subgroup reporting across all observed routes and fallback reasons, including count, metrics, and catalog coverage.
6. **`NegativeSampler`**:
   - Evaluates negative pools on demand from sorted condition-warm items minus all user training-rated items (no eager users x items list scaling).
   - Preflight check fails globally if no trainable positive examples exist in the dataset.
   - Uniform one-negative draw using supplied RNG.
   - Validates that positive pairs are present in user's condition-positive history.
   - Empty negative pool yields explicit skipped accounting; an empty/skipped batch returns `([], skipped_count)` without inventing positives or regularization updates.
7. **Reference Mathematics (`bpr_loss`, `endpoint_regularization`, `stable_softplus`)**:
   - `stable_softplus`: numerically stable softplus with strict finite input/output validation.
   - `bpr_loss`: stable mean softplus difference over supervised positive/negative score pairs.
   - `endpoint_regularization`: explicit `1e-5 * sum(squared L2 base row norms) / (unique typed endpoint rows)`.
   - Endpoint deduplication: unique batch users + union of positive/negative items.
   - Base-ID rows only; ignores Brand, CategoryPath, and neighbor-only rows; validates consistent nonzero dimensions.

## 3. Milestone Queue

| Phase / Milestone | Description | Status | Gate Requirement |
|---|---|---|---|
| **Milestone I1** | Core shared evaluator, Popularity, KG-only, reference sampling & loss | **implemented_pending_Pi_review** | Complete regression suite passing |
| **Milestone I2** | Learned baseline architectures (LightGCN, Heterogeneous KG+GNN) reference modules | Pending | Milestone I1 accepted by Pi |
| **Milestone I3** | Batch data loaders, batch positive-edge masking, and training loop harness | Pending | Milestone I2 verified |
| **Milestone I4** | Bounded profiling gate on Musical_Instruments | **NOT AUTHORIZED** | Explicit user approval & budget freeze |
| **Milestone I5** | Comparative training runs across categories and conditions | **NOT AUTHORIZED** | Profiling review & execution plan freeze |

## 4. Historical initial I1 verification (not Pi acceptance)

Execution command:
```sh
.venv/bin/python -B -c "import os,tempfile,unittest,json,sys; from pathlib import Path; p=Path('.test-output/recommendation-core-v1').resolve(); p.mkdir(parents=True,exist_ok=True); os.environ['TMPDIR']=str(p); tempfile.tempdir=None; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); r=unittest.TextTestRunner(verbosity=2).run(suite); (p/'fixture-result.json').write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'successful':r.wasSuccessful()},indent=2)+'\n'); sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)"
```

Outcome:
- **Tests run**: 63
- **Failures**: 0
- **Errors**: 0
- **Skipped**: 0
- **Successful**: True
- **Artifact**: `.test-output/recommendation-core-v1/fixture-result.json` (diagnostic text artifact generated by original approved wrapper)
- **Invocation timeout history**: Rev0 and Rev1 reached the 10-minute timeout during initial implementation exploration and initial revision; Rev2 targeted fixes executed cleanly in ~1.8s.
- **Known limitations**: Reference fixture core, not a full-data production adapter or training pipeline; no cross-platform bitwise guarantee from `math.fsum` alone.

## 5. Next Separate Bounded-Profiling Gate Requirements

Bounded profiling must NOT proceed without separate, explicit user authorization. Entry requirements before profiling on Musical_Instruments:
1. Prerequisite regression verification of all model encoders and batch positive-edge masking.
2. Explicitly approved profiling scope, hardware ceilings (conditional Kaggle capacity), time limits, and optimizer permission.
3. Provisional batch size, fanout, and initialization parameters (distinct from comparative run settings).
4. Validation-only ranking measurements (zero test split exposure).
5. Fresh, isolated profiling output paths.

## 6. Current Lifecycle State

- **Protocol Version**: `amazon4u-protocol-v2.0.0` (user-approved suite amendment;
  v1.0.0 snapshot and initial verification history preserved). Pi directly performed
  this config/documentation cleanup under the user's scoped exception; no agy task
  was launched. See [amendment](../../docs/bpr-mf-removal-amendment.md).
- **Lifecycle State**: `IMPLEMENTATION_REGRESSION_VERIFICATION`
- **Execution Revision**: `null` (not assigned)
- **Profiling Authorized**: `false`
- **Training Authorized**: `false`
- **Deployment Decision**: `NOT_AUTHORIZED`
- **Worker Milestone Status**: `implemented_pending_Pi_review`
