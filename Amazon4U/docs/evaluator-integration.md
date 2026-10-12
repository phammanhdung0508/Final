# Evaluator Integration (Milestone I1 / Step 1)

## Context and Decision

Under the user-approved implementation sequence recorded in `docs/implementation-roadmap.md`, the integration of the independently accepted recommendation evaluator (`scripts/recommendation_evaluation.py`) into the primary recommendation core (`scripts/recommendation_core.py`) is executed as Step 1:
1. Step 1: Integrate the independently accepted evaluator into `recommendation_core.py`, eliminating divergent candidate/ranking/evaluation logic while preserving public core APIs, KG metadata fallback, and candidate-hardening invariants.
2. Step 2: Independent verification of remaining Milestone I1 components (Popularity, KGOnlyScorer, NegativeSampler, BPR loss, endpoint regularization).
3. Step 3: Implementation and independent verification of learned models and training harness.
4. Step 4: Preparation of bounded Musical_Instruments profiling plan for separate execution authorization.

### Historical Provenance and Preserved Boundaries
- Main HEAD: `ed60be08f3d1f22a9d76eecbee282415f95a6a5c` (229 tracked files, 6,349,614 bytes, archive SHA256 `76508021f4d9383c723073b7126dcbd3027bf59ed76d297bfad8d60fb0bf2ece`).
- Historical task `303fdf53-8c5b-4013-86be-4fec983c809a` remains historical ESCALATE (acceptance_scope: task_only, deployment_decision: NOT_AUTHORIZED).
- Sparse preflight task `d0a8d369-f283-4a35-8ad7-b27f30370e4d` remains preserved as historical ESCALATE; monitor requirements resolved under standing setup approval by materializing tracked clone baseline sentinels (outside-project tracked files in the isolated clone are monitored sentinels).
- Older historical broad I1/evaluator/audit tasks `52a149ec-0df0-4577-9d45-29de06d4d259`, `a7b233bd-e2fc-4ceb-8119-ea62bbcd0508`, and `fb50c21d-766a-4771-b2a7-396200852f76` remain historical ESCALATE.
- Accepted standalone evaluator task `0f2f0b7d-6e2a-4726-9e6e-83a73e8da423` was task-only ACCEPT (deployment NOT_AUTHORIZED); accepted source SHA256 `b396bdc2a3f419ebb2ce66f65e97c59d2dab5a0f5775692a7536a9dffa3bf3f2` and tests SHA256 `5d522eddc5c16bfd0b487a6dda84d900b5a99aec092471ed95f570d6fff837b1`.
- Explicit user recovery authorization permits a fresh scoped task and recovery; it is not a retroactive waiver of historical ESCALATE decisions.
- Incomplete prior Worker invocation histories are preserved as such; passing current source benchmarks does not establish an exhaustive historical execution claim.
- During the isolated Worker tasks, sibling projects (Movie4U) and main checkout were read-only baseline sentinels; no Worker modifications or Git metadata edits were permitted. Subsequent Pi mechanical integration into main is recorded below; sibling sources remain untouched.
- Worker implements integration, focused recovery fixes, and regression fixtures only; Pi retains sole verification and acceptance authority. Deployment is **NOT_AUTHORIZED**.

## Architecture and Integration Design

### 1. Unified Evaluation Engine
`scripts/recommendation_evaluation.py` serves as the authoritative evaluation engine across the project:
- `TrainingContext`: Validated training-only context enforcing feedback conditions (`all_ratings` vs `rating_ge_4`), warm pool derivation, distinct positive vs all-rated history, and duplicate/invalid input failures.
- `ScoringFailureError`: Shared exception raised for non-finite scores, missing candidate scores, invalid routes, corrupt profiles, or spoofed fallbacks. (Note: Candidate shortages are counted in candidate_shortages audit without padding and do not raise scoring errors).
- `BaseEvaluator`: Enforces candidate isolation, validation vs test history filtering, exact tie-breaking (`score desc, parent_asin asc`), shortage accounting without padding, macro-averaged ranking metrics (NDCG and HitRate @ 5, 10, 20), CatalogCoverage@10 over the full condition-warm pool, and route/fallback subgroup auditing. By default, rejects all provider fallback claims; evaluator strictly owns empty-positive-history fallback.
- `DEFAULT_FALLBACK_REASONS`: Configured as `("empty_positive_history",)` on `BaseEvaluator`, ensuring standalone reports default to `empty_positive_history` only.

### 2. Core Evaluator Subclass and Guarded Metadata Fallback
In `scripts/recommendation_core.py`, the legacy whole-class `Evaluator` duplicate is replaced with a clean subclass of `BaseEvaluator`:
- `DEFAULT_FALLBACK_REASONS`: Overridden on core `Evaluator` to `("empty_positive_history", "unusable_metadata_profile")`. The inherited evaluation engine ensures both reasons are always emitted under `report['subgroups']['fallback']['by_reason']`, populating count 0 and all-None registered metrics when absent.
- `Evaluator(BaseEvaluator)` overrides `_validate_provider_result` to delegate ordinary routes (`personalized`, `non_personalized`) directly to `super()._validate_provider_result`, and implements a narrow, strictly guarded hook for legitimate KG metadata fallback (`unusable_metadata_profile`).
- Guarded conditions for metadata fallback:
  - Scorer must be a legitimate `KGOnlyScorer` instance.
  - Scorer `context` must be identical to `Evaluator.context`.
  - User must have non-empty condition-positive history (`empty_positive_history` remains evaluator-owned).
  - All user and candidate item profile dictionaries across all channels are validated for finiteness via the authoritative base validator `KGOnlyScorer._validate_profile_dict`; non-finite values even in unused channels are rejected. Overriding `_validate_profile_dict` on scorer instances or subclasses cannot bypass this interface validation gate.
  - Positively weighted channels must not be usable (i.e. if `alpha > 0` and category profile non-empty, or `beta > 0` and brand profile non-empty, fallback is rejected as invalid/spoofed).
  - Provider scores for all authoritative candidates must exactly equal condition-positive training popularity.
- All spoofed reasons, wrong contexts, corrupt profiles, and scorer failures are rejected with `ScoringFailureError` without fallback.

### 3. Import Canonicalization
Both `scripts/recommendation_evaluation.py` and `scripts/recommendation_core.py` implement canonical module aliasing:
- Supports both package imports (`from scripts.recommendation_core import ...`) and top-level directory imports (`from recommendation_core import ...`).
- Relative package import `if __package__:` cleanly avoids masking underlying dependency errors.
- Both import modes resolve to identical module instances in `sys.modules`, ensuring `TrainingContext`, `ScoringFailureError`, `Evaluator`, and `KGOnlyScorer` share exact object identity in the same interpreter.

## Verification

The full test suite executes under the exact pre-declared runner:
1. `tests/test_recommendation_core.py` (51 tests): Verifies Milestone I1 components including hand-calculated IDF, mixture normalization, BPR loss, endpoint regularization, and core evaluation invariants.
2. `tests/test_recommendation_evaluation.py` (21 tests): Accepted shared evaluator hardening suite verifying rank boundaries, catalog coverage, test-split validation history requirements, and deep mutation isolation.
3. `tests/test_evaluator_integration.py` (29 tests): 13 preserved prior unaccepted integration fixtures plus 16 focused recovery fixtures verifying:
   - Core fallback reason schema with absent reasons emitting count 0 and all-None metrics across empty targets, zero eligible users, and no-fallback populations.
   - Standalone evaluator default `empty_positive_history` reason retention and provider fallback rejection.
   - Adversarial validator override regression ensuring non-finite values in unused channels cannot bypass interface validation.
   - Authoritative outer profile container validation and missing user/candidate profile state rejection as `ScoringFailureError` (rather than `AttributeError`, `TypeError`, or `KeyError`).
   - Route validation super-delegation for ordinary routes (`personalized`, `non_personalized`).
   - Hand-calculated mixed routes, both fallback reasons, and coverage oracles across feedback conditions (`rating_ge_4` and `all_ratings`) and splits (`validation` and `test` with validation history).
   - Compact table-driven oracle asserting all 7 registered metrics across both feedback conditions and splits with literal expected values (no evaluator/scorer calls, and asserting count 0/all-None metrics for empty_positive_history under `all_ratings` without fabricating or clearing positive history).
   - Honest `KGOnlyScorer` subtype dispatch fixture exercising all THREE routes (`personalized`, `non_personalized`, `fallback`) and both fallback reasons in one call, verifying provider bypass and exact aggregate/subgroup/by_reason metrics and catalog coverage.
4. Baseline suite other tests (12 tests): In `tests/test_candidates.py`, `tests/test_kg.py`, and `tests/test_protocol_lifecycle.py`.

Baseline partition accounting:
- Original clean baseline: 63 tests (51 core + 12 baseline).
- Prior integration-draft suite: 97 tests (63 clean baseline + 21 shared accepted + 13 preserved prior unaccepted integration).
- Current suite total: 113 discovered tests (97 integration-draft suite + 16 focused recovery fixtures; or 63 clean baseline + 21 shared + 29 integration = 113).
All 113 tests pass with 0 failures, 0 errors, and 0 skips.

## Original Worker submission status (historical)

- Step 1 implementation: **Completed by Worker, pending independent Pi verification, review, and recorded decision**.
- Deployment: **NOT_AUTHORIZED**.
- Comparative training / profiling: **NOT_AUTHORIZED**.

## Current independent review and main integration

- Preserved artifact task `2562b8ec-cf57-4ace-b348-9fbd6c994262`, revision 0:
  **ACCEPT**, acceptance_scope **task_only**, deployment **NOT_AUTHORIZED**.
  Decision `agy-evidence-f04ab70a-955f-4b89-8da0-2a484c1a4740`;
  independent benchmark 113 tests, zero failures/errors/skips, 1.699 seconds.
- Pi mechanically copied the five accepted source/test/doc files into main under
  the original implementation-sequence approval; no new source implementation or
  retuning. Target core was clean against HEAD, other targets absent; originals
  preserved before copying and copied hashes independently matched the artifact.
- Main baseline: 63 tests / 1.213 seconds. Integrated main suite: **113 tests /
  1.334 seconds**, zero failures/errors/skips. Fixture scratch empty; other
  pre-existing tracked Amazon4U bytes and unrelated Git status records unchanged;
  HEAD/index unchanged. No stage/commit/push, sibling or historical source edits.
- Source/provenance and complete main logs:
  `data/reports/evaluator-main-integration-v1-provenance/`.
  This document's independent-status amendment follows the byte-identical transfer;
  accepted snapshot remains unchanged. See `evaluator-verification-runtime-separated.md`.
- **Step 1 evaluator integration is complete and regression-verified in main.**
  Separate remaining-I1 scorer/sampler/loss verification, learned model/harness
  implementation and bounded Musical_Instruments profiling preparation remain next.
  No broad I1, learned-model or profiling-execution acceptance is implied.
- Historical ESCALATEs remain unchanged. Deployment, real-data model training and
  profiling execution remain **NOT_AUTHORIZED** until their respective gates.
