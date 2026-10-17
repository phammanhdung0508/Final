# D2 detailed part-level blocker matrix

**Preparation artifact, not implementation acceptance or execution authority.** Current task `6efb0ae5-6a62-4376-91e9-7b358c51013f` remains ESCALATE at1/1. No Worker or candidate tests were run for this preparation. All proposed reproductions below are **unexecuted** unless explicitly marked historical.

## Pinned inputs and interpretation

Candidate root: `/home/sunf/FSB/Final/Amazon4U/.worktrees/checkpoint-payload-authorized-correction/Amazon4U`. Historical Pi benchmark283/283 (including249 unchanged baseline IDs),7.302s/9592ms, evidence `agy-evidence-bb5368ef-3750-4924-b99f-be9856c0bbbd`. This is not a fresh run or D2 ACCEPT. Exact file/function/test hashes, original criterion strings and command are in `data/reports/d2-part-level-preparation/baseline.json`.

| File | SHA256 |
|---|---|
| `scripts/checkpoint_payload.py` | `5113947b8716212942eb7592f5e014d964c1faf75c9d0256d5e2d3a905a28a68` |
| `tests/test_checkpoint_payload.py` | `c4b9fe4c224c4cecdcd1d19d2f62aa484d207ac11d9ca770c9cc093a3b97a2d5` |
| `docs/checkpoint-payload.md` | `66202875e2d98d7a68b8bad8ad3c0c9e26900980bb5cf9b16b07d952eef52554` |

The original859-line/59605-byte specification is pinned as SHA256 `a648d97f73e1213eb37f1a87fe56e76f1171f59baedcce6cd62b2c66f7dd64c4`. C1–C10 are unchanged, with detailed contracts at spec402–786. Candidate docs append is an untrusted prior claim, not acceptance evidence.

**Status is part-specific.** Verified means the cited invariant has source/historical evidence, with the basis stated; it does not mean all adversarial cases are covered. Unverified identifies missing evidence, not an established implementation defect. Failed static findings are not newly executed failures. Implementation preservation and additional tests are separate columns.

## Part-level navigation

| ID | Part | Criteria | Implementation status | Disposition |
|---|---|---|---|
| P01 | Accepted dependencies, baseline and import aliases | C1, C9 | verified | `preserve_dependency` |
| F01 | Valid fixture construction and guard reachability | C2, C6, C9 | failed | `add_tests_and_boundary_helper_only` |
| E01 | Public API, exact14-root envelope and applicability records | C2, C8 | verified | `preserve_logic_add_tests` |
| E02 | Actual/expected identity and source provenance binding | C1, C2, C9 | verified | `preserve_logic_add_tests` |
| E03 | Trusted descriptor list schemas | C2, C4, C5, C8 | failed | `targeted_declared_schema_fix` |
| E04 | Runtime/provenance/default-step-policy coherence | C2, C4, C5, C8 | verified | `preserve_logic_add_tests` |
| R01 | Trusted resource policy schema/configurability | C3, C8 | verified | `preserve_logic_add_tests` |
| R02 | Exact M/S/C metadata/storage/structure accounting | C3, C8, C9 | verified | `preserve_logic_add_tests` |
| R03 | Preflight/raw-validation/copy/revalidation ordering | C3, C7, C8 | verified | `preserve_logic_add_tests` |
| M01 | LightGCN complete current/best dense recipe | C3, C9 | verified | `preserve_logic_add_tests` |
| M02 | Hetero complete36-parameter current/best recipe | C3, C9 | verified | `preserve_logic_add_tests` |
| M03 | Sparse COO canonicality and trusted graph contents | C3, C9 | verified | `preserve_logic_add_tests` |
| A01 | Optional Adam param_names type/value binding | C2, C4, C8 | failed | `targeted_named_option_fix` |
| A02 | Adam packed IDs, groups/options and complete parameter binding | C2, C4, C9 | verified | `preserve_logic_add_tests` |
| A03 | Adam lazy states and aggregate update lower bound | C4, C9 | verified | `preserve_guard_add_tests` |
| A04 | Adam moment and native scalar step/precision | C4, C9 | verified | `preserve_logic_add_tests` |
| G01 | Canonical CUDA role and explicit CPU inactivity | C2, C5, C8 | failed | `targeted_canonical_CPU_guard_fix` |
| G02 | RNG registry, shared IDs, families and W0/W1/W2 | C5, C9 | verified | `preserve_logic_add_tests` |
| G03 | Python MT19937 v3 codec | C5, C9 | verified | `preserve_logic_add_tests` |
| G04 | NumPy RandomState MT19937 codec | C5, C9 | verified | `preserve_logic_add_tests` |
| G05 | NumPy PCG64 codec | C5, C9 | verified | `preserve_logic_add_tests` |
| G06 | Native Torch CPU codec and seed binding | C5, C9 | verified | `preserve_logic_add_tests` |
| G07 | Seed materials actual/expected/both and initialization lifetime | C2, C5, C9 | verified | `preserve_logic_add_tests` |
| G08 | Complete live/global RNG preservation | C5, C7, C9 | unverified | `preserve_implementation_add_tests` |
| S01 | Positive/empty-negative domain P/K/T oracles | C5, C9 | verified | `preserve_logic_add_tests` |
| S02 | Sampler orders, quiescence and batch/progress counters | C5, C6, C9 | verified | `preserve_logic_add_tests` |
| T01 | Selection first-check, improvement, ties and first patience stop | C6, C9 | verified | `preserve_function_add_tests` |
| T02 | Max epoch alone and simultaneous patience/max schedule | C6, C9 | unverified | `preserve_function_add_tests` |
| T03 | Current/best binding and exact-bit identity | C3, C6, C7, C9 | verified | `preserve_function_add_tests` |
| O01 | Owned output and separate input/expectation/container/storage aliases | C7, C9 | unverified | `preserve_implementation_add_tests` |
| O02 | Complete no-mutation on early/late failure; owned semantics | C7, C8, C9 | unverified | `preserve_implementation_add_tests` |
| D01 | Whole-path bounded domain diagnostics | C2, C8, C9 | failed | `targeted_bounded_rendering_only` |
| D02 | Reject custom objects/cycles without hooks | C8, C9 | verified | `preserve_type_dispatch_add_tests` |
| V01 | Full regression evidence and per-case closure coverage | C1, C9, C10 | unverified | `tests_and_evidence_only` |
| V02 | Disclosure, docs, limitations and action reconciliation | C10 | failed | `docs_and_report_correction_only` |
| A05 | Adam foreach/fused type checks before equality | C2, C4, C8 | failed | `targeted_type_before_comparison_fix` |
| G09 | Parent/local seed type validation before bound-role equality | C2, C5, C8 | failed | `targeted_seed_type_before_comparison_fix` |

## Exact blocking sites

Ranges below refer to the pinned candidate; complete function ranges remain in each row.

| Row | Primary defect/evidence location |
|---|---|
| F01 | `tests/test_checkpoint_payload.py:634–642,665–675`: E4 best snapshot versus earliest selected epoch; fixture defect, not selection-code defect. |
| E03 | `scripts/checkpoint_payload.py:2000–2001,2070–2071`: trusted list fields accept tuples. |
| A01 | `scripts/checkpoint_payload.py:897–902`: optional actual names coerced with list(). |
| A05 | `scripts/checkpoint_payload.py:938–946`: foreach/fused equality before type checks. |
| G01 | `scripts/checkpoint_payload.py:1133–1144`; accepted D1 `checkpoint_identity.py:224–235`: canonical name mismatch in CPU guard. |
| G09 | `scripts/checkpoint_payload.py:1233–1240,1247–1310`: parent seed comparison precedes type validation. |
| D01 | `scripts/checkpoint_payload.py:729,893,901,922,1137,1527,1982–1984,2065–2067,2089,2256`: raw collection/value diagnostics; use the recipes below to reach them. |
| V02 | Candidate `docs/checkpoint-payload.md:872–1004`; original `independent-review/revision1/worker-report.json` fields summary/test_results/unresolved_issues versus raw steps272/282/300/310 and initial review. **No runtime unit test proves disclosure**: compare the report with actual records. |

## Detailed cases and mandatory closure evidence

Common rule: build a valid complete public control first, independently detach expectations from mutable payload aliases, then apply exactly the described logical mutation. Assert the intended field/guard and domain family, not any unrelated refusal. Resource preflight may legitimately refuse device/layout/type before later semantics. Multiple-role corruptions are explicit only where listed. D1 security exceptions retain their existing family. Do not derive trusted expectations from the corrupted payload. Each case family expands to independently labeled cases, not one broad assertion.

### P01 — Accepted dependencies, baseline and import aliases

**Criteria:** C1, C9. **Implementation:** verified; `preserve_dependency`. **Tests:** historical_baseline_verified_recheck_unchanged.

**Source:** `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`).

**Existing tests:** `test_dual_import_canonicalization` at 953–957.

**Current evidence:** Historical283-suite retains249 IDs; current hashes match archived candidate.

**Valid control:** Existing249 baseline + dual-import case.

**Required cases / gaps:** Freeze D1/model/all249 test bytes, original criteria/protocols, source HEAD/index; future bootstrap must reproduce283 before edits. Preserve all34 existing payload method ASTs; additive tests are allowed.

**Expected outcome / mandatory evidence for ACCEPT:** Fresh original full discovery after setup; no skips/deleted assertions/unexpected scope or Git mutation. No D1/model edits. Subsystem acceptance remains pending.

### F01 — Valid fixture construction and guard reachability

**Criteria:** C2, C6, C9. **Implementation:** failed; `add_tests_and_boundary_helper_only`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:2120–2333` (`_validate_semantic_payload`).

**Existing tests:** `_make_synthetic_identity_and_provenance` at 99–393; `_make_valid_lightgcn_payload` at 410–697; `_make_valid_hetero_payload` at 700–943; `test_positive_lightgcn_payload` at 959–977; `test_positive_heterognn_payload` at 979–995; `test_fixture_alignment_and_normalization` at 997–1023.

**Current evidence:** E0/E2 controls pass historically. Static helper defect: E4 constant scores select2 but best_model.epoch is4; existing patience test overwrites it.

**Valid control:** Retain original E0/E2 helper behavior; add a separate coherent-boundary builder using accepted D1 build_canonical_identity and build_source_provenance(source_files,git_evidence,environment_records).

**Required cases / gaps:** Before each mutation validate a detached full control. For new E/F/history contexts rebuild fingerprints through D1; set sampler hash, epochs, G/history, Adam updates/steps, selected epoch/best weights coherently. Keep expectations detached from payload; no regenerated whole test file.

**Expected outcome / mandatory evidence for ACCEPT:** Control succeeds before mutation. A malformed fixture is corrected, never used to justify weakening guards. E4 tie must have best epoch2, not4.

### E01 — Public API, exact14-root envelope and applicability records

**Criteria:** C2, C8. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`); `scripts/checkpoint_payload.py:2120–2333` (`_validate_semantic_payload`).

**Existing tests:** `test_envelope_and_inapplicable_components` at 1260–1298; `test_strict_schemas_and_types` at 2166–2199.

**Current evidence:** Source plus historical missing/extra/schema/capability/active-AMP/scheduler negatives; full nested coverage absent.

**Valid control:** Complete LightGCN and Hetero controls, scheduler/AMP explicit inactive records.

**Required cases / gaps:** Each root missing/extra/wrong supported type; absent mandatory keyword (Python call TypeError is signature evidence, not malformed-payload evidence); root schema/capability invalid; each scheduler/AMP key missing/extra/wrong type, active state, non-None inactive state, incorrect reason.

**Expected outcome / mandatory evidence for ACCEPT:** D2 data refusals are field-specific CheckpointValidationError; required-call signature preserved; no manufactured expectations/default policy. Root-key diagnostic changes belong only to D01.

### E02 — Actual/expected identity and source provenance binding

**Criteria:** C1, C2, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:2120–2333` (`_validate_semantic_payload`).

**Existing tests:** `test_positive_lightgcn_payload` at 959–977; `test_positive_heterognn_payload` at 979–995.

**Current evidence:** D1 reconstruction/comparison calls present, D1 tests unchanged and green; D2 caller-specific mutation evidence incomplete.

**Valid control:** Independently constructed source/provenance/identity records, not expectations copied from a corrupted payload.

**Required cases / gaps:** Actual-only and expected-only missing/extra/wrong types; falsified fingerprints; rebuild a different valid identity (seed/settings/row membership/graph/history) and compare with original trusted identity; source bytes/environment/Git mismatch; keep D1 API/exception semantics.

**Expected outcome / mandatory evidence for ACCEPT:** Reject via existing D1 validation/mismatch/provenance/security exception families as appropriate. No D1 rewrite, source acquisition, physical-origin proof or safe-decoding claim.

### E03 — Trusted descriptor list schemas

**Criteria:** C2, C4, C5, C8. **Implementation:** failed; `targeted_declared_schema_fix`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1886–1959` (`_validate_expected_model_schema`); `scripts/checkpoint_payload.py:1962–2113` (`_validate_expected_runtime_schema`).

**Existing tests:** `test_strict_schemas_and_types` at 2166–2199.

**Current evidence:** Static: expected_runtime.optimizer_groups and rng_role_bindings values accept tuple despite original spec ordered-list declarations (spec511–523,620–630).

**Valid control:** List-based existing complete descriptors. Preserve supported row-order/parameter-name sequence alternatives where original contract permits them.

**Required cases / gaps:** All expected_model/runtime nested keysets missing/extra/wrong types; optimizer_groups tuple/empty/dict; each rng_role_bindings value tuple/string/scalar; duplicate/missing name/ID; row-order duplicates/wrong membership; Hetero empty User/Product versus valid empty metadata.

**Expected outcome / mandatory evidence for ACCEPT:** Exact originally declared list fields refuse tuple; no indiscriminate tuple ban. Role-specific CVE, valid controls remain green. Bound diagnostics via D01.

### E04 — Runtime/provenance/default-step-policy coherence

**Criteria:** C2, C4, C5, C8. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1962–2113` (`_validate_expected_runtime_schema`); `scripts/checkpoint_payload.py:2120–2333` (`_validate_semantic_payload`).

**Existing tests:** `test_strict_schemas_and_types` at 2166–2199; `test_positive_lightgcn_payload` at 959–977.

**Current evidence:** Static comparisons and existing profile negative; component-wide cases missing.

**Valid control:** Loaded CPU float32 native step policy; both declared batch policies in synthetic contexts.

**Required cases / gaps:** Each six runtime_profile fields actual-only/expected-only/both malformed; provenance version disagreement; unsupported batch policy/dtype; float64 public profile must refuse when loaded default is float32 without changing global dtype.

**Expected outcome / mandatory evidence for ACCEPT:** Field-specific domain refusal; no host acquisition, default-dtype mutation, implicit defaults or claim float64 profile verified. D01 changes diagnostic rendering only.

### R01 — Trusted resource policy schema/configurability

**Criteria:** C3, C8. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:285–338` (`validate_resource_policy`).

**Existing tests:** `test_resource_policy_validation` at 1072–1115.

**Current evidence:** Current schema/explicit smaller runtime profile controls pass; all-field matrix incomplete.

**Valid control:** Exact synthetic2097152/2097152/32/16384 and explicit smaller runtime profile.

**Required cases / gaps:** Each policy key missing/extra/wrong type; bool/0/negative for each four limits; invalid schema/kind/profile; synthetic constants changed; payload-carried policy cannot override trusted argument.

**Expected outcome / mandatory evidence for ACCEPT:** Domain refusal before unsafe processing; no universal production2MiB ceiling, hardware auto-sizing or larger profile selected.

### R02 — Exact M/S/C metadata/storage/structure accounting

**Criteria:** C3, C8, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:341–367` (`_tensor_descriptor`); `scripts/checkpoint_payload.py:370–559` (`preflight_resource_accounting`).

**Existing tests:** `test_exact_resource_boundaries_and_accounting` at 1117–1172; `test_exact_independent_projected_output_oracle` at 1747–1765; `test_exact_visit_limits_under_synthetic_profile` at 1767–1779; `test_storage_deduplication_same_address_different_capacity` at 2151–2164.

**Current evidence:** Historical exactB/B+1, M174/S1048576/C2096978, visits16384/16385, depth32/33, distinct StorageImpl same-address cases; static raw backing/copy accounting read.

**Valid control:** Keep existing literal independent oracles, not recomputed solely by candidate helper.

**Required cases / gaps:** Add sparse index+value backing/descriptor count oracle; aliases counted per metadata/tensor occurrence, input dedup only actual storage identity; tiny views/empty storages/noncontiguous dense views; separately tensor-bearing expected descriptors. Test integer byte charge and UTF8/bytes/container charge from explicit formula.

**Expected outcome / mandatory evidence for ACCEPT:** Equality accepted; next unit rejected; no densify/coalesce/clone before quota. One oversized negative at a time, accounted input or projected output <=2097152+4096. No process-memory sandbox claim.

### R03 — Preflight/raw-validation/copy/revalidation ordering

**Criteria:** C3, C7, C8. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`); `scripts/checkpoint_payload.py:370–559` (`preflight_resource_accounting`).

**Existing tests:** `test_raw_tensor_detachment_and_sparse_forgery_rejection` at 1671–1745.

**Current evidence:** Static7-phase ordering and raw grad/forged COO refusals historically pass; no complete tripwire evidence.

**Valid control:** Unpatched complete public controls for both models.

**Required cases / gaps:** Within new tests use restored unittest.mock context tripwires for copy/D1 reconstruction/native parser. Oversized actual and each expected input must domain-refuse before tripwire; raw invalid grad/coalescing before copy. Instrument owned copy to produce a bad copied RNG field: owned semantic revalidation must refuse without partial return.

**Expected outcome / mandatory evidence for ACCEPT:** No AssertionError/native error accepted as quota evidence. Intended preflight/raw/owned guard identified; all patched helpers restored after test. Do not alter entry-point phase order.

### M01 — LightGCN complete current/best dense recipe

**Criteria:** C3, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:589–612` (`_validate_tensor_properties`); `scripts/checkpoint_payload.py:710–770` (`_validate_model_state_dict`).

**Existing tests:** `test_lightgcn_model_parameter_and_buffer_mutations` at 1300–1354; `test_positive_lightgcn_payload` at 959–977.

**Current evidence:** 2parameters+buffer recipe matches accepted model; selected current-shape/NaN/meta checks pass, full matrix missing.

**Valid control:** Independent valid current/best LightGCN with correct row IDs and normalized sparse buffer.

**Required cases / gaps:** For EACH user_embeddings/item_embeddings in EACH current_model/best_model independently missing/extra/shape/float64/sparse-layout/meta-device/NaN/+Inf/-Inf/requires_grad; model epoch/row-order/schema corruptions separate. Use intended role/name/guard assertions.

**Expected outcome / mandatory evidence for ACCEPT:** Full named coverage manifest; property/recipe guard not unrelated best-weight equality. Keep numerical predicates and exact2-param recipe unchanged; key rendering D01 only.

### M02 — Hetero complete36-parameter current/best recipe

**Criteria:** C3, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:566–586` (`_get_hetero_expected_param_names`); `scripts/checkpoint_payload.py:589–612` (`_validate_tensor_properties`); `scripts/checkpoint_payload.py:710–770` (`_validate_model_state_dict`).

**Existing tests:** `test_heterognn_model_parameter_mutations` at 1356–1385; `test_valid_empty_hetero_metadata_tables` at 2063–2092; `test_positive_heterognn_payload` at 979–995.

**Current evidence:** Recipe matches accepted registration; current tests have only5shape categories, not complete negative matrix.

**Valid control:** All36 slots; User/Product nonempty; both metadata tables empty is another valid control.

**Required cases / gaps:** Expand all36 names from pinned model recipe, each current/best role: missing/extra/shape/dtype/layout/meta/finiteness/detachment; include all4 base tables, both layers of8relations and4node self weights/biases. Empty tables test applicable shape/type/layout cases; use nonempty metadata control for element-finiteness mutations.

**Expected outcome / mandatory evidence for ACCEPT:** No invented element access on empty tensors; parameter/name-specific CVE; no change to model,36 count, registration order, or valid empty metadata.

### M03 — Sparse COO canonicality and trusted graph contents

**Criteria:** C3, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:615–707` (`_validate_sparse_adj_buffer`); `scripts/checkpoint_payload.py:710–770` (`_validate_model_state_dict`).

**Existing tests:** `test_raw_tensor_detachment_and_sparse_forgery_rejection` at 1671–1745; `test_r6_btf_fixture_adjacency_mismatch_oracle` at 2624–2684.

**Current evidence:** Forged order/duplicates and corrected5vs6edge oracle pass; complete actual/expected/current/best cases absent.

**Valid control:** Four-edge normalized2x3 graph plus existing independent five-edge BTF graph.

**Required cases / gaps:** Mutate current adjacency, best adjacency and expected buffer separately: missing/dense/wrong dtype/meta/shape/NaN/Inf/grad, coalesced flag false, lex order, duplicate, row/column lower+upper bounds. For order/duplicate/range keep invalid indices with _coalesced_(True), never coalesce them away. Same-NNZ different canonical graph: replace (u2,p2) with (u2,p1), independently recompute its degree weights, retain original trusted buffer.

**Expected outcome / mandatory evidence for ACCEPT:** Intended canonical/range/trusted-content refusal. Valid current+expected agreeing on a different physically supplied graph is NOT an origin-proof oracle; do not invent that claim. Preserve numerical exact-bit comparisons.

### A01 — Optional Adam param_names type/value binding

**Criteria:** C2, C4, C8. **Implementation:** failed; `targeted_named_option_fix`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:777–1087` (`_validate_adam_optimizer`); `scripts/checkpoint_payload.py:1962–2113` (`_validate_expected_runtime_schema`).

**Existing tests:** `test_adam_optimizer_validation` at 1387–1504.

**Current evidence:** Static source898–901 calls list(act_pnames) without type check or exact comparison with optional trusted value. Same-key dict passes, scalar raises TypeError. No current named-group test.

**Valid control:** Enable names in BOTH actual group and trusted options; same ordered clean name list; validate public control first. Include2-group partition and both model families.

**Required cases / gaps:** Actual-only dict.fromkeys(valid_names,None),7,None,string,bytes,bool; missing/unexpected optional field; reordered/duplicate/wrong element names; malformed trusted named option; both-corrupt mapping. Match the explicitly supported trusted sequence type/value rather than coerce arbitrary iterable. Proposed handoff pins list/list and tuple/tuple (existing trusted sequence support), rejects cross-container mismatch; no blanket tuple policy outside declared fields.

**Expected outcome / mandatory evidence for ACCEPT:** Named-option CVE, not native conversion or acceptance. Full controls stay valid. Protect all other Adam conditions; do not rewrite selection. Compatibility detail is explicit in proposed assignment, not asserted as a newly executed result.

### A02 — Adam packed IDs, groups/options and complete parameter binding

**Criteria:** C2, C4, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:777–1087` (`_validate_adam_optimizer`); `scripts/checkpoint_payload.py:1962–2113` (`_validate_expected_runtime_schema`).

**Existing tests:** `test_adam_optimizer_validation` at 1387–1504; `test_strict_schemas_and_types` at 2166–2199.

**Current evidence:** Static checks plus selected moment/orphan/type tests; group/name/options matrices absent.

**Valid control:** 1-group and2-group controls with independent trusted options and native packed IDs0..P-1; model buffer never an optimizer parameter.

**Required cases / gaps:** Each binding/state/group field missing/extra/wrong type, bool IDs/updates, duplicate/orphan/unknown IDs/names, actual group swaps/param order, omitted parameter, buffer as parameter; each lr/betas/eps/weight_decay and supported flags wrong type/value in actual/expected/both; betas list/length/wrong values, foreach None/False/True positives, fused None/False positives.

**Expected outcome / mandatory evidence for ACCEPT:** Guard-specific CVE; unchanged frozen hyperparameters and supported flags. Preserve unrelated condition ASTs; A01/A05 permit their explicitly identified fixes, and D01 bounded diagnostic expression changes.

### A03 — Adam lazy states and aggregate update lower bound

**Criteria:** C4, C9. **Implementation:** verified; `preserve_guard_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:777–1087` (`_validate_adam_optimizer`).

**Existing tests:** `test_adam_mixed_lazy_and_updated_parameters` at 1781–1803; `test_adam_optimizer_validation` at 1387–1504.

**Current evidence:** Mixed lazy and initialized-zero controls pass; final aggregate restored versus initial regression, source guard1078–1087 pinned separately.

**Valid control:** G0 omitted/{}/complete-zero states; G4 mix one updated4 and others zero/omitted for both models.

**Required cases / gaps:** G>0 all updates0/omitted; all individual updates<=G but sum=G-1 with matching scalar steps; individual updates>G; missing/{}/None state with updates>0; partial/nonzero initialized zero-state. Also accept sum>G and heterogeneous per-parameter updates.

**Expected outcome / mandatory evidence for ACCEPT:** Exact aggregate or lazy-state guard CVE as intended; valid mixed states pass. NO total-updates==G or all-updates==G rule. Aggregate block byte hash must remain unchanged.

### A04 — Adam moment and native scalar step/precision

**Criteria:** C4, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:777–1087` (`_validate_adam_optimizer`).

**Existing tests:** `test_adam_optimizer_validation` at 1387–1504.

**Current evidence:** Negative first moment/shape/sign and float32 representable2**24+2/2**25 versus inexact2**25+1 controls pass. Full states/fields incomplete.

**Valid control:** Synthetic initialized zero and post-update mappings, no optimizer execution. Component helper tests only inside full unittest run.

**Required cases / gaps:** Both moments shape/dtype/layout/meta/finiteness/grad, second moment negative; zero moments allowed with updates>0; step missing/Python int/bool/tensor vector/wrong dtype/meta/NaN/Inf/negative/fraction/update disagreement/grad; huge invalid updates domain error; retain representable high-step controls and precision-specific inexact refusal.

**Expected outcome / mandatory evidence for ACCEPT:** No2**24 validity cap, no global default change, no optimizer/backward/load. Preserve lazy and precision predicates; D01 diagnostic edits only.

### G01 — Canonical CUDA role and explicit CPU inactivity

**Criteria:** C2, C5, C8. **Implementation:** failed; `targeted_canonical_CPU_guard_fix`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1094–1368` (`_validate_rng_state_dict`).

**Existing tests:** `test_rng_comprehensive_matrix_and_worker_w2` at 2094–2149.

**Current evidence:** Static wrong literal PyTorch_CUDA at1143 cannot match canonical PyTorch_CUDA_if_applicable; existing test only adds IDs to an inactive role. D1 consistency has no CPU/CUDA cross-check.

**Valid control:** Original canonical inactive role, empty IDs, original explicit reason, passes.

**Required cases / gaps:** Noncanonical actual name PyTorch_CUDA rejects ordered-name guard. Separately rebuild valid D1 identity with canonical CUDA role active(seed42,clean algorithm/derivation,reasonNone), update actual+expected identity and sampler hash, matching actual role and trusted role bindings=[gen_torch] with existing same-seed parent CPU generator; all other fields valid. This consistent metadata must fail explicit CPU inactivity, not identity mismatch.

**Expected outcome / mandatory evidence for ACCEPT:** CVE identifies canonical CUDA role/CPU unsupported. No CUDA tensor/allocation/init or D1 edits. Test construction through public builder, not editing fingerprint text.

### G02 — RNG registry, shared IDs, families and W0/W1/W2

**Criteria:** C5, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1094–1368` (`_validate_rng_state_dict`); `scripts/checkpoint_payload.py:1962–2113` (`_validate_expected_runtime_schema`).

**Existing tests:** `test_staged_worker_context_w1` at 1805–1821; `test_rng_comprehensive_matrix_and_worker_w2` at 2094–2149.

**Current evidence:** W1/W2 controls and selected duplicate/orphan/missing-family cases pass; all-role coverage missing.

**Valid control:** W0,W1,W2 and same generator ID intentionally shared across parent roles, one registry record.

**Required cases / gaps:** Each10role missing/extra/order/type/applicability/algorithm/derivation/reason/IDs; missing/duplicate/orphan generator; wrong codec/family/cardinality/parent worker context; W0 any worker; W1/W2 bool/negative/out-of-range/duplicate IDs and each missing family, keeping other bindings coherent to reach intended guard.

**Expected outcome / mandatory evidence for ACCEPT:** Missing state never silently dropped; shared IDs not forced to independent streams. Use only staged private contexts, no worker processes. G01/G09 and D01 are the identified implementation changes here.

### G03 — Python MT19937 v3 codec

**Criteria:** C5, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1371–1469` (`_validate_single_codec_state`).

**Existing tests:** `test_rng_and_codecs_validation` at 1506–1553.

**Current evidence:** Valid codec and version2 refusal historically pass; structural/internal component cases missing.

**Valid control:** Private Random(42).getstate(), finite Gaussian cache and None cache.

**Required cases / gaps:** State/container length/version bool and wrong version; internal tuple length; word bool/negative/>uint32; index -1/625/bool; gauss wrong scalar type/nonfinite. Independently corrupt each actual state field.

**Expected outcome / mandatory evidence for ACCEPT:** Codec/field-specific CVE; only private setstate, no draws, reseed reconstruction or global setter.

### G04 — NumPy RandomState MT19937 codec

**Criteria:** C5, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1371–1469` (`_validate_single_codec_state`).

**Existing tests:** `test_rng_and_codecs_validation` at 1506–1553.

**Current evidence:** Valid codec and wrong engine-name refusal pass; word/cursor/cache/type coverage missing.

**Valid control:** 624 uint32 word list, positions0/624, flags0/1; finite cached value may be nonzero when flag0.

**Required cases / gaps:** Tuple arity/engine; ndarray vs list; word count/range/bool; position -1/625/bool; flag -1/2/bool; cache int/nonfinite.

**Expected outcome / mandatory evidence for ACCEPT:** Private bounded array/setter only; correct flags/word bounds and valid stale cache retained.

### G05 — NumPy PCG64 codec

**Criteria:** C5, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1371–1469` (`_validate_single_codec_state`).

**Existing tests:** `test_rng_and_codecs_validation` at 1506–1553.

**Current evidence:** Valid PCG64 and even-inc refusal pass; other state components missing.

**Valid control:** Private PCG64 snapshot including nonzero inactive cached uint32.

**Required cases / gaps:** Outer/inner schema keys, engine PCG64DXSM mismatch, state/inc bool/negative/>uint128, even inc, has_uint32 -1/2/bool, uinteger negative/>uint32/bool.

**Expected outcome / mandatory evidence for ACCEPT:** Native private parser success or translated field-specific CVE, never global Generator use or invented cache-zero rule.

### G06 — Native Torch CPU codec and seed binding

**Criteria:** C5, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1371–1469` (`_validate_single_codec_state`).

**Existing tests:** `test_rng_and_codecs_validation` at 1506–1553; `test_r3_torch_worker_seed_integer` at 2303–2345.

**Current evidence:** Source validates native length/private setter/initial_seed; current test mainly lists-seed refusal, not bad native internals.

**Valid control:** Private CPU Generator(42) state, detached contiguous uint8 of loaded native length.

**Required cases / gaps:** Actual state wrong dtype/rank/length/contiguity/meta/grad where representable; all-zero native-length uint8 corruption must exercise native parser (no ABI offsets). W1: actual+expected worker seed changed42xxx->42xxx+1 but captured bytes unchanged, so native initial_seed mismatch reaches intended guard.

**Expected outcome / mandatory evidence for ACCEPT:** CVE from native parser or initial_seed, not only length; no global restore, CUDA, draws or legacy format conversion. No claim future cross-version continuation.

### G07 — Seed materials actual/expected/both and initialization lifetime

**Criteria:** C2, C5, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1094–1368` (`_validate_rng_state_dict`); `scripts/checkpoint_payload.py:1962–2113` (`_validate_expected_runtime_schema`).

**Existing tests:** `test_r3_torch_worker_seed_integer` at 2303–2345; `test_r4_python_worker_seed_integer` at 2347–2389.

**Current evidence:** Both-corrupt Python/Torch worker list repros pass. They can fail expected schema first, so do not prove all actual-only guards.

**Valid control:** W1 private captured Python/Torch/NumPy states; scalar seeds; NumPy-only worker nonempty uint32 word lists; archived initialization state supplied independently.

**Required cases / gaps:** Python/Torch actual-only/expected-only/both list,bool,negative; Torch uint64 and legacy MT uint32 intrinsic bounds; NumPy list empty/bad word and parent-list refusal. Python arbitrary large valid worker seed metadata permitted (no reachability proof). Missing init archive/binding refuses rather than fabricate/reseed.

**Expected outcome / mandatory evidence for ACCEPT:** Correct actual or expected seed guard identified; R3/R4 retained. No arbitrary master-seed cap, derivation-code execution, global worker launch or accepted-model changes.

### G08 — Complete live/global RNG preservation

**Criteria:** C5, C7, C9. **Implementation:** unverified; `preserve_implementation_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1371–1469` (`_validate_single_codec_state`); `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`).

**Existing tests:** `test_rng_and_codecs_validation` at 1506–1553.

**Current evidence:** Existing NumPy assertion at1520 compares only engine name, omits keys/cursor/cache; failure path omits NumPy entirely. Source uses private generators.

**Valid control:** Create all fixtures/mutations BEFORE snapshots; capture Python entire tuple, NumPy all5components with copied keys, Torch state bytes, default dtype.

**Required cases / gaps:** Compare immediately before/after successful both-model calls and late Adam/native-RNG/owned-revalidation failures. Use complete NumPy array/position/flag/cache equality, not engine string alone.

**Expected outcome / mandatory evidence for ACCEPT:** All states unchanged; no global setters even for teardown. This is an evidence gap, not a demonstrated implementation mutation.

### S01 — Positive/empty-negative domain P/K/T oracles

**Criteria:** C5, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1476–1495` (`_compute_condition_positive_and_empty_pool_pairs`).

**Existing tests:** `test_r5a_old_empty_pool_fixture_independent_derivation` at 2391–2427; `test_r5b_genuine_untrainable_t0_domain_refusal` at 2429–2622; `test_batch_then_filter_empty_chunk_and_partial_batch` at 1823–2061.

**Current evidence:** Historical independentP5/K2/T3, genuineP4/K4/T0 refusal and corrected BTF5edge controls pass.

**Valid control:** Keep independent literal pair sets/degrees, not only call candidate derivation.

**Required cases / gaps:** Retain T0 domain error and old6edge negative witness. Clearing order with P5/K2/T3 must be order failure, NOT mislabeled T0. Add changed low-rated all-rated exclusion case while retaining positive warm domain.

**Expected outcome / mandatory evidence for ACCEPT:** No domain algorithm or graph semantics rewrite; no real dataset or sampled negative execution.

### S02 — Sampler orders, quiescence and batch/progress counters

**Criteria:** C5, C6, C9. **Implementation:** verified; `preserve_logic_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1498–1651` (`_validate_sampler_and_progress`); `scripts/checkpoint_payload.py:276–278` (`_ceil_div`).

**Existing tests:** `test_sampler_and_progress_validation` at 1555–1573; `test_batch_then_filter_empty_chunk_and_partial_batch` at 1823–2061.

**Current evidence:** Current batch_cursor/permutation/large integer ceil and one BTF positive pass; field/history-policy matrix incomplete.

**Valid control:** E0 initial and completed controls for both policies; P5/K2/T3 B2 order has empty middle chunk and retained partial final chunk.

**Required cases / gaps:** Each sampler/progress key/type; algorithm/hash, source tuple element shape/type/duplicates/missing/extra, cursor,next_epoch, worker_ids and each4quiescent counters; E0 invalid histories/G/order; E>0 history length/sum/skips. Filter-before exact ceil; batch-then last exact count and earlier lower/upper bounds with last order unchanged.

**Expected outcome / mandatory evidence for ACCEPT:** Intended field/policy CVE. No float ceil, extra empty/regularization update, production batch-policy default or sampler/worker execution. D01 rendering only.

### T01 — Selection first-check, improvement, ties and first patience stop

**Criteria:** C6, C9. **Implementation:** verified; `preserve_function_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1658–1850` (`_validate_selection_and_best_model`).

**Existing tests:** `test_selection_history_and_patience_boundary` at 1575–1632; `test_best_model_snapshot_semantics` at 1634–1669.

**Current evidence:** First0.0, terminal22/10bad and epoch23 refusal, signed-zero inequality pass; function restored and pinned.

**Valid control:** E0 initial; E1 ready beforeF2; E2 first0.0; E4 strict improvement and exact tie; E22 patience.

**Required cases / gaps:** Independent all selection/history fields wrong type/missing/extra, finite range0..1/NaN/Inf, duplicate/missing/unscheduled/future checks, wrong summary/bad_checks9vs10/11, post-stop epoch/check including later apparent improvement, unsupported completed status.

**Expected outcome / mandatory evidence for ACCEPT:** Frozen strict > improvement, earliest tie and first-stop boundary preserved. COMPLETE FUNCTION text/AST hash unchanged in initial correction; fix fixtures, not selection logic.

### T02 — Max epoch alone and simultaneous patience/max schedule

**Criteria:** C6, C9. **Implementation:** unverified; `preserve_function_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1658–1850` (`_validate_selection_and_best_model`).

**Existing tests:** `test_selection_history_and_patience_boundary` at 1575–1632.

**Current evidence:** All3stop reasons present in source; no dedicated max-only/combined tests currently.

**Valid control:** E100/F3: increasing scores at3..99 selects99, max_epochs stop, no invented100check. E100/F2: strictly improve through80 then10ties82..100, selects80 and patience_and_max_epochs.

**Required cases / gaps:** Reject ready/initial/wrong reason at terminal; missing/extra final check; wrong selected/bad_checks; use valid progress histories, G=2*E, matching Adam scalar steps and best_model epoch.

**Expected outcome / mandatory evidence for ACCEPT:** Both valid terminal controls pass and each isolated corruption refuses correct selection guard; no max_epochs/F/default/protocol changes.

### T03 — Current/best binding and exact-bit identity

**Criteria:** C3, C6, C7, C9. **Implementation:** verified; `preserve_function_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1658–1850` (`_validate_selection_and_best_model`).

**Existing tests:** `test_best_model_snapshot_semantics` at 1634–1669.

**Current evidence:** Before-checkNone and equal-epoch signed-zero refusal pass; all best fields and both recipes not covered.

**Valid control:** No-check bestNone; equal-epoch bit-identical; older best epoch with either coincidentally equal or different finite weights valid.

**Required cases / gaps:** Missing best after check/present before; best epoch future or wrong selection; row order/schema/tensor cases; same-epoch dense/sparse content difference including +/-0; ensure current-resume weights not replaced by earlier-best snapshot.

**Expected outcome / mandatory evidence for ACCEPT:** No equality requirement for different epochs or tolerance for same epoch. Preserve selection function, add both-model tests.

### O01 — Owned output and separate input/expectation/container/storage aliases

**Criteria:** C7, C9. **Implementation:** unverified; `preserve_implementation_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1857–1883` (`_make_owned_output_copy`); `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`).

**Existing tests:** `test_ownership_and_no_mutation` at 1025–1070.

**Current evidence:** Existing test covers one current user tensor/epoch and two fields after failure; complete alias evidence absent. No known copy implementation defect from source read.

**Valid control:** Both complete model recipes, selected best, initialized Adam, RNG and shared input storage/views; each expected descriptor detached before mutation tests.

**Required cases / gaps:** Snapshot all actual/expected inputs. Independently mutate inputs and each expected_identity/provenance/model/runtime/policy container after return, verify result unaffected; mutate returned current/best/Adam step+moments/RNG bytes/nested metadata/sparse indices+values and verify original inputs/expectations unaffected. Also mutate one returned occurrence and check other copies not aliased.

**Expected outcome / mandatory evidence for ACCEPT:** Content AND mutable-storage independence, not pointer-only claims or scalar samples. Returned result remains mutable and requires revalidation after edits. No copy rewrite without a new demonstrated defect.

### O02 — Complete no-mutation on early/late failure; owned semantics

**Criteria:** C7, C8, C9. **Implementation:** unverified; `preserve_implementation_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`); `scripts/checkpoint_payload.py:1857–1883` (`_make_owned_output_copy`).

**Existing tests:** `test_ownership_and_no_mutation` at 1025–1070.

**Current evidence:** Static raw-before-copy and owned revalidation present, selected-scalar failure evidence insufficient.

**Valid control:** Unpatched complete controls, deep structural snapshots of built-in input/expected values and full RNG state.

**Required cases / gaps:** Early type/resource, late selection, Adam moment, final native RNG failure and deliberately bad owned copy; compare ALL input/expected contents and native states before/after; no partial result. For hostile objects use hook counters/identity sentinels instead of deepcopy invoking hooks.

**Expected outcome / mandatory evidence for ACCEPT:** No input/expectation/live mutation and no success return. Controlled test mocks restored. No live model/optimizer load or restoration.

### D01 — Whole-path bounded domain diagnostics

**Criteria:** C2, C8, C9. **Implementation:** failed; `targeted_bounded_rendering_only`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:181–214` (`_safe_repr`); `scripts/checkpoint_payload.py:176–178` (`_safe_sort_keys`); `scripts/checkpoint_payload.py:710–770` (`_validate_model_state_dict`); `scripts/checkpoint_payload.py:1962–2113` (`_validate_expected_runtime_schema`); `scripts/checkpoint_payload.py:1094–1368` (`_validate_rng_state_dict`); `scripts/checkpoint_payload.py:777–1087` (`_validate_adam_optimizer`); `scripts/checkpoint_payload.py:1498–1651` (`_validate_sampler_and_progress`); `scripts/checkpoint_payload.py:2120–2333` (`_validate_semantic_payload`).

**Existing tests:** `test_fail_closed_error_handling` at 1236–1258; `test_r2_huge_update_domain_error` at 2262–2301.

**Current evidence:** Static remaining raw list/repr/sorted paths at729,893,901,922,1137,1527,1982–1984,2065–2067,2089,2256; R2 fix is not whole-path evidence.

**Valid control:** Each valid public control passes first. H=10**5000 remains within logical metadata budget and interpreter settings unchanged.

**Required cases / gaps:** Replace ALL LightGCN state keys with homogeneous H,H+1,H+2 (not mixed int+str, which would fail earlier). Expected codec=H; actual role.name=H; sampler.algorithm=H; actual runtime_profile.profile_id=H; params[0]=H; betas=(H,0.999); expected role-binding keys all H+i; expected runtime_profile keys homogeneous tuples (0,), ("x",) to expose sorting TypeError. Add long clean strings and many keys to verify aggregate message bound; malformed Unicode must domain-refuse at preflight.

**Expected outcome / mandatory evidence for ACCEPT:** CheckpointValidationError with role/field and proposed <=4096-character diagnostic budget; no native ValueError/TypeError/OverflowError, interpreter-limit changes or integer validity cap. Preserve all guard predicates/numerical rules; central bounded collections formatter may be added. Budget is proposed test detail, not rewritten original C8.

### D02 — Reject custom objects/cycles without hooks

**Criteria:** C8, C9. **Implementation:** verified; `preserve_type_dispatch_add_tests`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:147–173` (`_safe_type_name`); `scripts/checkpoint_payload.py:181–214` (`_safe_repr`); `scripts/checkpoint_payload.py:370–559` (`preflight_resource_accounting`); `scripts/checkpoint_payload.py:285–338` (`validate_resource_policy`).

**Existing tests:** `test_custom_objects_and_type_safety_without_hooks` at 1174–1234; `test_r1_metaclass_hash_hook` at 2209–2260.

**Current evidence:** R1 zero metaclass-hash events and selected custom subclasses pass; exact identity dispatch fixed.

**Valid control:** Full control first; malicious instance only introduced afterward; reset counters after construction.

**Required cases / gaps:** Actual payload and EACH expected input plus policy: hostile hash/eq/repr/str/getattribute/getattr, dict/list/int/str/Tensor subclasses, ndarray/Parameter, cycles. Do not let snapshot/test rendering invoke hooks and then blame validator.

**Expected outcome / mandatory evidence for ACCEPT:** Existing exception family and exactly0hook events; reject before caller hooks/native parser/copy. Preserve _safe_type_name identity dispatch and no arbitrary object coercion.

### V01 — Full regression evidence and per-case closure coverage

**Criteria:** C1, C9, C10. **Implementation:** unverified; `tests_and_evidence_only`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`).

**Existing tests:** `test_positive_lightgcn_payload` at 959–977; `test_positive_heterognn_payload` at 979–995.

**Current evidence:** Historical283green is real but does not certify missing cases. No new tests executed in preparation.

**Valid control:** Fresh protected dependency baseline249, then exact starting candidate283 bootstrap before new edits.

**Required cases / gaps:** Add new labeled test class/helpers; keep all34 existing method ASTs and249 tests unchanged. Every planned case ID must map to actual execution/pass/failure and source/test hash; emit compact coverage ledger only after assertions succeed INSIDE subTest blocks, never outside swallowed failures. Final Pi full suite uses exact predeclared command,120s, no skips or truncated mandatory results.

**Expected outcome / mandatory evidence for ACCEPT:** Complete original C1–C10 review plus case manifest; method count alone not closure. All failures/repeats retained, no alternative candidate execution. No training or held-out policy authorized.

### V02 — Disclosure, docs, limitations and action reconciliation

**Criteria:** C10. **Implementation:** failed; `docs_and_report_correction_only`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:2340–2401` (`validate_checkpoint_payload`).

**Existing tests:** `test_positive_lightgcn_payload` at 959–977.

**Current evidence:** Worker report says completed/no issues, only310pass; raw272failed250/1,282failed283/2,300pass283,310pass283; initial152failure/191interrupted/timeout omitted. Doc872–1004 overclaims complete matrices.

**Valid control:** Frozen59605byte specification and all historical reports/artifacts retained unchanged.

**Required cases / gaps:** New active appended docs must distinguish source review, executed tests, pending gaps and unsupported capabilities; include all attempts from this history and future invocations with exact commands/results/status refs. Reconcile partial inventory with raw steps (315hashes checked;83previously recovered), dedup mirrors by step. Each new report limitation adjudicated verbatim/in order by Pi; unresolved items block ACCEPT.

**Expected outcome / mandatory evidence for ACCEPT:** No retroactive editing of original reports/criteria/decisions; no future-success prose before evidence. Same-UID partial records are not complete protected syscall history. Task-only acceptance separate from deployment NOT_AUTHORIZED.

### A05 — Adam foreach/fused type checks before equality

**Criteria:** C2, C4, C8. **Implementation:** failed; `targeted_type_before_comparison_fix`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:777–1087` (`_validate_adam_optimizer`).

**Existing tests:** `test_adam_optimizer_validation` at 1387–1504.

**Current evidence:** New static finding: source938–946 compares act_foreach/act_fused with trustedFalse BEFORE checking type. A multielement exact Tensor reaches ambiguous Tensor truth conversion. No new runtime probe executed.

**Valid control:** Set foreach=False or fused=False in BOTH actual and expected options, validate full control; also retain None/True foreach controls.

**Required cases / gaps:** Replace only actual flag with torch.tensor([0,1]) under the validFalse profile; independently test foreach/fused, plus scalar/bool-like integer and other wrong types. The generic preflight accepts this exact CPU tensor, so intended option type guard must handle it.

**Expected outcome / mandatory evidence for ACCEPT:** Field-specific CheckpointValidationError, not native ambiguous-bool RuntimeError. Type-check None/bool alternatives BEFORE equality; preserve all well-typed option semantics, lazy/aggregate/selection guards.

### G09 — Parent/local seed type validation before bound-role equality

**Criteria:** C2, C5, C8. **Implementation:** failed; `targeted_seed_type_before_comparison_fix`. **Tests:** additional_evidence_required.

**Source:** `scripts/checkpoint_payload.py:1094–1368` (`_validate_rng_state_dict`).

**Existing tests:** `test_rng_and_codecs_validation` at 1506–1553; `test_r3_torch_worker_seed_integer` at 2303–2345; `test_r4_python_worker_seed_integer` at 2347–2389.

**Current evidence:** New static finding: parent/local branch at1233–1240 compares r_seed with g.seed_material before codec-specific seed checks at1247+. A multielement exact Tensor causes native ambiguous truth conversion. Existing worker tests bypass this parent branch.

**Valid control:** Existing valid parent Python/NumPy/Torch seed42 controls; worker controls retain independent derived material.

**Required cases / gaps:** For EACH parent codec, change only actual seed_material to torch.tensor([42,43]); preserve trusted42 and valid states. Also test other malformed supported leaves, actual/expected/both where applicable.

**Expected outcome / mandatory evidence for ACCEPT:** Seed-field CheckpointValidationError before equality/native parser, no RuntimeError. Validate codec-specific seed type/range before parent equality; retain Python arbitrary-size support, intrinsic NumPy/Torch limits and distinct worker derivation semantics.

## Criterion coverage index

| Original criterion | Part IDs (all required) |
|---|---|
| C1 | P01, E02, V01 |
| C2 | F01, E01, E02, E03, E04, A01, A02, G01, G07, D01, A05, G09 |
| C3 | R01, R02, R03, M01, M02, M03, T03 |
| C4 | E03, E04, A01, A02, A03, A04, A05 |
| C5 | E03, E04, G01, G02, G03, G04, G05, G06, G07, G08, S01, S02, G09 |
| C6 | F01, S02, T01, T02, T03 |
| C7 | R03, G08, T03, O01, O02 |
| C8 | E01, E03, E04, R01, R02, R03, A01, G01, O02, D01, D02, A05, G09 |
| C9 | P01, F01, E02, R02, M01, M02, M03, A02, A03, A04, G02, G03, G04, G05, G06, G07, G08, S01, S02, T01, T02, T03, O01, O02, D01, D02, V01 |
| C10 | V01, V02 |

## Preservation and handoff

All original249 tests and34 payload method bodies are pinned; add tests instead of rewriting verified implementation. Exact source/function/block fingerprints and protected boundaries are specified in the companion [correction assignment](checkpoint-payload-correction-assignment.md). In particular, Adam aggregate and the entire selection function are protected. Any justified change to a protected part needs explicit Pi impact review before that edit; do not silently broaden the work.

The machine-readable counterpart is `data/reports/d2-part-level-preparation/matrix.json`. Raw-evidence reconciliation is in `evidence-audit.json`; the failed first metadata-helper attempt is preserved in `preparation-attempts.json` (not a candidate test). Source and histories remained unchanged.

Read [the proposed correction assignment](checkpoint-payload-correction-assignment.md) for order, exact controls, proposed limits and authorization gates. This preparation does not reopen an exhausted task or authorize a new Worker, main transfer, D3, profiling/training, GPU/Kaggle, held-out evaluation or deployment.
