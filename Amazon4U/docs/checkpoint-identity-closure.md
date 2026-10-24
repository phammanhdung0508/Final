# Focused identity/provenance correction

## Prospective approval and limits

The user explicitly agreed to proceed with the proposed focused correction:
retain the existing design, fix invalid-Unicode domain errors, complete required
adversarial coverage, correct docs/report, and repeat independent review and the
full benchmark. This records that approval BEFORE implementation. Routine in-scope
fixes are Pi's responsibility; do not ask the user to choose error messages or
fixtures. No new scientific choice or execution default is selected.

ONE whole-deliverable correction invocation, max_revisions=0, explicit long_running
Worker, finite 120s full-suite verification. Preserve task
`9617ef2b-5a76-4376-8aee-aa36a3e08a74` FINAL ESCALATE and all older decisions,
exhausted budgets, drafts, commands and runtime artifacts. No reopen/reset or
partial acceptance. Exactlythree substantive deliverables remain; this is still
the first whole identity/provenance deliverable, not separate test/error microtasks.

Fresh isolated Git root `Amazon4U/.worktrees/checkpoint-identity-closure-review`,
HEAD `fd15dc3b5fe665f45d8787ba779b9e9541690bde`, project subdir `Amazon4U/`.
Only `scripts/checkpoint_identity.py`, `tests/test_checkpoint_identity.py` and
`docs/checkpoint-identity.md` may change inside that project. Protected latest
rejected files are starting material, NOT accepted implementation. No broad rewrite,
new schema/architecture, new dependency or changes to accepted model/graph code.
Keep the structured RNG/worker/thread recording schema and literal hashes unchanged
for all previously valid inputs. No protocol freeze or invented actual settings.

No real data/source capture, profiling, training, optimizer steps, tensors in NEW
tests, CUDA/GPU/Kaggle/heldout, subagents, staging/commits/pushes, sibling writes,
payload validation, secureIO or live restoration/training-loop work. New tests are
standard-library CPU memory fixtures <=8 users/12 products/8 metadata nodes/type;
source/provenance bound 2MiB unchanged (small boundary-negative excess allowed).
Task-only acceptance; deployment NOT_AUTHORIZED, no deployment_policy.

## Consolidated correction worksheet

Retain all 204 baseline and 39 existing contract test methods and all earlier mandatory
requirements. The following are EXACT missing groups, not optional examples. Use
otherwise-valid COMPLETE fixtures; each invalid case changes only the intended
field (plus documented dependencies necessary for a valid ACTIVE-worker fixture).
Assert specific exception family AND stable guard/field diagnostic with
assertRaisesRegex/subTest or equivalent. No incidental failure can count as coverage.

### 1. source: malformed unicode and numeric error boundaries

- Preserve ensure_ascii=False and compact literal-UTF 8 CategoryPath spelling at
  BOTH validator and formatter; escaped-ASCII aliases remain rejected.
- Reject literal lone surrogate labels (e.g. Python string chr(0xD800) or chr(0xDFFF))
  clearly with CheckpointValidationError in category validator AND formatter.
- Canonical data/graph/validation/settings/composite JSON encoding must not leak
  UnicodeEncodeError or numeric JSON conversion ValueError/OverflowError for malformed
  caller content. Use small shared private text/serialization helpers or targeted
  guards, not blanket acceptance, replacements/Unicode normalization, ensure_ascii=True,
  silent defaults or process-global changes to integer limits/RNG/thread settings.
- Preserve domain errors in builder/from_dict and actual/expected validators. Cover
  bad text in user/item/category/brand/header/settings, plus a positive full identity
  and roundtrip containing snow/non-ASCII labels through graph nodes/edges/metadata.
- Test numeric overflow in training rating, edge weight, validation rating and
  numeric settings including betas. Include 10**400 and 10**5000: formatting an
  oversized offending int must not itself leak ValueError. JSON conversion of huge
  positive seed/timestamp must fail with a clear domain error, not raw serializer
  exception. Do not claim unbounded JSON integer support or set interpreter limits.
  Values are tiny memory synthetic scalars, not workload/resource expansion.

### 2. RNG RECORD schema, both applicability branches, semantic sensitivity

- Independently validate registered name/duplicate/missing/unknown component cases,
  preserving all other valid components. For EAC H6 record keys
  name/applicable/algorithm/seed/derivation/reason: omission; unknown and mixed-type
  extra record keys; wrong record container and strict applicable bool.
- Valid ACTIVE record: nonempty unpadded algorithm/derivation, nonnegative int seed
  not bool, reason=None. Invalid cases include empty/padded/non-string algorithm and
  derivation, bool/string/float/negative/None seed, non-None reason.
- Valid INACTIVE record: algorithm/seed/derivation=None, nonempty unpadded reason.
  Invalid cases independently change each None field and empty/padded/non-string
  reason. Name-only old records must be rejected, not silently upgraded.
- Include each registered role at least once in otherwise-valid case coverage;
  for DataLoader_workers choose consistent runtime worker fixtures. This is synthetic
  metadata, not execution of any RNG, GPU or worker. No new method-specific policy.
- Independently mutate ACTIVE algorithm, seed, derivation and INACTIVE reason;
  build valid new identities, assert changed settings AND composite fingerprints
  and mismatch against trusted base. Do not change several unrelated fields together.
- Retain ordered-stream sensitivity and tuple/list equivalence. Corrupt a nested
  record in BOTH actual and expected identity validation, not only root settings.

### 3. worker derivation/settings: valid active base, isolated field changes

- Build ONE complete valid positive-worker fixture with consistent runtime_workers,
  worker_settings, worker_derivation and DataLoader_workers record. Never launch it.
- worker_derivation EXACT 5 keys: each missing key, unknown/mixed key, wrong container,
  applicable strict bool, active algorithm/base_seed/derivation/reason type/shape,
  inactive None fields/reason. Base seed mismatch and applicability mismatches must
  reach their respective consistency guards.
- Active algorithm and derivation isolated valid changes; inactive reason sensitivity.
  Active base_seed==determinism.seed is a real dependency: a valid base-seed change
  changes BOTH bound fields, explicitly distinguish from independent field testing.
- worker_settings EXACT 4 keys: missing/unknown/mixed/wrong container; zero-worker
  constraints; positive-worker bool/nonpositive/nonint prefetch, nonbool persistent,
  empty/padded/nonstring context/init identifiers. Every invalid case begins from
  the complete positive-worker fixture, not an already inconsistent zero-worker one.
- From that valid active fixture, change EACH prefetch_factor, persistent_workers,
  multiprocessing_context and worker_init_function independently; assert changed
  settings/composite fingerprints and trusted expected mismatch. Independently vary
  runtime_workers (positive->different positive) and cpu_threads; reject bad types.
- Deep input/output worker_derivation/RNG/worker_settings aliasing and post-mutation
  expected comparison; do not assume shallow frozen dicts are immutable.

### 4. serialized identities and intended history guards

- from_dict wrong/missing schema, unknown string key, mixed keys0+unknown, missing
  retained field(s), malformed retained collection/nested types, and EACH falsified
  stored fingerprint/hash (valid 64 hexwrongvalue). Direct constructor mutation alone
  does NOT cover serialization. Require schema/key/type/content diagnostic intended.
- Preserve equivalent roundtrips and actual/expected corrupt nested config cases.
- Keep BOTH u1/u2 history keys and every unrelated valid item; replace only one u1
  item with dict/int in all-rated/positive history tests. Assert item-element guard,
  not key mismatch or earlier all-rated agreement failure.
- Preserve isolated metadata-only, graph-only, validation-only and multi-category
  ordering cases; do not regress them while adding tables.

### 5. provenance validator roles, preflight, snapshot isolation

- Maintain missing/unknown top/file/Git/environment keys and entry-path/size/hash/
  total/bytes cases. Add malformed untracked MEMBERS bool, int, dict/list, blank/padded
  strings and forbidden/traversal paths in BOTH actual AND expected validator roles.
  Start with dirty=True, otherwise-valid untracked evidence and byte coverage, so
  intended member/path guard is reached. SecurityErrors for forbidden paths;
  ProvenanceErrors for member types. Retain duplicate/missing-byte coverage.
- Builder and BOTH validator roles: valid small first file plus oversized later
  file; validator declared total within limit. Assert aggregate-content diagnostic
  AND ZERO hashlib.sha256 calls (not just any broad exception). Source bytes remain
  in memory; no conversion/acquisition. Preserve early declared-total and exact 2MiB.
- Deeply isolate nested library_versions and untracked lists, including output
  mutation compared against an independently built trusted expected record. No
  universal tamper-proof claim when caller mutates both actual and expected evidence.

### 6. documentation/report evidence accuracy

- Preserve correct actual list/tuple shapes, relative links, caller trust boundary,
  shallow frozen nested mutability and ordered stream/fanout exceptions. Record
  malformed Unicode and unsupported numeric serialization behavior/domain errors.
- Scope tamper detection to recomputed canonical content AND unchanged trusted
  expected evidence. Do not promise proof of source origin, automatic resource/
  determinism enforcement, all conceivable Python object behavior or universal
  mutation detection with aliased/untrusted expected evidence.
- Pending Pi acceptance; no 'strictly all checklist complete/comprehensive/no issues'
  assertion unless supported by the exact named tests. Required JSON report maps
  groups1..5 to actual test method names and honestly names remaining limitations.
- All failed/repeated full-suite runs and in-invocation fixes must be reported with
  exact command, result and count. Do not omit failures or claim Pi approval.

## Only approved code/test execution

Original and amended command records are preserved. Only the scratch path literal
changes prospectively under standing noncritical layout authorization; executable,
-B/-c program, discovery('tests'), verbosity, safeguards, exit semantics and 120s
ceiling stay unchanged. This is NOT byte-for-byte reuse of old absolute-path command.
Only submitted verification_commands[0] may execute code/tests, synchronously.
No subset unittest, smoke import, candidate oracle probe, ad-hoc Python, compileall,
detached/background tests, source acquisition or alternate wrapper. Read-only
navigation/copying protected starting files allowed; Git reads optional-lock-free.
Independent literal printf/sha 256 sum authoring arithmetic remains permitted if needed
and must be reported; no hash should change for existing valid fixtures in this fix.

Separate fresh external worker-runtime and fixture-scratch dirs; preserve every
old scratch/runtime artifact. No test writes to source/pre-existing artifacts/Git.
Pi will inspect actual complete changed functions/callers/dependencies/tests/docs,
independently run the EXACT submitted full-suite benchmark, evaluate every original
criterion and worksheet group, freshly inspect then record final ACCEPT/ESCALATE.
No partial acceptance, automatic followup after final review or deployment approval.

## Final independently reviewed result

Task `5eab9071-04ab-427f-a920-fe420f5972ee`, revision 0/max_revisions0;
invocation `6fc77521-5235-4c74-8591-662396bdf262`, completed normally in
1,725,209ms under explicit long_running. Worker reported TWO exact full-suite runs:
248 tests/1 error/12.215s (test subTest formatting of10**5000), then 248 OK/8.591s.
Pi independently ran the EXACT submitted full suite: 248 tests/0 failures/errors,
7.108s unittest/9,542ms process. All 204 baseline and 39 prior contract methods remain;
5 new named matrix methods were added. Benchmark
`agy-evidence-698f05db-0bd1-4db9-bc9a-e60c1175713c`.

Pi read ALL 2549 source/3057 test/265 doc lines plus relevant accepted typed-ID and
reproducibility dependencies. Independent AST literal/JSON/SHA256 review confirmed
unchanged valid settings/composite preimages/hashes, without candidate import/probe.
Fresh inspection `agy-evidence-135da011-2bb9-47df-a323-baf745bbd1ad`: exactly 3 allowed
paths, unauthorized[], Git metadata unchanged, 142 protectedfiles/86,536,910bytes and
130 outside sentinels. Previous draft/HEAD/index/status independently unchanged;
fixture scratch empty, runtime artifacts retained. Bounded monitoring is not an OS
sandbox or complete action audit; captured 150 action candidates are navigation only.

Final review `agy-evidence-29cd9f02-617d-4637-8169-a35d78f5c163`: **ESCALATE**,
acceptance_scope **task_only**, deployment **NOT_AUTHORIZED**. No main transfer,
automatic revision or subsequent invocation. Earlier final decisions remain intact.

### Concrete progress

Lone-surrogate validation/formatting and safe canonical encoding/domain diagnostics
implemented; full positive nonASCII identity/roundtrip passes. RNG/worker record
omission/type/activeinactive matrices, independently sensitive worker configuration,
serialized retained-type/allfingerprint cases, both-role untracked member/path
matrices, zero-hash lateaggregate guard assertions and independent nested output
isolation were actually added. This is significant closure of earlier omissions.

Actual new test method map (Pi independently verified, absent from Worker report):
1. `test_invalid_unicode_domain_errors_and_numeric_overflow`.
2. `test_rng_stream_record_matrix_and_independent_mutations`.
3. `test_worker_settings_and_derivation_matrix`.
4. `test_from_dict_and_serialized_matrix`, plus existing corrected history cases.
5. `test_provenance_untracked_members_and_aggregate_diagnostics`.

### Specific remaining blockers

1. New validation introduces an undocumented/arbitrary `bit_length()>14000` cutoff
   for headerseed/timestamps/settings, inaccurately labeled maximum JSON conversion.
   Static counterexample: 2**14000 has 14001 bits but 4215 decimal digits, BELOW the
   normal 4300 digit serializer limit; previous JSON path supports it, new guard
   rejects it. Preserve safe diagnostic truncation, but translate ACTUAL serializer
   failures without silently narrowing previously valid exactinteger evidence.
   This conclusion is static code/math review, not an unauthorized candidate probe.
2. Most newUnicode/numeric/RNG/worker/serialized/history negative cases still assert
   broad CheckpointValidationError only. subTest labels do not assert the rejection
   guard/field diagnostic. Only stored-fingerprint cases and THREE lateaggregate
   cases check diagnostic content. The predeclared intended-guard requirement is
   not satisfied by headings or passing counts.
3. Docs 52 wrongly describes10**400 as exceeding 4300 digits: it has 401 digits and
   overflows finite float conversion, not that integer-string limit. Actual 14000 bit
   cutoff is not documented. Worker summary claims all 6 groups complete with no
   unresolved issues and omits requested group-to-method mapping; that claim is not
   supported by remaining conformance evidence.

These findings are routine correctness/documentation fixes, NOT new scientific
choices. The one-invocation/zero-revision limit was prospectively set by Pi; Pi must
not disguise that workflow limit as an unresolvable scientific blocker, silently
reset it, or waive acceptance. No broad rewrite is justified. Any further recovery
needs a prospectively authorized scope/budget preserving this final evidence.

Evidence directory: `data/reports/checkpoint-identity-closure-setup/` (original/current
commands/spec, baseline, checksum-verified Worker report/full benchmark stderr,
independent literal preimages and preservation manifests). Exactlythree substantive
checkpoint deliverables remain; first unaccepted, later payload validation and
secureIO/validation-only integration still blocked. No profiling/training/heldout/
GPU/Kaggle/deployment permission was created.
