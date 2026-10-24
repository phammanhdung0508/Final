# Bounded identity/provenance repair

## Authorization and boundaries

The user approved the proposed tightly bounded repair of the existing rejected
identity/provenance draft and explicitly reaffirmed Pi's authority to decide and
act on routine blockers following the agreed design without repeated questions.
Only consequential scientific/security/data/workload decisions outside that design
require user intervention. This is not permission to waive criteria, reset budgets,
run endless retries or claim an ESCALATE is an acceptance.

One whole-deliverable repair invocation, zero automatic revisions, followed by Pi
independent actual-file review and benchmark. Preserve all previous FINAL ESCALATE
records and drafts, including task `54a7629c-75e8-47a7-a5d0-2bdb89873f65`.
Exactly three checkpoint deliverables remain. No main implementation transfer,
profiling/training, real data, optimizer steps, GPU/Kaggle/held-out execution,
subagents, staging/commits/pushes or deployment. Acceptance remains task-only;
deployment NOT_AUTHORIZED. Payload/state validation and secure IO/integration are
separate later deliverables, not live restoration or full training loops.

Clean Git root: `Amazon4U/.worktrees/checkpoint-identity-repair-review`, HEAD
`fd15dc3b5fe665f45d8787ba779b9e9541690bde`, project subdirectory `Amazon4U/`.
Only its `scripts/checkpoint_identity.py`, `tests/test_checkpoint_identity.py` and
`docs/checkpoint-identity.md` may change. Latest rejected files and this precise
checklist are protected references. Worker explicit long_running; original full
benchmark ceiling 120 seconds; source bound 2 MiB and bounded synthetic fixtures
unchanged. Fresh runtime and fixture dirs are separate outside the clone.

## Exact repair checklist: contract first, coverage, docs, benchmark/report

### A. source correctness and representational configuration schema

1. Restore `ensure_ascii=False` in BOTH category validator canonical spelling and
   category formatter. Preserve the established producer/model compact literal-UTF-8
   identities; no changes to accepted model/graph code. Test non-ASCII labels and
   reject escaped-ASCII aliases when literal spelling is required.
2. Validate EVERY serialized identity key is a string BEFORE schema/key-set
   diagnostic sorting in `from_dict`. Maintain common no-padding header/fingerprint
   checks. Otherwise-valid extra keys `0` and `"unknown"` must raise the domain
   error, not incidental TypeError. Audit other public validation entry points for
   the same mixed-key/unhashable/numeric-overflow pattern; targeted fixes only.
3. Retain all existing resolved fields, including the seven previously repaired
   numerical/model/sampling fields. Replace name-only RNG labels with actual strict
   caller-evidence configuration records. This is REPRESENTATION, not selection of
   scientific settings, RNG initialization/state capture or execution defaults.

The precise recording schema for this repair is:
- `determinism.rng_streams`: list/tuple of records with EXACT keys
  `name, applicable, algorithm, seed, derivation, reason`.
- Exactly one record for EACH configured reproducibility component:
  `Python`, `NumPy`, `PyTorch_CPU`, `PyTorch_CUDA_if_applicable`, `initialization`,
  `epoch_shuffle`, `negative_sampler`, `neighbor_sampler`, `DataLoader_generator`,
  `DataLoader_workers`. Reject missing, duplicate and unknown component names.
- `applicable`: strict bool. If true: algorithm and derivation are nonempty
  unpadded strings, seed is nonnegative int (not bool), reason is explicitly None.
  If false: algorithm/seed/derivation are explicitly None and reason is a nonempty
  unpadded string. No silent defaults or name-only reconstruction. Algorithms and
  derived seeds are caller-supplied evidence; no implementation of derivations.
- Preserve list order as recorded semantic stream registration order; equivalent
  tuple/list carriers normalize equally. Document this ordered exception clearly.
- `determinism.worker_derivation`: EXACT record with
  `applicable, algorithm, base_seed, derivation, reason`, with analogous strict
  active/inactive rules. Applicability must agree with runtime_workers > 0.
  Active base_seed must agree with determinism.seed, without forcing every stream
  to use the same derived seed. DataLoader_workers applicability must also agree.
- `execution.runtime_workers`: existing nonnegative int (not bool).
- Add `execution.worker_settings`: EXACT record
  `prefetch_factor, persistent_workers, multiprocessing_context, worker_init_function`.
  For zero workers: None, False, None, None explicitly. For positive workers:
  positive int prefetch_factor, strict bool persistent_workers, nonempty unpadded
  context/init-function identifiers. These record caller's resolved implementation
  settings without launching workers or choosing actual execution defaults.
- Add `execution.cpu_threads`: positive int (not bool), caller-evidence only; do
  not change process thread settings or pretend this enforces resource limits.

Synthetic fixtures explicitly label their values as synthetic. Inapplicable
components require reasons, not fabricated algorithms/seeds or omitted evidence.
No global protocol/version/execution freeze is changed. Update component schema
version if necessary and document compatibility rejection; do not alter the frozen
scientific protocol. Every supported recorded semantic change must affect identity
and reject against trusted expected content. No RNG-state/optimizer/payload APIs.

### B. mandatory meaningful coverage, not inflated wrappers

Keep existing tests; add small table-driven cases starting with otherwise-valid
complete fixtures, with message/guard assertions where meaningful:
- RNG component/field omissions, duplicates, unknowns, applicability, strict types,
  algorithm/seed/derivation mutations; runtime worker/prefetch/thread fields and
  consistency; nested actual AND expected corruption and snapshot aliasing.
- All settings sections: missing/unknown top section and missing/unknown fields.
- Serialized identity wrong schema/version, missing/unknown/mixed-key fields,
  malformed retained types and falsified fingerprints; header padding and overflow.
- Fix unhashable-history fixture WITHOUT dropping u2: only replace the intended
  u1 element so that the element guard, not user-key mismatch, is tested.
- Isolated equal-count metadata mutation: change ONLY metadata, keeping graph
  nodes/edges constant; separately test graph-only and validation-only changes.
- Multi-category metadata ordering through full validator and to_dict/from_dict;
  equivalent tuple/list inputs with identical ordered streams/fanouts.
- Provenance ACTUAL and EXPECTED mutations of entry path, missing/unknown top/file/
  Git/environment keys, totals/sizes/hashes/bytes, bool sizes, malformed/duplicate/
  forbidden untracked members and byte coverage; correct intended exception family.
- Builder and BOTH validator roles: valid small first file, oversized later file,
  declared total within limit so the aggregate-content guard is actually reached,
  mock asserting ZERO hashing/conversion before rejection. Keep existing early-
  declared-total tests and exact accepted 2 MiB boundary. Negative fixture payload
  should be only a small boundary excess, not a giant memory allocation.
- Nested provenance library/untracked snapshot isolation, metadata/validation/RNG/
  worker input and output aliasing; post-mutation detection. No real file capture,
  new tensors/optimizer steps/unsafe serialization or external test processes.

Preserve unchanged data/graph/validation literal oracles. Legitimate settings/hash
changes require independent exact canonical-content preimages and literal hashes,
not values computed by the candidate API or empty/nonempty assertions.

### C. documentation and report accuracy

Fix section3.1 categories/validation target items/validation records to actual LIST
shapes; preserve correct tuple shapes for training records/graph edges/node IDs.
Document all new fields, applicability, caller evidence, domain errors, bounds,
mutable nested state and ordered-list exceptions. Use relative source links, not
stale absolute clone URIs/line anchors. Remove unsupported 'comprehensive complete'
guarantees. Label pending Pi acceptance and state only actual verification evidence.
Do not assign restoration/training-loop implementation to later deliverables:
they are payload VALIDATION and exclusive no-clobber IO/validation-only integration.

### D. sole full-suite command and truthful complete report

Record scratch-literal relocation prospectively under standing noncritical recovery
permission. Preserve exact original and approved command records; executable,
-B/-c, discovery over tests, verbosity, environment safeguards and exit behavior
remain unchanged. This is not byte-for-byte reuse of the old absolute-path command.

Execute ONLY submitted verification_commands[0], synchronously, for ALL test/code
execution. No subset unittest, smoke import, candidate oracle probes, ad-hoc Python,
compileall, detached/background runs or modified wrappers. Read-only navigation
and copying the declared protected draft into allowed source paths are permitted;
use optional-lock-free Git reads. Independent literal-preimage SHA calculation
using printf/sha256sum is explicitly allowed as source-authoring arithmetic, NOT a
candidate test or benchmark; record those commands honestly in report limitations.
Do not scan real data/credentials or the entire workspace to find evidence.

Any failed full-suite runs and in-invocation corrections must be reported along
with every same-command run/result. Iterate within this ONE invocation until the
consolidated checklist is honestly met; no new delegation/revision budget. If a
remaining issue requires a consequential unapproved decision, explain it instead
of inventing defaults or approval. Required JSON must distinguish completed code,
actual tests, limitations/unresolved issues and pending Pi acceptance. A long
completed report cannot substitute for absent tests or inaccurate documentation.

Pi independently reads complete actual code/tests/docs and relevant dependencies,
inspects and reruns the predeclared benchmark, evaluates every original criterion
and all eight requirements, then records final ACCEPT or ESCALATE. No partial
acceptance, implicit deployment or automatic source tuning after final review.

## Recorded result: FINAL ESCALATE

Task `9617ef2b-5a76-4376-8aee-aa36a3e08a74`, revision0/max_revisions0;
invocation `caf4b67a-cb58-43f6-90e1-a477a023bb71`, completed normally in
2,110,512 ms under explicit long_running policy. Worker reported one full-suite
243-test passing run in14.113s. Pi independently reran the exact predeclared
command:243tests, zero failures/errors, 8.744s unittest/12,602ms process, all204
baseline preserved. Benchmark `agy-evidence-42eabb37-03b0-4200-9a3a-07ea7a336004`.

Pi read complete2520source/2070test/262doc lines plus relevant accepted typed-ID
and reproducibility dependencies, and independently verified new literal settings
and composite hashes using static AST literals and independent JSON/SHA256 without
candidate import/execution. Fresh inspection
`agy-evidence-0265e833-0303-47c6-8aa6-358e1eebe660`: exactly3allowedfiles,
unauthorized[],Gitunchanged, 139protectedfiles/86,476,445bytes and130outside
sentinels. Previous correction draft/HEAD/index/status unchanged. Runtime artifacts
retained; fixture scratch empty. Monitoring is not an OS sandbox/complete action
audit; captured163action candidates remain untrusted navigation.

Final decision `agy-evidence-7f432a65-0de4-4111-a7de-334e7af5ef42`:
**ESCALATE**, acceptance_scope **task_only**, deployment **NOT_AUTHORIZED**.
No accepted code transferred to main, no automatic revision or new invocation.

Substantial fixes verified: literalUTF8 compatibility restored at both sites,
from_dict mixed-key sorting guarded, full structured10component RNG/worker/thread
recording schema implemented, isolated metadata and multi-category cases, retained-u2
history guards and actual/expected late-file zero-hash bounds added, list/tuple docs
and ordered exceptions corrected. These improvements are not whole-deliverable
acceptance.

Remaining mandatory gates, identified from actual code/tests (not Worker claims):
- RNG record missing/unknown key matrix, strict applicability/seed types and
  algorithm/derivation/reason corruption/sensitivity are incomplete. Positive-worker
  settings mutate several fields together rather than isolate each semantic field.
- from_dict malformed retained collections/nested types and falsified stored hashes
  are not covered by direct-constructor tampering tests; numeric-overflow coverage
  is absent. Provenance malformed untracked MEMBERS and forbidden paths in both
  validator roles remain absent. Most cases lack intended-guard diagnostics.
- Statically identified malformed-UTF8 public error: literal lone-surrogate labels
  can pass scalar/category string checks, then canonical JSON UTF8 encoding raises
  incidental UnicodeEncodeError. Preserve ensure_ascii=False; reject invalid scalar
  evidence or translate serialization errors clearly, rather than regress validUTF8.
  No unauthorized candidate probe was run to reach this conclusion.
- Worker report's 'strictly satisfying all checklist items'/unresolved[] claim is
  unsupported by actual mandatory coverage. Passing243tests cannot waive it.

Evidence lives in `data/reports/checkpoint-identity-repair-setup/`, including original
and approved command/spec, baseline, checksum-verified materialized report/full
benchmark stderr, independent literal preimages and preservation manifests. Prior
FINAL ESCALATEs, exhausted budgets and exactlythree checkpoint deliverables remain
unchanged. Payload validation and secureIO/integration remain blocked; no profiling,
training, GPU/Kaggle, heldout or deployment permission was created.
