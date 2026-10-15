# Deliverable 2 — Payload Validation: scope and acceptance proposal

## Status and authority

**Scope definition only. Implementation NOT_AUTHORIZED; not implemented or accepted.**
The user requested this definition and explicitly reserved implementation for separate
authorization. The detailed contracts below are Pi's proposed implementation scope,
not newly approved scientific settings or permission to execute them. Companion
record: `configs/checkpoint-payload-scope.json`.

This remains the second of **exactly three checkpoint deliverables**, not a collection
of new micro-tasks. Fixtures, helpers, diagnostics and corrections belong to this
whole deliverable. Historical ESCALATE decisions and exhausted budgets are preserved.
No Worker, worktree preparation, candidate execution or tests are authorized by this
planning record. No stage/commit/push.

## Accepted dependency — do not redo Deliverable 1

Use the existing public identity/provenance APIs and validation exception family,
without modifying or reimplementing their schemas, fingerprints or canonicalization.
The source artifact task is `8a86cfbb-437b-4873-bc45-1e44da1d4b04`, ACCEPT evidence
`agy-evidence-5a2fc59d-465d-4fa3-b53b-6b81628a5236`. Its separate main working-tree
integration is ACCEPT, with **249 passing tests**, recorded in
`data/reports/checkpoint-identity-main-integration/review.json` and
[the integration record](checkpoint-identity-main-integration.md). That decision
supersedes older progress notes' 204-test count, not their preserved historical results.

Frozen dependency SHA256s:

- `scripts/checkpoint_identity.py`: `6c91bab8229be89453e9b485f3f8df758bfa91e2611296b68c61c5cbe5bb5b2c`
- `tests/test_checkpoint_identity.py`: `72c874a40f8686e5ab1a914cb9427ec0f9470c127221960195cf8d465532c468`
- `docs/checkpoint-identity.md`: `c84d28069e5877154534c32e2f67ca40d36544c0bf3027ab6f4021d57dce0601`

Retain their tests unchanged as regression dependencies; this is not a new identity
implementation or request to reopen its accepted task. An actual dependency defect
would block the dependent work and require explicit escalation, not permission to
silently edit these files.

## Product boundary

Proposed implementation files, relative to Amazon4U:

1. `scripts/checkpoint_payload.py` — in-memory CPU payload/state validation.
2. `tests/test_checkpoint_payload.py` — complete positive/adversarial contract tests.
3. `docs/checkpoint-payload.md` — this specification, extended with truthful API,
   schema, limitations and verification documentation while preserving authorization
   and criterion history.

Input: an **already decoded**, versioned state mapping plus independently trusted
identity/provenance expectations, model/parameter descriptors and resolved runtime
constraints. Expectations are mandatory; deriving them from the submitted payload
itself does not establish compatibility. Output: a complete, independently owned
validated state snapshot eligible for a later authorized restoration consumer, or a
domain validation error. There must be no partially successful return.

The runtime path must be usable without a live model/optimizer or dataset. The
supported learned models are the accepted CPU LightGCN and Heterogeneous KG+GNN;
no new architecture, optimizer family or optional comparison is introduced. Model
schemas must match their actual accepted implementations, not a fixture-only subset.

### Required state and cross-field rules

- **Envelope:** one explicit payload schema version and epoch-boundary capability
  label; exact required/optional keys at every level. Include the existing canonical
  identity and provenance records and validate actual and expected roles through
  Deliverable 1. Bind method, graph/warm-ID row ordering, resolved settings and
  protocol/execution identifiers; counts alone are insufficient.
- **Model:** complete named parameter/buffer state for each supported model,
  including every heterogeneous relation/self matrix and bias. Match independently
  expected names, shapes, CPU devices, layouts and dtypes; reject nonfinite values,
  missing/extra entries, unsupported tensor subclasses and unsupported layouts.
  Tensor contents are state, not reconstructable from parameter counts. In particular,
  LightGCN registers `adj_user_item` as a **sparse COO buffer**: support and validate
  its CPU float32 values, integer index tensor, shape, canonical index order,
  uniqueness/ranges and normalized weights against independently trusted training-
  graph expectations. Do not drop this buffer, densify it, trust a coalesced flag
  alone or accept a same-shape/same-count different graph. Sparse backing-storage
  accounting includes both index and value storages. Other unsupported sparse
  layouts are rejected explicitly.
- **Adam:** complete group membership/order and unambiguous parameter-ID/name
  binding; exact resolved hyperparameters and supported framework options; validate
  step representation and counters, first/second moments and any supported optional
  state. Moment shape/dtype/device must agree with its parameter, values finite and
  second moments nonnegative. Reject duplicate/unknown IDs, group swaps, orphan
  state, incompatible options and missing required moment fields. Respect Adam's
  lazy initialization: empty initial state and a never-updated parameter can be
  valid; absence is not valid for a parameter declared already updated. Explicit
  per-parameter update metadata must agree with state and global progress. Do not
  falsely require every parameter to have received every global update. Zero-step
  initialized/uninitialized cases need explicit, consistent rules.
- **RNG:** represent every stream role declared by the accepted identity, including
  Python, NumPy, PyTorch CPU and applicable initialization/shuffle/negative/neighbor/
  loader/worker streams. Bind algorithms, applicability, seed/derivation, worker IDs
  and actual state representation to trusted runtime descriptors. Check structural
  and supported-engine validity, not just existence or tensor length. Inapplicable
  roles, including CUDA in this CPU profile, require explicit absence/reason; do not
  silently drop applicable state. Private throwaway RNG instances may validate state;
  never set or advance global/live generators. No arbitrary RNG draw is an oracle
  for exact continuation.
- **Sampler/progress:** validate algorithm/domain bindings, applicable state, any
  recorded order as a valid permutation of the declared permitted example domain,
  cursor and epoch/step/update/skip counters. Only explicitly quiescent epoch-boundary
  state is supported: no partial batch, active prefetch or in-flight worker state.
  Reject mid-epoch capability claims. No sampler execution or training is implemented.
- **Selection:** record completed validation checks (epochs and finite NDCG@10),
  best score/selected epoch and patience state. Recompute their internal consistency
  using the frozen strict-improvement, earliest-exact-tie and 10-check patience rules,
  with at most 100 epochs and the explicitly supplied synthetic validation schedule.
  Failed checks are not patience increments. Initial/no-check state must be explicit
  (no invented best epoch/score). Distinguish current resume weights from best-selected
  weights: after a successful check, include a separately epoch-bound best-model
  snapshot with the same complete model schema; before the first check it is absent.
  If current and selected epochs coincide, their model states must agree. This checks
  supplied history/state consistency, not proof that recorded scores were measured
  or that tensors genuinely originated in a claimed historical run.
- **Applicability:** scheduler and AMP scaler are explicitly absent in this CPU,
  constant-learning-rate profile. A claim that either was active must fail, not lose
  state silently. CUDA state, mid-epoch restoration and unsupported optimizer options
  are rejected explicitly rather than converted to this profile.

Document the exact field names/types, tensor profile, supported framework/RNG
representations, counter meanings and all initial/boundary cases in this same
module/test/doc deliverable. Those are implementation contracts, not permission to
choose unresolved scientific execution settings. Batch size, fanout, validation
frequency and other pending settings must be explicitly supplied as synthetic
fixture values; never install them as production defaults or amend frozen protocols.

## Acceptance criteria — all mandatory

**C1 — Scope and dependency preservation.** Only the three proposed implementation
files change during Worker implementation. The accepted identity/provenance files,
model/graph implementations, frozen protocols, histories, Git metadata and sibling
projects remain unchanged. No prohibited execution or acquisition occurs.

**C2 — Strict envelope and trusted binding.** Exact versioned schemas and explicit
applicability are enforced. Missing/extra fields, wrong types (including bool-as-int),
unsupported versions, changed identity/provenance/settings and missing mandatory
expectations fail closed. Both malformed actual and malformed expected inputs are
covered. Delegate identity/provenance validation to the existing accepted APIs. The
API must not manufacture expectations from the payload; independently trusted
acquisition remains a caller boundary, not something matching records can prove.

**C3 — Complete model/tensor validation.** Both accepted learned model families pass
complete positive controls; every parameter/buffer category has independently
mutated negative coverage for names/shapes/dtypes/layout/device/finiteness. LightGCN's
sparse adjacency buffer must match trusted graph contents, with independent invalid
index/order/duplicate/weight and same-count graph-mutation cases. Resource
preflight precedes cloning or materialization, counts full backing storage rather
than only visible tensor elements, and refuses oversized/unsupported storage.

**C4 — Complete Adam validation.** Positive initial, lazily initialized and synthetic
post-update mappings pass without any optimizer execution. State/group/name/order,
hyperparameter, moment and per-parameter/global-step inconsistencies independently
fail. No blanket empty-state acceptance and no false requirement for uniform updates.

**C5 — Complete RNG/sampler validation.** Every applicable role has real state
validation and independent negative coverage; missing/extra roles, malformed engine
states, inconsistent seeds/algorithms/worker IDs, invalid domain/order/cursors and
nonquiescent or mid-epoch states fail. Inapplicable state is explicit. Global RNG,
worker and sampler execution state remains untouched on success and failure.

**C6 — Progress and selection consistency.** Initial, first-check, improvement,
exact-tie, non-improvement, patience-boundary and terminal cases are coherent.
Nonfinite/out-of-range scores, future or unscheduled selected epochs, inconsistent
history/counters and current/best state confusion fail. Fix fixtures, never weaken
selection guards to make them pass. No evaluation/selection engine or test-set input.

**C7 — No mutation or partial restoration.** No model/optimizer load, live/global RNG
restoration, gradient computation or optimizer step occurs. Success returns owned
container/tensor copies; later mutations of inputs, expectations or returned data
cannot silently alter the other objects. Failure leaves inputs and live/global state
unchanged, including late failures. Do not describe mutable tensors/containers or
caller-supplied evidence as immutable, unforgeable or proof of origin.

**C8 — Fail-closed error/resource behavior.** Malformed supported input types produce
the documented domain exception, not native formatting/conversion/shape errors.
Diagnostics identify the field/role and remain bounded for oversized integers,
invalid Unicode and pathological representations. Do not change interpreter integer
limits or introduce an arbitrary integer validity cap. Reject custom objects, cycles
and unsupported structures without executing caller hooks. Document the payload
content/storage accounting and reject excess before copying; no claim of a total
process-memory sandbox or safe decoder follows from validating an existing mapping.

**C9 — Meaningful complete regression.** Preserve all 249 current baseline tests and
all accepted identity tests unchanged. New tests use valid full controls, isolate each
mutation, assert the intended guard/domain error and cover both supported models and
both actual/expected roles where applicable. The prospectively pinned full unittest
suite must complete within 120 seconds, exit zero, with no failures/errors/skips or
removed assertions. Pi independently reads complete results and implementation,
callers, relevant framework dependencies and tests; green tests alone are insufficient.

**C10 — Evidence and truthful completion.** Complete schema/API/limits documentation,
Worker report and independently verified criterion-by-criterion review are required.
Disclose every known limitation/issue; Pi must assess all current limitations in exact
order/text against original scope, without waiving an assigned requirement. Any
unresolved in-scope issue or incomplete mandatory evidence blocks ACCEPT. Acceptance
is task_only; deployment NOT_AUTHORIZED. Do not claim exact continuation, complete
checkpoint security or full harness completion.

## Proposed verification and execution limits — not execution authorization

- One clean pinned isolated repository/project root, containing the exact accepted
  dependency as explicitly monitored read-only input. Preserve main's uncommitted
  work; no automatic commit to manufacture a clean dependency baseline. Pin and
  approve the setup method before delegation; if protection/cleanliness is unresolved,
  stop rather than launch a Worker.
- Baseline and candidate use one prospectively declared synchronous full unittest
  discovery command, absolute Python path, argv, cwd and fresh scratch path, unchanged
  across Worker/revisions/Pi verification. Pin these and a 120-second ceiling at
  submission after setup approval; no candidate execution occurs during this scope
  definition. No subsets, smoke imports, compileall, detached jobs or ad-hoc probes.
- At most 8 users, 12 products, 8 metadata nodes/type and 64 dimensions. Small CPU
  tensors, synthetic Adam mappings and private CPU RNG-state fixtures are proposed;
  **zero optimizer steps/backward passes/training**. Nonallocating meta-device tensors
  are proposed solely for wrong-device rejection tests; no CUDA initialization or
  execution. Existing baseline CPU tests remain unchanged.
- Retain the 2MiB fixture payload ceiling. Before submission, pin the exact content
  accounting (metadata/source bytes plus full distinct tensor backing storage),
  structural bounds and narrowly bounded just-over-limit negative fixtures. This
  ceiling is not Python heap usage; the separate serialized-file/reconstructed-storage
  decoding gate belongs to Deliverable 3. Boundary accounting must be explicit in the
  authorized specification, not a later Worker-selected shortcut.
- Declare only fresh test-generated paths/temp directories; no pre-existing artifact
  or source writes. Set `training_allowed=false`, disable bytecode and require scratch
  cleanup. Predeclare recursive-sentinel monitoring and protected ignored dependencies/
  historical references with 128MiB/file, 512MiB/aggregate, 60s/file, 300s/snapshot
  read-only hashing budgets. These do not enlarge the 2MiB payload fixture ceiling.
- One whole-deliverable Worker invocation and up to two consolidated corrective
  revisions after separate implementation authorization. Original criteria and
  benchmark commands remain fixed. No automatic replacement task or budget reset.
  Pi retains acceptance authority and all inspection/evidence/verification gates.

## Explicit exclusions and next authorization gate

Deliverable 3 owns checkpoint serialization/deserialization, `weights_only=True`
loading, pre-load file bounds, reconstructed storage limits during decoding,
allowlisted physical source acquisition, approved-root/descriptor traversal,
symlink/parent-race defense, atomic exclusive no-clobber publication, cleanup and
end-to-end serialization/validation-only integration. Deliverable 2 must not call
`torch.load`/`torch.save`, pickle/eval/shell, capture source/environment from the host,
or implement checkpoint filesystem APIs. An already decoded payload is not evidence
that decoding was safe or its physical provenance was proved.

No live restoration, continuation oracle, training loop, profiling, real datasets,
GPU/Kaggle, held-out evaluation, deployment or main transfer is authorized here.
Implementation approval must explicitly accept this whole scope, settle the listed
submission-time accounting/setup details, and predeclare exact commands/paths/budgets
before any Worker starts. Later main integration needs its own authorization.
There is no deployment policy; deployment/generalization remains NOT_AUTHORIZED.

## Recommended execution specification — awaiting separate authorization

The user asked for Pi's recommendation, not implementation. The following selects
concrete proposed setup, command and accounting rules; it authorizes none of their
execution. Existing C1–C10 remain unchanged.

### Setup and command

- Prepare a fresh local full clone at
  `/home/sunf/FSB/Final/Amazon4U/.worktrees/checkpoint-payload-review`, pinned to the
  currently observed source HEAD `f145e93abea96db7fb429e546c7c097d9dc78e86`.
  The accepted identity source/tests/doc and these planning records are now tracked
  at that HEAD; no commit, staging, dependency rewrite or ignore-rule workaround is
  needed. Verify their accepted hashes and a clean clone before the baseline. If
  that path becomes occupied or the pinned inputs disagree, stop instead of
  overwriting/reusing a checkout or silently selecting another HEAD.
- Pin the project/cwd to that clone's `Amazon4U/` directory. Only the three proposed
  implementation files are Worker-writable. All existing dependency/protocol/history
  files and Git state remain protected. No main-tree transfer is included.
- Use the existing main Python executable
  `/home/sunf/FSB/Final/Amazon4U/.venv/bin/python`, `-B`, and the exact `-c` script
  recorded in the companion JSON's `execution_recommendation.benchmark`.
  It sets both `TMPDIR` and `tempfile.tempdir`, disables bytecode, discovers all
  `tests`, runs with verbosity 2 and exits nonzero on failure. Pin cwd and a 120s
  timeout. Use exactly this command for the isolated baseline, Worker, corrective
  revisions and independent Pi benchmark; no additional candidate commands.
- The proposed sole fresh generated/temp subtree is Git-root-relative
  `Amazon4U/.test-output/checkpoint-payload-v1/`, inside the clone. Require it empty
  before and after each run. Benchmark logs/receipts are Pi-controlled evidence,
  not permission for Worker writes to other locations. Baseline must reproduce
  all 249 accepted tests before delegation. No setup or baseline has run here.

### Exact proposed 2MiB accounting

Use a **logical payload-content budget**, not pickle size or claimed process RAM:
`B = 2,097,152 bytes`. Reject unless BOTH of these are at most B:

1. metadata charge + incoming distinct full backing-storage bytes;
2. metadata charge + projected storage bytes of the owned output copies.

For incoming CPU tensors count the entire underlying storage, once per distinct
storage across the payload, not just `numel * element_size` of a visible view.
Sparse COO tensors contribute index and value storages, without densification.
For projected compact output copies count each tensor occurrence independently;
use dense logical element bytes, or sparse index/value logical element bytes.
This prevents shared input storage from disguising an oversized copied result.
Check before cloning/materializing. Reject unsupported devices/layouts first.

Define metadata charge by a type-directed traversal of the declared payload:

| Value | Charge in bytes |
|---|---|
| None / bool | 1 |
| exact integer | 1 sign byte + max(1, ceil(abs(integer).bit_length()/8)) |
| finite float | 8 |
| Unicode string | 8 + length of strict UTF8 encoding |
| bytes | 8 + byte length |
| dict | 8 + charges of every key and value |
| list / tuple | 8 + charges of every element |
| tensor | charge of its descriptor tuple, plus separately accounted storage |

A dense tensor descriptor is `(dtype_name, layout_name, device_name, shape_tuple,
stride_tuple, storage_offset)`. A sparse COO descriptor is `(dtype_name,
layout_name, device_name, shape_tuple, is_coalesced)` plus descriptors of its index
and value tensors. Names use the pinned framework's string representations; shape,
stride and offset components are exact integers. Descriptor tuple/element charges
use the same table. This is accounting, not a new identity hash or serialization.
Metadata aliases are charged at each occurrence; cycles/custom objects are rejected.
Traverse only allowed primitive containers and supported tensors; count safely and
stop at excess without first building an unbounded encoding or diagnostic.

Additional proposed structural limits: root container depth 1, maximum container
nesting depth **32**, and **16,384 visited values** (including keys and tensor
component descriptors). Tensor descriptors and their elements count under the same
traversal rules; tensor elements themselves are storage, not individual metadata
nodes. Repeated references count as repeated visits. These are technical bounds
for this small CPU profile, not new scientific settings or integer validity caps.

Boundary tests must cover exactly B and just over B, oversized backing storage
behind a tiny view, shared-storage output expansion, sparse index/value accounting,
and structural limits. Allow one deliberately oversized negative fixture at a time,
with combined accounted input no greater than B + **4096 bytes**; it must be rejected
before output allocation. This narrow exception is only for quota-rejection tests.

Embedded source/provenance bytes count in this payload budget. A record valid under
Deliverable 1's independent source-byte limit need not fit inside a complete 2MiB
payload with tensors and other metadata. This adds no change to Deliverable 1.
Passing this accounting does not promise a serialized file will fit within 2MiB:
Deliverable 3 must separately enforce its actual file and safe-decoding limits.
The budget also does not count every interpreter/framework temporary allocation or
establish an OS/process-memory sandbox.

If authorized, Pi first validates setup and the 249-test baseline, then submits the
whole deliverable with this fixed specification and at most two corrective revisions.
Any material specification change requires approval before delegation; no scope
changes, extra tasks or execution are inferred from this recommendation.

## User-agreed clarification — fixture budget versus runtime policy

The user approved Pi's recommendation to separate the synthetic verification budget
from runtime validation limits and requested that agreement be recorded. This is a
scope/policy agreement only, **not implementation or execution authorization**.

1. **Synthetic verification retains the 2MiB budget.** It bounds small correctness
   fixtures; it was inherited from `checkpoint-recovery-plan.md:157` and
   `checkpoint-three-part-recovery.md:74,100–101`, not derived from production model
   size or hardware capacity. Narrow oversized rejection fixtures remain subject to
   the separately specified test policy.
2. **Runtime validation has no universal hardcoded 2MiB architectural ceiling.**
   It must use an explicit, configurable, independently trusted resource policy,
   sized for the intended model and available memory. Assess current model weights,
   Adam state, selected-best weights and validation copies, not only serialized file
   size. Include metadata/provenance, complete backing storage and sparse index/value
   storage in the applicable accounting.
3. **Limits are selected and pinned before execution.** An untrusted payload cannot
   choose or enlarge them. Do not silently increase limits, choose an unlimited
   fallback, or automatically treat detected free RAM as allocation permission.
   Larger profiles require explicit authorization and verification before use.
4. **Input and output checks remain required under that policy.** Retain the safety
   objective of checking incoming full backing storage and projected owned output
   copies before allocation; the runtime threshold comes from the trusted profile,
   not a permanently embedded 2,097,152-byte constant.
5. **Deliverable 3 must enforce corresponding actual file/decoding limits.** This
   agreement does not itself enlarge its historical file limit or authorize larger
   file loading. A profile that permits larger runtime state needs aligned, separately
   authorized and verified IO/decoding limits before end-to-end use.

This clarification **supersedes the earlier recommendation insofar as its 2MiB
rejection rule could be read as a universal runtime limit**. The earlier formula and
numeric structural bounds remain preserved as a proposed small-fixture profile,
not newly approved production defaults. Exact runtime policy fields, selected larger
limits (none selected here) and detailed accounting still require specification
before delegation. Original C1–C10 remain unchanged and apply to the approved profile;
this clarification waives none of their assigned safety requirements.

No identity/provenance redo, source/schema change, broader device/model support,
hardware-capacity verification, Worker launch, tests, training, main transfer or
deployment is authorized. Exactly three checkpoint deliverables remain; Deliverable2
is still unimplemented and requires separate implementation authorization.
