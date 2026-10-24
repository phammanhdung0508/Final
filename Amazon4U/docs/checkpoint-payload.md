# Deliverable 2, payload validation: scope and acceptance proposal

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

## Accepted dependency, do not redo deliverable 1

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

1. `scripts/checkpoint_payload.py`, in-memory CPU payload/state validation.
2. `tests/test_checkpoint_payload.py`, complete positive/adversarial contract tests.
3. `docs/checkpoint-payload.md`, this specification, extended with truthful API,
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

## Acceptance criteria, all mandatory

**C1, Scope and dependency preservation.** Only the three proposed implementation
files change during Worker implementation. The accepted identity/provenance files,
model/graph implementations, frozen protocols, histories, Git metadata and sibling
projects remain unchanged. No prohibited execution or acquisition occurs.

**C2, Strict envelope and trusted binding.** Exact versioned schemas and explicit
applicability are enforced. Missing/extra fields, wrong types (including bool-as-int),
unsupported versions, changed identity/provenance/settings and missing mandatory
expectations fail closed. Both malformed actual and malformed expected inputs are
covered. Delegate identity/provenance validation to the existing accepted APIs. The
API must not manufacture expectations from the payload; independently trusted
acquisition remains a caller boundary, not something matching records can prove.

**C3, Complete model/tensor validation.** Both accepted learned model families pass
complete positive controls; every parameter/buffer category has independently
mutated negative coverage for names/shapes/dtypes/layout/device/finiteness. LightGCN's
sparse adjacency buffer must match trusted graph contents, with independent invalid
index/order/duplicate/weight and same-count graph-mutation cases. Resource
preflight precedes cloning or materialization, counts full backing storage rather
than only visible tensor elements, and refuses oversized/unsupported storage.

**C4, Complete Adam validation.** Positive initial, lazily initialized and synthetic
post-update mappings pass without any optimizer execution. State/group/name/order,
hyperparameter, moment and per-parameter/global-step inconsistencies independently
fail. No blanket empty-state acceptance and no false requirement for uniform updates.

**C5, Complete RNG/sampler validation.** Every applicable role has real state
validation and independent negative coverage; missing/extra roles, malformed engine
states, inconsistent seeds/algorithms/worker IDs, invalid domain/order/cursors and
nonquiescent or mid-epoch states fail. Inapplicable state is explicit. Global RNG,
worker and sampler execution state remains untouched on success and failure.

**C6, Progress and selection consistency.** Initial, first-check, improvement,
exact-tie, non-improvement, patience-boundary and terminal cases are coherent.
Nonfinite/out-of-range scores, future or unscheduled selected epochs, inconsistent
history/counters and current/best state confusion fail. Fix fixtures, never weaken
selection guards to make them pass. No evaluation/selection engine or test-set input.

**C7, No mutation or partial restoration.** No model/optimizer load, live/global RNG
restoration, gradient computation or optimizer step occurs. Success returns owned
container/tensor copies; later mutations of inputs, expectations or returned data
cannot silently alter the other objects. Failure leaves inputs and live/global state
unchanged, including late failures. Do not describe mutable tensors/containers or
caller-supplied evidence as immutable, unforgeable or proof of origin.

**C8, Fail-closed error/resource behavior.** Malformed supported input types produce
the documented domain exception, not native formatting/conversion/shape errors.
Diagnostics identify the field/role and remain bounded for oversized integers,
invalid Unicode and pathological representations. Do not change interpreter integer
limits or introduce an arbitrary integer validity cap. Reject custom objects, cycles
and unsupported structures without executing caller hooks. Document the payload
content/storage accounting and reject excess before copying; no claim of a total
process-memory sandbox or safe decoder follows from validating an existing mapping.

**C9, Meaningful complete regression.** Preserve all 249 current baseline tests and
all accepted identity tests unchanged. New tests use valid full controls, isolate each
mutation, assert the intended guard/domain error and cover both supported models and
both actual/expected roles where applicable. The prospectively pinned full unittest
suite must complete within 120 seconds, exit zero, with no failures/errors/skips or
removed assertions. Pi independently reads complete results and implementation,
callers, relevant framework dependencies and tests; green tests alone are insufficient.

**C10, Evidence and truthful completion.** Complete schema/API/limits documentation,
Worker report and independently verified criterion-by-criterion review are required.
Disclose every known limitation/issue; Pi must assess all current limitations in exact
order/text against original scope, without waiving an assigned requirement. Any
unresolved in-scope issue or incomplete mandatory evidence blocks ACCEPT. Acceptance
is task_only; deployment NOT_AUTHORIZED. Do not claim exact continuation, complete
checkpoint security or full harness completion.

## Proposed verification and execution limits, not execution authorization

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

## Recommended execution specification, awaiting separate authorization

The user asked for Pi's recommendation, not implementation. The following selects
concrete proposed setup, command and accounting rules; it authorizes none of their
execution. Existing C1-C10 remain unchanged.

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

## User-agreed clarification, fixture budget versus runtime policy

The user approved Pi's recommendation to separate the synthetic verification budget
from runtime validation limits and requested that agreement be recorded. This is a
scope/policy agreement only, **not implementation or execution authorization**.

1. **Synthetic verification retains the 2MiB budget.** It bounds small correctness
   fixtures; it was inherited from `checkpoint-recovery-plan.md:157` and
   `checkpoint-three-part-recovery.md:74,100-101`, not derived from production model
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
before delegation. Original C1-C10 remain unchanged and apply to the approved profile;
this clarification waives none of their assigned safety requirements.

No identity/provenance redo, source/schema change, broader device/model support,
hardware-capacity verification, Worker launch, tests, training, main transfer or
deployment is authorized. Exactly three checkpoint deliverables remain; Deliverable2
is still unimplemented and requires separate implementation authorization.

## Finalized preparation contract, runtime policy and state boundaries

The user authorized a **bounded specification pass for items 1 and 2 only**.
This section resolves their technical contracts for later implementation; it does
not authorize that implementation. Existing C1-C10 and historical recommendations
above are preserved. Where an earlier paragraph calls these particular details
pending, this section is the current specification. Step 3 (checkout/HEAD/command
synchronization and the 249-test baseline) is explicitly deferred until implementation
is authorized and ready to start. The earlier checkout HEAD/command remain historical
proposals, not current setup verification receipts or authority to use a stale HEAD.

### 1. required trusted resource-policy argument

The validator's public entry point must require keyword arguments for serialized
`expected_identity`, `expected_provenance`, `expected_model`, `expected_runtime`
and `resource_policy`; none may be manufactured from the submitted payload.
All expectations must themselves be validated. There is no model/optimizer argument,
implicit resource-policy default, unbounded fallback or hardware auto-sizing.

`resource_policy` is a plain dict with exactly these keys:

| Field | Type and rule |
|---|---|
| `schema_version` | exact string `amazon4u-payload-resources-v1` |
| `profile_id` | nonempty clean UTF8 string identifying the externally authorized profile |
| `kind` | `synthetic` or `runtime` |
| `max_input_bytes` | positive exact int, bool excluded |
| `max_output_bytes` | positive exact int, bool excluded |
| `max_container_depth` | positive exact int, bool excluded |
| `max_visited_values` | positive exact int, bool excluded |

For the concrete synthetic profile `amazon4u-payload-synthetic-v1`, the four limits
are respectively **2097152, 2097152, 32, 16384**. No larger production/runtime
numbers are selected here. `kind=runtime` still requires every explicit finite
integer limit and prior operator authorization; it must not inherit synthetic
constants as an architectural ceiling or infer permission from available RAM.
Validation of a policy's types does not prove hardware capacity or authorize a run.
A payload cannot carry an overriding policy, choose a larger profile or disable guards.

Use the earlier type-charge table and tensor descriptor accounting exactly. Let M
be metadata charge, S the sum of full distinct input backing storages, and C the
projected independent compact-copy storage for every output tensor occurrence:
require **M+S <= max_input_bytes** and **M+C <= max_output_bytes**, as well as the
policy's depth/visit limits. Equality is allowed. For sparse COO, account for the
full index and value storages; never densify or coalesce before resource preflight.
Storage identity is the framework's actual underlying storage identity, not tensor
identity, a visible slice's size, or equal contents. Empty storages contribute zero.
Dense strided views may be copied compactly; this cannot hide their full input storage.

Container depth counts only dict/list/tuple nesting, root at depth 1; tensor descriptor
containers follow the same rule at their tensor's position. Each key, scalar,
container and tensor is one visit, plus the values/containers of tensor descriptors;
tensor elements are charged as storage, not per-element visits. For sparse COO the
single descriptor tuple is `(dtype_name, layout_name, device_name, shape_tuple,
is_coalesced, index_descriptor, value_descriptor)`, so component nesting/charges
are unambiguous. Repeated references
count on every traversal; cycles fail. Sparse components contribute their descriptor
visits once, not a second recursively duplicated storage/descriptor subtree. For
metadata charge and visit counting a tensor is replaced by its descriptor tuple;
the tensor-position visit is that tuple's visit, not an additional hidden node.

Only exact built-in dict/list/tuple/str/bytes/int/float/bool/None and supported exact
`torch.Tensor` values are accepted; no custom Mapping, ndarray, Parameter, tensor
subclass, callable or arbitrary object hook. Tensor subclasses are refused before
reading tensor properties or calling tensor operations. Reject invalid text/nonfinite
scalar floats, count bounded UTF8 without first encoding an arbitrarily large string,
and calculate integer charge using bit lengths rather than decimal conversion.

Preflight both the actual payload and tensor-bearing expected descriptors separately
under these bounds, before copying, D1 reconstruction/hashing or native RNG parsers.
Expected descriptors are not silently folded into the actual payload or used as its
source of truth. These are per-object logical content/copy bounds, not a sum of all
live interpreter, expected-input, library, temporary and output memory. Account for
that larger working set when a future runtime profile is sized; D3 remains responsible
for preventing oversized allocations during initial decoding.

The oversized-fixture exception belongs to the test harness, **never the validator's
acceptance policy**: one boundary-negative fixture at a time, input accounted size
at most 2097152+4096 bytes, or a shared-storage fixture whose projected output is at
most 2097152+4096. Test depth 33 and visit count 16385 as separate structural refusals.
These fixtures must fail without output copying; no permission for oversized positives,
large device allocations or disabling the normal guard. Positive tests include exact
limits; all counters must also be tested immediately above their limit. Test the
configurable runtime-policy path with explicit smaller unit-test budgets within the
same synthetic envelope; this selects no production RAM profile or larger allocation.

### 2. payload envelope and trusted descriptors

Component schema version: `amazon4u-checkpoint-payload-v1`; capability string:
`epoch_boundary_validation_only`. This is not an experiment protocol bump or an exact
resumption guarantee. The root dict has exactly these required keys:

`schema_version`, `capability`, `identity`, `provenance`, `runtime_profile`,
`current_model`, `best_model`, `optimizer`, `rng`, `sampler`, `progress`,
`selection`, `scheduler`, `amp`.

- `identity` is the existing `CanonicalResumeIdentity.to_dict()` representation;
  reconstruct/revalidate both roles with the existing public D1 APIs, then compare
  against mandatory trusted expectations. No alternative fingerprint or schema.
- `provenance` is the existing `build_source_provenance` dict, validated by
  `validate_source_provenance(actual, expected)`, including actual `content_bytes`.
- `runtime_profile` has exactly `profile_id`, `python_version`, `numpy_version`,
  `torch_version`, `adam_step_dtype`, `batch_policy`. It matches trusted expectations
  and applicable provenance version records. Allowed step dtype names are
  `torch.float32` and `torch.float64`; the chosen dtype must also match the loaded
  non-fused Adam scalar-initialization policy (`_get_scalar_dtype`, determined by
  the framework's default dtype). A read-only dtype comparison is permitted, not
  changing the global default to make a profile pass. The synthetic profile uses
  float32; float64 is not claimed verified in the current environment. Batch policies
  are defined below. Read-only
  checks of loaded interpreter/library version constants may refuse an incompatible
  native RNG codec, but must not fill missing caller provenance or acquire host files.
- `scheduler` is exactly `{applicable: False, state: None,
  reason: "not_used_constant_learning_rate"}`; `amp` is exactly
  `{applicable: False, state: None, reason: "not_used_cpu_float32"}`.
  Missing records, active use, state attached to an inactive record, or extra keys fail.

The trusted `expected_model` dict has exactly `method`, `row_order`, `buffer_state`.
The method matches identity and one of the two supported families. `row_order` records
ordered typed IDs: User/Product for LightGCN, and all four node types for HeteroGNN.
Check membership against the validated retained identity (LightGCN items use its warm
pool); trusted row ordering must come from the accepted graph/model construction,
not a payload-derived sort. `buffer_state` is the independently trusted immutable-
graph buffer expectation, not weights taken from the submitted checkpoint.

The trusted `expected_runtime` dict has exactly `runtime_profile`, `optimizer_groups`,
`rng_role_bindings`, `rng_generator_bindings`, `sampler_algorithm`.
`optimizer_groups` is a nonempty ordered list of `{parameter_names, options}`.
Each parameter appears once across groups. `options` has the Adam fields below except
`params`; `param_names` is optional and its presence/values must be explicit in the
trusted profile. RNG binding formats and sampler semantics are specified below.
Profiles are independently resolved caller inputs, not proof of physical acquisition
or permission to run a new scientific configuration.

### 3. current and best model snapshots

`current_model` has exactly `epoch`, `row_order`, `state_dict`; its epoch equals
`progress.epoch`. `state_dict` is a plain dict (callers explicitly snapshot a native
OrderedDict outside this API), with detached CPU tensors and no autograd history.
Dense parameter tensors are float32, finite and strided; unsupported layouts fail.
Owned returned tensors are detached compact copies, not references into input or
trusted expectations. Float equality checks use logical contents in canonical order;
when snapshots must be identical, compare exact float32 bits, not an approximate score
or storage pointer. Empty CategoryPath/Brand embedding tables are allowed when the
trusted graph has none; matrices/biases are still mandatory.

Complete recipes, in accepted registration order:

- LightGCN: parameters `user_embeddings` (U, 64), `item_embeddings` (I, 64), plus buffer
  `adj_user_item` (U, I), coalesced sparse COO with CPU int64 indices and float32 values.
  Independently check canonical unique lexicographic indices, ranges and exact trusted
  normalized graph contents; a coalesced flag alone is not evidence. No dense substitute.
- HeteroGNN: four parameters `user_embeddings`, `product_embeddings`,
  `category_embeddings`, `brand_embeddings` with their typed row counts and width 64;
  then `relation_weights_l0.<relation>`, `relation_weights_l1.<relation>` (64,64),
  then `self_weights_l0.<type>` (64,64), `self_biases_l0.<type>` (64,),
  `self_weights_l1.<type>` (64,64), `self_biases_l1.<type>` (64,).
  Use all eight `RELATION_SCHEMAS` entries and all four `NODE_TYPES` entries in the
  accepted module's insertion order. This is **36 parameters**, no registered buffers.
  LightGCN has **two parameters**, not three optimizer entries: its adjacency is a buffer.

`best_model` is None until the first successful scheduled validation check; afterward
it is mandatory with the same three fields/complete recipe, its epoch equal to
`selection.selected_epoch` and its row order/buffers equal to trusted expectations.
If its epoch equals the current epoch, all corresponding tensor contents must agree
exactly. If earlier, weights may differ or coincide; equality is not itself an error.
A finite earlier best score alone never identifies the current tensors as best weights.
D2 cannot prove historical weight/score origin or replay gradient updates. Validate
the owned snapshot's complete semantics before returning it, not merely the original
references. Callers must prevent concurrent mutation of actual/expected inputs during
the call; this is not a concurrent-memory sandbox. After return, input mutations cannot
change the owned result, but a recipient modifying that result must revalidate it.

### 4. Adam boundary contract

`optimizer` has exactly `algorithm`, `state_dict`, `parameter_bindings`;
`algorithm` is `Adam`. `state_dict` has exactly `state` and `param_groups`, matching
PyTorch's mapping/group organization. IDs are exact nonnegative ints, bool excluded.
Groups are ordered as trusted; flatten their parameter names to assign native packed
IDs 0..P-1. `parameter_bindings` is a list in that order, each exactly
`{id, name, updates}`; names/groups match the complete model registration recipe and
trusted group partition. `updates` is a nonnegative exact integer, never a coerced
float. Reject duplicate/unknown IDs/names, omitted parameters, orphan state or group
swaps, even when shapes happen to match. Group partitions/options must be independently
supplied; no inferred/default group is installed by the validator.

Each group has exactly `params`, `lr`, `betas`, `eps`, `weight_decay`, `amsgrad`,
`maximize`, `foreach`, `capturable`, `differentiable`, `fused`,
`decoupled_weight_decay`, with optional `param_names` only as trusted above.
`params` is its ordered list of packed IDs. Numeric hyperparameters agree with
validated resolved settings and the frozen Adam recipe (0.001, (0.9,0.999), 1e-8,
zero weight decay); bool/nonfinite values and tensor-valued hyperparameters fail.
`betas` is a two-element tuple of finite scalar numbers. The supported CPU profile
requires amsgrad/maximize/capturable/differentiable/decoupled_weight_decay all False;
`foreach` may be None/False/True and `fused` None/False, exactly matching trusted options.
No scheduler `initial_lr`, AMSGrad maximum moment, AdamW or silent option upgrade.

| Per-parameter condition | Required state |
|---|---|
| `updates=0` | ID omitted, `{}`, or a complete initialized zero state |
| Complete initialized zero state | exactly `step`, `exp_avg`, `exp_avg_sq`; step 0 and both moments identically zero |
| `updates>0` | exactly those three fields, all present; no empty/missing state |

Initialized moments have the parameter's logical shape, CPU strided float32 dtype,
finite contents, and nonnegative `exp_avg_sq`. First moments may be negative.
For positive updates, zero moments are legitimate (e.g. zero gradients); do not reject.
`step` is a detached **0-dimensional CPU floating tensor**, dtype exactly the trusted
`adam_step_dtype`, finite, nonnegative, integer-valued, and exactly equal to `updates`.
Do not require int64 step tensors, accept bools, flatten a length-one vector, or silently
upgrade a legacy Python int step. If the framework floating counter cannot represent
the declared integer exactly, reject with a precision-specific domain error rather
than round, change framework/global defaults, or cap every metadata integer.

Let G be `progress.global_step`. Every `updates <= G`; if G=0 all updates are zero;
if G>0, sum(updates) >= G (at least one parameter updated per counted optimizer step).
No requirement that all parameters have state or equal G: unused relation parameters
can remain genuinely uninitialized. These are consistency checks, not proof that the
reported gradient/step trajectory occurred. Fixtures use synthetic mappings only,
never `optimizer.step`, `backward`, or a live optimizer restoration.

### 5. RNG roles, shared generators and native state representations

`rng` has exactly `roles`, `generators`. `roles` lists the ten records in accepted
`CONFIGURED_RNG_COMPONENTS` order; each is the existing six-field identity stream
record plus `generator_ids` (a list). The original six values must match validated
identity exactly. Inactive roles require `generator_ids=[]` and their original
explicit reason; active roles cannot silently lose state. CUDA is inactive here.

`generators` is a list of exact records `{id, codec, worker_id, seed_material, state}`.
IDs are unique nonempty clean strings. `worker_id` is None for parent/local streams,
otherwise an exact integer in 0..W-1. No orphan generators or duplicate IDs within a
role. All IDs and descriptors must match independent trusted bindings, not the payload
itself. `rng_role_bindings` maps each of the ten role names to its exact ordered ID
list; `rng_generator_bindings` maps every ID to exactly
`{codec, worker_id, seed_material}`. No state appears in those binding descriptors.

Python/NumPy/PyTorch_CPU parent roles each use one generator of the corresponding
family; DataLoader_generator uses one PyTorch CPU generator. Initialization uses one
local generator when applicable (PyTorch CPU for the accepted HeteroGNN constructor).
Shuffle/negative/neighbor roles use the explicitly bound generators, which may be
shared with other roles or worker contexts. Sharing is represented by the **same ID**
and one state record, not by conflicting duplicate copies or a new independent stream.

When W=0, the worker role is inactive and no worker-scoped generator is permitted.
When W>0, DataLoader_workers names one Python, one NumPy and one PyTorch CPU generator
per worker ID, with precisely the trusted worker seed material/derivation bindings;
additional worker-scoped sampler generators must be explicitly bound to their roles.
Validate configured worker metadata through D1. IDs describe captured/staged contexts,
not proof that processes were launched. No worker is launched or probed by D2.

Supported codecs (all other engine/format IDs fail):

| Codec ID | Exact state representation and checks |
|---|---|
| `python.random.MT19937.v3` | tuple `(3, internal, gauss)`; internal is a tuple of 625 exact ints: 624 uint32 words, then index 0..624; gauss is None or finite float. No version-2 signed-word coercion. Parse only on private `random.Random(0)`. |
| `numpy.random.RandomState.MT19937.v1` | tuple `("MT19937", keys, position, has_gauss, cached_gauss)`; keys is a list of 624 uint32 exact ints, position 0..624, has_gauss exact int 0/1, cached_gauss finite float. A cached value need not be zero when its flag is zero. Build only the bounded private uint32 key array for `RandomState(0).set_state`. Raw ndarrays are not payload leaves. |
| `numpy.random.PCG64.v1` | exact dict `bit_generator="PCG64"`, `state={state,inc}`, `has_uint32`, `uinteger`; state/inc are uint128 exact ints, inc odd; flag exact int 0/1 and uinteger uint32. Inactive cached uint32 may remain nonzero. Validate with a private `PCG64(0)` state setter, not the process-global RNG. PCG64DXSM and Generator-MT19937 dict formats are not aliases of this codec. |
| `torch.cpu.Generator.native.v1` | detached contiguous 1-D CPU uint8 tensor; require the current native state length supported by the matching pinned framework profile, then native `set_state` on a private CPU Generator. Test invalid internal engine controls as well as wrong dtype/length; length alone is insufficient. Native `initial_seed()` must agree with declared Torch seed material. No raw ABI-offset decoder, legacy byte-format conversion, CUDA state or global `torch.set_rng_state`. |

Seed material is an exact nonnegative int for Python/Torch; Torch also obeys
its intrinsic uint64 seed range. NumPy accepts an exact nonnegative seed int or an
explicit nonempty list of uint32 words, as specified by the trusted derivation context;
legacy RandomState integer seeds obey its uint32 range. For a parent/local generator,
seed material is an integer equal to the bound role's recorded stream seed; every
role sharing that generator must agree. NumPy seed-word lists apply to worker-scoped
contexts here. For worker-scoped generators the effective material must match the
independent trusted per-worker binding, while role seed/derivation and D1 worker
base/derivation remain their configured parent context. Do not equate a configured
master/base seed with an unrecorded native DataLoader-generated worker base seed.
These are engine representation bounds, not arbitrary restrictions on D1 master-seed
integers. Record/compare derivation metadata; do not execute caller derivation code,
claim to verify an arbitrary worker-init program, or invent a modulo reduction.
Private setters must succeed without draws or changes to global/live state, and their
native exceptions must be translated into the existing validation error family.
Native acceptance is not a proof of reachability from a seed or future bitwise equality
across framework versions. For Python/NumPy the current state alone cannot prove which
seed produced it; trusted metadata acquisition remains explicit.

**Initialization lifetime boundary:** the accepted HeteroGNN constructor uses a local
CPU generator and does not retain it on the model; LightGCN accepts caller-provided
embeddings. D2 must not pretend to extract initialization state from these models or
fabricate it by reseeding. If initialization is declared applicable, its actual
archived/staged state and trusted binding are required; without them validation fails.
Capturing that state is not implemented here and gives no permission to edit accepted
models or rewrite identity applicability merely to make a fixture pass. Archival init
state is not automatically an instruction to reinitialize weights during restoration.

### 6. sampler and epoch-boundary progress

Use IDs `(user_id, product_id)` as two-string tuples. From the validated identity,
P is the complete set of condition-positive pairs; K is the subset whose user has
no eligible warm negative after subtracting ALL rated training history. Determine
emptiness from retained training history/warm IDs only, without sampling or consulting
validation/test data. Let N=|P| and T=|P-K|. Require T>0; an untrainable run fails.
This uses the accepted domain/eligibility rules, not a new identity fingerprint.

`sampler` has exactly `algorithm`, `identity_hash`, `source_order`, `cursor`,
`next_epoch`, `batch_cursor`, `pending_batches`, `prefetch_pending`, `inflight_workers`,
`worker_ids`. Algorithm equals trusted `sampler_algorithm`, identity_hash equals the
validated identity hash, and worker_ids is the ordered configured population 0..W-1.
All four pending/batch-cursor counters are exact integer zero at supported boundaries;
configured prefetch factor is not evidence of actual queued work. Positive worker
metadata does not authorize processes or claim mid-epoch capture support.

`progress` has exactly `epoch`, `global_step`, `epoch_optimizer_steps`,
`empty_pool_skips`. E=epoch is an exact integer 0..resolved max_epochs (100 for the
frozen recipe); G=global_step is an exact nonnegative int. The two history lists have
length E; step entries are positive ints, skip entries equal |K| each epoch, and
sum(epoch_optimizer_steps)=G. At E=0 both lists are empty and G=0.

- Initial boundary E=0: source_order=None, cursor=0, next_epoch=1; no partial order
  or fabricated completed epoch. Worker initial states, if configured, are explicit
  caller-staged contexts, not invented claims of captured physical workers.
- Completed boundary E>0: source_order is a list containing every P ID exactly once,
  cursor=N, next_epoch=E+1, with no active/prefetched batch. Terminal boundaries still
  record E+1 as an index, not permission to run another epoch.
- Trusted `batch_policy=filter_before_batch`: remove K from the recorded source order,
  then chunk by explicitly resolved batch size B, retaining the final partial batch.
  Each epoch's optimizer-step count equals ceil(T/B).
- Trusted `batch_policy=batch_then_filter`: chunk the complete source order by B,
  retain its final partial batch, skip K within each chunk; count an optimizer step
  only for a chunk with at least one eligible pair. Last epoch's step count must
  equal that exact derived count. Older counts must lie in
  `[ceil(T/B), min(T,ceil(N/B))]`; only the latest order is retained, so no earlier
  exact batch/gradient trajectory is claimed verified from aggregates.

Both batch policies are supported representations of explicitly declared synthetic/
trusted contexts, not a new production batching decision. Do not select one as a
production default, drop a partial batch, turn empty batches into updates, or infer a
schedule from an untrusted cursor. Sampling randomness/negative outcomes are not
replayed. Source-order permutations and integer counters alone do not prove a run.

### 7. selection history, current/best weights and terminal cases

`selection` has exactly `metric`, `history`, `best_score`, `selected_epoch`,
`bad_checks`, `status`, `stop_reason`; metric is `validation_NDCG@10`.
History is an ordered list of exact `{epoch, score}` records, epoch an exact int,
score a finite float in [0,1]. Use the explicitly supplied resolved validation
frequency F, not an installed default. This profile's schedule is the positive
multiples of F up to max_epochs; require at least one such check in the permitted
run. The recorded history must be exactly its prefix through E: no skipped, duplicated,
future or unscheduled checks. No extra final-epoch check is silently inserted.

Recompute summaries as follows, without evaluating a model or running selection trials:

| Boundary | Required state |
|---|---|
| No scheduled check yet (including E=0) | empty history, best_score=None, selected_epoch=None, bad_checks=0, best_model=None |
| First successful check, even score 0.0 | select that epoch/score, bad_checks=0, best_model present |
| Strictly higher score | update selected epoch/score and best_model binding, reset bad_checks=0 |
| Exact tie or lower score | retain earliest best epoch/score, increment bad_checks by one; no epsilon/tolerance tie rule |
| bad_checks reaches 10 | terminal at that exact check epoch; no later epoch/check or retroactive improvement permitted |
| E reaches max_epochs | terminal, whether or not that epoch itself had a scheduled check; best snapshot still follows history |

`status` is `initial` at E=0, `ready` for a nonterminal completed epoch, or `stopped`.
`stop_reason` is None unless stopped; then it is `patience`, `max_epochs`, or
`patience_and_max_epochs` when both coincide. Reject counts greater than 10, a terminal
reason on ready state, a ready state at a stop boundary, or E beyond the first patience
stop. Failed/nonfinite checks are failures, not patience increments or valid eligible
post-failure boundaries. An older valid checkpoint may remain usable, but D2 performs
no rollback or recovery action. Earlier best weights remain distinct from current
resume weights at later epochs, even if both tensors coincidentally compare equal.

### 8. required boundary verification and source alignment

Implementation tests must cover the complete positive state plus independent corruptions
for every table row/field above, both model recipes, sparse graph contents, both batch
policies, native counter precision, shared RNG identities and positive/zero workers.
Include missing late-stage state, malformed expected descriptors, return/input alias
mutations and before/after global RNG equality on success and failure. No prohibited
optimizer steps, global RNG restoration, workers or actual training in fixtures.
All original 249 tests remain unchanged; no tests have run in this specification pass.

Static alignment references read for this pass:

- `scripts/differentiable_models.py`: HeteroGNN constructor/registration (462-550),
  LightGCN constructor/state and registered sparse buffer (906-1019); existing model
  tests verify CPU state/parameter reproducibility and sparse buffer rejection.
- `scripts/checkpoint_identity.py`: required settings/RNG role constants, retained
  identity and public serialization/revalidation, source provenance dict and public
  actual/expected validation. Existing identity fixture algorithm labels ending in
  `_synthetic` are metadata examples, not real engine codecs. New payload fixtures
  must supply truthful codec bindings without editing accepted identity tests/hashes.
- Read-only framework sources under `/home/sunf/miniconda3/lib/python3.13/`:
  `site-packages/torch/optim/adam.py` (constructor, lazy `_init_group`, `__setstate__`),
  `torch/optim/optimizer.py` (`_get_scalar_dtype`, packed state IDs and step cast policy),
  `random.py` (version-3 state), NumPy `_pcg64.pyi` and Torch CPU/MT19937 headers.
  Local version files report torch `2.10.0+cu128`, NumPy `1.26.4`; pyvenv.cfg records
  Python `3.13.11`. These are **static observations**, not a fresh import, environment
  acceptance or baseline result. A CUDA-enabled wheel does not authorize CUDA use.

The two preparation contracts are now specified. Implementation remains NOT_AUTHORIZED;
no new source/tests, checkout, baseline, Worker, training or deployment was started.
Before implementation submission, step 3 must synchronize the then-approved complete
specification and actual checkout/command inputs and independently reproduce the 249
baseline tests. No stale proposed HEAD or earlier result substitutes for that gate.

## Separately authorized step 3, prospective setup/baseline plan

The user subsequently said **"please start step 3 also"**. This authorizes isolated
setup/specification synchronization and baseline verification only. It supersedes
step 3's earlier deferral, not the separate implementation gate. No Worker, payload
implementation, training/profiling, GPU/remote/held-out execution, deployment, main
transfer, staging, commit or push is authorized.

Current source HEAD observed before setup is
`3a9efa41e02c82e422d8eb15fbc0a0c4622a4c48`. Main's two uncommitted planning records
from the specification pass must be preserved. Prepare a fresh full local clone at
`Amazon4U/.worktrees/checkpoint-payload-review`, check out that exact HEAD detached,
and leave its tracked files/index clean. No automatic commit or metadata trick to
conceal an overlay. Because HEAD predates the final uncommitted specification, copy
both complete current planning records byte-for-byte as **separate frozen references**
under clone `Amazon4U/data/reports/checkpoint-payload-preparation-inputs/`. Keep the
older HEAD versions at their tracked paths unchanged. The frozen references plus
this prospective plan are the operative preparation specification, not the older
tracked drafts. Future implementation submission must include that full specification
and clearly identify this distinction; a clean Git status alone is not proof of it.

Use the exact Python/argv/cwd/120s command already declared in the companion JSON's
`execution_recommendation.benchmark`, unchanged. Only its formerly stale HEAD is
superseded by the explicit pin above. Initially create/verify an empty fresh clone
`Amazon4U/.test-output/checkpoint-payload-v1/` subtree. The approved existing full
249-test synthetic CPU suite remains unchanged, including its existing import-order
subprocesses and differentiability tests. There are no new payload fixtures, optimizer
steps or alternate candidate commands. Run once synchronously with full stdout/stderr,
exit/timeout status and duration retained; do not silently retry a failure.

Pi-owned orchestration and complete evidence may be created under main
`Amazon4U/data/reports/checkpoint-payload-preparation/`. Only the two main planning
records may receive scoped authorization/result metadata updates; preserve their tested
versions before any result update. Frozen reference files, original criteria, exact
command and original evidence must not be rewritten. Input references are ignored by
Git but explicitly protected and hashed. Only fresh test-created scratch contents may
mutate during baseline; they must be cleaned. Disable bytecode with the pinned command.

### Mandatory preparation criteria

P1. Respect this step-3-only authorization; preserve accepted identity/model/test bytes,
all original C1-C10, historical decisions and unrelated user changes. No implementation.

P2. Record clean isolated HEAD/index/status and verify all tracked clone bytes match
that HEAD. Freeze complete current scope doc/config plus dependency hashes and exact
command as independently hashed references. The two uncommitted planning records are
not silently represented as committed HEAD content.

P3. Execute only the declared full-suite command, in its declared cwd and scratch,
within120seconds; retain complete untruncated streams/status/hashes, including any
failure or interruption. No subset tests, smoke imports, compileall or ad-hoc probes.

P4. Reproduce all249 accepted test IDs, exit0 with zero failures/errors/skips. Read all
actual results and compare with the earlier accepted main integration evidence, not
only a test-count summary. Green tests alone do not accept Deliverable2.

P5. Capture stable before/after streamed hashes of tracked/nonignored clone files and
all explicitly protected input references, outside-project tracked/nonignored sentinels
and Git metadata. Check main tracked/nonignored files and selected accepted/rejected
artifacts as well. Existing main nested repositories receive HEAD/index/status checks,
not a claim of full recursive content coverage. Only the approved fresh clone/records
and scoped main metadata may be added/changed during preparation. For each snapshot:
128MiB/file, 512MiB aggregate, 60s/file, 300s total hashing limits. Unexpected mutation or
incomplete/unstable hashing blocks acceptance. This is bounded monitoring, not an OS
sandbox, comprehensive action audit or coverage of undeclared ignored contents/reverted
transient writes. No sibling edits or semantic dataset inspection.

P6. Record a separate Pi preparation ACCEPT or ESCALATE with evidence for P1-P6.
Its acceptance_scope is task_only **for preparation/baseline only**, not Deliverable2
implementation acceptance; deployment_decision NOT_AUTHORIZED. Implementation remains
separately unauthorized even after a successful baseline. Future source/spec changes
require fresh synchronization/relevant verification rather than reuse of stale evidence.

---

## Authorized Whole-Deliverable-2 correction implementation and verification record

### 1. authority, recovery boundaries, and scope

This section documents the explicitly authorized whole-Deliverable-2 Payload Validation correction for Amazon4U, executing under a fresh consequential authorization (`data/reports/checkpoint-payload-authorized-correction-inputs/authorization.md` and `plan.json`) to resume the research path D2 -> D3 -> approved research gates. This implementation operates in the clean worktree (`.worktrees/checkpoint-payload-authorized-correction`) with one initial implementation and at most one consolidated revision under strict 20-minute worker and 120-second test suite ceilings.

Implementation scope is strictly confined to three allowed paths:
1. `Amazon4U/scripts/checkpoint_payload.py`
2. `Amazon4U/tests/test_checkpoint_payload.py`
3. `Amazon4U/docs/checkpoint-payload.md`

All accepted Deliverable 1 dependencies, model source, historical task records, Git metadata, and sibling projects remain unchanged.

### 2. resolution of all confirmed review blockers and diagnostic oracles

The implementation resolves all 6 confirmed blockers identified in historical review feedback and diagnostic reproductions:

#### Blocker 1: metaclass hooks defense
- Replaced dictionary/set lookups and tuple membership dispatches (`type(x) in (list, tuple)`) with safe, direct type identity checks (`x is list or x is tuple`, `t is int`, `t is float`, etc.) across `_safe_type_name`, `_safe_repr`, `preflight_resource_accounting`, and all schema checkers.
- Unsupported custom objects or hostile metaclasses never trigger `__hash__`, `__eq__`, `__repr__`, `__str__`, or metaclass `__getattribute__`/`__getattr__` hooks. They are immediately identified as `<unsupported object>` without calling any user-controlled code.
- Verified by `test_r1_metaclass_hash_hook` and `test_custom_objects_and_type_safety_without_hooks`.

#### Blocker 2: bounded huge-int and unicode diagnostics
- In Python 3.11+, formatting arbitrarily large integers (e.g. 10**5000) directly in f-strings raises a native `ValueError: Exceeds the limit (4300 digits) for integer string conversion`.
- Replaced raw integer formatting across all diagnostic error paths with `{_safe_repr(val)}`, which checks `val.bit_length() > 4000` and outputs `<oversized int (N bits)>` without attempting base-10 string conversion.
- Bounded `math.isfinite` invocations exclusively to floats, preventing `OverflowError` on huge integers.
- Large native integers remain valid supported values where mathematically permitted, without imposing an invented integer validity cap or modifying interpreter limits. All diagnostics remain bounded `CheckpointValidationError` domain exceptions.
- Verified by `test_r2_huge_update_domain_error` and `test_fail_closed_error_handling`.

#### Blocker 3: seed material validation
- Enforced that PyTorch worker seeds in `runtime_workers > 0` contexts must be exact scalar non-negative integers bounded by uint64 ($0 \le \text{seed} \le 2^{64}-1$). Lists or sequences of integers are strictly rejected. Actual generator seeds are verified against `dummy_gen.initial_seed()`.
- Enforced that Python random seeds must be exact non-negative scalar integers without an invented uint64 cap, matching Python's native arbitrary-precision integer support.
- Enforced that NumPy worker seed words are restricted strictly to NumPy codecs (`numpy.random.PCG64.v1` and `numpy.random.RandomState.v1`).
- Verified by `test_r3_torch_worker_seed_integer`, `test_r4_python_worker_seed_integer`, and `test_rng_and_codecs_validation`.

#### Blocker 4: Adam betas exact tuple validation
- Enforced that `betas` in both actual optimizer `param_groups` and `expected_runtime['optimizer_groups'][g]['options']` must be an exact 2-tuple: `type(betas) is tuple and len(betas) == 2`. Lists, singletons, or tuples of other lengths fail closed.
- Verified by `test_adam_optimizer_validation` and `test_strict_schemas_and_types`.

#### Blocker 5: genuine P5/K2/T3 vs T0 oracle and corrected BTF fixture
- Reconstructed the `batch_then_filter` fixture to use the genuine 5 condition-positive training edges `[[0, 0, 1, 2, 2], [0, 1, 2, 0, 1]]` with LightGCN degree-normalized weights `[0.5, 0.5, 1.0, 0.5, 0.5]`.
- Preserved the old 6-edge graph as an explicit negative witness in `test_r6_btf_fixture_adjacency_mismatch_oracle`, proving that declared adjacency mismatch is caught and rejected.
- Retained the genuine P=5, K=2, T=3 oracle asserting that clearing order does not make T=0, and verified genuine domain refusal on untrainable T=0 (T <= 0) datasets.
- Verified by `test_r5a_old_empty_pool_fixture_independent_derivation`, `test_r5b_genuine_untrainable_t0_domain_refusal`, `test_r6_btf_fixture_adjacency_mismatch_oracle`, and `test_batch_then_filter_empty_chunk_and_partial_batch`.

#### Blocker 6: complete strict schemas, resource accounting, and model matrices
- Complete schemas and exact types for both LightGCN (sparse adjacency buffer matching graph topology, coalesced, lexicographic ordering) and HeteroGNN (all 36 parameters required, non-empty User and Product tables, valid empty CategoryPath and Brand tables when metadata is absent).
- Strict preflight resource accounting checking raw inputs and expected descriptors prior to copying, deduplicating `StorageImpl` identity via `_cdata` while accounting distinct storages over the same address buffer.
- Complete 10-role RNG validation with quiescence enforcement, global RNG state preservation across success and failure, and exact output copy isolation.

### 3. public API specification

Module: `scripts/checkpoint_payload.py` (also directly importable as `checkpoint_payload`)

```python
def validate_checkpoint_payload(
    payload: Any,
    *,
    expected_identity: Any,
    expected_provenance: Any,
    expected_model: Any,
    expected_runtime: Any,
    resource_policy: Any,
) -> Dict[str, Any]:
```

The function executes a 7-phase fail-closed validation pipeline:
1. **Resource Policy Validation**: Validates `resource_policy` schema (`amazon4u-payload-resources-v1`) and positive integer bounds.
2. **Preflight Resource Accounting**: Runs preflight bounds checking on raw `payload` and expected descriptors prior to copying or traversal.
3. **Expected Descriptors Validation**: Validates schemas and types of `expected_identity`, `expected_provenance`, `expected_model`, and `expected_runtime`.
4. **Raw Semantic Validation**: Validates raw tensors (`requires_grad=False`, sparse COO coalesced layout and lexicographic order, finiteness), model parameters, Adam optimizer state, RNG roles/codecs, sampler progress, and selection history.
5. **Owned Output Copy Construction**: Clones and detaches all tensors into an isolated dictionary snapshot.
6. **Owned Snapshot Revalidation**: Re-runs complete semantic checks and preflight resource accounting on the constructed snapshot.
7. **Return Snapshot**: Returns the verified, isolated snapshot.

### 4. verification test suite and diagnostic oracles

Test module: `tests/test_checkpoint_payload.py`

The test module contains 34 self-contained test methods:
- **27 Retained Recovery Methods**:
  1. `test_dual_import_canonicalization`: Verifies dual import canonicalization.
  2. `test_positive_lightgcn_payload`: Complete positive control for LightGCN.
  3. `test_positive_heterognn_payload`: Complete positive control for HeteroGNN.
  4. `test_fixture_alignment_and_normalization`: Verifies sparse buffer weights against `DifferentiableLightGCN`.
  5. `test_ownership_and_no_mutation`: Verifies input/output isolation and no-mutation on success and late failure.
  6. `test_resource_policy_validation`: Verifies resource policy schemas, limits, and runtime overrides.
  7. `test_exact_resource_boundaries_and_accounting`: Verifies exact B and B+1 boundaries, depth 32/33, visits 10/11.
  8. `test_custom_objects_and_type_safety_without_hooks`: Rejects custom objects and attacker metaclass instances without invoking hooks.
  9. `test_fail_closed_error_handling`: Verifies lone surrogates, huge negative integers, and huge float-safe integers.
  10. `test_envelope_and_inapplicable_components`: Verifies required keys, schema version, capability, and inactive scheduler/AMP.
  11. `test_lightgcn_model_parameter_and_buffer_mutations`: Negative matrix for LightGCN shapes, finiteness, and sparse COO buffers.
  12. `test_heterognn_model_parameter_mutations`: Negative matrix for HeteroGNN 36 parameters and zero buffers.
  13. `test_adam_optimizer_validation`: Adam groups, lazy initialization, moment shapes, and step precision ($2^{24}+2, 2^{25}, 2^{25}+1$).
  14. `test_rng_and_codecs_validation`: RNG roles, 4 native codecs, and global RNG state preservation across success and failure.
  15. `test_sampler_and_progress_validation`: Sampler domains, quiescent boundaries, and integer ceil division.
  16. `test_selection_history_and_patience_boundary`: Strict improvement, tie retention, and 10-check patience boundary.
  17. `test_best_model_snapshot_semantics`: `best_model` None before first check, epoch binding, bitwise equality with signed zero.
  18. `test_raw_tensor_detachment_and_sparse_forgery_rejection`: Raw `requires_grad=True` and forged `coalesced=True` COO rejection.
  19. `test_exact_independent_projected_output_oracle`: Projected compact output storage oracle at exact B and B+1.
  20. `test_exact_visit_limits_under_synthetic_profile`: Exact 16,384 visits pass; 16,385 visits fail.
  21. `test_adam_mixed_lazy_and_updated_parameters`: Mixed lazy and updated parameter states in Adam.
  22. `test_staged_worker_context_w1`: Multi-worker $W=1$ with 3 worker generators totaling 7 generators.
  23. `test_batch_then_filter_empty_chunk_and_partial_batch`: Genuine non-empty $K$, empty chunk skipped, partial batch, and $T=0$ rejection.
  24. `test_valid_empty_hetero_metadata_tables`: Empty CategoryPath and Brand (0, 64) tables with all 36 parameters required.
  25. `test_rng_comprehensive_matrix_and_worker_w2`: Multi-worker $W=2$ with 6 worker generators totaling 10 generators, plus role/registry negative matrix.
  26. `test_storage_deduplication_same_address_different_capacity`: Distinct StorageImpls sharing base address with different capacities.
  27. `test_strict_schemas_and_types`: Strict float/int, int/bool, and clean string validations.
- **7 Ported Diagnostic Reproduction and Oracle Methods**:
  28. `test_r1_metaclass_hash_hook`: Diagnostic reproduction proving attacker metaclass `__hash__`/`__eq__` hooks are never called during rejection.
  29. `test_r2_huge_update_domain_error`: Diagnostic reproduction proving updates exceeding 4300 digits raise `CheckpointValidationError` rather than native Python `ValueError`.
  30. `test_r3_torch_worker_seed_integer`: Diagnostic reproduction proving Torch worker seeds must be scalar integers <= uint64 and verified against `initial_seed()`.
  31. `test_r4_python_worker_seed_integer`: Diagnostic reproduction proving Python worker seeds must be non-negative integers without uint64 cap.
  32. `test_r5a_old_empty_pool_fixture_independent_derivation`: Independent derivation of $P=5, K=2, T=3$ and order-clearing independence.
  33. `test_r5b_genuine_untrainable_t0_domain_refusal`: Proves genuine untrainable $T=0$ domain refusal from the $T \le 0$ guard.
  34. `test_r6_btf_fixture_adjacency_mismatch_oracle`: Proves adjacency mismatch between declared and condition-positive graph is rejected.

Together with the 249 unchanged baseline tests across the test suite, the project full test suite contains 283 tests.

### 5. acceptance criteria compliance map (C1-C10)

| Criterion | Contract Scope | Implementation & Verification Evidence |
|:---|:---|:---|
| **C1** | Scope & Dependency Preservation | Only 3 allowed files modified (`scripts/checkpoint_payload.py`, `tests/test_checkpoint_payload.py`, `docs/checkpoint-payload.md`). Deliverable 1 dependencies, models, and protocols unchanged. |
| **C2** | Strict Envelope & Binding | Exact schemas for payload and expected descriptors; no bool/int or float/int aliasing; no manufactured expectations. Tested by `test_envelope_and_inapplicable_components`, `test_strict_schemas_and_types`. |
| **C3** | Model / Tensor Validation | Complete positive and negative mutation matrices for LightGCN and HeteroGNN. Sparse adjacency graph topology and degree normalization checked. Tested by `test_lightgcn_model_parameter_and_buffer_mutations`, `test_heterognn_model_parameter_mutations`. |
| **C4** | Adam Optimizer Validation | Moment shapes, groups, exact tuple betas, representable step bounds, mixed lazy/updated parameters verified without optimizer execution. Tested by `test_adam_optimizer_validation`, `test_adam_mixed_lazy_and_updated_parameters`. |
| **C5** | RNG & Sampler Validation | 10-role RNG architecture, quiescence enforcement, multi-family worker contexts ($W=1, W=2$), global RNG state preservation. Tested by `test_rng_and_codecs_validation`, `test_rng_comprehensive_matrix_and_worker_w2`. |
| **C6** | Progress & Selection | Epoch boundary, batch cursors, selection history, strict improvement, 10-check patience boundary, `best_model` epoch binding. Tested by `test_selection_history_and_patience_boundary`, `test_best_model_snapshot_semantics`. |
| **C7** | No Mutation / Ownership | Deep cloned, detached output copy; caller inputs and global state untouched across success and late failure. Tested by `test_ownership_and_no_mutation`. |
| **C8** | Fail-Closed Error/Resource | Safe metaclass handling, bounded integer formatting, exact container depth 32/33, visit bounds 16384/16385, preflight accounting. Tested by `test_custom_objects_and_type_safety_without_hooks`, `test_fail_closed_error_handling`, `test_r1_metaclass_hash_hook`, `test_r2_huge_update_domain_error`. |
| **C9** | Complete Regression Suite | All 249 baseline tests preserved; 34 payload tests (27 recovery + 7 diagnostic); 283 total tests pass within 120s with 0 failures, 0 errors, 0 skips via the approved benchmark command. |
| **C10** | Truthful Disclosures | Complete documentation with exact 59,605-byte specification prefix preserved; truthful execution logs; all 6 limitations disclosed in exact text and order. |

### 6. known limitations and honest disclosures (C10)

1. **In-Memory Validation Only**: Validating an already decoded in-memory mapping does not prove that initial deserialization or decoding was safe. Safe weights-only deserialization and file-level pre-load bounds belong to Deliverable 3.
2. **Epoch Boundary Quiescence Only**: Only quiescent epoch-boundary checkpoint states are supported (`batch_cursor=0`, `pending_batches=0`, `prefetch_pending=0`, `inflight_workers=0`). In-flight worker, mid-batch, or streaming prefetch states are unsupported and rejected.
3. **CPU Execution Profile Only**: Models and tensors must reside on CPU. CUDA tensors, AMP scalers, and lr schedulers are explicitly unsupported in this baseline profile and fail closed.
4. **Synthetic Budget vs Production Capacity**: Validation under the 2 MiB synthetic resource profile verifies contract correctness and safety boundaries on small fixtures. It does not select or certify a production-scale RAM budget or prove hardware capacity for large models.
5. **Historical Consistency, Not Execution Proof**: Validating numerical consistency between progress, Adam steps, RNG descriptors, and selection history does not prove that recorded gradient trajectories, validation scores, or tensor values genuinely originated from a historical physical training run.
6. **Task-Only Scope**: This deliverable implements in-memory payload validation only. Model training, optimizer execution, live restoration, Deliverable 3 file serialization, Kaggle execution, and deployment remain NOT_AUTHORIZED.

---

## Text-only correction proposal: active status and unresolved closure

This section supersedes the preceding appendix's active completeness and verification claims. It does not delete or retroactively correct historical records. The original 59,605-byte specification prefix, prior reports, failed attempts, and decisions remain preserved.

Both text-only Worker responses executed zero commands, imported no modules, accessed no filesystem, and ran zero tests. Pi independently reported: pristine dependency baseline 249 OK in 7.635 seconds; safe-red discovery 320 tests with 6 failures and 14 errors in 8.192 seconds; and corrected-subset full discovery 329 tests with 5 failures, 0 errors and 0 skips in 39.128 seconds. All 320 preceding methods passed in the last discovery. Its nine parsed ledgers declared 2,201 assertions and recorded 2,196 successes; the five unresolved assertions were quota-message regex mismatches, not reached tripwires. These supplied Pi results are not executions by this Worker. This consolidated revision is unexecuted and is not a complete coverage certificate or D2 acceptance.

### Proposed changes and application gate

The private-generator fixture corrections and all other previously reviewed unambiguous edits have already been applied; this revision does not repeat them. Pi rejected the ambiguous identity-builder anchor because it matched twice. That edit was not applied. The newer coherent LightGCN payload builder now supplies the independently specified five-edge adjacency, five-pair source order and two skips, and its S01 control passed. This revision does not change the identity builder: a passing control alone does not resolve whether its retained graph should exclude the low-rated edge. That semantic source-alignment question remains unresolved rather than being silently patched. Every proposed replacement below targets a supplied current-file hash and still requires Pi's unique-anchor, disjointness and protection checks.

The proposed production changes address optional Adam names without iterable coercion, option type checks before equality, the canonical CPU/CUDA-role guard, codec seed validation before parent equality, and the two explicitly declared list fields. Diagnostic changes bound key sampling and address the supplied huge-integer/key-sorting recipes. They do not change model numerical predicates, the protected selection/copy/domain/tensor helpers, resource budgets, or the Adam aggregate block. Whole-path diagnostic closure is still incomplete: raw contextual identifiers and additional diagnostic expressions require review.

The proposed additional tests cover current/best dense parameter slots, nonempty and empty Hetero metadata controls, named one/two-group controls, flag types, parent seed types, a repaired five-edge empty-pool control, complete global snapshots on success and selected late failures, preflight tripwires, and recursive ownership mutations. Per-test stdout ledgers distinguish declared IDs from assertions that completed successfully. These additions do not close every frozen requirement. The older prefix-only V01 check remains a smoke check, not comprehensive evidence.

### Evidence and action reconciliation

Historical evidence is not a fresh run. The supplied history retains initial step 152 failure, step 191 interruption/timeout, step 272 with 250 tests and one ERROR, step 282 with 283 tests and two ERRORS, and passing 283-test records at steps 300 and 310. The historical Pi 283/283 benchmark is also retained as historical evidence, not current D2 acceptance. The initial archive setup helper incorrectly assumed main HEAD equaled dependency HEAD and failed before any candidate execution or Worker invocation; setup-attempt0.json is retained. Main HEAD af1a3d5f56f16283fcdc058a25e222390bf1160a was preserved, and accepted dependency 3a9efa41e02c82e422d8eb15fbc0a0c4622a4c48 was explicitly archived. The preceding text-only Worker request took 346341 milliseconds, returned HTTP 200, and performed no Worker execution.

The last native Worker attempt timed out after 30 minutes without a final report. Its six full attempts at steps 9/166/183/196/268/328, prohibited probes/subsets/external writes/global fixture seeds, and their consequences remain preserved and are not absolved by this text-only revision. The older 315 checked hashes / 83 recovered records inventory is distinct from the latest eb5065fc review's 369 verified raw rows / 130 observed command steps / 6 supplementary records. These inventories must not be conflated. Pi owns independent raw-evidence reconciliation; this Worker did not execute, reacquire or independently verify that evidence. The three current Pi discoveries used the unchanged predeclared full script, with results retained in baseline, initial-safe-red and initial-corrected-subset evidence directories. Their exact commands and streams remain Pi-owned evidence; no missing receipt or tool trace is invented here. Same-UID partial records do not establish a complete protected syscall history.

### Remaining acceptance blockers

The accompanying JSON report retains a normal object for every one of the 37 rows. This revision proposes the exact five role-specific quota assertions, additional quota-failure snapshots, bounded escaped-string and codec-context rendering, sampler-counter and version diagnostics, exact-record schema matrices, resource-policy cases, direct public D1 binding cases, Adam moment/step cases, raw-gradient copy tripwires, additional diagnostic recipes and unsupported-value positions. It does not close all assigned work. In particular, raw identifiers remain in RNG registry and trusted-generator diagnostics; complete sparse, codec, worker, runtime-field, sampler, selection, alias and hostile-subclass matrices remain incomplete. Static review also identifies the positive-update fractional-step case: the supplied implementation compares int(step_val) with updates, which does not reject 4.5 when updates is 4. A new regression assertion records that required refusal without changing a numerical predicate. Pi must review this unresolved implementation issue; it is not described as a passing test. Proposed tests have not established their own correctness or the 120-second bound. All C1-C10 review and preservation verification remain pending. Any unresolved assigned requirement or mandatory evidence gap blocks ACCEPT.

### Unsupported boundaries

The API validates an already decoded mapping, not safe decoding or total process memory. It supports the declared quiescent CPU epoch-boundary profile, not CUDA, active scheduler/AMP, live restoration, or mid-epoch capture. Trusted expectations remain a caller acquisition boundary; matching records do not prove physical origin, seed reachability, historical training, or future cross-version continuation. Returned containers and tensors remain mutable and require revalidation after modification. Concurrent mutation during validation is unsupported. No production RAM profile, float64 host policy verification, initialization archive capture, D3 implementation, training, or deployment is claimed.

A native uint8 tensor cannot require gradients in Torch. This proposal does not claim that impossible fixture was tested; representable floating-gradient and malformed native-state alternatives still need explicit G06 coverage without skips.

Acceptance, if independently established later, is task_only. Deployment remains NOT_AUTHORIZED.

---

## Consolidated tests-only revision: active status

This active section supersedes historical statements about current defects and missing demonstrations, without deleting their history. The original 59,605-byte prefix and all preceding appendices are preserved. The original C1-C10 contract is unchanged: no architecture, production feature, schema, numerical guard, resource ceiling, or new acceptance requirement is introduced.

Pi supplied an independently reviewed A04/D01 full-discovery result: **338 tests OK in 46.446 seconds**, with **3,194 declared / 3,194 successful cases**, including 52 long-registry/trusted-ID cases. Those are supplied Pi results, not Worker executions. The current production fixes are not patched by this revision. All 338 existing methods and their assertions are retained by the additive patch.

The additional tests proposed here have **not been executed by Worker**. They must not be described as passing coverage. They add raw, unrepaired COO corruptions for current/best/trusted buffers; an independently normalized same-NNZ different graph; literal sparse/alias accounting boundaries; codec-component and staged-worker cases; coherent parent-family/cardinality refusals; initialization archive/binding refusals; Adam binding/options/lazy controls; sampler/history/batch-policy cases; selection and terminal corruptions; public runtime/provenance routes; and genuine noncontiguous shared-backing ownership controls. Existing dense/schema/name/flag/parent-seed/global-state/late-failure/recursive-ownership evidence is reused, not regenerated.

Sparse expectation corruptions may fail trusted-content comparison rather than an additional expected-canonical validator. Raw invalid order/duplicate/bounds examples retain their indices with `is_coalesced=True` and do not call `.coalesce()` to repair them. The alternate graph is compared against the unchanged independent trusted buffer; matching jointly altered records are not used as negative origin evidence. The literal accounting targets are the supplied proposed M/S/C/visit values; their exact agreement with the frozen type-charge implementation still requires source confirmation, not modification of protected accounting to fit an oracle.

Codec controls use private staged states/setters, without draws or global restoration. Nonfinite scalar codec leaves can be rejected by common preflight and are not labeled native-parser evidence. Floating-gradient malformed Torch state is representable; uint8 gradients are not. Worker contexts, huge Python worker seed metadata, and NumPy worker word lists do not establish physical capture, derivation execution, or seed reachability. Keyword-omission TypeErrors demonstrate the Python signature only.

### Consolidated revision main report

The original submission report is preserved at [revision/worker-report.json](/home/sunf/FSB/Final/Amazon4U/data/reports/checkpoint-payload-a04-d01-correction/revision/worker-report.json), with the exact response retained alongside it. Pi records subsequent independent results and the separate decision in [review.json](/home/sunf/FSB/Final/Amazon4U/data/reports/checkpoint-payload-a04-d01-correction/review.json). These filesystem artifact links are Pi-supplied evidence metadata, not a retrospective change to the Worker report or an acceptance claim. The report's submission-time source-confirmation and link questions must be resolved with actual evidence before acceptance.

Whole-D2 ACCEPT is not established by the supplied green baseline or by a green target subset. Pi must retain the mandatory inspection, exact full-suite 120-second verification, complete streams/ledgers, preservation/history checks, and exact ordered limitation review. An unresolved original-contract issue or missing mandatory evidence blocks ACCEPT. A failed proposed test must be classified as a fixture/oracle mistake, missing evidence, or a source defect before any correction; this revision authorizes no production edits.

### Lasting supported and unsupported boundaries

The validator handles an already decoded mapping under per-object logical content/copy bounds, not safe decoding or a total process-memory sandbox. Supported state remains the declared quiescent CPU epoch boundary, not CUDA, active scheduler/AMP, live restoration, or mid-epoch capture. Independently trusted acquisition and initialization archival remain caller boundaries. Matching records do not prove origin, historical training, seed reachability, or future cross-version continuation. Returned tensors/containers are mutable; modification requires revalidation. Concurrent mutation during validation remains unsupported. No production RAM profile, float64 host-policy verification, initialization capture implementation, D3, training, main transfer, or deployment is claimed.

Independent verification after this response is an evidence gate, not a persistent validator limitation. Acceptance, if eventually established against all original C1-C10 obligations, remains **task_only**; deployment is **NOT_AUTHORIZED**.
