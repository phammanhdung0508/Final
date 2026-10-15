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
