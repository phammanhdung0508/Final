# GPU-READY design note (phase 1, deliverable 2)

Status: **AGREED (user, 2026-10-24); implemented, and GPU-READY accepted by the user on
2026-10-25** (see [`gpu-ready-traceability.md`](gpu-ready-traceability.md)). EX01, option C, the new
names and the GR06 checkpoint design are agreed. Heterogeneous aggregation uses full
neighborhoods for Musical_Instruments. No code is written or run by
this note. It covers
requirements GR01, GR02, GR06 and GR07 in [`gpu-ready-preparation.md`](gpu-ready-preparation.md)
and records one finding that changes the profiling path.

## Implementation status (2026-10-24, local CPU only)

User go-ahead on 2026-10-24: "You start right now." Nothing has run on a GPU or on
Amazon data. Each new module has CPU tests; frozen files are unchanged.

| Part | Module | Tests | Status |
|---|---|---|---|
| EX01 models | `scripts/vectorized_models.py` | `tests/test_vectorized_execution.py` | Done. Forward, masked forward, loss, gradients and 5 Adam steps match the accepted models on CPU for both models, 3 seeds and both feedback conditions |
| EX01 evaluation | `scripts/vectorized_evaluation.py` | same | Done for the validation split. The report is identical to the accepted evaluator when both rank the same scores, including ties, shortages, exclusions and fallbacks. Test split refused (not authorized) |
| EX01 trainer | `scripts/vector_trainer.py` | `tests/test_vector_trainer.py` | Done. Block 2 recipe, private CPU generators, epoch-boundary capture and restore. Restart is bit for bit identical on CPU |
| GR01, GR04, GR05 runtime | `scripts/gpu_runtime.py` | `tests/test_gpu_ready_parity.py` | Done: explicit device refusal, determinism policy, environment fingerprint, global RNG guard |
| GR03, GR07 parity run and result | `scripts/gpu_ready_parity.py` | same | Done on CPU: `amazon4u-H-result-cuda-v1` with `GPU_READINESS_GATES` |
| GR06 checkpoint profile with D2 and D3 | `scripts/checkpoint_payload_cuda.py`, `scripts/checkpoint_storage_cuda.py` | `tests/test_checkpoint_cuda.py` | Done on CPU (worktree `gr06-cuda-checkpoint`). Restart through publish, retrieve and restore is bit for bit identical for both models. CUDA parts are faked; real CUDA restore is run 3 |
| GR08 real transport and O-side cuda-v1 record | `multiautoresearch/adapters/amazon4u_kaggle.py`, `multiautoresearch/adapters/amazon4u_gpu.py` | `tests/test_amazon4u_gpu.py` | Done locally against a fake Kaggle CLI (same worktree). The bundle runs end to end on CPU and reports the expected run binding. No Kaggle call made; see [Kaggle execution](gpu-ready-kaggle-execution.md#gr08-real-transport) |

User decisions on 2026-10-24, after this implementation:

- Working in main instead of a worktree: accepted this once ("What's done is done");
  later implementation goes in a worktree (`AGENTS.md`, Worktrees).
- Frozen documents: the user asked to restore the nine protocol documents and
  `checkpoint-payload.md` to their frozen bytes. Done; the full suite passes (555
  tests, 81 s, peak 0.94 GB).
- Separate trainer: accepted on condition that it is verified with the same inputs,
  sampled triples and initialization, comparing optimizer steps, and that the new
  sampler's distribution is verified. Done in `tests/test_vector_trainer_equivalence.py`:
  - every step matches `CPUBackend.update` with `HCPUAdam` on the accepted model: loss,
    parameters and Adam moments, both models, 3 seeds, both conditions, 2 epochs;
  - the sampler's support equals `NegativeSampler.get_negative_pool`, including a
    low-rated warm item excluded under rating_ge_4;
  - draws are uniform (chi-square p > 1e-3) and match the accepted sampler's
    distribution (two-sample chi-square);
  - one user's draws do not depend on other users in the batch;
  - empty-pool skips match the accepted sampler.
  These tests found and fixed one bug: the heterogeneous parameter groups were
  registered in sorted order, not the accepted order, which misaligns optimizer state.
- Product-bias tolerance: agreed in principle, subject to a mathematical proof and
  targeted tests. After the corrected Adam bound the user replied "now it should lgtm"
  (2026-10-24). The clarification is agreed.

Deviations from this note, recorded for review:

- **Separate trainer, not a `CUDABackend` inside `Harness`.** `Harness` and
  `check_envelope` accept only the `H-REF-synthetic-cpu-v1` profile and build v1 CPU
  payloads, so a GPU backend cannot run through them without editing H28 files.
  `vector_trainer.py` reuses `Selection`, `validation_score` and `ADAM_OPTIONS` from
  `training_harness.py` instead. H28 stays unchanged.
- **Negative sampling (revised for GR06, 2026-10-24).** `NegativeSampler.get_negative_pool`
  builds each user's pool in Python for every positive, which is too slow at full scale.
  The trainer now draws through H-REF's own `PrivateRNG` with the generators
  `expected_runtime` declares, and finds the drawn pool position by binary search instead
  of building the pool. With the same generators it produces exactly the triples
  `NegativeSampler.prepare_batch` produces (`tests/test_vector_trainer_equivalence.py`).
  The earlier version used its own torch generators and matched only in distribution.
  That version is replaced, because a D2-valid checkpoint must hold the declared
  generators.
- **Aggregation without sparse matrix products.** Both models use `index_select` and
  `index_add`. PyTorch documents deterministic CUDA kernels for these under
  `use_deterministic_algorithms(True)`; it does not list sparse-dense products.

**Finding: one heterogeneous parameter has an exactly zero BPR gradient.** User
decision, 2026-10-24: "Agree in principle, but I will need subject to mathematical proof
and targeted tests." Proof and tests follow. Agreed by the user on 2026-10-24 after
review. No science changes.

### Proof: the layer-1 Product self bias cannot affect loss or rankings

Notation follows `heterogeneous-encoder.md`. Let `b = b_Product^(1)`, the self bias of
the second (identity) layer for Product nodes.

1. **`b` enters only Product outputs, once, additively.** Layer 1 computes, for every
   product `p`, `h_p^(2) = W_(self, Product)^(1) h_p^(1) + b + mean_r m_(p, r)^(1)`. The self
   term applies to every product, with or without active relations. Layer 1 is the last
   layer, so `h^(2)` never feeds another message. No other node's `h^(2)` and no `h^(1)`
   depends on `b`. Hence `h_p^(2) = g_p + b`, where `g_p` does not depend on `b`. The
   same holds for every batch-masked view, because masking only changes which edges
   enter the neighbor means.
2. **Scores shift by a per-user constant.** `s(u, p) = h_u^(2) . (g_p + b) = h_u^(2) . g_p
   + h_u^(2) . b`, and `h_u^(2)` does not depend on `b`. For a fixed user, every item's
   score moves by the same `c_u = h_u^(2) . b`.
3. **The loss does not depend on `b`.** Each BPR term uses one user and one masked view:
   `s(u, n) - s(u, p) = h_u^(2) . (g_n - g_p)`, so `c_u` cancels. The endpoint penalty
   uses only base User and Product ID rows, never `b`. So `L(b) = L(0)` for every `b`, and
   `dL/db = 0` exactly.
4. **Rankings and selection do not depend on `b`.** Validation ranks the candidates of
   one user by score, then by `parent_asin`. Adding the same `c_u` to every candidate
   preserves both the order and the exact ties. NDCG@10, HitRate and coverage are
   functions of those rankings, so selection is unaffected.
5. **Why it still moves in float32.** Autograd computes `dL/db` as a sum of `+h_u` and
   `-h_u` contributions, which leaves a rounding residual (about 1e-9 in float32, 1e-18 in
   float64). Adam rescales that residual into a full-size step in a noise-driven
   direction. The 1e-4 parameter tolerance cannot hold for `b`.

   **Bound on the drift (corrected 2026-10-24 after user review).** An earlier version
   said "at most `lr` per step". That holds at step 1 only. Per coordinate, Adam's step
   is `lr * m_hat_t / (sqrt(v_hat_t) + eps)` with
   `m_t = (1 - b1) * sum_k b1^k g_(t-k)` and `v_t = (1 - b2) * sum_k b2^k g_(t-k)^2`.
   Cauchy-Schwarz over the gradient history gives `|m_hat_t| / sqrt(v_hat_t) <= R_t`,
   where `R_t = sqrt(sum_(k<t) (1 - b1)^2 b1^(2k) / ((1 - b2) b2^k)) * sqrt(1 - b2^t) / (1 - b1^t)`.
   Adding `eps` only makes the step smaller. With `b1 = 0.9` and `b2 = 0.999`:

   | Step t | 1 | 2 | 3 | 4 | 5 |
   |---|---|---|---|---|---|
   | `R_t` | 1.0000 | 1.0014 | 1.0036 | 1.0068 | 1.0108 |
   | Sum up to t | 1.0000 | 2.0014 | 3.0050 | 4.0117 | 5.0225 |

   One implementation moves `b` by at most `lr * sum R_t`. Both start from the same
   value, so by the triangle inequality two implementations differ by at most
   `2 * lr * sum R_t`. After 5 steps that is `10.045 * lr`, about 1.0e-2. `R_t` keeps
   growing, toward about 7.27 for long runs, so `lr` per step is not a bound for
   training in general.
6. **No other parameter behaves this way.** In LightGCN, every parameter is a base
   embedding that the loss uses directly. In the heterogeneous model, the User layer-1
   bias does not cancel: `(g_u + b_U) . (h_n - h_p)` depends on `b_U`. The layer-1
   parameters that only produce CategoryPath or Brand outputs are never used by the loss.
   Autograd gives them no gradient (`None`), Adam never updates them, and both sides stay
   exactly equal.

Limit: point 4 is exact in real arithmetic. In float32, changing `b` changes rounding,
so a near-tie closer than float32 resolution could flip. This is the same rounding
limit as any other float difference, and the ranking checks cover it.

Targeted tests (`tests/test_product_bias_invariance.py`, float64 on the vectorized model,
which matches the accepted model in float32):

- loss, masked and unmasked, unchanged when `b` changes by a large random shift;
- computed `dL/db` at rounding level, while every other trained parameter's gradient is
  far larger;
- scores shift by exactly `h_u . delta` for every item, so top-20 lists are identical;
- `self_biases_l1.Product` is the only parameter with a computed, cancelling gradient;
  the other zero-gradient parameters are exactly the CategoryPath and Brand outputs and
  receive no gradient at all;
- the User layer-1 bias, as a control, does change the loss;
- the closed-form `R_t` equals the worst case found by Adam itself on the Cauchy-Schwarz
  extremal gradient sequence, and float32 drift of `b` stays within `lr * sum R_t`.

## Key finding: the accepted models are correctness references, not scale executors

Both accepted models in `scripts/differentiable_models.py` are written for clarity and
exact verification:

- `DifferentiableHeteroGNN.forward` loops in Python over every node of every type and
  over each node's relations and neighbors (around lines 647 to 744). It uses all
  neighbors; no neighbor sampling or fanout exists.
- `score_candidates` in both models runs a full forward pass per user, then scores each
  candidate with `torch.dot(...).item()` in a Python loop (around lines 865 and 1199).
- The accepted evaluator (`scripts/recommendation_evaluation.py`, around line 368)
  calls `score_candidates` once per eligible user and ranks a Python score dictionary.

Musical_Instruments KG-v1 has 57,439 users, 24,556 products and 396,958 rated edges
(`kg-input-audit.md`). Full-ranking validation every epoch would mean about 1.4 billion
Python-level scores per epoch, plus one full-graph forward per user. Moving these loops to
a GPU does not fix that: the cost is Python dispatch, not arithmetic.

**Consequence.** GPU-READY parity can use the accepted models on tiny fixtures, as
planned. Full-scale profiling needs a separate **vectorized execution path**:

- one propagation per evaluation pass, reused across users;
- batched user-by-item score matrices computed in chunks;
- a batched top-K step that reproduces the evaluator's history filtering, the
  score-descending then `parent_asin`-ascending tie order, the shortage counts and the
  fallback routes exactly;
- for the heterogeneous model, vectorized masked aggregation with batch positive-edge
  masking applied before aggregation at every hop. Musical_Instruments uses full
  neighborhoods; Electronics is decided after profiling; see
  [`gpu-ready-g2-g3-options.md`](gpu-ready-g2-g3-options.md).

This is new implementation beside the accepted models, verified against them on CPU.
It is not a change to model mathematics. If neighbor sampling is chosen, its rules
need a recorded amendment before use (`training-semantics.json` lists it as pending). This note proposes it as a separate requirement,
**EX01**, owned by the profiling prerequisites (G3), not by GPU-READY.

## GR01: device seam

The harness already routes placement, numeric calls and state transfer through a
backend object (`CPUBackend`, `scripts/training_harness.py` line 378). The controller
calls `require`, `verify_runtime`, `make_model`, `make_optimizer`, `update`,
`snapshot_model`, `load_model`, `load_optimizer` and `evaluate`.

Proposal:

- New module `scripts/gpu_backend.py` with `CUDABackend` implementing the same methods.
  `capabilities()` returns `{'device': 'cuda', 'accelerator': 'gpu', ...}` and the
  observed device name. `require()` refuses when `torch.cuda.is_available()` is false
  or the device or accelerator differ. There is no CPU fallback.
- `CPUBackend`, `HCPUAdam` and `training_harness.py` stay unchanged (H28 file).
- Backend selection is explicit in the run spec through a new profile ID,
  `gpu-ready-synthetic-cuda-v1`. The existing `H-REF-synthetic-cpu-v1` profile keeps
  selecting `CPUBackend` only.

Open question for the design review: `Harness.__init__` takes `backend=` already, so a
new backend may need no harness change. If any harness call path hard-codes CPU (for
example `torch.Generator(device='cpu')` in `PrivateRNG`), a small, separately scoped
change to the H28 file is needed. Phase 1 will list each such call before any edit.

## GR02: device-aware construction

The accepted models call `check_cpu_tensor` on parameters, propagated outputs and the
LightGCN adjacency, so they cannot run on CUDA unchanged.

Options:

| Option | What it means | Cost |
|---|---|---|
| A. Device parameter in the accepted models | Add `device=` and make checks device-aware | Edits frozen accepted code; re-review of models, D2 and D3 interactions |
| B. GPU sibling models | New classes that reuse the same math helpers but place tensors on a chosen device | Some duplication; accepted CPU classes untouched |
| C. Vectorized executors only | Build EX01 as device-aware from the start; use it for GPU parity and profiling | One new path instead of two; parity checked against the accepted CPU models |

**Agreed: C** (user, 2026-10-24). The scale finding means a new executor is needed anyway.
Building it device-aware avoids a GPU copy of code that cannot scale. The accepted CPU
models remain the oracle for both CPU and GPU parity.

## GR06: GPU checkpoint profile

Agreed by the user on 2026-10-24: a GPU checkpoint profile beside the accepted CPU
profile, and the design below. The user made two rules mandatory:

- **The GPU RNG state must not change during an epoch.** Every epoch compares the CUDA
  generator state at its start and end, and any change fails the run. This allows no
  random operation on the GPU. If neighbor sampling is adopted later, it must use a
  CPU generator, or this rule needs a recorded change first.
- **Resumed training must be bit-for-bit identical** to the uninterrupted run in the
  same environment. Any difference fails GR06. It is diagnosed, never covered by a
  tolerance.

### Names and versions

| Item | Name |
|---|---|
| Payload schema | `amazon4u-checkpoint-payload-cuda-v1`, beside the accepted `amazon4u-checkpoint-payload-v1` |
| Capability | `epoch_boundary_same_environment_resume` |
| Environment fingerprint | `amazon4u-gpu-environment-v1` |
| CUDA RNG codec | `torch.cuda.Generator.philox.v1` |
| Resource and IO limits | The existing D2 `amazon4u-payload-resources-v1` and D3 IO policy schemas, with limits pinned for each run before launch |

### States saved and verified

1. Current model parameters and buffers, stored as CPU float32 copies.
2. Best model, its epoch and the selection record, with the same exact-agreement rule
   as D2.
3. Adam groups, first and second moments and step counts, under the same rules as D2.
4. RNG: Python, NumPy and the PyTorch CPU generators.
   - Initialization, epoch shuffling and negative sampling stay on CPU generators, as
     in H-REF.
   - The CUDA generator state for each used device is saved too, with the device index
     and name.
   - The run checks that the CUDA state did not change during an epoch. A change would
     show hidden GPU randomness, and the run fails on it.
5. Sampler state at a quiet epoch boundary: no batch in progress, no prefetch, no
   DataLoader workers.
6. Progress: epoch, update count, validation history and patience counter.
7. Determinism settings: `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`,
   cuDNN `deterministic` and `benchmark`, and the TF32 flags.
8. Environment fingerprint:
   - GPU name and compute capability;
   - visible and used device counts;
   - driver, CUDA runtime and cuDNN versions;
   - Python, NumPy and PyTorch versions.
9. Run binding (G5): full bundle manifest hash and the run spec hash. This is recorded
   outside D1, because D1's source list does not cover new modules.
10. Scheduler and AMP are recorded as not applicable: constant learning rate, float32,
    with the reason `not_used_cuda_float32`.

Not saved: live CUDA tensors, caches and library handles, and the per-batch masked
adjacency. The masked adjacency is rebuilt from the fixed graph, whose hash is verified.

### Reusing D1, D2 and D3 unchanged

- **D1, called as is.** The identity records `execution_device` as `cuda:0`, the
  `PyTorch_CUDA_if_applicable` stream as active and deterministic algorithms as on. The
  first local test confirms that D1 accepts these values. If it does not, work stops
  for a decision; D1 is not edited.
- **D2, file unchanged.** The new module `scripts/checkpoint_payload_cuda.py` reuses D2's
  validators for the shared parts: model state, Adam, sampler and progress, and
  selection and best model. It reuses D2's codecs for the CPU RNG roles. Only the
  differing parts are new: the CUDA RNG role, the environment fingerprint, the
  determinism record, the AMP and scheduler records and the run binding.
- **D3, file unchanged.** D3's `publish_checkpoint` and `retrieve_checkpoint` call the v1
  validator directly, so the CUDA profile cannot pass through them.
  - A new module, `scripts/checkpoint_storage_cuda.py`, has two short functions. They
    reuse D3's root anchoring, bounded encoding and decoding, and no-clobber publishing,
    and call the CUDA validator instead.
  - The format, bounds and refusal rules are the same as D3. Each validator demands its
    own exact `schema_version`, so a v1 file never loads as CUDA and a CUDA file never
    loads as v1.
  - Reusing D3's private helpers ties the new module to D3's recorded hash. A change to
    D3 then fails a hash check instead of drifting silently.
  - Alternative: add schema dispatch inside D3. That edits frozen accepted code and is
    not recommended now.

### As implemented (2026-10-24)

The implementation is simpler than the plan above in two places, and stricter in one.

- **D2 is reused through its public validator, not its private helpers.** A cuda-v1
  payload is the v1 payload with three changed fields (`schema_version`, `capability`,
  `amp.reason`) and one added `device` section. `validate_cuda_checkpoint_payload`
  checks `device` against an independent expectation, then calls D2's
  `validate_checkpoint_payload` on the core view: the payload without `device` and with
  the three fields set to their v1 values. Nothing else is altered.
- **The CUDA generator is not an active D1 stream.** Training never consumes it (your
  rule), so `PyTorch_CUDA_if_applicable` stays inactive and D2's CPU-profile rule is
  satisfied truthfully. Its state is recorded in `device.global_rng`. Restore sets no
  global generator. It refuses unless the process already holds the recorded CPU and
  CUDA states, which a fresh process in the same environment does.
- **D3 is reused through its private helpers.** `checkpoint_storage_cuda.py` is D3's
  publish and retrieve with only the validator swapped. A test pins the recorded
  SHA-256 of D1, D2, D3 and H files (`configs/o-offline-implementation.json`).
- **D1 records `execution_device = cuda:0`** for CUDA runs. D1 accepts it.
- A cuda-v1 payload may also record `device = cpu`, with no CUDA states, so the same code
  path runs in local tests.

Mutation check: disabling any one of four device-section checks makes a test fail.

**Finding in accepted H-REF fixtures.** For the heterogeneous model, the accepted D2/D3
test fixture (`tests/test_checkpoint_storage._fixture`) declares `CategoryPath` rows as
`["Electronics"]`, `["Electronics","Audio"]`. The accepted graph builds them in sorted
order, `["Electronics","Audio"]` first. D2 checks row membership, not order, so an H-REF
heterogeneous checkpoint built from this fixture labels its category rows in the wrong
order. Restore is unaffected, because the same graph is rebuilt. The user approved a scoped correction on 2026-10-24:

- In `tests/test_checkpoint_payload.py`, only the non-empty `row_order` in
  `_make_valid_hetero_payload` changed, to the graph order. The D1 identity
  (`graph_nodes`) is a membership record and is unchanged.
- The file's SHA-256 changes from `41f70442...` (recorded in
  `configs/o-offline-implementation.json`, kept as history) to `1c28cbdd...`.
  `tests/test_checkpoint_cuda.py` pins the new value and names this correction.
- A regression test compares the fixture's row order with the accepted graph's order.
  It fails on the old fixture and passes on the corrected one.
- `checkpoint_fixture` now refuses any row-order difference instead of correcting it.
- Full suite: 574 tests, 0 failures.

### Resume scope

- **Epoch boundaries only.** A checkpoint is written after the epoch's validation and
  selection updates. There is no mid-epoch resume.
- **Same validated environment only.** Restore requires every fingerprint field, the
  bundle hash and the run spec hash to match. A mismatch is refused, with no relaxed
  mode. Continuing across a changed Kaggle image or driver would need a separate
  decision.
- **Restore in a fresh process.** Every tensor is validated on CPU first, then moved to
  the recorded device.
- **Exactness.** The claim is that a restored run equals the uninterrupted run bit for bit
  on the same environment. If phase-2 run 3 shows any difference, the claim is not made;
  no tolerance is added.
- **No other conversions.** No CPU-to-GPU or GPU-to-CPU restore, and no conversion
  between v1 and cuda-v1.

**Finding: fresh-process resume was impossible (found 2026-10-24, after run 1).** Test 11
below asks for restore in a new process. `restart_case()` restored in the same process,
so run 1 did not test it; the user's review of run 1 caught the gap. The new
`fresh_process_restart_case()` runs `straight`, `first` and `resume` as three processes
(`scripts/gpu_ready_child.py`). Its first CPU run refused the resume with "process global
generators differ from the checkpoint". Cause: PyTorch seeds the global CPU generator
randomly in every new process. The checkpoint records the global states, and
`restore_cuda` refuses unless the process holds exactly those states, so no new process
could ever resume. Fix, keeping the refusal: `gpu_runtime.pin_global_generators()` seeds
the global generators to `GLOBAL_RNG_SEED = 0` at the start of every process that writes
or resumes a checkpoint. Training never consumes them (checked every epoch), so every
such process holds the same states, and the refusal still catches anything that does.
After the fix, both models resume bit for bit across processes on CPU.
`tests/test_gpu_ready_parity.py` shows that two new processes differ without the pin and
match with it. Kaggle run 2 tests the fix on the GPU.

### Tests

Local, CPU only. The CUDA parts are faked with recorded byte tensors and a fake
fingerprint source:

1. Publish and retrieve round trip; every section is equal.
2. A v1 file refused by the CUDA reader, and a CUDA file refused by the v1 reader.
3. Each fingerprint field changed one at a time is refused.
4. A malformed CUDA RNG record is refused: wrong dtype, length, device index or count.
5. Tensors on a non-CPU device in the payload are refused, using meta-device stand-ins
   as D2 does.
6. A mid-epoch or non-quiet sampler state is refused.
7. Changed D1 identity fields, such as seed, device or fanout, are refused through D1.
8. An oversized file or tensor extent is refused before decoding.
9. Publishing to an existing name is refused.
10. D1, D2, D3 and H28 hashes are unchanged and the full suite passes.

On Kaggle (phase-2 run 3, one T4, both models, tiny fixtures):

11. Uninterrupted for E epochs compared with k epochs, checkpoint, restore in a new
    process, then E minus k epochs. Parameters, Adam state, all RNG states, validation
    history, best model and top-10 lists are compared bit for bit.
12. The CUDA RNG state is unchanged across each epoch.
13. After restore, every model and optimizer tensor is on `cuda:0`, and the fingerprint
    is in the run report.

## GR07: versioned result and record schema

- New H result schema `amazon4u-H-result-cuda-v1` and O record schema
  `amazon4u-O-record-cuda-v1`, selected by the run profile.
- Their remaining-gate list is a new constant, for example `GPU_READINESS_GATES`, in a
  new module. It omits the GPU-READY line only when that run's GR evidence is attached,
  and keeps the other four gates.
- `READINESS_GATES`, every `v1` schema and all `v1` validation stay byte-identical.
- O collection dispatches on schema version. A `v1` result can never satisfy a `cuda-v1`
  check, and the reverse also fails.

## Frozen files this design does not touch

The H28 bundle (including `training_harness.py` and `training_entrypoint.py`),
`differentiable_models.py`, `checkpoint_payload.py`, `checkpoint_storage.py`,
`recommendation_evaluation.py`, the O adapters and both O test modules. The one possible
exception is the H28 call list under GR01, which needs its own scope first.

## Decisions needed

1. Agreed: option C, and EX01 as a profiling prerequisite.
2. Agreed: the new profile, schema and checkpoint names.
3. Agreed: the GR06 design, with the two mandatory rules.
4. Agreed: full neighborhoods for Musical_Instruments; Electronics after profiling.
