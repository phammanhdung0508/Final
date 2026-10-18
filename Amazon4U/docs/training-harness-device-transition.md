# CPU reference first; GPU readiness before large-scale profiling

## Discussion decision and authority

The user agrees with a CPU reference harness first to verify real updates, but asks
that this not become a permanent CPU restriction or leave H/O nominally complete
while large-scale experiments remain infeasible. Pi agrees. This clarifies the
preparation proposal; it authorizes no implementation, Worker, test, GPU job,
profiling or numerical/scientific change. Earlier proposal/criteria/review remain
preserved, including `data/reports/training-harness-device-transition-v1/inputs/`.

**CPU-reference verified is a milestone, not overall experiment readiness.**
Existing CPU tensor guards are current accepted implementation boundaries, not a
methodological rule that Amazon4U must always run on CPU. They remain unchanged now.
Training is from scratch on either device; orchestration reuse is not methodology reuse.

## Sequence and honest completion labels

1. **H-REF: CPU reference correctness.** Verify sampler/graph/loss, actual Adam
   updates, validation selection and epoch capture/restore on tiny fixtures using
   accepted models. This is an independently reviewable correctness milestone.
2. **O-OFFLINE: orchestration correctness.** Adapt existing managed Kaggle packaging,
   entrypoint, failure/status/logging/collection paths with fake transport. Model
   device and requested accelerator must be explicit, not hardwired to CPU or GPU.
3. **GPU-READY: required readiness gate in the same overall harness roadmap.**
   Assess targeted device adaptation and verify tiny GPU execution, CPU-reference
   parity, determinism, state transfer/checkpoint/RNG completeness and actual Kaggle
   package/environment compatibility under separately approved bounds. This is not
   automatically another Worker task or a reset of H/O or D3 budgets.
4. **PROFILE-READY / feasibility:** after the readiness gates and separate execution
   approval, profile full-scope Musical_Instruments time/RAM/VRAM, including full-ranking
   validation. Then recheck Toys_and_Games and Electronics. GPU correctness alone
   does not establish full-data feasibility or authorize comparative runs.

H-REF and O-OFFLINE can have task-only acceptance for their declared scopes; do not
call the whole research harness ready for planned GPU experiments at that point.
The existing G1 gate is required work, not an optional afterthought or a scope waiver.
No promising full-scale readiness before measurement; no silent CPU fallback, sampled
LightGCN, interaction/user/candidate subsampling or unregistered numerical changes.

## Design H/O to avoid a second harness

Proposed H/O acceptance supplements (not new execution authority):
- Controller and orchestration depend on an explicit execution/device capability
  interface, separate from optimizer/selection/sampler mathematics. Initial backend
  reports CPU support honestly; unsupported GPU requests fail before launch.
- Keep tensor placement, graph/buffer movement, frozen-score cache and state transfer
  behind narrow execution adapters. Do not scatter `.cpu()` or global default-device
  assumptions through the epoch controller. Existing D2/D3 serialization boundaries
  remain explicit and may correctly require CPU snapshots.
- O retains one runner/profile path and carries requested accelerator/device/runtime
  through bundle, entrypoint, receipt, logs and artifacts. Do not clone a second GPU
  scheduler or silently request a GPU while the model still executes on CPU.
- Freeze these interface seams during H/O preparation. Test capability dispatch and
  device/accelerator mismatch using fakes during the CPU/offline milestone; no actual
  CUDA execution is implied by those tests.

A GPU implementation may need device-aware tensor/index/sparse-buffer construction
and transfer, plus targeted changes to CPU-specific operations or Python-loop execution.
That is not necessarily a mathematical GNN rewrite. Start with the smallest compatible
adaptation, use the accepted CPU implementation as the oracle, and identify any operation
that truly requires a new kernel/vectorization. Do not promise `.to('cuda')` alone will
work, bypass guards, or preemptively rewrite the GNN. Changes to accepted production
modules require their own explicit scope and independent preservation/parity review.

## Proposed GPU-READY evidence (future, unexecuted)

- Both model families: tiny forward scores, loss, gradients and actual Adam updates
  compared against the CPU reference; preserve graph normalization/masking, endpoint
  penalty, optimizer settings and evaluation semantics. Record justified numerical
  tolerances; no cross-device bitwise-equality promise or new ranking tie epsilon.
- Explicit placement of every parameter, sparse/index/metadata tensor and temporary;
  same graph population and fresh caches. Compare uninterrupted vs restarted operation
  within the selected GPU environment, not an unproved cross-hardware continuation.
- Fail unsupported deterministic CUDA operations first; exceptions need explicit
  approval, never automatic disabling. Record GPU/CUDA/driver/library versions and
  actual accelerator allocation. A GPU label in metadata is not observed GPU execution.
- Checkpoint completeness includes GPU-relevant RNG/runtime state, not merely copied
  model weights. Current `checkpoint_payload.py` explicitly rejects an active CUDA RNG
  role in its CPU profile. Copying tensors to CPU is not by itself a complete GPU resume
  contract. Assess a versioned GPU state/runtime profile under separate approval if
  required; preserve the accepted CPU profile and D3 bounded storage guarantees.
- Packaging/imports/native IO/artifact/durable-checkpoint compatibility must be observed
  in the approved Kaggle environment before claiming it is remotely ready. Offline
  transport tests do not establish this. No remote preflight or test is launched now.

GPU worker/test/job budgets are **not yet proposed numerically or authorized** by
this clarification. The existing H/O4-request/6-full-suite CPU/offline proposal is
unchanged and still unapproved. Define the later device milestone's exact paths,
resource/time limits, approvals and evidence before execution; no budget borrowing,
automatic retries or changes to old criteria/reports.

Status: **preparation clarification recorded; implementation on HOLD**.
`acceptance_scope: preparation_only_no_implementation_ACCEPT`.
`deployment_decision: NOT_AUTHORIZED`.
