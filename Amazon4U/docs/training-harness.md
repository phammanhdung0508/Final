# Existing H-REF — bounded CPU reference correction

## Status and authority

This is an additive implementation candidate and targeted correction, not an
execution receipt or acceptance. The Worker used no tools and ran no tests.
All three earlier responses, candidates, reports, ESCALATEs, D3 history and
inherited evidence remain immutable. This separately approved fourth response
neither resets budgets nor authorizes application, imports, probes or execution.
Pi independently reviews before any application; regression execution requires
separately released applicable authorization and fresh review.

No accepted scientific module, configuration, test, D1 allowlist or D2 schema is
modified. All original H1-H9, HC and matrix obligations remain assigned.
Deployment is NOT_AUTHORIZED.

The callable entrypoint implements recommendation training **from scratch**, not
LM pretraining, SFT, fine-tuning or automatic promotion. Only explicitly authorized,
owned synthetic CPU packets are accepted. `main()` refuses CLI/host acquisition.
No shell, credentials, network, subprocess, workers, thread tuning, global RNG,
default-dtype, determinism or security setters, native Torch checkpoint archives
or automatic host source capture are introduced.

## Boundaries and independent expectations

`TrainingData(actual_identity, expected_identity=..., resource_policy=...)` uses
unchanged D1 validation after bounded D2 resource preflight. Expected identity
comes from the trusted caller, not the checkpoint under examination. The adapter
retains complete training inputs and actual validation records; only training
records reach TrainingContext. All-rated history remains separate from condition
positives. Metadata is limited to canonical CategoryPath and Brand IDs. Duplicate
training pairs, train/validation overlap, missing actual validation records,
inconsistent targets and excessive populations are refused. The adapter does not
acquire real datasets or prove caller-supplied bytes originated synthetically.

A Harness requires separate identity, provenance, model row/buffer, runtime,
resource and D3 IO expectations. The actual run envelope is compared with a
separately expected envelope. Runtime versions are compared with current
Python/NumPy/Torch versions without environment changes. The fixture runtime is
`synthetic_profile_v1`, zero workers, CPU float32, `filter_before_batch` and exact
nonfused Adam options from independent runtime descriptors. Recorded CPU-thread
settings are declarations, not observed allocation or requests to change threads.

### External project closure

`bundle_manifest` describes caller-supplied bytes only. Its explicit 28-path
external manifest includes all twenty registered D1 source/config paths, accepted
`checkpoint_payload.py` (which D1 does not cover), `checkpoint_storage.py` and all
six H files. Thus identity, models, graph construction, recommendation core,
evaluation, D2, D3 and H runtime code are included independently of D1's schema.
The conservative registered source/config superset also retains auxiliary
registered files; describing their bytes does not import or execute them.

`verify_bundle` compares the entire manifest with an independently retained
expectation. The envelope separately binds its digest to run, approval, method,
input identity, initialization archive, device and fixture policy. Tests declare
expected closure membership independently of the production constant and compute
expected per-file lengths/hashes directly. Positive complete initial capture and
native staged restore controls precede missing/changed-D2 negatives. Only D2
closure bytes differ in those negatives; all other bytes and retained expectations
remain fixed. Fail-on-call model/optimizer/update/native-RNG seams ensure those
refusals precede initialization, update and native restore.

The two-file registered-source IO fixture tests only the unchanged D1/D3 API; it
is not full runtime closure. Synthetic packet hashes do not establish actual
installed/deployed source origin. A later trusted packager must acquire and
independently verify actual closure bytes under separate authority. There is no
physical collector, host snapshot, third-party source vendoring or D1/D2 amendment.

`owned_digest` is an internal semantic SHA-256 helper for owned/preflighted values.
It includes container types and tensor dtype/layout/shape/exact bytes. It is not a
physical D3 file hash, signature, authentication mechanism or untrusted decoder.
Publication receipts carry this semantic digest, which must be retained through
an independently trusted channel. Physical artifact hashing/collection remains O.

## Initialization and private RNG

LightGCN requires explicit small owned CPU float32 tensors. The complete initial
model state, including independently expected adjacency, is bound by the envelope.
This does not select a production initialization distribution.

Heterogeneous initialization invokes the unchanged seeded constructor: Normal ID
embeddings, Xavier transformations and zero biases. Its initial state is separately
archived/digest-bound. Its internal initializer is not replaced or exposed as a
continuing generator. Continuing RNG roles and aliases come from independent D2
runtime expectations; unused streams remain represented.

Private objects are created per generator ID. Python MT, NumPy RandomState MT,
NumPy PCG64 and Torch CPU states use accepted D2 codecs. Aliases refer to the same
object. Epoch shuffle and negative draws use their configured private streams;
the latter adapt uniform integer draws to the accepted sampler's choice API.
Empty pools consume no negative draws. Heldout information enters neither sampler
nor graph. Private native setters run only after complete unchanged D2 validation.

Private-RNG CPU fixture policy does not claim to enable global deterministic
algorithms or prove all-operation/cross-environment determinism.

## B1: narrow H-local CPU Adam compatibility guard

Installed Adam's ordinary step calls its CUDA-graph health hook. On the supplied
CUDA-built eager framework, the original hook can call CUDA availability, whose
default device-count route may initialize the driver API. This is an ancillary
framework compatibility issue, not a recommendation-model or Adam-formula defect.
Driver discovery is not itself a tensor allocation, GPU kernel or proof of a new
CUDA context. Nevertheless no driver-discovery/native-probe boundary is expanded,
and no actual target-driver behavior has been observed in this phase.

`HCPUAdam` inherits installed Adam construction, step arithmetic, lazy state,
parameter IDs and state_dict/load machinery unchanged. Its sole overridden method
is the CUDA-graph health hook. That hook performs CPU precondition checks instead
of availability/capture queries: all parameters, gradients and existing state
tensors must remain CPU, including state entries for inactive parameters.
The exact approved defaults and group options are mandatory. Capturable, fused,
differentiable, altered foreach/other options, AMP attributes and compiling modes
are refused. There is no global Torch/CUDA patch or second Adam implementation.

CPUBackend constructs this adapter and verifies exact model-parameter object
binding. Before an update it checks compatibility and binding before clearing
gradients, calling loss/backward or creating lazy state. The inherited step invokes
the same hook after backward. Fresh native optimizer loads receive the guard too.
Accepted model CPU checks and all Adam mathematics remain unchanged.

Scoped test tripwires raise on CUDA availability, native device count, stream
capture and the original base optimizer health hook. They surround both-model
fresh construction/first updates and actual D3 saved/restored continuation.
They are fail-on-call mocks, never successful no-op CUDA stubs. Tiny meta-device
parameter/state and incompatible-mode negatives require refusal before arithmetic
or model calls; valid CPU parameters, existing gradients and lazy state remain
unchanged. No CUDA tensor or hardware discovery is used by those negatives.
Baseline tripwires patch the actual H adapter, CPU optimizer factory and update
seam as well as the base Adam symbol, avoiding the pre-defined-subclass loophole.

These are proposed future controls, not executed evidence. Pi must still inspect
framework import/dispatch reachability; the local hook does not prove every
framework side effect safe or grant permission to run anything.

## Epoch, evaluator and selection semantics

Each epoch shuffles and retains all condition-positive pairs P. The accepted
NegativeSampler processes their order and K empty-pool skips. Trainable triples
are filtered before batching: updates equal ceil((|P|-|K|)/B). T=0 refuses;
empty batches never reach loss or optimizer step.

Each nonempty batch calls accepted fresh `compute_batch_loss`, backpropagates
`losses['total_loss']`, checks finite active gradients and performs actual Adam.
Ranking and endpoint regularization are not reimplemented. Adam remains 0.001,
(0.9,0.999), epsilon 1e-8, zero weight decay, no scheduler/AMP. Gradients are cleared
with set_to_none=True before and after updates. Inactive parameters retain None
gradients and lazy state; capture reads actual per-parameter steps. LightGCN keeps
its full positive graph; heterogeneous masking stays inside accepted loss calls.

Validation uses recommendation_core.Evaluator and actual validation records with
full ranking, unchanged eligibility, shortages, fallback, metrics and populations.
Unknown users are explicitly tested as excluded, with no per-user scoring record.
A separate rating_ge_4 control has a KNOWN user with only low-rated training
history and a relevant, warm, unobserved validation target. The authoritative
Evaluator supplies popularity fallback and must not call the provider. No positive
training interaction is fabricated for that user and no optimizer is needed.

Selection uses only overall_metrics['NDCG@10']; unavailable/all-N/A/nonfinite scores
refuse. Strict improvement creates an isolated best snapshot; ties retain earliest.
Patience is ten validation checks, maximum is 100 epochs, including the combined
terminal reason. Pure traces test boundaries without 100 training epochs. Fixture
completion is not convergence.

CorePopularity and KG-only retain separate deterministic routes with no optimizer.
KG-only uses the unchanged ordered grid and selected full report. No artificial
baseline fitting, test-based selection, refit, bootstrap or ensemble is added.

## Checkpoint and restart

`capture_epoch_checkpoint()` produces the exact fourteen-root D2 payload from
owned model state, actual Adam, private RNG, source order, skips, counters and
selection, then returns unchanged D2's owned validation result. Only initial or
successful quiescent epochs qualify. Training/validation failures poison the
handle; they are not rolled back, checkpointed or reported as success. A budget
refusal before an epoch leaves the handle unchanged.

Publication/retrieval call unchanged D3 with caller-approved roots and policies.
New restart controls use approved flat `.a4ucp` basenames. Successful publication
AND retrieval precede collision assertions. Collisions require the actual
CheckpointSecurityError OS-wrapper path and EEXIST cause; retrieval afterwards
checks that the existing payload remains intact. No generic Exception assertion
stands in for this guard. Registered source acquisition stays within D1's exact
allowlist; attempted H extension requires CheckpointSecurityError from that path.

CheckpointValidationError belongs to CheckpointError, not ValueError. D1/D2
structural/content/RNG/runtime/Adam negatives use its intended validation family;
H boundaries use HarnessError, accepted sampler/context refusals use ValueError,
and D3 security controls use CheckpointSecurityError. Valid controls precede the
mutations. Missing actual validation records are tested with a rebuilt canonical
identity so stale fingerprints cannot masquerade as the H records guard.

Restore checks independent envelope, current supplied source closure/runtime,
complete D2 payload and separately retained digest before constructing fresh
context/graph/model, adapter and private RNG objects. Native setters affect fresh
objects only. Full staged capture must match saved semantic state before returning
a replacement. LiveSlot's sole assignment follows that return. Late failures
therefore cannot partially mutate the active trajectory. Both-model actual
uninterrupted versus saved/restarted controls remain, including next draws,
order/losses, parameters, moments, lazy counters, selection and frozen validation.

No mid-epoch, GPU, cross-device or cross-environment continuation is claimed.
Latest evaluation reports are recomputed from selected frozen state, not stored as
new D2 fields. D3 publication failures may leave a completed published artifact
under the existing contract; callers must not ambiguously resubmit.

## Device/result seams and remaining obligations

CPUBackend owns capability checks, construction, numeric calls, evaluation,
snapshots and native transfer. The controller does not move tensors or bypass
accepted guards. Device/accelerator mismatches and unsupported capabilities refuse
before initialization/update; there is no silent CPU fallback.

The exact H run-spec allowlist requires separately expected specification and
explicit authorization. It contains no paths, credentials or test mounts.
`execute_fixture` returns a versioned in-memory result with truthful status,
actual updates, selection/report, capability/envelope and remaining gates.
Errors propagate. It is not a Kaggle CLI, runner or collector.

Bounds remain 8 users, 12 products, 8 nodes per metadata type, dimension 64,
CPU float32, 96 training records, 8 validation records and 512 directed edges.
Each file/decoded checkpoint is bounded at 2 MiB; supplied closure totals at most
8 MiB. D2/D3 retain their additional accounting and IO bounds. Each numeric
trajectory, including resumed progress, is capped at four actual Adam updates;
this does not change scientific max_epochs=100 or patience=10.

Static new full-suite Adam accounting is exactly **36**: first-step oracle 2;
both-model uninterrupted/prefix/restarted comparison 16; empty-pool epoch 2;
heldout branches 4; late-restore test 2; entrypoint 4; budget/evaluator failures 6.
All other methods, including new compatibility/closure/fallback controls, use zero
actual updates. Docstring and report match this enumeration. No hidden training or
inherited TestCase imports are added. The 128-step suite ceiling is not reset.

Physical fixtures remain private TemporaryDirectory(dir=tempfile.tempdir) trees,
0700 directories and 0600 files, with context cleanup and no external sentinels.
There are two concurrent checkpoint files per model branch and two small D1
acquisition files. Synthetic closure bytes are in memory, not host captures.

Pi must preserve all 389 inherited IDs, 3564 expanded cases, 34 D3 methods and
accepted bytes, independently inspect complete code/native/test safety and
adjudicate every original criterion and assigned limitation. No execution is
released by this correction. Any later authorized full regression must retain the
pinned complete discovery route and existing time/RSS/stream/scratch limits;
no subset, import probe, compile run, retry or budget reset is implied.

H-REF is a CPU-reference milestone only. GPU-READY, O-OFFLINE transport/packaging,
actual environment verification, production LightGCN initialization, full-scope
Musical_Instruments profiling, Toys/Electronics feasibility, comparative freeze
and final-test access remain separately required gates. There is no real-data,
CUDA, remote, profiling, deployment, main-transfer, commit or push authority.
