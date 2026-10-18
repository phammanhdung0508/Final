# Training Harness: preparation and bounded implementation proposal

## Authority, status, and evidence

**PREPARATION ONLY; implementation, Worker requests, tests, training, profiling,
Kaggle calls and legacy replacement are NOT AUTHORIZED by this document.**

User-agreed principle: **“Reuse orchestration, not inherited training methodology.”**
Exact current request: `data/reports/training-harness-preparation-v1/inputs/authority.txt`.
Planning HEAD: `d2938ba20ccc5cf5281197b2ef08f571dd14eae6`.
Requirements/evidence matrix: [training-harness-requirements.md](training-harness-requirements.md)
and `data/reports/training-harness-preparation-v1/requirements.json`.
Machine-readable proposal: `configs/training-harness-preparation.json`.

D1 identity/provenance, D2 decoded-payload validation, D3 secure IO, model/reference
and evaluator implementations remain authoritative, unchanged dependencies. D3 main
integration has task-only ACCEPT with389 tests; source decision:
`data/reports/checkpoint-storage-main-integration/review.json`. That is inherited
benchmark evidence, **not a fresh harness baseline**. Old4 Worker attempts/9 D3-associated
full runs, failed variants and ESCALATEs remain history, not a new budget pool.
The two D3 isolated trees remain retired/audit-only; do not execute or repurpose them.

This preparation reads code/docs and hashes existing files. It performs no candidate
imports, numerical experiments, regressions, bundle hydration, CLI preflight/dry-run,
credentials lookup, network call, profiling or training. Source findings below are
**static findings**, not executed reproductions or proof of Kaggle compatibility.

## Actual methodology and stage names

1. **Package and input verification**: approved source/data/config fingerprints,
   environment compatibility and explicit execution permission, not model fitting.
2. **Recommendation training from scratch**: LightGCN and Heterogeneous KG+GNN,
   each per category × feedback condition × registered seed. No pretrained model.
3. **Validation-only checkpoint selection**: frozen evaluator, strict NDCG@10
   improvement, earliest exact tie, patience10 checks, maximum100 epochs.
4. **Frozen-model final test evaluation**: separately authorized after execution/tuning
   freeze; no train+validation refit or test-based selection.
5. **Post-selection aggregation/uncertainty**: fixed per-user outputs, three-seed
   sample SD and registered paired bootstrap. No retraining/reselection here.

`resource_profile` is a separately authorized measurement mode before comparative
training; it is not fine-tuning, a new objective or a license to tune on test.
Popularity/KG-only fit deterministic training-derived statistics; they do not run
Adam or manufacture three identical seeded fits. No self-supervised pre-training,
SFT, reward training, LM loss, val_bpb promotion or inherited eval_score definition.
Physical legacy directory names do not define Amazon4U methodological stages.

## Reuse and targeted adaptation—not another control plane

Use the **existing** `multiautoresearch/post-training/scripts/kaggle_job.py` as the
single selected managed-runner base. It already has bundle/render, launch, status,
logs, output and job-record functions. Its historical `post-training` pathname is
retained for compatibility, not as a fine-tuning label. Do not implement a third
standalone job scheduler, agent dispatcher, promoted-master system or remote daemon.

Propose one explicit Amazon4U profile adapter:
`multiautoresearch/adapters/amazon4u.py` (plus package initializer). Give the existing
runner explicit profile hooks for project/source root, manifest, entrypoint, typed
argv, required artifacts, metrics schema and terminal-state interpretation. Do not
achieve adaptation by mutating module-global ROOT or monkeypatching model guards.
Amazon4U operations must select `--profile amazon4u` and an approved run specification;
missing profile/spec fails, rather than falling through to legacy train/evaluate.py.
The existing legacy profile, if retained, must have isolated state and no behavior
change except deliberately reviewed transport fixes. No existing legacy job is launched
as a regression test. Freeze this dispatch interface before the two work packages.

Reuse audited transport concepts/functions (resolve CLI, push, status, logs, output,
job receipts) with targeted fixes; **do not import inherited training methodology**.
The pre-training runner's promoted-master/diff categories/val_bpb path and
post-training MovieLens/LM train.py/model.py/evaluate.py/GEMINI agents are not run
or imported by the Amazon4U profile. No changes to pre-training/inference algorithms,
autonomous loops, credentials machinery, provider routing or the external CLI repository.

### Static transport findings requiring closure before use

| ID | Actual source | Finding / proposed disposition |
|---|---|---|
| L1 | both runners `build_kernel_metadata`, `launch_job` | timeout stored/passed to an unused metadata argument, but push argv omits `-t`; pass an explicit validated CLI timeout and test rendered argv |
| L2 | post runner generated `main` | evaluation nonzero is printed, but final return uses training rc; missing evaluation/final_model may also be skipped; Amazon4U requires every declared stage and nonzero failure propagation |
| L3 | both `poll_and_fetch` | unbounded polling, CLI errors ignored, substring terminal detection; parse exact supported statuses, bound attempts/deadline, distinguish unknown from terminal error |
| L4 | both `launch_job`, `resolve_kaggle_username` | dry-run resolves identity/credentials and writes last-job state before push; Amazon4U render-only must be pure/offline with supplied synthetic owner and no active-job mutation |
| L5 | post `collect_project_files`, pre equivalent | legacy files/root/src/models discovery misses authoritative Amazon4U scripts/configs; replace Amazon4U collection with explicit bounded source manifest; no recursive host capture |
| L6 | generated `install_dependencies` | install errors only warn; no frozen dependency/environment proof; fail before training on install/version/determinism/capability mismatch |
| L7 | `parse_metrics`, `fetch_output_impl` | scalar regex and warnings allow missing metrics; require typed complete run/result/artifact schema and registered Amazon4U metrics, not eval_score/val_bpb |
| L8 | `LAST_JOB_PATH`, output `-o`, `persist_job_state` | mutable implicit latest/slug collision and overwrite-prone collection; bind immutable run/attempt/version and fresh output directories; no accidental cross-run reuse |
| L9 | generated training/eval argument handling | free-form `.split()` is not a typed argv contract; use validated argv arrays, no shell or inherited arbitrary experiment command |

These are not reasons to rewrite both full runners or to execute the legacy programs.

## Authoritative components and implementation seams

**Protected**: every existing `scripts/*.py`, existing tests, accepted D1–D3 docs,
scientific configs/frozen manifests and prior reports. Only the specifically proposed
managed-runner path may change under later approval; other production additions are
new adapters/harness files. Never relax CPU guards or edit accepted tests to fit a harness.

Proposed new files (not created by preparation):
- `scripts/training_harness.py`: CPU epoch controller, exact sampling/update/selection
  and capture/restore integration; imports authoritative model/evaluator/checkpoint APIs.
- `scripts/training_data.py`: explicit full-split/KG input adapter, training-only
  context and separate evaluation targets; counts/fingerprints and no sampling shortcut.
- `scripts/training_entrypoint.py`: strict run-spec/authorization dispatch and exit code;
  no default fitting action, credential discovery or generic shell execution.
- `multiautoresearch/adapters/{__init__.py,amazon4u.py}`: packaging and run/artifact
  profile for the selected existing runner, not another training or evaluation engine.
- `tests/test_training_harness.py`, `tests/test_training_data.py`,
  `tests/test_kaggle_orchestration.py`: additive synthetic/offline evidence.
- `docs/training-harness.md` and framework operator documentation; any instruction-file
  replacement is separately scoped, not silently authorized by this proposal.

No new GPU backend, sampling kernel, bootstrap engine, UI or native agent-control
system is bundled into this first implementation. Existing cold-start/optional-MF/
metadata-free-control exclusions remain unchanged. Frozen final-test/postprocessing
entrypoint contracts are defined, but cannot execute on real held-out data in tests.

### One handoff, two substantive work packages

**H — training/state/data integration**: the three new scripts, their additive tests
and harness API documentation. Preserve existing numeric implementations; real tiny
CPU updates only if the future test policy is explicitly approved.

**O — managed orchestration integration**: targeted selected-runner changes plus
Amazon4U profile and offline transport/artifact tests. Reuse existing transport,
not H's model/controller logic. H and O have a frozen run-spec/result interface;
O follows H acceptance or consumes a frozen fake entrypoint until joint verification.
No helper/security micro-tasks or restart of D3 correction budgets. Joint final
verification is integration, not a replacement Worker deliverable.

## Proposed run contract and execution behavior

Strict versioned operator-supplied run spec: run/attempt ID, mode, protocol/manifest
and execution-revision IDs, method/category/condition/seed, source bundle hash,
input version/fingerprints/counts, expected model/runtime/identity/provenance,
explicit IO/D2 resource policies, initialization archive reference, batch policy/size,
validation schedule, device/thread/worker/determinism profile, time/RAM/disk/artifact
budgets, Kaggle owner/slug/version bindings and authorization evidence. No unapproved
null field becomes a default, guessed quota or checkpoint-derived expectation.
Profiling specs are provisional and distinctly labeled, not comparative execution freezes.

- Feed all training observations to existing TrainingContext; derive positive views
  without losing all-rated negative exclusions. Raw splits and approved KG features
  stay unchanged. Input chunking does not prove bounded whole-graph/model memory.
- Propose **filter_before_batch** for the first explicit execution profile, subject
  to approval: shuffle all condition-positive pairs P, retain full source_order,
  derive K (empty-pool pairs), batch trainable pairs T=P−K without replacing skips,
  use existing NegativeSampler and record |K|/ceil(T/B). Do not silently choose this
  over D2's also-supported batch_then_filter; the selected policy is identity-bound.
- Each valid update: zero grads, accepted fresh model loss/forward, backward, Adam
  step. No step/regularization-only update on an empty batch. Recompute differentiable
  propagation after updates; no stale graph/embedding cache or retained-gradient trick.
- Adam constant0.001, betas0.9/0.999, eps1e-8, weight_decay0; no scheduler, AMP,
  altered loss or transformations. Track actual parameter IDs/state step counts,
  including lazy state and inactive Hetero parameters—not global_step copied to all36.
- Use explicit owned RNG streams and recorded derivations, never process hash seeding.
  Worker processes/prefetch remain0 for v1 fixtures; explicit nonapplicability is recorded.
  Fail unsupported deterministic operations. No silently disabled determinism.
- Validate using the accepted evaluator, full warm candidates and frozen populations.
  An additive frozen-scoring adapter may cache ONE no-grad unmasked forward result
  for an immutable selected snapshot, using identical dot products/routes; compare
  every tiny score/rank with the authoritative scorer. Invalidate after any update,
  restore or graph change. Never reuse validation representations for training.
  Existing per-user score_candidates recomputes forward: do not assume it scales.
- Validation failure/nonfinite/absent eligible score is an error, not a patience check
  or score0. Exact ties retain earliest best snapshot. Best and current state remain
  distinct; terminal counters/status match unchanged D2.

## Packaging, Kaggle entrypoint and failures

Manifest-only bundling preserves `scripts/` import layout plus explicit required
configs/protocol snapshots/adapter entrypoints; it does not bundle `.git`, `.venv`,
credentials, data, other projects, worktrees, legacy model weights or whole directories.
Reject absolute/traversal/symlink/nonregular/duplicate paths; enforce source count and
compressed/uncompressed byte limits before hydration. No downloaded executable is
selected from arbitrary checkpoint metadata. Validate bundle hash and paths before use.

Inputs use operator-approved versioned Kaggle dataset/kernel sources and content
fingerprints. No upload, acquisition, scan of real raw data or external publication
occurs now. Train/profile bundle must not mount/use real test targets; later final-test
mode requires separate authorization. Kaggle source/version/mount-path mapping must be
explicit; dataset slug alone is not reproducible data identity. No silent fallback to
MovieLens downloads or LM prepare.py. Package/import verification is offline/fake-fixture
work only until remote execution approval.

Generated kernel runs `scripts/training_entrypoint.py` for one declared stage with
validated argv, explicit cwd and owned output root. Dependency install/preflight,
input mismatch, training, validation, checkpoint, determinism and result-manifest
failures all return nonzero. Always attempt bounded diagnostic/failure-artifact
recording without overwriting completed artifacts or hiding the primary error.
No broad exception→success, missing evaluator skip or optional success metric.

CLI reference is **read-only** `/home/sunf/my-project/kaggle-cli/`, pinned in the
preparation evidence (HEAD6894864ff4095cf98e9e36d3a694ff5fc9ba3818). Reviewed
`src/kaggle/cli.py`, `api/kaggle_api_extended.py`, `docs/kernels*.md`:
- push supports `-t/--timeout`, `--accelerator`, and `--no-run`; **even --no-run is a
  remote upload and is not authorized now**. Account quota/hardware are unverified.
- parse successful owner/slug/**version** receipt; reference API can print a push error,
  so subprocess exit0 alone is not sufficient acknowledgment. Missing/ambiguous receipt
  yields submission_unknown, never automatic re-push or a guessed latest job.
- status/output/logs can address a version; current reference includes kernels logs.
  Installed CLI version/capabilities remain unknown, so explicitly validate/record
  compatibility later. No unsupported-logs fallback is mislabeled as successful logging.
- local poll timeout does not prove remote job cancellation or stop quota usage.
  Record submission/running/complete/error/cancel/unknown distinctly, plus bounded
  polling/retrieval errors; fetch failure evidence where available without making a
  failed training job successful because diagnostics downloaded.

Only an authorized launch calls Kaggle. Render-only never resolves credentials,
contacts CLI/network, launches a subprocess or changes last-active-job state. Tests
inject fake CLI/process/clock responses; do not run the external CLI's server-facing
test suite or any installed kaggle command during this preparation/initial regression.

## Frozen handoff interface proposal (approve before H/O implementation)

Use strict versioned JSON for operator/run/transport records, not arbitrary Python
objects or executable plugins from artifacts. Worker may choose internal dataclasses,
but may not omit required semantic fields or derive expectations from a checkpoint.
Reject unknown fields/versions rather than silently ignoring configuration mistakes.

`amazon4u-run-spec-v1` required sections:
- `identity`: run_id, attempt_id, method, category, feedback_condition, stage, seed;
  deterministic methods explicitly mark seed nonapplicable, not an arbitrary new seed.
- `authority`: approved mode/scope/receipt hash and permission kind; source is a
  separately operator-approved immutable input. A Worker-written approved=true flag
  or artifact-carried hash alone is not authority. Fixtures use explicit fake approvals
  whose scope cannot launch a real job; protected approval records stay nonwritable.
- `protocol`: frozen version/manifest hashes, lifecycle, execution revision and
  approved execution-profile ID. Null execution revision is legal for labeled synthetic
  fixtures only, not comparative fitting.
- `inputs`: explicit versioned sources/mount paths, split/content/KG/warm-pool hashes,
  expected counts and permitted metadata; train/profile test input must be absent.
- `execution`: CPU/dtype/thread/worker settings, batch policy/size, schedule, optimizer
  options, sampling, RNG derivation/roles and determinism requirements. Missing real
  initialization/device/capacity approval refuses launch, not an implicit default.
- `initialization`: approved policy, row-order/parameter binding, owned initial archive
  identifier/hash; Hetero initialization follows accepted constants.
- `expected_checkpoint_contracts`: independently trusted D1 identity/provenance,
  expected_model, expected_runtime, resource_policy and io_policy, or immutable references
  with separately expected hashes; exact D2 schemas unchanged.
- `code`: complete bounded bundle/snapshot manifest reference/hash and accepted-component
  hashes. `limits`: explicit wall/poll/network/file/count/storage/RAM/disk limits and
  VRAM nonapplicability for CPU; no universal production capacities selected here.
- `transport`: explicitly approved owner/slug, unique run-attempt association, accelerator,
  CLI/dependency versions, source bindings and private/internet policy. Default Amazon4U
  packaging is private with internet disabled; dependency installation requires approved
  pinned offline wheels or a separately permitted network policy, never legacy pip defaults.

`amazon4u-run-result-v1` required sections: matching run/attempt/stage and code/input/
protocol hashes; status and actual exit code; completed epochs/actual updates/skips;
validation schedule/history/selected epoch; registered metrics/population/route records
when applicable; actual environment/resources; checkpoint/artifact manifest links; failure
stage and bounded safe diagnostics. Initial state permits best_checkpoint=null before
validation. Failed/partial runs never masquerade as complete fits or three-seed summaries.
Only final-test mode may emit actual test metrics under its separate permission.

`amazon4u-artifacts-v1`: exact relative paths, type, bytes, SHA256 and role for complete
source/config snapshots, initial/current/selected checkpoint where applicable, run/result/
input/environment/approval references, stdout/stderr and failure history. Bind these to
an independently retained submission receipt (owner/slug/version/bundle hash/run-attempt).
A self-consistent downloaded manifest alone is not authenticated origin. No extra D2
root keys or expanded D1 source allowlist are introduced by this external envelope.

Transport states: rendered → submission_pending → submitted(version-bound) → queued/
running → remote_complete or remote_error/cancelled; collection has its own complete/
partial/error status. submission_unknown and poll_deadline_exceeded are not remote
success/cancellation. Training/evaluation/checkpoint failure dominates a successful
artifact download; absence of required artifacts dominates remote_complete.

**Budget enforcement seam:** plain legacy `kernels output` followed by a directory
scan does not enforce a hard pre-download aggregate byte cap. Amazon4U collection must
use the existing CLI/SDK listing/auth/download transport with a targeted bounded stream
adapter that counts actual bytes/files/pages/deadlines into fresh quarantine, or fail
closed when that cap cannot be enforced. Do not claim a manifest-declared size or
post-hoc scan is complete mediation. No new authentication/client framework or sandbox
is proposed. Mock streamed chunks/HTTP errors/pagination/size lies in offline tests;
no real downloads or filesystem-security experiments. Likewise bundle hydration must
limit actual decoded bytes incrementally, not call unbounded gzip.decompress first
and check afterward. Use smaller-policy synthetic negatives, never actual bombs.

## Checkpoint and artifact integration

D3 APIs remain acquire_sources/publish_checkpoint/retrieve_checkpoint. Caller-approved
scripts/configs directory FDs must be independently established; direct-child lookup
preserves full D1 logical IDs. Do not rebuild the obsolete projectRoot/scripts bootstrap
or claim continuous project-path ancestry. Check filesystem/procfs/ABI compatibility
before fitting in a later authorized environment; unsupported facilities fail closed,
not a pickle, named-temp or privileged fallback.

Capture quiescent initial/completed-epoch CPU payloads: current/best, actual Adam IDs,
state/options/update counters, RNG roles/states/aliases, full source_order/cursor,
progress/skip counts and selection; scheduler/AMP explicitly nonapplicable. Reuse D2
before D3 publication and after bounded retrieval; independent expectations/policies
come from the trusted run spec. Live restoration is NEW harness functionality, not
previous D3 acceptance: stage/reconstruct a fresh model/optimizer/private RNG set from
fully validated owned data, never half-mutate an active training object. Failure leaves
existing live state untouched; no promised mid-epoch restore or cross-hardware bitwise
continuation. Test uninterrupted vs save/restart CPU trajectories independently.

**D1 provenance closure gap:** D1's20 registered logical paths intentionally exclude
checkpoint_storage.py, new harness modules and multiautoresearch paths. Do not add names
or fields to D1/D2 to make packaging pass. Package a separate bounded complete source
manifest and source snapshots, bind bundle/authorization/run IDs and checkpoint hashes
in an independently expected external run envelope, and verify it before restore.
A manifest downloaded with its own bytes is not independent authenticity proof. Do
not claim D1 alone attests new harness code. If canonical D1 identity itself must include
that closure, seek a specific schema/allowlist amendment first; it is not this proposal.

Collection downloads only the bound kernel version into a fresh owned quarantine,
never force-overwrites previous outputs. Validate relative filenames/types and all
pages/counts/bytes/hash expectations, then required run/result/source/input/checkpoint
manifest cross-links. D3 acquisition rules apply to checkpoint leaves. Do not unpickle
weights, execute artifact hooks or trust metrics parsed from a log line. Keep raw logs
bounded and credential-redacted; retain exit/failure/restart/partial-output evidence.
Validate missing/corrupt/incomplete artifacts as failure/partial, not successful run.
Only verified artifacts are marked collected/complete; no automatic promotion/deploy.

Durability caveat: Kaggle working output is ephemeral while a kernel runs. D3's atomic
link/file fsync is NOT proof of remote/directory crash durability. A durable permitted
checkpoint sink and interruption recovery policy must be approved/verified before
claiming recovery across terminated Kaggle sessions; no automatic upload of secrets or
checkpoints to a new service is authorized. Keep failures explicit if output unavailable.

## Blocking choices and delivery boundaries

| Gate | Current evidence / decision still needed |
|---|---|
| G1 CPU vs GPU | accepted check_cpu_tensor/model parameters/LightGCN COO explicitly reject CUDA; Hetero uses CPU temporaries and full Python-neighbor loops. First harness acceptance is CPU-only/offline orchestration, NOT GPU readiness. A GPU execution adapter/backend needs separate scope, mathematical parity/determinism evidence and approval; never bypass guards or select CPU silently for the planned GPU study |
| G2 initialization | LightGCN requires caller-supplied base embeddings; scientific initialization distribution remains unresolved. Tests use explicit tiny archived tensors only; real run fails missing approved initialization. Hetero Normal0/0.1, Xavier gain1, zero biases already frozen |
| G3 execution choices | batch policy, batch sizes, RNG derivation, threads/determinism, fanouts/sampling and final validation frequency require explicit recorded profile.1024/fanout10 are suggestions, not defaults. No sampled core LightGCN; new Hetero sampler is outside first CPU-reference harness |
| G4 Kaggle environment/data | allowed account/quota, accelerator, CLI/dependency/native-IO compatibility, immutable full-data mounts and durable sink unverified. No live preflight now |
| G5 provenance | external complete bundle/run envelope proposal above must be approved and independently tested; D1/D2 untouched. Resume source/runtime mismatch fails before live mutation |
| G6 execution authority | future synthetic optimizer/backward/restore tests, implementation/Worker budgets, exact roots/commands and240s proposal require explicit implementation approval. Real profiling/training/test remain separate |

Thus preparation can be ready for a **bounded CPU harness/offline orchestration**
implementation decision without pretending the full GPU research runner is complete.
Do not label known GPU/durability/source-closure gaps out-of-scope for overall study
readiness. At harness review, unresolved assigned CPU/transport guarantees block ACCEPT;
record the explicit CPU-only acceptance scope and remaining profiling prerequisites.

## Proposed budgets and independent verification (not approved)

- Fresh independent harness project beneath `.worktrees/training-harness-implementation-v1/`
  only after approval; pin then-current HEAD/source hashes and all accepted dependencies.
  Do not reuse retired D3 trees or create a replacement D3 task.
- H and O: one initial text-only patch response plus at most one consolidated revision
  each,30min/request,no automatic retry. At most4 new requests total; no Worker tools,
  source execution/tests, subagents, commits or controller/provider changes.
- At most6 full discoveries across this proposed implementation:1 fresh389 baseline,
  H initial+optional revision (2), O initial+optional revision (2),1 joint final (1).
  No subsets/smoke/import/native probes/compileall/repeated runs. Each **proposed240s**
  covers entire invocation; D3's240s authority does NOT automatically approve this stage.
  Main transfer is a separate later bounded integration, not a seventh implicit run.
- Proposed fixture envelope: CPU,float32,<=8users/12products/8metadata nodes per type,
  exactly64 dimensions; actual fixture file<=2MiB,concurrent<=8MiB/128files,decoded
  checkpoint storage<=2MiB. Miniature bundle<=8MiB unpacked/128files; use synthetic
  package trees, not live host/source/credential capture by tests.
- New actual numeric branch<=4epochs/4 Adam updates,all new tests combined<=128
  Adam steps per full discovery;at most5post-baseline discoveries give a phase ceiling
  of640 new actual Adam steps. Baseline389 runs no newly added update tests;inherited
  accepted differentiation exceptions are unchanged. Propose a2GiB peak process-RSS
  acceptance ceiling,64MiB newly generated
  scratch per full invocation and8MiB per complete stdout/stderr stream. Measure RSS
  only with a separately reviewed future parent-wrapper OS accounting receipt;this is
  not a new diagnostic/profiler run,global resource-limit setter or memory sandbox.
  Missing measurement is an unverified gate,not an invented passing counter;never
  truncate logs silently or retune accepted tests to fit. These limits are proposals,
  not an amendment to old D3 runtime-investigation/benchmark history.
  Patience/max-epoch controllers use injected traces (up to101 events),
  not100 real epochs; keep max_epochs100 semantic tests without expensive training.
  No new CUDA, worker threads/processes, network, remote CLI, optimizer on real data,
  global RNG/dtype/umask/security setters or unsafe decoding. Existing389 exceptions
  unchanged; the tiny new backward/Adam/private restore operations need explicit approval.
- Runtime time/RAM/VRAM/disk/checkpoint budgets are REQUIRED run-spec inputs, not
  invented production2MiB caps. Conditional60GPU-hours/week and20% reserve remain
  unverified planning, not spending authority. Numeric real-job/profile budgets remain
  null until a separate measured plan; failure must not trigger subsampling/tuning.
- Snapshot proposal:128MiB/file512MiB aggregate,60s/file300s/snapshot; explicit ignored
  new implementation/scratch/test-output paths, accepted D3 archived trees/reports,
  accepted D1/D2 evidence, Git HEAD/index/status and outside tracked/nonignored sentinels.
  Actual approval must pin paths/budgets; snapshots are not a sandbox/transient-write log.

Proposed sole benchmark form (NOT executed; re-pin path/argv before approval):

```text
cwd=/home/sunf/FSB/Final/Amazon4U/.worktrees/training-harness-implementation-v1/Amazon4U
/home/sunf/FSB/Final/Amazon4U/.venv/bin/python -B -c
import os,tempfile,unittest,sys; from pathlib import Path; p=Path('/home/sunf/FSB/Final/Amazon4U/.worktrees/training-harness-implementation-v1/Amazon4U/.test-output/training-harness-v1'); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); os.environ['TMPDIR']=str(p); os.environ['PYTHONDONTWRITEBYTECODE']='1'; suite=unittest.defaultTestLoader.discover('tests'); result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not (result.wasSuccessful() and not result.skipped))
```

Pi reads complete changed functions/callers/dependencies/tests and generated-entrypoint
source before any execution. Keep generated bootstrap thin and route into shared
ordinary runtime functions tested with fake process/CLI/clock/file interfaces;do not
execute embedded arbitrary generated Python or add subprocess smoke probes. Offline
packaging evidence is not observed successful remote generation/execution traffic.
AST/static scans supplement, never replace semantic
review. Independently inspect scope/preserved bytes and complete stdout/stderr/method
IDs/ledgers,128-step budget,fixture cleanup and full389 inheritance. Use exact positive
controls and boundary-specific tripwires, not jointly corrupted expected records.
Compare actual updates/Adam and restarted trajectories with independent expected
values, and mock CLI call/exit/status/artifact cases without actual account operations.

Before acceptance evaluate every H/O and joint criterion, security/integration and all
current Worker limitations exactly/in order against original approved scope quotes.
Any assigned unresolved issue, unsafe mandatory test, missing/stale/timeout/cancelled
verification means ESCALATE, not partial ACCEPT. Consolidated revision only before any
held-out exposure within budget; no new task/retry to evade exhaustion.
Acceptance is **task_only**, with explicit CPU/offline boundary; deployment
**NOT_AUTHORIZED** and no deployment_policy. Later deployment/generalization requires
a prospectively declared separate policy/one-shot evaluation; benchmark is not approval.

## Exit and next gates

Preparation exit: whole matrix, source references/static findings, protected hashes,
proposed H/O boundaries/criteria/budgets/controls, explicit open decisions and zero
execution disclosure recorded. This does not resolve G1–G6 or approve implementation.

After separate authorization: H/O implementation → Pi independent full regression /
ordered limitation adjudication → separately approved main integration. Only then,
and after device/environment/data/init/durability decisions, prepare the exact bounded
Musical_Instruments profiling execution request (full downloaded scope; validation
ranking every epoch first). Toys_and_Games then Electronics require feasibility checks,
not extrapolated acceptance. Comparative tuning/search counts/seed/reuse policy,
execution revision and final test policy must be frozen before comparative exposure.
No real profiling/training/final test is part of this preparation or implementation
benchmark, and old test splits are never reclassified as unseen.
