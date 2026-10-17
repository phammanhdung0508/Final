# Deliverable 3 — secure checkpoint storage/retrieval and D1/D2 integration

## Current authority and D2 conclusion

The user directed Pi to conclude D2 and prepare D3, with this explicit boundary:

> D2 validates checkpoint data after it has been decoded, while D3 is responsible for the secure storage/retrieval of checkpoints and their integration with D1/D2.

**D2 is concluded:** implementation and separate main integration are ACCEPT,
`task_only`. Main integration verified355 tests, zero failures/errors/skips,
54.139s unittest /56.643s process, within120s. The authoritative decision remains
`data/reports/checkpoint-payload-main-integration/review.json`.
No new D2 error has been identified in this preparation. Do not initiate another
D2 correction, re-audit, Worker round or security-controller investigation. D1/D2
source, tests, exact accepted payload documentation and historical reports remain
unchanged dependencies. If a concrete new dependency error is subsequently found,
record its evidence and impact before deciding whether any change is necessary;
D3 is not permission to rewrite an accepted dependency.

**Current authorization: documentation/specification preparation only.** This is
a D3 scope and evidence-plan draft, not a completed implementation specification,
Worker submission, fresh baseline receipt, implementation acceptance or permission
to execute checkpoint IO. Companion record: `configs/checkpoint-storage-scope.json`.

D3 remains **one substantive deliverable**: secure checkpoint IO plus its D1/D2
integration verification. Fixtures/helpers/corrections stay inside that deliverable,
not new micro-tasks. Historical task decisions and revision budgets remain intact.

## Responsibility boundary

| Component | Owns | Does not establish |
|---|---|---|
| D1 — accepted identity/provenance | Canonical identities, supplied source bytes and explicit Git/environment evidence, actual/expected compatibility | Physical acquisition of bytes merely because a record is internally consistent |
| D2 — accepted payload validation | Already-decoded state schemas, models/Adam/RNG/sampler/progress/selection, per-object logical resource preflight, independently owned validated output | Safety of decoding, file/path security, live restoration, historical origin or exact continuation |
| D3 — this deliverable | Approved-root physical IO and source acquisition, bounded serialization/decoding, atomic exclusive checkpoint publication, corruption/race refusal, owned temporary cleanup, D1/D2 pipeline integration | A model training loop, restoring live models/optimizers/RNG, a process-memory sandbox, or a general agent security controller |

The important ordering is:

**Read:** trusted root/path/policy → anchored open and file bounds → decoding-time
allocation/storage bounds and restricted decoding → D1/D2 validation with mandatory
independent expectations → return validated owned state, never a partially accepted
checkpoint or restored live object.

**Write:** mandatory trusted expectations/policies → D1/D2 validation of supplied
state → bounded serialization to an exclusively created owned temporary → complete
file/format checks → atomic **no-clobber** publication within the approved destination.
No invalid or partial checkpoint becomes the completed destination.

D1/D2 checks must not be replaced by D3-specific weaker schemas or expectations
manufactured from the checkpoint. Loading a file and only then invoking D2 is
**not** sufficient to enforce decoding-time allocation safety.

## Proposed product scope

Proposed implementation paths, to be frozen before submission:

1. `scripts/checkpoint_storage.py`
2. `tests/test_checkpoint_storage.py`
3. `docs/checkpoint-storage.md` — extend this record without erasing authority/history.

Three public operations are proposed; exact signatures remain to be specified:

- **Acquire approved source snapshots:** read a caller-specified explicit allowlist
  beneath an independently approved source root, bounded before reads/hashing; return
  the actual bytes used with existing D1 provenance APIs. No directory-wide capture,
  credentials, siblings, datasets, shell/Git invocation or invented host environment
  evidence. Git/environment records remain explicit caller-supplied inputs.
- **Publish a checkpoint:** validate state through accepted D1/D2 contracts, serialize
  within explicit IO bounds, and publish once without replacing an existing artifact.
- **Read a checkpoint:** retrieve from an approved root, enforce safe decoding/file/
  storage limits, then call accepted D1/D2 contracts before returning owned state.

The IO policy must be independently trusted and mandatory, separate from the D2
`resource_policy`. The checkpoint cannot choose either policy or enlarge limits.
No implicit root, default unlimited budget, free-RAM auto-sizing or production
capacity selection. D3 must align its accepted state/format with D2 rather than
altering D2 to make serialized fixtures pass.

## Required contracts and evidence mapping

These draft criteria map the already approved D3 boundary in
`checkpoint-three-part-recovery.md` and the D3 exclusion in `checkpoint-payload.md`.
They are not a new D2 contract or an excuse to widen checkpoint scope.

| ID | D3 requirement | Planned independent evidence |
|---|---|---|
| IO1 | Preserve accepted D1/D2, models,355 baseline tests, protocols, historical reports, Git metadata and siblings; only declared D3 files/outputs change | Pinned dependency references and before/after scope evidence; no D2 re-audit or source rewrite |
| IO2 | Narrow independently approved roots; reject escapes, unapproved paths/types, symlinks and unsupported objects; descriptor-anchored traversal and explicit parent-race refusal | Valid paths plus absolute/traversal/symlink/nonregular and deterministic parent-swap cases; no check-then-reopen pathname shortcut |
| IO3 | Enforce file bounds before decoding, and reconstructed-storage/allocation bounds during decoding; corrupt/truncated/unsupported formats fail closed | Literal small valid controls; independently crafted malformed files/storage declarations; exact and one-over small-policy limits; evidence that forbidden allocation/decoder paths are not reached |
| IO4 | Restricted loading with `weights_only=True` for the existing Torch-loading contract; no unsafe fallback, arbitrary globals/allowlist expansion, shell/eval or attacker hook execution | Unsupported/corrupt object and decoder refusal cases; complete decoder/caller/dependency review; weights-only is not treated as a storage/RAM bound by itself |
| IO5 | Atomic exclusive no-clobber publication; bounded writes; cleanup only owned incomplete temporaries | Existing destination unchanged on collision, independent interrupted/failing-write and competing-publication controls, no partial destination, no unrelated deletion; atomic visibility distinguished from any separately specified durability promise |
| IO6 | Allowlisted physical source acquisition supplies actual bounded bytes to D1, without broad host capture | Tiny approved synthetic source tree and changed-byte, forbidden-path, symlink, missing-file and bound refusals; existing D1 validation reused |
| IO7 | Full serialization → retrieval → decoding → D1/D2 validation pipeline, mandatory independent expectations, owned returned data and no live restoration | Both supported model families and coherent initial/epoch-boundary mappings; wrong trusted expectations, corruption and invalid payload refusals; complete unchanged baseline plus D3 cases |
| IO8 | Bounded meaningful full regression, complete streams/report, truthful API/limits documentation and Pi acceptance | Predeclared full discovery≤120s, no failures/errors/skips or weakened assertions; complete changed functions/callers/dependencies/tests read; ordered limitation adjudication; incomplete evidence blocks ACCEPT |

Shared guards may support more than one case; do not manufacture an unbounded
Cartesian coverage target, new arbitrary numerical caps or origin/continuation proofs.
A refusal fixture must have a genuine valid control and reach its intended boundary.

## Details to settle before implementation submission

These are **D3 preparation work**, not new controller research or D2 auditing.
Do not let a Worker silently choose them while implementing:

1. **Format and bounded decoder.** Specify the supported on-disk envelope/version,
   tensor/storage representation, restricted-loader options and how untrusted declared
   sizes are bounded **before allocation**. File length alone, archive-member sizes
   alone, `weights_only=True` alone, and post-decode D2 checks are not interchangeable
   protections. Review the relevant pinned framework APIs statically. Preserve the
   existing restricted-loading requirement; a format change needs an explicit recorded
   decision, not an unnoticed unsafe fallback. Specify corruption/integrity claims
   separately from authenticity: a checksum stored with attacker-controlled bytes is
   not independently trusted proof of origin. If the proposed path cannot meet the
   promised bounds, identify that blocker before authorizing execution.
2. **Exact IO policy and accounting.** Specify required fields, strict types, byte and
   reconstructed-storage limits, container/archive structural bounds where applicable,
   alias/view handling, read/write counting, and their relation to D2's M/S/C policy.
   Do not reuse D2's2MiB logical budget as proof of a2MiB serialized file or total RAM
   bound. No production limits are selected in this draft.
3. **Root/path/platform contract.** Specify approved-root bootstrap, supported platform
   primitives/filesystems, component traversal, final open/publication operations,
   permissions and descriptor ownership. State exactly which namespace-mutation
   adversary is handled and which root-ownership/concurrency assumptions are required.
   Holding a directory descriptor is not by itself proof of every claimed parent-race
   guarantee. Do not silently weaken the assigned symlink/parent-race refusal contract.
4. **Publication/failure contract.** Choose the actual no-replace atomic primitive and
   its supported-file-system checks; specify owned temporary identification and cleanup
   on each failure. Separate atomic visibility from crash durability; do not invent a
   crash-recovery guarantee or destructive rollback. Do not use a race-prone existence
   check followed by an overwriting rename.
5. **Exact APIs and test setup.** Pin signatures, exception family, format/policy schema,
   path limits, synthetic controls, fault-injection hooks and resource oracles. Select
   an isolated checkout, baseline355, exact executable/argv/cwd/scratch, output paths,
   dependency hashes and bounded response/revision/execution budgets before any Worker
   or candidate execution. No stale HEAD or old benchmark is a fresh D3 baseline.

## Prospective execution boundaries — not execution permission

Retain the existing correctness-fixture envelope: CPU only, at most8 users,
12 products,8 metadata nodes/type,64 dimensions; new fixture files≤2MiB. Synthetic
Adam mappings/private RNG states only, **no optimizer steps/backward/training**, live
restoration, worker processes, CUDA, real datasets, remote jobs or held-out evaluation.
The existing accepted baseline's differentiability/import-order tests remain unchanged.

Oversized-file refusal can be demonstrated against a smaller explicit test IO budget
while the actual fixture still fits≤2MiB; D2's narrowly approved logical negative
exceptions do not automatically authorize larger D3 physical files. Aggregate fixture
storage, decode quotas, structural limits and exact small negative exceptions must be
specified before execution. No production RAM/serialized-file capacity claim follows.

Tests may later serialize/decode only after separate D3 implementation/test approval,
using fresh declared synthetic scratch trees and controlled filesystem fault injection.
No live source capture or filesystem attack/probe is authorized during preparation.
All ignored accepted dependencies/evidence that matter must be explicitly monitored;
`.worktrees/` ignore rules do not exempt them from future verification.

Reuse the already established text-only patch channel when implementation is authorized;
no tool-controller, authentication, sandbox or Worker-architecture detour. Pi alone
reviews/applies and executes prospectively approved commands. Keep one whole-D3 task
and consolidated revisions under its approved budget; never reset old tasks or create
security-fix micro-tasks to evade the bound. Main transfer/regression remains a separate
approval gate after D3 acceptance, not an additional implementation deliverable.

## Status and next gate

**D2 concluded; D3 preparation started.** No D2/D3 code, baseline/test run, serialization,
loading, source acquisition, Worker, deployment, staging, commit or push occurred in
this preparation. Next: finish the five bounded D3 specification decisions above,
record the exact part-level source/evidence/test plan, then obtain implementation
and execution authorization. This draft does not grant that later authority.
Acceptance, if later earned, is **task_only**; deployment **NOT_AUTHORIZED**.
