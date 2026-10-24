# Deliverable 3, secure checkpoint storage/retrieval and D1/D2 integration

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
| D1, accepted identity/provenance | Canonical identities, supplied source bytes and explicit Git/environment evidence, actual/expected compatibility | Physical acquisition of bytes merely because a record is internally consistent |
| D2, accepted payload validation | Already-decoded state schemas, models/Adam/RNG/sampler/progress/selection, per-object logical resource preflight, independently owned validated output | Safety of decoding, file/path security, live restoration, historical origin or exact continuation |
| D3, this deliverable | Approved-root physical IO and source acquisition, bounded serialization/decoding, atomic exclusive checkpoint publication, corruption/race refusal, owned temporary cleanup, D1/D2 pipeline integration | A model training loop, restoring live models/optimizers/RNG, a process-memory sandbox, or a general agent security controller |

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
3. `docs/checkpoint-storage.md`, extend this record without erasing authority/history.

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
| IO1 | Preserve accepted D1/D2, models, 355 baseline tests, protocols, historical reports, Git metadata and siblings; only declared D3 files/outputs change | Pinned dependency references and before/after scope evidence; no D2 re-audit or source rewrite |
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

## Prospective execution boundaries, not execution permission

Retain the existing correctness-fixture envelope: CPU only, at most8 users,
12 products, 8 metadata nodes/type, 64 dimensions; new fixture files≤2MiB. Synthetic
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

## Subsequent user approval, implementation-first, evidence-driven correction

The user subsequently agreed to this plan and explicitly directed execution within
**IO1-IO8**, allowing the Worker to propose and implement technical solutions.
Pi verifies actual guarantees through source review and testing; specific evidenced
obstacles receive targeted correction, not expansion of the whole design. Unsupported
safety claims and unsafe testing remain forbidden.

This supersedes the earlier preparation-only implementation hold and the suggestion
that Pi must settle every technical choice before a Worker proposal. The earlier
scope, IO1-IO8 and history remain unchanged. Worker discretion covers exact APIs,
format, explicit IO policy and technical mechanisms **within** those criteria, not
unsafe fallbacks, silent weakening, new scientific defaults or dependency rewrites.

Prospective plan, complete exact user approval, original preparation text, pinned
HEAD, three patch paths, command/cwd/scratch and fixture/response/execution bounds:
`data/reports/checkpoint-storage-implementation/plan.json` and its `inputs/`.
One initial text-only response and at most one consolidated revision, 30minutes each,
one generation request each/no retries; no architecture or controller changes.
Pi alone screens before each original full discovery (120s, at most6per round,
including the initial355-test dependency baseline). This is a fresh D3 dependency
baseline, not reopening or auditing D2.

Only fresh owned synthetic scratch IO may run after source/test semantic review.
No unsafe pickle/loader fallback, attacker-hook execution, giant allocation or
bomb-decompression test, live host/source/credential/data target, restoration,
training, CUDA, worker process, remote/heldout work, main transfer or commit/push.
New files≤2MiB; concurrent new fixtures≤8MiB/128files and decoded tensor storage≤2MiB;
D2's synthetic logical bounds remain unchanged. Small explicit IO-policy negatives
and nonexecuting boundary tripwires must replace dangerous test workloads.
These are verification budgets, not universal production-capacity guarantees.

Current status: **D3 implementation authorized, no implementation accepted yet**.
Do not claim a path/race/decoder/allocation/publication guarantee until evidence
supports it. If safe mandatory evaluation is blocked, report that concrete obstacle
and use the bounded correction process; never run an unsafe test merely to get red.
Acceptance remains task_only; deployment NOT_AUTHORIZED.

## D3 implementation proposal, current API, format and limits

This section records the implementation-first technical choices; it does not change
any preceding requirement or historical result. The approved prospective D3 full-suite
ceiling is now **240 seconds**, not the historical 120 seconds above. Pi supplied a
fresh unchanged 355-test baseline receipt: zero failures/errors/skips, 52.493s unittest,
55.645s process, and 3564/3564 expanded cases across 35 ledgers. That receipt is a
dependency gate, not D3 acceptance. Worker executions/tests: **0**.

### Public apis and trust inputs

All three operations live in `scripts/checkpoint_storage.py`; canonical and top-level
imports share the module and accepted exception classes.

- `acquire_sources(root_fd, source_sizes, *, git_evidence, environment_records,
  expected_provenance, io_policy) -> dict`: `source_sizes` is an exact nonempty dict
  from D1-registered two-component POSIX source/config path to independently approved
  **exact** nonnegative byte length. Every request path/type and aggregate declared
  length is checked before acquisition; all physical sizes are checked before reads
  or hashing. Actual bytes feed D1 `build_source_provenance`, followed by
  `validate_source_provenance(actual, expected_provenance)`. The returned provenance
  includes the actual bytes. No D1 allowlist extension, Git invocation, environment
  discovery, directory scan or live source capture is implemented.
- `publish_checkpoint(root_fd, name, payload, *, expected_identity,
  expected_provenance, expected_model, expected_runtime, resource_policy, io_policy)
  -> str`: D2 validates and owns the input first; bounded serialization precedes any
  output creation. The return value is the published basename.
- `retrieve_checkpoint(root_fd, name, *, expected_identity, expected_provenance,
  expected_model, expected_runtime, resource_policy, io_policy) -> dict`: anchored
  bounded acquisition, complete wire preflight, CPU reconstruction, then accepted
  D2 validation (including D1 identity/provenance validation), returning D2-owned state.

Every expected descriptor and both policies are mandatory independent caller inputs.
Neither a checkpoint nor its digest supplies trusted origin, expectations or policy.
There is no restoration, model/optimizer argument, automatic rollback or training API.
Filesystem failures become `CheckpointSecurityError`; malformed wire/native tensor
reconstruction failures become `CheckpointValidationError`. Accepted D1/D2 errors
retain their existing family. Unexpected programming failures are not broadly caught.

### Explicit format decision: A4UCP v1, no native archive ingestion

The proposed on-disk format deliberately replaces native Torch/pickle archives at
this D3 boundary. This is an explicit format choice, not an unsafe fallback or an
assertion that `weights_only` bounds memory. **No `torch.load` path exists.** Native
Torch archives, legacy archives, pickle streams, compression, arbitrary globals,
object reducers and extensions are unsupported and refused. Consequently there is
no call that could omit `weights_only=True` or CPU mapping; a future native-loading
path would require separate review and explicit `weights_only=True, map_location='cpu'`
plus independent preallocation proof. Whether this no-native-loader interpretation
satisfies the retained IO4 Torch-loading wording requires Pi's explicit adjudication.

The 48-byte envelope is magic `A4UCP\\0\\1\\n` (8 bytes), big-endian uint64 body
length, and SHA-256 of the body. Length must match exactly, with no trailing bytes.
The digest detects accidental corruption; it is not independent authenticity evidence.

The body uses one-byte tags: `n/t/f` for None/booleans; `i` for sign byte plus
uint64-length minimal unsigned big-endian magnitude; `r` for finite IEEE binary64;
`s/b` for uint64-length strict UTF8/raw bytes; `m/l/u` for uint32-count insertion-ordered
plain dict/list/tuple. Dict keys are supported primitive keys or recursively primitive
tuples; duplicate/equality-colliding keys are refused. Large exact integers, integer
optimizer keys, bytes and list/tuple distinctions survive without JSON coercion.
D2 retains authority over its stricter semantic/container rules.

`D` is a dense tensor: dtype byte, uint32 rank, uint64 dimensions, uint64 byte count,
and the complete raw compact storage. Dtype codes are 1 uint8, 2 int64, 3 float32,
4 float64, 5 int32, 6 int16, 7 int8, 8 bool, 9 float16 and 10 bfloat16. Raw tensor
bytes currently use native endianness; **cross-endian interchange is unsupported**.
This is a concrete format limitation, not a portable interchange claim. The format
implicitly requires canonical contiguous strides and offset zero; arbitrary strided
storage declarations, aliases, views, tensor subclasses and storage resize instructions
cannot be expressed. D2 checks full input storage before its owned compact copy;
publication does not silently hide a large input backing storage.

`C` is coalesced sparse COO: logical shape followed by dense index and value records.
Descriptor compatibility is checked before tensor reconstruction. Coordinates are then
checked for bounds and strict lexicographic uniqueness before the native sparse
constructor, also using `check_invariants=True`. No densification or coalescing occurs.
The supported logical sparse shape itself must satisfy the configured element bound.

The entire file is parsed into bounded primitive metadata and internal raw-byte tensor
specifications before **any** tensor reconstruction. Every dense record must have
actual raw bytes exactly equal to its checked shape extent times dtype width. Rank,
every dimension, canonical stride extent, aggregate full storage and projected copy
bytes are checked first. Zero dimensions cannot conceal enormous trailing extents.
Only after complete preflight does reconstruction use explicit CPU `empty` or bounded
`frombuffer` backing bytes; sparse construction also explicitly selects CPU. There is
no pickle parser, archive parser, arbitrary-global resolution or native tensor rebuild
whose untrusted shape can expand storage. Exact float32 bits are preserved by raw byte
encoding, including signed zero and current/best bindings checked by D2.

Serialization makes a complete counting pass before creating tensor byte copies or
output files, followed by a bounded encoding pass. Caller payload/expectation mutation
during either public operation is unsupported, as in D2. The count comparison detects
size changes but is not a concurrent-memory sandbox or semantic mutation detector.

### Mandatory independent IO policy

An exact dict has exactly `schema_version='amazon4u-checkpoint-io-v1'`, a nonempty
clean UTF8 `profile_id` (at most 256 characters), and these positive exact integer
limits, each at most signed-64-bit maximum:

| Field | Accounting |
|---|---|
| `max_file_bytes` | Complete envelope plus body, on read and write |
| `max_scalar_bytes` | Each encoded string, bytes value or integer magnitude |
| `max_nodes` | Every wire value occurrence, including dict keys and sparse components |
| `max_depth` | Wire-value nesting, root depth 1, including sparse components |
| `max_storage_bytes` | Sum of complete raw dense storage records, including COO components |
| `max_copy_bytes` | Sum of independent compact tensor bytes projected for D2 output |
| `max_tensor_elements` | Per-tensor canonical extent product, also for sparse logical shapes |
| `max_dimension` | Every individual tensor dimension |
| `max_rank` | Every tensor rank |
| `max_source_bytes` | Aggregate approved source bytes, no greater than D1's 2MiB invariant |
| `max_source_files` | Source request count, no greater than D1's registered set size |

Equality is permitted. V1 also restricts wire depth to 128, rank to 32 and
node count to uint32 representation. These are format/parser bounds, not selected
production byte budgets. Runtime byte capacities require independently authorized
explicit profiles; no free-RAM sizing or default profile exists. D2's separate
metadata/storage/copy/depth/visit policy remains unchanged and is enforced separately.
IO counters are not D2 counters. Multiple bounded copies, input/expected objects,
Python metadata, sparse invariant checks and library overhead can coexist: neither
policy is a total process-memory, time or OS sandbox guarantee.

### Directory descriptors, parent races and supported ownership contract

Linux descriptor-relative open/stat/link/unlink, `O_NOFOLLOW`, `O_NONBLOCK`,
`O_DIRECTORY` and `O_CLOEXEC` are required. The caller independently approves and
opens a narrow directory **inode**, then passes its descriptor; D3 duplicates it and
never resolves an absolute root pathname or its ancestors. The caller retains its
original descriptor. Ancestor pathname renames do not redefine that approved object.
Checkpoint names are only `[A-Za-z0-9][A-Za-z0-9_-]{0,95}.a4ucp`, a direct child of
that object: publication has no traversed parent components to swap.

Directories must belong to the effective UID and not be group/world writable.
Opened files must be same-device, caller-owned, single-link regular files and not
group/world writable. Symlinks and cross-device descendants are refused. Source
parents are opened relative to the held root, never through pathname resolve/reopen.
Before reads and after acquisition/validation, root metadata, every parent edge,
held directory inode and opened file identity/size/mtime/ctime are compared with the
original snapshots. Observed parent replacements, renames and file mutation fail
closed. Final leaf opens are nonblocking so a FIFO cannot block while being classified.
Tests use only synthetic FIFOs, never device nodes or host targets.

This contract requires caller-controlled directories and no malicious same-UID or
privileged namespace/file mutator, including writers that can chmod directories,
substitute a special device, rewrite open files, or swap and restore namespace entries
between checks. Callers must prevent concurrent namespace mutation; competing
publishers may add distinct names or race for the same destination but must not
modify/delete each other's owned temporaries or completed files. Read/source checks
may conservatively refuse unrelated concurrent directory changes. Metadata snapshots
are not an atomic ancestry lock, and cannot prove refusal of every transient hostile
parent race. **The strongest adversarial interpretation of IO2 is therefore an
unresolved issue, not a proved guarantee or an implicit criterion waiver.** The tests
exercise a deterministic observed parent swap without escaping overall scratch.

### Publication, cleanup and durability

A random exclusively created 0600 temporary is written with bounded chunks and
fsynced. Its held descriptor is compared with the temporary name before linking.
A same-directory hard link publishes the completed inode atomically and exclusively;
existing files, directories and symlinks cause collision rather than replacement.
There is no exists-then-overwriting-rename sequence. Successful publication removes
only its owned temporary name. Error cleanup verifies the temporary inode before
unlinking it, refusing to delete a substituted unrelated object. No directory scan or
cleanup of previous callers' incomplete files occurs.

Same-directory hard-link atomicity on a local filesystem with ordinary POSIX semantics
is required; network/distributed filesystems are not claimed supported. File fsync is
requested, but directory fsync, crash recovery, persistence after power loss and
filesystem fault tolerance are not established. An error after the publication link
can leave a complete destination; it is never automatically removed or rolled back.
Hostile temporary-name substitution remains outside the ownership contract above.

### Proposed verification and current status

The new test module imports only existing fixture functions and independently copies
each argument. Proposed cases cover complete LightGCN and 36-parameter Hetero mappings
at initial/completed boundaries; physical-source/D1/D2 integration; exact returned
bits/types and ownership; unchanged global RNG/default dtype; independent expectation
and payload corruption; source request/physical aggregate preflight; missing/symlink
sources; flat path refusal, directories and nonblocking FIFO refusal; deterministic
parent swap; exact/one-over small-policy limits; malformed storage/extents and complete
preflight-before-native tripwires; unsupported formats/hooks; no-clobber collisions,
competing publication and interrupted-write cleanup. None has been executed by Worker.

All generated fixtures and sentinels are fresh children of
`TemporaryDirectory(dir=tempfile.tempdir)`. No execution-bearing pickle, giant tensor,
compression bomb, process/thread, live source read or actual host escape target is
constructed. Existing 355 tests/dependencies remain untouched. Pi must statically
screen the complete implementation/tests before the authorized full discovery under
240 seconds. The baseline receipt alone establishes no new D3 test result.

Current status: implementation submitted, runtime evidence pending. Two concrete
adjudications remain assigned: IO2's strongest hostile-parent-race requirement is not
established by this ownership-and-snapshot mechanism; IO4's retained native-loading
wording must be reconciled explicitly with the proposed no-pickle/no-native-archive
format. Neither is silently marked satisfied. No D3 ACCEPT is claimed. Acceptance
scope remains task_only; deployment NOT_AUTHORIZED.

## User-approved prefix-root correction, fourth request overall

This appended section supersedes the earlier proposal's current root API, leaf-open,
publication, byte-profile and unresolved-adjudication descriptions, not their history
or the original IO1-IO8. The user explicitly approved these conditions:

> Each D1 prefix must have an independently approved directory FD; D3 shall only expose direct-child basenames while preserving logical identifiers. No claims of continuous ancestry regarding the project-root pathname are permitted.

> Scope changes to collecting terminal-error metadata and validating request/context bounds; do not unilaterally alter models, controllers, or routing.

> Proceed from the existing candidate, conduct a comprehensive review against criteria IO1-IO8, and ensure no history is reset, nor should any additional, unrequested correction cycles be initiated.

The second condition concerns the separately bounded SDK wrapper, which these product
patches do not modify. This is the same OriginalD3 deliverable. Three prior request
attempts, two prior technical responses, the incomplete completion, unapplied reference,
old ESCALATE reports and five prior full-suite-associated runs remain history. This is
one additional response, fourth overall, with no retry, follow-up or revision requested.
The supplied initial candidate and all 355 accepted dependencies remain the baseline.

### Current source-root API and acquisition ordering

`acquire_sources(source_roots, source_sizes, *, git_evidence, environment_records,
expected_provenance, io_policy) -> dict` replaces the single project-root source API.
`source_roots` must be an exact dict with exactly the exact-string keys `scripts` and
`configs`, even for a request using only one prefix. Each value must be an exact
nonnegative integer directory descriptor, independently approved by the caller for
that logical prefix. Both bindings are validated and duplicated; their directory
inodes must be distinct. Wrong-type, missing, unknown, non-directory and aliased-inode
bindings are refused. D3 closes only its duplicate descriptors. Binding approval is
an external trust input, not something inferred from descriptor numbers, checkpoint
contents or a project pathname.

`source_sizes` keeps the existing nonempty exact-dict contract: registered full D1
logical identifier to independently approved exact nonnegative byte length. All 20
existing identifiers remain supported without extending D1. For example,
`scripts/checkpoint_identity.py` selects `source_roots['scripts']` by mapping lookup
and opens only `checkpoint_identity.py` relative to its held duplicate. Likewise,
`configs/compute-plan.json` selects the independently approved configs directory.
The full logical identifiers remain provenance keys and entry paths; they are not
flattened. No source-directory component is traversed and no source file descriptor
must be preopened by the caller. No live source files are moved.

All request syntax, declared aggregate bounds and root bindings are preflighted before
source-leaf acquisition. Each leaf is first opened with `O_PATH|O_NOFOLLOW|O_CLOEXEC`.
A symlink, FIFO, device, directory or other unsupported object is refused by metadata
classification before any data-capable open. Caller ownership, no group/world write,
single-link regular type, same-device membership within its own prefix root, and exact
physical size remain required. All physical sizes precede the first source read or
hash. A trusted procfs descriptor transport reopens the classified held inode, not the
mutable leaf filename; the readable descriptor's full signature must match. Leaf
namespace and held-file signatures and approved-root signatures are checked before
reads, after acquisition and after D1 validation. Observed mutations fail before an
accepted return. Root permissions are also checked when the acquisition snapshot is
created, rather than merely assuming the earlier descriptor-duplication check persists.

Ancestor pathname replacement cannot redirect a held approved root to a replacement
directory. The approved object is the directory inode. There is no continuous ancestry
claim concerning a project-root pathname and no remaining intermediate source-parent
edge to traverse. The earlier source-parent-escape issue is therefore superseded by
the explicitly approved object model, not silently waived under the obsolete API.
Snapshots can conservatively refuse unrelated directory activity. They do not constitute
an atomic namespace lock or detect every change restored between observations.

Roots, expected provenance, Git evidence, environment records and both policies remain
independently supplied. Actual acquired bounded bytes still pass through unchanged D1
build/validation APIs; returned payloads still pass unchanged D2 validation. No Git,
environment discovery, directory-wide capture, live restoration or model changes occur.

### Current platform and trusted descriptor transport

Filesystem operations require little-endian Linux x86_64 with the glibc native ABI,
Linux descriptor flags, genuine trusted procfs, and a local filesystem implementing
`O_TMPFILE` and ordinary atomic exclusive hard-link semantics. There is no named or
privileged fallback if required facilities fail. The process and its mount namespace
must be trusted; `/proc/self/fd` is an internal held-descriptor transport, not permission
to follow caller-supplied symlinks. Other operating systems, architectures, libcs and
network/distributed filesystem semantics are not claimed supported.

Before native calls, the implementation checks Linux, `uname().machine == 'x86_64'`,
glibc identification from `confstr`, int width 4, long/unsigned-long/pointer widths 8,
and the explicit statfs structure's 120-byte size, 8-byte alignment and key offsets.
The structure has two longs, five unsigned longs, a two-int fsid, three further longs
and four spare longs; fsid/namelen/flags/spare offsets are 56/64/80/88. This is the
Linux x86_64 glibc ABI, not an assertion that all LP64 ABIs are interchangeable.
`fstatfs` has a typed pointer to that checked structure and must report procfs magic
`0x9fa0`. `linkat` has explicit int/char-pointer argument types and int return type;
its only flag is Linux `AT_SYMLINK_FOLLOW` (`0x400`) on the internal procfs fd link.
Native errors and missing facilities fail closed without capability, mount, privilege
or OS-security changes. These declarations still require Pi's independent review.

### Current publication and cleanup mechanism

After unchanged D2 validation and complete bounded serialization, publication creates
an anonymous `O_TMPFILE|O_RDWR|O_CLOEXEC` inode relative to the approved destination FD,
requesting mode 0600. `O_EXCL` is deliberately absent because it would prohibit later
linking of an O_TMPFILE inode. Initial regular type, ownership, same-device membership,
zero link count, zero size and non-writable-to-others mode are checked. The process
umask may restrict the requested permissions further; it is never changed.

Bounded writes and exact final length precede file fsync. Native `linkat` follows the
internal procfs reference to this held descriptor and exclusively creates the flat
destination basename. There is no mutable temporary directory entry to check and
then link, no check/unlink cleanup window, no overwrite and no named fallback. Every
cleanup path only closes the owned descriptor. An incomplete anonymous inode is
released by close; after publication, close leaves the completed destination intact.
An error after the atomic link may leave a complete artifact, never an automatic
completed rollback. Collisions preserve every existing destination type. Directory
fsync and crash durability remain unpromised.

### Current wire profile: A4UCP v2 L

The 48-byte envelope now starts with the eight literal bytes `A4UCP\x00\x02L`.
`L` explicitly selects little-endian dense numeric storage. Envelope lengths, structural
fields, integer metadata and scalar IEEE binary64 encoding retain their previous
big-endian definitions. Dense numeric bytes are little-endian; opaque uint8 native
RNG bytes are preserved unchanged and are not byte-swapped or reinterpreted by D3.
Both encoding and decoding refuse a non-little-endian native platform. Old ambiguous
v1 magic, unknown versions and unsupported profiles are refused before parsing or
native tensor construction; there is no archive recreation, native loading or legacy
conversion. This is a marked supported profile, not universal RNG-ABI portability.

Only the codec entry guards and magic change. `_policy`, `_shape`, `_Meter`, `_key`,
`_Encoder`, `_Parser`, `_materialize`, `_read_fd` and `_write_all` remain unchanged.
The full counting pass, exact canonical extents and storage, complete aggregate
preflight before the first tensor rebuild, CPU reconstruction, owned output, exact
metadata types/bits and independent D1/D2 expectations remain in force. The retained
Pi adjudication permits this non-pickle format for IO4: no Torch load exists or is
needed merely to supply its flag. Any future Torch-loading path would still require
separate review, explicit weights_only=True and CPU placement plus preallocation proof.

### Corrected synthetic evidence and unexecuted verification gate

Fixture directories are explicitly private 0700 and regular fixture files explicitly
0600, without a global umask setter. Independently approved synthetic prefix FDs are
bootstrapped before syscall tripwire contexts. Every source-tree negative starts from
a successful full physical-source control. The retained source cases now use direct
prefix roots; additional cases cover all 20 logical IDs, malformed root bindings,
non-directory/aliased roots, both-prefix FIFO refusal, ancestor replacement without
redirection and both-prefix leaf substitution before a readable open. Device-mode
classification uses mocked metadata on an ordinary synthetic file, never a device.

Publication controls retain collisions, bounded serialization, partial-write failure
and deterministic competition, and add held-inode/decoy identity, close-only cleanup,
required-facility failure and completed-artifact preservation after an injected
post-link error. Literal independently constructed little-endian float/int records,
opaque uint8 bytes, unsupported-profile dispatch and non-little-endian tripwires
supplement the existing quota, malformed-record and aggregate-preflight controls.
All targets remain fresh owned children of the configured overall scratch. No new
subprocess, thread, training, unsafe pickle, huge allocation or live-host source test
is introduced. Both families' initial/epoch full pipeline tests and all 355 inherited
files remain unchanged except the declared D3 test file's targeted fixture/API edits.

Worker executions/tests remain **0**. User-supplied historical evidence is retained:
the initial 379-test run had one failure, two errors and zero skips; all 355 inherited
tests and 3564 expanded cases passed. Those results do not establish revised-product
success. Pi must review exact bindings, complete changed code/callers/native assumptions
and fixtures before the sole preauthorized full discovery, with a 240-second ceiling,
no failures/errors/skips, complete streams, scope checks and empty-before/after scratch.
At most the sixth aggregate full-suite-associated run is authorized; history is not
reset. No candidate probe, subset or additional correction cycle is requested.

### Ordered current limitations

1. Filesystem IO requires Linux x86_64 glibc, genuine trusted procfs and local O_TMPFILE/exclusive-link support; there is no portable, named or privileged fallback.
2. Directory-inode and prefix-binding approval is supplied independently by the caller; D3 does not establish continuous project-root pathname ancestry.
3. Metadata observations refuse observed namespace/content mutations, but are not an atomic namespace lock and cannot detect every mutation restored between checks or defend against a compromised process or privileged mount mutator.
4. IO and D2 quotas bound specified logical content, storage and copies, not total process RAM, wall time or an OS sandbox.
5. File fsync is requested; directory durability, power-loss persistence, crash recovery and automatic rollback are not provided.
6. The embedded digest detects corruption, not authenticity, historical origin or authorized provenance acquisition.
7. Only A4UCP v2 L is accepted; native archives, old ambiguous profiles, cross-endian conversion and universal native-RNG ABI portability are unsupported.
8. Validation returns owned data without live restoration and does not prove historical training trajectories or exact future continuation.

Current status: targeted correction submitted for independent Pi review, not ACCEPT.
The old root-model uncertainty and IO4 wording adjudication above are historical, not
current assigned gaps. Revised IO3/IO5 evidence, IO6/IO7 pipeline success, full regression
and scope/native review remain unverified mandatory gates; no runtime success is
asserted. Acceptance remains task_only; deployment NOT_AUTHORIZED. No main transfer,
research, training, staging, commit, push, release or rollback is authorized here.
