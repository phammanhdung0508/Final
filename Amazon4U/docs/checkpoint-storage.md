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

## Subsequent user approval — implementation-first, evidence-driven correction

The user subsequently agreed to this plan and explicitly directed execution within
**IO1–IO8**, allowing the Worker to propose and implement technical solutions.
Pi verifies actual guarantees through source review and testing; specific evidenced
obstacles receive targeted correction, not expansion of the whole design. Unsupported
safety claims and unsafe testing remain forbidden.

This supersedes the earlier preparation-only implementation hold and the suggestion
that Pi must settle every technical choice before a Worker proposal. The earlier
scope, IO1–IO8 and history remain unchanged. Worker discretion covers exact APIs,
format, explicit IO policy and technical mechanisms **within** those criteria, not
unsafe fallbacks, silent weakening, new scientific defaults or dependency rewrites.

Prospective plan, complete exact user approval, original preparation text, pinned
HEAD, three patch paths, command/cwd/scratch and fixture/response/execution bounds:
`data/reports/checkpoint-storage-implementation/plan.json` and its `inputs/`.
One initial text-only response and at most one consolidated revision,30minutes each,
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

### Pre-Worker gate result — blocked, no implementation claim

The newly pinned D3 dependency baseline was attempted twice with the exact original
full-discovery command, unchanged355-test bytes and120s limit. Both were incomplete:
120.117s/120.116s, termination exit−15, no completed suite footer. The captured streams
are byte-identical and stop during the optional-name matrix;2631 expanded assertions
completed before each stop, with no failure/error traceback observed before termination.
This is **not** a passing baseline or a demonstrated D2 correctness defect.

Pi recorded **ESCALATE, task_only not accepted**, in
`data/reports/checkpoint-storage-implementation/review.json`. The initial timeout,
one bounded exact-command verification recovery, full streams and stable snapshots
are retained. The earlier archive pathspec/cwd setup failure is also retained; it
occurred before project creation/candidate execution and was corrected without any
scope/HEAD change. Observed host load is not proof of the timeout's cause.

No D3 Worker generation request or revision has run; both response slots remain
unused. No D3 source/test module, serialization/decoding, unsafe IO test, main transfer
or D2 correction was performed. Stop here rather than silently extending timeout,
pruning/changing tests, changing thread/environment policy, repeating indefinitely,
starting a Worker without its baseline, or expanding the D3 design. The specific
next obstacle is the fresh baseline execution precondition; further recovery must be
prospectively assessed/recorded with appropriate authority. D2 remains concluded.

### Later explicitly authorized diagnostic — data collection only

The user separately approved **one full355-test diagnostic**,120seconds, collecting
per-test elapsed times, CPU/RSS/fault/context-switch/swap metrics and stack snapshots.
Implementation/assertions were unchanged; no subset or D2 audit was authorized.
The user expressly distinguished instrumentation overhead/timeouts from correctness.

Data and full analysis: `data/reports/checkpoint-storage-implementation/full-suite-diagnostic/`.
The diagnostic selected355 tests, completed60 OK, and stopped with one active/294 not
yet started at120.167s. The dominant completed method was
`test_dense_parameter_slots`:73.97s wall/73.72s CPU,0 major faults. The optional-name
matrix began only110.26s into the run and had9.74s left. Thus its repeated position
as the last log entry did not establish a hang or defect in that method.

Process CPU was118.88s, sampled peak RSS about607MiB, sampled process swap0. Host
paging occurred but is not established as the dominant process bottleneck. Stacks
showed differing fingerprint/copy/tensor-check/resource-preflight paths. Seven periodic
plus one near-deadline Python snapshots were captured; two have incomplete unwinding.
Noninteractive sudo required a password, so no elevated/native C++ backtrace was
claimed; tests and accessible monitoring ran as the ordinary user without bypass.

These observations support an expensive CPU-workload bottleneck in this diagnostic,
not a deadlock finding or an explanation of all earlier timing variation. Instrumentation
overhead was not separately calibrated. Full per-test/counter/stack evidence is retained.
This is **not** a new acceptance benchmark, does not waive the original baseline gate,
and authorizes no further run, optimization, timeout/environment change, D2 correction,
Worker, main transfer or deployment. Original baseline reviews remain unchanged.

### Runtime investigation concluded; prospective D3 budget amendment proposed

The user has directed closure of the runtime investigation on the existing evidence:
no further diagnostics/profiler, C++ backtrace, sudo, thread-policy experiment or
accepted-test modification. D2 remains concluded; no correctness error is inferred
from a timeout or the instrumented run.

The proposed D3-only amendment is **120s →240s per full-suite invocation**, including
both one fresh unchanged uninstrumented355-test baseline and later full suites with
additional D3 tests.240s is an upper ceiling, not expected runtime or a per-test
allowance. See [the formal proposal](checkpoint-storage-execution-budget-amendment.md)
and `configs/checkpoint-storage-execution-budget-amendment.json`.

**Pending explicit approval; not effective.** No new run, verifier timeout change or
Worker has occurred. Original120s plans/criteria/failed results and diagnostic reports
remain unchanged; nothing exceeding120s is retroactively compliant. After approval,
record a versioned prospective verifier/receipt, run the fresh full355 baseline with
unchanged tests/assertions, and proceed with the already authorized Worker-led IO1–IO8
process only if that baseline passes. Preserve all other safety/resource/review/request
limits and histories; no automatic further increase or reset. Deployment NOT_AUTHORIZED.

### Subsequent approval and bounded implementation result — ESCALATE

The user's **“Approve. please procced”** prospectively approved the240s ceiling.
Original proposal/120s plan/verifier/reports remain unchanged; `effective-plan-240.json`
and `verify_240.py` bind only future executions to that approval. Fresh unchanged,
uninstrumented dependency baseline: **355OK,52.493s unittest/55.645s process**,
zero failures/errors/skips;35 ledgers/3564 successful cases. See `baseline-240-review.json`
in `data/reports/checkpoint-storage-implementation/`. This satisfied launch, not D3 acceptance.

Initial text-only Worker source/tests/docs were independently reviewed and mechanically
applied in the isolated D3 project, never main. Full approved regression completed
**379tests,1failure/2errors/0skips,53.213s unittest/55.993s process**. All355 inherited
IDs and3564 expanded cases pass; streams/snapshots are complete. Fresh source-fixture
permissions caused early refusals and prevented the parent-swap injection being reached.
Pi's safety prescreen missed that mode mismatch; it was not a D2 defect. No dangerous
allocation/decoder probe, runtime investigation or environment tuning was performed.

Pi's consolidated review also identified mutable-temp check/link and check/unlink
windows, special-file opens before classification, incomplete parent-race proof and
unidentified native byte order. See `consolidated-feedback.md`. Pi explicitly recorded
the non-pickle format as a permissible original technical choice: there is no native
archive loader to relax, no required native archive interoperability, and every future
Torch load would still require explicit weights_only=True/CPU plus preallocation proof.
The original IO1–IO8 texts and initial report are not rewritten by that interpretation.

The **sole consolidated revision** proposes explicit private fixtures, anonymous
O_TMPFILE/descriptor-bound publication, O_PATH classification and an explicit v2
little-endian wire marker. It was read completely as an **inspection-only preview**,
not applied or executed: its production-source baseline SHA is65characters and does
not match the independently pinned64-character hash. The original response/hash/report
were not silently repaired. More importantly, independent source review confirms the
assigned **IO2 source-parent ancestry/race guarantee remains unresolved**; this alone
blocks acceptance even if that metadata typo were corrected. Both response slots are
consumed; no third Worker, automatic source/hash fix, new task or budget reset.

Final recorded decision: **ESCALATE, acceptance_scope task_only, task_accepted false**;
**deployment_decision NOT_AUTHORIZED**. Current report:
`data/reports/checkpoint-storage-implementation/final-review-evidence/review.json`.
Original pre-Worker `review.json` remains historical and byte-preserved. Initial tested
candidate and nonapplicable revision are retained separately. No D3 main source/test
transfer, training/profiling, held-out execution, commit/push/release/rollback. Further
implementation requires separately explicit scoped authority; review-only reassessment
does not permit implementation fixes. D1/D2 remain concluded, accepted and protected.

### User-approved bounded correction amendment — no reset

The subsequent user conditions explicitly authorize one additional bounded correction,
not replacement of the original task/history. Exact authority, prior counters and
Pi-verified baseline hashes are recorded under
`data/reports/checkpoint-storage-implementation/bounded-correction-amendment-v1/`;
see [the amendment](checkpoint-storage-bounded-correction-amendment.md). Initial
candidate and both ESCALATE reports remain unchanged. A byte-identical new isolated
copy supplies independently checked baseline hashes before Worker implementation.
One additional30minute text request, no revisions/retries; at most one reviewed240s
full discovery with the same body/interpreter, relocated declared scratch only.
Preserve verified D3 logic and355 dependencies; no rewrite, unsafe filesystem
experiment, controller or separate race research. Worker must resolve original IO2
with clear guarantees or precisely identify the unmet clause and stop. Pi reviews all
IO1–IO8, especially IO2/3/5/6/7. No automatic follow-up if IO2 remains unresolved;
no acceptance/main transfer/research/deployment is granted by this amendment.

The additional request subsequently ended with SDK stop_reason:error,HTTP200 and
0response bytes after371.113s. No patch or precise feasibility report returned;
no candidate code changed and no new full suite or filesystem experiment ran. Exact
underlying error text was not retained by the credential-safe client. Independent
request/context hashes and before/after snapshots confirm both candidates and all
prior history preserved. **ESCALATE/task_only/not accepted**, deployment NOT_AUTHORIZED:
`data/reports/checkpoint-storage-implementation/bounded-correction-amendment-v1/review.json`.
IO2 remains the prior unresolved assigned blocker, not proven infeasible by this
incomplete completion. Three overall request attempts used, two technical responses;
no automatic retry, follow-up, source/hash fix, new task or budget reset.

### Focused IO2/domain and SDK assessment — proposal only

Before any additional execution privileges, the user requested a specific technical
decision, not race-condition research/controller work. See
[the decision and bounded-correction proposal](checkpoint-storage-io2-technical-decision.md).
Naively dropping scripts/configs prefixes would violate accepted D1. The recommended
specification uses independently approved directory-inode roots per existing logical
prefix and direct-child physical opens, retaining every registered D1 identifier and
existing actual-byte/provenance validation. No mandatory deeper logical source path
was found among the20 registered paths. Root-inode approval must be explicit; this is
insufficient if a single-project-root ancestry proof is instead mandatory. No original
IO2/IO6 clause or threat-model requirement is silently amended.

Local static SDK review verified full context in the observed failed payload; no
max-output-token field was sent despite requested maxTokens32768. Current SDK supports
terminal error metadata that our wrapper did not retain. Exact historical failure
cause remains unknown. Small wrapper preflight/error-metadata corrections and one
future bounded patch are proposed, not executed/authorized. No further model call,
product change or regression in this assessment; all histories and used counts stay.
Current IO2 remains unresolved, IO3/IO5 unverified, IO6/IO7 unpassed. Full IO1–IO8 and
ordered limitation review still govern any task_only ACCEPT; deployment NOT_AUTHORIZED.

### Subsequent approved prefix-root bounded correction

The user has now finalized the technical choice: independently approved directory FD
per D1 prefix;direct-child physical basenames,unchanged logical identifiers,no continuous
project-root pathname ancestry claims. Scope SDK wrapper changes to terminal-error
metadata and request/context validation;no model/controller/routing changes. Authorize
one specific correction phase from the existing candidate,not additional cycles/reset.
See [approved phase](checkpoint-storage-prefix-root-correction.md),its exact authority
and plan under `data/reports/checkpoint-storage-implementation/prefix-root-correction-v1/`.
Three old request attempts stay used;one fourth request30minutes/no retry or revision;
Pi reviews complete IO1–IO8 before at most one pinned full240s suite. No test/probe,
unsafe FS experiment,main transfer or research work beyond that explicit scope.

The approved phase returned a valid bound Worker patch;Pi read full785-line source,
815-line tests,docs/callers/native references and applied only3reviewed candidate files.
The sole full run completed389tests,1failure/0errors/0skips,55.852s unittest/62.611s
process;all355 and3564expanded cases pass. Actual root/decoder/anonymous-publication and
source→D1→checkpoint→D2 controls pass;9preserved nodes remain exact. A new sourceFIFO
oracle globally blocks a valid configs `_readable` before reaching its scripts FIFO
negative. This is not evidence of FIFO data opening. Pi's prescreen missed the ordering;
no accepted dependency defect or unsafe filesystem experiment was found.
Current **ESCALATE/task_only/not accepted**,deployment NOT_AUTHORIZED:
`data/reports/checkpoint-storage-implementation/prefix-root-correction-v1/review.json`.
IO8zero-failure gate still blocksACCEPT. No source/oracle fix,rerun,extraWorker or reset;
4request attempts and6conservativefullruns retained. Further test-only correction and
verification,if desired,require explicit bounded authority;no main/research transfer.

### Subsequent Pi-only test-oracle direction

The user recommends approving a Pi-implemented test-oracle-only correction followed
by full regression,no additionalWorker unless newproductiondefects are discovered.
This is recorded prospectively as that specific continuation:
[Pi-only oracle plan](checkpoint-storage-oracle-only-correction.md). Only the sourceFIFO
method changes;independent singleton zero-byte regular-file positive then same-size
FIFO makes the global tripwire target-specific and excludes size-mismatch shortcuts.
Production/othermethods/355dependencies/old389failure stayexact. One seventh full240s
run after complete review,no Worker/reset/additional cycles. Full IO1–IO8 and exact
ordered current limitation review still required;task_only,NOT_AUTHORIZED deployment.

### Final isolated D3 acceptance after authorized Pi oracle correction

The only newproduct-artifact change was Pi's namedFIFO testmethod:independent singleton
emptyregular positive,then same-sizeFIFO andregular-type reason,withno-readable/read
tripwires andownedrestoration. Production andcandidateAPI docs remain byte-identical to
Worker's reviewed prefix-root implementation;allothermethodAST/355accepted files exact.
One authorized seventh full discovery: **389OK,0failures/errors/skips**,53.292s unittest /
55.890s process under240s;35ledgers/3564expanded cases preserved andstdout identical to
acceptedbaseline. Complete streams/scope/Git/ignored-evidence/scratch checks retained.

Pi verified every originalIO1–IO8 andeachcurrentWorker limitation in exactorder/text,
usingactualcode/native dependencies/results andoriginalscopequotes. No assigned requirement
waived. Pendingreport evidence issues closed for this user-authorized derivative,not a
rewrite orretroacceptance ofWorker's earlier failed389variant. Alloldreports/ESCALATEs and
budgets remain preserved;4Worker attempts,7conservativefullruns,no newWorker thisphase.
**ACCEPT,acceptance_scope task_only**,isolated D3 only:
`data/reports/checkpoint-storage-implementation/oracle-only-correction-v1/review.json`.
**deployment_decision NOT_AUTHORIZED**;no main source/test transfer,training/profiling,
heldout,stage/commit/push/release/rollback. Main integration requires separateapproval.
