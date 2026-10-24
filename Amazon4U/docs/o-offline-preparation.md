# O-OFFLINE, preparation and bounded implementation proposal

## Current decision: preparation only

User approved Pi-only source/interface audit and an approval-ready O proposal.
**No O implementation, Worker/SDK, test/import/compile probe, CLI (even dry-run/help),
credential lookup, network, GPU, real-data work or deployment is authorized.**
Records: `../data/reports/o-offline-preparation-v1/`.
Machine proposal: `../configs/o-offline-preparation.json`.
Requirement/evidence matrix: [o-offline-requirements.md](o-offline-requirements.md).
All future budgets/commands below are PROPOSED, not released by this preparation.

Main H-REF is accepted/task_only; its current six files and26inherited Python modules
still match accepted hashes. The last existing full result is412PASS/75.717s/794771456
bytes RSS. No tests were rerun here. H history remains SDK4/native1/7full invocations,
108successful Hsteps inferred, with0remaining released full/mock slots. O starts with
0invocations/0tests, not a reset or transfer of H/D3 budgets. Deployment NOT_AUTHORIZED.

## Start point and actual static findings

Main HEAD: `9071b9bfdfeda2b65074b24c7add545f9c3d8c45`.
Selected existing runner:
`multiautoresearch/post-training/scripts/kaggle_job.py` (781lines, completely read).
Read its CLI dispatch/render/generated-program/launch/status/log/output callees,
actual kickoff/program instructions and dependencies. Legacy LM/Movie instructions
are reference only; none was executed. No pre-training/inference runner is selected.

L1-L9 remain present in current source, with precise qualifications:

| ID | Current code / consequence |
|---|---|
| L1 | `build_kernel_metadata` ignores timeout;`launch_job` push argv lacks `-t`/explicit accelerator; stored timeout is not enforced. |
| L2 | Generated train failure already returns nonzero and must be preserved. Evaluation failure only prints; missing evaluation/final_model can skip; final return is training rc. |
| L3 | `poll_and_fetch` loops forever, ignores CLI return code and treats substrings as terminal/success. A failure message containing complete can mislead it. |
| L4 | `launch_job` resolves user/credentials, renders files and mutates last/job state BEFORE checking dry_run. It is not pure offline rendering. |
| L5 | `collect_project_files` collects legacy root/src/models, not explicit Amazon4U scripts/configs/outer closure; no source byte/count/type policy and is_file follows symlinks. |
| L6 | Generated pip failure warns and continues; requirements are not pinned; no approved environment/input/capability gate. Metadata internet defaults true. |
| L7 | Log-regex eval_score and optional metrics do not establish typed complete Amazon4U results/artifacts. Fetch can return success without metrics. Unsupported logs can return0 but are not availability proof. |
| L8 | Fixed prepare slug, second-resolution experiment slug, mutable last/slug state, render overwrite and forced output download allow collisions/cross-run reuse. |
| L9 | `.split()` loses typed argument boundaries. Current subprocess calls use argv lists, NOT shell=True; do not invent a shell-execution finding. |

Additional O integration findings:
- `run_command` has no subprocess timeout/capture cap. Generated log streaming and
  output/log parsing have no aggregate limits. A bounded O path cannot blindly reuse it.
- Hydration calls unbounded gzip.decompress before writing; bounded streaming decoding
  must reject malformed/over-limit input before exceeding the approved decoded budget.
- Accepted H exposes `execute_fixture(...)` with independently supplied owned objects,
  expectations, authorization/envelope and bundle arguments. Its `main()` deliberately
  refuses host/CLI acquisition; the module has NO __main__ guard. Running the file directly
  can exit0 without doing any work. O needs an explicit callable adapter, not an H edit or
  a success claim from `python scripts/training_entrypoint.py`.
- Main scripts is a namespace package:there is NO scripts/__init__.py. Preserve actual
  import layout; do not add a protected H file to satisfy an old illustrative fixture.
- H's fixed28-path supplied-byte manifest is NOT the full O runtime closure. An O outer
  manifest must also bind the runner, profile/kernel adapter, generated code and relevant
  dependency/config/source snapshots without changing D1's20paths, D2 schemas or H's28.

Full locations, hashes and evidence are in `source-audit.json`/`inspection.json`.
These are source-level findings, not observed deployed failures or new H defects.

## Existing transport reference: useful, but not deployed capability

Read-only external reference remains HEAD
`6894864ff4095cf98e9e36d3a694ff5fc9ba3818` at `/home/sunf/my-project/kaggle-cli/`.
No external file was modified, no CLI/client imported and no installed capability/account
was queried. Current source supports push timeout/accelerator and owner/slug/version
for files/status/logs/output, normalizing numeric versions to API vN labels. Some help
text still says latest; source and unexecuted reference tests establish only static support.

Push --no-run still uploads a version. Push CLI can print an error or omit a version
without a nonzero return; require an exact acknowledged version and matching owner/slug.
Output currently streams bytes and supports pages, but its helper lacks the required hard
aggregate/file/page/deadline policy and includes retries/resume/overwrite machinery.
Logs may materialize whole bodies; follow reconnects. Do NOT claim all transport is missing,
and do NOT treat these helpers as bounded merely because they stream/check Content-Length.
Use existing transport primitives through a targeted counted adapter, or fail closed when
capability/version/bounds cannot be provided. No new authentication/client framework.

## Proposed architecture and H/O seam

**Reuse orchestration, not inherited methodology.** Keep the selected runner as the single
CLI/control facade. Add explicit `--profile amazon4u`/run-spec dispatch BEFORE any user,
CLI, credential or mutable-state resolution. Unknown/missing Amazon-specific profile/spec
fails closed. Legacy no-profile behavior remains legacy, not an implicit Amazon4U run;
Amazon-specific flags without its profile must not fall through into train/evaluate.py.

One profile adapter owns bounded plans, metadata, receipt/status/artifact schemas and
profile-specific transport hooks. One thin kernel-runtime adapter owns bounded hydration,
preflight/stage dispatch and failure-result construction. Neither implements models,
sampling, optimizer, selection, checkpoint math, a second scheduler or an agent/research loop.
Do not mutate module-global ROOT to make Amazon4U work. Shared legacy functions must not
change behavior incidentally; profile-specific fixes avoid changing old launches.

Freeze these proposed interfaces before implementation (names may be implemented inside
approved modules, not a separate service):
- `render_plan(spec, approved_spec, source_packets, expected_sources, capabilities)`:
  pure deterministic in-memory output, explicit synthetic owner; no credential/env/CLI or
  active-state mutation. `write_bundle` is a separate explicit fresh-owned-directory step.
- Existing runner dispatch to profile `launch/status/logs/output` hooks with injected
  transport, clock and state sink. All tests use fake transport/fake clock, no real sleeps,
  provider requests or native commands. Default real-operation route refuses absent
  separately approved execution/capability transport; no discovery fallback.
- `run_kernel(plan, runtime_adapter)` validates actual bounded hydrated bytes, independent
  approval/runtime/input bindings and declared stages, then invokes the adapter's H callable.
  Typed argv is retained as arrays, never free-form train/eval strings or arbitrary shell.
- Runtime adapter seam explicitly separates data/expected-state object bootstrap from
  H's unchanged execute_fixture API. O tests bind fakes carrying the exact argument keys;
  a fixture success is labelled simulated/offline, never real fitting or actual H metrics.
  No new real-data loader, production initialization or fabricated D2 objects are implemented.
  Missing live/bootstrap support fails nonzero before work; no direct-script no-op success.
- A separate outer artifact/transport envelope contains the unchanged H result/envelope.
  Keep H schema/result keys intact; no D1/D2 extension, extra checkpoint root or global patch.

The lack of real-data/bootstrap/GPU support is a later readiness gate, NOT an excuse to
skip offline callable handoff/path-layout/stage/error controls assigned to O. A user
requiring a real production kernel at this milestone would need a scope amendment;
O-OFFLINE must not silently claim that capability.

## Proposed contracts and failure rules

Strict versioned O plan fields:profile/mode, run+attempt IDs, explicit fake/live evidence
kind, protocol/execution-profile/approval references, Hspec/envelope references, device+
accelerator, input/source/bundle expectations, required stages, limits, transport capabilities
and owner/slug/version binding. Private=true/internet=false by default. No credential,
real test target, LM prepare/train/final_model, val_bpb promotion or arbitrary path/command.

Outer source manifest separates input-source payload hashes from generated kernel and
manifest hashes to avoid self-reference. Independently retained expected bundle/plan/
script digests bind the later submission receipt. A self-carried downloaded hash is not
independent authenticity. Physical D3-file SHA and H semantic payload digest are distinct.
O must preserve both roles/expected cross-links, not relabel H's semantic receipt as a
physical file checksum or decode arbitrary artifacts/execute their hooks.

Required state axes are separate:
rendered -> submission_pending -> submitted(version-bound) -> queued/running ->
remote_complete/error/cancelled, plus submission_unknown/poll_deadline_exceeded.
Collection is independently complete/partial/error. Remote complete is NOT verified
run completion; successful collection never hides stage failure or missing required
results/source/checkpoint artifacts. Ambiguous push cannot automatically resubmit.
Use exact registered status tokens/version-bound references on EVERY page/log/status
operation; unknown text, nonzero CLI, error bodies and misleading complete substrings fail.

Collection checks all pages/tokens/counts/deadlines and counts actual chunks BEFORE
writing past per-file/aggregate limits into a fresh quarantine. Validate names/types,
independent lengths/hashes/roles and complete run/attempt/version/source/input/approval/
result/checkpoint/log cross-links before marking verified collection. Reject path escapes,
duplicates, links, partial pages, bad hashes, size lies, over-budget streams and old-output reuse.
Keep bounded redacted diagnostics and primary failure. No force-overwrite, implicit latest,
SDK automatic download retry/resume, arbitrary URL/auth fallback or unbounded decompress.
Use tiny/mock limits for negatives, not real bombs or unsafe filesystem experiments.

Kaggle working output is ephemeral. Missing durable checkpoint/output must remain an
explicit unrecoverable/partial outcome. Offline receipts cannot prove remote crash
recovery, source authenticity, installed versions/accelerator allocation or dataset mounts.
No new storage upload service or real durability test belongs to this milestone.

## Proposed future scope and verification, not approved

Future implementation occurs in a fresh byte-preserving copied project under
`.worktrees/o-offline-implementation-v1/Amazon4U/` WITHOUT creating/changing Git metadata.
Use the enclosing existing Git root with canonical project_root set to that explicit
copy and Git-root-relative writable paths strictly inside it. Main, Movie4U and old
artifacts are read-only sentinels. This is a distinct O deliverable, not replacing H.

Only candidate-relative implementation paths proposed:
1. existing `multiautoresearch/post-training/scripts/kaggle_job.py` (targeted dispatch/hooks)
2. new `multiautoresearch/adapters/__init__.py`
3. new `multiautoresearch/adapters/amazon4u.py`
4. new `multiautoresearch/adapters/amazon4u_kernel.py`
5. new `tests/test_kaggle_orchestration.py`
6. new `tests/test_amazon4u_bundle.py`
7. new `docs/o-offline-orchestration.md`

No accepted scripts/tests/configs, instructions, legacy train/model/evaluate/prepare,
pre-training/inference, requirements/dependencies, external CLI or extension edits.
No new source files/packages outside this list without a prospective scope decision.
Main integration of accepted O is a later separate gate, not automatic from isolation.

Proposed first native round:at most2actual invocations (initial+one consolidated revision),
30minutes each, max_revisions1 through supported delegate/inspect/benchmark/review workflow.
Worker is source/AST/patch only:no tests, target imports, CLI, network, packages, credentials,
Git writes or runtime probes. Pi reviews complete changes/callers/generated programs/tests
before original full verification. Reports remain untrusted/truthful; no limitation/issue
wording is removed or scope label invented to force ACCEPT. All native workflow approval,
baseline, inspection, history, terminal and limitation gates remain mandatory. If a tool
cannot represent a needed disposition, stop for supported handling, not replacement tasks.
No SDK fallback or second recovery round is approved by this preparation.

Proposed maximum3fresh full discoveries:one unchanged412baseline, one initial O-augmented
suite, and at mostone after an authorized consolidated revision. Exact child discovery
and supervisor command proposals/generated paths are in the configuration. Do not run
subsets/smokes/import probes/compileall or execute generated real kernels. O tests may
execute only bounded generated FAKE programs with injected fake entrypoint/preflight,
no target H/model execution. The old412suite's inherited tiny CPU work must be explicitly
approved as part of those full commands; O adds0numerical updates. Static H bound36/suite
would add at most108 to prior108=216, within unchanged whole-H256cap. This is an upper
bound/inference, not observed O execution or permission to spend remaining steps later.

Each proposed full child <=240s/2GiB child peakRSS/8MiB complete stream each/64MiB scratch.
Tiny O fixtures <=2MiB per file/decoded,<=8MiB concurrent/128files,<=8mock pages,
<=5fake polls,<=64KiB chunks (smaller limits for boundary negatives). Zero actual network,
credentials, pip, CUDA, process workers/threads or new RNG/dtype/security setters.
Use the accepted race-tolerant monitor and exact OwnedCache cleanup semantics in a new
versioned parameterized wrapper, with declared fresh source/output paths; do not rerun
frozen consumed H verifiers. Hashing <=128MiB/file, 512MiBaggregate, 60s/file, 300s/snapshot;
include recursive Git/outside-project sentinels and protected ignored source/evidence.
Monitoring is not an OS sandbox or total-process-tree/transient-write guarantee.

Any failed/unsafe/stale/incomplete mandatory gate stops for RCA within approved budget.
Actual started calls/runs count, including failures; reserved slots do not. No automatic
extra tests/recovery/rollback/deployment. Acceptance would be task_only_O_OFFLINE_fake
transport correctness, not remote/GPU/study readiness. Require every O criterion/matrix
row/current exact ordered limitation/issue independently adjudicated before ACCEPT.

## Next approval

Preparation alone authorizes nothing above. A subsequent implementation approval must
cover isolated copy/project root, seven writable paths, first native round, full-suite/tiny
inherited-H execution permissions, wrapper/outputs/bounds and independent acceptance.
The actionable contract is in the companion JSON. Deployment remains NOT_AUTHORIZED.
