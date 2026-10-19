# Agent instructions — Amazon4U

Scope: this directory and all descendants. Amazon4U is independent of Movie4U;
do not copy MovieLens split rules, candidate rules or ID assumptions here, and do
not change sibling projects unless explicitly requested.

## Current Worker governance — prospective, no execution authority

Read `docs/worker-delegation-policy.md` and `configs/worker-delegation-policy.json`
with `docs/delegation-and-execution.md`. **Native agy Worker invocation through the
delegation extension is the default for substantive implementation.** Pi plans,
scopes, orchestrates, independently verifies and decides acceptance; native agy
implements/corrects/tests/documents only within explicitly authorized scope.

Use distinct labels: `native_agy_worker_invocation`,
`sdk_text_only_generation_request`, and `pi_independent_verification`.
SDK direct completion is NOT native agy. Retain SDK only as an optional fallback:
**explicit user approval before every SDK request**, with why native agy is unsuitable,
exact task/unresolved criteria, model/channel, source/output paths, request ceiling,
time/resource/verification limits and independent acceptance procedure. No automatic
fallback, silent switch, budget transfer or inherited SDK permission.

Prospective native recovery maximum is **2 + 2 + 2**: round1 attempts1/2 initial and
consolidated correction; round2 attempts3/4 targeted continuation and correction;
round3 attempts5/6 final continuation and correction. Maximum two per round/six per
deliverable, not blanket permission. Authorize each task/round prospectively; stop on
acceptance, no credible resolution path, an authorization boundary or critical safety.
After every substantive ESCALATE preserve evidence/counts, identify actual root cause
(including Pi handoff vs Worker failure), remaining gaps, material next-round changes,
technical justification and needed approval. After ESCALATE #3 present an evidenced
stop/exceptional-native-continuation/SDK recommendation, never execute fallback.
Repeated reports of one decision are not new ESCALATEs. Completed round counters and
same-task cumulative history never reset; all extension gates remain mandatory.

This policy is prospective and grants NO new native/SDK invocation, application,
regression, experiment, integration or deployment. Historical H-REF recorded four SDK
requests, NOT four native agy calls; they remain consumed. No new H native allocation
or SDK request follows. Preserve original plans/evidence/budgets and accepted science,
models/evaluator/D1–D3. No extension/SDK/controller/source/test changes, Kaggle/GPU/
training/profiling, stage/commit/push from this governance correction.

## Current H-REF — final Pi ACCEPT/task_only

The user explicitly authorized a separate append-only final Pi H-REF acceptance review
using existing independently verified evidence. Hierarchy: User authorization → Pi final
acceptance authority → native agy implementation/execution evidence. Current decision:
ACCEPT/task_only at `data/reports/training-harness-h-ref-v1/pi-final-acceptance-v1/review.json`;
authority/artifact/report in that directory and `docs/h-ref-pi-final-acceptance.md`.

All original H1–H9 independently verified. Ordered Worker limitations0/2 are verified
external boundaries with original quotes;assigned Pi duties1/3 and the original deferred-
review issue are fulfilled with evidence,NOT outside scope. No substantive acceptance
blocker remains. Existing412overallPASS/81.938s/870412288bytes,389inherited/34D3/3564cases,
empty fresh scratch/owner-safe cleanup/preservation;all322isolated files/H8/science exact.
No new Worker,SDK,implementation,regression or extension change for this final review.

Native task3606687c-beba-4d3f-bd24-03ae5238da37 remains historical ESCALATE with original
failed verification/report/budgets unchanged. No native ACCEPT/reopen/reassessment/receipt
substitution or extension restriction bypass. The latest user amendment permits a
separate Pi deliverable record,not a native-tool schema/status change. SDK4/native1,
old3full+separate1full=4 retained. Main H integration,O/GPU/profiling/experiments/deployment
remain unapproved;deployment NOT_AUTHORIZED. No stage/commit/push authority.

## Previous cache lifecycle review — overallPASS,formal acceptance hold

User approved Pi-only cache path/cleanup infrastructure and ONE separately allocated
full run. No training/H8 change. Fresh owned cache:creator/manager PiPID1363626,
consumer childPID1363657;private held-FD/inode/dev/UID/mode checks,empty-only rmdir
finally after child settled. Nine mock-only checks cover failure/refusal paths;no real
foreign/pre-existing/nonempty directory deleted. Oldcache3413649 remains unchanged.
Overall verifierPASS:412tests/0failures/errors/skips,74.875s unittest/81.938s child/
870412288bytesRSS;389inherited/34D3/3564expanded and preservation/emptyscratchPASS.
All322isolated files exact;source/model/BPR/Adam/science/H8 remain unchanged.
Latest additive `data/reports/training-harness-h-ref-v1/cache-lifecycle-v1/final-adjudication-v1/acceptance-packet.json`.

The prior all-H1–H9-verified wording overstated H8: runtime PASS, but full acceptance
closure is pending. Two external limitation boundaries have original-scope quotes;
two assigned Pi review duties must NOT be called out of scope. The tool has no separate
issue-resolution field, blocks a nonempty original unresolved_issues array, and allows
only out_of_scope/unresolved limitation labels. See final-adjudication-v1/operator-decision-required.md:
an explicit procedural decision is needed, not new H code, Worker requests or tests.
No workflow, source, report rewrite or policy exception is inferred.
HREF ACCEPT NOTrecorded:existingnative task
3606687c-beba-4d3f-bd24-03ae5238da37 remains terminalESCALATE withfailedoriginalcommand.
This approved amended Pi verification does not reopen/reassess/retroaccept it or permit
rewritingoldverifier/receipts/replacementtasks. See acceptance-gate-review.json:operator-
supported handling is needed;original-command-only reassessment is not permission to
substitute newcommand. No furtherWorker/SDK/run/extensionedit or maintransfer authority.
CountersSDK4/native1(secondunused),old3full+separate1full=4,allconsumed. Hstillnotaccepted;
deploymentNOT_AUTHORIZED. Preserve alloldreports/failures/cache andcurrentpassingproof.

## Previous native correction —412functionalPASS,cleanupESCALATE

User separately approved at most2native invocations on canonical dictionary adaptation/
related fixtures and NaN exception ordering,plus Pi's one full regression. One actual
native invocation corrected only isolated training_data.py and two H test modules;
no accepted D1–D3/model/BPR/Adam/science changes. Native task
`3606687c-beba-4d3f-bd24-03ae5238da37` is ESCALATE/task_only,not accepted.
Latest `data/reports/training-harness-h-ref-v1/native-correction-v1/review.json`.

All412functionaltests/389inherited/34D3methods/3564expanded cases PASS,0failures/errors/
skips,72.582s unittest/79.580s child/863043584bytespeakRSS. Numerical/private D3 restart/
device/closure/entrypoint controls now pass. Mandatory supervisor nevertheless exits1:
EMPTY torchinductor_sunf directory remains in required-empty scratch. Source-only RCA
traces native optimizer's lazy Dynamo import to module-global cache_dir creation;Pi
missed framework-cache lifecycle,not a model/math/native-correction defect. Preserve
candidate,complete logs,empty directory and all history;no cleanup/fix/rerun/rollback.

Counts:SDK4/native1of2(secondunused),fullH3of3used(1baseline+2candidates). User explicitly
forbids automatic Worker requests on re-test issues. No second Worker/SDK/new test or
main H transfer. Proposed only:Pi-only verification-infrastructure/cache ownership and
fresh-scratch plan plus explicit extra-suite budget;respect terminal/operator/original-
command gates,no task replacement/reset or edits to frozen old verifier/evidence.
H remains unaccepted;deployment NOT_AUTHORIZED. Original24-error regression preserved.

## Historical task record — scope-limited, not current channel authority

The following accumulated approvals/status snapshots retain their original histories.
Their "current/latest/now" wording is local to the recorded phase, not a live launch
permission. In the H-REF text-only sequence, generic "Worker/agy" labels refer to SDK
generation, not native agy. D2/D3 SDK exceptions never establish a future default.
Original task criteria, counts and decisions remain unchanged; the policy above governs
new channel/recovery decisions without reopening or resetting any old task.

**Subsequent user direction permits one limited same-H continuation**,not a newproject
orhistoryreset. See `data/reports/training-harness-h-ref-v1/continuation-v1/plan.json`.
Pi must finish source/APIhandoff audit beforeoneadditional30mintext-onlyresponse,no
retry/revision;Hrequests2used→ceiling3. Same316-filebaseline/root/sixnewpaths;one reviewed
candidatefullsuite inoriginalunusedinitialslot,unchanged240s/2GiB/fixturebudgets.
Originaltwo-responseESCALATE remainshistorical;no O/GPU/remote/maintransferauthority.

**Subsequent user direction/Pi approval grants one targeted same-H correction:**see
`data/reports/training-harness-h-ref-v1/correction-v1/decision.json`. Prior3 requests,
candidate and ESCALATEs remain unchanged;one fourth30mintext-onlyresponse,no retry or
revision. B1 is a library compatibility/capability-check boundary,not a model defect;
Pi prospectively approves only an H-local CPU Adam health-check adapter preserving
native math/state,not driver probes/global patches/new packages. One F2–F5 brief,
no new Package H/planning matrix. No application,test,CUDA or other execution authority
is granted by this phase;review first and full regression only if separately released.
**Fourth response now received:**four proposed files corrected,two retained exactly;
Pi reviewed B1/F2–F5 statically and preserved native Adam math/accepted components.
See `data/reports/training-harness-h-ref-v1/correction-v1/review.json`. Decision ESCALATE
only pending mandatory authorized verification,not ACCEPT. No source application,
candidate tests,updates or driver probes. Four requests used/ceiling4,none remaining;
old389baseline remains the sole Hsuite. Next gate is isolated application plus the
original full regression if separately released,no new Package H or automatic fifthcall.

**That prior continuation is ESCALATE,not accepted:**third response delivered all six
files,but they remain unapplied previews. Complete static review found invalid D3
basenames,wrong checkpoint-exception assertions,unknown-user/fallback oracle confusion
and omitted D2 source closure. Installed Torch CPU Adam can enter a CUDA-driver
availability path;Pi missed this native dependency in the pre-request audit and owns
that preparation omission. No candidate import/test/update or driver probe was run.
See `data/reports/training-harness-h-ref-v1/continuation-v1/review.json`. All316 baseline
files/main accepted code/Git/siblings/old reports remain exact. Three Hrequests used,
none remaining;one old baseline and zero candidate suites. No fourth request,Pi source
fix,extra test,CUDA permission orhistoryreset automatically. Deployment NOT_AUTHORIZED.

**Historical H-REF result: ESCALATE/not accepted;two responses consumed,zero files.**
Fresh389baseline passed57.732s unittest/64.917s process,661983232bytespeakRSS;all3564
expanded cases pass. FirstWorker requestedfrozencontracts;finalWorker refusedto guess
loss-returnkeyomittedbyPiinfreshcontext. No Hcode/newtests/updates/restore/candidatesuite.
Pi responsibility andexactcompleteAPI supplement recorded in
`data/reports/training-harness-h-ref-v1/review.json`. Selected316-filebaseline,accepted
code/history/Git/siblings preserved. No thirdrequest,Pi sourcefix,extra test/replacement
orbudgetreset automatically. H1preservation/H9process verified;assignedH2–H8incomplete.
Scope task_only_H_REF_CPU_reference_not_accepted;deployment NOT_AUTHORIZED.

**Historical latest authority: Package H (H-REF) bounded implementation approved**,including
CPU-reference device-capability interfaces. See `docs/training-harness-h-ref.md` and
`data/reports/training-harness-h-ref-v1/plan.json`. Pi delegates six new H files to agy
through existing tool-free text-only channel,one initial+one consolidated revision max,
30min each,no retries. Pi-only pre-reviewed389baseline/Hinitial/optionalHrevision full
suites<=240s/2GiBpeakRSS withbounded tinyCPUupdates/private stagedrestore now permitted
only under that plan. Allaccepted code/tests/history remainprotected;oldD3archivesretired.
Out-of-task packages/frameworks/structures ask Pi first. No PackageO/GPU/Kaggle/realdata/
profiling/comparative training/main source transfer/stage/commit/push authority.

**Device clarification:** CPU reference first is a correctness milestone,not a
permanent CPU architecture rule. See `docs/training-harness-device-transition.md`.
H-REF/O-OFFLINE must retain explicit device/tensor/state-transfer interfaces;GPU-READY
is required before planned large-scale GPU profiling. Do not declare the whole harness
experiment-ready from CPU/offline acceptance alone. No GNN rewrite/guard relaxation,
Worker,CUDA test,remote preflight or execution is authorized by this discussion.

**Current user direction: Training Harness preparation only.** Reuse existing
`multiautoresearch` orchestration (Kaggle packaging/submission/status/logging/output),
not inherited LM/Movie methodology. See `docs/training-harness-preparation.md`,
`docs/training-harness-requirements.md` and `configs/training-harness-preparation.json`.
The34-part/10-criterion plan proposes scoped CPU harness and managed-runner adapters;
no production/legacy component replacement,Worker,test,optimizer/restore,training,
profiling,Kaggle call/credential lookup or deployment is authorized now. External
`/home/sunf/my-project/kaggle-cli/` is read-only reference,not a live CLI permission.
Accepted models/evaluator/D1–D3 and389main baseline remain protected. CPU-only guards,
LightGCN initialization,complete bundle provenance outside D1's20paths,Kaggle runtime/
durability and execution settings remain explicit gates;do not hide these behind
"GPU-ready"/"fine-tuning" labels. Future budgets/240s/Worker scopes in the proposal
need separate approval and do not reset4Worker/9D3-run history. No automatic launch.

**Current D3 authority: bounded main integration.** The user directed retirement of
isolated D3 implementation/execution and transition to main. Both original D3 trees
and all evidence/history remain audit-only at their original paths;no delete/move,
chmod,Git metadata changes or further isolated runs. Pi may run one355 main baseline,
copy exactly the3accepted D3files (archive current main doc),then one389 full main
regression,each<=240s. No Worker/code/test rewrite,retry,training/profiling/deployment.
Four Worker attempts/seven old full runs remain retained;two integration runs are
separately bounded,not a reset. See `docs/checkpoint-storage-main-integration.md`,
`configs/checkpoint-storage-main-integration.json` and its frozen report plan.
**D3 main integration is now ACCEPT/task_only** in
`data/reports/checkpoint-storage-main-integration/review.json`:355baselineOK
56.293s process,389integratedOK57.452s process,0failures/errors/skips under240s each.
All355 IDs/bytes and3564expanded cases/34D3methods preserved;only accepted3files
transferred byte-identically. Old main D3doc archived;separate integration doc/receipt
supersedes preserved Worker submission-status paragraphs in the transferred API doc.
Both retired D3 trees/all old reports/Git/siblings unchanged. AllMI1–MI5/IO1–IO8 and
8exact ordered limitation reviews verified. Four Worker attempts/seven old runs plus
2main runs retained (nine total),no reset. No further execution/correction automatically.
Deployment NOT_AUTHORIZED;profiling/training/heldout/remote/stage/commit/push remainblocked.

Compare KG-only and KG + GNN methods for **top-K Amazon product recommendation**,
with popularity and collaborative-filtering baselines under shared evaluation rules.
Agreed scope: **full downloaded splits** for Electronics, Toys_and_Games and
Musical_Instruments, evaluated independently. Develop/resource-check on
Musical_Instruments first, then Toys_and_Games and Electronics. No research
interaction subsampling or evaluation-user sampling; fixtures are correctness-only.
Record and obtain approval for any resource-driven scope amendment before using
it or observing comparative results. See `configs/dataset-scope.json`.

Full 5-core `last_out` benchmark splits and root-level Parquet product metadata
are downloaded. Separate, all-rating training KGs are built and validated.
Scripts cover downloading, inspection, audits, graph construction and visualization.
No model training or learned embeddings have been run. The approved baseline
protocol is frozen as `amazon4u-protocol-v2.0.0` after user-approved removal of
BPR-MF (see `docs/bpr-mf-removal-amendment.md`); v1.0.0 remains preserved. Lifecycle state
`IMPLEMENTATION_REGRESSION_VERIFICATION`. Milestone I1 reference components are
implemented and independently verified task-only: Step 1 evaluator integration is
accepted in main, and Step 2 remaining Popularity/KG-only scorer, sampler, BPR
loss/reference utility and endpoint-regularization verification is accepted under
`cc64f3cb-9a4f-40aa-abbf-5ced6c1bd000` with the v2 suite at 115 passing
regressions. Step 3a learned-model foundation and core LightGCN reference
implementation is accepted under `8153df65-e2e2-4571-a4ce-340d0ffeb1c2`; Step
3b Heterogeneous KG+GNN reference encoder foundation is accepted under
`d35b3d7c-3cd6-4f8e-8ecf-adcb963f4b93`, with the current suite at 163 passing
regressions. Differentiable CPU models/training-only graphs have frozen-verification
ACCEPT under `f709d062-ae0e-4eb6-960c-f94728180d1b` and separately reviewed main
integration (204 tests at that milestone). Separately accepted D1 identity/provenance
is integrated in main with **249 passing tests**; do not redo it.
The separately authorized A04/D01 round now has **whole-D2 ACCEPT, task_only**, in
its isolated candidate:355 passing tests/zero failures/errors/skips,111.539s unittest,
116.571s complete process under the unchanged120s limit. All337 inherited tests and
3564 expanded cases pass. Six ordered limitations and both submission-time evidence
questions are independently resolved in
`data/reports/checkpoint-payload-a04-d01-correction/review.json`.
The user subsequently authorized exact D2 main integration, now **ACCEPTED,
task_only** with355 tests/zero failures/errors/skips,54.139s unittest/56.643s process
under120s. Source/tests/documentation are byte-identical to the accepted artifact;
all249 baseline IDs/bytes and3564 expanded cases pass. See
`data/reports/checkpoint-payload-main-integration/review.json` and
`docs/checkpoint-payload-main-integration.md`. No Worker/source revision or commit/push.
Profiling/training remain unauthorized and deployment **NOT_AUTHORIZED**.
Latest user approval now authorizes **D3 implementation-first within IO1–IO8**,
using the existing text-only Worker channel and evidence-driven targeted correction.
The Worker may propose/implement technical choices within scope; Pi verifies actual
code/guarantees and screens before any safe synthetic test IO. Pinning every technical
choice before a proposal is no longer a prerequisite. See
`data/reports/checkpoint-storage-implementation/plan.json` and the appended approval in
`docs/checkpoint-storage.md`. D3 was initially **blocked before Worker launch**: its fresh355-test
dependency baseline timed out twice at the original120s limit, with no completed
footer or demonstrated D2 error. Phase review is ESCALATE in
`data/reports/checkpoint-storage-implementation/review.json`; at that historical gate,
zero Worker requests and both response slots were unused. No automatic third run, timeout extension,
test/source change or environment/thread retuning. Address only this concrete
execution precondition prospectively; do not reopen D2 or expand D3 design.
A subsequent user-authorized single full355-test diagnostic collected timings,
resources and Python stacks under120s, not an acceptance benchmark. Dense matrix
completed in73.97s; optional-name matrix started at110.26s, so its final-log position
is not a hang finding. No source/assertion/environment changes. See
`data/reports/checkpoint-storage-implementation/full-suite-diagnostic/analysis.json`.
The user has now **concluded the runtime investigation**: no more diagnostics,
profilers, C++ traces, sudo operations, thread experiments or accepted-test edits.
A D3-only **120s→240s full-suite ceiling** is formally proposed in
`docs/checkpoint-storage-execution-budget-amendment.md` and its companion JSON.
It covers the fresh uninstrumented355 baseline and later suites with added D3 tests;
240s is an upper limit, not expected runtime. The user's subsequent **“Approve.
please procced”** now explicitly approves it; see the frozen approval and
`effective-plan-240.json` in the D3 report directory. Preserve original120s criteria,
plan/verifier/results and all used budgets. Run one fresh unchanged uninstrumented355
baseline under240s; only a pass permits bounded Worker-led IO1–IO8 continuation.
No further runtime investigation or environment/thread experiments.
**Latest implementation result: ESCALATE, task_only/not accepted.** Approved fresh
baseline355 passed55.645s process. Initial379 full regression completed55.993s with
1failure/2errors/0skips; all355 inherited tests and3564 expanded cases pass. New source
fixture permissions caused early refusals. The sole consolidated Worker revision is
inspection-only: malformed65-character source baseline hash prevents application;
assigned IO2 source-parent-race refusal is still unresolved independently of that typo.
Both generation slots are consumed; no third Worker, automatic source/hash repair,
repeated benchmark, replacement task or budget reset. See
`data/reports/checkpoint-storage-implementation/final-review-evidence/review.json`.
Preserve the initial candidate, nonapplicable revised preview and all original reports.
No main source/test transfer; deployment NOT_AUTHORIZED. Further implementation needs
separately explicit scoped authority, not a review-only reassessment.
**Subsequent explicit authority now grants one additional bounded D3 correction
response**, not a reset of the two used responses or either ESCALATE. See
`docs/checkpoint-storage-bounded-correction-amendment.md` and its pinned plan/hashes.
Preserve the initial candidate unchanged; use its byte-identical isolated copy.
Worker must resolve original IO2 with clear guarantees or identify the exact infeasible
clause. Preserve verified logic; no D3 rewrite, scope weakening, unsafe experiment or
controller. At most one additional30minute text request, no revisions/retries; one
reviewed240s full suite maximum, retaining prior five runs. Stop on unresolved IO2.
Pi reviews original IO1–IO8, especially IO2/3/5/6/7, before any acceptance. Old candidate,
reports, budget and baseline hashes remain preserved; no research/main-transfer authority.
**That additional request is now used and ended INCOMPLETE_OR_BLOCKED**: HTTP200,
SDK stop_reason:error,0response bytes, no patch or feasibility report. Exact cause
not retained by credential-safe logger. No new code/tests ran; both candidates and
all prior history remain unchanged. Current ESCALATE is
`data/reports/checkpoint-storage-implementation/bounded-correction-amendment-v1/review.json`.
Three overall request attempts, two completed technical responses; no further request,
retry, production/hash fix or test is authorized automatically. IO2 remains unresolved;
no infeasibility conclusion can be drawn from a failed model completion.
**Latest user direction is focused technical assessment only, before more execution
privileges.** See `docs/checkpoint-storage-io2-technical-decision.md` and its JSON.
Recommend direct-child physical acquisition from independently approved per-prefix
root inodes, retaining all20 D1 logical scripts/configs identifiers. A basename-only
logical namespace would violate D1/IO6. Exact root model must be agreed, not silently
substituted for mandatory single-project-root ancestry if that is intended.
Static SDK review found no prompt truncation evidence, no wire max-output-token cap,
and terminal error metadata discarded by our wrapper; cause of old failure unknown.
A minimal wrapper correction and one further bounded correction are PROPOSED ONLY.
No Worker/candidate test/product change/retry authority is granted by this assessment;
all3 used request attempts, old candidates and ESCALATE history remain preserved.
IO2 is not implemented by a design note; IO3/IO5 unverified, IO6/IO7 unpassed; full
IO1–IO8 review remains required before any task_only ACCEPT. Deployment NOT_AUTHORIZED.
**Subsequent user approval now finalizes per-prefix approved directory FDs/direct-child
physical acquisition with unchanged D1 logical identifiers; no continuous project-root
pathname ancestry claim.** See `docs/checkpoint-storage-prefix-root-correction.md`.
One specific bounded phase is authorized: existing candidate,fresh Pi hashes,one fourth
request30minutes/no retry or revision,then comprehensive IO1–IO8 source/fixture review
before at most one pinned240s full suite. Three used requests/history stay preserved.
SDK changes only safe terminal-error metadata and request/context bounds;same model,
controller-free channel and routing. No additional unrequested correction cycle,
unsafe filesystem experiment,source dependency rewrite,main transfer or research work.
Root decision does not satisfy IO3/IO5/IO6/IO7 or grant implementation acceptance.
**Latest prefix-root phase is ESCALATE solely on a new test oracle in the full suite**:
389tests,1failure/0errors/0skips,62.611s process;all355 inherited tests and3564expanded
cases pass. Reviewed Worker source785lines/test815lines applied only3candidatefiles;
9specified nodes unchanged. Root mechanism,IO3/IO5 controls andIO6/7positive pipeline
now have actual passing evidence;no overallACCEPT because IO8requires zero failures.
New sourceFIFO testglobally blocks _readable for a valid configs file before its later
scripts FIFO target. Pi missed that oracle ordering before execution;no FIFOdata-open
observed. No fix/rerun permitted automatically;fourrequests andsixconservativefullruns
used. Currentreview: `data/reports/checkpoint-storage-implementation/prefix-root-correction-v1/review.json`.
Retain code/failing test/allrawresults;no extra cycle,maintransfer or researchauthority.
**Subsequent user direction now authorizes a Pi-authored test-oracle-only correction
and one full regression**,not a Worker/production cycle. See
`docs/checkpoint-storage-oracle-only-correction.md` and its frozen plan/diff. Only the
named sourceFIFO test method may change;singleton zero-byte regular-file positive
then same-size FIFO ensures target-specific type refusal. Allproduction,othermethods,
355dependencies andfailed389history retained. FourWorker attempts/sixfullruns stayused;
one seventh full240s run only,after freshcomplete review. No extraWorker unless a new
productiondefect is identified and appropriate consequential authority is settled.
No automatic furtherfix/repeat;allIO1–IO8/orderedlimitations stillgateACCEPT.
**D3 now has ACCEPT/task_only in the isolated candidate after that authorized Pi-only
oracle correction:**389OK,0failures/errors/skips,53.292s unittest/55.890s process under240s.
All355 baseline IDs/bytes and3564expanded cases preserved;production4be67c43 andcandidate
API documentation unchanged;only namedtestmethod altered. Pi reviewed allIO1–IO8 andall8
currentWorker limitations exact/inorder withoriginalscopequotes;submission evidence
issues resolved without rewriting reports. Currentdecision:
`data/reports/checkpoint-storage-implementation/oracle-only-correction-v1/review.json`.
FourWorker attempts andsevenconservativefullruns retained;no newWorker or budgetreset.
AlloldESCALATEs andfailed389 preserved,notretroaccepted. D1/D2 remainclosed/protected.
Deployment NOT_AUTHORIZED. D3 main source/test transfer/regression needsseparateapproval;
no automatic profiling/training,heldout,stage/commit/push/release/rollback or extra cycles.
No unsupported safety claim, unsafe testing, controller investigation,
accepted-dependency rewrite, main transfer or new research authority.
The preceding **D2 concluded; D3 preparation-only** direction is retained in
`docs/checkpoint-storage.md` and `configs/checkpoint-storage-scope.json`. No new
D2 error has been identified; do not start another D2 correction/re-audit/Worker or
security-controller investigation absent a concrete new error. D2 validates decoded
state; D3 owns secure storage/retrieval, decoding-time bounds and D1/D2 integration.
Prepare D3 as one substantive deliverable without modifying accepted dependencies.
Its earlier requirement to settle every technical detail before implementation is
superseded by the latest implementation-first approval above, not by scope expansion.
All candidate executions still require Pi's prior semantic/safety review and exact
pinned command, scratch and bounds. Production execution remains unauthorized.
All earlier ESCALATE decisions/budgets and failed attempts remain preserved.
Historical explicitly bounded D2 correction `6efb0ae5-6a62-4376-91e9-7b358c51013f` remains **ESCALATE at1/1**:
its isolated283-test suite passes, but required schema/applicability/diagnostic/coverage/
disclosure contracts remain unmet. No rejected payload source transferred to main;
D3/profiling/training remain blocked. See `docs/checkpoint-payload-authorized-correction.md`.
The Worker-free37-part preparation matrix and correction assignment are recorded in
`docs/checkpoint-payload-blocker-matrix.md` and `docs/checkpoint-payload-correction-assignment.md`.
The subsequently approved matrix-guided task `eb5065fc-234e-4d6f-ac63-b788d55b1502`
is now **ESCALATE after its initial30minute timeout**:320tests/6failures/14errors,
with all283old cases preserved and no implementation fixes. Raw records prove forbidden
probes/subsets and external helper writes. **One revision remains unused; not budget
exhaustion.** Do not automatically spend it/reopen/replace a task whose historical
process violations block acceptance. See `docs/checkpoint-payload-matrix-correction.md`.
The subsequently authorized text-only correction is also **ESCALATE at1/1**. Two
completed no-tool SDK responses produced real fixes/coverage without Worker commands;
Pi screened private fixture RNG changes before execution. Final337tests/2failures/
0errors/0skips in46.429s: both new negatives expose pre-existing Adam step4.5 accepted
with updates4. All320inherited cases pass and3140expanded assertions succeed, but raw
identifier diagnostics and full37-row evidence remain incomplete. Preserve this useful
isolated artifact; no automatic third response, numerical Pi fix, task replacement or
main transfer. This was response/revision-count exhaustion, not another Worker timeout.
See `docs/checkpoint-payload-text-correction.md` and its independent review.
Do not reopen/reset/automatically replace exhausted tasks. Checkpoint/provenance task
`e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a` remains ESCALATE/unaccepted in isolation;
both corrective revisions used after operator-authorized same-task reopening;
final benchmark still has six errors, no draft transferred to main. Further source
work requires an explicit consequential plan; no automatic task/budget reset. See
`docs/training-harness-review.md`, `docs/differentiable-models-recovery.md`,
`docs/evaluator-integration.md`, `docs/implementation-roadmap.md`
and `docs/protocol-lifecycle.md`. This does not authorize profiling/training or
deployment. Remaining full training loops/optimizer execution, checkpoints beyond
manifest contracts, reproducibility integration, execution settings, tuning policy,
profiling, and comparative training remain pending and not authorized. Read `configs/protocol-lifecycle.json` and
`docs/protocol-lifecycle.md`; do not interpret a state/version change, graph
construction, an audit, or milestone implementation as permission to train.

## Version and authorization rules

- Protocol version, lifecycle state and execution revision are separate identifiers.
  Component/schema versions are not the global experiment version.
- Preserve frozen snapshots/manifests under `configs/protocols/`; append amendments
  with approval, rationale, changed rules and held-out exposure. Never overwrite
  earlier baselines to match later code/configs.
- Major bumps cover split/scope/relevance/candidate/core-comparison changes; minor
  bumps cover substantive methodological amendments/additions; patches are
  behavior-preserving clarifications. State changes alone do not bump versions.
- Bounded profiling needs explicit scope, limits, provisional settings, optimizer
  permission and fresh outputs, after prerequisite regression review. No test model
  evaluation or performance-driven tuning during profiling.
- Before comparative runs, freeze pending initialization/execution/tuning settings,
  verify implementation and record explicit training authorization. Run artifacts
  must include baseline/execution identifiers and hashes, authorization and complete
  code/data/config provenance. Deployment remains separately NOT_AUTHORIZED.

## Read before changing the pipeline

- `README.md`: structure, dependencies and reproduction commands.
- `docs/evaluation-protocol.md`: pre-training protocol and unresolved decisions.
- `configs/dataset-scope.json`: approved full-data scope and execution order.
- `configs/evaluation-metrics.json`: frozen metrics, ranking and reporting rules.
- `docs/reproducibility-and-uncertainty.md` and `configs/reproducibility.json`:
  agreed Block 3 seeds, provenance, resumption, failure and uncertainty rules.
- `docs/compute-and-tuning-plan.md` and `configs/compute-plan.json`: conditional
  Kaggle capacity, profiling-first validation and budget-freeze approach.
- `docs/training-semantics.md` and `configs/training-semantics.json`: agreed
  sampling eligibility, required regression checks and pending training choices.
- `docs/candidates-and-cold-start.md`: authoritative candidate/filtering policy.
- `docs/models-and-baselines.md` and `configs/model-suite.json`: agreed core
  membership, optional comparisons, attribution limits and model claim boundaries.
- `docs/kg-only-scoring.md`: frozen KG-only formulas, missing-channel policy,
  whole-user fallback and required hand-calculated tests.
- `docs/heterogeneous-encoder.md`: frozen relation operator, inverses, activations,
  initialization and required tests; execution/sampling settings remain pending.
- `docs/lightgcn-protocol.md`: explicit core Option B fixed-graph training-edge
  reuse, exception to heterogeneous masking, variant labels and comparison limits.
- `docs/kg-schema.md` and `configs/kg-v1.json`: implemented structural allowlist.
- `docs/kg-input-audit.md`: actual split/metadata findings and limitations.

Keep documentation, configuration, implementation and tests consistent. Do not
resolve protocol ambiguities silently. Record amendments before new experiments;
never choose thresholds, features or metrics based on test performance.

## Delegation and execution roles

The current `docs/worker-delegation-policy.md` applies here: native agy is the default,
2 + 2 + 2 is a prospective maximum with separately authorized rounds, and SDK requires
explicit user approval before every request. Historical channel preferences below do
not override this policy or supply new implementation/execution authority.

**Standing user preference:** Pi prioritizes planning/assignment/execution management,
receiving results/correction coordination and independent verification. Delegate clear,
bounded utility/test/local-fix implementation to agy within approved scope,grouping
related code and tests rather than micromanaging tiny requests. Pi retains review and
acceptance;no reduced verification or new Worker authority for preparation-only work.
See `docs/delegation-and-execution.md#standing-user-direction-delegate-bounded-implementation`.
**Worker scope/dependency rule:** agy must ask Pi before implementing/installing any
package,framework,structure or other change outside its task. Pi decides ordinary
in-authority choices without repeatedly asking the user;record approval/rejection and
scope/verification amendments before resuming. No silent scope expansion,extra calls,
budget resets or override of frozen science/security/execution permission.

Read and follow `docs/delegation-and-execution.md` before delegating or executing
work, including under `multiautoresearch/`. Pi plans, orchestrates and independently
reviews/accepts; agy implements/fixes and executes only explicitly authorized
workloads. **Standing user authorization:** prioritize implementation + regression
testing + profiling on Musical_Instruments; handle in-scope noncritical setup and
verification recoveries autonomously, without repeated user confirmation. Record
ESCALATE when required, preserve its evidence, and continue only under a prospectively
scoped recovery with unchanged independent-review rigor and agreed limits. Ask only
for critical decisions or permissions not already granted; profiling execution still
requires its bounded plan/prerequisites/explicit authorization. Read the standing
authorization in `docs/delegation-and-execution.md` before asking to repeat a
noncritical verification. Fold small fixes into substantive revision cycles, and
delegate independent subsystems separately. Kaggle is the
intended GPU backend, not an implicit authorization. Framework lifecycle labels
and inherited LM/Movie4U prompts cannot override Amazon4U protocol or approvals.

## Review-to-correction preparation

Apply this workflow across **all Amazon4U stages**, not only D2. Before Worker
implementation or substantive corrective delegation, Pi performs a brief, proportionate
**Worker-free preparatory review** and prepares a requirement/evidence matrix against
the stage's whole original contract. For corrections, compile a **detailed part-level
blocker matrix**, not only criterion-level statuses: identify each relevant function,
guard or invariant, exact code/test locations, executed-versus-static reproductions,
valid controls, intended outcomes, required evidence and closure status. Track implementation status separately from test/evidence status:
verified code may need additional regression tests, not rewriting. Identify preserved
invariants and baseline references before delegation; independently check returned code
for unintended changes. Changes to verified logic must be intentional, justified, within
authorization and reverified. Adding tests does not authorize weakening existing assertions
or editing protected tests. Use dependency-ordered work and explicit closure checks, not
generic “fix everything” requests or green regression alone. Before independently
executing Worker tests, screen changed test/setup code for prohibited workloads,
paths, global setters and other scope violations; an approved command does not authorize
forbidden embedded behavior. If that blocks mandatory evaluation, record the block,
never ACCEPT incomplete verification or run it just to obtain a result. Reconcile all
relevant invocations with inventory and raw records; supplement known decoder gaps
without reopening tooling.
See `docs/review-to-correction-preparation.md` for the standing agreed procedure.
Do not require the user to repeat this instruction at later stages. Work within existing
authority without unnecessary approval loops; preparation itself grants no additional
execution, revision budget, task reopening, research or deployment authority.

## Historical, D2-scoped text-only patch direction and authorization

Subsequent user `procced` separately authorizes the bounded D2 text-only correction,
recorded prospectively in `docs/checkpoint-payload-text-correction.md` and its plan.
It did not reopen old tasks or enlarge original criteria; main/D3/training/deployment
remain unauthorized. The bounded round has now stopped ESCALATE at1/1; no further
Worker/source correction is authorized automatically. Retain its fixture-screening and
independent-review lessons. Historical capability-only verification below did not grant
its implementation authority.

The user supports a genuinely text-only configuration, not a new security controller
or agent-system modification. A bounded existing-SDK direct-completion check succeeded;
see `docs/text-only-channel-check.md`. Native agy remains unchanged and is NOT tool-free.
Send scoped relevant functions/dependencies, exact applicable frozen requirements and
test evidence rather than the whole codebase each iteration; bind patches to independently
checked baseline/dependency hashes. Missing context must be requested, not guessed.
Pi must review paths, contents, protected invariants and test changes before mechanically
applying any future authorized patch, and screen the entire selected execution path before
running it. Never execute inherited unsafe fixtures merely as a bootstrap. D2 patches
remain limited to its three approved files; other stages use their own approved scope.
The check used synthetic content only, with no actual D2 patch/test/Worker execution.
It grants no implementation/revision-budget/deployment authority and does not inherit
native delegation receipts or permit bypassing mandatory acceptance gates. Do not restart
controller/tooling research. The newly recorded D2 authority applies only to that bounded
text-only correction; further work outside it still requires appropriate authorization.

## Historical next-D2 direction — frozen contract, no standing SDK authority

A subsequent user `procced` now separately authorizes the new bounded round in
`docs/checkpoint-payload-a04-d01-correction.md` and its pinned plan. Preserve the337-test
baseline; prioritize A04/D01 with the unchanged text-only channel. This new authority
supersedes the prior discussion-only hold for that round, not old decisions or budgets.

User-agreed after the337-test text-only round: preserve that candidate and suite;
do not regenerate tests, rewrite verified fixes or alter unrelated protected guards.
No further Worker architecture/controller work is needed. Under a separate new bounded
authorization, reuse patch-only proposals and prioritize A04integer-valued Adam steps
and D01bounded diagnostics. Address ledger-backed coverage gaps additively; missing
tests are not new implementation errors or permission to rewrite source.
Do not introduce technical requirements beyond original C1–C10/frozen D2specification.
Every proposed new test must demonstrate an existing requirement. The37-part matrix
maps evidence; it cannot independently expand acceptance. Flag any unsupported reviewer
request without changing original records or waiving original mandatory gates.
See the next-round direction in `docs/checkpoint-payload-text-correction.md`.
This discussion grants no Worker/source/test execution, budget reset or task reopening.

## Mandatory decision recording

- Always record project decisions made during discussions and implementation;
  do not leave agreed choices only in chat or rely on conversation memory.
- Update the relevant document in `docs/` and configuration, where applicable,
  before implementing or running work governed by that decision. This applies
  throughout the project, not only to evaluation and training.
- Record what was decided, why, its scope, and its status: proposed, agreed,
  implemented or superseded. Distinguish user-approved decisions from agent
  recommendations and unresolved questions; never present a proposal as agreed.
- Record changes to earlier decisions with their rationale and affected artifacts;
  preserve the previous decision's context instead of silently rewriting history.
- Keep implementation progress, verification results and remaining work current
  in the relevant documentation. Link supporting scripts, tests and reports.
- Before concluding a task, check that all new decisions and implemented changes
  are documented. Report unresolved choices explicitly. Documentation updates
  do not authorize staging, committing or pushing without a user request.

## Data and graph invariants

- Raw data in `data/raw/` is immutable. Preserve downloaded files, publisher split
  assignments and provenance. Never randomly re-split or move a held-out target.
- Join reviews/interactions and products using `parent_asin`; users use `user_id`.
  Benchmark CSVs contain rating/timestamp interactions, **not full review text**.
- Keep category graphs separate. Cross-category histories/embeddings are not
  approved: one category's training event can occur after another's target.
- KG-v1 nodes: User, Product, CategoryPath, Brand. Relations: `rated`, `belongs_to`,
  `category_child_of`, `has_brand`. Only training users/products become nodes.
- Metadata allowlist: `parent_asin`, observed `categories` path prefixes and
  explicit string `details.Brand`. Do not treat `store`/`Manufacturer` as brand.
  Category identities are complete JSON path prefixes, not ambiguous leaf names.
- Missing metadata does not justify fabricated edges or silently removing items.
  Fail on duplicate training pairs/metadata IDs and invalid structural inputs.
- KG-v1 stores all training ratings as observations, not binary likes. It has no
  approved node features/embeddings. The >=4 graph is a distinct future artifact.
- Use bounded-memory/chunked or disk-backed processing. Existing DuckDB audits
  cap memory at 512 MB and spill to temporary workspaces. Do not load full datasets
  into RAM or download additional large collections without explaining the cost.

## Leakage safeguards

- Training graphs, inverse edges, user profiles, statistics and graph embeddings
  derive only from permitted training data. Never expose held-out labels/reviews.
- A target review/rating must not become an input for predicting itself. The
  heterogeneous model's agreed training-target shortcut safeguard is batch
  positive-edge masking:
  exclude direct forward/reverse/duplicate equivalents before or during sampling
  at every hop, not after neighborhood retrieval. Preserve unrelated edges and
  valid remaining multi-hop KG paths. Both BPR scores, if BPR is adopted, use
  the same masked view; no cached unmasked representations may bypass it.
  Interaction encoder edges are binary, with no rating-value inputs. Masks are
  batch-scoped, not permanent history deletion. Add regression checks for effective
  adjacency, sampled neighborhoods and preservation of valid metadata paths.
  Core LightGCN is an explicit approved exception: fixed condition-positive
  training graph with supervised-edge reuse and no batch masks. Held-out edges
  are still forbidden in both models; this is not validation/test leakage.
- Exclude `average_rating`, `rating_number`, bestseller ranks, `bought_together`,
  full details dictionaries and dataset-wide/review-derived aggregates. Other
  metadata/features require an audited, explicitly registered allowlist change.
- Fit preprocessing, vocabularies, normalization, PCA and learned encoders only
  on permitted training data. Record provenance for external frozen encoders.
- Validation selects hyperparameters/checkpoints; test never selects them.
- Preserve regression checks showing held-out changes and excluded-field changes
  cannot alter graph/model-input tables.

## Feedback, candidates and cold start

- Primary: all-rating recorded behavior. Sensitivity: training/held-out ratings
  >=4 as positive feedback. Low ratings/unobserved items are not verified negatives.
- Preserve raw splits in both conditions; do not replace low-rated held-out items.
- For >=4 training, keep positive history separate from all-rating observed
  training history. Positives/forward/reverse-positive edges use >=4 only;
  negative eligibility is the condition warm pool minus ALL user training-rated
  items. A globally warm item rated below 4 by that user is forbidden as their
  negative. Never consult held-out data. Detect/report empty pools without
  relaxing exclusions. Block 2 uses uniform eligible sampling, one negative per
  positive; skip/report empty-pool positive updates and fail if no trainable
  examples remain. Never create synthetic or regularization-only updates.
- Warm pool derives solely from permitted training interactions. Current >=4
  policy uses condition-specific warmth; explicitly report the resulting pool
  difference, not a fixed-candidate threshold-only ablation.
- Full candidate ranking, **no sampled evaluation negatives**. Use identical
  candidates and relevance rules across methods within each condition.
- Validation filters all user training items. Test filters all user training
  **and validation** items, including low ratings. Validation IDs are allowed only
  for candidate subtraction, not model updates, features or message passing.
- Freeze the selected model for test; no train+validation refit under this policy.
- Exclude cold/history-overlapping targets from primary warm metrics and report
  counts, proportions and eligible-user denominators. Never selectively restore
  a target or silently omit empty positive profiles. Their agreed common fallback
  is evaluation/inference-only condition-positive training popularity with normal
  candidate/history filters. No synthetic personalized training positives/edges,
  no popularity-vector model input and no personalized-prediction claim for these
  users. Report non-personalized fallback counts and subgroup metrics separately;
  scoring failures must still fail.
- A heterogeneous GraphSAGE-style recommender using learned node-ID embeddings
  is transductive, not automatically an inductive cold-start model. Learned-ID
  inputs and encoder initialization are now agreed in Block 1; no learned
  features/embeddings have been generated in KG-v1.
  Report capability based on the implemented inputs/inference pathway, not the
  GraphSAGE name alone. See `docs/models-and-baselines.md`.
- Cold performance is not currently supported/evaluated. Only register it after
  verifying metadata-inductive inference, no cold-ID embeddings, learned warm
  metadata vocabulary and training-only preprocessing. ID-only models are N/A,
  not valid cold baselines. Report unsupported items/unseen metadata separately.
- Audit unknown users rather than assuming 5-core makes them impossible. Full
  5-core does not imply training alone is 5-core. Timestamp ties and static
  metadata availability remain documented limitations.

## Agreed core models

- Core: Popularity, LightGCN, KG-only and Heterogeneous KG + GNN.
  LightGCN is not optional. Block 1 freezes 64d LightGCN
  with 3 layers/mean 0–3/dot product, and 2-layer 64d learned-ID heterogeneous
  GraphSAGE-style/dot product. Sampling/profiling execution details remain pending.
  Exact heterogeneous aggregation/initialization
  are now frozen in `docs/heterogeneous-encoder.md`: relation-specific transforms,
  one node-type self term, within-relation means and mean over active relations;
  explicit inverses, ReLU then identity, Normal(0,std=0.1) IDs, Xavier-uniform
  gain-1 matrices, zero self biases, no dropout/layer/output normalization.
  Interaction inputs are binary and rating-free.
- KG-only uses training-warm IDF profiles and cosine, no rating weighting or
  popularity bonus. Tune category/brand mixtures in order 1/0, 0.75/0.25, 0.5/0.5
  by validation NDCG@10 per category/condition; exact ties select the first entry.
  Keep profile/IDF/missing-channel/candidate rules identical across trials.
  IDF is 1+ln((N+1)/(df+1)) over all condition-warm products, with distinct-product
  df. Normalize product channels separately; user channels are normalized sums
  over distinct positive-history products. Missing-channel cosine is zero, with
  no per-item mixture renormalization. Whole-user condition-positive popularity
  fallback applies only if no positively weighted user profile is usable, never
  per-item or for scoring failures. Report fallback counts/reasons/subgroup metrics.
  Empty-positive-history fallback is shared across methods; KG-only's additional
  missing-metadata fallback is distinct. Neither creates training examples or
  model input features. Do not use test outcomes to select weights.
- Core LightGCN uses approved Option B, exact propagation on the fixed training
  graph without batch target masks. Profile execution, not methodological identity.
  Masked Option A is a target-edge-masked variant, not the core baseline. Sampled
  propagation requires an explicit amendment and sampled-variant labeling.
  Heterogeneous KG + GNN retains masking; do not attribute comparison gains solely
  to metadata when architecture and training-edge use also differ.
  GraphSAGE batch size/fanout are profiling starting points, not frozen constants.
- No MF baseline or metadata variant is included in the active study. A matched
  encoder without metadata relations is a recommended
  optional control; do not claim metadata attribution from architecture-changing
  comparisons alone. No optional experiment has a finalized runnable configuration.

## Agreed Block 2 training recipe

- Learned methods: mean stable BPR via softplus(s_negative-s_positive), plus the
  explicit endpoint-only penalty below; uniform eligible negatives, one per positive.
- Adam, betas (0.9,0.999), epsilon 1e-8, zero weight decay; constant learning rate
  0.001, maximum 100 epochs. Stop after 10 validation checks without strictly better
  validation NDCG@10; exact ties retain the earliest checkpoint and do not reset
  patience. Failed validation is a failure, not a valid patience check.
- Empty negative pools skip/report affected positives; fail if no examples remain
  trainable. Invalid empty batches have no optimizer/regularization-only update.
- Model-specific batch sizes, fanouts, validation frequency, LightGCN initialization
  and tuning budgets still require specification/profiling before training.
  Numerical approvals do not authorize training until the overall protocol is ready.

## Agreed endpoint regularization

- Learned methods use explicit 1e-5 times the mean squared L2 row norm of unique
  supervised base-ID endpoints: users plus union of positive/negative item IDs.
  Deduplicate per typed ID across roles/examples; no extra dimension or half factor.
- Do not select rows from the sampled computation graph or full propagation table.
  Brand/Category rows are always excluded; a neighbor user/product row is included
  only if it independently is a supervised endpoint. Use base rows, not propagated
  representations.
- Transformation matrices/biases and global optimizer weight decay are zero.
  No hidden table-wide or decoupled weight decay may replace the explicit penalty.
  Ranking-loss gradients through neighbors remain allowed; do not detach them.
- Test endpoint deduplication, fanout independence of the penalty, base-row selection
  and optimizer groups. No synthetic/regularization-only update for invalid empty
  batches. Other training settings remain pending; see training-semantics documents.

## Frozen metrics and reporting

- Primary: NDCG@10. Secondary: HitRate@10 and CatalogCoverage@10.
  Supplementary: NDCG/HitRate at 5 and 20. Select checkpoints/hyperparameters with
  validation NDCG@10 only; do not choose K using test outcomes.
- Binary relevance follows the feedback condition. Macro-average NDCG/HitRate
  over eligible users; coverage is a unique-recommendation union divided by the
  condition's entire warm pool, not a per-user average.
- Filter history before Top-K. Sort score descending, then parent_asin ascending
  for exact ties. Reject non-finite scores. Do not pad lists shorter than K;
  report affected-user counts and never silently drop scoring failures.
- Report separately by category × feedback condition; no pooled primary score.
  Use the same eligible users across methods within a group. Separate validation
  from final test reporting and include eligibility/exclusion, empty-profile,
  fallback and candidate-shortage counts. Label overlapping exclusion reasons.
- Block 3 freezes seeds 42/2026/3407 across learned methods/groups. Record RNG
  streams/workers, environment and reconstructable source provenance (not just
  dirty flags). Fail first on unsupported determinism; explicitly approve/log
  exceptions. Claim resumption only at boundaries independently verified.
- Report all registered metrics per seed, mean ± sample SD, and fallback subgroups.
  Bootstrap fixed seed-averaged per-user test metrics after validation-only selection:
  10,000 shared paired-user resamples, seed 12345, 95% percentile intervals and
  full-test-population point estimates. Do not bootstrap coverage per user or
  conflate user/seed uncertainty. The contrast list is frozen: KG + GNN minus
  Popularity/LightGCN/KG-only for NDCG@10/HitRate@10 per category/condition
  (36 planned intervals). Share resamples across all contrasts/metrics within a
  group; no bootstrap for @5/@20 or coverage and no optional-method contrasts
  without pre-experiment amendment. Unadjusted intervals are descriptive,
  empty populations N/A, failed seeds explicit.
  No hidden population changes, seed replacement or favorable-result retries.
- Other training choices remain unresolved. Conditional 60 GPU-hours/week must
  be verified as permitted/available; reserve roughly 20%. Profile full-ranking
  validation every epoch on Musical_Instruments first, then recheck larger
  categories. Freeze trial budgets/search spaces before comparative results;
  2–3 configurations is only a planning range. Frequency amendments need approval,
  with patience counted in validation checks. Freezing metrics does not authorize
  training or silently reduce dataset/candidate scope.

## Development and verification

Save reusable tooling in `scripts/`, policies in `configs/`/`docs/`, tests in
`tests/`, generated graphs in `data/processed/`, reports in `data/reports/` and
visualizations in `data/visualizations/`. Visualizations are illustrative training
neighborhoods, not random or representative evaluation samples.

From `Amazon4U/`:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q scripts tests
```

When data/graph logic changes, also run the affected full-data audits as feasible:

```sh
.venv/bin/python scripts/audit_kg_inputs.py
.venv/bin/python scripts/audit_candidates.py
.venv/bin/python scripts/validate_kg_splits.py
```

Install dependencies with `.venv/bin/python -m pip install -r requirements.txt`
when necessary; `curl` is required for downloads and Graphviz `dot` for rendering.
Builders do not overwrite existing graphs: use a new `--output` for experimental
builds and validate that artifact explicitly. Preserve source hashes, configuration,
split/condition identity and eventually seeds/checkpoint protocol with each run.

Keep changes small and tested. Do not commit raw data, large generated graphs,
virtual environments or caches. Check root and local ignore rules with
`git check-ignore` before assuming reports are tracked. Do not stage, commit or
push changes unless requested; leave unrelated user changes untouched.
