# Review-to-correction preparation

Current workflow follows [direct single-agent ownership](worker-delegation-policy.md).
Mandatory Pi–agy handoffs and the prospective 2 + 2 + 2 process are retired.
The routing statements below are historical. Preparation and verification lessons
still apply to direct work; they do not require new delegation tasks.

> Retired channel/recovery policy: [worker-delegation-policy.md](worker-delegation-policy.md).
> Native agy is the default for future substantive implementation; SDK is an exception
> requiring explicit user approval before every request. Native recovery maximum is
> separately authorized 2 + 2 + 2, with root-cause review after each substantive
> ESCALATE and mandatory recommendation after #3. Historical D2 text-only examples
> below are scope-limited, not a standing SDK route or authority to reset budgets.

## Decision and status

Standing user-agreed workflow for **all Amazon4U stages and future Worker assignments**,
not only D2: improve preparation and closure of review findings, reduce avoidable Worker
errors and unsupported completion claims, and preserve independent review. The D2
failure pattern motivated this procedure but does not limit its applicability. The user
need not repeat it at later stages. This is not a guarantee of bug-free Worker output.

For the current D2 case, the **37-part matrix and proposed correction assignment are
prepared for authorization review**, not D2 acceptance. See
`checkpoint-payload-blocker-matrix.md`, `checkpoint-payload-correction-assignment.md`
and `data/reports/d2-part-level-preparation/review.json` for evidence and the three
practical-outcome assessments. New reproductions remain unexecuted; no Worker or
candidate tests ran during preparation. This process decision grants no additional
implementation, candidate-execution, revision-budget, research or deployment authority.

## Before implementation or substantive corrective delegation

Pi performs a brief, proportionate Worker-free source/evidence review and prepares a
requirement/evidence matrix against the **whole original contract for that stage**,
not only the latest discovered bugs. For new implementation, identify planned component
and test locations; for corrections, inspect current complete functions, relevant
callers/dependencies and tests and build a **detailed part-level blocker matrix**.
Track individual relevant functions, guards and invariants, not only criterion-level
labels. Each part needs its verified/failed/unverified status, supporting evidence and
baseline reference, intended implementation disposition, separate test/evidence gaps,
and explicit closure conditions. Preserve original criteria,
reports, decisions and exhausted budgets unchanged. D2's C1–C10 remain D2-specific;
do not transplant its schemas or criteria into a different stage.

Each row must include:

- Original requirement and acceptance criterion.
- Exact current source location and relevant caller/test locations.
- Concrete error-demonstrating case, marked **executed reproduction**, **static
  finding**, or **not yet verified**; never imply an unexecuted case was run.
- Valid positive control and independent negative/mutation cases.
- Expected outcome: accepted owned result or the intended bounded domain error and
  guard, rather than an unrelated earlier rejection.
- Mandatory evidence for ACCEPT, present evidence, remaining gap and closure status.

Choose coverage appropriate to each stage, including correctness, integration, failure
paths, boundaries, resource constraints and truthful disclosure. For D2 specifically,
mandatory areas include Adam schema/types; canonical/noncanonical role validation
and canonical active-CUDA rejection under the CPU-only profile; bounded malformed/
huge-integer diagnostics for actual and expected inputs; ownership/input immutability/
output independence including late failures; complete model/RNG/sampler/selection
oracles and matrices; resource accounting; and truthful execution disclosure.

## Preparatory-review completion conditions

Preparation is ready for a bounded implementation/correction proposal when Pi records:

1. **Pinned baseline and complete requirement mapping:** identify the exact artifact
   reviewed; map every original mandatory criterion to relevant parts and evidence,
   including already verified parts. Do not substitute a list of today's bugs for the
   full contract.
2. **Completed part-level matrix:** each relevant part has a source/test location,
   verified/failed/unverified status with evidence, an implementation disposition
   (preserve, targeted fix or investigation), separate test gaps and closure conditions.
3. **Actionable controls and oracles:** each blocker/test gap has a concrete valid
   control and negative/mutation case where applicable, intended outcome/guard and
   criterion. Distinguish previously executed cases from static findings and proposed
   tests. Missing runtime proof must have an explicit prospectively authorized check;
   preparation does not require running unapproved probes or fixing the code.
4. **Preservation and impact plan:** specify verified invariants/baseline references,
   allowed implementation changes, additive protection tests and affected callers or
   dependencies requiring revalidation. Missing tests alone do not imply code changes.
5. **Evidence reconciliation:** reconcile relevant reports with available inventory/raw
   records, retain all observed attempts and identify remaining coverage/decoding gaps.
   Any material gap needs a bounded resolution plan; never imply complete execution
   history or treat Worker claims as verified facts.
6. **Executable bounded handoff:** document dependency order, implementation/test-only
   work packages, proposed paths/commands, resource/time/revision limits, required
   closure/full-regression checks and independent review gates. Distinguish existing
   permissions from new authority needed; leave no unrecorded ambiguity requiring the
   Worker to invent an acceptance rule or silently expand scope.

Pi then records **preparation ready** with artifact links and remaining technical
blockers. If missing evidence or a scope/contract ambiguity prevents a bounded handoff,
record **preparation blocked** and the specific missing input/permission instead.
Do not search indefinitely or claim that no further bugs can exist. Known defects may
remain: this milestone means the correction is well specified, not that the artifact
passes. It is not task ACCEPT, new Worker authorization or deployment approval.

## Preserve implementation separately from strengthening tests

Track **implementation status** separately from **evidence/test status** in the matrix.
Verified implementation can remain unchanged while additional regression tests are
needed to protect its existing behavior. Missing coverage alone is not a reason to
rewrite implementation. Use explicit dispositions such as preserve implementation /
add tests, targeted implementation fix / add closure tests, or investigate missing
evidence before deciding whether code changes are needed.

Before delegation, identify verified invariants, their source locations and baseline
snapshot/hash references, and the intended implementation-change boundary. Adding
regression coverage within authorized paths is compatible with preserving verified
implementation; it does not authorize weakening existing assertions or modifying
protected tests. Tests and execution remain subject to the original scope/commands.

Pi must independently compare the returned implementation with that baseline, not
infer preservation from passing tests or Worker claims. Review complete changed
functions and affected callers/dependencies. Any change to verified logic must be
explicitly identified, justified, within authorization and independently assessed;
unexplained or unrelated changes are not silently accepted. Reverify affected
invariants and run the mandatory final regression/integration checks even when their
implementation was intentionally left unchanged.

For example, an Adam `param_names` correction should be narrowly scoped. Add or
strengthen tests protecting already verified Adam update/group/step guards and
selection behavior without rewriting those guards merely to add coverage. Compare
those implementations against the baseline; any necessary changes must be intentional,
recorded and supported by fresh evidence. The same distinction applies at every stage.

## Correction sequencing and evidence

Use the matrix to prepare dependency-ordered, bounded implementation or correction
with explicit closure checks, not a generic instruction to “fix everything.” Plan valid
controls and independent positive/negative oracles before spending implementation/revision
budget. Each finding must map to a correction and verifiable closure evidence; Worker
claims and green regression alone do not close it. Review complete relevant code and
actual results independently before deciding that a requirement is satisfied. Screen
changed test/setup code for prohibited execution and scope violations **before** running
it, not only afterward; a pinned full-suite command does not authorize forbidden
operations embedded in tests. A blocked mandatory evaluation is not ACCEPT.

Use existing authorization for routine in-scope work without unnecessary approval loops.
Any additional execution or expanded scope still requires prospective authority; D2's
existing full-suite-only command restriction remains unchanged. Pi retains independent
review and acceptance authority and all original verification/deployment gates.

Reconcile disclosure against all relevant invocations, inventory **and raw records**, retaining
failed, repeated, cancelled, timed-out and successful attempts. Inventory decoding is
partial/version-specific; its known global-budget gap requires supplementary bounded
raw-evidence checks when needed, not another tooling project. Flags are navigation,
not proof of violations or completeness.

Preparation does not reopen an exhausted task, authorize an automatic replacement or
reset budgets. Where new correction authority is required, first prepare the matrix and
closure plan rather than request a generic retry. Across stages, incomplete or unresolved
mandatory evidence still blocks acceptance; task acceptance is not deployment approval.
D3 and research execution remain behind their existing acceptance and authorization
gates. See `checkpoint-payload-authorized-correction.md` for the current unaccepted
D2 artifact and immutable decision references.

## Historical D2-scoped text-only patch context and application boundaries

The user agreed this architecture is promising but explicitly withheld new D2 launch
authority. Do not build a controller merely to impose patch-only behavior. Verify an
existing genuinely text-only model configuration instead; never call native agy tool-free
while its native tools remain. The one-shot existing SDK direct-completion verification
and synthetic patch demonstration succeeded; see `text-only-channel-check.md`.

For any future authorized text-only iteration, provide relevant complete functions,
necessary callers/dependencies, exact applicable frozen requirements and pertinent
failure/control evidence, not the entire codebase every time. Retain all original
requirements as authoritative; scoped context is not a waiver or semantic rewrite.
Require baseline hashes and verify all changed-file/relevant-dependency hashes before
application; reject drift instead of applying fuzzily. Request missing context explicitly.

Pi checks exact allowed paths, patch contents, protected implementation and test changes
before mechanical application. For D2 only its three approved files are eligible; no
undeclared helpers, path escapes, renames/deletions or assertion weakening/skips. Review
all selected test/setup execution paths, including inherited fixtures, for prohibited
actions such as global `torch.manual_seed`; do not reexecute the unsafe320-case artifact
just to bootstrap. Static screening supplements full semantic review, not a universal
unsafe-code detector or sandbox. No actual patch or tests ran in the synthetic demo.

A direct API channel does not automatically inherit native delegation receipts. Future
D2 authority must explicitly predeclare bounded requests/context/patch/history records
and all mandatory independent inspection, verification and acceptance gates without
silently bypassing them. Do not alter old task reports, decisions, criteria or budgets,
and do not turn this into a tooling project or automatic D2 relaunch.

## Bounded text-only D2 follow-up — result, not new authority

Subsequent separate user `procced` authorized the pinned text-only correction in
`checkpoint-payload-text-correction.md`. That round stopped ESCALATE at1/1, with no
native Worker tools/commands and no new controller. All320inherited tests pass, but
final337tests retain2valid fractional-Adam-step failures;3140expanded assertions pass
without closing all37rows. D01raw identifier bounds and coverage remain incomplete.
Preserve the useful fixes/tests and exact current reports; do not manufacture green,
change acceptance criteria, extend budgets or automatically replace the round.

The two private-generator fixture edits were independently reviewed/applied before
safe red discovery. One ambiguous anchor was rejected rather than silently patched
into an older helper. Reports honestly declared partial work. Capability removal
successfully prevented direct Worker probes/writes; it did not guarantee complete
implementation/coverage. Neither response timed out. Final next-source work requires
appropriate bounded authority; no D3/main transfer/training/deployment follows.

## Frozen-contract evidence, not expanding requirements

The user explicitly reaffirmed this for the next D2round: no additional Worker
architecture work; preserve the337-test candidate and verified fixes; prioritize the
well-defined A04/D01implementation defects under new authorization. Use ledger-backed
coverage additions without confusing missing evidence with a demonstrated source bug.
Do not regenerate the suite or rewrite unrelated protected guards to gain coverage.

C1–C10 and the original frozen specification are the acceptance authority, not an
expanding reviewer wish list. Every proposed test must demonstrate an existing contract
clause. A matrix/ledger item is evidence navigation, not independent authority to add
technical requirements. If a request lacks an original-contract basis, identify the
review discrepancy prospectively rather than silently create an acceptance blocker.
Keep original reports/criteria/decisions unchanged and retain all mandatory independent
review, execution-safety and verification gates. This direction grants no new execution
or revision budget; see `checkpoint-payload-text-correction.md`.

## Matrix-guided execution follow-up

The subsequently approved attempt `eb5065fc-234e-4d6f-ac63-b788d55b1502` is ESCALATE;
see `checkpoint-payload-matrix-correction.md`. The matrix's six implementation blocker
groups were reproduced and protected code stayed unchanged, but the Worker timed out
before fixes and used prohibited probes/subsets/external helper scripts. One revision
was deliberately left unused because source changes cannot undo historical process
violations. Thus the third practical outcome—compliant correction execution—was not
achieved; a detailed assignment alone is not proof of effective Worker follow-through.
Pi's benchmark also ran new global-seeding fixture operations before they were noticed;
the screening-order mistake is explicitly recorded. Do not respond with another generic
automatic retry or reopen tooling. Historical preparation assessments remain unchanged;
they were qualified plan-level assessments, not promises of execution success.
