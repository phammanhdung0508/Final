# Review-to-correction preparation

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
actual results independently before deciding that a requirement is satisfied.

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
