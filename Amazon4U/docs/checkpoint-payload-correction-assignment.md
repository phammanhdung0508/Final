# D2 matrix-guided correction assignment, proposal, NOT_AUTHORIZED

## Status and prerequisites

This is a prospective **whole-D2** correction proposal, not a new active task, historical
reopen or budget reset. User authorization currently covers preparation only. Do not
invoke a Worker, execute a candidate, modify candidate source/tests, create the proposed
checkout or start D3/research merely because this document exists. Current task
`6efb0ae5-6a62-4376-91e9-7b358c51013f` remains ESCALATE at 1/1; all earlier decisions,
reports, criteria and artifacts are preserved.

Inputs:

- [Part-level matrix](checkpoint-payload-blocker-matrix.md): all 37 rows, not just the
  failed implementation rows. Each row defines source/test locations, valid controls,
  intended outcomes and closure evidence.
- `data/reports/d2-part-level-preparation/baseline.json`: source/function/test hashes,
  exact original C1-C10 strings, historical benchmark and original command.
- `data/reports/d2-part-level-preparation/evidence-audit.json`: original byte/hash
  references for failed/repeated runs; not an edited Worker report.
- Frozen full 859-line/59605-byte original specification, SHA256
  `a648d97f73e1213eb37f1a87fe56e76f1171f59baedcce6cd62b2c66f7dd64c4`.

A future task must include the original full criteria/specification unchanged and
this prospective plan, explicitly distinguishing historical authorization paragraphs
from the new approval. New tests demonstrate original requirements; this does not
amend the scientific protocol or replace original requirements with easier ones.

## Proposed authorization envelope, must be granted before execution

- One initial Worker and at most **one consolidated revision, 30 minutes each**.
  This is a proposed 60 minute maximum, versus the exhausted prior 40 minute envelope;
  the increase requires explicit prospective approval. Use one uniform per-invocation
  timeout compatible with the task's pinned execution policy, not an assumed different
  revision timeout. No automatic deadline extension or replacement task. Prefer a
  truthful partial report to running past the bound.
- Exactly 3 Worker-writable implementation paths, Git-root-relative:
  `Amazon4U/scripts/checkpoint_payload.py`,
  `Amazon4U/tests/test_checkpoint_payload.py`,
  `Amazon4U/docs/checkpoint-payload.md`.
- Proposed fresh root (currently unused):
  `/home/sunf/FSB/Final/Amazon4U/.worktrees/checkpoint-payload-matrix-correction`;
  project/cwd its `Amazon4U/` subdirectory. Separately approve setup of a clean local
  clone at dependency HEAD `3a9efa41e02c82e422d8eb15fbc0a0c4622a4c48`, without commits,
  staging, index tricks or modifying old checkouts. If occupied, stop rather than
  overwrite. Pin actual HEAD/index and accepted source/test/protocol hashes.
- Freeze the latest 3-file candidate, matrix, full original specification, original
  reports/reviews and both invocation evidence references under new project's
  `data/reports/checkpoint-payload-matrix-inputs/` as protected ignored read-only inputs.
  Keep historical files at their original locations unchanged. Use project-scoped
  recursive sentinels and explicit artifact hashing budgets 128MiB/file, 512MiB/total,
  60s/file, 300s/snapshot. These are monitoring limits, not a sandbox or fixture budget.
- Fresh test output/temp subtree only:
  `Amazon4U/.test-output/checkpoint-payload-matrix-v1/`; empty before/after every run.
  Wrapper/runtime evidence must be separately predeclared in the future task, not
  silently placed in implementation/test paths. Training in new tests is forbidden.
- Keep fixtures <=8 users, 12 products, 8 metadata nodes/type, 64 dimensions; actual/expected
  logical limits 2097152 bytes/depth 32/visits 16384. Only original one-at-a-time negative
  exception at most2097152+4096 accounted input or projected output, depth 33 or visits 16385.
  Smaller explicit runtime-policy test budgets permitted; no production capacity claim.
- No new training/backward/optimizer steps, model/optimizer restoration, global RNG
  setters/default-dtype changes, real datasets, GPU/CUDA initialization, workers,
  profiling, Kaggle/remote jobs, source acquisition, serialization/decoding, D3,
  held-out evaluation, main integration or deployment. Existing 249 baseline tests,
  including their already accepted synthetic differentiability checks, remain unchanged.

### Proposed sole candidate-execution command

Future cwd is the new root's `Amazon4U/`. Executable:
`/home/sunf/FSB/Final/Amazon4U/.venv/bin/python` with argv `-B`, `-c`, and this exact
script, pinned at future submission for baseline/Worker/revision/Pi:

```python
import os,tempfile,unittest,sys; from pathlib import Path; p=Path('/home/sunf/FSB/Final/Amazon4U/.worktrees/checkpoint-payload-matrix-correction/Amazon4U/.test-output/checkpoint-payload-matrix-v1'); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); os.environ['TMPDIR']=str(p); os.environ['PYTHONDONTWRITEBYTECODE']='1'; suite=unittest.defaultTestLoader.discover('tests'); result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not (result.wasSuccessful() and not result.skipped))
```

120 seconds maximum per synchronous full discovery; at most 6 attempts per Worker
invocation, all attempts retained. No subsets, smoke imports, compileall, candidate
probes, detached tests or substitute commands. Pure file/text/AST/hash inspection is
not permission to import/evaluate candidate code. The new scratch path and fail-on-skip
exit detail are prospective; the exhausted task's original command is not changed.
Pi must inspect and independently verify with the new task's approved original command.
No predeclared deployment policy: deployment remains NOT_AUTHORIZED.

## Baseline, preservation boundaries and allowed edits

1. Setup independently verifies the clean accepted 249-test dependency baseline after
   separate setup/execution approval. Preserve receipts and check source/protocol hashes.
2. Worker begins by ordinary byte-for-byte copying the frozen latest 3 candidate files
   into its 3 allowed paths, not regenerating them. Record hashes matching preparation
   baseline. Its first full discovery must reproduce the 283 starting cases before new
   edits; a bootstrap failure is reported before attempting speculative corrections.
   Original 34 payload test bodies and all 249 baseline tests must remain unchanged.
3. Add a new test class/helpers for the matrix. Missing coverage does **not** authorize
   rewriting verified guards. Existing E0/E2 fixture controls are preserved; add a new
   coherent boundary helper for E1/E4/E100 etc. Never relax validator semantics to make
   an invalid fixture pass.
4. **Protected function bodies**, exact text/AST hashes in `baseline.json`:
   `_validate_selection_and_best_model`, `_make_owned_output_copy`,
   `_compute_condition_positive_and_empty_pool_pairs`, `_ceil_div`,
   `_get_hetero_expected_param_names`, `_validate_tensor_properties`,
   `_validate_sparse_adj_buffer`, `_tensor_descriptor`, `_safe_type_name`.
   **Protected Adam aggregate block:** the last top-level `if global_step==0 ... else`
   in `_validate_adam_optimizer`, including its `sum(updates)>=G` refusal; exact range/hash
   is recorded separately. Add protective tests for these implementations.
5. Initial implementation edits are limited to:
   - **A01:** optional actual `param_names` shape/type/order/value binding.
   - **A05:** type-check actual foreach/fused options before tensor-valued equality.
   - **G01:** correct canonical CUDA-role CPU-inactivity guard, not D1 constants.
   - **G09:** validate parent/local seed type/range before bound-role equality.
   - **E03:** enforce original explicitly declared list fields in expected runtime.
   - **D01:** centralized bounded error rendering/type-safe key summaries and affected
     D2 error sites. Guard predicates/numerical behavior outside the listed targeted fixes above
     must remain unchanged; no broad catches that hide unexpected native errors.
   - Active appended API/result documentation and test-module descriptions/counts,
     grounded in actual results. Keep the original 59605-byte doc prefix exact.
6. Hashes are review tripwires, not proof of semantics. Pi compares complete changed
   functions and callers, not only whole-file hashes/diff hunks. If a new valid-control
   test demonstrates a real defect in a protected part, Worker reports the exact case
   rather than silently changing it. Pi may explicitly scope a necessary, justified
   change within the remaining declared revision authority after impact review; an
   exhausted budget or outside-scope dependency defect still stops work. Do not freeze
   a demonstrated bug forever, or treat a new test as automatic permission to rewrite.

## Concrete technical pins for the known fixes

### A01, actual named-option control and rejection

Use a valid E2 payload. Set trusted group `options.param_names` and actual group
`param_names` to separate copies of the same ordered names. Validate the complete
control before mutation. Then change only actual names to `dict.fromkeys(names,None)`,
`7`, `None`, a string, bytes or bool; each must fail at the named-option guard with a
bounded domain exception. A mapping whose keys happen to match is not a name sequence.

Pin this proposed compatibility detail explicitly for future approval: preserve the
existing supported trusted list/tuple sequence alternatives, require actual optional
names to have the same concrete supported sequence type and ordered values as that
trusted optional value, reject cross-container mismatches and malformed elements.
Test list/list and tuple/tuple controls, actual-only/expected-only/both-invalid cases.
This pin clarifies the correction assignment; do not retroactively claim that tuple
behavior has already been executed or that every other field permits tuples.

Do not change packed IDs, partitions, frozen options, lazy states, update/step precision,
aggregate bounds or selection. Tests for A02-A04 and T01-T03 protect those behaviors.

### G01, actual canonical active-cuda rejection, not just a spelling test

Noncanonical actual role name `PyTorch_CUDA` must fail the ordered-role guard.
That alone does not exercise the CPU inactivity requirement.

For the real negative, start from a passing CPU control and rebuild identity via the
accepted public D1 builder with only the canonical `PyTorch_CUDA_if_applicable` role
active, seed 42, clean algorithm/derivation, reason None. Supply the rebuilt identity as
both actual and expected, update sampler.identity_hash, matching actual role metadata
and actual/trusted role IDs `[gen_torch]`. Existing parent generator has seed 42; all
other bindings/states remain valid. D1 does not impose this CPU/CUDA cross-check.
The D2 CPU guard must reject the canonical active role. Do not initialize CUDA or edit
D1 to produce this fixture. Preserve explicit inactive-role rejection and W0/W1/W2.

### D01, demonstrate the actual failing diagnostic paths

Let `H=10**5000`, never decimal-format H in the test itself. Keep all other fields
valid. Mutate each separately, with intended-field assertions:

- Entire LightGCN state keyset becomes homogeneous integers H, H+1, H+2 with the same
  original tensor values. Adding H to existing string keys would be rejected earlier
  by mixed-key preflight and would NOT demonstrate source729's defect.
- Trusted `rng_generator_bindings[gid].codec=H` reaches source 2089.
- Actual role.name=H, sampler.algorithm=H, runtime_profile.profile_id=H,
  optimizer group params[0]=H and betas=(H, 0.999) exercise distinct raw paths.
- Trusted role-binding keyset uses homogeneous H+i keys; trusted runtime-profile
  keyset uses homogeneous tuple keys `(0,)` and `('x',)` to expose unsafe sorting.
- Long clean strings and many keys check aggregate message size, not just truncation
  of one element. Keep within the original logical fixture limits.

Proposed diagnostic assertion budget is4096characters per D2-origin error, with a
field/role identifier. Original C8 says bounded, without that numeric constant; this
is an explicit prospective technical test pin, **not** an integer validity cap or a
rewrite of original criteria. Keep valid arbitrary-size integers where allowed and
existing D1 exception families. Never modify interpreter digit limits.

### A05/G09, supported tensor leaves must not trigger native truth conversion

Generic preflight allows exact CPU tensors because valid model/optimizer/RNG fields
need them. It does not make a tensor a valid optimizer bool option or seed integer.

For A05 first validate a control with actual AND expected foreach=False, then replace
only actual foreach by `torch.tensor([0,1])`. Repeat independently for fused=False.
Current equality-before-type checks 938-946 attempt truth conversion of a multielement
Tensor. Reject wrong types BEFORE equality, retaining all supported well-typed options.

For G09 first validate each parent codec's seed 42 control, then replace only actual
seed_material by `torch.tensor([42,43])`. Current parent bound-role comparison 1233-1240
precedes codec type validation 1247+. Move or add codec-specific validation before that
comparison. Retain worker-derived seed distinctions and intrinsic limits. These are
static source findings; no new reproductions were executed in preparation.

### Deterministic test expansion, not assumed coverage

For every schema field listed in the original contract, label missing/extra and wrong
supported-type cases separately. Wrong-type representatives are None, bool, int,
float, string, bytes, dict, list, tuple, scalar CPU tensor and two-element CPU tensor;
exclude representatives explicitly valid for that field/context, and test value/range
errors separately. Preserve original allowed alternatives; A01's precise compatibility
pin and E03's explicit list fields are stated above. Do not use only the factory's
None foreach/fused defaults and miss False-profile comparisons.

Expand model cases by exact pinned parameter names and current/best role, codec cases
by each stated component, and expected-descriptor cases independently of actual input.
Use nonempty metadata controls for finiteness tests, not out-of-range indexing into
empty embeddings. Do not fabricate unconstructible native states (e.g. requires_grad
on uint8 RNG bytes) or claim a constructor-normalized input exercised a D2 rejection.
Explain intrinsic representation constraints with source evidence; Pi must verify that
an assigned negative category is still covered. Do not introduce unittest skips.

Use case IDs `<row>/<role>/<field-or-parameter>/<mutation>`. Never place raw huge
integers or hostile objects in subTest parameters/assertion formatting; use safe labels
so the test harness cannot hide the original result with its own formatting error.
Check full control success and intended failure stage. Emit successful case IDs only inside the successful
assertion block; reconcile declared/generated/executed cases in the final ledger.
Pointer differences alone do not prove ownership, and immutable primitive/empty-tuple
identity sharing is not a mutable-alias failure.

## Dependency-ordered work and stopping checkpoints

These are phases of ONE whole deliverable, not new tasks for each small fix.

1. **Bootstrap:** copy pinned candidate, reproduce 283 before edits, verify positive
   controls. If it fails, preserve results and report the precise discrepancy.
2. **Test-first closure:** add named-case matrices and independent oracles covering all 37
   rows. Execute only the full command. Demonstrate specified current A01/A05/G01/G09/E03/D01
   failures; distinguish expected red findings from fixture/import/unrelated errors.
   Do not manufacture an exact failure count before these proposed cases are run.
3. **Minimal fixes:** apply only known localized changes above. Keep protected blocks
   exact; use matrix protection tests before declaring closure. New in-scope findings
   require a concrete valid-control reproduction and explicit impact accounting.
4. **Full evidence:** complete every required matrix family, run original full discovery,
   inspect all results and retained case IDs. Emit a compact per-family executed-case
   ledger only AFTER assertions succeed inside subTest blocks; append outside a failing
   subTest would falsely count a swallowed assertion as success. Pi verifies loops,
   guard reachability, case completeness and original test hashes independently.
5. **Truthful handoff:** reserve final 5 minutes of each invocation
   for the <=120s full run, evidence reconciliation and a complete report. Stop editing
   earlier if necessary. If incomplete, return partial with exact remaining row IDs;
   do not consume the deadline writing success prose or silently omit failed runs.
6. **Pi decision:** inspect diff then actual complete functions/callers/tests, check
   protected hashes/condition changes, run independent approved full benchmark, read
   full evidence and evaluate unchanged C1-C10. A single consolidated revision, if
   needed and unexposed, addresses explicit remaining rows. Any unresolved issue or
   incomplete mandatory evidence still blocks ACCEPT at the final bound.

## Report required to prevent another unsupported completion claim

Use the required native report fields, with row/case IDs in summary/test_results and
unresolved_issues; do not invent a different report schema. Report every execution
attempt (bootstrap/red/failed/repeated/timed-out/cancelled/successful), exact command,
source snapshot, exit/status and raw-evidence references. Also acknowledge the preserved
prior attempt 152 failure/191 interruption/initial timeout, Pi283/26 errors and revision
272(250/1 error), 282(283/2 errors), 300(283OK), 310(283OK). Original reports remain unchanged.

For every matrix row identify preserve/fix/test-only, original and resulting source
hashes, added tests, observed intended result and remaining gap. `unresolved_issues=[]`
is justified only when assigned work is actually closed; no broad "all blockers fixed"
claim from aggregate test count. Current limitations are independently adjudicated by
Pi verbatim/in order against original scope; in-scope defects cannot be reclassified
outside scope. Supplement partial inventory using bounded raw-record review without
retuning shared tooling. No record completeness/syscall-sandbox claim.

Even eventual task ACCEPT is **task_only**. Main transfer, D3 and scientific execution
require their later independent gates; deployment remains **NOT_AUTHORIZED**.
