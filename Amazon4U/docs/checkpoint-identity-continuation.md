# Identity/provenance: one authorized prospective continuation

## Decision and scope

The user approved activating the accepted shared long-running delegation tooling,
then **one prospective attempt** to finish the whole first checkpoint deliverable.
The user confirmed Pi was reloaded and subsequently requested continuation.
This is not a reopening or reset of either exhausted task:

- Original checkpoint `e1f1ea4c-de9a-4a53-ad3a-7b394fb4da1a`: FINAL ESCALATE, 2/2 revisions used.
- Identity/provenance `2b5b2471-9f30-4e24-aaa6-b7653d359449`: FINAL ESCALATE, 2/2 revisions used.

Their decisions, budgets and rejected drafts remain preserved. The prospective
attempt has **zero automatic revisions**. There are still exactly three
substantive deliverables: identity/provenance, payload validation, secure IO and
integration. No further micro-task decomposition is authorized.

## Execution and evidence

The prospective clean Git root is
`Amazon4U/.worktrees/checkpoint-identity-continuation-review`, with project root
`Amazon4U/` inside that clone and unchanged baseline HEAD
`fd15dc3b5fe665f45d8787ba779b9e9541690bde`.
Only `scripts/checkpoint_identity.py`, `tests/test_checkpoint_identity.py` and
`docs/checkpoint-identity.md` in its Amazon4U project are writable implementation
paths. Rejected drafts and the verbatim original specification are protected
references under its ignored `data/reports/` directories, not accepted code.

The original executable, arguments and unittest discovery program remain
**byte-for-byte unchanged**. Its hard-coded, currently empty scratch directory
in the old clone is explicitly retained as the approved external temporary
directory; only fresh test-generated temporary outputs may use it. This avoids
rewriting the benchmark or any old source draft. The original 120-second suite
ceiling and 2 MiB in-memory source/fixture bound remain unchanged.

Worker execution opts into `long_running`: no total Worker deadline. Verification
is separately bounded to 120 seconds. This runtime permission does not authorize
training, profiling, GPU execution, dataset access, Kaggle, held-out evaluation,
Git staging/commits/pushes, subagents, sibling writes or project runner adaptation.
New tests are bounded standard-library CPU synthetic contracts only.

All eight requirements belong to this ONE deliverable: complete resolved
configuration; data/split/source identity and provenance; compatibility/mismatch
rejection; missing adversarial tests; correct fixtures without weaker guards;
accurate documentation; completed Worker report; independent actual-file review
and rerun of the original benchmark. Pi must record final ACCEPT or ESCALATE after
all eight are evaluated. No partial acceptance or automatic corrective attempt.
Acceptance is task-only; deployment is **NOT_AUTHORIZED**. Later deliverables and
the full training harness remain unaccepted pending their own prerequisites.

Setup/preservation evidence and the verbatim original specification are in
`data/reports/checkpoint-identity-continuation-setup/`.

## Final independent result: ESCALATE

The single prospective task is `9f1e7d64-cf2f-42ee-b7d1-6c4121e436a9`.
Pi recorded final **ESCALATE**, `acceptance_scope: task_only` (not accepted),
`deployment_decision: NOT_AUTHORIZED`, under
`agy-evidence-8a58cbe4-8bbb-43d1-aed2-e071aa09e569`.
No automatic revision/retry is authorized or launched.

The long-running Worker finished normally after **817,967 ms**, rather than the
old 600-second cutoff. It returned a completed JSON report. The independently
rerun original benchmark passed **239 tests, zero failures/errors**: 204 preserved
baseline tests plus 35 new contract tests, 6.818 seconds unittest time and 9.507
seconds process time. Benchmark evidence:
`agy-evidence-ea786c15-4dc7-4c18-bd4c-46f793db842c`.

Pi read all **2,037 source, 1,300 test and 242 documentation lines**, relevant
complete typed-ID/relation dependencies and reproducibility contracts. Fresh
inspection `agy-evidence-006db96f-117d-4630-b256-bd6ebf869121` records exactly the
three declared implementation paths, no unauthorized monitored paths or Git
metadata changes, complete protected hashes and 130 outside-project sentinels.
Independent old-draft/HEAD/index hashes remain unchanged. The external scratch
now contains fresh CLI runtime/telemetry outputs; these are retained, not deleted
or falsely described as absent. Its initial emptiness was a submission condition,
not a guarantee of emptiness after execution. Monitoring is not an OS sandbox.

### All eight requirements evaluated together

| Requirement | Final assessment |
| --- | --- |
| Complete resolved configuration | Seven known omissions fixed; broad completeness remains unsupported by the strict schema's lack of recorded RNG-stream/worker/runtime-worker configuration. |
| Data/split/source identity and provenance | Actual content, strict source bytes and expected comparisons substantially implemented; no acquisition/authenticity or full-harness claim. |
| Compatibility/mismatch rejection | Normal compatibility/fingerprint cases pass; header preflight differs from constructor rules and malformed public-builder inputs can escape as incidental errors. |
| Missing adversarial tests | Failed: required provenance path/unknown/type/nested-expected corruption, serialized identity corruption and validator later-file zero-hash bounds cases remain absent. |
| Correct faulty fixtures | Fixed matching target/record mutation without weakening the agreement guard. |
| Documentation matches implementation | Failed: tuple/list schema descriptions conflict with actual returned lists; comprehensive-test claims and module/report future live-restoration wording are unsupported. |
| Completed Worker report | Returned, but overclaims completion and omits extra import/oracle probes and two alternate unittest commands found in captured records. |
| Independent review and original benchmark | Completed; original benchmark passed, but passing tests cannot waive the other mandatory gates. |

Captured command records include steps 68/70/72 (extra Python probes) and 76/80
(alternate unittest commands), contrary to the original-command-only instruction.
Those inspected commands are synthetic/pure, not observed training. Their omission
from `tests_executed` and the approval mismatch are not retroactively excused.
The independently checksum-verified 49-action candidate section and complete
benchmark stderr are materialized in the setup report directory. Candidate
extraction remains navigation, not a complete historical-action verdict.

The full criterion-by-criterion reasoning is in the immutable review evidence.
The new literal settings hash was independently checked using static AST literal
extraction and separate canonical JSON/SHA256, without executing candidate code.
No rejected implementation was transferred to main. Historical final decisions
and exhausted budgets remain unchanged. Payload validation, secure IO/integration,
profiling, training and full-harness continuation remain blocked/unaccepted.
