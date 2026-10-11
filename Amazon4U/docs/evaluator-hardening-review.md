# Pi review — evaluator candidate-input isolation recovery

## Lead operating decision

The user approved Pi handling noncritical setup corrections and in-scope revisions
without repeated lead permission, following the existing independent review loop.
The goal remains implementation, regression verification and eventually bounded
profiling on Musical_Instruments. This does not waive mandatory evidence, critical
correctness/security/leakage checks, revision limits or separate profiling/training
permissions. Historical ESCALATE decisions remain preserved, not retroactively accepted.

## Recorded acceptance

Pi recorded **ACCEPT**, acceptance scope **task_only**, for fresh task
`0f2f0b7d-6e2a-4726-9e6e-83a73e8da423`, revision 2.
Review evidence: `agy-evidence-e0b47b09-2b95-4f2b-9ffb-eedad289eec8`.
**Deployment decision: NOT_AUTHORIZED** (no deployment policy was predeclared).

Acceptance covers only candidate-input isolation in the shared evaluator, its
synthetic regression fixtures and corresponding documentation in the isolated
repository `.worktrees/evaluator-hardening-clean/Amazon4U/`. The evaluator owns an
immutable candidate tuple; both scorer interfaces receive private list copies.
Missing authoritative scores fail; added cold/history keys cannot enter rankings,
candidate counts or coverage. Both feedback conditions and synthetic validation/test
filtering, multi-user/repeated calls and fixed hand-calculated oracles were reviewed.

Pi read actual complete source, tests, relevant callers/interfaces/policies and
changed documentation, independently verified all five original criteria, and ran
the unchanged approved full-suite command after final revision. Result: **33 tests,
zero failures/errors/skips**, unittest 2.442s, wrapper 3.113s, exit 0.
Benchmark evidence: `agy-evidence-5b9efccc-d6c3-4dd4-b9de-111869ab96e7`.
`git diff --check` passed. Source was unchanged after substantive code review;
the final test revision only removed EOF whitespace. No new dependencies.

Protected hashes, original 357-file snapshot and historical files remain stable.
Only authorized evaluator source/tests/documentation and declared synthetic result
output changed in the isolated task. Baseline content-capture commit remains
`a869003f832810ef6fa033be6bfae2dd9757285e`; it is not an implementation-acceptance
commit. No later commit, push or main-checkout code integration was performed.
Main-checkout change for this review is this decision record only.

## Scope boundaries and remaining work

This is not broad I1 acceptance, full-data production readiness, complete historical
CLI execution decoding, an OS sandbox or protection against out-of-band mutable
state access. Synthetic test-history fixtures are not held-out Amazon generalization.
Prior broad I1, separate evaluator and audit-tooling tasks remain ESCALATE/unaccepted;
preflight failures remain preserved in the isolated provenance reports. The scoped
candidate-mutation correctness issue now has an independently accepted recovery.

Next priority: independently verify the remaining I1 baseline scorers and reference
utilities under separate scopes, then models/encoders and training-harness regression
prerequisites. Audit-tool completion remains a separate lower-priority deliverable.
Musical_Instruments profiling has not run and remains unauthorized until its
prerequisites pass and an explicit bounded scope/budget/optimizer/output approval
is recorded. No training, profiling, deployment or real-data model evaluation occurred.
