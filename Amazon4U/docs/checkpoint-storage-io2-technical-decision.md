# D3 IO2 technical decision and minimal SDK review

Status: **specification-level recommendation / proposed correction only**. The user authorized this focused evaluation before any additional execution privileges, not a Worker call, product change, test, scope amendment or budget reset. Current D3 remains ESCALATE, task_only/not accepted; deployment NOT_AUTHORIZED. Exact authority and static evidence: `data/reports/checkpoint-storage-implementation/io2-technical-decision-v1/`.

## Decision: flat physical acquisition, unchanged logical D1 paths

**Reject** the naive proposal to accept only basename source identifiers under one project root. Accepted D1 requires the explicit registered `scripts/...` and `configs/...` identifiers. Dropping those prefixes, relocating files into a staging directory and implicitly relabeling them, or removing one prefix would not establish the original IO6 contract.

**Recommend** independently approved **directory-inode roots per existing D1 prefix**, with direct-child acquisition only:

- Keep every D1 logical identifier unchanged, e.g. `scripts/checkpoint_identity.py` and `configs/reproducibility.json`.
- A mandatory, independently trusted caller mapping binds `scripts` and/or `configs` to their respective approved directory descriptors. For each registered `prefix/basename`, D3 opens only `basename` relative to that already-approved descriptor. The prefix lookup is a trusted mapping lookup, not a filesystem traversal.
- Reject extra logical components, absolute/traversal paths, unregistered identifiers, unknown/missing prefix bindings, symlinks and nonregular leaves before data reads. Preserve exact types, ownership/permission policy, file/aggregate bounds and descriptor ownership.
- Read actual physical bytes from those directories and assemble the original full logical-path→bytes mapping for **one existing D1 provenance build/validation**, with mandatory independent expected provenance and explicit caller Git/environment evidence. No D1 schema or allowlist change; no real-source capture during tests.

This is compatible at the **API/domain specification level** with the original IO2/IO6: IO2 allows narrow independently approved roots; IO6 requires allowlisted physical acquisition, not a particular single-project-root API. The original plan explicitly leaves exact APIs and root bootstrap to scoped technical choice. All20 currently registered D1 paths have exactly one prefix and one basename; no registered deeper logical source path is being removed. The criteria do not mandate arbitrary-depth traversal. Logical two-component source IDs remain mandatory even though each physical lookup is flat.

### Exact guarantee and threat-model boundary

The trust anchor must be the independently approved **directory inode**, not its mutable pathname. A rename of an ancestor cannot retarget a held directory descriptor. There is no separately acquired mutable intermediate parent below that root; nested physical traversal is rejected before acquisition. Thus the previously identified detached-parent lookup is eliminated, not excused by snapshots.

D3 must still classify a direct leaf before a data-capable open, bind the readable descriptor to that checked inode, and refuse observed leaf/directory substitutions before returning data. Root approvals and logical-prefix bindings must not come from the checkpoint. D3 must **not** implement this as “open projectRoot/scripts with an ordinary racy lookup, then call it independently approved”: that recreates the unresolved bootstrap problem. The caller supplies independently approved roots, as required by the original trust boundary; it does not supply preopened source-file descriptors instead of D3 acquisition.

This design does not claim to detect every transient namespace event, lock filesystem ancestry forever, make file contents an atomic multi-file snapshot, or resist a privileged attacker changing the kernel/mount namespace or process descriptor table. Flatness alone does not address untrusted mount manipulation. Existing Linux/local-filesystem/trusted-mount assumptions and the exact approved-inode root model must be explicit and independently reviewed, not silently added as a limitation waiver. If the intended contract instead requires proving membership under **one project-root pathname/descriptor throughout acquisition**, a prefix-root mapping is insufficient; that interpretation must be settled before implementation, not hidden in the patch.

**Recommended technical choice is the prefix-root/direct-child model above**, subject to explicit agreement on that root model before another Worker request. This records feasibility of a bounded design, not proof that current code implements IO2 or authorization to change scope.

## Minimal request/context and SDK findings — static inspection only

1. The failed request retained the full358,676-byte context as one user input: independently compared byte-for-byte with the observed payload. It requested `gpt-6.1-sol`, no tools, `tool_choice:none`, streaming and `store:false`. No evidence of client-side context truncation was found.
2. The current SDK's character heuristic estimates about102,526 tokens including instructions. Its current bundled catalog lists500,000 context tokens and128,000 maximum output tokens. These are **not exact token counts or proof of the failed request's effective server limits**; runtime overrides/model metadata were not saved then. Do not diagnose the failure as context overflow or declare capacity verified from byte counts.
3. Although the client passed `maxTokens:32768`, the actual payload has **neither max_output_tokens nor max_tokens**. The inspected Codex adapter constructs no such field. Treat32768 as a requested option, not a demonstrated server-enforced ceiling; do not blindly inject a potentially unsupported parameter. Existing one-request/30minute/client-byte bounds are separate controls, not token/cost guarantees.
4. The SDK exposes terminal error messages through the error event and final `AssistantMessage` (`errorMessage`, `rawStopReason`, optional diagnostics). Its `.result()` can resolve to an error message rather than throw. Our Pi-owned wrapper ignores the error event, omits those fields, then throws its own generic incomplete-response error. That loses useful failure classification. HTTP200 establishes headers, not successful streamed completion. The historical precise cause cannot now be recovered from the retained evidence.

### Proposed minimal client correction, before any new call

Keep the same direct ModelRuntime/no-tools/SSE/one-fetch/no-retry channel. No AgentSession, controller, authentication change, routing experiment, or provider-library modification.

- Save safe effective model limit metadata and preflight request/context sizes before fetching. Proposed context cap256KiB and serialized-payload cap320KiB are lower operational limits, not production checkpoint caps. Remove duplicated historical context only by sending one exact authoritative copy of required criteria/authority/commands and complete selected implementation/dependencies. If critical context does not fit, stop; never silently truncate or semantically compress requirements/evidence.
- Capture allowlisted terminal event/status/error-code metadata and stage markers before the generic error branch; retain known raw-stop categories and usage-presence flags. Classify known SDK EOF/timeout/provider terminal errors; unknown remains unknown. Fingerprint/length an unrecognized error message if necessary rather than storing free-form secrets.
- Never log authorization headers, credentials, arbitrary raw events, or reasoning/signature blocks. Do not persist an entire assistant/error object as a shortcut. Recheck the existing1MiB text bound on both streamed and final text, and label partial text unaccepted. Preserve HTTP status separately from completion status.
- No health-check generation, retry or credential/debugging investigation is authorized by this review. Review the exact small wrapper diff locally before any future approved request. No claim that these changes would have prevented the historical provider failure.

## Proposed bounded correction — NOT AUTHORIZED

Only after agreement on the root model and a separately explicit execution amendment:

- One additional Worker generation, at most30minutes, no retry/revision. The existing **three request attempts/two technical responses remain used**; this would be a fourth attempt, never a reset. Reuse the hash-verified existing amendment candidate; independently refresh all baseline hashes before sending.
- Patch the same three D3 product paths, not a rewrite. Preserve verified codec/preflight/type/bit/ownership logic and all355 inherited files. Carry forward the still-unexecuted targeted private-fixture, special-leaf classification, anonymous inode-bound no-clobber publication and explicit byte-profile corrections, plus the direct-child prefix-root API. Prior proposals are references, not accepted code.
- Pi reviews complete IO1–IO8 before execution, particularly **IO2 containment/refusal; IO3 allocation/format bounds; IO5 atomic publication/owned cleanup; IO6 actual bounded source acquisition; IO7 full D1/D2 pipeline**. No current limitation or assigned unresolved issue is waived by this design.
- At most the one previously specified full discovery,240s, all355 plus all D3 tests; prior conservative run count5 is retained. No subset, smoke/native probe, unsafe filesystem experiment, new subprocess/thread, controller, profiling/training or privilege change. Before that run, screen the entire patch/native dependencies/fixtures and bind a safety receipt to exact bytes.
- Source tests must cover both prefixes with genuine controls, rejected nested/escape/type inputs, root-ancestor substitution without redirection to a replacement directory, and leaf substitution refusal. Adapt only internal hooks made obsolete by the deliberate API change; preserve the assigned safety invariants rather than counting unrelated early refusals as success.
- Stop with the precise blocker on unresolved IO2, invalid binding, unsafe/incomplete review or any failed/incomplete mandatory evaluation. No iterative repair loop. ACCEPT requires every original IO1–IO8 and exact ordered limitation review; solving IO2 alone does not accept D3. IO3/IO5 remain unverified and IO6/IO7 remain failed/unpassed today.

No new Worker request, product patch or regression execution occurred in this technical review. No main source/test transfer, deployment, training/profiling, held-out work, stage/commit/push/release/rollback is authorized.
