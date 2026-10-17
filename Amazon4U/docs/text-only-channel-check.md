# Text-only channel: bounded capability verification

## Decision and authority

User-approved direction: do not build a controller or modify the native agent merely
to force patch-only behavior. Check an existing genuinely text-only configuration;
use scoped context and baseline-bound patches; review paths/content/tests before
application or execution. The user explicitly said this is **not new D2 authorization**.

**Verified for this bounded demonstration:** the installed Pi SDK's existing
`ModelRuntime.completeSimple` direct-completion interface, using the already selected
`openai-codex/gpt-6.1-sol`. This is **not native agy** and does not call
`createAgentSession`, load workspace resources/extensions/MCP, create an agent loop,
register tools or dispatch model output. No controller, extension, provider/account
configuration, proxy or agent-system modification was made.

Local review: `data/reports/text-only-channel-check/review.json`, ACCEPT/task_only
**for capability verification and synthetic demonstration only**. Deployment remains
NOT_AUTHORIZED. D2 remains ESCALATE under its unchanged previous review.

## Actual observed request and result

One generation request, no retries,30second deadline; HTTP200, stop_reason=stop,
3001ms. Observed actual outgoing request has **no tool definitions** and
`tool_choice: "none"`; response contains text only. There is no tool dispatcher even
if a provider unexpectedly returns tool-call content (the check rejects it).
`parallel_tool_calls: true` is an adapter default present in the recorded body; it
supplies no tools or host execution authority with zero definitions/tool_choice none.
This is a claim about the client channel's granted capabilities, not an inspection
of all provider-internal infrastructure or an OS sandbox for generated code.

The prompt was632UTF8bytes: a synthetic constant-assignment excerpt, virtual path and
whole-virtual-file SHA256. Actual D2 source/specification bytes sent: **zero**.
Response:184bytes, a JSON replacement patch tied to that synthetic baseline. SDK
reported188input/75output tokens; these are not audited billing or a production D2
cost-saving measurement. The requested1024output-token option is not represented as
an enforced server cap by the inspected Codex adapter; the time bound/no-retry policy
and the actual small response are recorded without claiming otherwise.

Exact synthetic request/response and outcome:
`data/reports/text-only-channel-check/{synthetic-input.json,observed-request-body.json,response.txt,channel-result.json}`.
No headers/credential values were logged or supplied in model context. Authentication
used the existing SDK-managed provider configuration, without login/MFA/new scopes.

## Synthetic patch submission and checks

All file contents were synthetic, held in memory under the three approved D2 path
names. No patch was applied to those real files. The model received only the relevant
`VALUE = 1` excerpt, not the synthetic protected function or any source codebase.

Submitted format:

```json
{"path":"Amazon4U/scripts/checkpoint_payload.py","baseline_sha256":"<whole virtual-file SHA256>","old_text":"VALUE = 1\n","new_text":"VALUE = 2\n"}
```

Before any real publication or execution the demonstration checks exact schema/path
allowlist, current baseline hash, unique replacement context, protected-function AST,
retained test assertions and forbidden test syntax. It accepts the valid model patch
**in memory only** and rejects12negative cases:

- wrong project/path, traversal and absolute path;
- stale baseline and nonunique replacement;
- protected guard change;
- `torch.manual_seed`, another global setter, subprocess and filesystem calls;
- removed assertion and skipped test.

All negatives are rejected without writing a target file or executing generated code.
Actual three D2 candidate-file hashes and its existing ESCALATE review hash were checked
unchanged before/after. Evidence: `patch-demo-result.json`, `synthetic-accepted.diff`
and `virtual-baseline.json` in the same report directory.

This is **not a production patch engine or universal unsafe-code detector**. The toy
requires no calls/imports, so the synthetic checker forbids all of them. Real D2 uses
legitimate private generators; that toy policy must not be installed as its validator.
No actual filesystem application/symlink/parent-race guarantee was demonstrated.
Full Pi semantic review of implementation, tests, setup and affected dependencies is
still required; static screening alone does not exclude obfuscation or indirect effects.

## Future D2 boundaries — still require authorization

- Use a bounded text-only request plan, with relevant complete functions, necessary
  callers/dependencies, exact applicable frozen requirement clauses and pertinent test
  results. Do not resend the whole codebase per iteration or silently compress/change
  original requirements. Request missing context rather than guess. Pi retains the
  complete original C1–C10/specification as authoritative acceptance requirements.
- Each patch identifies the exact baseline; Pi independently checks whole-file hashes
  and relevant dependency hashes immediately before mechanical application. Drift means
  reject/rebuild context, not fuzzy application to a newer artifact.
- Accept only the three pre-approved D2 paths; reject alternate/absolute/traversal paths,
  undeclared new files, renames/deletions, scope changes and unintended protected-guard
  edits. Check canonical target/parent/symlink status and contents before real writes.
  This is reviewer-controlled application, not blanket permission to run patch commands.
- Inspect test changes for deleted/weakened assertions, skips, invalid controls, hidden
  helper scripts, imports/process/filesystem operations and global RNG setters. Review
  the **entire selected execution path, including inherited fixtures**, not merely
  changed lines. The current320-test artifact contains global seed calls3141/3284:
  do NOT execute it as a bootstrap just because it exists or a command was once approved.
- Generated patches remain untrusted text. Pi reviews and mechanically applies
  Worker-authored changes; substantive Pi-authored corrections must be identified, not
  presented as independently authored Worker code.
- Only after approved scope and semantic screening may Pi execute predeclared checks.
  Keep all failures/repeats and exact original criteria, revision/history/acceptance
  gates. No implicit deployment, main transfer, D3 or training permission.
- The current agy delegation tool is unchanged and is **not tool-free**. This direct
  SDK check does not inherit its task/inspection/benchmark receipts. Before a real D2
  round, explicitly predeclare the text-channel request/budget/history records and how
  mandatory independent inspection/verification/acceptance gates will be recorded;
  do not silently bypass them or retrofit acceptance into old exhausted tasks.

Verification stopped after the one generation and synthetic checks. No native Worker,
D2 implementation/candidate tests, actual source patch, controller work, training,
remote jobs, staging/commit/push or deployment followed.
