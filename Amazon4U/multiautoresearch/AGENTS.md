# Agent Instructions

## Current Amazon4U Worker policy — prospective, no launch permission

Read `../docs/worker-delegation-policy.md` and
`../configs/worker-delegation-policy.json`. **Native agy via the delegation extension
is the default implementation channel.** Pi plans/scopes/orchestrates, independently
verifies and decides acceptance; native agy implements/corrects/tests/documents only
within authorized scope. Distinguish `native_agy_worker_invocation`,
`sdk_text_only_generation_request` and `pi_independent_verification`.

Recovery maxima are **2 + 2 + 2** native invocations over three separately authorized
rounds, six total per deliverable, not blanket permission. Preserve cumulative history;
never reset a completed round. Review root cause/gaps/material changes/technical merit/
needed authority after each substantive ESCALATE. After #3 present an evidenced stop,
exceptional-native-continuation or SDK recommendation, not automatic fallback. Stop
sooner on acceptance, no credible path, authorization boundary or critical safety.

SDK is optional only with explicit user approval before EVERY request, specifying
native unsuitability, task/unresolved criteria, model/channel, source/output paths,
request/time/resource/verification limits and independent acceptance procedure.
No silent switch, budget transfer or inherited SDK authorization. The historical four
H-REF SDK requests remain SDK calls and consumed; no new round is authorized here.
Legacy autonomous/subagent/LM workflows in nested AGENTS/GEMINI/prompts are reference
only for Amazon4U and cannot override this policy, science, isolation or execution gates.
This governance correction authorizes no Worker/SDK/test/remote execution or code changes.

## Amazon4U operating agreement

For Amazon4U work, inherit `../AGENTS.md` and read
`../docs/delegation-and-execution.md`. Pi is planner/orchestrator/independent
reviewer with acceptance authority; agy implements/fixes and may execute only
explicitly authorized Kaggle workloads. Follow the three delegation principles
and separate task acceptance from execution/deployment authorization.

The framework's pre-training/post-training/inference layout is intended for
adaptation, not already verified as an Amazon4U runner. Existing LM/Movie4U
instructions (including `GEMINI.md` and nested prompts) do not override Amazon4U
protocol, scoped task commands or approvals. Do not launch inherited autonomous
loops, subagents or training merely because a framework document describes them.

## Historical preparation direction and still-unassigned O-OFFLINE scope

User-agreed: **Reuse orchestration, not inherited training methodology.** The earlier
preparation permission was phase-scoped;see `../docs/training-harness-preparation.md` and
`../docs/training-harness-requirements.md`. Proposed adaptation uses the existing
post-training/scripts/kaggle_job.py transport with an explicit Amazon4U profile,
not its LM/Movie objectives/evaluator/model/promoted-master logic. Its physical
folder name must not label recommender training from scratch as fine-tuning.
No production/legacy replacement,Worker,CLI dry-run,credentials read,Kaggle call,
training/profiling or test execution is authorized by this document. Later H-REF
approvals remain historical/task-scoped; O-OFFLINE is still unassigned. Keep accepted Amazon4U
APIs authoritative;CPU/GPU and source-provenance/runtime gates remain explicit.
The external `/home/sunf/my-project/kaggle-cli/` is read-only reference only.

## Existing framework layout

This repository is a three-project workspace.

- `pre-training/` contains the existing Autolab pre-training experiment project.
- `post-training/` contains the single-model nano chat post-training benchmark.
- `inference/` contains the llama.cpp local inference optimization setup.

When working on pre-training, read and follow `pre-training/AGENTS.md` and run project commands from `pre-training/`.

When working on post-training, read and follow `post-training/AGENTS.md` and run project commands from `post-training/`.

When working on inference, read and follow `inference/AGENTS.md` and run project commands from `inference/`.

Keep the three project directories separate. Do not move benchmark files back to the repository root.
