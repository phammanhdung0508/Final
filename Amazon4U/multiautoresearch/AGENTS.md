# Agent instructions

## Current Amazon4U implementation policy

Inherit `../AGENTS.md`. Read `../docs/worker-delegation-policy.md`,
`../configs/worker-delegation-policy.json`, and
`../docs/delegation-and-execution.md`.

Direct single-agent ownership is the default. Pi may diagnose, implement, test,
and verify authorized Amazon4U changes directly. Mandatory Pi–agy delegation,
the planner/Worker split, and the prospective 2 + 2 + 2 recovery process are retired.
Delegation and second-model review are optional, subject to authorization and
original independent-review requirements. Self-review is not independent review.
Optional native delegation still requires all original extension gates.

Legacy agent prompts, control planes, autonomous workflows, and nested LM or
Movie4U instructions are reference only for Amazon4U. They do not require agent
fan-out or authorize execution. SDK generation still requires explicit user
approval before each request. Historical evidence, decisions, and consumed counts
remain unchanged. No stopped task is reopened by this policy.

## Amazon4U execution boundaries

Reuse orchestration, not inherited training methodology. The existing
`post-training/scripts/kaggle_job.py` uses LM or Movie4U objectives, evaluators,
and model conventions. It is not automatically an approved Amazon4U runner.
Keep accepted Amazon4U APIs and the frozen recommendation protocol authoritative.
CPU and GPU capability, provenance, runtime, and durability gates remain explicit.
The external `/home/sunf/my-project/kaggle-cli/` is read-only reference only.

This workflow approval grants no source correction on stopped candidates, test
run, CLI preflight, credential access, training, profiling, GPU or remote job,
deployment, cleanup, stage, commit, or push. Existing phase approvals and budgets
remain task-specific. Historical H-REF or O-OFFLINE snapshots are not launch authority.

## Existing framework layout

This directory contains three independent framework projects.

- `pre-training/` contains the existing Autolab pre-training experiment project.
- `post-training/` contains the single-model nanochat post-training benchmark.
- `inference/` contains the llama.cpp local inference optimization setup.

Before working in one of these directories, read its local `AGENTS.md` and run
its applicable authorized commands from that directory. Keep the three projects
separate. Do not move their benchmark files to this directory's root. Their
legacy workflows cannot override the Amazon4U implementation and execution policy.
