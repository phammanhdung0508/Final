# Agent Instructions

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

## Existing framework layout

This repository is a three-project workspace.

- `pre-training/` contains the existing Autolab pre-training experiment project.
- `post-training/` contains the single-model nano chat post-training benchmark.
- `inference/` contains the llama.cpp local inference optimization setup.

When working on pre-training, read and follow `pre-training/AGENTS.md` and run project commands from `pre-training/`.

When working on post-training, read and follow `post-training/AGENTS.md` and run project commands from `post-training/`.

When working on inference, read and follow `inference/AGENTS.md` and run project commands from `inference/`.

Keep the three project directories separate. Do not move benchmark files back to the repository root.
