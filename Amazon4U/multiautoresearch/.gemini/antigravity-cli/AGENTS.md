# Antigravity subagent system architecture

> **Amazon4U scope override:** `../../AGENTS.md` and
> `../../../docs/worker-delegation-policy.md` govern. Native agy through the delegation
> extension is the default; SDK generation is a distinct exception needing explicit
> user approval before EVERY request. Recovery maximum is separately authorized
> 2 + 2 + 2 with cumulative history and root-cause review, not automatic calls.
> The subagent roles and `invoke_subagent` research loop below are legacy architecture
> reference, not authorization for Amazon4U. Do not launch them, inherited training,
> Kaggle jobs or model requests from these instructions. Pi retains independent review
> and acceptance; this correction authorizes governance edits only.

This directory defines the agent roles for Antigravity during research in `multiautoresearch`.

## Agent roles

| Role | Spec file | Purpose | Permitted actions |
|---|---|---|---|
| **`autolab`** | `agents/autolab.md` | Master coordinator of the research loop | Delegates tasks, inspects queues, oversees loop |
| **`planner`** | `agents/planner.md` | Hypothesis generation & prioritization | Reads notes, proposes 1-variable hypotheses |
| **`experiment-worker`** | `agents/experiment-worker.md` | Execution & Kaggle runner management | Modifies `train.py`, launches & polls Kaggle jobs |
| **`reviewer`** | `agents/reviewer.md` | Benchmark integrity auditor | Read-only verification; checks that `evaluate.py` is intact |
| **`reporter`** | `agents/reporter.md` | Metric logging & documentation | Appends to `results.tsv`, maintains `notes.md` |

## Delegation protocol

When Antigravity uses `invoke_subagent`:
1. **Plan.** `planner` inspects previous results in `research/results.tsv` and `research/notes.md`, then formulates the next testable hypothesis.
2. **Review.** `reviewer` verifies the proposed edit adheres to benchmark constraints.
3. **Execute.** `experiment-worker` edits `train.py`, launches on Kaggle via `kaggle_job.py`, and waits for completion.
4. **Report.** `reporter` logs the resulting `eval_score` into `research/results.tsv`.
