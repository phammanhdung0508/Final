# Antigravity Subagent System Architecture

This directory defines the specialized agent roles for Antigravity when executing
autonomous research in `multiautoresearch`.

## Agent Roles

| Role | Spec File | Purpose | Permitted Actions |
|---|---|---|---|
| **`autolab`** | `agents/autolab.md` | Master coordinator of the research loop | Delegates tasks, inspects queues, oversees loop |
| **`planner`** | `agents/planner.md` | Hypothesis generation & prioritization | Reads notes, proposes 1-variable hypotheses |
| **`experiment-worker`** | `agents/experiment-worker.md` | Execution & Kaggle runner management | Modifies `train.py`, launches & polls Kaggle jobs |
| **`reviewer`** | `agents/reviewer.md` | Benchmark integrity auditor | Read-only verification: ensures `evaluate.py` is intact |
| **`reporter`** | `agents/reporter.md` | Metric logging & documentation | Appends to `results.tsv`, maintains `notes.md` |

## Delegation Protocol

When Antigravity uses `invoke_subagent`:
1. **Plan**: `planner` inspects previous results in `research/results.tsv` and `research/notes.md`, then formulates the next testable hypothesis.
2. **Review**: `reviewer` verifies the proposed edit adheres to benchmark constraints.
3. **Execute**: `experiment-worker` edits `train.py`, launches on Kaggle via `kaggle_job.py`, and waits for completion.
4. **Report**: `reporter` logs the resulting `eval_score` into `research/results.tsv`.
