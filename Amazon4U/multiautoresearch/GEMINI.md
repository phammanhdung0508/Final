# Antigravity Project Instructions: Multi-AutoResearch

Welcome to `multiautoresearch`, an autonomous ML research and optimization framework
for Knowledge Graph and Heterogeneous Graph Neural Network recommendation systems (`Movie4U`).

## Lifecycle Stages

The project is structured into three distinct lifecycle directories:

1. **`post-training/`** (Primary Optimization Target):
   - **Task**: Post-train HeteroGraphSAGE for macro-averaged recommendation ranking.
   - **Primary Metric**: `eval_score` (held-out NDCG@10 on validation split).
   - **Fixed Benchmark**: `prepare.py`, `evaluate.py`, `model.py`.
   - **Mutable Surface**: `train.py` (loss objectives, negative sampling, learning rate schedules, regularizations).
   - **Execution Engine**: `scripts/kaggle_job.py` (managed Kaggle Kernels runner).

2. **`pre-training/`**:
   - **Task**: Self-supervised representation learning on Knowledge Graph topology.
   - **Primary Metric**: `val_bpb` (validation link prediction loss; lower is better).
   - **Fixed Benchmark**: `prepare.py`, `evaluate.py`, `model.py`.
   - **Mutable Surface**: `train.py`.

3. **`inference/`**:
   - **Task**: Recommendation serving throughput and latency optimization.
   - **Primary Metric**: `tok/s` (queries per second / recommendations per second) and `latency_p50_ms`.
   - **Benchmark Script**: `scripts/benchmark_serving.py`.

---

## Operating Rules for Antigravity

- **Benchmark Integrity**:
  - NEVER edit `prepare.py` or `evaluate.py`.
  - NEVER evaluate against or tune on the test split during the research loop.
  - The single mutable target for model improvements is `train.py`.
- **One Hypothesis Per Run**:
  - Each experiment should test exactly one clear hypothesis (e.g. BPR loss, temperature scaling, hard negative mining).
- **Execution via Kaggle**:
  - Managed GPU runs are executed via `uv run scripts/kaggle_job.py launch --mode experiment`.
  - Stream and inspect logs using `uv run scripts/kaggle_job.py logs <kernel_slug>`.
- **Ledger Recording**:
  - Always record completed runs, hypotheses, and metrics into `research/results.tsv` and `research/notes.md`.
- **Version Control Guardrail**:
  - DO NOT commit or push to git unless explicitly requested by the user.

---

## Subagents Architecture

Subagent specifications live under `.gemini/antigravity-cli/agents/`:

- `autolab.md`: Master coordinator managing the autonomous research loop.
- `planner.md`: Proposes hypotheses and prioritizes the experiment queue.
- `experiment-worker.md`: Implements changes in `train.py` and manages Kaggle job lifecycles.
- `reviewer.md`: Read-only auditor verifying benchmark rules and diff integrity.
- `reporter.md`: Records runs into `research/results.tsv` and updates research notes.
