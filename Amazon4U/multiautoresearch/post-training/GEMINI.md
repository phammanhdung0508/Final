# Antigravity Instructions: Post-Training Stage

You are operating within the `post-training/` lifecycle stage of `multiautoresearch`.

## Mission

Maximize macro-averaged held-out `eval_score` (NDCG@10) on the validation split for the
MovieLens Knowledge Graph + HeteroGraphSAGE recommendation model.

## Benchmark Contract

- **Fixed Benchmark Files (DO NOT MODIFY)**:
  - `prepare.py`
  - `evaluate.py`
  - `model.py`
- **Single Experiment File (MODIFICATIONS ALLOWED)**:
  - `train.py`

## Managed Execution Engine

- Dry-run validation:
  ```bash
  uv run scripts/kaggle_job.py launch --dry-run
  ```
- Launch managed GPU training on Kaggle:
  ```bash
  uv run scripts/kaggle_job.py launch --mode experiment
  ```
- Follow execution logs:
  ```bash
  uv run scripts/kaggle_job.py logs <kernel_slug>
  ```
- Download artifacts & metrics:
  ```bash
  uv run scripts/kaggle_job.py output <kernel_slug>
  ```

Always log results into `research/results.tsv` and maintain findings in `research/notes.md`.
Do not commit or push to git without explicit user instruction.
