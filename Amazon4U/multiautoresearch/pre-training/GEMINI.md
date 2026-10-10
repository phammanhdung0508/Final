# Antigravity Instructions: Pre-Training Stage

You are operating within the `pre-training/` lifecycle stage of `multiautoresearch`.

## Mission

Pre-train graph representations on the MovieLens Knowledge Graph using self-supervised
link prediction, minimizing `val_bpb` (validation link prediction loss).

## Benchmark Contract

- **Fixed Benchmark Files (DO NOT MODIFY)**:
  - `prepare.py`
  - `evaluate.py`
  - `model.py`
- **Single Experiment File (MODIFICATIONS ALLOWED)**:
  - `train.py`
- **Promotion Source of Truth**:
  - `train_orig.py`
  - `research/live/master.json`
  - `research/results.tsv`

## Local Verification & Managed Runs

- Run local experiment:
  ```bash
  uv run train.py
  uv run evaluate.py --model-path final_model
  ```
- Record locally & promote if beating current master:
  ```bash
  uv run scripts/submit_patch.py --comment "<hypothesis description>"
  ```

Always log results into `research/results.tsv` and maintain findings in `research/notes.md`.
Do not commit or push to git without explicit user instruction.
