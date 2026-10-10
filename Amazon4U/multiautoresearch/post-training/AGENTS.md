# Agent Instructions

This is the post-training project. It is separate from `../pre-training`.

## Goal

Improve `eval_score` on the held-out post-training benchmark. Follow the
PostTrainBench-style contract: fixed task evaluator, fixed base architecture,
free-form post-training work, and a final `final_model/` artifact.

## Rules

- Edit `train.py` by default, or add supporting files in the current task
  directory for a post-training method.
- Prefer nanochat-style post-training patterns: task mixtures, masked assistant
  losses, chat special tokens, task-specific evaluation, and optional reward
  training.
- Do not modify `evaluate.py`, `src/eval/tasks/*/evaluate.py`, or generated eval
  records.
- Do not modify `model.py` to change the submitted architecture unless the task
  explicitly asks for benchmark or architecture changes.
- Do not train on the eval split.
- Store the best trained model in `final_model/`.
- Run the benchmark with `uv run evaluate.py --model-path final_model` before
  claiming an improvement.
- Keep this project independent from `../pre-training`; do not move benchmark
  files between the two project directories.

## Workflow

1. Prepare fixed data and base checkpoint:
   - `uv run prepare.py`
2. Edit `train.py`.
3. Train:
   - `uv run train.py`
4. Evaluate:
   - `uv run evaluate.py --model-path final_model`

Use `--limit` on `evaluate.py` and `--eval-limit` on `train.py` for quick
iteration, then run the full evaluation before reporting a result.

## Managed Runner

The managed cloud path is Kaggle Kernels via `kaggle_job.py`:

- Configure Kaggle credentials at `~/.kaggle/kaggle.json` or via `KAGGLE_USERNAME`.
- Launch with `python scripts/kaggle_job.py launch --mode experiment`.
- Poll status with `python scripts/kaggle_job.py status`.
- View logs with `python scripts/kaggle_job.py logs`.
- Download outputs with `python scripts/kaggle_job.py output`.
- The job runs `train.py` and the fixed `evaluate.py`.
- Results are persisted under `.runtime/kaggle-logs/` including `final_model/`, `metrics.json`, and logs.

Use `--train-args` and `--eval-args` for smoke runs or controlled variants, but
do not bypass the final fixed evaluator before claiming a score.

## Pi Agent Control Plane

Project-scoped Pi assets live in `.pi/`.

- Start Pi from this project root.
- Use `/posttrain "nanochat sft improvements" 1 3` for the parent flow.
- Use `.pi/agents/planner.md` for read-only method queues.
- Use `.pi/agents/reviewer.md` for benchmark-integrity checks.
- Use `.pi/agents/reporter.md` for Hugging Face Jobs status.
- Use `.pi/agents/memory-keeper.md` for `research/notes.md` and
  `research/results.tsv`.
- Use `.pi/agents/experiment-worker.md` for one coherent method change and one
  managed run.

Do not use the pre-training `/autolab` prompt or pre-training worker scripts in
this project.

## Task Sandbox

For PostTrainBench-style operation, create an isolated task directory:

- `uv run scripts/create_task.py --num-hours 1`

Then work inside the printed task directory. It contains `prompt.txt`,
`timer.sh`, the fixed `evaluate.py`, and a starter `train.py`.
