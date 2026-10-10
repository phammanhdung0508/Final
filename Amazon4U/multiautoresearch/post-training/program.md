# Program

We want to post-train the Knowledge Graph + HeteroGraphSAGE recommendation model
to maximize held-out `eval_score` (macro-averaged NDCG@10).

The fixed benchmark files are:

- `prepare.py`
- `evaluate.py`
- `model.py`

The starter experiment file is:

- `train.py`

Start with:

```bash
uv run prepare.py
```

Then improve the post-training method in `train.py`, or create additional
supporting files. Prefer ranking and collaborative filtering methods:
Bayesian Personalized Ranking (BPR) loss, hard negative sampling, popularity-aware
negative sampling, temperature scaling, embedding regularizations, learning rate
schedules, or multi-objective loss combinations.
You may change data mixing, schedules, optimization, loss shaping, or other
training strategy details, but keep the benchmark honest:

- do not train on the validation or test split
- do not modify `evaluate.py`
- do not modify generated eval records
- do not substitute another architecture
- save the best checkpoint to `final_model/`

Always finish with:

```bash
uv run evaluate.py --model-path final_model
```

For managed Kaggle runs, use the post-training Kaggle runner:

```bash
uv run scripts/kaggle_job.py launch --mode experiment
uv run scripts/kaggle_job.py logs <JOB_ID>
```

For Pi Agent orchestration, start Pi in this project root and run:

```text
/posttrain "hetero-graphsage ranking improvements" 1 3
```

The Pi flow is project-local and should not use the pre-training `/autolab`
workflow.
