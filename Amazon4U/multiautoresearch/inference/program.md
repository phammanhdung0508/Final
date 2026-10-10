# Program

Optimize recommendation serving and Top-K dot-product inference for speed.

The benchmark target is serving HeteroGraphSAGE user and movie embeddings for
real-time recommendation scoring across the full catalog.

Start with:

```bash
uv run scripts/benchmark_serving.py --runs 20 --warmup-runs 5
```

Then benchmark and record hypotheses:

```bash
uv run scripts/benchmark_serving.py \
  --batch-size 64 \
  --precision fp32 \
  --threads 4 \
  --experiment-id "exp-batch64-fp32" \
  --append-tsv research/results.tsv
```

Improve one variable at a time: batch size, thread parallelism, FP16/INT8
quantization, candidate pre-filtering, or top-k algorithms (torch.topk,
argpartition, or vector indexes). Do not claim a speedup until the same benchmark
has been run before and after the change.
