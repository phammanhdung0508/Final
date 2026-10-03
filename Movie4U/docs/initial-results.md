# Initial MVP experiment

Configuration: seed 42, 50 CPU training epochs, 64 hidden channels, two GraphSAGE layers, checkpoint selected at epoch 50 by validation NDCG@10. The final layer keeps signed embeddings, permitting negative dot-product logits for binary cross-entropy. An earlier smoke implementation applied final-layer ReLU; that decoder limitation was corrected before this recorded run. No hyperparameter search was performed.

| Method | Test Precision@10 | Test Recall@10 | Test NDCG@10 | Test HitRate@10 |
|---|---:|---:|---:|---:|
| KG-only | 0.006798 | 0.010132 | 0.011959 | 0.060823 |
| KG + GraphSAGE | 0.012165 | 0.025350 | 0.018920 | 0.107335 |

559 users have positive test interactions; 51 are excluded from test metrics. Validation evaluates 555 users. The GNN outperforms the genre baseline on these ranking metrics in this single run, but both have low full-catalog ranking quality. GNN catalog coverage is lower (about 1.37%, versus 8.85% for KG-only). This is not evidence of SOTA performance or guaranteed general improvement.

Machine-readable outputs are generated at `artifacts/metrics/comparison.json`. Training history is in `artifacts/models/training_history.json`. Rerunning with different parameters may change results; update this summary rather than treating it as a permanent benchmark.

Next model work should use validation-only tuning, multiple seeds, stronger baselines, and uncertainty estimates. Keep the test set isolated from model selection. The current test has been used for MVP inspection and reporting; future confirmatory research should use a fresh held-out protocol.
