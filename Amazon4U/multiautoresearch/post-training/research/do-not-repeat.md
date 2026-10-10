# Do Not Repeat: Post-Training Experiments

This document records dead ends and negative results to avoid repeating failed directions.

---

### 1. Pure Pairwise BPR Loss (Run `dungsunf/posttrain-exp-20261007-081554`)
* **Hypothesis**: Replacing pointwise BCE with pairwise BPR loss $\mathcal{L} = \text{softplus}(-(x_{\text{pos}} - x_{\text{neg}}))$ would directly optimize ranking.
* **Result**: `eval_score` dropped from **0.04769** to **0.00291** (coverage collapsed from 1.5% to 0.2%).
* **Root Cause**: The dot-product decoder produces signed logits with no absolute boundary. Without BCE's sigmoid targets ($y=1$ for positives, $y=0$ for negatives), BPR allows unbounded embedding magnitude growth and drift, collapsing catalog coverage.
* **Lesson**: Pointwise BCE is essential for anchoring logit magnitudes. If using pairwise margins, combine them with BCE (hybrid BCE + margin) or enforce cosine normalization.

---

### 2. Unweighted Negative Sampling (neg_ratio=2) with Premature Cosine Decay (Run `dungsunf/posttrain-exp-20261007-082139`)
* **Hypothesis**: Increasing negative sampling to 2:1 and decaying learning rate via CosineAnnealingLR over 70 epochs would improve tail ranking discrimination.
* **Result**: `eval_score` dropped from **0.04769** to **0.03546**.
* **Root Cause**: Unweighted BCE with `neg_ratio=2` biased gradients heavily toward negative suppression (loss stalled at 0.3707 vs baseline 0.3402), while premature LR decay prevented the model from escaping early flat saddle points before epoch 30.
* **Lesson**: Keep `neg_ratio=1` (or apply `pos_weight=neg_ratio` in BCE), and maintain sufficient learning rate (0.005) through at least 40-50 epochs.
