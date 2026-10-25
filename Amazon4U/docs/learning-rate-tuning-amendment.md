# Amendment: tune the learning rate (3 values, both learned methods)

Status: **AGREED by the user (2026-10-26)**: "ok, i agreed. Option A as your recommend look
good to me." It was recorded before any comparative training or tuning result exists, and
no validation or test result has been seen. This record authorizes no training.

Machine-readable companion:
[`../configs/learning-rate-tuning-amendment.json`](../configs/learning-rate-tuning-amendment.json).
Settings it completes:
[`musical-instruments-training-settings.md`](musical-instruments-training-settings.md).

## What changes

| | Before (Block 2, frozen) | After (this amendment) |
|---|---|---|
| Adam learning rate | Constant 0.001 | Tuned over **0.0005, 0.001, 0.002**; constant within a run |
| What the 3 tuning configurations vary | Not defined | The learning rate only, one value per configuration |
| Applies to | — | LightGCN and Heterogeneous KG + GNN, the same three values for both |

Everything else in Block 2 is unchanged: Adam's betas, epsilon and zero weight decay, the
endpoint penalty, the 100-epoch maximum, and patience of 10 validation checks. Block 1,
the frozen metrics and the candidates are also unchanged. KG-only keeps its own fixed
grid of 3 mixtures. Popularity has nothing to tune.

## Selection rule (fixed now, before any result)

- Per method, per feedback condition, run the three configurations once each with the
  tuning seed (pending; see below).
- Select by **validation NDCG@10 only**, at each run's own selected checkpoint.
- An exact tie keeps the earlier entry in the order 0.0005, 0.001, 0.002.
- The test split is never used for selection.
- The final runs train the selected learning rate with seeds 42, 2026 and 3407.

## Why

The user agreed with the agent's recommendation:

- The learning rate is the standard setting to tune.
- Tuning it treats both methods the same.
- The three values bracket the frozen 0.001.

The alternatives were:

- **LightGCN starting values.** Only LightGCN has an open setting of that kind, so tuning
  it would favour one method.
- **No tuning.** The user wanted 3 configurations.

## How tuning is run

Also agreed, as an approach, in the same exchange: the agent proposes the full plan, the
user approves and freezes it, and the agent then runs every configuration in the plan. The
agent does not choose a next configuration after seeing scores, and does not give one
method more tries than the other.

Approving the whole frozen plan at once, instead of each Kaggle run, was proposed. It will
be put to the user as a separate approval when the training bundle exists. Until then, the
existing per-run approval rule applies.

## Consequences to carry out before training

1. **Protocol version.** This is a substantive methodological amendment, so under
   `protocol-lifecycle.md` it becomes **`amazon4u-protocol-v2.1.0`**, with a new snapshot
   under `configs/protocols/v2.1.0/`. v2.0.0 stays intact. Creating the snapshot is part
   of the build step. It has not been done yet.
2. **Code and checkpoint checks pin 0.001 today.**
   - The training harness's Adam options use `lr: 0.001`.
   - The checkpoint validation, D2 v1 and the v2 checks derived from it, refuses any
     identity whose learning rate is not 0.001.

   Allowing exactly the three approved values is a scoped change to frozen D2 and harness
   files. It follows from this amendment and will be made with the same proof standard as
   before: v1 left as it is, tests and deliberate breaks. No other value will be accepted.
3. **Still pending.**
   - The tuning seed.
   - Whether the selected configuration's tuning run may count as a final seed.
   - Confirming validation every epoch.
   - The real-data training run, and explicit training approval.

## Budget

There are still 3 configurations, so the worst case is unchanged at about 24.7 GPU-hours
against the 24-hour cap. The settings record explains this.

## Further decisions (2026-10-26)

User: "1. seed 42 is ok to me. 2. yes, wining tuning should count as a final run. 3. oh, yes please use validation every epoch."

| Decision | Agreed |
|---|---|
| Tuning seed | **42**, for every tuning configuration of both methods and both conditions |
| Trial reuse | The **selected configuration's tuning run counts as final seed 42**. Final runs then train only seeds 2026 and 3407 |
| Validation frequency | **Full-ranking validation every epoch**. Patience of 10 checks is therefore 10 epochs |

Trial reuse is valid because a tuning run is a complete final-style run: the same recipe,
100-epoch maximum, early stopping, checkpoints and selected-checkpoint rule, under the
frozen seed. Selection uses validation only, so reusing the run adds no test exposure.

**Budget after these decisions** (worst case, every run reaches 100 epochs):

| Part | Runs | GPU-hours |
|---|---|---|
| Tuning: 3 learning rates × 2 methods × 2 conditions, seed 42 | 12 | 12.2 |
| Final: seeds 2026 and 3407 × 2 methods × 2 conditions | 8 | 8.1 |
| Checkpoints | — | about 0.3 |
| **Total** | 20 | **about 20.6**, under the 24 h cap |
