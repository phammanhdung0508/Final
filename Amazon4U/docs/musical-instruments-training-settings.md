# Musical_Instruments comparative-training settings

Status: **AGREED by the user (2026-10-26)** for the settings in the table below. The
user's answer was "I would like `Tuning configurations` increase to 3. everything else
lgtm.", given in reply to the agent's per-model table. This record does **not** authorize
training. Explicit training approval, bound to a built bundle, is still required.

Machine-readable companion:
[`../configs/musical-instruments-training-settings.json`](../configs/musical-instruments-training-settings.json).
The pinned `configs/compute-plan.json` and `configs/training-semantics.json` are left
unchanged. Where they list these items as pending, this record holds the decision.

## Frozen settings

| Setting | LightGCN | Heterogeneous KG + GNN |
|---|---|---|
| Batch size | 2048 | 1024 |
| Tuning configurations per method, per condition | 3 | 3 |
| Checkpoint every | 20 epochs | 10 epochs |
| Also kept | Final and best-model checkpoints | Final and best-model checkpoints |
| Checkpoint format | Identity v2 (`checkpoint-identity-v2-plan.md`) | Identity v2 |

The GPU budget is shared by both learned methods:

- **Cap:** 24 GPU-hours for Musical_Instruments comparative training. That is the verified
  30 hours a week less the plan's 20 % reserve.
- **Booking:** each run books its measured worst case plus 25 %, about 0.7 h for a
  LightGCN run and 2.1 h for a KG + GNN run. Actual use is recorded after each run, and
  the ledger then counts it in place of the booking.
- **Limit:** no run starts if its booking would take the total past the cap. The agent
  stops and asks the user instead.

KG-only and Popularity are unaffected. They need no GPU training, batches or checkpoints.

## Why (agent recommendations the user accepted)

- **LightGCN 2048, not 4096.** 4096 halves the Adam updates per epoch under the frozen
  learning rate and the 100-epoch limit. That risks an undertrained baseline, and it
  saves only about 1 to 2 GPU-hours.
- **KG + GNN 1024.** This is the only size profiled. Its cost is a full-neighborhood
  pass per batch.
- **Same number of tuning configurations for both methods.** Giving one method more
  tries would make the comparison unfair. The user chose 3, the top of the planning
  range (2 to 3) in `compute-and-tuning-plan.md`.
- **Checkpoint spacing.** Run 3 proved resume exact, so spacing affects cost, not
  results. A LightGCN checkpoint costs about one LightGCN epoch, so its spacing is wider.

## Consequence the user should know: the worst case was slightly over the cap (superseded by trial reuse below)

Worst case means every run reaches 100 epochs (profiling report, plus run 3 checkpoint
costs):

| Part | Runs | GPU-hours |
|---|---|---|
| Tuning: 3 configs × 2 methods × 2 conditions × 1 seed | 12 | 12.2 |
| Final: 3 seeds × 2 methods × 2 conditions | 12 | 12.2 |
| Checkpoints, at the agreed spacing | — | about 0.3 |
| **Total** | 24 | **about 24.7** |

This is about 0.7 h over the 24 h cap. Early stopping (patience: 10 checks) will usually
end runs sooner, but by how much is unknown before training. Under the agreed limit rule,
the agent would stop and ask before the run that crosses the cap. This is a known
possibility, not a hidden overrun.

## Still pending before training

- ~~What the 3 tuning configurations vary~~: **agreed 2026-10-26**. The learning rate is
  0.0005, 0.001 or 0.002 for both methods
  ([`learning-rate-tuning-amendment.md`](learning-rate-tuning-amendment.md)).
- ~~Trial reuse~~ and ~~validation frequency~~: agreed 2026-10-26 (see the end of this
  record). The tuning seed is 42.
- **The real-data training run** (implementation) and **explicit training approval**.

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
