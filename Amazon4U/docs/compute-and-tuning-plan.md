# Compute and tuning plan — agreed approach, numerical budget pending

## Conditional Kaggle capacity

**Decision:** plan around approximately **60 GPU-hours per week only if** the
available account/session usage is permitted under Kaggle's current rules. Verify
actual quota accounting, available hardware and session limits first. This record
is not an instruction to bypass account or quota restrictions.

GPU-hours are allocated compute time, not necessarily wall-clock time. Parallel
sessions can shorten elapsed time but do not reduce aggregate GPU-hours. Hardware
can differ between runs; record GPU model, software environment and runtime.
GPU is the initial platform recommendation; TPU compatibility is not established.

## Agreed experiment-planning approach

- Keep **three final training seeds** for each learned method/category/feedback
  condition: **42, 2026, 3407**. Block 3's RNG/provenance and conditional paired
  bootstrap framework are agreed in [reproducibility](reproducibility-and-uncertainty.md).
  The bootstrap contrast list is now frozen: KG + GNN minus each of four core
  baselines for NDCG@10/HitRate@10, separately within all six experiment groups.
- Use a small, profiling-informed hyperparameter search. **2–3 configurations per
  learned method per category × condition** is the planning range, not a frozen
  numerical trial budget or permission to pick its size after observing results.
- Freeze search spaces, tuning seed, trial counts and stopping/budget rules after
  resource profiling, **before comparative tuning/model results**. Select models
  and checkpoints using validation NDCG@10 only; never use test performance or
  bootstrap intervals for selection.
- Reserve approximately **20%** of available capacity for interruptions, debugging
  and reruns. At a verified 60-hour allocation, roughly 48 hours remain for planned
  runs and 12 hours for contingency. These are estimates, not measured capacity.
- Preserve full-data scope, full-ranking evaluation and condition-specific rules.
  Additional capacity does not automatically double the tuning budget or authorize
  sampling users/interactions/candidates.
- Parallelize independent runs when permitted. Use distinct immutable run
  directories and equivalent recorded environments/configurations; do not merge
  checkpoints, RNG state or logs between runs.
- Plan execution over multiple weeks if needed. Save resumable checkpoints,
  optimizer/RNG state, configurations and logs to durable permitted storage, not
  only an ephemeral runtime. Log failures and restarts; never silently change the
  scientific configuration in response to an OOM or quota interruption.

Three learned core methods (BPR-MF, LightGCN, heterogeneous KG + GNN) across six
category × condition groups and three seeds imply **54 final fits**, before extra
tuning. Two or three configurations per learned method/group would imply 36 or
54 tuning trials, respectively. Do not assume tuning and final-run costs are
identical, or that previously computed trials may be reused without a registered
reuse policy. Popularity and KG-only have separate runtime/weighting budgets.

## Lifecycle and authorization

The baseline is frozen as `amazon4u-protocol-v1.0.0`; this compute plan remains
pending measured execution settings and numerical tuning policy. See
[protocol lifecycle](protocol-lifecycle.md). The profiling-first approach below is
agreed, but **no bounded profiling run is authorized yet**. Approval must declare
scope, compute limits, provisional settings, permitted optimizer updates and outputs
following prerequisite regression verification. Comparative training is separately
unauthorized until execution/tuning freeze and explicit approval.

## Profile first: full-ranking validation

The previously agreed validation-profiling decision is now recorded in Amazon4U:

1. Start with full-ranking validation every epoch on Musical_Instruments.
2. Measure training-epoch time, end-to-end validation time (embedding inference,
   candidate/history filtering, ranking and metrics), and peak memory. Include
   validation cost when estimating complete-run hours.
3. Keep the per-epoch schedule if practical; do not predesign a workaround.
4. Recheck feasibility on Toys_and_Games and Electronics. Musical_Instruments
   feasibility does not establish Electronics feasibility.
5. Batch/stream full ranking rather than materializing the entire user–item score
   matrix. This is an execution strategy, not a smaller candidate universe.
6. Any necessary frequency change requires an agreed, documented amendment before
   comparative experiments. Early-stopping patience is measured in validation
   checks, not epochs. Block 2 now approves patience **10 validation checks**;
   the final profiled validation frequency is still pending.

Estimate required GPU-hours by summing measured/projected hours over all planned
runs, documenting extrapolation assumptions. Report actual trial counts and
compute as well as nominal search budgets; equal trial counts need not mean equal
compute or equal model complexity.

**Rationale:** a second permitted allocation makes experiments more feasible, but
full-ranking validation and larger datasets may dominate costs. Prioritize final
seed variability and fair, modest tuning rather than an unmeasured large search.

**Status:** user-approved planning approach; quota permission/availability not
verified, profiling not performed, and exact tuning budgets/search seed not frozen.
Final training seeds are frozen in `configs/reproducibility.json`.
No models have been trained. Configuration: `configs/compute-plan.json`.
This decision does not remove the overall pre-training protocol gate.
