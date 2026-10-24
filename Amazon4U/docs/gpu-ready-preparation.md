# Execution Readiness, phase 1: GPU-READY preparation plan

Status: **Phase 1 decisions agreed (2026-10-24).** Next: local CPU implementation. Nothing in this
plan changes code or launches a job. Each later step needs the approval named in the
gate table below. Deployment remains `NOT_AUTHORIZED`.
Machine-readable companion: [`../configs/gpu-ready-preparation.json`](../configs/gpu-ready-preparation.json).

## User decisions recorded on 2026-10-24

- **Scope (agreed).** "Execution Readiness" means GPU-READY plus the open gates that
  profiling needs (G2 to G5 in `../configs/training-harness-preparation.json`). It ends
  with an approved Musical_Instruments profiling plan, not a profiling run.
- **Two phases (agreed).** Phase 1 is this preparation. Phase 2 is the bounded tiny
  GPU runs on Kaggle and their review, only after phase 1 is approved.
- **Orchestration path (agreed).** Use the accepted `multiautoresearch` Kaggle runner
  and Amazon4U adapters. Do not use its LM or Movie4U methodology and do not add a
  second runner or GPU scheduler.
- **`READINESS_GATES` stays frozen (agreed).** Both copies, in
  `scripts/training_harness.py` and `multiautoresearch/adapters/amazon4u.py`, and the
  `v1` result format stay byte-identical. They are the honest scope label of CPU and
  offline results. GPU-READY adds a separate, versioned profile with its own gate list.

## User decisions on the phase-1 deliverables, 2026-10-24

| Item | Decision |
|---|---|
| EX01 | Agreed as a profiling prerequisite |
| Heterogeneous aggregation | Agreed: full neighborhoods for Musical_Instruments profiling. Electronics is decided from measured step time before comparative training; sampling only through a recorded amendment |
| GR02 | Agreed: option C, GPU support built into EX01 |
| Names and checkpoints | Agreed: the new profile and schema names and a GPU checkpoint profile beside the CPU one. The detailed GR06 design in `gpu-ready-design.md` is approved, with two user-required rules: the GPU RNG state must not change during an epoch, and resumed training must be bit-for-bit identical |
| G2 | Agreed: LightGCN IDs use Normal(0, std 0.1) (option A). The formal protocol amendment is written with the other pre-comparative settings |
| G3 | Agreed as provisional starting values for profiling |
| G4 | Agreed: "you can check access using kaggle-cli and create kernel-metadata.json to start training if need." The agent's check was refused by the local permission classifier. The user ran it instead, so access is verified for account `dungsunf`. Weekly quota: the full 30 GPU-hours are available (user-reported) |
| Phase-2 budget | Agreed as proposed |

Kaggle offers T4 GPUs as a pair (T4 x2). "One T4" in the budget means one of the pair,
selected with `CUDA_VISIBLE_DEVICES=0`. The run records both the visible and the used
device counts.

## Why this is needed

H-REF proved training correctness on CPU and O-OFFLINE proved the job pipeline with a
fake Kaggle. Neither shows that the code runs on a GPU, matches the CPU results, resumes
correctly, or works in the real Kaggle environment. The sequence and required evidence
are fixed in [`training-harness-device-transition.md`](training-harness-device-transition.md).

## Current state found during preparation

- **No local GPU.** This machine has no `nvidia-smi` and PyTorch reports CUDA
  unavailable (`torch 2.10.0+cu128`). Every GPU run therefore needs remote Kaggle
  execution, which needs gate G4 and Kaggle credentials. None are granted.
- **CPU guards in accepted code** (correct for the CPU scope, unchanged now):
  - `scripts/differentiable_models.py`: rejects CUDA tensors and non-CPU devices for
    parameters and the LightGCN adjacency (around lines 201 to 210 and 1022 to 1031).
  - `scripts/training_harness.py`: `CPUBackend` reports only `cpu`/`none` and
    `require()` refuses other devices with no fallback (line 378 onward). The H-local
    Adam and RNG streams are CPU-only (`torch.Generator(device='cpu')`).
  - `scripts/checkpoint_payload.py`: the CPU profile rejects tensors off CPU and an
    active `PyTorch_CUDA_if_applicable` RNG role (around line 1166).
- **Locks in the accepted O-OFFLINE adapter** (`multiautoresearch/adapters/amazon4u.py`):
  `accelerator` must be `none` and `device` must be `cpu` in specs, plans, H envelopes
  and collected results. Only `FakeKaggleTransport` exists; a live run is refused (OP18).
  Allowed profiles are `H-REF-synthetic-cpu-v1` and `amazon4u`; mode is only
  `fixture_verification` with `owned_synthetic_packet` data.
- **Legacy runner.** `multiautoresearch/post-training/scripts/kaggle_job.py` can build
  Kaggle GPU metadata (`enable_gpu`, T4 machine shape). Its legacy live path still has
  the L1 to L9 defects found by O-OFFLINE. A real transport must keep O-OFFLINE's
  bounded, fail-closed behavior and must not reuse those legacy live paths.

## Frozen artifacts and preservation rules

- The 28 H28 harness files, `amazon4u.py`, `amazon4u_kernel.py`, `adapters/__init__.py`,
  both O test modules and all accepted models, evaluator and D1 to D3 code keep their
  recorded hashes unless a step below explicitly scopes a change and its review.
- New GPU code goes beside the accepted CPU code, selected by an explicit profile or
  schema version. The CPU path and every `v1` schema must validate exactly as now.
- Every phase-2 change must keep the full suite passing (currently 515 tests, including
  the 103 O-only tests), plus new GPU-profile tests. No assertion is weakened.
- Science stays frozen: models, BPR, endpoint penalty, Adam settings, masking, LightGCN
  Option B and evaluation rules. GPU work changes placement, not mathematics.

## Requirement and evidence matrix

Status values: `open` (not started), `decision` (needs a recorded choice first).

| ID | Requirement | Evidence required for closure | Status |
|---|---|---|---|
| GR01 | Explicit device seam: a GPU backend beside `CPUBackend`, selected by profile; unsupported requests fail before launch | Unit tests with fakes on CPU; CPU path byte-identical | open |
| GR02 | Device-aware tensor, index and sparse-buffer construction for both models, no `.cpu()` scattered in the controller | Source review; CPU suite unchanged | open |
| GR03 | Numerical parity on tiny fixtures: scores, loss, gradients and Adam updates vs the CPU oracle, both models | Kaggle run report with recorded tolerances; no bitwise promise; no new tie epsilon | decision (tolerances) |
| GR04 | Determinism: deterministic algorithms on; unsupported CUDA ops fail loudly; exceptions only with explicit approval | Run log showing settings and any refusal | open |
| GR05 | Recorded environment: GPU model, driver, CUDA and library versions and the accelerator actually allocated | Collected run record; a metadata label alone is not evidence | open |
| GR06 | GPU checkpoint profile: versioned payload profile including CUDA RNG/runtime state beside the accepted CPU profile; D3 storage bounds kept | Unit tests on CPU for schema; GPU restart vs uninterrupted comparison in the same Kaggle environment | decision (profile design) |
| GR07 | Versioned GPU result and record schema with its own remaining-gate list; `v1` and `READINESS_GATES` untouched | Collection tests for both versions; v1 fixtures pass unchanged | open |
| GR08 | Real Kaggle transport behind the existing transport interface: exact submission receipt, bounded status polling, bounded logs and output collection, no overwrite | Fake-transport tests still pass; one live receipt reviewed | decision (G4) |
| GR09 | Kaggle packaging, imports, native IO and artifact collection work in the real environment | Phase-2 run report | open |
| GR10 | Budgets enforced: run count, wall time, GPU-hours and output size, with no automatic retry | Plan limits recorded before launch; run receipts within them | decision (budget) |

## Profiling-prerequisite gates handled in phase 1 (decisions only)

| Gate | What phase 1 prepares | Decision owner |
|---|---|---|
| G2 LightGCN initialization | Options with rationale for the pending `LightGCN_ID_initialization_distribution` | User (frozen-science change) |
| G3 Execution settings | Provisional profiling settings: batch sizes, GraphSAGE fanout, LightGCN exact propagation execution, validation every epoch | User; provisional for profiling only, not comparative runs |
| G4 Remote execution | Kaggle use, account, credential handling (never printed or stored in reports) and a quota check | User |
| G5 Run provenance | Complete bundle/source/data/config/checkpoint binding for real runs, beyond D1's 20 paths | Agent proposes; user approves |

G6 (approval for each new execution) applies throughout: every phase-2 run is pinned
in advance with exact commands, limits and outputs.

## Phase-2 budget (agreed 2026-10-24)

The weekly 60 GPU-hours in `../configs/compute-plan.json` is unverified (`quota_verified:
false`). GPU-READY uses tiny synthetic fixtures only, so the proposed ceiling is small:

- At most **3 Kaggle GPU runs**, at most **30 minutes wall time each**, **1.5 GPU-hours
  total**, one T4 GPU. No automatic retries; each further run needs a recorded reason.
- No real Amazon data, no held-out data, no profiling and no comparative training.
- Local CPU verification (full suite) is allowed under existing limits: 240 s and
  2 GiB per full run.

## Phase-1 deliverables

All five are written and await review (2026-10-24):

Reviewed by the user on 2026-10-24; decisions are in the table at the top.

1. This plan and its JSON companion.
2. [Design note](gpu-ready-design.md) for GR01, GR02, GR06 and GR07. It also records
   that the accepted models cannot run full-scale profiling and proposes **EX01**, a
   vectorized execution path.
3. [G2 and G3 options](gpu-ready-g2-g3-options.md), with recommendations.
4. [Kaggle execution and provenance](gpu-ready-kaggle-execution.md) for G4, G5 and GR08.
5. [Phase-2 run plan](gpu-ready-phase2-run-plan.md), with fixtures, tolerances, limits
   and outputs ([JSON](../configs/gpu-ready-phase2-run-plan.json)).

## Approval gates

| Step | Needs |
|---|---|
| Phase-1 notes (deliverables 2 to 5) | Nothing new; preparation only |
| Writing GPU-profile code beside accepted code | User approval of the design notes |
| Any change to a frozen accepted file | Separate explicit scope and preservation review |
| First Kaggle run or credential access | G4 and the budget are agreed; a run also needs the local CPU work and full suite passing first |
| GPU-READY acceptance | Independent review of GR01 to GR10 evidence |
| Musical_Instruments profiling plan, then its run | Separate approvals after GPU-READY |
