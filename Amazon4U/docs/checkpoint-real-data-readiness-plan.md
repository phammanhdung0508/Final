# Real-data checkpoint readiness plan

Status: **AGREED by the user (2026-10-25)**: "Approve the checkpoint plan; targets 60 s /
60 s / 8 GiB; use the profiling ledger." Earlier the user decided that checkpoints keep the
full copy of the training data for now, the speed fix is tried first, and the question
comes back to the user if loading still fails.
Machine-readable companion:
[`../configs/checkpoint-real-data-readiness-plan.json`](../configs/checkpoint-real-data-readiness-plan.json).

## Why it is needed

The reproducibility rules require durable, resumable checkpoints
(`configs/compute-plan.json`, `durable_resumable_checkpoints: true`). D1 to D3 are accepted
on synthetic fixtures, and GPU-READY proved CUDA resume on them. On the full
Musical_Instruments data a local probe found
(`data/reports/profiling-mi-v1/checkpoint-scale-probe.txt`):

| Step | Result |
|---|---|
| D1 identity build | 30 to 35 s, 2.4 GB peak |
| Payload build | 19 to 25 s |
| D2 validation (under `cProfile`) | 1,180 s |
| D3 publish, unprofiled | 437 s, 3.8 GB peak |
| D3 retrieve | Not measured; the system stopped it for low memory |

Profiling on the T4 measured epochs of 11 to 60 s (`musical-instruments-profiling-report.md`),
so a checkpoint today costs 7 to 40 epochs.

## Where the time goes (from the profile)

| Hot spot | Calls | Time | Cause |
|---|---|---|---|
| `_validate_clean_text` (D1) | 62.7 M | 648 s | Checks every character of every ID in Python, about 1 billion `ord` calls |
| `compute_graph_fingerprint` (D1) | 8 | 624 s | Recomputed for every identity rebuild |
| `build_canonical_identity` via `from_dict` (D1) | 4 | 474 s | D2 rebuilds both actual and expected identity, and semantic validation runs twice |
| `compute_data_fingerprint` (D1) | 8 | 208 s | Same repetition |
| `preflight_resource_accounting` (D2) | 6 | 145 s | Walks the whole payload tree each time |
| `_compute_condition_positive_and_empty_pool_pairs` (D2) | 2 | 104 s | Builds `warm_items - all_rated` for every user (57,439 × 24,556 items) only to test emptiness. Equivalent: the pool is empty exactly when every warm item is in the user's rated set |

Times overlap: the first two are mostly inside the third. The fix targets repetition and
per-character Python loops, not what is checked.

## Scope

In scope:

- **C1. Real-data builders (new code, not frozen).** Build the D1 identity, provenance and
  runtime expectations from `ProfilingData` for both conditions and both models, and a D3
  IO policy sized for the real data (`max_tensor_elements` at least 2^31, because
  LightGCN's sparse adjacency counts as 57,439 × 24,556 cells). The probe did this by hand;
  this makes it a tested module.
- **C2. Speed fix in D1 and D2 (frozen files; needs approval).** Behavior-preserving only:
  - the per-character surrogate check becomes one precompiled regular-expression search
    with the same accept and reject results;
  - within one validation call, each identity is built once and each fingerprint is
    computed once, and the results are reused;
  - the resource accounting walks the payload once per validation;
  - `_compute_condition_positive_and_empty_pool_pairs` tests emptiness by counting the
    user's rated items that are warm, instead of building the difference set (read on
    2026-10-25; same result by set algebra).
  No check is removed, weakened or reordered in a way that changes which error is raised.
- **C3. Retrieve and memory.** Measure retrieve, which has never completed on real data.
  If memory is the limit, reduce copies held at once (for example, decode and validate
  without a second deep copy), still behavior-preserving.
- **C4. Real-data proof.** Locally on CPU: publish and retrieve for both models and both
  conditions, timed. Then one Kaggle run (its own approval): train 2 epochs, publish,
  retrieve in a fresh process, resume and finish 4, and compare against an uninterrupted
  4-epoch run, as GR06 did on fixtures. Record publish, retrieve, size and memory.

Out of scope:

- Changing what a checkpoint stores. The full copy stays (user, 2026-10-25). Storing only
  hashes would change the frozen D1 design and is a separate decision, raised only if C3
  shows loading still fails.
- Any change to science, training, selection or metrics.
- Toys_and_Games and Electronics. Electronics has about 27× the edges; its checkpoint cost
  needs its own check before its training (guess: the full copy may not scale there).

## How behavior preservation is proven

1. **The frozen tests stay unchanged and pass**: `tests/test_checkpoint_identity.py`,
   `tests/test_checkpoint_payload.py` (including V02's byte-exact documentation prefix),
   `tests/test_checkpoint_storage.py`, `tests/test_checkpoint_cuda.py` and the H-REF tests.
2. **A differential test** runs the accepted D1/D2 (from `master`, copied into a test-only
   module) and the changed ones on the same inputs: the accepted fixtures, every rejection
   case the frozen tests build, and new mutations aimed at the changed code (lone
   surrogates at the start, middle and end of an ID; whitespace padding; a changed graph
   edge; a changed fingerprint). Accept or reject, the exception class and the message must
   be identical.
3. **Real-data equality**: on Musical_Instruments, the changed code accepts the same
   payload and produces byte-identical published files as the accepted code (expected,
   because D3's encoding is canonical; step 2 checks that two publishes of the same payload
   are identical before relying on it). Locally the
   accepted code takes about 7 minutes per publish, which is slow but feasible once.
4. **Deliberate breaks**: each changed check is broken on purpose once and must fail a test.

## Frozen-file procedure

D1 to D3 are pinned by `FrozenFileTests` in `tests/test_checkpoint_cuda.py`, against the
hashes in `configs/o-offline-implementation.json` plus its `AMENDED` table. After approval,
each changed file gets an `AMENDED` entry with its new SHA-256, and this plan records the
previous and new hashes, the change and the user's approval words, as the O-OFFLINE pin
amendment did. The recorded historical hashes are not edited. Work happens in a new
worktree.

## Targets (agreed by the user)

On Musical_Instruments, per checkpoint, on Kaggle:

- publish at most 60 s and retrieve at most 60 s;
- peak host memory at most 8 GiB during either.

60 s is about one KG + GNN epoch. If the speed fix cannot meet these, the agent stops and
reports the measured numbers rather than lowering the bar or changing what is stored.

How often to checkpoint is a training-budget decision, not part of this plan. Profiling
showed every Musical_Instruments run fits one Kaggle session (longest about 1.7 hours for
100 epochs), so checkpoints mostly guard against a failed session.

## Order of work

| Step | Work | Ends with |
|---|---|---|
| 1 | C1 builders and tests | Real-data identity, policy and IO policy accepted by the accepted D1 to D3 on fixtures and on real data (slow path) |
| 2 | Measure the accepted code once on real data | Baseline publish, retrieve and published-file hashes (retrieve may need the machine's RAM freed; the user is asked first) |
| 3 | C2 speed fix | Frozen tests, differential test and deliberate breaks pass; real-data files byte-identical |
| 4 | C3 if needed | Retrieve completes within the memory target |
| 5 | C4 local, then Kaggle with approval | Times, sizes, memory and resume equality recorded |
| 6 | Update the comparative-training projection with checkpoint cost | Report addendum |

## Approval steps

| Step | Approval |
|---|---|
| This plan | Needed |
| C2 and C3 changes to frozen D1 to D3 files | Covered by approving this plan, limited to the behavior-preserving changes above |
| Step 2 local real-data run (and freeing RAM if needed) | Covered by approving this plan; local CPU only, no weights kept |
| The C4 Kaggle run | Its own approval, bound to a built bundle; booked in the profiling ledger (4.0 GPU-hours remain under its 6-hour cap) |

## User decisions (2026-10-25)

| Decision | User's words |
|---|---|
| Plan approved, including the scoped frozen-file changes | "Approve the checkpoint plan" |
| Targets 60 s publish, 60 s retrieve, 8 GiB host | "targets 60 s / 60 s / 8 GiB" |
| C4 Kaggle run booked in the profiling ledger | "use the profiling ledger." |

## Status note (2026-10-25)

After this draft, the user wrote: "Im Headache, you can decide by yourself. and handle as
you be in order to finish profiling. Im going to take a rest". The agent did not treat this
as approval of the frozen-file changes. The plan stays PROPOSED until the user approves it
in so many words. No code was written. The user then approved it (above).

## Progress (2026-10-25)

Work is in worktree `.worktrees/checkpoint-real-data` (branch `checkpoint-real-data`), not
committed. Evidence is in `data/reports/checkpoint-readiness-v1/` (git-ignored). All local
CPU timings are on a 7.8 GB machine that was swapping, so they are approximate.

**C1 done.** `scripts/checkpoint_real_data.py` builds the D1 identity, expected model and
policies from `ProfilingData`. It keys `positive_history` by every training user, because
D1 requires it; under ≥4, users without positives map to `[]`. `scripts/checkpoint_timing.py`
times one publish, or a retrieve in a new process. `tests/test_checkpoint_real_data.py`
covers resume equality for both methods and both conditions, refusal across condition or
method, and refusal of a changed record.

**Baseline, accepted code** (LightGCN, all ratings, epoch 0): publish took 468 s, with a peak RSS of
3.3 GB. The file is 368,964,132 bytes, SHA-256 `0287358e…b919`. Retrieve was not measured:
swap was 7 of 8 GB used, and this plan asks the user before RAM is freed.

**C2 done, target not met.** Changes:

- D1 `_validate_clean_text`: one regex search for exact `str`.
- D2 empty-pool test: `warm <= rated`.
- D1 `_validate_identity_pair(content_verified=True)`, used only by D2 right after
  `from_dict`. It skips the three recomputes that `from_dict` has just done.
- D2 builds the expected identity once for both passes.

The public `validate_canonical_identity` is unchanged. New hashes:

| File | SHA-256 |
|---|---|
| `checkpoint_identity.py` | `da31e178…c5a` (was `6c91bab8…b2c`) |
| `checkpoint_payload.py` | `1f2a385e…8da` (was `6bc4d2f0…844`) |

Both are pinned in the `AMENDED` table of `tests/test_checkpoint_cuda.py`.

Proof:

- **Byte-identical file.** The changed code publishes the same file as the accepted code
  (same SHA-256).
- **Dual run.** All 207 frozen checkpoint tests ran with every D1/D2 call executed by both
  the accepted copy and the changed code (`tests/checkpoint_dual.py`). There were 0 real
  differences. Two reported differences came from a test that poisons the live module's
  owned copy only.
- **Equivalence tests.** `tests/test_checkpoint_speed_equivalence.py` has 7 tests. One is
  a 400-trial property test showing that retained content has the fingerprints `from_dict`
  checked.
- **Deliberate breaks.** 5 of 6 fail a test (`c2-deliberate-breaks.jsonl`). The sixth
  breaks the exact-`str` surrogate regex, and no test can detect it: `encode("utf-8")`
  already rejects every surrogate, so that check is unreachable in the accepted code too.
  It is kept.
- **Full suite.** 636 tests pass.

**Result.** Publish takes 168 s, down from 468 s, with a peak RSS of 3.2 GB. The target is
60 s. Measured remaining cost (`c2-step1`): the D3 `_encode` takes about 60 s on its own (two
Python passes over about 11 M nodes). Three identity builds take about 72 s, and six
resource walks about 47 s. D3 encode speed is not in this plan's listed changes. Following
the plan, the agent stopped here and reports instead of lowering the bar or changing what
is stored.

### User decisions after C2 (2026-10-25)

| Decision | User's words |
|---|---|
| Option A: behavior-preserving speed-up of D3 save and load (`checkpoint_storage.py`, frozen), proven the same way as C2 | "ok pick A then" |
| Retrieve is measured on Kaggle in the C4 run, not locally; C4 still needs its own run approval | "Kaggle please" |

### Option A done (2026-10-25)

Also in the `checkpoint-real-data` worktree, not committed. All changes are
behavior-preserving:

- **D3 save** (`_Encoder.node`, `_tagged_scalar`): the meter, reserve and scalar checks are
  inlined in their original order, with the same messages.
- **D3 load** (`_Parser.node`): the reads are inlined for scalar and container tags.
- **D3 sparse rebuild** (`_check_coo_coordinates`): the per-column loop is vectorized. The
  first failing column decides the error, bounds checked before order, as in the loop.
- **D2 walk**: an exact-`str` fast path (`_text`) with the same checks and order.
- **D1 graph fingerprint**: each exact-`str` CategoryPath value is validated once per call.

New hashes, pinned in `tests/test_checkpoint_cuda.py` `AMENDED`. The previous hashes stay
recorded in `configs/o-offline-implementation.json`.

| File | SHA-256 |
|---|---|
| `checkpoint_identity.py` | `e699a770…` |
| `checkpoint_payload.py` | `3cf065bf…` |
| `checkpoint_storage.py` | `35cebc79…` |

Proof:

- **D3 differential tests.** `tests/test_checkpoint_storage_equivalence.py` compares the
  current D3 with the pinned accepted copy:
  - 1,500 random saves under tight policies;
  - 600 resealed corrupted bodies;
  - 1,500 random sparse coordinate sets;
  - targeted duplicate-key and coordinate cases.
- **Walk and category tests.** New tests compare the walk and the graph-fingerprint category
  checks with the accepted copy.
- **Dual run.** 210 frozen and equivalence tests showed 0 real differences.
- **Deliberate breaks.** All 9 real breaks fail a test. The first pass found two test gaps,
  for duplicate keys and lexicographic coordinates. Tests were added, and the rerun detects
  both. One planned break turned out to be an equivalent rewrite and was replaced. See
  `d3-deliberate-breaks*.jsonl`.
- **Full suite.** 642 tests pass.

Local publish, all ratings (`d3-step2`):

| Model | Publish | Peak RSS | Notes |
|---|---|---|---|
| LightGCN | 125 s (accepted code: 468 s) | 3.7 GB | Byte-identical to the accepted code |
| KG + GNN | 118 s | 3.8 GB | 362,570,061 bytes |

The 60 s target is still not met. The remaining cost is about 60 s for three identity
builds, about 27 s for resource walks and about 25 s for the save. Retrieve is to be
measured on Kaggle in C4 (user decision).

### User decisions after option A (2026-10-25)

| Decision | User's words |
|---|---|
| Build the C4 Kaggle proof bundle; the launch still needs approval of that built bundle | "Go ahead with the Kaggle proof run. Yes." |
| Commit the work: code on branch `checkpoint-real-data` (commit `817108f`, not pushed), plan updates on `master` | "Save the work to git. Yes" |
