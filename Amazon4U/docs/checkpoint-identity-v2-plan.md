# Checkpoint identity v2 plan: store fingerprints, not the data copy

Status:

- **Direction agreed by the user (2026-10-25):** "Ok, let's start with 'Store hashes
  instead of the full data copy'. I agree on this one, it ok to modify older checkpoint
  design if it's no longer work effectively (note as new version) - write me the plan."
- **This plan: AGREED by the user (2026-10-25):** "Approve the plan? APPROVE, I agree with your
  plan. Book Kaggle run 3 in the profiling budget? Yes use run 3 profiling budget for this
  plan implement ok too me". The Kaggle run 3 launch still needs approval of its built
  bundle.

Machine-readable companion:
[`../configs/checkpoint-identity-v2-plan.json`](../configs/checkpoint-identity-v2-plan.json).
Context: [`checkpoint-real-data-readiness-plan.md`](checkpoint-real-data-readiness-plan.md).

## Why

Checkpoint proof run 2 (Kaggle T4, Musical_Instruments) showed that resume is exact in all
four cases. It also showed that publish takes 155 to 182 s and retrieve 141 to 170 s, against
the agreed 60 s targets. Files are 428 to 467 MB. One checkpoint costs 3 to 10 epochs.

The local profile (readiness plan, option A) puts most of the remaining cost on the data
copy that the D1 identity carries in every checkpoint: about 400,000 training records,
about 1,040,000 graph edges, every user's histories, the product metadata and the
validation inputs. Every publish and every retrieve deep-copies it, walks it to check
memory limits, recomputes three fingerprints from it, compares it field by field, and
encodes or decodes it. Electronics has about 27× the edges, so this cost would grow with it.

## What changes

Version 2 keeps the identity's meaning and drops its bulk.

| | v1 (today, unchanged) | v2 (new) |
|---|---|---|
| Identity in the checkpoint | Header, five fingerprints, resolved settings **and the full data copy** | Header, five fingerprints and resolved settings only |
| Where the content comes from at resume | The checkpoint's own copy, rechecked against the caller's expected copy | The caller's trusted copy, built from the hash-verified dataset (`ProfilingData`). It is never read from the checkpoint |
| How a wrong dataset is caught | Stored fingerprints recomputed from the stored copy, then compared with the expected identity | Stored fingerprints compared with fingerprints the caller computed from its own data |

The same five fingerprints are kept and computed by the same D1 functions: data, graph,
validation, settings and the identity hash. A v2 identity therefore carries exactly the
`identity_hash` a v1 identity of the same data carries. This is checked by a test.

### Checks that use the content

Some D2 checks read the identity's content. In v2 they read the trusted expected content
instead. This content is held in memory, built once per process, and never stored:

- the positive-pair domain P and the empty-pool pairs K for the sampler and progress
  checks;
- model row order against users, warm items and graph nodes.

What a v2 checkpoint must match is unchanged. Only where the reference content lives
changes.

### What is lost, stated plainly

1. **A v2 checkpoint cannot be checked without the dataset.** In v1, a checkpoint could
   prove its own content against its stored fingerprints. In v2 it can only be matched
   against data the caller supplies. Every resume already loads and verifies the dataset,
   so nothing in the agreed workflow uses the v1 ability.
2. **Less detail on a mismatch.** A v1 mismatch can name the field that differs. A v2
   mismatch names only the fingerprint that differs: data, graph, validation or settings.
3. **No content check of the checkpoint against the fingerprints.** Integrity of the
   checkpoint bytes still rests on D3's SHA-256 envelope. Binding to the data rests on
   fingerprint equality with the caller's recomputation.

### What stays

- **v1 stays as it is.** D1/D2/D3 v1 code, their frozen tests and pinned hashes are
  unchanged. Runs 1 and 2 and GPU-READY stay reproducible. v2 is added beside v1 under new
  schema names:
  - `amazon4u-checkpoint-identity-v2`;
  - `amazon4u-checkpoint-payload-v2`;
  - a cuda-v2 profile.
- **v1 and v2 refuse each other's files.** A v2 reader refuses a v1 file, and a v1 reader
  refuses a v2 file.
- **Everything else in the payload is unchanged:** the model, optimizer, RNG, sampler,
  progress, selection, provenance, device record and run binding, with all their v1 checks.
- **The D3 format and its checks are unchanged:** the bounded format, no pickle, the
  exclusive publish and the anchored retrieve. Only the payload validator it calls changes.

## Scope

In scope:

- **V1. D1 v2 identity.** A small identity record. Validate it against a trusted
  `CanonicalResumeIdentity` that the caller builds from data. The fingerprint functions are
  reused unchanged.
- **V2. D2 v2 payload validation.** v1's checks, with the identity part replaced. The P, K
  and row-order checks read the trusted expected content.
- **V3. cuda-v2 profile and D3 wiring.** Publish and retrieve call the v2 validator.
- **V4. Real-data builders and trainer.** `checkpoint_real_data` and `VectorTrainer` emit
  and restore v2 payloads. v1 builders stay for v1 tests.
- **V5. Proof, local then Kaggle.** Re-measure locally, then one Kaggle proof run, run 3,
  with the same cases and comparison as run 2.

Out of scope:

- Changing any training or evaluation science, seeds or metrics.
- The sampler's stored epoch order: about 400,000 pairs as strings in each checkpoint. It
  is kept as v1 stores it. If v2 still misses the target, it is the next candidate (guess).
- Toys_and_Games and Electronics. v2 should scale with model size instead of data size, but
  this is not measured here.

## How v2 is proven

1. **v1 untouched.** The frozen v1 tests and pinned hashes pass unchanged.
2. **Same checks outside the identity.** Every v1 rejection case that does not change the
   identity content is replayed through v2. The class and message must match v1, so the
   two versions accept and reject alike outside the identity. The v1 test fixtures are
   used.
3. **Identity changes caught.** Every v1 identity-content mutation is replayed against v2:
   a changed rating, edge, history, metadata or validation target, a changed seed or a
   changed setting. v2 must refuse each one through a fingerprint mismatch against the
   trusted data. Also tested:
   - a v2 checkpoint resumed against the other condition, method or dataset;
   - a v1 file read by v2;
   - a v2 file read by v1.
4. **Same identity hash.** On fixtures and on Musical_Instruments, a v2 identity carries the
   same `identity_hash` as the v1 identity of the same data.
5. **Exact resume.** CPU resume equality for both models and both conditions, then Kaggle
   run 3, as in run 2.
6. **Deliberate breaks.** Each new v2 check is broken on purpose once and must fail a test.

## Targets

Publish ≤ 60 s, retrieve ≤ 60 s and peak host memory ≤ 8 GiB per checkpoint on Kaggle.
These are unchanged from the readiness plan, as agreed by the user. Measured per case, as
in run 2. If v2 misses them, the agent stops and reports the numbers.

The agent's estimate: publish and retrieve fall to well under 60 s, and the file to about
100 MB (guess). The tensors are about 50 to 100 MB. The sampler order and the rest are
small next to the data copy.

## Order of work

| Step | Work | Ends with |
|---|---|---|
| 1 | V1 and V2 in a new worktree | v2 tests, the v1-equivalence replay and the identity-hash equality pass; v1 frozen tests unchanged |
| 2 | V3 and V4 | CPU resume equality for 4 cases on synthetic data |
| 3 | Local real-data timing | Publish and retrieve times, file size and memory, against run 2 |
| 4 | Kaggle run 3 bundle | Built and handed to the user; launched only after the user approves that bundle |
| 5 | Record | Readiness plan and profiling-report checkpoint cost updated |

## Approvals

| Item | Approval |
|---|---|
| Direction: store fingerprints, not the data copy, as a new version | Given (user, 2026-10-25) |
| This plan | Given (user, 2026-10-25) |
| Kaggle run 3 | Its own approval, bound to the built bundle |
| Ledger for run 3 | Agreed: the profiling ledger. 7,200 s remain of its 21,600 s cap, enough for exactly one more run at full booking |

## User decisions (2026-10-25)

| Decision | User's words |
|---|---|
| Plan approved, v1 kept unchanged beside v2 | "APPROVE, I agree with your plan." |
| Run 3 booked in the profiling ledger | "Yes use run 3 profiling budget for this plan implement ok too me" |
