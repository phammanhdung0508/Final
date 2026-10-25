# KG-Only + Neo4j + GraphRAG MVP

## Authority and boundaries

User request: “Can you implement end-to-end proposed plan?”

Implementation is authorized for the explanation-first extension. Work happens in
`Amazon4U/.worktrees/kg-graphrag-mvp-v1/Amazon4U/`, not the active profiling tree.
The main scientific protocol, core models, datasets, ranking and frozen files stay
unchanged. This document does not grant a protocol amendment or deployment approval.
No external service, package installation, provider request, real-data import,
profiling, training or held-out evaluation is authorized by this implementation pass.
Those require an explicit environment and resource/API budget. No full H regression
allocation is reused. No staging, commit or push is authorized.

## Contract

1. Reuse KGOnlyScorer and TrainingContext without changing their source. Rank the
   complete condition-warm pool, filter all training history and, in test mode,
   supplied validation IDs. Preserve deterministic ties, shortages and fallback.
2. Import the four KG-v1 node types and four relations into a fresh, versioned
   Neo4j namespace. Stream input/import batches, retain provenance, refuse reuse,
   independently reconcile identities/properties/edges before marking ready.
   Real input additionally requires a separately approved manifest/eight-file SHA
   lock; a graph's own boolean validation receipt is not source authentication.
3. Retrieve bounded evidence through static parameterized read-only queries.
   Check namespace/version and metadata equality against the scoring source.
4. Separate numerical score contributions from descriptive evidence. Zero-weight
   channels are not reasons; fallback is non-personalized training popularity.
5. Supply a deterministic explanation baseline. The LLM may select evidence IDs
   only; the application renders vetted sentences. Free-form generated claims,
   Cypher, user IDs, API secrets and held-out records are not model inputs.
6. Preserve ranking across unavailable graph/provider, invalid output, timeout,
   missing metadata and malicious metadata. Invalid generated output falls back
   visibly; scoring/data corruption must fail, not become popularity fallback.
7. Provide a runnable synthetic demo, CLI/library, optional real Neo4j and HTTPS
   model adapters, fixtures, tests and honest verification records. Live integrations
   and full-data feasibility remain unverified until separately authorized.

## Implementation sequence

Graph input/provenance → unchanged reference scoring → template evidence → Neo4j
import/read adapter → constrained HTTPS generator → CLI → adversarial fixture tests.
Fixtures are correctness-only, not research subsampling. The real input adapter
consumes complete category KG-v1 Parquet artifacts and requires explicit opt-in.
Only Brand and complete CategoryPath metadata are allowed. No titles, reviews,
prices, ratings-as-likes, inferred attributes, embeddings or cross-category history.

## Local verification envelope

New fixture-only tests and the existing pure-reference recommendation/evaluation
modules may run with `-B`, no network, no package installation and no H/Torch imports.
A second fixture pass may use the already installed DuckDB package to create/read
small synthetic Parquet tables under fresh scratch. It never reads project raw or
processed datasets. This verifies the real input adapter without real-data execution.
Each verification child: 60 seconds, 512 MiB address space, 8 MiB per output stream.
Temporary files go under a fresh phase-owned report scratch directory. Source and
baseline hashes are recorded; results and failed attempts are retained. Tests may
be rerun only after a recorded in-scope correction, not to select research results.

## Unresolved operational choices

Neo4j host/edition/database and separate import/read credentials; LLM provider/model
and approved data egress; dependency installation; exact real-data import/query and
inference limits; independent live integration verification; main integration and
deployment. Neo4j is CPU/database infrastructure, not a GPU scalability guarantee.

## Status

Isolated implementation is available, but the full end-to-end request is **ESCALATE,
not accepted**. Deployment remains **NOT_AUTHORIZED**. No main source integration.

- Initial implementation pure verification: **99 tests pass**, comprising 27 new MVP checks and
  72 unchanged reference recommendation/evaluation checks, zero failures/errors/skips.
  Guards report no network/H/Torch work. Source hashes and fresh scratch checks pass.
- The actual synthetic CLI demo produces ranked products, score components and
  cited template explanations. It uses a memory graph, not a real Neo4j service.
- Two native Parquet fixture children exited **-11** under the same 512 MiB
  address-space envelope. The cause remains unverified. One narrow fixture-writer
  correction preserved all assertions but did not close the blocker. Both attempts,
  full streams and residual scratch are retained. No more native reruns occurred.
- Neo4j and HTTPS adapters pass contract/mocked tests only. No package installation,
  actual Neo4j connection, provider generation, real-data import or H allocation ran.
- Code review strengthened retrieval to apply LIMIT before aggregation. The final
  pure pass verifies that contract without rerunning the failed native suite.

[Runbook and limitations](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/docs/kg-graphrag-mvp.md).
[Full review](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/review.json).
[Canonical phase decisions](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/decisions.tsv).

Next requires approval for a bounded synthetic Parquet diagnostic, not an assumed
memory-limit fix. Proposed envelope: one 60-second child, 2 GiB virtual address space,
512 MiB monitored RSS, 8 MiB per stream, 32 MiB scratch, no network/real data/H work.
That higher address-space limit is **not yet authorized**. Neo4j host/version and
LLM provider/model/data-egress/usage budgets must also be agreed before live checks.
The existing profiling agent's work and every historical acceptance remain unchanged.

## Adversarial review v1

The user requested the interrogate review workflow. Two serial same-model self-review
passes were used because this harness exposes no model-selectable reviewer subagent.
Neither Claude nor Grok was launched. This is not independent cross-family consensus.

Verdict is **HOLD**. Original implementation decision remains **ESCALATE** and deployment
remains **NOT_AUTHORIZED**. No code fixes, test reruns, native diagnostics, installations,
service connections or provider calls occurred during this review.

Act on the file-backed adapter's cached-SHA/lazy-read integrity boundary, insecure
remote Neo4j transport defaults, and the existing incomplete native/live gates before
claiming end-to-end acceptance. The source-lock issue is a static mutation scenario,
not observed dataset tampering. No new normal-path KG-Only mathematical regression was
found; this is not proof of real-data quality or scalability. Consider repeated whole-
graph work, safe error diagnostics and the exact scope of namespace reconciliation.

[Adversarial report](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/interrogate-v1/report.md).
[Review verdict](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/interrogate-v1/verdict.json).
[Active review trail](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/interrogate-v1/decisions.tsv)
preserves the previous phase's complete trail as an immutable prefix. Earlier reviews,
criteria, source hashes, failed attempts and consumed budgets are unchanged.

## TDD correction scope

User request: “tdd on Act On”. A fresh `tdd-act-on-v1` evidence phase preserves
pre-correction source/tests. Correct the two code findings in the existing isolated
worktree. Use configuration-only TLS checks and one focused synthetic artifact-
mutation regression, not a retry of the crashed full native suite. The source check
uses existing DuckDB, closes its fixture-writer connection before opening the reader,
and disables only that reader's external file cache to make the mutation signal
deterministic. It never reads real datasets. Each child retains the existing
60-second/512-MiB-address-space/8-MiB-stream limits. If a focused native check crashes,
record it and stop; do not infer the intended red signal or raise limits.

Run failing checks before production edits. Record intended failures, then make
minimal source-boundary and transport-policy changes and run focused green checks
plus nearby pure-reference regressions. Existing assertions, scientific models and
all previous terminal reviews remain unchanged. The third Act On item is still an
unresolved mandatory native/live verification gate. This request does not authorize
new service connections, provider calls, package installation, higher resource
limits, research execution or main integration.

## TDD correction results

Act On code findings 1 and 2 have direct failing-before/passing-after evidence.

- The source regression actually read rating 3.0 after the independent lock approved
  bytes containing 5.0. It now passes. The reader hashes and consumes the same pinned
  read-only handles, not independently reopened mutable paths. It checks inode/size/
  modification metadata around reads, bounds hashing/manifest reads, and releases
  handles on close and rejected initialization. This input adapter now explicitly
  requires Linux `/proc/self/fd`; it is not an OS sandbox or hostile-filesystem proof.
- The remote transport regression initially accepted ten insecure remote URI cases.
  It now rejects them before driver creation. A further red test exposed that plain
  `neo4j://` routing must also be rejected on loopback because discovery can route
  remotely. All routing now requires TLS. Plain direct `bolt://` remains available
  on loopback. The positive control was narrowed accordingly, not weakened to match
  an unsafe implementation. Earlier test/source versions are preserved.
- Final nearby pure verification passes **108 tests**, zero failures/errors/skips:
  27 MVP, 72 unchanged reference, four transport and five file-pin lifecycle checks.
  The artifact-mutation regression and an existing synthetic Parquet-to-recommendation
  happy path also pass as two separate targeted native checks. No claim of a passing
  whole Parquet suite or real Neo4j server is made.
- Every check kept the existing 60-second/512-MiB-address-space/8-MiB-stream limits.
  No higher diagnostic allocation, package installation, network request, real-data
  execution, H update, main source integration, staging, commit or push occurred.

**Act On 3 remains unresolved.** Original -11 native-suite attempts remain preserved
and unexplained; actual Neo4j and provider checks still require their own approved
setup and budgets. Current overall decision remains **ESCALATE**, not full acceptance.
Deployment remains **NOT_AUTHORIZED**. Consider findings were not expanded into fixes.

[TDD report](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/tdd-act-on-v1/report.md).
[TDD review](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/tdd-act-on-v1/review.json).
[Active correction trail](../.worktrees/kg-graphrag-mvp-v1/Amazon4U/data/reports/kg-graphrag-mvp-v1/tdd-act-on-v1/decisions.tsv)
preserves the entire previous review trail as an immutable prefix. Earlier reviews
and criteria are not rewritten; these results describe a newly authorized correction.
