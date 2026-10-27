# Plan: KG-v2 enrichment from `store` and `details`, evaluated with KG-only

Status: **AGREED by the user (2026-10-27).** All three rule changes approved: "Ok please
use store as fallback. you can change the rule." / "Agree, using some details field not
just Brand." / "agree too" (the attribute channel and its weight options). This approval
covers CPU validation work only. It authorizes no GPU run and no test run.

The user asked to enrich the
data with `store` and `details` and to test whether it raises the recommendation score,
working with KG-only and Popularity first. KG + GNN and LightGCN are not retrained until
this plan's result is known. Options and the column audit:
[`kg-metadata-enrichment-options.md`](kg-metadata-enrichment-options.md) (Option 1).

## Goal and the number

- **Question.** Does a cleaner, richer graph raise KG-only's validation NDCG@10?
- **Yardsticks.** Core KG-only (KG-v1): 0.0052 (all ratings), 0.0057 (ratings ≥ 4).
  Popularity: 0.0157 and 0.0169. Popularity does not use metadata, so it does not change.
- **Data.** Musical_Instruments, validation split only. The test split is not read.
- **Not a hillclimb on validation.** The attempts below are fixed now, before any KG-v2
  number exists. Each is measured once and reported, win or lose. Further attempts need a
  new approval. Repeated tuning on validation would overfit it.

## The data shape

One fact table per product, built from training products only:

| Field | KG-v1 | KG-v2 |
|---|---|---|
| `categories` | observed path prefixes | unchanged |
| `brand` | `details.Brand`, exact string | normalized brand key, filled from `store` when `details.Brand` is missing |
| `attributes` | none | set of `field=value` facts from selected `details` fields |

KG-only gains one optional channel, `attribute`, computed exactly like the category
channel: IDF `1 + ln((|W|+1)/(df+1))` over condition-warm products, L2-normalized product
vectors, user vectors as normalized sums over positive history, cosine, zero for a missing
channel.

## Normalization rules (fixed now)

**Brand key.** Unicode NFKC, trim, collapse internal whitespace, casefold. The displayed
name is the most frequent original spelling among training products, ties broken
alphabetically. This merges the 86 case or spacing splits found in the audit, such as
"sela" and "Sela".

**Store fill.** If `details.Brand` is missing, use `store` under the same key rule. Skip a
store that is empty or whose key is one of `unknown`, `generic`, `n/a`, `none`. The audit
step lists every store value containing "visit", "store", "unknown" or "generic" (70
training products) before the build, so any further exclusion is recorded, not guessed.

**Attributes.** Fields and merges:

| Fact field | Source keys in `details` |
|---|---|
| `color` | `Color`, `Color Name` |
| `material` | `Material`, `Material Type`, `Body Material` |
| `strings` | `Number of Strings` |
| `power` | `Power Source` |
| `connector` | `Connector Type` |

Values are trimmed, casefolded and split on commas, so "Adapter,Electric" gives two facts.
A value is kept only if at least 5 training products have it, to drop typos and one-offs.
`Size` is excluded: its values are mostly unusable ("-inch", "0"). Excluded as before:
ratings, rating counts, Best Sellers Rank, `bought_together`, price, dates, text fields.

## Attempts (fixed now)

| Attempt | Graph | KG-only grid (category / brand / attribute) |
|---|---|---|
| Baseline | KG-v1 | core: 1/0/0, 0.75/0.25/0, 0.5/0.5/0 (results exist) |
| **A. Brand fix** | KG-v1 categories, KG-v2 brand | the same core 3 mixtures, formula unchanged |
| **B. Brand fix + attributes** | full KG-v2 | 0.5/0.5/0, 0.4/0.4/0.2, one third each |

Selection inside each attempt: best validation NDCG@10, exact ties keep the earliest grid
entry, separately per feedback condition. Popularity is rerun once as a control and must
reproduce its earlier numbers exactly.

## How a gain is judged

- For each attempt against the KG-v1 baseline: the change in NDCG@10 with a **paired
  bootstrap interval** over validation users (1,000 resamples, seed 12345, 95%
  percentile). A gain counts only if the interval is above zero in both conditions.
- Also reported: HitRate@10, CatalogCoverage@10, fallback counts, and brand and attribute
  coverage before and after.
- The five-shopper page is rebuilt with the same seed (20261027), to check that the
  picks changed in a sensible way.

## Correctness checks

- Hand-calculated tests for the attribute channel, written from the formula before the
  code, in the style of `docs/kg-only-scoring.md`.
- With the attribute weight at 0 and KG-v1 brands, the new code reproduces core KG-only
  exactly.
- The normalization rules have unit tests, including the 86 merges and the store
  exclusions.
- The 1-in-200 spot check against the per-user reference, as in the earlier runs.
- The full test suite passes.

## Where the work goes

- The existing `training-mi` worktree. A new worktree would branch from committed code and
  miss the uncommitted fast KG-only scoring this work builds on; nothing is committed yet.
  Merged into master on 2026-10-28 (commit `8844951`); the worktree is archived.
- `scripts/kg_v2_facts.py` builds the fact table into a new directory under
  `data/processed/` and never overwrites it.
- `scripts/kg_only_fast.py` gains the optional attribute channel.
- A validation runner writes new files under `data/reports/kg-v2-enrichment-v1/`.
- Results are recorded in this document and a companion JSON.

## Rule changes this needs (approval required)

1. KG-v2 uses `store` as a brand fallback. The current rule says "do not treat `store` as
   brand". The audit supports it: the two agree 98% of the time when both exist.
2. KG-v2 uses selected `details` fields. The current allowlist is `categories` and
   `details.Brand` only.
3. KG-only gains an attribute channel and the attempt B grid.

All three apply to a labeled variant, "KG-only on KG-v2", designed after validation
results were seen. Core KG-only and protocol v2.1.0 stay unchanged and reported. The
protocol snapshot is frozen with the research questions before the test run.

## Cost and what comes next

- CPU only. My estimate: about 1 day for A and 1 more day for B, including tests. Each
  validation run takes about 20 minutes and 2 GB of memory.
- **If A or B gives a clear gain:** propose retraining KG + GNN on KG-v2, plus the
  "GNN without KG" control, as a separate plan with GPU approval.
- **If neither does:** stop, report it, and keep the current results. The LLM text
  option stays available as the next candidate.

## Results (validation, 2026-10-27)

Run: `scripts/kg_v2_validation.py` on KG-v2 facts `data/processed/kg-v2-facts-mi-v2`.
Output: `data/reports/kg-v2-enrichment-v1/validation.json`. 35 minutes on CPU, peak
memory 4.1 GB. The test split was not loaded. Popularity and core KG-only reproduced
their recorded v2.1.0 numbers exactly. Every mixture passed the 1-in-200 per-user spot
check against a reference scorer.

**Facts built.** Brand on 24,430 of 24,556 training products (99.5%, was 69%): 16,981
from `details.Brand`, 7,449 from `store`. 116 brand keys merged several spellings.
414 attribute values kept of 4,781 seen. 19,016 products (77%) have at least one
attribute.

**Correction during the build.** The first build (`kg-v2-facts-mi-v1`) skipped placeholder
names such as "Unknown" only for `store`, so 11 `details.Brand` values of "Unknown"
became a brand. KG-v1 excludes them. The rule now applies KG-v1's placeholder list, plus
"generic" and "artist unknown", to both sources, and a test checks that it contains every
KG-v1 placeholder. The first build is kept, marked `SUPERSEDED.txt`.

Validation NDCG@10:

| Condition | Popularity | KG-only KG-v1 | A. brand fix | B. brand fix + attributes |
|---|---|---|---|---|
| All ratings | 0.0157 | 0.0052 | 0.0053 (0.5/0.5) | **0.0064** (0.4/0.4/0.2) |
| Ratings ≥ 4 | 0.0169 | 0.0057 | 0.0056 (0.5/0.5) | **0.0070** (0.4/0.4/0.2) |

Gain over KG-only KG-v1 (paired bootstrap, 1,000 resamples, seed 12345, 95% interval):

| Attempt | All ratings | Ratings ≥ 4 | Counts as a gain |
|---|---|---|---|
| A | +0.00005 [-0.00030, +0.00042] | -0.00006 [-0.00045, +0.00032] | No |
| B | +0.00120 [+0.00079, +0.00169] | +0.00129 [+0.00076, +0.00181] | **Yes**, above zero in both |

B's gain is about 23% over KG-only KG-v1. B remains about 2.4 times below Popularity.
B also raises CatalogCoverage@10 from 0.57 to 0.79 (all ratings). Other B mixtures:
0.5/0.5/0 equals A; one third each gives 0.0060 and 0.0063. Brand coverage alone (A)
changes almost nothing; the gain comes from the attribute facts.

Per this plan, no further attempts are made without a new approval. The next step is a
separate plan for retraining KG + GNN on KG-v2, with the "GNN without KG" control and
GPU approval.

## Attempt C: accessory links (last enrichment attempt)

Status: **AGREED by the user (2026-10-28)**: "I want to try with "accessory for" links in
KG-only. this will the last try we will improve." The design below was fixed by the agent
before any result for it existed. It follows the relation audit
([`kg-v3-relations-audit.md`](kg-v3-relations-audit.md)), which measured a lift of 1.5 for
the rule-based link. No further enrichment attempt follows, whatever the result.

**Instrument roles.** From `scripts/relation_lift.py` `instrument_roles`, with two fixes
from the audit's hand check, decided before this attempt's result:

1. A product that **is** an instrument family serves no family.
2. A "for ..." phrase runs to the end of the clause, so a comma list such as "for all
   instruments, ukulele, guitar, bass" counts every family in it. The phrase stops at `;`,
   `(`, `)`, `|` or 80 characters.

**Channel.** A fourth KG-only channel, `accessory`. History side: a product contributes
tokens `own:F` for each family F it is and `serve:F` for each family it serves. Candidate
side: the flipped tokens, `serve:F` for families it is and `own:F` for families it serves.
So owning a guitar matches guitar accessories, and owning guitar accessories matches
guitars. Each side uses the frozen KG-only construction: IDF over condition-warm products,
L2-normalized product vectors, user vectors as normalized sums over positive history,
cosine, zero when missing.

**Grid** (category / brand / attribute / accessory): 0.4/0.4/0.2/0 (attempt B's selection,
the comparison), 0.35/0.35/0.15/0.15, 0.3/0.3/0.15/0.25. Selection by validation NDCG@10,
ties earliest.

**Gain rule.** Paired bootstrap against attempt B's selected mixture (1,000 resamples, seed
12345, 95% interval), above zero in both conditions. Also reported against KG-v1.

### Attempt C results (validation, 2026-10-28)

Run: `scripts/kg_c_validation.py`. Output:
`data/reports/kg-v2-enrichment-v1/validation-C.json`. 30 minutes on CPU, peak memory
3.7 GB. Popularity, KG-only on KG-v1 and attempt B reproduced their recorded NDCG@10
exactly. Every mixture passed the 1-in-200 spot check. The test split was not loaded.
Of 24,556 training products, 4,524 are an instrument and 5,013 serve one.

| Condition | Popularity | KG-v1 | B | C 0.4/0.4/0.2/0 | **C 0.35/0.35/0.15/0.15** | C 0.3/0.3/0.15/0.25 |
|---|---|---|---|---|---|---|
| All ratings | 0.0157 | 0.0052 | 0.0064 | 0.0064 | **0.0067** | 0.0062 |
| Ratings ≥ 4 | 0.0169 | 0.0057 | 0.0070 | 0.0070 | **0.0073** | 0.0068 |

Gain of C's selection (paired bootstrap, 1,000 resamples, seed 12345, 95% interval):

| Against | All ratings | Ratings ≥ 4 |
|---|---|---|
| B | +0.00028 [+0.00012, +0.00044] | +0.00031 [+0.00013, +0.00050] |
| KG-v1 | +0.00148 [+0.00106, +0.00196] | +0.00161 [+0.00106, +0.00214] |

C counts as a gain over B in both conditions, about +4.5%. Against KG-v1, the enriched
KG-only gains about 28%. It remains about 2.3 times below Popularity. A heavier accessory
weight (0.25) is worse than B. CatalogCoverage@10 rises to 0.80 (all ratings).

This was the last enrichment attempt. The enriched KG-only for later comparison is
attempt C at 0.35/0.35/0.15/0.15, labeled "designed after validation results".
