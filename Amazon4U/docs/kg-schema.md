# KG-v1: structural training graph

**Implemented graph schema**, fixed in `configs/kg-v1.json` before any model
training. This freezes graph construction only; the evaluation protocol remains
a draft and training is not authorized. See [evaluation protocol](evaluation-protocol.md)
and [input audit](kg-input-audit.md).

## Scope and provenance

One independent graph per category: Electronics, Toys_and_Games,
Musical_Instruments. Build from complete published training CSVs, not a random
1,000-row sample. No validation/test file is read by the builder. Those files are
read by a separate integrity validator only.

Only users and products present in that category's training split become nodes.
Metadata-only catalog products are not included. Some held-out items are absent
from training; evaluation must explicitly decide how to handle cold-start items
before training. Do not silently remove such targets to improve results.

All training ratings (1–5) remain observations; there is no positive threshold,
negative sampling, rating prediction objective or ranking objective chosen yet.
A `rated` edge is **not** a claim that the user likes the item. Its rating and
timestamp are edge properties, not approved node features.

## Node files (Parquet)

| File | Node type | Columns | Identity |
|---|---|---|---|
| `users.parquet` | User | node_id, user_id | `u:` + user_id |
| `products.parquet` | Product | node_id, parent_asin | `p:` + parent_asin |
| `categories.parquet` | CategoryPath | node_id, label, path_json | `c:` + JSON array of path labels |
| `brands.parquet` | Brand | node_id, label | `b:` + explicit trimmed Brand |

IDs are identities, not ordinal numeric features. No learned embeddings or
numeric/text node features have been generated.

Category nodes identify **full observed path prefixes**, not just leaf names.
Thus `["Electronics", "Accessories"]` differs from
`["Musical Instruments", "Accessories"]`. Whitespace is trimmed; spelling/case
is otherwise preserved. JSON arrays avoid delimiter collisions.

Brand is extracted only from a nonempty JSON string at `details["Brand"]`.
Blank and case-insensitive `none`, `null`, `nan`, `n/a`, `unknown` values are
excluded. `store` and `Manufacturer` are not treated as brand. Brand identities
are exact trimmed strings, not verified canonical entities; aliases may remain.

## Edge files (Parquet)

| File / relation | Source → target | Extra columns |
|---|---|---|
| `rated.parquet` | User → Product | rating, timestamp |
| `belongs_to.parquet` | Product → CategoryPath | none |
| `category_child_of.parquet` | CategoryPath → preceding path prefix | none |
| `has_brand.parquet` | Product → Brand | none |

A product is linked to **every prefix** of its nonempty observed category path.
`category_child_of` records adjacent prefixes. This models the ordered category
paths found in the data, not an externally verified taxonomy. `main_category`
is not used because it often differs from the root of `categories`.

There are no reverse interaction edges in storage. Future message-passing
implementations may add inverse relations from training edges only; they must
never derive reverse edges from held-out interactions.

Missing categories/brands yield no corresponding edges. Missing metadata does
not remove a training product or invent attributes. Duplicate training pairs,
duplicate matching metadata IDs, invalid interactions or blank category labels
cause construction to fail instead of silently picking records.

## Approved and excluded fields

The structural metadata allowlist is **parent_asin, categories, details.Brand**.
The remaining details dictionary is discarded, including `Best Sellers Rank`.
Product `average_rating`, `rating_number`, `bought_together`, and all other
metadata fields are excluded from KG-v1 features/relations. Text and general
`has_attribute` edges are deferred until separately audited and registered.

Metadata is a snapshot with unknown historical availability. Even brand and
category assignments might differ from their values at interaction time.
KG-v1 is a static-metadata benchmark, not a claim of historical causal validity.

## Cross-category safeguards

User IDs can overlap across categories. Separate output graphs prevent graph
message passing from bringing a user's future interaction in another category
into the current experiment. Do not merge these graphs or share fitted user
representations without a new time-safe protocol. A merged training graph is
not automatically safe merely because all its rows are named `train`.

## Artifacts and checks

Outputs: `data/processed/kg-v1/<Category>/` and a shared `manifest.json` with
config, input SHA-256 hashes, counts and validation results. Files are sorted
before export. Atomic publication prevents a failed build from being mistaken
for a complete graph. Existing outputs are not overwritten.

Checks performed:

- Unique IDs and unique relation pairs; valid typed edge endpoints.
- Persisted rated edges match training rows exactly, including rating/time.
- Persisted node tables match the strict column allowlist.
- Independent split validation: no held-out user–product pairs in rated edges;
  source hashes still match the build manifest.
- Small-fixture regression tests: held-out changes and excluded metadata changes
  cannot change graph tables; duplicates fail; missing attributes remain absent.

These checks establish graph integrity, not a complete model leakage audit.
Embeddings, objectives, candidate sets and evaluation still need their own checks.
