# Full-data audit before graph construction

Machine-readable results: `data/reports/kg-input-audit.json`.
Separate graph/split check: `data/reports/kg-split-validation.json`.
Reproduce using `scripts/audit_kg_inputs.py` and `scripts/validate_kg_splits.py`.
No models or embedding encoders were fitted during this audit.

## Published split integrity

All three categories have:

- No invalid interaction records, within-split duplicate user–item pairs, or
  user–item pair overlap between train/valid/test.
- Exactly one validation and one test interaction for every user, with remaining
  interactions in training; no per-user time inversions. This is consistent
  with leave-last-two-out (last interaction for test, preceding for validation).
- Full-collection minimum user/item degrees of 5. Training minimum user degree
  is 3 and training minimum item degree is 1: training alone is **not 5-core**.
- Complete product-ID matching against downloaded metadata.

Timestamp equality at split boundaries occurs for **44 Electronics users,
17 Toys_and_Games users and 6 Musical_Instruments users**. Published assignments
are preserved, but these are not strictly ordered timestamps. The original
publisher's tie-breaking rule is not established by this audit.

Held-out interactions whose item is absent from the training catalog:

| Category | Validation | Test |
|---|---:|---:|
| Electronics | 2,344 | 5,101 |
| Toys_and_Games | 720 | 1,441 |
| Musical_Instruments | 62 | 106 |

KG-v1 includes training products only. Freeze cold-start handling and candidate
coverage before evaluating a model; do not interpret dropped targets as success.

## Metadata quality

No missing or duplicated product IDs, conflicting duplicate structural records,
blank category path entries, malformed details JSON or non-object details were
found in these downloaded files.

| Category | Full metadata rows | Empty category lists | Empty descriptions | Explicit scalar Brand rows |
|---|---:|---:|---:|---:|
| Electronics | 1,610,012 | 128,442 | 682,748 | 1,153,897 |
| Toys_and_Games | 890,874 | 88,952 | 305,664 | 346,569 |
| Musical_Instruments | 213,593 | 18,974 | 60,859 | 124,082 |

Nonempty strings/lists are not guarantees of semantic quality. Explicit brand
coverage above counts nonempty strings before removing placeholder labels.
`details` also contains `Best Sellers Rank`; the complete dictionary must not be
passed blindly as node features. `store` is not a verified brand field.
`main_category` frequently differs from the first `categories` label, so those
fields must not be assumed to describe one consistent taxonomy.

Price has no SQL null/blank rows but frequently contains missing-value strings:
1,083,247 Electronics, 462,209 Toys_and_Games and 128,677 Musical_Instruments
rows contain placeholder values. Additional prices cannot be parsed as a single
number. This demonstrates why simple null counts are insufficient. Price is
excluded from KG-v1.

## Cross-category overlap and timing

| Category pair | Shared users | First-category training not before other target | Reverse direction |
|---|---:|---:|---:|
| Electronics / Toys_and_Games | 118,911 | 51,744 | 35,261 |
| Electronics / Musical_Instruments | 32,300 | 17,977 | 6,881 |
| Toys_and_Games / Musical_Instruments | 5,129 | 2,465 | 1,599 |

A timing-risk count means the user's maximum training timestamp in one category
is greater than or equal to the earliest held-out timestamp in the other. These
are user counts, not leaked event counts. This is strong evidence against simply
combining category-specific training graphs. No shared metadata product IDs were
found between these category collections.

## Constructed KG-v1

| Category | Users | Products | Category paths | Brands | Rated edges |
|---|---:|---:|---:|---:|---:|
| Electronics | 1,641,026 | 367,052 | 1,245 | 43,617 | 12,191,484 |
| Toys_and_Games | 432,264 | 161,656 | 908 | 14,955 | 2,997,358 |
| Musical_Instruments | 57,439 | 24,556 | 633 | 3,316 | 396,958 |

Additional relations connect products to category prefixes and explicit brands,
plus category-prefix hierarchy edges. See [schema](kg-schema.md) for semantics.
The three persisted graphs occupy approximately **284 MiB**.

All structural checks passed, and independent validation found **zero validation
or test user–product pairs in the training graphs**. No text/numeric node features
or embeddings were created. Evaluation remains blocked pending protocol freeze.
