# KG metadata enrichment options

Status: **PROPOSED, not approved (2026-10-27).** Recorded at the user's request ("Please
note down my 3 question about extract facts from free text using LLM as a option to
increate metadata"). This note changes no frozen rule and authorizes no build, run, package
or provider call. KG-v1 (brand and category only) and protocol v2.1.0 stay as they are.

## Why

On Musical_Instruments validation, KG-only scored below Popularity (NDCG@10 0.0052 against
0.0157, all ratings). KG-v1 has two kinds of fact, and they are incomplete and noisy:
31% of training products have no brand, some brands are wrong, and 86 brands are split
only by letter case. A richer graph might help KG-only and KG + GNN. It might also not:
shoppers mostly buy popular and complementary products.

## Column audit (training products, Musical_Instruments)

Measured on 2026-10-27 over the 24,556 products in the training split. All of them have
a metadata row. The test split was not read.

| Column | Coverage | Notes |
|---|---|---|
| `store` | 99.6% | Equals the brand (ignoring case) for 98% of the 16,979 products that have both. Present for 99% of the 7,564 products without a brand. |
| `details` | per field | Color Name 58%, Color 50%, Material Type 31%, Size 29%, Material 20%, Power Source 16%, Number of Strings 15%, Connector Type 14%, Body Material 11%. Values need cleaning. |
| `title` | 100% | Free text, for example "DIMARZIO dp415 W Pickup Micro for Electric Guitar 6 Strings White". |
| `features` | 95% | Free text, median 5 bullet points. |
| `description` | 72% | Free text, median 246 characters. |

Not allowed: `average_rating`, `rating_number`, `details.Best Sellers Rank` and
`bought_together` use information from after the predicted purchase. `price` is a single
2023 snapshot. Images and videos are out of scope.

## Options

### Option 1. KG-v2 from structured columns (agent recommendation)

- Fill a missing brand from `store`, as a labeled rule. This changes the current rule
  "do not treat store as brand".
- Merge brand names that differ only in case or spacing.
- Add 4 to 6 cleaned `details` fields as new fact types, such as color, material, size,
  number of strings, power source and connector type.

### Option 2. Facts from free text with an LLM (the user's proposal)

Use an LLM to read the three free-text columns (`title`, `features`, `description`) and
return structured facts that become new KG nodes and relations. Examples: instrument type,
material, number of strings, pickup type, compatible instrument and power source.

**What it could add.** Most products have text even when `details` is empty, so it could
fill many gaps. Unlike text embeddings, the output stays as named facts, so the method
remains a knowledge graph and the facts can be shown as explanations.

**Questions to settle before any approval:**

1. **Correctness.** LLMs can invent facts. A fixed list of allowed fact types and values,
   a rule that the answer must be stated in the text, and a hand-checked sample with a
   measured error rate are needed before use.
2. **Reproducibility.** A pinned model version, fixed prompts, temperature 0, and every
   output stored once with its hash. Training reads the stored file and never calls the
   model again. The project rules require provenance for external frozen models.
3. **Leakage.** Use the seller text only, never review text. The text is a 2023 snapshot,
   which is the same static-metadata limitation as KG-v1.
4. **Cost and approvals.** About 24,600 Musical_Instruments training products, far more
   for Toys_and_Games and Electronics. It needs a provider, a key, network egress and a
   spending limit, each approved. Alternatively, a local model needs a new package
   approval and enough memory.
5. **Timing.** Like any graph change, it must be built and validated before the single
   test run, and reported as designed after validation results were seen.

### Option 3. Text embeddings (follow-up)

Turn the text into vectors with a pretrained text model, as node features. This changes the
frozen KG + GNN design, which has learned IDs and no node features. It would be a new
method, "KG + text", with its own plan.

## Status and next step

- Agent recommendation: Option 1 first. Keep Option 2 as the next candidate, to be decided
  after Option 1's validation result.
- No option is approved. Any option needs a written plan, a new protocol version, a leakage
  and quality audit, and GPU approval for retraining.
