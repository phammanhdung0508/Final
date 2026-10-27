# KG relation lift audit (step 1 toward relationship links)

Status: **measurement only (2026-10-27).** The user agreed to the order "rules and category
links first, measure lift, then an LLM for what rules miss". This step builds nothing into
a graph and changes no rule. Building new relations needs a plan and approval.

Tool: `scripts/relation_lift.py` (in master since 2026-10-28, commit `8844951`), tests `tests/test_relation_lift.py`.
Output: `data/reports/kg-v3-relations-audit-v1/lift.json`. Training split only.

## Method

For 50,000 sampled training purchases that are not a user's first (seed 20261027), an
event is **linked** when the purchase shares a relation value with the user's earlier
purchases. The **random** rate uses 20 products per event, drawn in proportion to training
purchase counts. **Lift** = linked rate / random rate. A lift near 1 means the relation
does not predict the next purchase.

"Instrument" relations use a simple rule: a product **is** a family when its category path
has the family's label, and **serves** it when its title says "for ... <word>" or its
`Compatible Devices` names a word. The vocabulary is `INSTRUMENTS` in the script.

## Results

| Relation | Products with it | Linked | Random | Lift |
|---|---|---|---|---|
| Brand (KG-v1) | 69% | 0.103 | 0.027 | **3.8** |
| Brand (KG-v2, with store) | 99% | 0.157 | 0.043 | **3.6** |
| Category leaf | 97% | 0.201 | 0.063 | **3.2** |
| Category level 3 | 96% | 0.569 | 0.329 | 1.7 |
| Category level 2 | 97% | 0.710 | 0.530 | 1.3 |
| Attribute: connector | 13% | 0.070 | 0.030 | 2.4 |
| Attribute: strings | 15% | 0.149 | 0.095 | 1.6 |
| Attribute: material | 38% | 0.089 | 0.065 | 1.4 |
| Attribute: power | 16% | 0.122 | 0.085 | 1.4 |
| Attribute: color | 53% | 0.274 | 0.254 | 1.1 |
| Instrument: serves or is the same family | 38% | 0.308 | 0.205 | 1.5 |
| Instrument: serves a family the user owns | 23% | 0.124 | 0.084 | 1.5 |

## Reading

- Brand and the exact category leaf are the strongest links. Broad category levels are
  weak, which fits core KG-only's poor category-only score.
- Color barely predicts anything (lift 1.1). Connector is the strongest attribute.
- The rule-based "serves an owned instrument" link is modest (lift 1.5), weaker than brand.
- A hand check of 15 random "serves" products found most correct. Errors: an acoustic
  guitar marked as serving acoustic guitars ("guitar for beginner ... acoustic guitar"),
  a tuner "for all instruments, ukulele, guitar, bass..." recorded as guitar only because
  commas end the phrase, and a microphone marked as serving amplifiers through
  `Compatible Devices`. The rule is rough. Better extraction (an LLM) could raise its
  precision and coverage, but the measured lift gives no sign that this link is strong.
- Lift measures whether a link appears; it does not measure ranking among equally linked
  products. That may be why attributes with low lift still raised KG-only's NDCG@10
  (inferred, not tested).
