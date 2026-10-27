"""KG-v2 product facts: normalized brands with a `store` fallback, and `details` attributes.

Rules: docs/kg-v2-enrichment-plan.md (agreed 2026-10-27). Only products in the training
split are read, from the static product metadata. No rating, review, rank, price, date or
text field is used, and the held-out splits are never opened.

- **Brand.** `details.Brand` when it is a string with a usable key, else `store`. Both
  pass `brand_key`: Unicode NFKC, trim, collapsed whitespace, casefold. A key that is
  empty or in `BRAND_PLACEHOLDERS` is unusable, from either source. The list extends
  KG-v1's `brand_placeholders_excluded` (configs/kg-v1.json). The display name of a key
  is its most frequent original spelling, ties alphabetical.
- **Attributes.** The `ATTRIBUTE_FIELDS` keys of `details`, values trimmed, casefolded
  and split on commas. A `field=value` fact is kept only if at least `MIN_PRODUCTS`
  training products have it.

The builder writes `facts.parquet` (parent_asin, relation, value) and `manifest.json`
into a new directory and refuses to overwrite one.

Usage: python -m scripts.kg_v2_facts --meta '<raw_meta glob>' --train <train csv> --out <new dir>
"""

import argparse
import glob
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = 'amazon4u-kg-v2-facts-v1'
KG_V1_PLACEHOLDERS = frozenset({'none', 'null', 'nan', 'n/a', 'unknown'})
BRAND_PLACEHOLDERS = KG_V1_PLACEHOLDERS | {'generic', 'artist unknown'}
ATTRIBUTE_FIELDS = {
    'color': ('Color', 'Color Name'),
    'material': ('Material', 'Material Type', 'Body Material'),
    'strings': ('Number of Strings',),
    'power': ('Power Source',),
    'connector': ('Connector Type',),
}
MIN_PRODUCTS = 5


def brand_key(value):
    if not isinstance(value, str):
        return None
    key = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip().casefold()
    return key or None


def attribute_values(raw):
    if not isinstance(raw, str):
        return set()
    values = (re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', v)).strip().casefold() for v in raw.split(','))
    return {v for v in values if v}


def product_facts(details, store):
    """`(brand_spelling, brand_source, attributes)` for one product's raw metadata."""
    spelling, source = None, None
    for value, origin in ((details.get('Brand'), 'details.Brand'), (store, 'store')):
        if brand_key(value) and brand_key(value) not in BRAND_PLACEHOLDERS:
            spelling, source = value, origin
            break
    attributes = set()
    for field, keys in ATTRIBUTE_FIELDS.items():
        for key in keys:
            attributes |= {f'{field}={v}' for v in attribute_values(details.get(key))}
    return spelling, source, attributes


def build_facts(rows):
    """Facts and counts from `(parent_asin, details_json, store)` rows of training products."""
    spellings = defaultdict(Counter)
    brands, sources, attributes = {}, Counter(), {}
    for asin, raw, store in rows:
        try:
            details = json.loads(raw) if raw else {}
        except ValueError:
            details = {}
        if not isinstance(details, dict):
            details = {}
        spelling, source, attrs = product_facts(details, store)
        if spelling is not None:
            key = brand_key(spelling)
            brands[asin] = key
            spellings[key][spelling] += 1
            sources[source] += 1
        attributes[asin] = attrs
    df = Counter(a for attrs in attributes.values() for a in attrs)
    kept = {a for a, n in df.items() if n >= MIN_PRODUCTS}
    display = {k: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, c in spellings.items()}
    facts = sorted([(p, 'brand', display[k]) for p, k in brands.items()]
                   + [(p, 'attribute', a) for p, attrs in attributes.items() for a in attrs if a in kept])
    counts = {
        'products': len(attributes),
        'brand_sources': dict(sorted(sources.items())),
        'products_with_brand': len(brands),
        'distinct_brands': len(display),
        'brand_keys_with_several_spellings': sum(len(c) > 1 for c in spellings.values()),
        'attribute_values_seen': len(df),
        'attribute_values_kept': len(kept),
        'products_with_attributes': sum(bool(a & kept) for a in attributes.values()),
        'attribute_products_by_field': {
            f: sum(any(a.startswith(f'{f}=') for a in attrs & kept) for attrs in attributes.values())
            for f in ATTRIBUTE_FIELDS},
    }
    return facts, counts


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(meta_glob, train_csv, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f'{out_dir} exists; overwrite refused')
    meta_files = sorted(glob.glob(meta_glob))
    if not meta_files:
        raise SystemExit(f'no metadata files match {meta_glob}')
    con = duckdb.connect()
    con.execute("SET memory_limit='512MB'")
    rows = con.execute(
        'select parent_asin, details, store from read_parquet(?) where parent_asin in '
        '(select distinct parent_asin from read_csv_auto(?)) order by parent_asin',
        [meta_files, str(train_csv)]).fetchall()
    trained = con.execute('select count(distinct parent_asin) from read_csv_auto(?)', [str(train_csv)]).fetchone()[0]
    if len(rows) != trained or len({r[0] for r in rows}) != trained:
        raise SystemExit('every training product needs exactly one metadata row')
    facts, counts = build_facts(rows)
    out_dir.mkdir(parents=True)
    pq.write_table(pa.table({k: [f[i] for f in facts] for i, k in enumerate(('parent_asin', 'relation', 'value'))}),
                   out_dir / 'facts.parquet')
    manifest = {
        'schema_version': SCHEMA, 'rules': 'docs/kg-v2-enrichment-plan.md',
        'brand_placeholders': sorted(BRAND_PLACEHOLDERS), 'attribute_fields': ATTRIBUTE_FIELDS,
        'min_products_per_attribute_value': MIN_PRODUCTS,
        'sources': {'train_csv': {'name': Path(train_csv).name, 'sha256': _sha256(train_csv)},
                    'metadata': [{'name': Path(f).name, 'sha256': _sha256(f)} for f in meta_files]},
        'facts_sha256': _sha256(out_dir / 'facts.parquet'), 'counts': counts,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_facts(out_dir):
    """`(brands, attributes)` maps from a built directory, after checking its hash."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / 'manifest.json').read_text())
    if manifest.get('schema_version') != SCHEMA or _sha256(out_dir / 'facts.parquet') != manifest['facts_sha256']:
        raise ValueError(f'{out_dir} is not an intact KG-v2 facts build')
    brands, attributes = {}, defaultdict(list)
    for r in pq.read_table(out_dir / 'facts.parquet').to_pylist():
        if r['relation'] == 'brand':
            brands[r['parent_asin']] = r['value']
        else:
            attributes[r['parent_asin']].append(r['value'])
    return brands, dict(attributes), manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    print(json.dumps(build(args.meta, args.train, args.out)['counts'], indent=2))


if __name__ == '__main__':
    main()
