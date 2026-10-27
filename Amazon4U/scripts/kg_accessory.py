"""Accessory links for KG-only attempt C (docs/kg-v2-enrichment-plan.md).

Roles follow `relation_lift.instrument_roles` (the audited rule) with the two fixes fixed
before attempt C's result:

1. a product that **is** an instrument family serves no family;
2. a "for ..." phrase runs to `;`, `(`, `)`, `|` or 80 characters, so a comma list
   such as "for all instruments, ukulele, guitar, bass" counts every family in it.

`accessory_tokens` gives each product two token sets for the KG-only accessory channel.
History side: `own:F` for each family it is, `serve:F` for each it serves. Candidate side:
the same facts flipped, so owning a guitar matches guitar accessories and owning guitar
accessories matches guitars.
"""

import json
import re

import duckdb

from scripts.relation_lift import INSTRUMENTS

_FOR = re.compile(r'\bfor\b([^;()|]{0,80})')


def roles(title, categories, compatible):
    """`({families it is}, {families it serves})` for one product."""
    labels = set(categories or ())
    is_ = {f for f, (cats, _) in INSTRUMENTS.items() if labels & set(cats)}
    if is_:
        return is_, set()
    text = ' '.join(m.group(1) for m in _FOR.finditer((title or '').lower()))
    text += ' ' + (compatible or '').lower()
    serves = {f for f, (_, words) in INSTRUMENTS.items()
              if any(re.search(rf'\b{re.escape(w)}s?\b', text) for w in words)}
    return set(), serves


def accessory_tokens(product_roles):
    """`(history_tokens, candidate_tokens)` maps from `{product: (is, serves)}`."""
    history, candidate = {}, {}
    for p, (is_, serves) in product_roles.items():
        h = sorted({f'own:{f}' for f in is_} | {f'serve:{f}' for f in serves})
        c = sorted({f'serve:{f}' for f in is_} | {f'own:{f}' for f in serves})
        if h:
            history[p], candidate[p] = h, c
    return history, candidate


def load_roles(meta_files, products):
    """Roles for `products` from their raw metadata title, categories and compatibility."""
    con = duckdb.connect()
    con.execute("SET memory_limit='512MB'")
    out = {}
    for asin, title, cats, raw in con.execute(
            'select parent_asin, title, categories, details from read_parquet(?) where parent_asin in (select unnest(?))',
            [list(meta_files), sorted(products)]).fetchall():
        try:
            compatible = json.loads(raw).get('Compatible Devices') if raw else None
        except (ValueError, AttributeError):
            compatible = None
        out[asin] = roles(title, cats, compatible if isinstance(compatible, str) else None)
    return out
