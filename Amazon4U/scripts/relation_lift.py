"""Lift check for candidate KG relations on Musical_Instruments training purchases.

A relation gives each product a set of values (its brand, its category leaf, its
`color=` facts, the instruments it is or serves). For a user's n-th training purchase
(n > 1, time order, ties by parent_asin), the event is **linked** when the purchase
shares a value with the union over the user's earlier purchases. Lift is the linked rate
divided by the rate for `DRAWS` random products per event, drawn in proportion to their
training purchase counts. Lift near 1 means the relation does not predict the next
purchase. Training split only; nothing is built into a graph.

`INSTRUMENTS` is the candidate "serves instrument" relation: a product **is** an
instrument family when its category path holds one of the family's labels, and **serves**
it when its title says "for ... <word>" or its `Compatible Devices` names a word.

Usage: python -m scripts.relation_lift --data <dataset dir> --manifest-sha256 <sha> \\
    --meta '<raw_meta glob>' --facts <kg-v2 facts dir> --out <new file>
"""

import argparse
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import duckdb

from scripts.kg_v2_facts import load_facts
from scripts.profiling_data import ProfilingData

SCHEMA = 'amazon4u-relation-lift-v1'
EVENTS, DRAWS, SEED = 50_000, 20, 20261027

INSTRUMENTS = {
    'electric guitar': (('Electric Guitars',), ('electric guitar', 'strat', 'stratocaster', 'telecaster', 'les paul',
                                                 'floyd rose')),
    'acoustic guitar': (('Acoustic Guitars', 'Classical & Nylon-String Guitars'),
                        ('acoustic guitar', 'acoustic', 'classical guitar')),
    'guitar': (('Guitars',), ('guitar',)),
    'bass guitar': (('Bass Guitars',), ('bass guitar', 'electric bass', 'bass')),
    'ukulele': (('Ukuleles',), ('ukulele', 'uke')),
    'banjo': (('Banjos',), ('banjo',)),
    'mandolin': (('Mandolins',), ('mandolin',)),
    'orchestral strings': (('Violins', 'Violas', 'Cellos', 'Orchestral Strings'), ('violin', 'viola', 'cello')),
    'drums': (('Drums & Percussion', 'Drum Sets & Set Components'), ('drum', 'drums', 'drum set', 'snare', 'cymbal')),
    'keyboard': (('Keyboards & MIDI', 'Pianos', 'Electronic Keyboards'), ('keyboard', 'piano')),
    'microphone': (('Microphones',), ('microphone', 'mic', 'blue yeti')),
    'amplifier': (('Guitar & Bass Amplifiers & Preamps',), ('amplifier', 'amp')),
}
_FOR = re.compile(r'\bfor\b([^,;()|]{0,60})')


def instrument_roles(title, categories, compatible):
    """`({families it is}, {families it serves})` for one product."""
    labels = set(categories or ())
    is_ = {f for f, (cats, _) in INSTRUMENTS.items() if labels & set(cats)}
    text = ' '.join(m.group(1) for m in _FOR.finditer((title or '').lower()))
    text += ' ' + (compatible or '').lower()
    serves = {f for f, (_, words) in INSTRUMENTS.items()
              if any(re.search(rf'\b{re.escape(w)}s?\b', text) for w in words)}
    return is_, serves


def lift(events, values, weights, next_values=None, *, draws=DRAWS, seed=SEED):
    """Linked rate, random rate and lift over `(history, next)` events for one relation.

    `next_values` views the next product differently from the history, for a directional
    relation; by default both use `values`.
    """
    next_values = values if next_values is None else next_values
    rng = random.Random(seed)
    population = list(weights)
    cum = list(weights.values())
    linked = chance = eligible = 0
    for history, nxt in events:
        held = set().union(*(values.get(p, ()) for p in history))
        if not held:
            continue
        eligible += 1
        linked += bool(held & next_values.get(nxt, set()))
        chance += sum(bool(held & next_values.get(p, set())) for p in rng.choices(population, cum, k=draws)) / draws
    if not eligible:
        return {'events_with_history_value': 0}
    return {'events_with_history_value': eligible, 'linked_rate': linked / eligible,
            'random_rate': chance / eligible, 'lift': (linked / chance) if chance else None}


def run(data_dir, manifest_sha256, meta_glob, facts_dir, *, n_events=EVENTS):
    data = ProfilingData(data_dir, manifest_sha256)
    brands_v2, attributes, _ = load_facts(facts_dir)
    products = sorted({p for _, p, *_ in data.train})
    con = duckdb.connect()
    con.execute("SET memory_limit='512MB'")
    meta = {a: (t, c, d) for a, t, c, d in con.execute(
        'select parent_asin, title, categories, details from read_parquet(?) where parent_asin in (select unnest(?))',
        [sorted(glob.glob(meta_glob)), products]).fetchall()}

    def leaf(p):
        cats = data.product_categories.get(p)
        return {max(cats, key=len)} if cats else set()

    def depth(p, k):
        cats = [c for c in data.product_categories.get(p, ()) if len(json.loads(c[2:])) == k]
        return set(cats)

    relations = {
        'category leaf': {p: leaf(p) for p in products},
        'category level 3': {p: depth(p, 3) for p in products},
        'category level 2': {p: depth(p, 2) for p in products},
        'brand (KG-v1)': {p: {b} for p, b in data.product_brands.items()},
        'brand (KG-v2)': {p: {b} for p, b in brands_v2.items()},
    }
    for field in ('color', 'material', 'strings', 'power', 'connector'):
        relations[f'attribute {field}'] = {p: {a for a in attrs if a.startswith(f'{field}=')}
                                           for p, attrs in attributes.items()}
    roles = {}
    for p in products:
        title, cats, raw = meta.get(p, (None, None, None))
        try:
            compatible = json.loads(raw).get('Compatible Devices') if raw else None
        except (ValueError, AttributeError):
            compatible = None
        roles[p] = instrument_roles(title, cats, compatible if isinstance(compatible, str) else None)
    relations['instrument: serves or is the same family'] = {p: i | s for p, (i, s) in roles.items()}
    relations['instrument: serves a family the user owns'] = None

    by_user = defaultdict(list)
    for u, p, _, t in data.train:
        by_user[u].append((t, p))
    events = []
    for u in sorted(by_user):
        seq = [p for _, p in sorted(by_user[u])]
        events += [(seq[:i], seq[i]) for i in range(1, len(seq))]
    events = random.Random(SEED).sample(events, min(n_events, len(events)))
    weights = defaultdict(int)
    for _, p, *_ in data.train:
        weights[p] += 1

    result = {}
    for name, values in relations.items():
        if values is None:
            owned = {p: {('own', f) for f in roles[p][0]} | {('serve', f) for f in roles[p][1]} for p in products}
            flipped = {p: {('serve', f) for f in roles[p][0]} | {('own', f) for f in roles[p][1]} for p in products}
            stats = lift(events, owned, weights, flipped)
        else:
            stats = lift(events, values, weights)
        stats["products_with_value"] = sum(bool(values.get(p)) for p in products) / len(products) if values else \
            sum(bool(roles[p][1]) for p in products) / len(products)
        result[name] = stats
    return {'schema_version': SCHEMA, 'category': data.manifest['category'], 'split': 'train',
            'events': len(events), 'draws_per_event': DRAWS, 'seed': SEED, 'instrument_vocabulary': INSTRUMENTS,
            'relations': result}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for name in ('--data', '--manifest-sha256', '--meta', '--facts', '--out'):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; overwrite refused')
    result = run(args.data, args.manifest_sha256, args.meta, args.facts)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    for name, s in result['relations'].items():
        print(f"{name:45s} products {s['products_with_value']:.0%}  linked {s.get('linked_rate', 0):.3f}  "
              f"random {s.get('random_rate', 0):.3f}  lift {s.get('lift') or 0:.1f}")


if __name__ == '__main__':
    main()
