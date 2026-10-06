#!/usr/bin/env python3
"""Render a bounded, real training-KG neighborhood as HTML, SVG and PNG.

Requires DuckDB and Graphviz's `dot` executable. This is an illustrative,
non-random graph sample, not a training/evaluation subset. No held-out data read.
"""
import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import duckdb

from build_kg import literal

COLORS = {'User': '#c4b5fd', 'Product': '#93c5fd', 'CategoryPath': '#86efac', 'Brand': '#fdba74'}


def quoted(value):
    return json.dumps(str(value), ensure_ascii=False)


def render(graph, category, output, max_products=6, user_id=None):
    if not shutil.which('dot'):
        raise RuntimeError('Install Graphviz (dot) to render visualizations')
    if not 1 <= max_products <= 12:
        raise ValueError('max_products must be between 1 and 12')
    manifest = json.loads((graph / 'manifest.json').read_text())
    if category not in manifest['categories']:
        raise ValueError(f'Unknown category: {category}')
    con = duckdb.connect()
    con.execute("SET memory_limit='512MB'; SET threads=1; SET preserve_insertion_order=false")
    for name in ['rated', 'belongs_to', 'category_child_of', 'has_brand', 'categories', 'brands']:
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({literal(graph / category / (name + ".parquet"))})')
    if user_id is None:
        # Deterministic illustrative user selection, bounded degree for readability.
        candidate = con.execute("""SELECT source FROM rated GROUP BY source
            HAVING count(*) BETWEEN 6 AND 20 ORDER BY hash(source),source LIMIT 1""").fetchone()
        if not candidate:
            candidate = con.execute('SELECT source FROM rated ORDER BY source LIMIT 1').fetchone()
        if not candidate:
            raise ValueError('Graph has no interactions')
        user_node = candidate[0]
    else:
        user_node = 'u:' + user_id
    events = con.execute('SELECT source,target,rating,timestamp FROM rated WHERE source=? ORDER BY timestamp,target LIMIT ?',
                         [user_node, max_products]).fetchall()
    if not events:
        raise ValueError(f'No training interactions for {user_node}')
    products = [event[1] for event in events]
    product_list = ','.join(literal(p) for p in products)
    memberships = con.execute(f'SELECT source,target FROM belongs_to WHERE source IN ({product_list}) ORDER BY source,target').fetchall()
    brand_edges = con.execute(f'SELECT source,target FROM has_brand WHERE source IN ({product_list}) ORDER BY source,target').fetchall()
    category_ids = sorted({dst for _, dst in memberships})
    brand_ids = sorted({dst for _, dst in brand_edges})
    nodes = {user_node: {'type': 'User', 'label': 'User\n' + user_node[2:10] + '…'}}
    nodes.update({p: {'type': 'Product', 'label': 'Product\n' + p[2:]} for p in products})
    hierarchy = []
    if category_ids:
        ids = ','.join(literal(c) for c in category_ids)
        for node, label, path in con.execute(f'SELECT node_id,label,path_json FROM categories WHERE node_id IN ({ids})').fetchall():
            nodes[node] = {'type': 'CategoryPath', 'label': label, 'path': json.loads(path)}
        hierarchy = con.execute(f'SELECT source,target FROM category_child_of WHERE source IN ({ids}) AND target IN ({ids}) ORDER BY source,target').fetchall()
    if brand_ids:
        ids = ','.join(literal(b) for b in brand_ids)
        for node, label in con.execute(f'SELECT node_id,label FROM brands WHERE node_id IN ({ids})').fetchall():
            nodes[node] = {'type': 'Brand', 'label': label}
    con.close()
    edges = [{'source': s, 'target': t, 'relation': 'rated', 'rating': r, 'timestamp': ts} for s, t, r, ts in events]
    for relation, pairs in [('belongs_to', memberships), ('category_child_of', hierarchy), ('has_brand', brand_edges)]:
        edges.extend({'source': s, 'target': t, 'relation': relation} for s, t in pairs)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / category
    sample = {'category': category, 'scope': 'illustrative training-only neighborhood; not a random/evaluation sample',
              'selection': 'bounded-degree user ordered by DuckDB hash; earliest training products, capped' if user_id is None else 'specified user; earliest training products, capped',
              'full_graph_counts': manifest['categories'][category]['counts'],
              'nodes': [{'id': key, **value} for key, value in sorted(nodes.items())], 'edges': edges}
    stem.with_suffix('.json').write_text(json.dumps(sample, indent=2) + '\n')
    dot = ['digraph KG {', 'graph [rankdir=LR, bgcolor="#f8fafc", pad="0.4", nodesep="0.35", ranksep="0.9", splines=true];',
           'node [fontname="Helvetica", fontsize=11, style="filled,rounded", color="#64748b", margin="0.12"];',
           'edge [fontname="Helvetica", fontsize=9, color="#94a3b8", arrowsize=0.6];']
    for key, value in sorted(nodes.items()):
        shape = 'ellipse' if value['type'] == 'User' else 'box'
        tooltip = key if 'path' not in value else ' > '.join(value['path'])
        dot.append(f'{quoted(key)} [label={quoted(value["label"])}, shape={shape}, fillcolor={quoted(COLORS[value["type"]])}, tooltip={quoted(tooltip)}];')
    dot.append('{ rank=same; ' + '; '.join(quoted(p) for p in products) + '; }')
    for edge in edges:
        relation = edge['relation']
        label = f'rated {edge["rating"]:g}/5' if relation == 'rated' else relation
        color = '#7c3aed' if relation == 'rated' else '#16a34a' if relation in ('belongs_to', 'category_child_of') else '#ea580c'
        dot.append(f'{quoted(edge["source"])} -> {quoted(edge["target"])} [label={quoted(label)}, color={quoted(color)}, tooltip={quoted(json.dumps(edge))}];')
    dot.append('}')
    dot_path = stem.with_suffix('.dot')
    dot_path.write_text('\n'.join(dot) + '\n')
    for extension in ['svg', 'png']:
        subprocess.run(['dot', '-T' + extension, str(dot_path), '-o', str(stem.with_suffix('.' + extension))], check=True)
    svg = stem.with_suffix('.svg').read_text()
    svg = svg[svg.index('<svg'):]
    legend = ' '.join(f'<span style="background:{color};padding:6px 12px;border-radius:6px">{kind}</span>' for kind, color in COLORS.items())
    counts = html.escape(json.dumps(sample['full_graph_counts'], indent=2))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Amazon4U — {html.escape(category)}</title>
<style>body{{font-family:system-ui;background:#f8fafc;color:#0f172a;margin:24px}} .canvas{{overflow:auto;border:1px solid #cbd5e1;background:white}}svg{{height:auto}}button{{margin:12px;padding:6px 12px}}p{{max-width:1000px}}</style>
<h1>Amazon4U: {html.escape(category)}</h1><p>Real training-KG neighborhood: one user, up to {max_products} products, their category paths and explicit brands. This is a bounded illustrative view, not the entire graph or an evaluation sample. Product titles/text are not node features in KG-v1.</p>
<p>{legend}</p><p>Hover over nodes/edges for full IDs, paths and edge details. Ratings are observations, not a binary “likes” relation. Category arrows follow observed path prefixes, not a verified external taxonomy. Missing attributes have no invented edges.</p>
<button onclick="zoom(1.2)">Zoom +</button><button onclick="zoom(1/1.2)">Zoom −</button><button onclick="reset()">Reset</button>
<div class="canvas">{svg}</div><h2>Full graph counts (not pictured in full)</h2><pre>{counts}</pre>
<script>const svg=document.querySelector('svg');const base=svg.viewBox.baseVal.width;let scale=1;function draw(){{svg.style.width=(base*scale)+'px';}}function zoom(f){{scale=Math.max(.1,Math.min(5,scale*f));draw();}}function reset(){{scale=1;draw();}}draw();</script></html>'''
    stem.with_suffix('.html').write_text(document)
    print(f'Saved {stem.with_suffix(".html")} ({len(nodes)} nodes, {len(edges)} edges)')
    return sample


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=root / 'data/processed/kg-v1')
    parser.add_argument('--output', type=Path, default=root / 'data/visualizations')
    parser.add_argument('--category', default='Musical_Instruments', help='Category name, or all')
    parser.add_argument('--max-products', type=int, default=6)
    parser.add_argument('--user-id')
    args = parser.parse_args()
    categories = json.loads((args.graph / 'manifest.json').read_text())['categories'] if args.category == 'all' else [args.category]
    for category in categories:
        render(args.graph, category, args.output, args.max_products, args.user_id)
