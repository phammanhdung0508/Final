#!/usr/bin/env python3
"""Independently check persisted KG edges against published split boundaries.

Held-out files are read only here for validation, never by build_kg.py.
"""
import argparse
import json
from pathlib import Path

import duckdb

from build_kg import literal, sha256


def run(raw, graph, output):
    manifest = json.loads((graph / 'manifest.json').read_text())
    report = {'passed': True, 'categories': {}}
    con = duckdb.connect()
    con.execute("SET memory_limit='512MB'; SET threads=1; SET preserve_insertion_order=false")
    for category, entry in manifest['categories'].items():
        edges = graph / category / 'rated.parquet'
        con.execute(f"CREATE OR REPLACE VIEW edges AS SELECT * FROM read_parquet({literal(edges)})")
        result = {'input_hashes_match': all(sha256(raw / p) == digest for p, digest in entry['input_sha256'].items())}
        for split in ['valid', 'test']:
            csv = raw / 'benchmark/5core/last_out' / f'{category}.{split}.csv'
            con.execute(f"""CREATE OR REPLACE VIEW heldout AS SELECT user_id,parent_asin FROM
                read_csv({literal(csv)},header=true,
                columns={{'user_id':'VARCHAR','parent_asin':'VARCHAR','rating':'DOUBLE','timestamp':'BIGINT'}})""")
            result[f'{split}_pairs_in_training_graph'] = con.execute("""SELECT count(*) FROM heldout h
                WHERE EXISTS (SELECT 1 FROM edges e WHERE e.source='u:'||h.user_id
                    AND e.target='p:'||h.parent_asin)""").fetchone()[0]
        passed = result['input_hashes_match'] and not any(result[f'{s}_pairs_in_training_graph'] for s in ['valid', 'test'])
        result['passed'] = passed
        report['passed'] &= passed
        report['categories'][category] = result
    con.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise ValueError('KG split validation failed')
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=root / 'data/raw')
    parser.add_argument('--graph', type=Path, default=root / 'data/processed/kg-v1')
    parser.add_argument('--output', type=Path, default=root / 'data/reports/kg-split-validation.json')
    args = parser.parse_args()
    run(args.raw, args.graph, args.output)
