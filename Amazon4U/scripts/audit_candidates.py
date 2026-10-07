#!/usr/bin/env python3
"""Count warm/cold targets, filtered-history targets, users and metadata vocabulary.

No ranking or model training. Warmth is condition-specific: at least one permitted
positive training edge. History filtering always uses all recorded train/valid items.
Metadata-only vocabulary uses exactly KG-v1 category prefixes and explicit brands.
"""
import argparse
import json
import tempfile
from pathlib import Path

import duckdb
from build_kg import literal


def run(raw, config_path, output):
    config = json.loads(config_path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'warm_definition': 'at least one positive training interaction under the condition',
              'history': {'valid': 'all training items', 'test': 'all training and validation items'},
              'metadata_vocabulary': 'KG-v1 category path prefixes and explicit Brand only', 'conditions': {}}
    with tempfile.TemporaryDirectory(prefix='.work-candidates-', dir=output.parent) as work:
        con = duckdb.connect(str(Path(work) / 'audit.duckdb'))
        con.execute("SET memory_limit='512MB'; SET threads=1; SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory={literal(Path(work) / 'spill')}")
        for category in config['categories']:
            print(f'Auditing candidates: {category}', flush=True)
            con.execute('DROP TABLE IF EXISTS events')
            for i, split in enumerate(['train', 'valid', 'test']):
                csv = raw / 'benchmark/5core/last_out' / f'{category}.{split}.csv'
                query = f"""SELECT *,{literal(split)} AS split FROM read_csv({literal(csv)},header=true,
                    columns={{'user_id':'VARCHAR','parent_asin':'VARCHAR','rating':'DOUBLE','timestamp':'BIGINT'}})"""
                con.execute(('CREATE TABLE events AS ' if i == 0 else 'INSERT INTO events ') + query)
            con.execute(f"""CREATE OR REPLACE TABLE metadata AS SELECT parent_asin,
                list_transform(categories, x -> trim(x)) AS path,
                CASE WHEN json_type(try_cast(details AS JSON), '$.Brand')='VARCHAR'
                THEN nullif(trim(json_extract_string(try_cast(details AS JSON), '$.Brand')), '') END AS brand
                FROM read_parquet({literal(raw / ('raw_meta_' + category) / '*.parquet')})
                WHERE parent_asin IN (SELECT parent_asin FROM events)""")
            placeholders = ','.join(literal(v) for v in config['brand_placeholders_excluded'])
            con.execute(f'UPDATE metadata SET brand=NULL WHERE lower(brand) IN ({placeholders})')
            for condition, positive in [('all_ratings', 'true'), ('rating_ge_4', 'rating>=4')]:
                con.execute(f"CREATE OR REPLACE TABLE warm AS SELECT DISTINCT parent_asin FROM events WHERE split='train' AND {positive}")
                con.execute(f"CREATE OR REPLACE TABLE positive_users AS SELECT DISTINCT user_id FROM events WHERE split='train' AND {positive}")
                result = {'warm_items': con.execute('SELECT count(*) FROM warm').fetchone()[0], 'splits': {}}
                for split in ['valid', 'test']:
                    history = "('train')" if split == 'valid' else "('train','valid')"
                    con.execute(f"""CREATE OR REPLACE TABLE targets AS SELECT e.*,
                        EXISTS(SELECT 1 FROM warm w WHERE w.parent_asin=e.parent_asin) AS is_warm,
                        EXISTS(SELECT 1 FROM events h WHERE h.split IN {history} AND h.user_id=e.user_id
                            AND h.parent_asin=e.parent_asin) AS in_filtered_history,
                        EXISTS(SELECT 1 FROM events t WHERE t.split='train' AND t.user_id=e.user_id) AS known_user,
                        EXISTS(SELECT 1 FROM positive_users t WHERE t.user_id=e.user_id) AS positive_history_user
                        FROM events e WHERE split={literal(split)}""")
                    query = f"""SELECT count(*) AS rows,
                        count(*) FILTER (WHERE NOT is_warm) AS cold_rows,
                        count(*) FILTER (WHERE in_filtered_history) AS targets_in_filtered_history,
                        count(DISTINCT user_id) FILTER (WHERE NOT known_user) AS users_absent_from_raw_training,
                        count(DISTINCT user_id) FILTER (WHERE NOT positive_history_user) AS users_without_positive_training_history,
                        count(*) FILTER (WHERE {positive}) AS relevant_rows,
                        count(*) FILTER (WHERE {positive} AND NOT is_warm) AS relevant_cold_rows,
                        count(*) FILTER (WHERE {positive} AND in_filtered_history) AS relevant_filtered_history_rows,
                        count(*) FILTER (WHERE {positive} AND is_warm AND NOT in_filtered_history AND known_user) AS eligible_warm_rows
                        FROM targets"""
                    cursor = con.execute(query)
                    counts = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))
                    counts['cold_ratio_all_rows'] = counts['cold_rows'] / counts['rows'] if counts['rows'] else None
                    counts['relevant_cold_ratio'] = counts['relevant_cold_rows'] / counts['relevant_rows'] if counts['relevant_rows'] else None
                    result['splits'][split] = counts
                # Inspect metadata vocabulary for cold items in either holdout, without
                # admitting their metadata to the training graph or fitting anything.
                con.execute("""CREATE OR REPLACE TABLE scoped_metadata AS SELECT m.*,
                    EXISTS(SELECT 1 FROM warm w WHERE w.parent_asin=m.parent_asin) AS is_warm FROM metadata m
                    WHERE m.parent_asin IN (SELECT parent_asin FROM warm)
                       OR m.parent_asin IN (SELECT parent_asin FROM events WHERE split IN ('valid','test'))""")
                con.execute("""CREATE OR REPLACE TABLE prefixes AS SELECT path,is_warm,
                    unnest(range(1,len(path)+1)) AS depth FROM scoped_metadata""")
                con.execute("""CREATE OR REPLACE TABLE vocab AS SELECT node_id,
                    bool_or(is_warm) AS warm, bool_or(NOT is_warm) AS cold FROM (
                        SELECT 'c:'||to_json(list_slice(path,1,depth))::VARCHAR AS node_id,is_warm FROM prefixes
                        UNION ALL
                        SELECT 'b:'||brand,is_warm FROM scoped_metadata WHERE brand IS NOT NULL)
                    GROUP BY node_id""")
                result['warm_metadata_nodes'] = con.execute('SELECT count(*) FROM vocab WHERE warm').fetchone()[0]
                result['cold_only_metadata_nodes'] = con.execute('SELECT count(*) FROM vocab WHERE cold AND NOT warm').fetchone()[0]
                report['conditions'].setdefault(condition, {})[category] = result
            print(json.dumps({c: report['conditions'][c][category] for c in report['conditions']}, indent=2), flush=True)
        con.close()
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=root / 'data/raw')
    parser.add_argument('--config', type=Path, default=root / 'configs/kg-v1.json')
    parser.add_argument('--output', type=Path, default=root / 'data/reports/candidate-audit.json')
    args = parser.parse_args()
    run(args.raw, args.config, args.output)
