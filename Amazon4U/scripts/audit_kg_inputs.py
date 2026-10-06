#!/usr/bin/env python3
"""Audit complete metadata and published splits without fitting any model.

DuckDB spills to a temporary on-disk workspace; memory is capped at 512 MB.
Reports label/split properties for integrity checks, not model selection.
"""
import argparse
import json
import tempfile
from pathlib import Path

import duckdb

CATEGORIES = ["Electronics", "Toys_and_Games", "Musical_Instruments"]
SPLITS = ["train", "valid", "test"]


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def one(con, sql):
    result = con.execute(sql)
    return dict(zip([d[0] for d in result.description], result.fetchone()))


def run(raw, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"categories": {}, "cross_category": {},
              "notes": ["Audits use held-out labels only for data integrity, never graph/features.",
                        "Per-category graphs are required initially; cross-category features are not approved."]}
    with tempfile.TemporaryDirectory(prefix=".work-audit-", dir=output.parent) as work:
        con = duckdb.connect(str(Path(work) / "audit.duckdb"))
        con.execute("SET memory_limit='512MB'; SET threads=1; SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory={literal(Path(work) / 'spill')}")
        con.execute("CREATE TABLE user_bounds(category VARCHAR,user_id VARCHAR, train_max BIGINT, heldout_min BIGINT)")
        con.execute("CREATE TABLE catalog_ids(category VARCHAR, parent_asin VARCHAR)")
        for category in CATEGORIES:
            print(f"Auditing {category}", flush=True)
            glob = raw / f"raw_meta_{category}" / "*.parquet"
            con.execute(f"CREATE OR REPLACE VIEW source_metadata AS SELECT * FROM read_parquet({literal(glob)})")
            schema = [list(r) for r in con.execute("DESCRIBE source_metadata").fetchall()]
            quality = one(con, """
                SELECT count(*) AS rows,
                    count(*) FILTER (WHERE parent_asin IS NULL OR trim(parent_asin)='') AS missing_ids,
                    count(*) FILTER (WHERE title IS NULL OR trim(title)='') AS empty_titles,
                    count(*) FILTER (WHERE categories IS NULL OR len(categories)=0) AS empty_categories,
                    count(*) FILTER (WHERE features IS NULL OR len(features)=0) AS empty_features,
                    count(*) FILTER (WHERE description IS NULL OR len(description)=0) AS empty_descriptions,
                    count(*) FILTER (WHERE store IS NULL OR trim(store)='') AS empty_stores,
                    count(*) FILTER (WHERE price IS NULL OR trim(price)='') AS empty_prices,
                    count(*) FILTER (WHERE lower(trim(price)) IN ('none','null','nan','n/a','unknown')) AS placeholder_prices,
                    count(*) FILTER (WHERE try_cast(price AS DOUBLE) IS NULL) AS nonnumeric_prices,
                    count(*) FILTER (WHERE details IS NULL OR trim(details)='') AS empty_details,
                    count(*) FILTER (WHERE details IS NOT NULL AND trim(details)<>'' AND NOT json_valid(details)) AS malformed_details,
                    count(*) FILTER (WHERE json_valid(details) AND json_type(try_cast(details AS JSON)) <> 'OBJECT') AS nonobject_details,
                    count(*) FILTER (WHERE json_type(try_cast(details AS JSON), '$.Brand')='VARCHAR'
                        AND trim(json_extract_string(try_cast(details AS JSON), '$.Brand'))<>'') AS scalar_brand_rows,
                    count(*) FILTER (WHERE json_exists(try_cast(details AS JSON), '$."Best Sellers Rank"')) AS bestseller_rank_rows
                FROM source_metadata
            """)
            # Only small columns are materialized for graph-quality checks.
            con.execute("""CREATE OR REPLACE TABLE metadata AS SELECT parent_asin, categories, main_category,
                CASE WHEN json_type(try_cast(details AS JSON), '$.Brand')='VARCHAR'
                THEN nullif(trim(json_extract_string(try_cast(details AS JSON), '$.Brand')), '') END AS brand
                FROM source_metadata""")
            quality.update(one(con, """SELECT count(DISTINCT parent_asin) AS unique_products,
                count(*) - count(DISTINCT parent_asin) AS duplicate_id_excess_rows FROM metadata"""))
            quality["duplicate_conflicting_records"] = con.execute("""SELECT count(*) FROM (
                SELECT parent_asin FROM metadata GROUP BY parent_asin
                HAVING count(DISTINCT to_json(categories))>1 OR count(DISTINCT brand)>1
                    OR count(DISTINCT main_category)>1)""").fetchone()[0]
            quality["blank_category_entries"] = con.execute("""SELECT count(*) FROM metadata,
                unnest(categories) AS c(label) WHERE label IS NULL OR trim(label)=''""").fetchone()[0]
            quality["main_category_differs_from_path_root"] = con.execute("""SELECT count(*) FROM metadata
                WHERE len(categories)>0 AND main_category IS NOT NULL AND main_category<>categories[1]""").fetchone()[0]
            con.execute(f"INSERT INTO catalog_ids SELECT {literal(category)}, parent_asin FROM metadata")
            detail_keys = con.execute("""SELECT key, count(*) AS n FROM
                (SELECT unnest(json_keys(try_cast(details AS JSON))) AS key FROM source_metadata)
                GROUP BY key ORDER BY n DESC, key LIMIT 30""").fetchall()
            con.execute("DROP TABLE IF EXISTS events")
            for i, split in enumerate(SPLITS):
                path = raw / "benchmark/5core/last_out" / f"{category}.{split}.csv"
                select = f"""SELECT user_id, parent_asin, rating, timestamp, {literal(split)} AS split
                    FROM read_csv({literal(path)}, header=true,
                    columns={{'user_id':'VARCHAR','parent_asin':'VARCHAR','rating':'DOUBLE','timestamp':'BIGINT'}})"""
                con.execute(("CREATE TABLE events AS " if i == 0 else "INSERT INTO events ") + select)
            splits = {}
            for split in SPLITS:
                where = f"WHERE split={literal(split)}"
                splits[split] = one(con, f"""SELECT count(*) AS rows, count(DISTINCT user_id) AS users,
                    count(DISTINCT parent_asin) AS products, min(timestamp) AS earliest_timestamp,
                    max(timestamp) AS latest_timestamp,
                    count(*) FILTER (WHERE user_id IS NULL OR user_id='' OR parent_asin IS NULL OR parent_asin=''
                        OR timestamp IS NULL OR timestamp<=0 OR rating IS NULL OR NOT isfinite(rating)
                        OR rating<1 OR rating>5) AS invalid_rows
                    FROM events {where}""")
                splits[split]["duplicate_user_item_pairs"] = con.execute(f"""SELECT count(*) FROM
                    (SELECT user_id,parent_asin FROM events {where} GROUP BY user_id,parent_asin HAVING count(*)>1)""").fetchone()[0]
                splits[split]["missing_metadata_interactions"] = con.execute(f"""SELECT count(*) FROM events e
                    {where} AND NOT EXISTS (SELECT 1 FROM metadata m WHERE m.parent_asin=e.parent_asin)""").fetchone()[0]
                splits[split]["interactions_with_items_absent_from_training"] = con.execute(f"""SELECT count(*) FROM events e
                    {where} AND NOT EXISTS (SELECT 1 FROM events t
                        WHERE t.split='train' AND t.parent_asin=e.parent_asin)""").fetchone()[0]
            pairs = {}
            for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
                pairs[f"{a}_{b}"] = con.execute(f"""SELECT count(*) FROM (
                    SELECT user_id,parent_asin FROM events WHERE split={literal(a)}
                    INTERSECT SELECT user_id,parent_asin FROM events WHERE split={literal(b)})""").fetchone()[0]
            con.execute("""CREATE OR REPLACE TABLE bounds AS SELECT user_id,
                count(*) FILTER (WHERE split='train') AS train_n,
                count(*) FILTER (WHERE split='valid') AS valid_n,
                count(*) FILTER (WHERE split='test') AS test_n,
                max(timestamp) FILTER (WHERE split='train') AS train_max,
                min(timestamp) FILTER (WHERE split='valid') AS valid_min,
                max(timestamp) FILTER (WHERE split='valid') AS valid_max,
                min(timestamp) FILTER (WHERE split='test') AS test_min FROM events GROUP BY user_id""")
            chronology = one(con, """SELECT
                count(*) FILTER (WHERE train_n=0 OR valid_n<>1 OR test_n<>1) AS users_not_standard_leave_last_two,
                count(*) FILTER (WHERE train_max>valid_min OR valid_max>test_min) AS users_with_time_inversions,
                count(*) FILTER (WHERE train_max=valid_min OR valid_max=test_min) AS users_with_boundary_ties,
                min(train_n) AS minimum_training_user_degree FROM bounds""")
            item_degree = con.execute("""SELECT min(n) FROM (SELECT parent_asin,count(*) AS n FROM events
                WHERE split='train' GROUP BY parent_asin)""").fetchone()[0]
            full_degrees = one(con, """SELECT
                (SELECT min(n) FROM (SELECT user_id,count(*) AS n FROM events GROUP BY user_id)) AS full_min_user_degree,
                (SELECT min(n) FROM (SELECT parent_asin,count(*) AS n FROM events GROUP BY parent_asin)) AS full_min_item_degree""")
            train_catalog = one(con, """SELECT count(*) AS products,
                count(*) FILTER (WHERE len(categories)>0) AS products_with_categories,
                count(*) FILTER (WHERE brand IS NOT NULL) AS products_with_brand
                FROM metadata WHERE parent_asin IN (SELECT parent_asin FROM events WHERE split='train')""")
            con.execute(f"""INSERT INTO user_bounds SELECT {literal(category)},user_id,train_max,
                least(valid_min,test_min) FROM bounds""")
            report["categories"][category] = {"metadata_schema": schema, "metadata_quality": quality,
                "top_detail_keys": detail_keys, "splits": splits, "pair_overlap": pairs,
                "chronology": chronology, "full_collection_degrees": full_degrees,
                "minimum_training_item_degree": item_degree, "training_catalog_metadata": train_catalog}
            print(json.dumps(report["categories"][category], indent=2), flush=True)
        for i, a in enumerate(CATEGORIES):
            for b in CATEGORIES[i+1:]:
                report["cross_category"][f"{a}__{b}"] = one(con, f"""SELECT count(*) AS shared_users,
                    count(*) FILTER (WHERE a.train_max>=b.heldout_min) AS users_with_a_training_not_before_b_target,
                    count(*) FILTER (WHERE b.train_max>=a.heldout_min) AS users_with_b_training_not_before_a_target
                    FROM user_bounds a JOIN user_bounds b USING(user_id)
                    WHERE a.category={literal(a)} AND b.category={literal(b)}""")
                report["cross_category"][f"{a}__{b}"]["shared_metadata_product_ids"] = con.execute(f"""SELECT count(*) FROM (
                    SELECT parent_asin FROM catalog_ids WHERE category={literal(a)} INTERSECT
                    SELECT parent_asin FROM catalog_ids WHERE category={literal(b)})""").fetchone()[0]
        con.close()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data/raw")
    parser.add_argument("--output", type=Path, default=root / "data/reports/kg-input-audit.json")
    args = parser.parse_args()
    run(args.raw, args.output)
