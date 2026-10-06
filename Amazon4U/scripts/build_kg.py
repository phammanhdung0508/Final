#!/usr/bin/env python3
"""Build and validate KG-v1 from training CSVs and approved metadata only.

Does not read validation/test files, create embeddings, or train a model.
Output is published atomically only after structural validation succeeds.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import duckdb


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_table(con, name, output, ordering):
    con.execute(f"COPY (SELECT * FROM {name} ORDER BY {ordering}) TO {literal(output / (name + '.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)")


def validate(con, output):
    tables = ["users", "products", "categories", "brands", "rated", "belongs_to", "category_child_of", "has_brand"]
    for name in tables:
        con.execute(f"CREATE OR REPLACE VIEW saved_{name} AS SELECT * FROM read_parquet({literal(output / (name + '.parquet'))})")
    counts = {name: con.execute(f"SELECT count(*) FROM saved_{name}").fetchone()[0] for name in tables}
    for name in tables[:4]:
        bad = con.execute(f"SELECT count(*)-count(DISTINCT node_id) FROM saved_{name}").fetchone()[0]
        if bad:
            raise ValueError(f"Duplicate node IDs in {name}")
    for edges, src, dst in [("rated", "users", "products"), ("belongs_to", "products", "categories"),
                            ("category_child_of", "categories", "categories"), ("has_brand", "products", "brands")]:
        bad = con.execute(f"""SELECT count(*) FROM saved_{edges} e
            WHERE NOT EXISTS (SELECT 1 FROM saved_{src} n WHERE n.node_id=e.source)
               OR NOT EXISTS (SELECT 1 FROM saved_{dst} n WHERE n.node_id=e.target)""").fetchone()[0]
        duplicate = con.execute(f"""SELECT count(*) FROM (SELECT source,target FROM saved_{edges}
            GROUP BY source,target HAVING count(*)>1)""").fetchone()[0]
        if bad or duplicate:
            raise ValueError(f"Invalid {edges}: {bad} dangling endpoints, {duplicate} duplicate pairs")
    if counts["rated"] != con.execute("SELECT count(*) FROM training").fetchone()[0]:
        raise ValueError("Training interaction count changed")
    # Compare persisted observations including ratings/timestamps, not merely counts.
    difference = con.execute("""SELECT count(*) FROM (
        (SELECT 'u:'||user_id AS source,'p:'||parent_asin AS target,rating,timestamp FROM training
         EXCEPT ALL SELECT source,target,rating,timestamp FROM saved_rated)
        UNION ALL
        (SELECT source,target,rating,timestamp FROM saved_rated EXCEPT ALL
         SELECT 'u:'||user_id,'p:'||parent_asin,rating,timestamp FROM training))""").fetchone()[0]
    if difference:
        raise ValueError("Saved interaction edges differ from training records")
    for name, cols in [("users", ["node_id", "user_id"]), ("products", ["node_id", "parent_asin"]),
                       ("categories", ["node_id", "label", "path_json"]), ("brands", ["node_id", "label"])]:
        actual = [r[0] for r in con.execute(f"DESCRIBE saved_{name}").fetchall()]
        if actual != cols:
            raise ValueError(f"Unexpected node features in {name}: {actual}")
    return {"counts": counts, "checks": {"unique_node_ids": True, "valid_typed_endpoints": True,
            "unique_relation_pairs": True, "rated_exactly_matches_training": True,
            "node_feature_allowlist": True}}


def build_category(con, raw, category, output, config):
    output.mkdir()
    training_path = raw / "benchmark/5core/last_out" / f"{category}.train.csv"
    meta_paths = sorted((raw / f"raw_meta_{category}").glob("*.parquet"))
    if not meta_paths:
        raise FileNotFoundError(f"No metadata for {category}")
    con.execute(f"""CREATE OR REPLACE TABLE training AS SELECT * FROM read_csv({literal(training_path)},
        header=true, columns={{'user_id':'VARCHAR','parent_asin':'VARCHAR','rating':'DOUBLE','timestamp':'BIGINT'}})""")
    invalid = con.execute("""SELECT count(*) FROM training WHERE user_id IS NULL OR trim(user_id)=''
        OR parent_asin IS NULL OR trim(parent_asin)='' OR timestamp IS NULL OR timestamp<=0
        OR rating IS NULL OR NOT isfinite(rating) OR rating<1 OR rating>5""").fetchone()[0]
    duplicate = con.execute("""SELECT count(*) FROM (SELECT user_id,parent_asin FROM training
        GROUP BY user_id,parent_asin HAVING count(*)>1)""").fetchone()[0]
    if invalid or duplicate:
        raise ValueError(f"Invalid training records: {invalid}; duplicate user-item pairs: {duplicate}")
    con.execute("CREATE OR REPLACE TABLE users AS SELECT DISTINCT 'u:'||user_id AS node_id,user_id FROM training")
    con.execute("CREATE OR REPLACE TABLE products AS SELECT DISTINCT 'p:'||parent_asin AS node_id,parent_asin FROM training")
    # Read only approved metadata projections; do not materialize entire details as features.
    con.execute(f"""CREATE OR REPLACE TABLE metadata AS
        SELECT parent_asin,
            list_transform(categories, x -> trim(x)) AS path,
            CASE WHEN json_type(try_cast(details AS JSON), '$.Brand')='VARCHAR'
            THEN nullif(trim(json_extract_string(try_cast(details AS JSON), '$.Brand')), '') END AS brand
        FROM read_parquet({literal(raw / ('raw_meta_' + category) / '*.parquet')}) m
        WHERE parent_asin IN (SELECT parent_asin FROM products)""")
    duplicate_meta = con.execute("SELECT count(*)-count(DISTINCT parent_asin) FROM metadata").fetchone()[0]
    blank_labels = con.execute("""SELECT count(*) FROM (SELECT unnest(path) AS label FROM metadata)
        WHERE label IS NULL OR label=''""").fetchone()[0]
    if duplicate_meta or blank_labels:
        raise ValueError("Duplicate metadata IDs or blank path labels: audit/resolve before graph construction")
    placeholders = ','.join(literal(v) for v in config["brand_placeholders_excluded"])
    con.execute(f"UPDATE metadata SET brand=NULL WHERE lower(brand) IN ({placeholders})")
    con.execute("""CREATE OR REPLACE TABLE paths AS SELECT parent_asin,path,
        unnest(range(1, len(path)+1)) AS depth FROM metadata""")
    con.execute("""CREATE OR REPLACE TABLE prefixes AS SELECT parent_asin,depth,
        to_json(list_slice(path,1,depth))::VARCHAR AS path_json,
        path[depth] AS label,
        CASE WHEN depth>1 THEN to_json(list_slice(path,1,depth-1))::VARCHAR END AS parent_json
        FROM paths""")
    con.execute("""CREATE OR REPLACE TABLE categories AS SELECT DISTINCT
        'c:'||path_json AS node_id,label,path_json FROM prefixes""")
    con.execute("CREATE OR REPLACE TABLE brands AS SELECT DISTINCT 'b:'||brand AS node_id,brand AS label FROM metadata WHERE brand IS NOT NULL")
    con.execute("""CREATE OR REPLACE VIEW rated AS SELECT 'u:'||user_id AS source,
        'p:'||parent_asin AS target,rating,timestamp FROM training""")
    con.execute("""CREATE OR REPLACE TABLE belongs_to AS SELECT DISTINCT 'p:'||parent_asin AS source,
        'c:'||path_json AS target FROM prefixes""")
    con.execute("""CREATE OR REPLACE TABLE category_child_of AS SELECT DISTINCT 'c:'||path_json AS source,
        'c:'||parent_json AS target FROM prefixes WHERE depth>1""")
    con.execute("""CREATE OR REPLACE TABLE has_brand AS SELECT 'p:'||parent_asin AS source,
        'b:'||brand AS target FROM metadata WHERE brand IS NOT NULL""")
    for name in ["users", "products", "categories", "brands"]:
        copy_table(con, name, output, "node_id")
    for name in ["rated", "belongs_to", "category_child_of", "has_brand"]:
        copy_table(con, name, output, "source,target")
    validation = validate(con, output)
    validation["training_products_without_metadata"] = con.execute("""SELECT count(*) FROM products p
        WHERE NOT EXISTS (SELECT 1 FROM metadata m WHERE m.parent_asin=p.parent_asin)""").fetchone()[0]
    validation["input_sha256"] = {str(p.relative_to(raw)): sha256(p) for p in [training_path, *meta_paths]}
    return validation


def run(raw, output, config_path):
    config = json.loads(config_path.read_text())
    if (config["schema_version"] != "kg-v1" or config["graph_scope"] != "separate_per_category"
            or config["interaction_split"] != "train" or config["product_scope"] != "training_products_only"
            or config["approved_node_features"] or config["cross_category_history"]):
        raise ValueError("This builder supports only the fixed structural, per-category KG-v1 policy")
    if config["approved_metadata_fields"] != ["parent_asin", "categories", "details.Brand"]:
        raise ValueError("Unsupported metadata allowlist")
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "config_sha256": sha256(config_path), "categories": {}}
    with tempfile.TemporaryDirectory(prefix=".work-kg-", dir=output.parent) as work:
        stage = Path(work) / "graph"
        stage.mkdir()
        con = duckdb.connect(str(Path(work) / "build.duckdb"))
        con.execute("SET memory_limit='512MB'; SET threads=1; SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory={literal(Path(work) / 'spill')}")
        for category in config["categories"]:
            if not category.replace('_', '').isalnum():
                raise ValueError(f"Invalid category name {category}")
            print(f"Building {category}", flush=True)
            manifest["categories"][category] = build_category(con, raw, category, stage / category, config)
            print(json.dumps(manifest["categories"][category]["counts"], indent=2), flush=True)
        con.close()
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(output)
    print(f"Built and structurally validated: {output}", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data/raw")
    parser.add_argument("--output", type=Path, default=root / "data/processed/kg-v1")
    parser.add_argument("--config", type=Path, default=root / "configs/kg-v1.json")
    args = parser.parse_args()
    run(args.raw, args.output, args.config)
