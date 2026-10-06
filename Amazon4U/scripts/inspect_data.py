#!/usr/bin/env python3
"""Inspect downloaded metadata schemas and interaction-to-metadata coverage.

Requires pyarrow. Does not load review text or change source files.
"""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

CATEGORIES = ["Electronics", "Toys_and_Games", "Musical_Instruments"]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--data", type=Path, default=root / "data")
    args = parser.parse_args()
    report = {}
    for category in args.categories:
        print(f"Inspecting {category}", flush=True)
        paths = sorted((args.data / "raw" / f"raw_meta_{category}").glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No metadata for {category}")
        ids = set()
        rows = 0
        schemas = []
        for path in paths:
            parquet = pq.ParquetFile(path)
            schema = str(parquet.schema_arrow)
            if schema not in schemas:
                schemas.append(schema)
            rows += parquet.metadata.num_rows
            for batch in parquet.iter_batches(columns=["parent_asin"], batch_size=65536):
                ids.update(value for value in batch.column(0).to_pylist() if value)
        result = {"metadata_files": len(paths), "metadata_rows": rows,
                  "unique_metadata_product_ids": len(ids), "metadata_schemas": schemas, "splits": {}}
        for split in ("train", "valid", "test"):
            products = set()
            users = set()
            count = matched = 0
            examples = []
            path = args.data / "raw" / "benchmark" / "5core" / "last_out" / f"{category}.{split}.csv"
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                for row in reader:
                    count += 1
                    matched += row["parent_asin"] in ids
                    products.add(row["parent_asin"])
                    users.add(row["user_id"])
                    if len(examples) < 3:
                        examples.append(row)
            result["splits"][split] = {"columns": columns, "interactions": count,
                                       "users": len(users), "products": len(products),
                                       "matched_interactions": matched,
                                       "matched_products": len(products & ids),
                                       "metadata_coverage": matched / count if count else None,
                                       "first_three_examples_not_random_sample": examples}
        report[category] = result
        print(json.dumps({"category": category, "metadata_rows": rows, "splits": result["splits"]}, indent=2), flush=True)
    output = args.data / "reports" / "inspection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
