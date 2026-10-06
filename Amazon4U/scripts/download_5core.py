#!/usr/bin/env python3
"""Download size-verified, resumable Amazon 2023 benchmark splits and metadata.

Default: both 5-core last_out CSVs and full raw_meta_* Parquet folders.
Requires curl; otherwise uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REPO = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORIES = ["Electronics", "Toys_and_Games", "Musical_Instruments"]


def listing(path, revision):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{path}?limit=1000"
    result = []
    while url:
        with urlopen(url, timeout=60) as response:
            result.extend(json.load(response))
            link = response.headers.get("Link", "")
        url = next((part.split("<", 1)[1].split(">", 1)[0]
                    for part in link.split(",") if 'rel="next"' in part), None)
    return [item for item in result if item["type"] == "file"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    parser.add_argument("--kind", choices=["all", "benchmark", "metadata"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for category in args.categories:
        if not category.replace("_", "").isalnum():
            parser.error(f"Invalid category: {category}")
    # Pin one revision so a run cannot mix files from different commits.
    with urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60) as response:
        revision = json.load(response)["sha"]
    files = []
    if args.kind in ("all", "benchmark"):
        available = {item["path"].rsplit("/", 1)[-1]: item
                     for item in listing("benchmark/5core/last_out", revision)}
        for category in args.categories:
            for split in ("train", "valid", "test"):
                files.append(available[f"{category}.{split}.csv"])
    if args.kind in ("all", "metadata"):
        for category in args.categories:
            files.extend(item for item in listing(f"raw_meta_{category}", revision)
                         if item["path"].endswith(".parquet"))
    total = sum(item["size"] for item in files)
    print(f"Revision {revision}; {len(files)} files; {total / 1e9:.3f} GB", flush=True)
    for item in files:
        print(f"  {item['path']}: {item['size'] / 1e6:.1f} MB", flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    remaining = sum(max(0, item["size"] - (args.output / item["path"]).stat().st_size)
                    if (args.output / item["path"]).exists() else item["size"] for item in files)
    if shutil.disk_usage(args.output).free < remaining + 1024**3:
        raise RuntimeError("Insufficient disk space (including 1 GiB safety margin)")
    manifest = {"repository": REPO, "revision": revision, "categories": args.categories, "files": files}
    (args.output / f"download_manifest_{args.kind}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for item in files:
        target = args.output / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == item["size"]:
            print(f"Verified size / skipping: {target}", flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".part")
        if target.exists():
            if partial.exists():
                raise RuntimeError(f"Both partial and incomplete target exist: {target}")
            target.rename(partial)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(item['path'])}"
        print(f"Downloading/resuming: {target}", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "5", "--connect-timeout", "30", "--continue-at", "-",
                        "--output", str(partial), url], check=True)
        if partial.stat().st_size != item["size"]:
            raise RuntimeError(f"Size mismatch: {partial}")
        partial.rename(target)
    print("All downloads complete and sizes verified.", flush=True)


if __name__ == "__main__":
    main()
