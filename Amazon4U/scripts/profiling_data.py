"""Profiling P1/P2: the Musical_Instruments profiling dataset, staged, verified and loaded.

A profiling dataset is one flat directory holding exactly the KG-v1 parquet files
(`kg.<table>.parquet`), the training and validation CSV splits and
`profiling-data-manifest.json`. It is flat because `kaggle datasets create` skips
subdirectories by default. The manifest records every file's size and SHA-256. Loading
refuses a missing, extra, changed or non-regular file, and any name containing `test`: the
test split is never staged. The one other name allowed is Kaggle's `dataset-metadata.json`,
which the upload directory needs; its content is not data and is never read.

Both feedback conditions are built in memory by the accepted `TrainingContext`
(positives >=4, observed history all ratings, condition-specific warm pool). No new
on-disk graph is written. H-REF's bounded synthetic input adapter is not used or changed.
See docs/musical-instruments-profiling-plan.md.
"""

import csv
import hashlib
import json
import os
import shutil
import stat
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.recommendation_evaluation import TrainingContext

SCHEMA = 'amazon4u-profiling-data-v1'
CATEGORY = 'Musical_Instruments'
MANIFEST_NAME = 'profiling-data-manifest.json'
KAGGLE_METADATA_NAME = 'dataset-metadata.json'
KG_TABLES = ('users', 'products', 'categories', 'brands', 'rated', 'belongs_to',
             'category_child_of', 'has_brand')
SPLITS = ('train', 'valid')
CSV_HEADER = ['user_id', 'parent_asin', 'rating', 'timestamp']
CONDITIONS = ('all_ratings', 'rating_ge_4')
_CHUNK = 1 << 20


class ProfilingDataError(ValueError):
    """The profiling dataset is incomplete, changed or not the approved content."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def expected_names(category=CATEGORY):
    return sorted([f'kg.{t}.parquet' for t in KG_TABLES]
                  + [f'{category}.{s}.csv' for s in SPLITS])


def _refuse_test(name):
    if 'test' in name.lower():
        raise ProfilingDataError(f'{name}: test-split files are never staged or loaded')


def stage_dataset(kg_root, split_dir, out_dir, *, category=CATEGORY):
    """Copy the approved files into a new directory and write their manifest."""
    kg_root, split_dir, out_dir = Path(kg_root), Path(split_dir), Path(out_dir)
    kg_manifest = json.loads((kg_root / 'manifest.json').read_text())
    out_dir.mkdir(mode=0o700)
    sources = {f'kg.{t}.parquet': kg_root / category / f'{t}.parquet' for t in KG_TABLES}
    sources.update({f'{category}.{s}.csv': split_dir / f'{category}.{s}.csv' for s in SPLITS})
    files = {}
    for name in expected_names(category):
        _refuse_test(name)
        shutil.copyfile(sources[name], out_dir / name)
        files[name] = {'bytes': (out_dir / name).stat().st_size, 'sha256': _sha256(out_dir / name)}
    manifest = {
        'schema_version': SCHEMA, 'category': category, 'files': files,
        'kg_config_sha256': kg_manifest['config_sha256'],
        'kg_counts': kg_manifest['categories'][category]['counts'],
        'kg_input_sha256': kg_manifest['categories'][category]['input_sha256'],
        'test_split_included': False,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def verify_dataset(root, manifest_sha256):
    """Check the directory against its manifest, itself pinned by SHA-256; return it."""
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if _sha256(manifest_path) != manifest_sha256:
        raise ProfilingDataError('profiling-data manifest differs from the pinned SHA-256')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != SCHEMA or manifest.get('test_split_included') is not False:
        raise ProfilingDataError('unsupported profiling-data manifest')
    category = manifest['category']
    if sorted(manifest['files']) != expected_names(category):
        raise ProfilingDataError('manifest does not list exactly the approved files')
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            rel = (Path(dirpath) / name).relative_to(root).as_posix()
            _refuse_test(rel)
            mode = os.lstat(Path(dirpath) / name).st_mode
            if stat.S_ISLNK(mode):
                raise ProfilingDataError(f'{rel}: symlinks are refused')
            if name in filenames:
                if not stat.S_ISREG(mode):
                    raise ProfilingDataError(f'{rel}: not a regular file')
                found.append(rel)
    if KAGGLE_METADATA_NAME in found:
        found.remove(KAGGLE_METADATA_NAME)
    if sorted(found) != sorted([*manifest['files'], MANIFEST_NAME]):
        raise ProfilingDataError(f'dataset files differ from the manifest: {sorted(found)}')
    for name, record in manifest['files'].items():
        path = root / name
        if path.stat().st_size != record['bytes'] or _sha256(path) != record['sha256']:
            raise ProfilingDataError(f'{name}: content differs from the manifest')
    return manifest


def _read_split(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        if next(reader) != CSV_HEADER:
            raise ProfilingDataError(f'{path.name}: header must be {CSV_HEADER}')
        rows = []
        for row in reader:
            if len(row) != 4:
                raise ProfilingDataError(f'{path.name}: malformed row')
            rows.append((row[0], row[1], float(row[2]), int(row[3])))
    return rows


def _strip(value, prefix):
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ProfilingDataError(f'node id {value!r} lacks prefix {prefix!r}')
    return value[len(prefix):]


class ProfilingData:
    """Verified profiling inputs. `context(condition)` builds the accepted TrainingContext."""

    def __init__(self, root, manifest_sha256):
        root = Path(root)
        self.manifest = verify_dataset(root, manifest_sha256)
        category = self.manifest['category']
        counts = self.manifest['kg_counts']
        tables = {t: pq.read_table(root / f'kg.{t}.parquet').to_pylist() for t in KG_TABLES}
        for t in KG_TABLES:
            if len(tables[t]) != counts[t]:
                raise ProfilingDataError(f'kg.{t}.parquet row count differs from the KG-v1 manifest')
        self.train = _read_split(root / f'{category}.train.csv')
        self.validation = _read_split(root / f'{category}.valid.csv')
        rated = Counter((_strip(r['source'], 'u:'), _strip(r['target'], 'p:'), r['rating'],
                         r['timestamp']) for r in tables['rated'])
        if rated != Counter(self.train):
            raise ProfilingDataError('KG rated edges differ from the training split')
        if ({_strip(r['node_id'], 'u:') for r in tables['users']} != {u for u, *_ in self.train}
                or {_strip(r['node_id'], 'p:') for r in tables['products']}
                != {p for _, p, *_ in self.train}):
            raise ProfilingDataError('KG user or product nodes differ from the training split')
        self.product_categories = {}
        for r in tables['belongs_to']:
            self.product_categories.setdefault(_strip(r['source'], 'p:'), []).append(r['target'])
        self.product_categories = {p: sorted(c) for p, c in self.product_categories.items()}
        self.product_brands = {}
        for r in tables['has_brand']:
            product = _strip(r['source'], 'p:')
            if product in self.product_brands:
                raise ProfilingDataError(f'product {product} has more than one brand')
            self.product_brands[product] = r['target']
        users = [u for u, *_ in self.validation]
        if len(set(users)) != len(users):
            raise ProfilingDataError('validation split has more than one target per user')

    def context(self, condition):
        if condition not in CONDITIONS:
            raise ProfilingDataError(f'unknown feedback condition {condition!r}')
        return TrainingContext([(u, p, r) for u, p, r, _ in self.train], condition=condition)

    def validation_targets(self):
        return [(u, p, r) for u, p, r, _ in self.validation]
