"""Profiling P1/P2: dataset staging, verification and both feedback conditions."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.profiling_data import (
    MANIFEST_NAME, ProfilingData, ProfilingDataError, _sha256, stage_dataset,
)
from scripts.recommendation_evaluation import TrainingContext

TRAIN = [('ua', 'p1', 5.0, 1), ('ua', 'p2', 2.0, 2), ('ub', 'p2', 4.0, 3), ('ub', 'p3', 1.0, 4),
         ('uc', 'p1', 3.0, 5)]
VALID = [('ua', 'p3', 4.0, 9), ('ub', 'p1', 5.0, 9), ('uc', 'p2', 2.0, 9)]
CAT = 'c:["Instruments"]'


def _csv(path, rows):
    path.write_text('user_id,parent_asin,rating,timestamp\n'
                    + ''.join(f'{u},{p},{r},{t}\n' for u, p, r, t in rows))


def _kg_root(tmp, train=TRAIN):
    root = tmp / 'kg-v1'
    (root / 'Musical_Instruments').mkdir(parents=True)
    users = sorted({u for u, *_ in train})
    products = sorted({p for _, p, *_ in train})
    tables = {
        'users': [{'node_id': f'u:{u}', 'user_id': u} for u in users],
        'products': [{'node_id': f'p:{p}', 'parent_asin': p} for p in products],
        'categories': [{'node_id': CAT, 'label': 'Instruments', 'path_json': '["Instruments"]'}],
        'brands': [{'node_id': 'b:Acme', 'label': 'Acme'}],
        'rated': [{'source': f'u:{u}', 'target': f'p:{p}', 'rating': r, 'timestamp': t}
                  for u, p, r, t in train],
        'belongs_to': [{'source': 'p:p1', 'target': CAT}, {'source': 'p:p2', 'target': CAT}],
        'category_child_of': [{'source': 'c:["Instruments","Guitars"]', 'target': CAT}],
        'has_brand': [{'source': 'p:p1', 'target': 'b:Acme'}],
    }
    counts = {}
    for name, rows in tables.items():
        pq.write_table(pa.Table.from_pylist(rows), root / 'Musical_Instruments' / f'{name}.parquet')
        counts[name] = len(rows)
    (root / 'manifest.json').write_text(json.dumps({
        'config_sha256': 'c' * 64,
        'categories': {'Musical_Instruments': {'counts': counts, 'input_sha256': {'x': 'd' * 64}}}}))
    return root


class ProfilingDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        splits = self.tmp / 'splits'
        splits.mkdir()
        _csv(splits / 'Musical_Instruments.train.csv', TRAIN)
        _csv(splits / 'Musical_Instruments.valid.csv', VALID)
        _csv(splits / 'Musical_Instruments.test.csv', VALID)
        self.splits = splits
        self.data = self.tmp / 'dataset'
        stage_dataset(_kg_root(self.tmp), splits, self.data)
        self.sha = _sha256(self.data / MANIFEST_NAME)

    def test_staging_holds_only_the_approved_files_and_never_the_test_split(self):
        names = sorted(p.relative_to(self.data).as_posix() for p in self.data.rglob('*') if p.is_file())
        self.assertNotIn('Musical_Instruments.test.csv', names)
        self.assertEqual(len(names), 11)
        manifest = json.loads((self.data / MANIFEST_NAME).read_text())
        self.assertIs(manifest['test_split_included'], False)

    def test_both_conditions_equal_the_accepted_training_context(self):
        data = ProfilingData(self.data, self.sha)
        records = [(u, p, r) for u, p, r, _ in TRAIN]
        for condition in ('all_ratings', 'rating_ge_4'):
            ours, accepted = data.context(condition), TrainingContext(records, condition=condition)
            for field in ('users', 'warm_items', 'positive_history', 'all_rated_history',
                          'popularity', 'sorted_warm_items'):
                self.assertEqual(getattr(ours, field), getattr(accepted, field), (condition, field))
        ge4 = data.context('rating_ge_4')
        self.assertIn('p2', ge4.warm_items)
        self.assertNotIn('p2', ge4.positive_history['ua'])
        self.assertIn('p2', ge4.all_rated_history['ua'])
        self.assertNotIn('p3', ge4.warm_items)
        self.assertEqual(data.product_categories, {'p1': [CAT], 'p2': [CAT]})
        self.assertEqual(data.product_brands, {'p1': 'b:Acme'})
        self.assertEqual(data.validation_targets(), [(u, p, r) for u, p, r, _ in VALID])

    def assertRefused(self, pattern, sha=None):
        with self.assertRaisesRegex(ProfilingDataError, pattern):
            ProfilingData(self.data, sha or self.sha)

    def test_pinned_manifest_and_changed_content_are_refused(self):
        self.assertRefused('pinned SHA-256', sha='0' * 64)
        path = self.data / 'Musical_Instruments.valid.csv'
        path.write_text(path.read_text().replace('ub,p1', 'ub,p2'))
        self.assertRefused('content differs')

    def test_extra_missing_test_and_symlinked_files_are_refused(self):
        (self.data / 'notes.txt').write_text('x')
        self.assertRefused('differ from the manifest')
        os.remove(self.data / 'notes.txt')
        os.replace(self.data / 'Musical_Instruments.valid.csv', self.tmp / 'valid.csv')
        self.assertRefused('differ from the manifest')
        os.replace(self.tmp / 'valid.csv', self.data / 'Musical_Instruments.valid.csv')
        (self.data / 'Musical_Instruments.test.csv').write_text('x')
        self.assertRefused('test-split')
        os.remove(self.data / 'Musical_Instruments.test.csv')
        target = self.data / 'kg.brands.parquet'
        os.replace(target, self.tmp / 'brands.parquet')
        os.symlink(self.tmp / 'brands.parquet', target)
        self.assertRefused('symlinks')

    def test_only_kaggle_dataset_metadata_is_tolerated_as_an_extra_file(self):
        (self.data / 'dataset-metadata.json').write_text('{}')
        ProfilingData(self.data, self.sha)
        (self.data / 'dataset-metadata.txt').write_text('{}')
        self.assertRefused('differ from the manifest')

    def test_kg_edges_that_differ_from_the_training_split_are_refused(self):
        other = self.tmp / 'other'
        other.mkdir()
        stage_dataset(_kg_root(other, train=TRAIN[:-1] + [('uc', 'p1', 4.0, 5)]), self.splits,
                      other / 'dataset')
        with self.assertRaisesRegex(ProfilingDataError, 'rated edges differ'):
            ProfilingData(other / 'dataset', _sha256(other / 'dataset' / MANIFEST_NAME))


if __name__ == '__main__':
    unittest.main()
