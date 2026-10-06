"""Small-fixture KG correctness and leakage regression tests (no full data needed)."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_kg import build_category
from validate_kg_splits import run as validate_splits


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.category = 'Fixture'
        self.csv_dir = self.raw / 'benchmark/5core/last_out'
        self.csv_dir.mkdir(parents=True)
        self.meta_dir = self.raw / 'raw_meta_Fixture'
        self.meta_dir.mkdir()
        self.config = json.loads((ROOT / 'configs/kg-v1.json').read_text())
        self.records = [
            {'parent_asin': 'p1', 'categories': ['Root', 'Shared'], 'details': '{"Brand":" Acme ","Best Sellers Rank":42}', 'average_rating': 4.0},
            {'parent_asin': 'p2', 'categories': ['Other', 'Shared'], 'details': '{"Brand":"None"}', 'average_rating': 2.0},
            {'parent_asin': 'unused', 'categories': ['Hidden'], 'details': '{"Brand":"Never"}', 'average_rating': 5.0},
        ]
        self.train = [('u1', 'p1', 5, 100), ('u1', 'p2', 2, 200), ('u2', 'missing', 3, 100)]
        self.write_csv('train', self.train)
        self.write_csv('valid', [('u1', 'unused', 5, 300)])
        self.write_csv('test', [('u1', 'heldout', 1, 400)])
        self.write_meta()

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, split, rows):
        with (self.csv_dir / f'Fixture.{split}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['user_id', 'parent_asin', 'rating', 'timestamp'])
            writer.writerows(rows)

    def write_meta(self):
        pq.write_table(pa.Table.from_pylist(self.records), self.meta_dir / 'full.parquet')

    def build(self, name):
        output = self.root / name
        con = duckdb.connect()
        try:
            report = build_category(con, self.raw, self.category, output, self.config)
        finally:
            con.close()
        tables = {p.stem: pq.read_table(p).to_pylist() for p in output.glob('*.parquet')}
        return report, tables

    def test_graph_semantics_and_missing_metadata(self):
        report, tables = self.build('graph')
        self.assertEqual(report['counts']['rated'], 3)
        self.assertEqual(report['counts']['users'], 2)
        self.assertEqual(report['training_products_without_metadata'], 1)
        self.assertEqual(tables['has_brand'], [{'source': 'p:p1', 'target': 'b:Acme'}])
        self.assertNotIn('unused', [row['parent_asin'] for row in tables['products']])
        # Same leaf label under different parents remains two distinct categories.
        shared = [r for r in tables['categories'] if r['label'] == 'Shared']
        self.assertEqual(len(shared), 2)
        self.assertNotEqual(shared[0]['node_id'], shared[1]['node_id'])
        self.assertEqual({row['rating'] for row in tables['rated']}, {2.0, 3.0, 5.0})
        self.assertEqual(len(tables['belongs_to']), 4)
        self.assertEqual(len(tables['category_child_of']), 2)

    def test_heldout_mutations_do_not_change_graph(self):
        _, before = self.build('before')
        self.write_csv('valid', [('different_user', 'different_product', 1, 999)])
        self.write_csv('test', [('u1', 'p1', 5, 1)])
        _, after = self.build('after')
        self.assertEqual(before, after)

    def test_excluded_metadata_does_not_change_graph(self):
        _, before = self.build('before')
        for row in self.records:
            row['average_rating'] = 1.0
            row['rating_number'] = 999999
            row['store'] = 'Not a brand'
            row['description'] = ['Target review says five stars']
            details = json.loads(row['details'])
            details['Best Sellers Rank'] = 987654
            row['details'] = json.dumps(details)
        self.write_meta()
        _, after = self.build('after')
        self.assertEqual(before, after)

    def test_independent_split_validation_detects_overlap(self):
        graph = self.root / 'published'
        graph.mkdir()
        con = duckdb.connect()
        try:
            report = build_category(con, self.raw, self.category, graph / self.category, self.config)
        finally:
            con.close()
        (graph / 'manifest.json').write_text(json.dumps({'categories': {self.category: report}}))
        output = self.root / 'split-report.json'
        self.assertTrue(validate_splits(self.raw, graph, output)['passed'])
        self.write_csv('test', [('u1', 'p1', 1, 999)])
        with self.assertRaisesRegex(ValueError, 'split validation failed'):
            validate_splits(self.raw, graph, output)
        self.assertEqual(json.loads(output.read_text())['categories'][self.category]['test_pairs_in_training_graph'], 1)

    def test_duplicate_training_pair_rejected(self):
        self.write_csv('train', self.train + [self.train[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate user-item'):
            self.build('invalid')

    def test_duplicate_metadata_rejected(self):
        self.records.append(dict(self.records[0]))
        self.write_meta()
        with self.assertRaisesRegex(ValueError, 'Duplicate metadata'):
            self.build('invalid')

    def test_empty_attributes_are_not_fabricated(self):
        self.records = [{'parent_asin': 'p1', 'categories': [], 'details': 'not JSON'},
                        {'parent_asin': 'p2', 'categories': [], 'details': '{}'}]
        schema = pa.schema([('parent_asin', pa.string()), ('categories', pa.list_(pa.string())), ('details', pa.string())])
        pq.write_table(pa.Table.from_pylist(self.records, schema=schema), self.meta_dir / 'full.parquet')
        report, _ = self.build('empty')
        self.assertEqual(report['counts']['categories'], 0)
        self.assertEqual(report['counts']['brands'], 0)


if __name__ == '__main__':
    unittest.main()
