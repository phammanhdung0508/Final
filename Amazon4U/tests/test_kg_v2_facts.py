"""KG-v2 fact rules: brand keys, store fallback, attribute cleaning and the frequency floor."""

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import kg_v2_facts as kf


def row(asin, store=None, **details):
    return asin, json.dumps(details), store


class BrandTests(unittest.TestCase):
    def test_brand_keys_merge_case_spacing_and_width(self):
        self.assertEqual({kf.brand_key(s) for s in ('Sela', 'sela', ' SELA ', 'Ｓｅｌａ')}, {'sela'})
        self.assertEqual(kf.brand_key('PDP  by\tDW'), kf.brand_key('PDP BY DW'))
        self.assertIsNone(kf.brand_key('   '))
        self.assertIsNone(kf.brand_key(None))

    def test_details_brand_wins_and_store_fills_only_a_gap(self):
        self.assertEqual(kf.product_facts({'Brand': 'Fender'}, 'Guitar Center')[:2], ('Fender', 'details.Brand'))
        self.assertEqual(kf.product_facts({}, 'Boss')[:2], ('Boss', 'store'))
        self.assertEqual(kf.product_facts({'Brand': '  '}, 'Boss')[:2], ('Boss', 'store'))
        self.assertEqual(kf.product_facts({'Brand': ['Fender']}, 'Boss')[:2], ('Boss', 'store'))

    def test_placeholders_give_no_brand_from_either_source(self):
        for value in ('Unknown', 'GENERIC', 'n/a', 'None', 'NaN', 'null', 'Artist  Unknown', '', None):
            with self.subTest(value=value):
                self.assertEqual(kf.product_facts({}, value)[:2], (None, None))
                self.assertEqual(kf.product_facts({'Brand': value}, None)[:2], (None, None))
        self.assertEqual(kf.product_facts({'Brand': 'Unknown'}, 'Boss')[:2], ('Boss', 'store'))
        self.assertEqual(kf.product_facts({}, 'Rinastore')[:2], ('Rinastore', 'store'))

    def test_kg_v1_placeholders_are_all_kept(self):
        v1 = json.loads((Path(__file__).resolve().parents[1] / 'configs/kg-v1.json').read_text())
        self.assertLessEqual(set(v1['brand_placeholders_excluded']), kf.BRAND_PLACEHOLDERS)

    def test_the_most_frequent_spelling_names_a_merged_brand(self):
        rows = [row('p1', Brand='BOSS'), row('p2', Brand='Boss'), row('p3', 'Boss'), row('p4', Brand='boss')]
        facts, counts = kf.build_facts(rows)
        self.assertEqual({v for _, r, v in facts if r == 'brand'}, {'Boss'})
        self.assertEqual(counts['brand_sources'], {'details.Brand': 3, 'store': 1})
        self.assertEqual(counts['brand_keys_with_several_spellings'], 1)


class AttributeTests(unittest.TestCase):
    def test_values_are_cleaned_split_and_merged_across_keys(self):
        _, _, attrs = kf.product_facts({'Color': ' Black ', 'Color Name': 'black', 'Power Source': 'Adapter,Electric',
                                        'Material Type': 'Alloy Steel', 'Size': 'Large', 'Item Weight': '2 lb'}, None)
        self.assertEqual(attrs, {'color=black', 'power=adapter', 'power=electric', 'material=alloy steel'})

    def test_values_on_fewer_than_five_products_are_dropped(self):
        rows = [row(f'p{i}', **{'Number of Strings': '6'}) for i in range(5)]
        rows += [row(f'q{i}', **{'Number of Strings': '7'}) for i in range(4)]
        facts, counts = kf.build_facts(rows)
        self.assertEqual({v for _, r, v in facts if r == 'attribute'}, {'strings=6'})
        self.assertEqual((counts['attribute_values_seen'], counts['attribute_values_kept']), (2, 1))

    def test_bad_details_json_gives_no_facts(self):
        facts, counts = kf.build_facts([('p1', '{not json', 'Boss'), ('p2', '[1, 2]', None)])
        self.assertEqual(facts, [('p1', 'brand', 'Boss')])
        self.assertEqual(counts['products'], 2)


class BuildTests(unittest.TestCase):
    def test_build_reads_training_products_only_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            meta = [row(f'p{i}', 'Boss' if i % 2 else None, Color='Black') for i in range(6)] + [row('held_out', 'Fender')]
            pq.write_table(pa.table({'parent_asin': [m[0] for m in meta], 'details': [m[1] for m in meta],
                                     'store': [m[2] for m in meta], 'average_rating': [4.0] * len(meta)}),
                           tmp / 'meta.parquet')
            (tmp / 'train.csv').write_text('user_id,parent_asin,rating,timestamp\n'
                                           + ''.join(f'u{i},p{i},5.0,{i}\n' for i in range(6)))
            manifest = kf.build(str(tmp / 'meta.parquet'), tmp / 'train.csv', tmp / 'out')
            brands, attributes, _ = kf.load_facts(tmp / 'out')
            self.assertEqual(brands, {'p1': 'Boss', 'p3': 'Boss', 'p5': 'Boss'})
            self.assertEqual(attributes, {f'p{i}': ['color=black'] for i in range(6)})
            self.assertEqual(manifest['counts']['products'], 6)
            with self.assertRaises(SystemExit):
                kf.build(str(tmp / 'meta.parquet'), tmp / 'train.csv', tmp / 'out')
            (tmp / 'out' / 'facts.parquet').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                kf.load_facts(tmp / 'out')


if __name__ == '__main__':
    unittest.main()
