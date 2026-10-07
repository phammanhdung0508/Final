"""Regression checks for candidate counting; no trained models or full data."""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_candidates import run


class CandidateAuditTests(unittest.TestCase):
    def test_warmth_history_unknown_users_and_metadata_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw'
            csv_dir = raw / 'benchmark/5core/last_out'
            csv_dir.mkdir(parents=True)
            rows = {'train': [('u1', 'p1', 5, 100), ('u2', 'low', 2, 100)],
                    'valid': [('u1', 'p2', 5, 200), ('u2', 'low', 5, 200)],
                    'test': [('u1', 'p2', 5, 300), ('u2', 'cold', 5, 300), ('unknown', 'p1', 5, 300)]}
            for split, records in rows.items():
                with (csv_dir / f'Fixture.{split}.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['user_id', 'parent_asin', 'rating', 'timestamp'])
                    writer.writerows(records)
            folder = raw / 'raw_meta_Fixture'
            folder.mkdir()
            records = [{'parent_asin': p, 'categories': ['Root', leaf], 'details': json.dumps({'Brand': brand})}
                       for p, leaf, brand in [('p1', 'Known', 'Warm'), ('low', 'Known', 'Warm'),
                                              ('p2', 'New', 'NewBrand'), ('cold', 'Known', 'Warm')]]
            pq.write_table(pa.Table.from_pylist(records), folder / 'full.parquet')
            config = json.loads((ROOT / 'configs/kg-v1.json').read_text())
            config['categories'] = ['Fixture']
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            output = root / 'report.json'
            with contextlib.redirect_stdout(io.StringIO()):
                run(raw, path, output)
            report = json.loads(output.read_text())
            all_ratings = report['conditions']['all_ratings']['Fixture']
            positive = report['conditions']['rating_ge_4']['Fixture']
            self.assertEqual(all_ratings['warm_items'], 2)
            self.assertEqual(positive['warm_items'], 1)
            test = all_ratings['splits']['test']
            self.assertEqual(test['cold_rows'], 2)
            self.assertEqual(test['targets_in_filtered_history'], 1)
            self.assertEqual(test['users_absent_from_raw_training'], 1)
            self.assertEqual(test['eligible_warm_rows'], 0)
            self.assertEqual(all_ratings['cold_only_metadata_nodes'], 2)
            # Low training ratings still count as known history for filtering.
            self.assertEqual(positive['splits']['valid']['targets_in_filtered_history'], 1)
            self.assertEqual(positive['splits']['test']['users_without_positive_training_history'], 2)


if __name__ == '__main__':
    unittest.main()
