"""O-OFFLINE files keep their pinned hashes (configs/o-offline-frozen-pins.json).

H28 and D1 to D3 were already pinned; O-OFFLINE was not (user, 2026-10-24). A change
needs explicit approval and an amendment entry that ends at the new pinned hash.
"""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / 'configs/o-offline-frozen-pins.json').read_text())


class OOfflinePinTests(unittest.TestCase):
    def test_files_match_their_pins(self):
        for path, digest in PINS['sha256'].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_every_amendment_is_approved_and_ends_at_the_pin(self):
        for entry in PINS['amendments']:
            with self.subTest(path=entry['path']):
                self.assertEqual(entry['approved_by'], 'user')
                self.assertEqual(entry['new_sha256'], PINS['sha256'][entry['path']])
                self.assertNotEqual(entry['previous_sha256'], entry['new_sha256'])


if __name__ == '__main__':
    unittest.main()
