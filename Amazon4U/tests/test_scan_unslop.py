"""The prose scanner must refuse byte-frozen documents instead of inviting edits."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scan_unslop import frozen_documents


class FrozenDocumentTests(unittest.TestCase):
    def test_every_frozen_baseline_document_is_refused(self):
        frozen = frozen_documents()
        # The documents edited by the 2026-10-24 prose pass (commits 08fc438, 0143f83).
        for path in ("docs/evaluation-protocol.md", "docs/training-semantics.md",
                     "docs/heterogeneous-encoder.md", "docs/lightgcn-protocol.md",
                     "docs/checkpoint-payload.md"):
            self.assertIn(path, frozen)
        out = subprocess.run([sys.executable, str(ROOT / "scripts/scan_unslop.py"),
                              str(ROOT / "docs/evaluation-protocol.md")],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("FROZEN, not scanned", out)

    def test_ordinary_documents_are_still_scanned(self):
        self.assertNotIn("docs/gpu-ready-design.md", frozen_documents())


if __name__ == "__main__":
    unittest.main()
