"""Read-only regression checks for versioned protocol records; no model execution."""

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProtocolLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / "configs/protocol-lifecycle.json").read_text())

    def test_version_and_state_are_separate(self):
        self.assertRegex(self.record["protocol_version"], r"^amazon4u-protocol-v\d+\.\d+\.\d+$")
        self.assertIn(self.record["lifecycle_state"], self.record["lifecycle_order"])
        self.assertFalse(self.record["version_policy"]["state_transition_changes_version"])
        self.assertFalse(self.record["version_policy"]["state_transition_grants_authorization"])
        self.assertEqual(len(set(self.record["lifecycle_order"])), len(self.record["lifecycle_order"]))

    def test_baseline_freeze_is_not_training_permission(self):
        if self.record["lifecycle_state"] == "BASELINE_PROTOCOL_FROZEN":
            self.assertIsNone(self.record["execution_revision"])
            self.assertFalse(self.record["profiling_authorized"])
            self.assertFalse(self.record["training_authorized"])
            self.assertIsNone(self.record["profiling_scope"])
            self.assertIsNone(self.record["comparative_training_scope"])
        self.assertEqual(self.record["deployment_decision"], "NOT_AUTHORIZED")
        for name in ("training-semantics", "compute-plan"):
            cfg = json.loads((ROOT / f"configs/{name}.json").read_text())
            self.assertEqual(cfg["training_authorized"], self.record["training_authorized"])
            self.assertEqual(cfg["lifecycle_reference"], "configs/protocol-lifecycle.json")
        compute = json.loads((ROOT / "configs/compute-plan.json").read_text())
        self.assertEqual(compute["profiling_authorized"], self.record["profiling_authorized"])

    def test_frozen_snapshot_hashes_and_paths(self):
        manifest_path = ROOT / self.record["baseline_manifest"]
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["protocol_version"], self.record["protocol_version"])
        self.assertIsNone(manifest["execution_revision"])
        self.assertFalse(manifest["profiling_authorized_by_freeze"])
        self.assertFalse(manifest["training_authorized_by_freeze"])
        self.assertGreater(len(manifest["files"]), 0)
        sources = set()
        for entry in manifest["files"]:
            source = Path(entry["source_path"])
            snapshot = Path(entry["snapshot_path"])
            self.assertFalse(source.is_absolute())
            self.assertFalse(snapshot.is_absolute())
            self.assertNotIn("..", source.parts)
            self.assertNotIn("..", snapshot.parts)
            self.assertIn(source.parts[0], ("configs", "docs"))
            self.assertEqual(snapshot.parts[0], "snapshot")
            self.assertNotIn(str(source), sources)
            sources.add(str(source))
            content = (manifest_path.parent / snapshot).read_bytes()
            self.assertEqual(len(content), entry["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
        self.assertNotIn("configs/protocol-lifecycle.json", sources)
        self.assertNotIn("configs/compute-plan.json", sources)

    def test_unresolved_execution_is_explicit(self):
        pending = self.record["pending_before_comparative_training"]
        for choice in (
            "BPR_MF_and_LightGCN_ID_initialization_distribution",
            "model_specific_batch_sizes",
            "validation_frequency_after_authorized_profiling",
            "tuning_search_spaces_trial_budgets_seed_and_reuse_policy",
            "explicit_comparative_training_authorization",
        ):
            self.assertIn(choice, pending)
        self.assertNotIn("bootstrap_comparison_metric_plan", pending)
        bootstrap = json.loads((ROOT / "configs/reproducibility.json").read_text())["bootstrap"]
        self.assertEqual(len(bootstrap["predeclared_comparisons"]), 4)
        self.assertEqual(bootstrap["predeclared_metrics"], ["NDCG@10", "HitRate@10"])
        self.assertEqual(bootstrap["planned_interval_count"], 48)


if __name__ == "__main__":
    unittest.main()
