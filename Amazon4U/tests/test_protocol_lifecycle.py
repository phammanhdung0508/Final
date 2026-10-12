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
            "LightGCN_ID_initialization_distribution",
            "model_specific_batch_sizes",
            "validation_frequency_after_authorized_profiling",
            "tuning_search_spaces_trial_budgets_seed_and_reuse_policy",
            "explicit_comparative_training_authorization",
        ):
            self.assertIn(choice, pending)
        self.assertNotIn("bootstrap_comparison_metric_plan", pending)
        bootstrap = json.loads((ROOT / "configs/reproducibility.json").read_text())["bootstrap"]
        self.assertEqual(len(bootstrap["predeclared_comparisons"]), 3)
        self.assertEqual(bootstrap["predeclared_metrics"], ["NDCG@10", "HitRate@10"])
        self.assertEqual(bootstrap["planned_interval_count"], 36)
        self.assertEqual(
            [c["right"] for c in bootstrap["predeclared_comparisons"]],
            ["Popularity", "LightGCN", "KG-only"],
        )
        self.assertTrue(all(c["left"] == "Heterogeneous KG+GNN" for c in bootstrap["predeclared_comparisons"]))
        self.assertEqual(bootstrap["planned_interval_count"], len(bootstrap["predeclared_comparisons"]) * len(bootstrap["predeclared_metrics"]) * 6)

    def test_removed_model_absent_but_shared_bpr_recipe_retained(self):
        suite = json.loads((ROOT / "configs/model-suite.json").read_text())
        self.assertEqual(suite["core_models"], ["Popularity", "LightGCN", "KG-only", "Heterogeneous KG+GNN"])
        self.assertEqual(set(suite["model_specifications"]), set(suite["core_models"]))
        self.assertNotIn("BPR-MF", json.dumps(suite))
        self.assertEqual([c["name"] for c in suite["optional_comparisons"]], ["Matched KG+GNN without metadata relations"])
        semantics = json.loads((ROOT / "configs/training-semantics.json").read_text())
        learned = ["LightGCN", "Heterogeneous KG+GNN"]
        self.assertEqual(semantics["learned_model_training_recipe"]["applies_to"], learned)
        self.assertEqual(semantics["regularization"]["applies_to"], learned)
        self.assertEqual(semantics["learned_model_training_recipe"]["loss"]["name"], "BPR")
        self.assertEqual(semantics["regularization"]["coefficient"], 1e-5)
        self.assertIn("LightGCN_ID_initialization_distribution", semantics["pending_choices"])
        self.assertNotIn("BPR_MF", json.dumps(semantics))

    def test_v2_amendment_preserves_original_frozen_baseline(self):
        self.assertEqual(self.record["protocol_version"], "amazon4u-protocol-v2.0.0")
        self.assertEqual(self.record["baseline_manifest"], "configs/protocols/v2.0.0/baseline-manifest.json")
        old_path = ROOT / "configs/protocols/v1.0.0/baseline-manifest.json"
        self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(), "7a19eec510050417700c462ac502c41811bdedd9a541b5f527ee9d6eb144475a")
        old = json.loads(old_path.read_text())
        for entry in old["files"]:
            content = (old_path.parent / entry["snapshot_path"]).read_bytes()
            self.assertEqual(len(content), entry["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
        original_suite = json.loads((old_path.parent / "snapshot/configs/model-suite.json").read_text())
        self.assertIn("BPR-MF", original_suite["core_models"])
        new_path = ROOT / self.record["baseline_manifest"]
        new = json.loads(new_path.read_text())
        self.assertEqual(new["predecessor_manifest"], "configs/protocols/v1.0.0/baseline-manifest.json")
        self.assertEqual(new["predecessor_manifest_sha256"], hashlib.sha256(old_path.read_bytes()).hexdigest())
        self.assertFalse(new["heldout_model_results_exposed"])
        for entry in new["files"]:
            self.assertEqual((ROOT / entry["source_path"]).read_bytes(), (new_path.parent / entry["snapshot_path"]).read_bytes())
        self.assertEqual(self.record["lifecycle_state"], "IMPLEMENTATION_REGRESSION_VERIFICATION")
        self.assertIsNone(self.record["execution_revision"])
        self.assertFalse(self.record["profiling_authorized"])
        self.assertFalse(self.record["training_authorized"])


if __name__ == "__main__":
    unittest.main()
