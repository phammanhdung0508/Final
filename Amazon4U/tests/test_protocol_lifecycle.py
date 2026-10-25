"""Read-only regression checks for versioned protocol records; no model execution."""

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
V2_0_MANIFEST_SHA256 = "f1ce495923f58a15c1c6a0574b88e6b0d343ffad1594d66e7f752d39a50a6743"


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
        resolved = self.record["resolved_before_comparative_training_v2_1_0"]
        self.assertIn("explicit_comparative_training_authorization", pending)
        for choice in (
            "LightGCN_ID_initialization_distribution",
            "model_specific_batch_sizes",
            "validation_frequency_after_authorized_profiling",
            "tuning_search_spaces_trial_budgets_seed_and_reuse_policy",
        ):
            self.assertIn(choice, resolved)
            self.assertNotIn(choice, pending)
        semantics = json.loads((ROOT / "configs/training-semantics.json").read_text())
        recipe = semantics["learned_model_training_recipe"]
        self.assertEqual(recipe["learning_rate"], [0.0005, 0.001, 0.002])
        self.assertEqual(recipe["validation_frequency"], "every_epoch_full_ranking")
        self.assertEqual(recipe["batch_size"], {"LightGCN": 2048, "Heterogeneous KG+GNN": 1024})
        self.assertEqual(semantics["tuning"]["tuning_seed"], 42)
        self.assertEqual(semantics["execution_choices"]["LightGCN_ID_initialization"],
                         {"distribution": "normal", "mean": 0, "std": 0.1})
        self.assertFalse(self.record["training_authorized"])
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
        self.assertIn("LightGCN_ID_initialization_distribution", semantics["resolved_choices_v2_1_0"])
        self.assertEqual(semantics["pending_choices"], [])
        self.assertNotIn("BPR_MF", json.dumps(semantics))

    def test_v2_1_amendment_preserves_the_v2_0_baseline(self):
        self.assertEqual(self.record["protocol_version"], "amazon4u-protocol-v2.1.0")
        self.assertEqual(self.record["baseline_manifest"], "configs/protocols/v2.1.0/baseline-manifest.json")
        prior_path = ROOT / "configs/protocols/v2.0.0/baseline-manifest.json"
        self.assertEqual(hashlib.sha256(prior_path.read_bytes()).hexdigest(), V2_0_MANIFEST_SHA256)
        prior = json.loads(prior_path.read_text())
        for entry in prior["files"]:
            content = (prior_path.parent / entry["snapshot_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
        prior_semantics = json.loads((prior_path.parent / "snapshot/configs/training-semantics.json").read_text())
        self.assertEqual(prior_semantics["learned_model_training_recipe"]["learning_rate"], 0.001)
        current = json.loads((ROOT / self.record["baseline_manifest"]).read_text())
        self.assertEqual(current["predecessor_manifest"], "configs/protocols/v2.0.0/baseline-manifest.json")
        self.assertEqual(current["predecessor_manifest_sha256"], V2_0_MANIFEST_SHA256)
        self.assertFalse(current["heldout_model_results_exposed"])
        self.assertTrue({e["source_path"] for e in prior["files"]} <= {e["source_path"] for e in current["files"]})
        for entry in current["files"]:
            self.assertEqual((ROOT / entry["source_path"]).read_bytes(),
                             (ROOT / self.record["baseline_manifest"]).parent.joinpath(entry["snapshot_path"]).read_bytes())
        event = self.record["history"][-1]
        self.assertEqual((event["prior_protocol_version"], event["protocol_version"]),
                         ("amazon4u-protocol-v2.0.0", "amazon4u-protocol-v2.1.0"))
        self.assertFalse(event["heldout_model_results_exposed"])
        self.assertFalse(event["profiling_or_training_authorized_by_event"])

    def test_v2_amendment_preserves_original_frozen_baseline(self):
        old_path = ROOT / "configs/protocols/v1.0.0/baseline-manifest.json"
        self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(), "7a19eec510050417700c462ac502c41811bdedd9a541b5f527ee9d6eb144475a")
        old = json.loads(old_path.read_text())
        for entry in old["files"]:
            content = (old_path.parent / entry["snapshot_path"]).read_bytes()
            self.assertEqual(len(content), entry["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
        original_suite = json.loads((old_path.parent / "snapshot/configs/model-suite.json").read_text())
        self.assertIn("BPR-MF", original_suite["core_models"])
        new_path = ROOT / "configs/protocols/v2.0.0/baseline-manifest.json"
        new = json.loads(new_path.read_text())
        self.assertEqual(new["predecessor_manifest"], "configs/protocols/v1.0.0/baseline-manifest.json")
        self.assertEqual(new["predecessor_manifest_sha256"], hashlib.sha256(old_path.read_bytes()).hexdigest())
        self.assertFalse(new["heldout_model_results_exposed"])
        self.assertEqual(self.record["lifecycle_state"], "IMPLEMENTATION_REGRESSION_VERIFICATION")
        self.assertIsNone(self.record["execution_revision"])
        self.assertFalse(self.record["profiling_authorized"])
        self.assertFalse(self.record["training_authorized"])


if __name__ == "__main__":
    unittest.main()
