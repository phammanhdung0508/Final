"""Unit, adversarial, and regression tests for pure checkpoint identity and provenance contracts.

Python standard library only. Zero PyTorch tensor allocations or optimizer steps.
Tests:
- Dual import canonicalization for checkpoint_identity and scripts.checkpoint_identity.
- Literal hand-calculated / fixed SHA-256 oracles for canonical identity and provenance.
- Complete roundtrip serialization (to_dict <-> from_dict) without loss or distortion.
- Deterministic ordering invariance under permutation of records, users, items, histories,
  graph nodes, edges, metadata, validation inputs, and settings.
- Delimiter collision adversaries: structured typed IDs prevent delimiter injection/confusion.
- Mutation matrix: sensitivity to every semantic component (method, category, condition, seed,
  record user/item/rating/timestamp, graph endpoints/weights/relations, metadata, validation targets,
  and settings parameters).
- Semantic consistency enforcement: condition positivity (rating >= 4 vs all_ratings),
  warm-pool completeness, positive history subset of all-rated history, and record agreement.
- Partial-record and history inconsistency detection across all record/history combinations.
- Rejection of duplicates: duplicate typed IDs, nodes, categories, edges, users, and metadata.
- Strict type validation: exact integer timestamps (rejection of float, bool, non-finite),
  finite ratings (rejection of bool, NaN, Inf), strict bool dirty flags, and integer seeds.
- Excluded metadata enforcement: rejection of average_rating, rating_number, store, Manufacturer,
  details, bestseller_rank, and bought_together.
- Graph contract enforcement: typed endpoints in declared node sets, relation schemas,
  valid 'c:[...]' CategoryPath IDs, and 'b:...' Brand IDs.
- Validation selection inputs: target/record agreement, missing/extra fields, duplicate pairs.
- Comprehensive resolved settings matrix: every single section and field tested for mutation,
  omission, extra unknown fields, invalid types (runtime_workers, rng_streams, worker_derivation),
  and rng_streams order sensitivity.
- Tuple vs list sequence interchangeability for rng_streams and neighbor_fanouts.
- Header padding / whitespace rejection on method, category, and lowercase 64-char hex fingerprints.
- Fingerprint recomputation and tamper detection: recomputing all fingerprints from retained
  actual content, detecting falsified fingerprints, and verifying caller expectations.
- Domain-specific exception handling: corrupt actual/expected objects fail with domain errors,
  never unhandled KeyError, TypeError, or AttributeError.
- Source provenance byte hashing, 2 MiB boundary enforcement (exact 2,097,152 bytes accepted, 2,097,153 bytes rejected),
  reconstructable bytes preservation, Git evidence (commit/dirty/diff/untracked), and environment records.
- Aggregate byte-bound rejection BEFORE any hashing/conversion using mock oracles for both builder and validator.
- Path security: strict allowlist, path traversal rejection ('..', '.'), absolute path rejection,
  credential pattern rejection, and disallowed directory rejection.
- Input/output aliasing isolation and post-mutation validation detection.
- Bounded synthetic fixtures: <= 8 users, <= 12 products, <= 8 metadata nodes per type.
"""

from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import checkpoint_identity
import scripts.checkpoint_identity

from checkpoint_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    MAX_PROVENANCE_SOURCE_BYTES,
    NODE_TYPES,
    PROVENANCE_SCHEMA_VERSION,
    REQUIRED_SETTINGS_FIELDS,
    REQUIRED_SETTINGS_SECTIONS,
    CanonicalResumeIdentity,
    CheckpointError,
    CheckpointIdentityMismatchError,
    CheckpointProvenanceError,
    CheckpointSecurityError,
    CheckpointValidationError,
    build_canonical_identity,
    build_source_provenance,
    compute_data_fingerprint,
    compute_graph_fingerprint,
    compute_settings_fingerprint,
    compute_validation_fingerprint,
    format_canonical_brand_id,
    format_canonical_category_id,
    validate_allowlisted_relative_path,
    validate_brand_id,
    validate_canonical_identity,
    validate_category_id,
    validate_source_provenance,
)


def _make_valid_settings_fixture() -> Dict[str, Any]:
    """Create a minimal, valid synthetic settings fixture covering all 6 mandatory sections."""
    return {
        "protocol": {
            "candidate_policy": "full_candidate_ranking",
            "lifecycle_state": "IMPLEMENTATION_REGRESSION_VERIFICATION",
            "protocol_version": "amazon4u-protocol-v2.0.0",
        },
        "execution": {
            "cpu_threads": 4,
            "execution_device": "cpu",
            "execution_label": "synthetic_contract_verification",
            "runtime_mode": "benchmark_verification",
            "runtime_workers": 0,
            "worker_settings": {
                "multiprocessing_context": None,
                "persistent_workers": False,
                "prefetch_factor": None,
                "worker_init_function": None,
            },
        },
        "model": {
            "architecture": "LightGCN",
            "embedding_dim": 64,
            "initialization": "normal_0_0.1_xavier_uniform_gain_1",
            "num_layers": 3,
        },
        "training": {
            "adam_betas": [0.9, 0.999],
            "adam_epsilon": 1e-08,
            "batch_size": 128,
            "endpoint_regularization": 1e-05,
            "learning_rate": 0.001,
            "loss_function": "BPR",
            "max_epochs": 100,
            "optimizer": "Adam",
            "patience": 10,
            "validation_frequency": 1,
            "weight_decay": 0.0,
        },
        "sampling": {
            "negative_strategy": "uniform_eligible",
            "negatives_per_positive": 1,
            "neighbor_fanouts": [15, 10],
            "pool_policy": "condition_warm_excluding_rated",
        },
        "determinism": {
            "deterministic_algorithms": True,
            "reproducibility_version": "amazon4u-reproducibility-v1",
            "rng_streams": [
                {
                    "name": "Python",
                    "applicable": True,
                    "algorithm": "stdlib_random_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "NumPy",
                    "applicable": True,
                    "algorithm": "numpy_pcg64_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "PyTorch_CPU",
                    "applicable": True,
                    "algorithm": "torch_cpu_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "PyTorch_CUDA_if_applicable",
                    "applicable": False,
                    "algorithm": None,
                    "seed": None,
                    "derivation": None,
                    "reason": "cuda_not_applicable_cpu_synthetic",
                },
                {
                    "name": "initialization",
                    "applicable": True,
                    "algorithm": "model_init_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "epoch_shuffle",
                    "applicable": True,
                    "algorithm": "shuffle_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "negative_sampler",
                    "applicable": True,
                    "algorithm": "uniform_neg_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "neighbor_sampler",
                    "applicable": True,
                    "algorithm": "fanout_sampler_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "DataLoader_generator",
                    "applicable": True,
                    "algorithm": "torch_generator_synthetic",
                    "seed": 42,
                    "derivation": "fixed_seed",
                    "reason": None,
                },
                {
                    "name": "DataLoader_workers",
                    "applicable": False,
                    "algorithm": None,
                    "seed": None,
                    "derivation": None,
                    "reason": "single_process_zero_workers_synthetic",
                },
            ],
            "seed": 42,
            "worker_derivation": {
                "applicable": False,
                "algorithm": None,
                "base_seed": None,
                "derivation": None,
                "reason": "single_process_zero_workers_synthetic",
            },
        },
    }


def _make_valid_identity_fixture() -> Dict[str, Any]:
    """Create a minimal, valid, complete synthetic identity fixture with <= 8 users, <= 12 items."""
    training_records = [
        {"user_id": "u1", "parent_asin": "p1", "rating": 4.0, "timestamp": 1000},
        {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
        {"user_id": "u2", "parent_asin": "p2", "rating": 3.0, "timestamp": 3000},
        {"user_id": "u2", "parent_asin": "p3", "rating": 4.0, "timestamp": 4000},
    ]
    positive_history = {
        "u1": ["p1", "p2"],
        "u2": ["p2", "p3"],
    }
    all_rated_history = {
        "u1": ["p1", "p2"],
        "u2": ["p2", "p3"],
    }
    warm_items = ["p1", "p2", "p3"]
    graph_nodes = {
        "User": ["u1", "u2"],
        "Product": ["p1", "p2", "p3"],
        "CategoryPath": ['c:["Electronics"]', 'c:["Electronics","Headphones"]'],
        "Brand": ["b:Sony"],
    }
    graph_edges = [
        ("User", "u1", "rated", "Product", "p1", 1.0),
        ("Product", "p1", "rated_by", "User", "u1", 1.0),
        ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics","Headphones"]', 1.0),
        ("CategoryPath", 'c:["Electronics","Headphones"]', "contains_product", "Product", "p1", 1.0),
        ("Product", "p1", "has_brand", "Brand", "b:Sony", 1.0),
        ("Brand", "b:Sony", "brand_of", "Product", "p1", 1.0),
        ("CategoryPath", 'c:["Electronics","Headphones"]', "category_child_of", "CategoryPath", 'c:["Electronics"]', 1.0),
        ("CategoryPath", 'c:["Electronics"]', "category_parent_of", "CategoryPath", 'c:["Electronics","Headphones"]', 1.0),
    ]
    product_metadata = {
        "p1": {"categories": ['c:["Electronics","Headphones"]'], "brand": "b:Sony"},
        "p2": {"categories": ['c:["Electronics"]'], "brand": None},
        "p3": {"categories": [], "brand": None},
    }
    validation_inputs = {
        "targets": {"u1": ["p3"], "u2": ["p1"]},
        "records": [("u1", "p3", 5.0, 5000), ("u2", "p1", 4.0, 6000)],
    }
    resolved_settings = _make_valid_settings_fixture()

    return {
        "method": "LightGCN",
        "category": "Musical_Instruments",
        "feedback_condition": "all_ratings",
        "seed": 42,
        "training_records": training_records,
        "positive_history": positive_history,
        "all_rated_history": all_rated_history,
        "warm_items": warm_items,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "product_metadata": product_metadata,
        "validation_inputs": validation_inputs,
        "resolved_settings": resolved_settings,
    }


def _make_valid_positive_worker_settings_fixture() -> Dict[str, Any]:
    """Create a minimal, valid synthetic settings fixture with positive workers (runtime_workers > 0)."""
    f = _make_valid_settings_fixture()
    f["execution"]["runtime_workers"] = 2
    f["execution"]["worker_settings"] = {
        "prefetch_factor": 2,
        "persistent_workers": True,
        "multiprocessing_context": "fork",
        "worker_init_function": "worker_init_fn",
    }
    f["determinism"]["worker_derivation"] = {
        "applicable": True,
        "algorithm": "worker_seed_fn",
        "base_seed": 42,
        "derivation": "base_plus_worker_id",
        "reason": None,
    }
    for stream in f["determinism"]["rng_streams"]:
        if stream["name"] == "DataLoader_workers":
            stream["applicable"] = True
            stream["algorithm"] = "worker_stream_synthetic"
            stream["seed"] = 1042
            stream["derivation"] = "offset_seed"
            stream["reason"] = None
            break
    return f


def _make_valid_positive_worker_identity_fixture() -> Dict[str, Any]:
    """Create an otherwise-valid complete synthetic identity fixture with positive workers and all bindings consistent."""
    f = _make_valid_identity_fixture()
    f["resolved_settings"] = _make_valid_positive_worker_settings_fixture()
    return f


def _make_valid_provenance_fixture() -> Tuple[Dict[str, bytes], Dict[str, Any], Dict[str, Any]]:
    """Create a minimal, valid synthetic source provenance fixture."""
    source_files = {
        "scripts/checkpoint_identity.py": b"# mock content for checkpoint identity\nprint(1)\n",
        "configs/training-semantics.json": b'{"version": "v2"}\n',
    }
    git_evidence = {
        "commit": "fd15dc3b5fe665f45d8787ba779b9e9541690bde",
        "dirty": False,
        "tracked_diff": "",
        "untracked_sources": [],
    }
    environment_records = {
        "python_version": "3.10.12",
        "platform_system": "Linux",
        "platform_release": "5.15.0",
        "platform_machine": "x86_64",
        "cpu_count": 8,
        "torch_version": "2.2.0",
        "library_versions": {"torch": "2.2.0", "duckdb": "1.0.0"},
    }
    return source_files, git_evidence, environment_records


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class CanonicalIdentityContractTests(unittest.TestCase):
    """Core contract tests for canonical resume identity."""

    def test_dual_import_canonicalization(self):
        self.assertIs(checkpoint_identity, scripts.checkpoint_identity)
        self.assertIs(
            checkpoint_identity.CanonicalResumeIdentity,
            scripts.checkpoint_identity.CanonicalResumeIdentity,
        )

    def test_literal_canonical_identity_oracles(self):
        fixture = _make_valid_identity_fixture()
        ident = build_canonical_identity(**fixture)

        # Exact, literal hand-calculated / verified SHA-256 oracles
        expected_dfp = "e3bbac61d597e55a203a1c6603696d3973380d033fb25a0dfda37a0645b05056"
        expected_gfp = "6aeccf0475a17517e592aba5775660cf617fe8ddd3ad4b8ac0e1b8845e49156b"
        expected_vfp = "dc524e535dbe28cd5edca98f1e18f2045e078ddaef1c9bdb458bb86c58471afe"
        # Settings fingerprint is SHA-256 of canonical resolved settings payload containing all 6 complete sections:
        expected_sfp = "c71aa5ce98966d9ba2599b08a920a28e54e22f35dc15a7ffad4a7110550cab82"
        # Composite identity hash is SHA-256 of canonical identity payload:
        expected_id_hash = "1bdaea12c95868dd3a5b0159193c448e64834f0acfedb32bbabbee0e62ab13a8"

        self.assertEqual(ident.data_fingerprint, expected_dfp)
        self.assertEqual(ident.graph_fingerprint, expected_gfp)
        self.assertEqual(ident.validation_fingerprint, expected_vfp)
        self.assertEqual(ident.settings_fingerprint, expected_sfp)
        self.assertEqual(ident.identity_hash, expected_id_hash)

        # Validation against self must pass
        validate_canonical_identity(ident, ident)

    def test_roundtrip_to_dict_and_from_dict(self):
        fixture = _make_valid_identity_fixture()
        ident = build_canonical_identity(**fixture)

        d = ident.to_dict()
        self.assertEqual(d["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(d["method"], "LightGCN")
        self.assertEqual(d["identity_hash"], ident.identity_hash)

        reconstructed = CanonicalResumeIdentity.from_dict(d)
        validate_canonical_identity(ident, reconstructed)
        self.assertEqual(ident, reconstructed)

        # from_dict must validate all keys are strings BEFORE schema/key diagnostics
        # Otherwise-valid extra keys 0 and 'unknown' must raise domain CheckpointValidationError, not TypeError
        d_mixed = ident.to_dict()
        d_mixed[0] = "extra_int_key"
        d_mixed["unknown"] = "extra_str_key"
        with self.assertRaises(CheckpointValidationError):
            CanonicalResumeIdentity.from_dict(d_mixed)

        # Non-dict data
        for bad_data in ("not_a_dict", [1, 2, 3], None):
            with self.assertRaises(CheckpointValidationError):
                CanonicalResumeIdentity.from_dict(bad_data)

        # Wrong schema version
        d_bad_schema = ident.to_dict()
        d_bad_schema["schema_version"] = "wrong-schema-v2"
        with self.assertRaises(CheckpointValidationError):
            CanonicalResumeIdentity.from_dict(d_bad_schema)

        # Missing key
        d_missing = ident.to_dict()
        del d_missing["seed"]
        with self.assertRaises(CheckpointValidationError):
            CanonicalResumeIdentity.from_dict(d_missing)

    def test_from_dict_and_serialized_matrix(self):
        f = _make_valid_identity_fixture()
        ident = build_canonical_identity(**f)
        base_dict = ident.to_dict()

        # 1. Wrong / missing schema
        bad_schema = copy.deepcopy(base_dict)
        bad_schema["schema_version"] = "wrong-schema-v9"
        with self.assertRaisesRegex(CheckpointValidationError, r"Schema version mismatch: expected 'amazon4u-checkpoint-identity-v1'"):
            CanonicalResumeIdentity.from_dict(bad_schema)

        missing_schema = copy.deepcopy(base_dict)
        del missing_schema["schema_version"]
        with self.assertRaisesRegex(CheckpointValidationError, r"Schema version mismatch"):
            CanonicalResumeIdentity.from_dict(missing_schema)

        # 2. Unknown string key
        bad_unk_key = copy.deepcopy(base_dict)
        bad_unk_key["unknown_key"] = "extra"
        with self.assertRaisesRegex(CheckpointValidationError, r"CanonicalResumeIdentity keys mismatch.*Extra: \['unknown_key'\]"):
            CanonicalResumeIdentity.from_dict(bad_unk_key)

        # 3. Mixed keys: 0 + unknown
        bad_mixed = copy.deepcopy(base_dict)
        bad_mixed[0] = "zero_key"
        bad_mixed["unknown"] = "extra"
        with self.assertRaisesRegex(CheckpointValidationError, r"CanonicalResumeIdentity dictionary keys must be strings.*0"):
            CanonicalResumeIdentity.from_dict(bad_mixed)

        # 4. Missing EACH of the 19 retained fields
        all_19_fields = (
            "schema_version",
            "method",
            "category",
            "feedback_condition",
            "seed",
            "data_fingerprint",
            "graph_fingerprint",
            "validation_fingerprint",
            "settings_fingerprint",
            "identity_hash",
            "training_records",
            "positive_history",
            "all_rated_history",
            "warm_items",
            "graph_nodes",
            "graph_edges",
            "product_metadata",
            "validation_inputs",
            "resolved_settings",
        )
        for field in all_19_fields:
            with self.subTest(missing_field=field):
                bad_d = copy.deepcopy(base_dict)
                del bad_d[field]
                if field == "schema_version":
                    with self.assertRaisesRegex(CheckpointValidationError, r"Schema version mismatch"):
                        CanonicalResumeIdentity.from_dict(bad_d)
                else:
                    with self.assertRaisesRegex(CheckpointValidationError, rf"CanonicalResumeIdentity keys mismatch.*Missing: \['{field}'\]"):
                        CanonicalResumeIdentity.from_dict(bad_d)

        # 5. Malformed retained collection / nested types
        for col_field, bad_val in (
            ("training_records", {"not": "sequence"}),
            ("training_records", 123),
            ("positive_history", ["not_a_mapping"]),
            ("positive_history", 123),
            ("all_rated_history", ["not_a_mapping"]),
            ("all_rated_history", 123),
            ("warm_items", {"not": "sequence"}),
            ("warm_items", 123),
            ("graph_nodes", ["not_a_mapping"]),
            ("graph_nodes", 123),
            ("graph_edges", {"not": "sequence"}),
            ("graph_edges", 123),
            ("product_metadata", ["not_a_mapping"]),
            ("product_metadata", 123),
            ("validation_inputs", ["not_a_mapping"]),
            ("validation_inputs", 123),
            ("resolved_settings", ["not_a_mapping"]),
            ("resolved_settings", 123),
        ):
            with self.subTest(col_field=col_field, bad_type=type(bad_val).__name__):
                bad_d = copy.deepcopy(base_dict)
                bad_d[col_field] = bad_val
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"{col_field} in serialized identity must be a (?:sequence|mapping), got {type(bad_val).__name__}",
                ):
                    CanonicalResumeIdentity.from_dict(bad_d)

        # 6. EACH falsified stored 64hex fingerprint/hash with intended diagnostic
        falsified_fp = "0" * 64
        for fp_field in (
            "data_fingerprint",
            "graph_fingerprint",
            "validation_fingerprint",
            "settings_fingerprint",
            "identity_hash",
        ):
            with self.subTest(falsified_fp=fp_field):
                bad_d = copy.deepcopy(base_dict)
                bad_d[fp_field] = falsified_fp
                expected_recomputed = getattr(ident, fp_field)
                with self.assertRaises(CheckpointIdentityMismatchError) as ctx:
                    CanonicalResumeIdentity.from_dict(bad_d)
                msg = str(ctx.exception)
                self.assertIn(f"Falsified or corrupt {fp_field} in serialized identity", msg)
                self.assertIn(falsified_fp, msg)
                self.assertIn(expected_recomputed, msg)

    def test_header_padding_and_whitespace_rejection(self):
        f = _make_valid_identity_fixture()

        # Method whitespace / empty / non-string
        for bad_m in (" LightGCN", "LightGCN ", "LightGCN\n", "   ", "", 123, None):
            bad_f = copy.deepcopy(f)
            bad_f["method"] = bad_m
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject method {bad_m!r}"):
                build_canonical_identity(**bad_f)

        # Category whitespace / empty / non-string
        for bad_c in (" Musical_Instruments", "Musical_Instruments ", "   ", "", 123, None):
            bad_f = copy.deepcopy(f)
            bad_f["category"] = bad_c
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject category {bad_c!r}"):
                build_canonical_identity(**bad_f)

        # Feedback condition whitespace / invalid
        for bad_fc in (" all_ratings", "all_ratings ", "rating_ge_4\t", "unknown_condition", 123):
            bad_f = copy.deepcopy(f)
            bad_f["feedback_condition"] = bad_fc
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject feedback_condition {bad_fc!r}"):
                build_canonical_identity(**bad_f)

        # Fingerprint whitespace, uppercase, invalid length, invalid chars in constructor & from_dict
        ident = build_canonical_identity(**f)
        d = ident.to_dict()

        for fp_field in ("data_fingerprint", "graph_fingerprint", "validation_fingerprint", "settings_fingerprint", "identity_hash"):
            for bad_fp in (
                " " + "a" * 64,
                "a" * 64 + " ",
                "A" * 64,
                "a" * 63 + "F",
                "a" * 63,
                "a" * 65,
                "g" * 64,
                12345,
            ):
                bad_d = copy.deepcopy(d)
                bad_d[fp_field] = bad_fp
                with self.assertRaises(CheckpointValidationError, msg=f"from_dict should reject {fp_field}={bad_fp!r}"):
                    CanonicalResumeIdentity.from_dict(bad_d)

                bad_kwargs = dict(
                    method=ident.method,
                    category=ident.category,
                    feedback_condition=ident.feedback_condition,
                    seed=ident.seed,
                    data_fingerprint=bad_fp if fp_field == "data_fingerprint" else ident.data_fingerprint,
                    graph_fingerprint=bad_fp if fp_field == "graph_fingerprint" else ident.graph_fingerprint,
                    validation_fingerprint=bad_fp if fp_field == "validation_fingerprint" else ident.validation_fingerprint,
                    settings_fingerprint=bad_fp if fp_field == "settings_fingerprint" else ident.settings_fingerprint,
                    identity_hash=bad_fp if fp_field == "identity_hash" else ident.identity_hash,
                    training_records=ident.training_records,
                    positive_history=ident.positive_history,
                    all_rated_history=ident.all_rated_history,
                    warm_items=ident.warm_items,
                    graph_nodes=ident.graph_nodes,
                    graph_edges=ident.graph_edges,
                    product_metadata=ident.product_metadata,
                    validation_inputs=ident.validation_inputs,
                    resolved_settings=ident.resolved_settings,
                )
                with self.assertRaises(CheckpointValidationError, msg=f"Constructor should reject {fp_field}={bad_fp!r}"):
                    CanonicalResumeIdentity(**bad_kwargs)

        # Header padding rejected in both actual and expected for validate_canonical_identity
        corrupt_act = copy.deepcopy(ident)
        object.__setattr__(corrupt_act, "method", " LightGCN")
        with self.assertRaises(CheckpointValidationError):
            validate_canonical_identity(corrupt_act, ident)

        corrupt_exp = copy.deepcopy(ident)
        object.__setattr__(corrupt_exp, "category", "Musical_Instruments ")
        with self.assertRaises(CheckpointValidationError):
            validate_canonical_identity(ident, corrupt_exp)

    def test_tuple_vs_list_sequence_interchangeability(self):
        f = _make_valid_identity_fixture()

        # Pass tuples instead of lists for sequence settings
        f["resolved_settings"]["sampling"]["neighbor_fanouts"] = (15, 10)
        f["resolved_settings"]["determinism"]["rng_streams"] = tuple(f["resolved_settings"]["determinism"]["rng_streams"])
        f["resolved_settings"]["training"]["adam_betas"] = (0.9, 0.999)

        ident_tuple = build_canonical_identity(**f)

        f_list = _make_valid_identity_fixture()
        ident_list = build_canonical_identity(**f_list)

        self.assertEqual(ident_tuple.settings_fingerprint, ident_list.settings_fingerprint)
        self.assertEqual(ident_tuple.identity_hash, ident_list.identity_hash)
        validate_canonical_identity(ident_tuple, ident_list)

    def test_delimiter_collision_adversaries(self):
        # 1. Delimiter in user vs product: user "u:1" item "p" vs user "u" item "1:p"
        dfp1 = compute_data_fingerprint(
            feedback_condition="all_ratings",
            users=["u:1", "u2"],
            training_records=[
                {"user_id": "u:1", "parent_asin": "p", "rating": 5.0, "timestamp": 1000},
                {"user_id": "u2", "parent_asin": "p", "rating": 5.0, "timestamp": 2000},
            ],
            positive_history={"u:1": ["p"], "u2": ["p"]},
            all_rated_history={"u:1": ["p"], "u2": ["p"]},
            warm_items=["p"],
        )

        dfp2 = compute_data_fingerprint(
            feedback_condition="all_ratings",
            users=["u", "u2"],
            training_records=[
                {"user_id": "u", "parent_asin": "1:p", "rating": 5.0, "timestamp": 1000},
                {"user_id": "u2", "parent_asin": "1:p", "rating": 5.0, "timestamp": 2000},
            ],
            positive_history={"u": ["1:p"], "u2": ["1:p"]},
            all_rated_history={"u": ["1:p"], "u2": ["1:p"]},
            warm_items=["1:p"],
        )

        self.assertNotEqual(dfp1, dfp2, "Structured typed IDs must prevent delimiter collision between user and product IDs.")

        # 2. Delimiter in CategoryPath: ["A > B"] vs ["A", "B"]
        cat_single = 'c:["A > B"]'
        cat_double = 'c:["A","B"]'
        validate_category_id(cat_single)
        validate_category_id(cat_double)

        gfp1 = compute_graph_fingerprint(
            graph_nodes={"User": ["u1"], "Product": ["p1"], "CategoryPath": [cat_single]},
            graph_edges=[("Product", "p1", "belongs_to", "CategoryPath", cat_single, 1.0)],
            product_metadata={"p1": {"categories": [cat_single], "brand": None}},
        )
        gfp2 = compute_graph_fingerprint(
            graph_nodes={"User": ["u1"], "Product": ["p1"], "CategoryPath": [cat_double]},
            graph_edges=[("Product", "p1", "belongs_to", "CategoryPath", cat_double, 1.0)],
            product_metadata={"p1": {"categories": [cat_double], "brand": None}},
        )
        self.assertNotEqual(gfp1, gfp2, "CategoryPath JSON prefix must prevent collision between label with ' > ' and nested labels.")

    def test_mutation_matrix_detects_all_single_component_mutations(self):
        base_f = _make_valid_identity_fixture()
        base_ident = build_canonical_identity(**base_f)

        # 1. Method mutated
        m_f = copy.deepcopy(base_f)
        m_f["method"] = "Heterogeneous KG+GNN"
        m_ident = build_canonical_identity(**m_f)
        self.assertNotEqual(base_ident.identity_hash, m_ident.identity_hash)
        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(m_ident, base_ident)

        # 2. Category mutated
        c_f = copy.deepcopy(base_f)
        c_f["category"] = "Toys_and_Games"
        c_ident = build_canonical_identity(**c_f)
        self.assertNotEqual(base_ident.identity_hash, c_ident.identity_hash)
        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(c_ident, base_ident)

        # 3. Seed mutated
        s_f = copy.deepcopy(base_f)
        s_f["seed"] = 2026
        s_ident = build_canonical_identity(**s_f)
        self.assertNotEqual(base_ident.identity_hash, s_ident.identity_hash)
        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(s_ident, base_ident)

        # 4. Rating mutated
        r_f = copy.deepcopy(base_f)
        r_f["training_records"][0]["rating"] = 5.0
        r_ident = build_canonical_identity(**r_f)
        self.assertNotEqual(base_ident.data_fingerprint, r_ident.data_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, r_ident.identity_hash)

        # 5. Timestamp mutated
        t_f = copy.deepcopy(base_f)
        t_f["training_records"][0]["timestamp"] = 9999
        t_ident = build_canonical_identity(**t_f)
        self.assertNotEqual(base_ident.data_fingerprint, t_ident.data_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, t_ident.identity_hash)

        # 6. Graph edge weight mutated
        gw_f = copy.deepcopy(base_f)
        gw_f["graph_edges"][0] = ("User", "u1", "rated", "Product", "p1", 0.5)
        gw_ident = build_canonical_identity(**gw_f)
        self.assertNotEqual(base_ident.graph_fingerprint, gw_ident.graph_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, gw_ident.identity_hash)

        # 7. Validation target mutated (consistently mutating both target and matching record)
        vt_f = copy.deepcopy(base_f)
        vt_f["validation_inputs"]["targets"]["u1"] = ["p2"]
        vt_f["validation_inputs"]["records"][0] = ("u1", "p2", 5.0, 5000)
        vt_ident = build_canonical_identity(**vt_f)
        self.assertNotEqual(base_ident.validation_fingerprint, vt_ident.validation_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, vt_ident.identity_hash)

        # 8. Settings learning_rate mutated
        set_f = copy.deepcopy(base_f)
        set_f["resolved_settings"]["training"]["learning_rate"] = 0.002
        set_ident = build_canonical_identity(**set_f)
        self.assertNotEqual(base_ident.settings_fingerprint, set_ident.settings_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, set_ident.identity_hash)

    def test_same_count_mutations_and_full_ordering_and_roundtrip(self):
        base_f = _make_valid_identity_fixture()
        base_ident = build_canonical_identity(**base_f)

        # 1. ISOLATED equal-count metadata mutation: change ONLY metadata, keeping graph nodes/edges constant
        iso_meta_f = copy.deepcopy(base_f)
        iso_meta_f["product_metadata"]["p1"]["categories"] = ['c:["Electronics"]']
        iso_meta_f["product_metadata"]["p2"]["categories"] = ['c:["Electronics","Headphones"]']
        iso_meta_ident = build_canonical_identity(**iso_meta_f)
        self.assertNotEqual(base_ident.graph_fingerprint, iso_meta_ident.graph_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, iso_meta_ident.identity_hash)
        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(iso_meta_ident, base_ident)

        # 2. Metadata brand swap with corresponding graph edge update
        meta_f = copy.deepcopy(base_f)
        meta_f["product_metadata"]["p1"]["brand"] = None
        meta_f["product_metadata"]["p2"]["brand"] = "b:Sony"
        new_edges = []
        for e in meta_f["graph_edges"]:
            if e[:4] == ("Product", "p1", "has_brand", "Brand"):
                new_edges.append(("Product", "p2", "has_brand", "Brand", "b:Sony", 1.0))
            elif e[:4] == ("Brand", "b:Sony", "brand_of", "Product"):
                new_edges.append(("Brand", "b:Sony", "brand_of", "Product", "p2", 1.0))
            else:
                new_edges.append(e)
        meta_f["graph_edges"] = new_edges
        meta_ident = build_canonical_identity(**meta_f)
        self.assertNotEqual(base_ident.graph_fingerprint, meta_ident.graph_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, meta_ident.identity_hash)

        # 3. SEPARATE graph-only edge weight mutation (metadata and graph nodes constant)
        edge_f = copy.deepcopy(base_f)
        edge_f["graph_edges"][0] = ("User", "u1", "rated", "Product", "p1", 0.5)
        edge_f["graph_edges"][1] = ("Product", "p1", "rated_by", "User", "u1", 0.5)
        edge_ident = build_canonical_identity(**edge_f)
        self.assertNotEqual(base_ident.graph_fingerprint, edge_ident.graph_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, edge_ident.identity_hash)
        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(edge_ident, base_ident)

        # 4. SEPARATE validation-only mutation (data, graph, and metadata constant)
        val_f = copy.deepcopy(base_f)
        val_f["validation_inputs"]["records"][0] = ("u1", "p3", 5.0, 9999)
        val_ident = build_canonical_identity(**val_f)
        self.assertNotEqual(base_ident.validation_fingerprint, val_ident.validation_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, val_ident.identity_hash)
        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(val_ident, base_ident)

        # 5. Same-count training record ratings swap: swap rating between record 0 and 1
        swap_f = copy.deepcopy(base_f)
        swap_f["training_records"][0]["rating"] = 5.0
        swap_f["training_records"][1]["rating"] = 4.0
        swap_ident = build_canonical_identity(**swap_f)
        self.assertNotEqual(base_ident.data_fingerprint, swap_ident.data_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, swap_ident.identity_hash)

        # Full ordering invariance across all collections
        perm_f = copy.deepcopy(base_f)
        perm_f["training_records"] = list(reversed(base_f["training_records"]))
        perm_f["positive_history"] = {
            "u2": list(reversed(base_f["positive_history"]["u2"])),
            "u1": list(reversed(base_f["positive_history"]["u1"])),
        }
        perm_f["all_rated_history"] = {
            "u2": list(reversed(base_f["all_rated_history"]["u2"])),
            "u1": list(reversed(base_f["all_rated_history"]["u1"])),
        }
        perm_f["warm_items"] = list(reversed(base_f["warm_items"]))
        perm_f["graph_nodes"] = {
            "Brand": list(base_f["graph_nodes"]["Brand"]),
            "Product": list(reversed(base_f["graph_nodes"]["Product"])),
            "User": list(reversed(base_f["graph_nodes"]["User"])),
            "CategoryPath": list(reversed(base_f["graph_nodes"]["CategoryPath"])),
        }
        perm_f["graph_edges"] = list(reversed(base_f["graph_edges"]))
        perm_f["product_metadata"] = {
            "p3": base_f["product_metadata"]["p3"],
            "p1": base_f["product_metadata"]["p1"],
            "p2": base_f["product_metadata"]["p2"],
        }
        perm_f["validation_inputs"] = {
            "records": list(reversed(base_f["validation_inputs"]["records"])),
            "targets": {
                "u2": list(base_f["validation_inputs"]["targets"]["u2"]),
                "u1": list(base_f["validation_inputs"]["targets"]["u1"]),
            },
        }

        perm_ident = build_canonical_identity(**perm_f)
        self.assertEqual(base_ident.data_fingerprint, perm_ident.data_fingerprint)
        self.assertEqual(base_ident.graph_fingerprint, perm_ident.graph_fingerprint)
        self.assertEqual(base_ident.validation_fingerprint, perm_ident.validation_fingerprint)
        self.assertEqual(base_ident.settings_fingerprint, perm_ident.settings_fingerprint)
        self.assertEqual(base_ident.identity_hash, perm_ident.identity_hash)
        self.assertEqual(base_ident, perm_ident)
        validate_canonical_identity(base_ident, perm_ident)

    def test_multi_category_metadata_ordering_and_sequence_equivalence(self):
        f1 = _make_valid_identity_fixture()
        f1["product_metadata"]["p1"]["categories"] = [
            'c:["Electronics","Headphones"]',
            'c:["Electronics"]',
        ]
        ident1 = build_canonical_identity(**f1)

        f2 = _make_valid_identity_fixture()
        f2["product_metadata"]["p1"]["categories"] = [
            'c:["Electronics"]',
            'c:["Electronics","Headphones"]',
        ]
        ident2 = build_canonical_identity(**f2)

        self.assertEqual(ident1.graph_fingerprint, ident2.graph_fingerprint)
        self.assertEqual(ident1.identity_hash, ident2.identity_hash)
        validate_canonical_identity(ident1, ident2)
        validate_canonical_identity(ident2, ident1)

        # to_dict and from_dict roundtrip equivalence
        d1 = ident1.to_dict()
        d2 = ident2.to_dict()
        self.assertEqual(d1, d2)
        reconstructed1 = CanonicalResumeIdentity.from_dict(d1)
        reconstructed2 = CanonicalResumeIdentity.from_dict(d2)
        self.assertEqual(reconstructed1, reconstructed2)
        validate_canonical_identity(reconstructed1, reconstructed2)


class CanonicalIdentitySemanticValidationTests(unittest.TestCase):
    """Validation tests for protocol semantics, consistency rules, and failure modes."""

    def test_invalid_unicode_domain_errors_and_numeric_overflow(self):
        f = _make_valid_identity_fixture()
        base_ident = build_canonical_identity(**f)

        # 1. Lone surrogates in validate_category_id and format_canonical_category_id
        for bad_cat in ('c:["\ud800"]', 'c:["\udfff"]', 'c:["prefix", "\ud800"]'):
            with self.subTest(bad_cat=bad_cat):
                with self.assertRaisesRegex(CheckpointValidationError, r"CategoryPath ID contains invalid Unicode / lone surrogates"):
                    validate_category_id(bad_cat)

        for bad_labels in (["\ud800"], ["\udfff"], ["Electronics", "\ud800"]):
            with self.subTest(bad_labels=bad_labels):
                with self.assertRaisesRegex(CheckpointValidationError, r"contains invalid Unicode / lone surrogates"):
                    format_canonical_category_id(bad_labels)

        # Escaped aliases rejected
        for escaped_cat in ('c:["\\u00e9"]', 'c:["\\u96ea"]', 'c:["\\ud800"]'):
            with self.subTest(escaped_cat=escaped_cat):
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"(?:CategoryPath ID is not in canonical compact JSON representation|contains invalid Unicode / lone surrogates)",
                ):
                    validate_category_id(escaped_cat)

        # 2. Malformed / lone surrogate text across builder, from_dict, actual and expected
        # A. Bad user ID
        for bad_u in ("u1\ud800", " u1", 123):
            with self.subTest(role="builder", bad_u=bad_u):
                bad_f = copy.deepcopy(f)
                bad_f["positive_history"] = {bad_u if k == "u1" else k: v for k, v in f["positive_history"].items()}
                bad_f["all_rated_history"] = {bad_u if k == "u1" else k: v for k, v in f["all_rated_history"].items()}
                bad_f["training_records"] = [
                    dict(r, user_id=bad_u) if r["user_id"] == "u1" else r for r in f["training_records"]
                ]
                bad_f["graph_nodes"]["User"] = [bad_u if u == "u1" else u for u in f["graph_nodes"]["User"]]
                bad_f["graph_edges"] = [
                    (e[0], bad_u, e[2], e[3], e[4], e[5]) if e[0] == "User" and e[1] == "u1" else (
                        (e[0], e[1], e[2], e[3], bad_u, e[5]) if e[3] == "User" and e[4] == "u1" else e
                    )
                    for e in f["graph_edges"]
                ]
                bad_f["validation_inputs"]["targets"] = {bad_u if k == "u1" else k: v for k, v in f["validation_inputs"]["targets"].items()}
                bad_f["validation_inputs"]["records"] = [
                    (bad_u, r[1], r[2], r[3]) if r[0] == "u1" else r for r in f["validation_inputs"]["records"]
                ]
                with self.assertRaisesRegex(CheckpointValidationError, r"(?i)(user ID|user_id)"):
                    build_canonical_identity(**bad_f)

            with self.subTest(role="from_dict", bad_u=bad_u):
                d = base_ident.to_dict()
                d["positive_history"] = {bad_u if k == "u1" else k: v for k, v in d["positive_history"].items()}
                d["all_rated_history"] = {bad_u if k == "u1" else k: v for k, v in d["all_rated_history"].items()}
                d["training_records"] = [
                    dict(r, user_id=bad_u) if r["user_id"] == "u1" else r for r in d["training_records"]
                ]
                d["graph_nodes"]["User"] = [bad_u if u == "u1" else u for u in d["graph_nodes"]["User"]]
                d["validation_inputs"]["targets"] = {bad_u if k == "u1" else k: v for k, v in d["validation_inputs"]["targets"].items()}
                d["validation_inputs"]["records"] = [
                    dict(r, user_id=bad_u) if r["user_id"] == "u1" else r for r in d["validation_inputs"]["records"]
                ]
                with self.assertRaisesRegex(CheckpointValidationError, r"(?i)(user ID|user_id)"):
                    CanonicalResumeIdentity.from_dict(d)

            with self.subTest(role="validator_actual", bad_u=bad_u):
                corrupt_act = copy.deepcopy(base_ident)
                object.__setattr__(corrupt_act, "positive_history", {bad_u if k == "u1" else k: v for k, v in base_ident.positive_history.items()})
                with self.assertRaisesRegex(CheckpointValidationError, r"actual\.positive_history key"):
                    validate_canonical_identity(corrupt_act, base_ident)

            with self.subTest(role="validator_expected", bad_u=bad_u):
                corrupt_exp = copy.deepcopy(base_ident)
                object.__setattr__(corrupt_exp, "positive_history", {bad_u if k == "u1" else k: v for k, v in base_ident.positive_history.items()})
                with self.assertRaisesRegex(CheckpointValidationError, r"expected\.positive_history key"):
                    validate_canonical_identity(base_ident, corrupt_exp)

        # B. Bad item ID
        for bad_p in ("p1\ud800", "p1 ", 123):
            with self.subTest(bad_p=bad_p):
                bad_f = copy.deepcopy(f)
                bad_f["warm_items"] = [bad_p if p == "p1" else p for p in f["warm_items"]]
                bad_f["graph_nodes"]["Product"] = [bad_p if p == "p1" else p for p in f["graph_nodes"]["Product"]]
                bad_f["positive_history"]["u1"] = [bad_p if p == "p1" else p for p in f["positive_history"]["u1"]]
                bad_f["all_rated_history"]["u1"] = [bad_p if p == "p1" else p for p in f["all_rated_history"]["u1"]]
                bad_f["training_records"] = [
                    dict(r, parent_asin=bad_p) if r["parent_asin"] == "p1" else r for r in f["training_records"]
                ]
                with self.assertRaisesRegex(CheckpointValidationError, r"item ID in warm_items"):
                    build_canonical_identity(**bad_f)

        # C. Bad category ID in graph & metadata
        bad_c_f = copy.deepcopy(f)
        bad_c_f["graph_nodes"]["CategoryPath"].append('c:["\ud800"]')
        with self.assertRaisesRegex(CheckpointValidationError, r"contains invalid Unicode / lone surrogates"):
            build_canonical_identity(**bad_c_f)

        # D. Bad brand ID in graph & metadata
        bad_b_f = copy.deepcopy(f)
        bad_b_f["graph_nodes"]["Brand"].append("b:Sony\ud800")
        with self.assertRaisesRegex(CheckpointValidationError, r"contains invalid Unicode / lone surrogates"):
            build_canonical_identity(**bad_b_f)

        # E. Bad header text (method, category)
        for field, bad_val in (("method", "LightGCN\ud800"), ("category", "Musical_Instruments\ud800")):
            with self.subTest(field=field, bad_val=bad_val):
                bad_h_f = copy.deepcopy(f)
                bad_h_f[field] = bad_val
                with self.assertRaisesRegex(CheckpointValidationError, rf"Invalid build_canonical_identity\.{field}.*lone surrogates"):
                    build_canonical_identity(**bad_h_f)

                d = base_ident.to_dict()
                d[field] = bad_val
                with self.assertRaisesRegex(CheckpointValidationError, rf"Invalid from_dict\.{field}.*lone surrogates"):
                    CanonicalResumeIdentity.from_dict(d)

                corrupt_act = copy.deepcopy(base_ident)
                object.__setattr__(corrupt_act, field, bad_val)
                with self.assertRaisesRegex(CheckpointValidationError, rf"Invalid actual\.{field}.*lone surrogates"):
                    validate_canonical_identity(corrupt_act, base_ident)

                corrupt_exp = copy.deepcopy(base_ident)
                object.__setattr__(corrupt_exp, field, bad_val)
                with self.assertRaisesRegex(CheckpointValidationError, rf"Invalid expected\.{field}.*lone surrogates"):
                    validate_canonical_identity(base_ident, corrupt_exp)

        # F. Bad settings text
        for sec, fld, bad_val in (
            ("protocol", "candidate_policy", "full_candidate_ranking\ud800"),
            ("training", "optimizer", "Adam\ud800"),
            ("execution", "execution_device", "cpu\ud800"),
        ):
            with self.subTest(sec=sec, fld=fld, bad_val=bad_val):
                bad_s_f = copy.deepcopy(f)
                bad_s_f["resolved_settings"][sec][fld] = bad_val
                with self.assertRaisesRegex(CheckpointValidationError, rf"Field {sec}\.{fld}.*lone surrogates"):
                    build_canonical_identity(**bad_s_f)

                corrupt_act = copy.deepcopy(base_ident)
                corrupt_act.resolved_settings[sec][fld] = bad_val
                with self.assertRaisesRegex(CheckpointValidationError, rf"Field {sec}\.{fld}.*lone surrogates"):
                    validate_canonical_identity(corrupt_act, base_ident)

                corrupt_exp = copy.deepcopy(base_ident)
                corrupt_exp.resolved_settings[sec][fld] = bad_val
                with self.assertRaisesRegex(CheckpointValidationError, rf"Field {sec}\.{fld}.*lone surrogates"):
                    validate_canonical_identity(base_ident, corrupt_exp)

        # 3. Positive full identity + roundtrip with non-ASCII graph/metadata
        non_ascii_f = copy.deepcopy(f)
        non_ascii_f["graph_nodes"]["CategoryPath"] = ['c:["楽器"]', 'c:["楽器","ギター"]']
        non_ascii_f["graph_nodes"]["Brand"] = ["b:ヤマハ"]
        non_ascii_f["graph_nodes"]["User"] = ["u_日本語", "u2"]
        non_ascii_f["graph_nodes"]["Product"] = ["p_雪", "p2", "p3"]
        non_ascii_f["warm_items"] = ["p_雪", "p2", "p3"]
        non_ascii_f["positive_history"] = {"u_日本語": ["p_雪", "p2"], "u2": ["p2", "p3"]}
        non_ascii_f["all_rated_history"] = {"u_日本語": ["p_雪", "p2"], "u2": ["p2", "p3"]}
        non_ascii_f["training_records"] = [
            {"user_id": "u_日本語", "parent_asin": "p_雪", "rating": 4.0, "timestamp": 1000},
            {"user_id": "u_日本語", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p2", "rating": 3.0, "timestamp": 3000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 4.0, "timestamp": 4000},
        ]
        non_ascii_f["graph_edges"] = [
            ("User", "u_日本語", "rated", "Product", "p_雪", 1.0),
            ("Product", "p_雪", "rated_by", "User", "u_日本語", 1.0),
            ("Product", "p_雪", "belongs_to", "CategoryPath", 'c:["楽器","ギター"]', 1.0),
            ("CategoryPath", 'c:["楽器","ギター"]', "contains_product", "Product", "p_雪", 1.0),
            ("Product", "p_雪", "has_brand", "Brand", "b:ヤマハ", 1.0),
            ("Brand", "b:ヤマハ", "brand_of", "Product", "p_雪", 1.0),
            ("CategoryPath", 'c:["楽器","ギター"]', "category_child_of", "CategoryPath", 'c:["楽器"]', 1.0),
            ("CategoryPath", 'c:["楽器"]', "category_parent_of", "CategoryPath", 'c:["楽器","ギター"]', 1.0),
        ]
        non_ascii_f["product_metadata"] = {
            "p_雪": {"categories": ['c:["楽器","ギター"]'], "brand": "b:ヤマハ"},
            "p2": {"categories": ['c:["楽器"]'], "brand": None},
            "p3": {"categories": [], "brand": None},
        }
        non_ascii_f["validation_inputs"] = {
            "targets": {"u_日本語": ["p3"], "u2": ["p_雪"]},
            "records": [("u_日本語", "p3", 5.0, 5000), ("u2", "p_雪", 4.0, 6000)],
        }
        non_ascii_ident = build_canonical_identity(**non_ascii_f)
        validate_canonical_identity(non_ascii_ident, non_ascii_ident)
        d_na = non_ascii_ident.to_dict()
        rec_na = CanonicalResumeIdentity.from_dict(d_na)
        validate_canonical_identity(non_ascii_ident, rec_na)
        self.assertEqual(non_ascii_ident, rec_na)

        # 4. Numeric overflow in ratings, weights, settings: BOTH 10**400 and 10**5000
        for exp in (400, 5000):
            huge_num = 10**exp
            with self.subTest(huge_num_exp=exp):
                # A. training record rating
                bad_r_f = copy.deepcopy(f)
                bad_r_f["training_records"][0]["rating"] = huge_num
                with self.assertRaisesRegex(CheckpointValidationError, r"Numeric overflow in rating at index 0"):
                    build_canonical_identity(**bad_r_f)

                # B. graph edge weight
                bad_w_f = copy.deepcopy(f)
                bad_w_f["graph_edges"][0] = ("User", "u1", "rated", "Product", "p1", huge_num)
                with self.assertRaisesRegex(CheckpointValidationError, r"Numeric overflow in edge weight at index 0"):
                    build_canonical_identity(**bad_w_f)

                # C. validation record rating
                bad_vr_f = copy.deepcopy(f)
                bad_vr_f["validation_inputs"]["records"][0] = ("u1", "p3", huge_num, 5000)
                with self.assertRaisesRegex(CheckpointValidationError, r"Numeric overflow in rating in validation record at index 0"):
                    build_canonical_identity(**bad_vr_f)

                # D. settings adam_betas
                bad_b1_f = copy.deepcopy(f)
                bad_b1_f["resolved_settings"]["training"]["adam_betas"] = [huge_num, 0.999]
                with self.assertRaisesRegex(CheckpointValidationError, r"Numeric overflow in training\.adam_betas"):
                    build_canonical_identity(**bad_b1_f)

                bad_b2_f = copy.deepcopy(f)
                bad_b2_f["resolved_settings"]["training"]["adam_betas"] = [0.9, huge_num]
                with self.assertRaisesRegex(CheckpointValidationError, r"Numeric overflow in training\.adam_betas"):
                    build_canonical_identity(**bad_b2_f)

                # E. settings float fields
                for fld in ("learning_rate", "endpoint_regularization", "weight_decay", "adam_epsilon"):
                    bad_flt_f = copy.deepcopy(f)
                    bad_flt_f["resolved_settings"]["training"][fld] = huge_num
                    with self.assertRaisesRegex(CheckpointValidationError, rf"Numeric overflow in field training\.{fld}"):
                        build_canonical_identity(**bad_flt_f)

        # 5. Huge positive seed and timestamp JSON conversion (10**5000)
        huge_int = 10**5000
        for target, expected_role in (
            ("training_timestamp", "data fingerprint"),
            ("validation_timestamp", "validation fingerprint"),
            ("settings_seed", "settings fingerprint"),
            ("header_seed", "identity hash"),
        ):
            with self.subTest(huge_int_target=target):
                bad_int_f = copy.deepcopy(f)
                if target == "training_timestamp":
                    bad_int_f["training_records"][0]["timestamp"] = huge_int
                elif target == "validation_timestamp":
                    bad_int_f["validation_inputs"]["records"][0] = ("u1", "p3", 5.0, huge_int)
                elif target == "settings_seed":
                    bad_int_f["resolved_settings"]["determinism"]["seed"] = huge_int
                elif target == "header_seed":
                    bad_int_f["seed"] = huge_int

                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"Numeric or serialization conversion failure in {expected_role}",
                ):
                    build_canonical_identity(**bad_int_f)

        # 6. Supported large exact integers: 2**14000 and neighbors across all 8 validation locations
        val_2_14000 = 2**14000
        self.assertEqual(val_2_14000.bit_length(), 14001)
        self.assertEqual(len(str(val_2_14000)), 4215)
        neighbor_integers = (val_2_14000 - 1, val_2_14000, val_2_14000 + 1)

        # Location 1: Header seed
        for s_val in neighbor_integers:
            with self.subTest(location="header_seed", s_val=s_val):
                h_f = copy.deepcopy(f)
                h_f["seed"] = s_val
                h_ident = build_canonical_identity(**h_f)
                self.assertEqual(h_ident.seed, s_val)
                self.assertIsInstance(h_ident.seed, int)
                self.assertNotIsInstance(h_ident.seed, (bool, float))
                validate_canonical_identity(h_ident, h_ident)
                d_h = h_ident.to_dict()
                rec_h = CanonicalResumeIdentity.from_dict(d_h)
                self.assertEqual(h_ident, rec_h)
                self.assertEqual(rec_h.seed, s_val)
                self.assertNotEqual(h_ident.identity_hash, base_ident.identity_hash)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"seed"):
                    validate_canonical_identity(h_ident, base_ident)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"seed"):
                    validate_canonical_identity(base_ident, h_ident)

        # Location 2: Training record timestamp
        for ts_val in neighbor_integers:
            with self.subTest(location="training_timestamp", ts_val=ts_val):
                t_f = copy.deepcopy(f)
                t_f["training_records"][0]["timestamp"] = ts_val
                t_ident = build_canonical_identity(**t_f)
                self.assertEqual(t_ident.training_records[0]["timestamp"], ts_val)
                self.assertIsInstance(t_ident.training_records[0]["timestamp"], int)
                self.assertNotIsInstance(t_ident.training_records[0]["timestamp"], (bool, float))
                validate_canonical_identity(t_ident, t_ident)
                d_t = t_ident.to_dict()
                rec_t = CanonicalResumeIdentity.from_dict(d_t)
                self.assertEqual(t_ident, rec_t)
                self.assertEqual(rec_t.training_records[0]["timestamp"], ts_val)
                self.assertNotEqual(t_ident.data_fingerprint, base_ident.data_fingerprint)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"data_fingerprint"):
                    validate_canonical_identity(t_ident, base_ident)

        # Location 3: Validation record timestamp
        for vts_val in neighbor_integers:
            with self.subTest(location="validation_timestamp", vts_val=vts_val):
                v_f = copy.deepcopy(f)
                v_f["validation_inputs"]["records"][0] = ("u1", "p3", 5.0, vts_val)
                v_ident = build_canonical_identity(**v_f)
                self.assertEqual(v_ident.validation_inputs["records"][0]["timestamp"], vts_val)
                self.assertIsInstance(v_ident.validation_inputs["records"][0]["timestamp"], int)
                self.assertNotIsInstance(v_ident.validation_inputs["records"][0]["timestamp"], (bool, float))
                validate_canonical_identity(v_ident, v_ident)
                d_v = v_ident.to_dict()
                rec_v = CanonicalResumeIdentity.from_dict(d_v)
                self.assertEqual(v_ident, rec_v)
                self.assertEqual(rec_v.validation_inputs["records"][0]["timestamp"], vts_val)
                self.assertNotEqual(v_ident.validation_fingerprint, base_ident.validation_fingerprint)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"validation_fingerprint"):
                    validate_canonical_identity(v_ident, base_ident)

        # Location 4: General integer settings (training.batch_size)
        for bs_val in neighbor_integers:
            with self.subTest(location="general_settings_batch_size", bs_val=bs_val):
                s_f = copy.deepcopy(f)
                s_f["resolved_settings"]["training"]["batch_size"] = bs_val
                s_ident = build_canonical_identity(**s_f)
                self.assertEqual(s_ident.resolved_settings["training"]["batch_size"], bs_val)
                self.assertIsInstance(s_ident.resolved_settings["training"]["batch_size"], int)
                self.assertNotIsInstance(s_ident.resolved_settings["training"]["batch_size"], (bool, float))
                validate_canonical_identity(s_ident, s_ident)
                d_s = s_ident.to_dict()
                rec_s = CanonicalResumeIdentity.from_dict(d_s)
                self.assertEqual(s_ident, rec_s)
                self.assertEqual(rec_s.resolved_settings["training"]["batch_size"], bs_val)
                self.assertNotEqual(s_ident.settings_fingerprint, base_ident.settings_fingerprint)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
                    validate_canonical_identity(s_ident, base_ident)

        # Location 5: Positive worker prefetch_factor
        pos_f = _make_valid_positive_worker_identity_fixture()
        base_pos_ident = build_canonical_identity(**pos_f)
        for pf_val in neighbor_integers:
            with self.subTest(location="positive_worker_prefetch", pf_val=pf_val):
                pw_f = copy.deepcopy(pos_f)
                pw_f["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = pf_val
                pw_ident = build_canonical_identity(**pw_f)
                self.assertEqual(pw_ident.resolved_settings["execution"]["worker_settings"]["prefetch_factor"], pf_val)
                self.assertIsInstance(pw_ident.resolved_settings["execution"]["worker_settings"]["prefetch_factor"], int)
                self.assertNotIsInstance(pw_ident.resolved_settings["execution"]["worker_settings"]["prefetch_factor"], (bool, float))
                validate_canonical_identity(pw_ident, pw_ident)
                d_pw = pw_ident.to_dict()
                rec_pw = CanonicalResumeIdentity.from_dict(d_pw)
                self.assertEqual(pw_ident, rec_pw)
                self.assertEqual(rec_pw.resolved_settings["execution"]["worker_settings"]["prefetch_factor"], pf_val)
                self.assertNotEqual(pw_ident.settings_fingerprint, base_pos_ident.settings_fingerprint)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
                    validate_canonical_identity(pw_ident, base_pos_ident)

        # Location 6: Sampling neighbor_fanouts
        for fo_val in neighbor_integers:
            with self.subTest(location="sampling_neighbor_fanouts", fo_val=fo_val):
                nf_f = copy.deepcopy(f)
                nf_f["resolved_settings"]["sampling"]["neighbor_fanouts"] = [fo_val, 10]
                nf_ident = build_canonical_identity(**nf_f)
                self.assertEqual(nf_ident.resolved_settings["sampling"]["neighbor_fanouts"], [fo_val, 10])
                self.assertIsInstance(nf_ident.resolved_settings["sampling"]["neighbor_fanouts"][0], int)
                self.assertNotIsInstance(nf_ident.resolved_settings["sampling"]["neighbor_fanouts"][0], (bool, float))
                validate_canonical_identity(nf_ident, nf_ident)
                d_nf = nf_ident.to_dict()
                rec_nf = CanonicalResumeIdentity.from_dict(d_nf)
                self.assertEqual(nf_ident, rec_nf)
                self.assertEqual(rec_nf.resolved_settings["sampling"]["neighbor_fanouts"], [fo_val, 10])
                self.assertNotEqual(nf_ident.settings_fingerprint, base_ident.settings_fingerprint)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
                    validate_canonical_identity(nf_ident, base_ident)

        # Location 7: RNG stream seed
        for rs_val in neighbor_integers:
            with self.subTest(location="rng_stream_seed", rs_val=rs_val):
                rng_f = copy.deepcopy(f)
                rng_f["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = rs_val
                rng_ident = build_canonical_identity(**rng_f)
                self.assertEqual(rng_ident.resolved_settings["determinism"]["rng_streams"][0]["seed"], rs_val)
                self.assertIsInstance(rng_ident.resolved_settings["determinism"]["rng_streams"][0]["seed"], int)
                self.assertNotIsInstance(rng_ident.resolved_settings["determinism"]["rng_streams"][0]["seed"], (bool, float))
                validate_canonical_identity(rng_ident, rng_ident)
                d_rng = rng_ident.to_dict()
                rec_rng = CanonicalResumeIdentity.from_dict(d_rng)
                self.assertEqual(rng_ident, rec_rng)
                self.assertEqual(rec_rng.resolved_settings["determinism"]["rng_streams"][0]["seed"], rs_val)
                self.assertNotEqual(rng_ident.settings_fingerprint, base_ident.settings_fingerprint)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
                    validate_canonical_identity(rng_ident, base_ident)

        # Location 8: Worker base_seed (coupled with determinism.seed)
        for wbs_val in neighbor_integers:
            with self.subTest(location="worker_base_seed", wbs_val=wbs_val):
                wbs_f = copy.deepcopy(pos_f)
                wbs_f["resolved_settings"]["determinism"]["seed"] = wbs_val
                wbs_f["resolved_settings"]["determinism"]["worker_derivation"]["base_seed"] = wbs_val
                wbs_ident = build_canonical_identity(**wbs_f)
                self.assertEqual(wbs_ident.resolved_settings["determinism"]["worker_derivation"]["base_seed"], wbs_val)
                self.assertEqual(wbs_ident.resolved_settings["determinism"]["seed"], wbs_val)
                self.assertIsInstance(wbs_ident.resolved_settings["determinism"]["worker_derivation"]["base_seed"], int)
                self.assertNotIsInstance(wbs_ident.resolved_settings["determinism"]["worker_derivation"]["base_seed"], (bool, float))
                validate_canonical_identity(wbs_ident, wbs_ident)
                d_wbs = wbs_ident.to_dict()
                rec_wbs = CanonicalResumeIdentity.from_dict(d_wbs)
                self.assertEqual(wbs_ident, rec_wbs)
                self.assertEqual(rec_wbs.resolved_settings["determinism"]["worker_derivation"]["base_seed"], wbs_val)
                self.assertNotEqual(wbs_ident.settings_fingerprint, base_pos_ident.settings_fingerprint)
                with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
                    validate_canonical_identity(wbs_ident, base_pos_ident)

        # Cross-field dependency check: uncoupled worker base_seed mutation alone fails
        uncoupled_f = copy.deepcopy(pos_f)
        uncoupled_f["resolved_settings"]["determinism"]["worker_derivation"]["base_seed"] = val_2_14000
        # determinism.seed remains 42
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.worker_derivation\.base_seed.*must agree with determinism\.seed",
        ):
            build_canonical_identity(**uncoupled_f)

    def test_rng_stream_record_matrix_and_independent_mutations(self):
        f = _make_valid_identity_fixture()
        base_ident = build_canonical_identity(**f)

        # 1. EACH of 6 keys missing in rng_streams record
        for key in ("name", "applicable", "algorithm", "seed", "derivation", "reason"):
            with self.subTest(missing_key=key):
                bad_f = copy.deepcopy(f)
                del bad_f["resolved_settings"]["determinism"]["rng_streams"][0][key]
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"determinism\.rng_streams\[0\] record keys mismatch.*Missing: \['{key}'\]",
                ):
                    build_canonical_identity(**bad_f)

        # 2. Unknown and mixed extra record keys
        bad_extra_k = copy.deepcopy(f)
        bad_extra_k["resolved_settings"]["determinism"]["rng_streams"][0]["unknown_key"] = "extra"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.rng_streams\[0\] record keys mismatch.*Extra: \['unknown_key'\]",
        ):
            build_canonical_identity(**bad_extra_k)

        bad_mixed_k = copy.deepcopy(f)
        bad_mixed_k["resolved_settings"]["determinism"]["rng_streams"][0][0] = "zero_key"
        bad_mixed_k["resolved_settings"]["determinism"]["rng_streams"][0]["unknown"] = "extra"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.rng_streams\[0\] record keys must be strings, got int",
        ):
            build_canonical_identity(**bad_mixed_k)

        # 3. Wrong container
        for bad_container in ("not_a_record", [1, 2, 3], 123):
            with self.subTest(bad_container=type(bad_container).__name__):
                bad_c_f = copy.deepcopy(f)
                bad_c_f["resolved_settings"]["determinism"]["rng_streams"][0] = bad_container
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"determinism\.rng_streams\[0\] must be a mapping, got {type(bad_container).__name__}",
                ):
                    build_canonical_identity(**bad_c_f)

        # 4. Strict bool applicability
        for bad_bool in (1, 0, "True", "False", None):
            with self.subTest(bad_bool=bad_bool):
                bad_app_f = copy.deepcopy(f)
                bad_app_f["resolved_settings"]["determinism"]["rng_streams"][0]["applicable"] = bad_bool
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.rng_streams\[0\] \(Python\)\.applicable must be a strict boolean",
                ):
                    build_canonical_identity(**bad_app_f)

        # 5. Component names: all 10 registered components appear in valid fixture
        present_names = [s["name"] for s in f["resolved_settings"]["determinism"]["rng_streams"]]
        self.assertEqual(len(present_names), 10)
        self.assertEqual(set(present_names), set(checkpoint_identity.CONFIGURED_RNG_COMPONENTS))

        # Missing component, unknown component, duplicate component
        bad_miss_comp = copy.deepcopy(f)
        bad_miss_comp["resolved_settings"]["determinism"]["rng_streams"] = bad_miss_comp["resolved_settings"]["determinism"]["rng_streams"][:-1]
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.rng_streams components mismatch.*Missing: \['DataLoader_workers'\]",
        ):
            build_canonical_identity(**bad_miss_comp)

        bad_unk_comp = copy.deepcopy(f)
        bad_unk_comp["resolved_settings"]["determinism"]["rng_streams"][0]["name"] = "unknown_component"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"Invalid stream name in determinism\.rng_streams\[0\]: 'unknown_component'",
        ):
            build_canonical_identity(**bad_unk_comp)

        bad_dup_comp = copy.deepcopy(f)
        bad_dup_comp["resolved_settings"]["determinism"]["rng_streams"][1]["name"] = bad_dup_comp["resolved_settings"]["determinism"]["rng_streams"][0]["name"]
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"Duplicate stream name in determinism\.rng_streams: 'Python'",
        ):
            build_canonical_identity(**bad_dup_comp)

        # 6. Active branch invalid values (tested on stream 0: "Python")
        # algorithm: empty, padded, non-string
        for bad_alg in ("", "  ", " stdlib ", 123):
            with self.subTest(bad_alg=bad_alg):
                bad_a_f = copy.deepcopy(f)
                bad_a_f["resolved_settings"]["determinism"]["rng_streams"][0]["algorithm"] = bad_alg
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.rng_streams\[0\] \(Python\)\.algorithm",
                ):
                    build_canonical_identity(**bad_a_f)

        # seed: bool, string, float, negative, None
        for bad_seed in (True, False, "42", 42.5, -1, None):
            with self.subTest(bad_seed=bad_seed):
                bad_s_f = copy.deepcopy(f)
                bad_s_f["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = bad_seed
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.rng_streams\[0\] \(Python\)\.seed must be a non-negative integer when applicable=True",
                ):
                    build_canonical_identity(**bad_s_f)

        # derivation: empty, padded, non-string
        for bad_drv in ("", "  ", " fixed_seed ", 123):
            with self.subTest(bad_drv=bad_drv):
                bad_d_f = copy.deepcopy(f)
                bad_d_f["resolved_settings"]["determinism"]["rng_streams"][0]["derivation"] = bad_drv
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.rng_streams\[0\] \(Python\)\.derivation",
                ):
                    build_canonical_identity(**bad_d_f)

        # reason: non-None
        bad_r_f = copy.deepcopy(f)
        bad_r_f["resolved_settings"]["determinism"]["rng_streams"][0]["reason"] = "should_be_none"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.rng_streams\[0\] \(Python\)\.reason must be None when applicable=True",
        ):
            build_canonical_identity(**bad_r_f)

        # 7. Inactive branch invalid values (tested on stream 3: "PyTorch_CUDA_if_applicable")
        # Each None field separately violated
        for non_none_field, non_none_val in (("algorithm", "torch_cuda"), ("seed", 42), ("derivation", "fixed_seed")):
            with self.subTest(inactive_violation=non_none_field):
                bad_in_f = copy.deepcopy(f)
                bad_in_f["resolved_settings"]["determinism"]["rng_streams"][3][non_none_field] = non_none_val
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"determinism\.rng_streams\[3\] \(PyTorch_CUDA_if_applicable\)\.{non_none_field} must be None when applicable=False",
                ):
                    build_canonical_identity(**bad_in_f)

        # reason: blank, padded, non-string, None
        for bad_rsn in ("", "  ", " cuda_reason ", 123, None):
            with self.subTest(bad_reason=bad_rsn):
                bad_rsn_f = copy.deepcopy(f)
                bad_rsn_f["resolved_settings"]["determinism"]["rng_streams"][3]["reason"] = bad_rsn
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.rng_streams\[3\] \(PyTorch_CUDA_if_applicable\)\.reason",
                ):
                    build_canonical_identity(**bad_rsn_f)

        # 8. Independent valid mutations of ACTIVE algorithm/seed/derivation and INACTIVE reason
        # Mutate active algorithm
        mut_alg_f = copy.deepcopy(f)
        mut_alg_f["resolved_settings"]["determinism"]["rng_streams"][0]["algorithm"] = "stdlib_random_synthetic_v2"
        mut_alg_ident = build_canonical_identity(**mut_alg_f)
        self.assertNotEqual(base_ident.settings_fingerprint, mut_alg_ident.settings_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, mut_alg_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_alg_ident, base_ident)

        # Mutate active seed
        mut_sd_f = copy.deepcopy(f)
        mut_sd_f["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = 43
        mut_sd_ident = build_canonical_identity(**mut_sd_f)
        self.assertNotEqual(base_ident.settings_fingerprint, mut_sd_ident.settings_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, mut_sd_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_sd_ident, base_ident)

        # Mutate active derivation
        mut_drv_f = copy.deepcopy(f)
        mut_drv_f["resolved_settings"]["determinism"]["rng_streams"][0]["derivation"] = "custom_derivation_v1"
        mut_drv_ident = build_canonical_identity(**mut_drv_f)
        self.assertNotEqual(base_ident.settings_fingerprint, mut_drv_ident.settings_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, mut_drv_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_drv_ident, base_ident)

        # Mutate inactive reason
        mut_rsn_f = copy.deepcopy(f)
        mut_rsn_f["resolved_settings"]["determinism"]["rng_streams"][3]["reason"] = "cuda_not_applicable_synthetic_v2"
        mut_rsn_ident = build_canonical_identity(**mut_rsn_f)
        self.assertNotEqual(base_ident.settings_fingerprint, mut_rsn_ident.settings_fingerprint)
        self.assertNotEqual(base_ident.identity_hash, mut_rsn_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_rsn_ident, base_ident)

        # 9. Nested invalid record cases in BOTH actual and expected validators
        for role in ("actual", "expected"):
            with self.subTest(nested_role=role):
                corrupt_ident = copy.deepcopy(base_ident)
                corrupt_ident.resolved_settings["determinism"]["rng_streams"][0]["seed"] = -1
                if role == "actual":
                    with self.assertRaisesRegex(
                        CheckpointValidationError,
                        r"determinism\.rng_streams\[0\] \(Python\)\.seed must be a non-negative integer",
                    ):
                        validate_canonical_identity(corrupt_ident, base_ident)
                else:
                    with self.assertRaisesRegex(
                        CheckpointValidationError,
                        r"determinism\.rng_streams\[0\] \(Python\)\.seed must be a non-negative integer",
                    ):
                        validate_canonical_identity(base_ident, corrupt_ident)

    def test_worker_settings_and_derivation_matrix(self):
        # Build one otherwise-valid synthetic POSITIVE worker fixture with ALL bindings consistent
        pos_f = _make_valid_positive_worker_identity_fixture()
        pos_ident = build_canonical_identity(**pos_f)
        validate_canonical_identity(pos_ident, pos_ident)

        zero_f = _make_valid_identity_fixture()
        zero_ident = build_canonical_identity(**zero_f)

        # 1. worker_derivation 5 keys: missing each
        for key in ("applicable", "algorithm", "base_seed", "derivation", "reason"):
            with self.subTest(missing_wd_key=key):
                bad_f = copy.deepcopy(pos_f)
                del bad_f["resolved_settings"]["determinism"]["worker_derivation"][key]
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"determinism\.worker_derivation fields mismatch.*Missing: \['{key}'\]",
                ):
                    build_canonical_identity(**bad_f)

        # Unknown and mixed extra keys in worker_derivation
        bad_wd_extra = copy.deepcopy(pos_f)
        bad_wd_extra["resolved_settings"]["determinism"]["worker_derivation"]["unknown"] = "bad"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.worker_derivation fields mismatch.*Extra: \['unknown'\]",
        ):
            build_canonical_identity(**bad_wd_extra)

        bad_wd_mixed = copy.deepcopy(pos_f)
        bad_wd_mixed["resolved_settings"]["determinism"]["worker_derivation"][0] = "zero"
        bad_wd_mixed["resolved_settings"]["determinism"]["worker_derivation"]["unknown"] = "bad"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.worker_derivation keys must be strings, got int",
        ):
            build_canonical_identity(**bad_wd_mixed)

        # Wrong container for worker_derivation
        for bad_c in ("not_dict", [1, 2], 123):
            with self.subTest(bad_wd_container=type(bad_c).__name__):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["determinism"]["worker_derivation"] = bad_c
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"Field determinism\.worker_derivation must be a mapping, got {type(bad_c).__name__}",
                ):
                    build_canonical_identity(**bad_f)

        # Strict bool applicability for worker_derivation
        for bad_app in (1, 0, "True", "False", None):
            with self.subTest(bad_wd_applicable=bad_app):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["determinism"]["worker_derivation"]["applicable"] = bad_app
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.worker_derivation\.applicable must be a strict boolean",
                ):
                    build_canonical_identity(**bad_f)

        # Active worker_derivation invalid fields
        for bad_alg in ("", "  ", " alg ", 123):
            with self.subTest(bad_wd_alg=bad_alg):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["determinism"]["worker_derivation"]["algorithm"] = bad_alg
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.worker_derivation\.algorithm",
                ):
                    build_canonical_identity(**bad_f)

        for bad_bs in (True, -1, 42.5, "42", None):
            with self.subTest(bad_wd_base_seed=bad_bs):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["determinism"]["worker_derivation"]["base_seed"] = bad_bs
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.worker_derivation\.base_seed must be a non-negative integer when applicable=True",
                ):
                    build_canonical_identity(**bad_f)

        for bad_drv in ("", "  ", " drv ", 123):
            with self.subTest(bad_wd_drv=bad_drv):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["determinism"]["worker_derivation"]["derivation"] = bad_drv
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.worker_derivation\.derivation",
                ):
                    build_canonical_identity(**bad_f)

        bad_wd_rsn = copy.deepcopy(pos_f)
        bad_wd_rsn["resolved_settings"]["determinism"]["worker_derivation"]["reason"] = "should_be_none"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.worker_derivation\.reason must be None when applicable=True",
        ):
            build_canonical_identity(**bad_wd_rsn)

        # Inactive worker_derivation invalid fields (tested on zero-worker fixture)
        for field, bad_val in (("algorithm", "alg"), ("base_seed", 42), ("derivation", "drv")):
            with self.subTest(inactive_wd_violation=field):
                bad_f = copy.deepcopy(zero_f)
                bad_f["resolved_settings"]["determinism"]["worker_derivation"][field] = bad_val
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"determinism\.worker_derivation\.{field} must be None when applicable=False",
                ):
                    build_canonical_identity(**bad_f)

        for bad_rsn in ("", "  ", " reason ", 123, None):
            with self.subTest(bad_inactive_wd_reason=bad_rsn):
                bad_f = copy.deepcopy(zero_f)
                bad_f["resolved_settings"]["determinism"]["worker_derivation"]["reason"] = bad_rsn
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"determinism\.worker_derivation\.reason",
                ):
                    build_canonical_identity(**bad_f)

        # Isolated sensitivity: active algorithm, active derivation
        mut_wd_alg_f = copy.deepcopy(pos_f)
        mut_wd_alg_f["resolved_settings"]["determinism"]["worker_derivation"]["algorithm"] = "worker_seed_fn_v2"
        mut_wd_alg_ident = build_canonical_identity(**mut_wd_alg_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_wd_alg_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_wd_alg_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_wd_alg_ident, pos_ident)

        mut_wd_drv_f = copy.deepcopy(pos_f)
        mut_wd_drv_f["resolved_settings"]["determinism"]["worker_derivation"]["derivation"] = "custom_worker_derivation"
        mut_wd_drv_ident = build_canonical_identity(**mut_wd_drv_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_wd_drv_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_wd_drv_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_wd_drv_ident, pos_ident)

        # Base-seed consistency: mutating base_seed alone raises consistency error
        mut_bs_alone_f = copy.deepcopy(pos_f)
        mut_bs_alone_f["resolved_settings"]["determinism"]["worker_derivation"]["base_seed"] = 43
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"determinism\.worker_derivation\.base_seed \(43\) must agree with determinism\.seed \(42\)",
        ):
            build_canonical_identity(**mut_bs_alone_f)

        # Mutating bound determinism.seed together with base_seed is valid and changes hashes
        mut_bs_bound_f = copy.deepcopy(pos_f)
        mut_bs_bound_f["seed"] = 43
        mut_bs_bound_f["resolved_settings"]["determinism"]["seed"] = 43
        mut_bs_bound_f["resolved_settings"]["determinism"]["worker_derivation"]["base_seed"] = 43
        mut_bs_bound_ident = build_canonical_identity(**mut_bs_bound_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_bs_bound_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_bs_bound_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_bs_bound_ident, pos_ident)

        # Inactive reason isolated sensitivity (on zero-worker fixture)
        mut_wd_rsn_f = copy.deepcopy(zero_f)
        mut_wd_rsn_f["resolved_settings"]["determinism"]["worker_derivation"]["reason"] = "alt_zero_worker_reason"
        mut_wd_rsn_ident = build_canonical_identity(**mut_wd_rsn_f)
        self.assertNotEqual(zero_ident.settings_fingerprint, mut_wd_rsn_ident.settings_fingerprint)
        self.assertNotEqual(zero_ident.identity_hash, mut_wd_rsn_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_wd_rsn_ident, zero_ident)

        # 2. worker_settings 4 keys: missing each
        for key in ("prefetch_factor", "persistent_workers", "multiprocessing_context", "worker_init_function"):
            with self.subTest(missing_ws_key=key):
                bad_f = copy.deepcopy(pos_f)
                del bad_f["resolved_settings"]["execution"]["worker_settings"][key]
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"execution\.worker_settings fields mismatch.*Missing: \['{key}'\]",
                ):
                    build_canonical_identity(**bad_f)

        # Unknown and mixed extra keys in worker_settings
        bad_ws_extra = copy.deepcopy(pos_f)
        bad_ws_extra["resolved_settings"]["execution"]["worker_settings"]["unknown"] = "bad"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"execution\.worker_settings fields mismatch.*Extra: \['unknown'\]",
        ):
            build_canonical_identity(**bad_ws_extra)

        bad_ws_mixed = copy.deepcopy(pos_f)
        bad_ws_mixed["resolved_settings"]["execution"]["worker_settings"][0] = "zero"
        bad_ws_mixed["resolved_settings"]["execution"]["worker_settings"]["unknown"] = "bad"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"execution\.worker_settings keys must be strings, got int",
        ):
            build_canonical_identity(**bad_ws_mixed)

        # Wrong container for worker_settings
        for bad_c in ("not_dict", [1, 2], 123):
            with self.subTest(bad_ws_container=type(bad_c).__name__):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["execution"]["worker_settings"] = bad_c
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"Field execution\.worker_settings must be a mapping, got {type(bad_c).__name__}",
                ):
                    build_canonical_identity(**bad_f)

        # Zero worker constraints (from zero-worker fixture)
        for k, bad_v in (
            ("prefetch_factor", 2),
            ("persistent_workers", True),
            ("multiprocessing_context", "fork"),
            ("worker_init_function", "init_fn"),
        ):
            with self.subTest(zero_worker_invalid=k):
                bad_f = copy.deepcopy(zero_f)
                bad_f["resolved_settings"]["execution"]["worker_settings"][k] = bad_v
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    rf"execution\.worker_settings\.{k} must be (?:None|strictly False) when runtime_workers == 0",
                ):
                    build_canonical_identity(**bad_f)

        # Positive-worker invalid constraints STARTING FROM VALID POSITIVE FIXTURE
        for bad_pf in (None, 0, -1, True, 2.5, "2"):
            with self.subTest(positive_worker_bad_prefetch=bad_pf):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = bad_pf
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"execution\.worker_settings\.prefetch_factor must be a positive integer when runtime_workers > 0",
                ):
                    build_canonical_identity(**bad_f)

        for bad_pw in (1, 0, "True", "False", None):
            with self.subTest(positive_worker_bad_persistent=bad_pw):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["execution"]["worker_settings"]["persistent_workers"] = bad_pw
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"execution\.worker_settings\.persistent_workers must be a strict boolean",
                ):
                    build_canonical_identity(**bad_f)

        for bad_mc in ("", "  ", " fork ", 123, None):
            with self.subTest(positive_worker_bad_context=bad_mc):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["execution"]["worker_settings"]["multiprocessing_context"] = bad_mc
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"execution\.worker_settings\.multiprocessing_context",
                ):
                    build_canonical_identity(**bad_f)

        for bad_wif in ("", "  ", " init_fn ", 123, None):
            with self.subTest(positive_worker_bad_init_fn=bad_wif):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["execution"]["worker_settings"]["worker_init_function"] = bad_wif
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"execution\.worker_settings\.worker_init_function",
                ):
                    build_canonical_identity(**bad_f)

        # 3. Independent mutations on valid positive worker fixture
        # prefetch_factor
        mut_pf_f = copy.deepcopy(pos_f)
        mut_pf_f["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = 4
        mut_pf_ident = build_canonical_identity(**mut_pf_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_pf_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_pf_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_pf_ident, pos_ident)

        # persistent_workers
        mut_pw_f = copy.deepcopy(pos_f)
        mut_pw_f["resolved_settings"]["execution"]["worker_settings"]["persistent_workers"] = False
        mut_pw_ident = build_canonical_identity(**mut_pw_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_pw_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_pw_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_pw_ident, pos_ident)

        # multiprocessing_context
        mut_mc_f = copy.deepcopy(pos_f)
        mut_mc_f["resolved_settings"]["execution"]["worker_settings"]["multiprocessing_context"] = "spawn"
        mut_mc_ident = build_canonical_identity(**mut_mc_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_mc_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_mc_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_mc_ident, pos_ident)

        # worker_init_function
        mut_wif_f = copy.deepcopy(pos_f)
        mut_wif_f["resolved_settings"]["execution"]["worker_settings"]["worker_init_function"] = "alt_worker_init_fn"
        mut_wif_ident = build_canonical_identity(**mut_wif_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_wif_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_wif_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_wif_ident, pos_ident)

        # runtime_workers (mutate to 4)
        mut_rw_f = copy.deepcopy(pos_f)
        mut_rw_f["resolved_settings"]["execution"]["runtime_workers"] = 4
        mut_rw_ident = build_canonical_identity(**mut_rw_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_rw_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_rw_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_rw_ident, pos_ident)

        # cpu_threads (mutate to 8)
        mut_ct_f = copy.deepcopy(pos_f)
        mut_ct_f["resolved_settings"]["execution"]["cpu_threads"] = 8
        mut_ct_ident = build_canonical_identity(**mut_ct_f)
        self.assertNotEqual(pos_ident.settings_fingerprint, mut_ct_ident.settings_fingerprint)
        self.assertNotEqual(pos_ident.identity_hash, mut_ct_ident.identity_hash)
        with self.assertRaisesRegex(CheckpointIdentityMismatchError, r"settings_fingerprint"):
            validate_canonical_identity(mut_ct_ident, pos_ident)

        # Invalid runtime_workers and cpu_threads
        for bad_rw in (-1, 1.5, True, "2", None):
            with self.subTest(bad_rw=bad_rw):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["execution"]["runtime_workers"] = bad_rw
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"Field execution\.runtime_workers",
                ):
                    build_canonical_identity(**bad_f)

        for bad_ct in (0, -1, True, 4.0, "4", None):
            with self.subTest(bad_ct=bad_ct):
                bad_f = copy.deepcopy(pos_f)
                bad_f["resolved_settings"]["execution"]["cpu_threads"] = bad_ct
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    r"Field execution\.cpu_threads",
                ):
                    build_canonical_identity(**bad_f)

        # 4. Deep input/output worker aliasing and postmutation against independent expected
        pos_f_mut = copy.deepcopy(pos_f)
        ident_alias = build_canonical_identity(**pos_f_mut)

        pos_f_mut["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = 999
        pos_f_mut["resolved_settings"]["determinism"]["worker_derivation"]["algorithm"] = "mutated"
        pos_f_mut["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = 999

        # Internal state remains unchanged
        validate_canonical_identity(ident_alias, pos_ident)
        self.assertEqual(ident_alias.resolved_settings["execution"]["worker_settings"]["prefetch_factor"], 2)
        self.assertEqual(ident_alias.resolved_settings["determinism"]["worker_derivation"]["algorithm"], "worker_seed_fn")

        # Output to_dict aliasing
        d = ident_alias.to_dict()
        d["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = 999
        d["resolved_settings"]["determinism"]["worker_derivation"]["algorithm"] = "mutated"
        validate_canonical_identity(ident_alias, pos_ident)

        # Post-mutation of internal dict triggers mismatch against independently built expected
        ident_alias.resolved_settings["execution"]["worker_settings"]["prefetch_factor"] = 999
        independent_expected = build_canonical_identity(**_make_valid_positive_worker_identity_fixture())
        with self.assertRaisesRegex(
            CheckpointIdentityMismatchError,
            r"settings_fingerprint in actual identity",
        ):
            validate_canonical_identity(ident_alias, independent_expected)

    def test_rating_ge_4_positivity_and_warm_pool_consistency(self):
        records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 4.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 3.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 5.0, "timestamp": 3000},
        ]
        all_hist = {"u1": ["p1", "p2"], "u2": ["p3"]}
        # In rating_ge_4, u1 positive item is p1 only (p2 rating is 3.0 < 4.0)
        pos_hist = {"u1": ["p1"], "u2": ["p3"]}
        warm_items = ["p1", "p3"]

        nodes = {"User": ["u1", "u2"], "Product": ["p1", "p3"]}
        edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("User", "u2", "rated", "Product", "p3", 1.0),
            ("Product", "p3", "rated_by", "User", "u2", 1.0),
        ]
        metadata = {"p1": {"categories": [], "brand": None}, "p3": {"categories": [], "brand": None}}
        val_in = {"targets": {"u1": ["p3"]}}
        settings = _make_valid_settings_fixture()

        # Valid rating_ge_4 identity
        ident = build_canonical_identity(
            method="LightGCN",
            category="Musical_Instruments",
            feedback_condition="rating_ge_4",
            seed=42,
            training_records=records,
            positive_history=pos_hist,
            all_rated_history=all_hist,
            warm_items=warm_items,
            graph_nodes=nodes,
            graph_edges=edges,
            product_metadata=metadata,
            validation_inputs=val_in,
            resolved_settings=settings,
        )
        validate_canonical_identity(ident, ident)

        # 1. Negative rating (< 4.0) wrongfully placed in positive_history
        bad_pos_hist = {"u1": ["p1", "p2"], "u2": ["p3"]}
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(
                method="LightGCN",
                category="Musical_Instruments",
                feedback_condition="rating_ge_4",
                seed=42,
                training_records=records,
                positive_history=bad_pos_hist,
                all_rated_history=all_hist,
                warm_items=warm_items,
                graph_nodes=nodes,
                graph_edges=edges,
                product_metadata=metadata,
                validation_inputs=val_in,
                resolved_settings=settings,
            )

        # 2. Warm items containing non-positive item in rating_ge_4
        bad_warm = ["p1", "p2", "p3"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(
                method="LightGCN",
                category="Musical_Instruments",
                feedback_condition="rating_ge_4",
                seed=42,
                training_records=records,
                positive_history=pos_hist,
                all_rated_history=all_hist,
                warm_items=bad_warm,
                graph_nodes=nodes,
                graph_edges=edges,
                product_metadata=metadata,
                validation_inputs=val_in,
                resolved_settings=settings,
            )

        # 3. Item with rating >= 4.0 missing from positive_history
        bad_missing_pos = {"u1": [], "u2": ["p3"]}
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(
                method="LightGCN",
                category="Musical_Instruments",
                feedback_condition="rating_ge_4",
                seed=42,
                training_records=records,
                positive_history=bad_missing_pos,
                all_rated_history=all_hist,
                warm_items=warm_items,
                graph_nodes=nodes,
                graph_edges=edges,
                product_metadata=metadata,
                validation_inputs=val_in,
                resolved_settings=settings,
            )

    def test_partial_record_and_history_inconsistencies(self):
        f = _make_valid_identity_fixture()

        # 1. User declared in records missing from positive_history
        bad_pos_missing_user = copy.deepcopy(f)
        del bad_pos_missing_user["positive_history"]["u2"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_pos_missing_user)

        # 2. User declared in records missing from all_rated_history
        bad_all_missing_user = copy.deepcopy(f)
        del bad_all_missing_user["all_rated_history"]["u2"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_all_missing_user)

        # 3. Extra user in positive_history not in records
        bad_pos_extra_user = copy.deepcopy(f)
        bad_pos_extra_user["positive_history"]["u3_unknown"] = ["p1"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_pos_extra_user)

        # 4. Incomplete user history: records have p1 and p2, all_rated_history only has p1
        bad_partial_all = copy.deepcopy(f)
        bad_partial_all["all_rated_history"]["u1"] = ["p1"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_partial_all)

        # 5. In all_ratings, positive_history missing an observed item
        bad_partial_pos = copy.deepcopy(f)
        bad_partial_pos["positive_history"]["u1"] = ["p1"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_partial_pos)

        # 6. Warm items missing an observed item
        bad_warm_missing = copy.deepcopy(f)
        bad_warm_missing["warm_items"] = ["p1", "p2"]  # missing p3
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_warm_missing)

        # 7. Warm items containing an unobserved item
        bad_warm_extra = copy.deepcopy(f)
        bad_warm_extra["warm_items"] = ["p1", "p2", "p3", "p4_unobserved"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_warm_extra)

    def test_duplicate_typed_node_category_and_metadata_identities(self):
        f = _make_valid_identity_fixture()

        # Duplicate User node
        dup_u = copy.deepcopy(f)
        dup_u["graph_nodes"]["User"].append("u1")
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_u)

        # Duplicate Product node
        dup_p = copy.deepcopy(f)
        dup_p["graph_nodes"]["Product"].append("p1")
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_p)

        # Duplicate CategoryPath node
        dup_c = copy.deepcopy(f)
        dup_c["graph_nodes"]["CategoryPath"].append('c:["Electronics"]')
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_c)

        # Duplicate Brand node
        dup_b = copy.deepcopy(f)
        dup_b["graph_nodes"]["Brand"].append("b:Sony")
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_b)

        # Duplicate category inside product_metadata
        dup_meta_cat = copy.deepcopy(f)
        dup_meta_cat["product_metadata"]["p1"]["categories"].append('c:["Electronics","Headphones"]')
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_meta_cat)

        # Duplicate graph edge
        dup_edge = copy.deepcopy(f)
        dup_edge["graph_edges"].append(dup_edge["graph_edges"][0])
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_edge)

        # Duplicate (user_id, parent_asin) record
        dup_rec = copy.deepcopy(f)
        dup_rec["training_records"].append(dup_rec["training_records"][0])
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_rec)

        # Duplicate warm items
        dup_warm = copy.deepcopy(f)
        dup_warm["warm_items"].append("p1")
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**dup_warm)

    def test_fractional_and_boolean_timestamps_and_ratings(self):
        f = _make_valid_identity_fixture()

        # Fractional timestamp
        frac_ts_f = copy.deepcopy(f)
        frac_ts_f["training_records"][0]["timestamp"] = 1000.5
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**frac_ts_f)

        # Boolean timestamp
        bool_ts_f = copy.deepcopy(f)
        bool_ts_f["training_records"][0]["timestamp"] = True
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bool_ts_f)

        # Boolean rating
        bool_r_f = copy.deepcopy(f)
        bool_r_f["training_records"][0]["rating"] = True
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bool_r_f)

        # Non-finite rating
        nan_r_f = copy.deepcopy(f)
        nan_r_f["training_records"][0]["rating"] = float("nan")
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**nan_r_f)

        # Boolean seed
        bool_s_f = copy.deepcopy(f)
        bool_s_f["seed"] = True
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bool_s_f)

    def test_excluded_and_unknown_metadata_rejection(self):
        f = _make_valid_identity_fixture()

        for excluded_field in (
            "average_rating",
            "rating_number",
            "store",
            "Manufacturer",
            "details",
            "bought_together",
            "bestseller_rank",
        ):
            bad_meta_f = copy.deepcopy(f)
            bad_meta_f["product_metadata"]["p1"][excluded_field] = 4.5
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject {excluded_field}"):
                build_canonical_identity(**bad_meta_f)

        # Unknown field
        bad_meta_unknown = copy.deepcopy(f)
        bad_meta_unknown["product_metadata"]["p1"]["unknown_attr"] = "val"
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_meta_unknown)

        # Product in graph_nodes but missing in product_metadata
        missing_meta_p = copy.deepcopy(f)
        del missing_meta_p["product_metadata"]["p3"]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**missing_meta_p)

        # Product in product_metadata but not declared in Product nodes
        extra_meta_p = copy.deepcopy(f)
        extra_meta_p["product_metadata"]["p_undeclared"] = {"categories": [], "brand": None}
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**extra_meta_p)

    def test_graph_contract_and_endpoint_failures(self):
        f = _make_valid_identity_fixture()

        # Unknown relation
        bad_rel_f = copy.deepcopy(f)
        bad_rel_f["graph_edges"].append(("User", "u1", "likes", "Product", "p1", 1.0))
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rel_f)

        # Endpoint not in nodes
        missing_node_f = copy.deepcopy(f)
        missing_node_f["graph_edges"].append(("User", "u1", "rated", "Product", "p_unknown", 1.0))
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**missing_node_f)

        # Relation schema type mismatch
        mismatch_rel_f = copy.deepcopy(f)
        mismatch_rel_f["graph_edges"].append(("Product", "p1", "rated", "User", "u1", 1.0))
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**mismatch_rel_f)

        # Invalid CategoryPath ID format
        with self.assertRaises(CheckpointValidationError):
            validate_category_id("c:Electronics > Headphones")
        with self.assertRaises(CheckpointValidationError):
            validate_category_id("c:[]")  # empty list
        with self.assertRaises(CheckpointValidationError):
            validate_category_id('c:[" Electronics"]')  # whitespace padding

        # UTF-8 literal category ID: producer-compatible oracle and escaped alias rejection
        parsed = validate_category_id('c:["雪"]')
        self.assertEqual(parsed, ["雪"])
        self.assertEqual(format_canonical_category_id(["雪"]), 'c:["雪"]')
        with self.assertRaises(CheckpointValidationError):
            validate_category_id('c:["\\u96ea"]')

        # Invalid Brand ID format
        with self.assertRaises(CheckpointValidationError):
            validate_brand_id("b:")
        with self.assertRaises(CheckpointValidationError):
            validate_brand_id("b: Sony ")  # whitespace padding
        with self.assertRaises(CheckpointValidationError):
            validate_brand_id("Sony")  # missing prefix

    def test_validation_extra_missing_fields_and_target_record_inconsistency(self):
        f = _make_valid_identity_fixture()

        # 1. Extra unknown field in validation_inputs
        bad_val_extra = copy.deepcopy(f)
        bad_val_extra["validation_inputs"]["extra_field"] = "bad"
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_val_extra)

        # 2. Both targets and records None
        bad_val_empty = copy.deepcopy(f)
        bad_val_empty["validation_inputs"] = {"targets": None, "records": None}
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_val_empty)

        # 3. Validation record dict missing a key
        bad_rec_missing_k = copy.deepcopy(f)
        bad_rec_missing_k["validation_inputs"]["records"] = [
            {"user_id": "u1", "parent_asin": "p3", "rating": 5.0}  # missing timestamp
        ]
        bad_rec_missing_k["validation_inputs"]["targets"] = {"u1": ["p3"]}
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rec_missing_k)

        # 4. Validation record tuple with length != 4
        bad_rec_len = copy.deepcopy(f)
        bad_rec_len["validation_inputs"]["records"] = [("u1", "p3", 5.0)]
        bad_rec_len["validation_inputs"]["targets"] = {"u1": ["p3"]}
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rec_len)

        # 5. Target / record inconsistency: target has u1/p3, record has u1/p2
        bad_val_mismatch = copy.deepcopy(f)
        bad_val_mismatch["validation_inputs"]["targets"]["u1"] = ["p2"]
        # records still has u1 -> p3
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_val_mismatch)

        # 6. Target / record inconsistency: user mismatch
        bad_val_user = copy.deepcopy(f)
        bad_val_user["validation_inputs"]["targets"] = {"u1": ["p3"]}
        bad_val_user["validation_inputs"]["records"] = [("u2", "p3", 5.0, 5000)]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_val_user)

        # 7. Duplicate target item
        bad_dup_target = copy.deepcopy(f)
        bad_dup_target["validation_inputs"]["targets"] = {"u1": ["p3", "p3"]}
        bad_dup_target["validation_inputs"]["records"] = None
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_dup_target)

        # 8. Duplicate validation record pair
        bad_dup_v_rec = copy.deepcopy(f)
        bad_dup_v_rec["validation_inputs"]["targets"] = None
        bad_dup_v_rec["validation_inputs"]["records"] = [
            ("u1", "p3", 5.0, 1000),
            ("u1", "p3", 5.0, 2000),
        ]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_dup_v_rec)

    def test_every_settings_section_field_and_mismatch(self):
        f = _make_valid_identity_fixture()
        base_ident = build_canonical_identity(**f)

        # 1. Test unknown top-level section in resolved_settings
        bad_top = copy.deepcopy(f)
        bad_top["resolved_settings"]["unknown_top_section"] = {"foo": "bar"}
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_top)

        # 2. Test every section and field
        for sec_name, fields in REQUIRED_SETTINGS_FIELDS.items():
            # Test missing section
            bad_sec = copy.deepcopy(f)
            del bad_sec["resolved_settings"][sec_name]
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject missing section {sec_name}"):
                build_canonical_identity(**bad_sec)

            # Test extra field in section
            extra_fld = copy.deepcopy(f)
            extra_fld["resolved_settings"][sec_name]["extra_field"] = "bad"
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject extra field in section {sec_name}"):
                build_canonical_identity(**extra_fld)

            for fld in fields:
                # Test missing field
                miss_fld = copy.deepcopy(f)
                del miss_fld["resolved_settings"][sec_name][fld]
                with self.assertRaises(CheckpointValidationError, msg=f"Should reject missing field {sec_name}.{fld}"):
                    build_canonical_identity(**miss_fld)

                # Test mutating field changes fingerprint and causes mismatch error
                mut_f = copy.deepcopy(f)
                val = mut_f["resolved_settings"][sec_name][fld]

                if sec_name == "execution" and fld == "runtime_workers":
                    mut_f["resolved_settings"]["execution"]["runtime_workers"] = 2
                    mut_f["resolved_settings"]["execution"]["worker_settings"] = {
                        "prefetch_factor": 2,
                        "persistent_workers": True,
                        "multiprocessing_context": "fork",
                        "worker_init_function": "worker_init_fn",
                    }
                    mut_f["resolved_settings"]["determinism"]["worker_derivation"] = {
                        "applicable": True,
                        "algorithm": "worker_seed_fn",
                        "base_seed": mut_f["resolved_settings"]["determinism"]["seed"],
                        "derivation": "base_plus_worker_id",
                        "reason": None,
                    }
                    for stream in mut_f["resolved_settings"]["determinism"]["rng_streams"]:
                        if stream["name"] == "DataLoader_workers":
                            stream["applicable"] = True
                            stream["algorithm"] = "worker_stream_synthetic"
                            stream["seed"] = 1042
                            stream["derivation"] = "offset_seed"
                            stream["reason"] = None
                elif sec_name == "execution" and fld == "worker_settings":
                    mut_f["resolved_settings"]["execution"]["runtime_workers"] = 1
                    mut_f["resolved_settings"]["execution"]["worker_settings"] = {
                        "prefetch_factor": 4,
                        "persistent_workers": True,
                        "multiprocessing_context": "spawn",
                        "worker_init_function": "worker_init_fn_v2",
                    }
                    mut_f["resolved_settings"]["determinism"]["worker_derivation"] = {
                        "applicable": True,
                        "algorithm": "worker_seed_fn",
                        "base_seed": mut_f["resolved_settings"]["determinism"]["seed"],
                        "derivation": "base_plus_worker_id",
                        "reason": None,
                    }
                    for stream in mut_f["resolved_settings"]["determinism"]["rng_streams"]:
                        if stream["name"] == "DataLoader_workers":
                            stream["applicable"] = True
                            stream["algorithm"] = "worker_stream_synthetic"
                            stream["seed"] = 1042
                            stream["derivation"] = "offset_seed"
                            stream["reason"] = None
                elif sec_name == "execution" and fld == "cpu_threads":
                    mut_f["resolved_settings"]["execution"]["cpu_threads"] = 8
                elif sec_name == "determinism" and fld == "worker_derivation":
                    mut_f["resolved_settings"]["execution"]["runtime_workers"] = 1
                    mut_f["resolved_settings"]["execution"]["worker_settings"] = {
                        "prefetch_factor": 2,
                        "persistent_workers": False,
                        "multiprocessing_context": "fork",
                        "worker_init_function": "worker_init_fn",
                    }
                    mut_f["resolved_settings"]["determinism"]["worker_derivation"] = {
                        "applicable": True,
                        "algorithm": "worker_seed_fn_alt",
                        "base_seed": mut_f["resolved_settings"]["determinism"]["seed"],
                        "derivation": "derived_offset_v2",
                        "reason": None,
                    }
                    for stream in mut_f["resolved_settings"]["determinism"]["rng_streams"]:
                        if stream["name"] == "DataLoader_workers":
                            stream["applicable"] = True
                            stream["algorithm"] = "worker_stream_synthetic"
                            stream["seed"] = 1042
                            stream["derivation"] = "offset_seed"
                            stream["reason"] = None
                elif sec_name == "determinism" and fld == "rng_streams":
                    mut_f["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = 43
                elif isinstance(val, bool):
                    mut_f["resolved_settings"][sec_name][fld] = not val
                elif isinstance(val, int):
                    mut_f["resolved_settings"][sec_name][fld] = val + 1
                elif isinstance(val, float):
                    mut_f["resolved_settings"][sec_name][fld] = val + 0.01 if val == 0.0 else val * 2.0
                elif isinstance(val, str):
                    mut_f["resolved_settings"][sec_name][fld] = val + "_mutated"
                elif isinstance(val, list):
                    if fld == "adam_betas":
                        mut_f["resolved_settings"][sec_name][fld] = [0.8, 0.99]
                    elif fld == "neighbor_fanouts":
                        mut_f["resolved_settings"][sec_name][fld] = [5, 5]

                mut_ident = build_canonical_identity(**mut_f)
                self.assertNotEqual(
                    base_ident.settings_fingerprint,
                    mut_ident.settings_fingerprint,
                    f"Mutation of {sec_name}.{fld} must alter settings_fingerprint",
                )
                with self.assertRaises(CheckpointIdentityMismatchError, msg=f"Mismatch error on {sec_name}.{fld}"):
                    validate_canonical_identity(mut_ident, base_ident)

        # Test specific invalid types
        # 1. bool for int fields
        for sec, fld in [
            ("execution", "runtime_workers"),
            ("execution", "cpu_threads"),
            ("model", "embedding_dim"),
            ("model", "num_layers"),
            ("training", "batch_size"),
            ("training", "max_epochs"),
            ("training", "patience"),
            ("training", "validation_frequency"),
            ("sampling", "negatives_per_positive"),
            ("determinism", "seed"),
        ]:
            bad_type = copy.deepcopy(f)
            bad_type["resolved_settings"][sec][fld] = True
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject bool for {sec}.{fld}"):
                build_canonical_identity(**bad_type)

        # 2. non-finite for float fields
        for sec, fld in [
            ("training", "adam_epsilon"),
            ("training", "endpoint_regularization"),
            ("training", "learning_rate"),
            ("training", "weight_decay"),
        ]:
            bad_flt = copy.deepcopy(f)
            bad_flt["resolved_settings"][sec][fld] = float("nan")
            with self.assertRaises(CheckpointValidationError, msg=f"Should reject NaN for {sec}.{fld}"):
                build_canonical_identity(**bad_flt)

        # 3. invalid adam_betas
        for bad_beta in ([0.9], [0.9, 0.999, 0.9999], [0.0, 0.999], [0.9, 1.0], "0.9,0.999"):
            bad_betas_f = copy.deepcopy(f)
            bad_betas_f["resolved_settings"]["training"]["adam_betas"] = bad_beta
            with self.assertRaises(CheckpointValidationError):
                build_canonical_identity(**bad_betas_f)

        # 4. invalid neighbor_fanouts
        for bad_fanout in ([-1], [True], "15,10", {"fanout": 10}):
            bad_fanout_f = copy.deepcopy(f)
            bad_fanout_f["resolved_settings"]["sampling"]["neighbor_fanouts"] = bad_fanout
            with self.assertRaises(CheckpointValidationError):
                build_canonical_identity(**bad_fanout_f)

        # 5. invalid runtime_workers
        for bad_rw in (-1, 0.0, 1.5, "0", None):
            bad_rw_f = copy.deepcopy(f)
            bad_rw_f["resolved_settings"]["execution"]["runtime_workers"] = bad_rw
            with self.assertRaises(CheckpointValidationError):
                build_canonical_identity(**bad_rw_f)

        # 6. invalid cpu_threads
        for bad_ct in (0, -1, 4.0, "4", None):
            bad_ct_f = copy.deepcopy(f)
            bad_ct_f["resolved_settings"]["execution"]["cpu_threads"] = bad_ct
            with self.assertRaises(CheckpointValidationError):
                build_canonical_identity(**bad_ct_f)

        # 7. invalid worker_settings
        # Zero workers with non-None prefetch, persistent_workers True, etc.
        bad_ws1 = copy.deepcopy(f)
        bad_ws1["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = 2
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_ws1)

        bad_ws2 = copy.deepcopy(f)
        bad_ws2["resolved_settings"]["execution"]["worker_settings"]["persistent_workers"] = True
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_ws2)

        bad_ws3 = copy.deepcopy(f)
        bad_ws3["resolved_settings"]["execution"]["worker_settings"]["multiprocessing_context"] = "fork"
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_ws3)

        bad_ws4 = copy.deepcopy(f)
        bad_ws4["resolved_settings"]["execution"]["worker_settings"]["worker_init_function"] = "init_fn"
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_ws4)

        # Positive workers with invalid worker_settings
        bad_pos_ws = copy.deepcopy(f)
        bad_pos_ws["resolved_settings"]["execution"]["runtime_workers"] = 2
        bad_pos_ws["resolved_settings"]["execution"]["worker_settings"] = {
            "prefetch_factor": None,  # must be positive int
            "persistent_workers": True,
            "multiprocessing_context": "fork",
            "worker_init_function": "init_fn",
        }
        bad_pos_ws["resolved_settings"]["determinism"]["worker_derivation"] = {
            "applicable": True,
            "algorithm": "w_alg",
            "base_seed": 42,
            "derivation": "w_drv",
            "reason": None,
        }
        for stream in bad_pos_ws["resolved_settings"]["determinism"]["rng_streams"]:
            if stream["name"] == "DataLoader_workers":
                stream["applicable"] = True
                stream["algorithm"] = "w_alg"
                stream["seed"] = 1042
                stream["derivation"] = "w_drv"
                stream["reason"] = None
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_pos_ws)

        # 8. invalid rng_streams
        # Missing component, unknown component, duplicate component
        bad_rng_missing = copy.deepcopy(f)
        bad_rng_missing["resolved_settings"]["determinism"]["rng_streams"] = bad_rng_missing["resolved_settings"]["determinism"]["rng_streams"][:-1]
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rng_missing)

        bad_rng_extra = copy.deepcopy(f)
        bad_rng_extra["resolved_settings"]["determinism"]["rng_streams"].append({
            "name": "unknown_stream",
            "applicable": False,
            "algorithm": None,
            "seed": None,
            "derivation": None,
            "reason": "synthetic_reason",
        })
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rng_extra)

        bad_rng_dup = copy.deepcopy(f)
        bad_rng_dup["resolved_settings"]["determinism"]["rng_streams"][1] = copy.deepcopy(bad_rng_dup["resolved_settings"]["determinism"]["rng_streams"][0])
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rng_dup)

        # Inapplicable stream with non-None algorithm/seed/derivation
        bad_rng_inapp = copy.deepcopy(f)
        bad_rng_inapp["resolved_settings"]["determinism"]["rng_streams"][3]["algorithm"] = "should_be_none"
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rng_inapp)

        # Applicable stream with invalid seed / reason
        bad_rng_app_seed = copy.deepcopy(f)
        bad_rng_app_seed["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = -1
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rng_app_seed)

        bad_rng_app_rsn = copy.deepcopy(f)
        bad_rng_app_rsn["resolved_settings"]["determinism"]["rng_streams"][0]["reason"] = "should_be_none"
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_rng_app_rsn)

        # 9. rng_streams order sensitivity (permutation changes fingerprint)
        rng_order_f = copy.deepcopy(f)
        rng_order_f["resolved_settings"]["determinism"]["rng_streams"] = list(
            reversed(f["resolved_settings"]["determinism"]["rng_streams"])
        )
        rng_order_ident = build_canonical_identity(**rng_order_f)
        self.assertNotEqual(
            base_ident.settings_fingerprint,
            rng_order_ident.settings_fingerprint,
            "Permuting rng_streams must alter settings_fingerprint",
        )
        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(rng_order_ident, base_ident)

        # 10. Consistency failures:
        # worker_derivation applicable=True when runtime_workers=0
        bad_wd_incon = copy.deepcopy(f)
        bad_wd_incon["resolved_settings"]["determinism"]["worker_derivation"] = {
            "applicable": True,
            "algorithm": "alg",
            "base_seed": 42,
            "derivation": "drv",
            "reason": None,
        }
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_wd_incon)

        # base_seed mismatch with determinism.seed
        bad_wd_seed = copy.deepcopy(f)
        bad_wd_seed["resolved_settings"]["execution"]["runtime_workers"] = 1
        bad_wd_seed["resolved_settings"]["execution"]["worker_settings"] = {
            "prefetch_factor": 2,
            "persistent_workers": False,
            "multiprocessing_context": "fork",
            "worker_init_function": "fn",
        }
        bad_wd_seed["resolved_settings"]["determinism"]["worker_derivation"] = {
            "applicable": True,
            "algorithm": "alg",
            "base_seed": 9999,  # != determinism.seed (42)
            "derivation": "drv",
            "reason": None,
        }
        for stream in bad_wd_seed["resolved_settings"]["determinism"]["rng_streams"]:
            if stream["name"] == "DataLoader_workers":
                stream["applicable"] = True
                stream["algorithm"] = "alg"
                stream["seed"] = 1042
                stream["derivation"] = "drv"
                stream["reason"] = None
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_wd_seed)

        # DataLoader_workers applicability mismatch with worker_derivation
        bad_dl_incon = copy.deepcopy(f)
        bad_dl_incon["resolved_settings"]["execution"]["runtime_workers"] = 1
        bad_dl_incon["resolved_settings"]["execution"]["worker_settings"] = {
            "prefetch_factor": 2,
            "persistent_workers": False,
            "multiprocessing_context": "fork",
            "worker_init_function": "fn",
        }
        bad_dl_incon["resolved_settings"]["determinism"]["worker_derivation"] = {
            "applicable": True,
            "algorithm": "alg",
            "base_seed": 42,
            "derivation": "drv",
            "reason": None,
        }
        # Leave DataLoader_workers inapplicable (mismatch!)
        with self.assertRaises(CheckpointValidationError):
            build_canonical_identity(**bad_dl_incon)

    def test_falsified_and_tampered_fingerprints_rejected(self):
        f = _make_valid_identity_fixture()
        ident = build_canonical_identity(**f)

        for fp_attr in ("data_fingerprint", "graph_fingerprint", "validation_fingerprint", "settings_fingerprint", "identity_hash"):
            tampered = CanonicalResumeIdentity(
                method=ident.method,
                category=ident.category,
                feedback_condition=ident.feedback_condition,
                seed=ident.seed,
                data_fingerprint="0" * 64 if fp_attr == "data_fingerprint" else ident.data_fingerprint,
                graph_fingerprint="0" * 64 if fp_attr == "graph_fingerprint" else ident.graph_fingerprint,
                validation_fingerprint="0" * 64 if fp_attr == "validation_fingerprint" else ident.validation_fingerprint,
                settings_fingerprint="0" * 64 if fp_attr == "settings_fingerprint" else ident.settings_fingerprint,
                identity_hash="0" * 64 if fp_attr == "identity_hash" else ident.identity_hash,
                training_records=ident.training_records,
                positive_history=ident.positive_history,
                all_rated_history=ident.all_rated_history,
                warm_items=ident.warm_items,
                graph_nodes=ident.graph_nodes,
                graph_edges=ident.graph_edges,
                product_metadata=ident.product_metadata,
                validation_inputs=ident.validation_inputs,
                resolved_settings=ident.resolved_settings,
            )
            with self.assertRaises(CheckpointIdentityMismatchError, msg=f"Should reject tampered {fp_attr}"):
                validate_canonical_identity(tampered, ident)

    def test_corrupt_actual_and_expected_objects_domain_errors(self):
        f = _make_valid_identity_fixture()
        ident = build_canonical_identity(**f)

        # Pass non-CanonicalResumeIdentity
        with self.assertRaises(CheckpointValidationError):
            validate_canonical_identity("not_an_identity", ident)
        with self.assertRaises(CheckpointValidationError):
            validate_canonical_identity(ident, {"not": "an_identity"})

        # Pass corrupt retained collections inside actual or expected
        for corrupt_field in (
            "positive_history",
            "all_rated_history",
            "training_records",
            "warm_items",
            "graph_nodes",
            "graph_edges",
            "product_metadata",
            "validation_inputs",
            "resolved_settings",
        ):
            # Test in actual
            corrupt_actual = copy.deepcopy(ident)
            object.__setattr__(corrupt_actual, corrupt_field, "corrupt_string_not_collection")
            with self.assertRaises(CheckpointValidationError, msg=f"Should raise CheckpointValidationError on corrupt {corrupt_field}"):
                validate_canonical_identity(corrupt_actual, ident)

            # Test in expected
            corrupt_expected = copy.deepcopy(ident)
            object.__setattr__(corrupt_expected, corrupt_field, "corrupt_string_not_collection")
            with self.assertRaises(CheckpointValidationError, msg=f"Should raise CheckpointValidationError on corrupt expected {corrupt_field}"):
                validate_canonical_identity(ident, corrupt_expected)

        # Corrupt header fields in actual and expected
        for header_field in (
            "method",
            "category",
            "feedback_condition",
            "seed",
            "data_fingerprint",
            "graph_fingerprint",
            "validation_fingerprint",
            "settings_fingerprint",
            "identity_hash",
        ):
            # In actual
            corrupt_act = copy.deepcopy(ident)
            object.__setattr__(corrupt_act, header_field, "   " if "fingerprint" not in header_field and header_field != "seed" else (123 if "fingerprint" in header_field else "not_int"))
            with self.assertRaises(CheckpointValidationError, msg=f"Should raise CheckpointValidationError on corrupt actual.{header_field}"):
                validate_canonical_identity(corrupt_act, ident)

            # In expected
            corrupt_exp = copy.deepcopy(ident)
            object.__setattr__(corrupt_exp, header_field, "   " if "fingerprint" not in header_field and header_field != "seed" else (123 if "fingerprint" in header_field else "not_int"))
            with self.assertRaises(CheckpointValidationError, msg=f"Should raise CheckpointValidationError on corrupt expected.{header_field}"):
                validate_canonical_identity(ident, corrupt_exp)

        # Unhashable / corrupt elements inside collections in actual and expected
        # 1. Non-string user key in positive_history without dropping u2 and keeping valid p1
        c_act1 = copy.deepcopy(ident)
        object.__setattr__(c_act1, "positive_history", {123: ["p1"], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"actual\.positive_history key must be a string, got int",
        ):
            validate_canonical_identity(c_act1, ident)

        c_exp1 = copy.deepcopy(ident)
        object.__setattr__(c_exp1, "positive_history", {123: ["p1"], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"expected\.positive_history key must be a string, got int",
        ):
            validate_canonical_identity(ident, c_exp1)

        # 1b. Non-string user key in all_rated_history without dropping u2 and keeping valid p1
        c_act1_all = copy.deepcopy(ident)
        object.__setattr__(c_act1_all, "all_rated_history", {123: ["p1", "p2"], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"actual\.all_rated_history key must be a string, got int",
        ):
            validate_canonical_identity(c_act1_all, ident)

        c_exp1_all = copy.deepcopy(ident)
        object.__setattr__(c_exp1_all, "all_rated_history", {123: ["p1", "p2"], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"expected\.all_rated_history key must be a string, got int",
        ):
            validate_canonical_identity(ident, c_exp1_all)

        # 2. Unhashable / corrupt items in history list without dropping u2 and keeping valid p1
        c_act2 = copy.deepcopy(ident)
        object.__setattr__(c_act2, "all_rated_history", {"u1": ["p1", {"bad": "dict"}], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"Invalid item ID in all_rated_history for user 'u1' must be a string, got dict",
        ):
            validate_canonical_identity(c_act2, ident)

        c_exp2 = copy.deepcopy(ident)
        object.__setattr__(c_exp2, "all_rated_history", {"u1": ["p1", {"bad": "dict"}], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"Invalid item ID in all_rated_history for user 'u1' must be a string, got dict",
        ):
            validate_canonical_identity(ident, c_exp2)

        # 3. Corrupt positive_history elements without dropping u2 and keeping valid p1
        c_act_pos = copy.deepcopy(ident)
        object.__setattr__(c_act_pos, "positive_history", {"u1": ["p1", 123], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"Invalid item ID in positive_history for user 'u1' must be a string, got int",
        ):
            validate_canonical_identity(c_act_pos, ident)

        c_exp_pos = copy.deepcopy(ident)
        object.__setattr__(c_exp_pos, "positive_history", {"u1": ["p1", 123], "u2": ["p2", "p3"]})
        with self.assertRaisesRegex(
            CheckpointValidationError,
            r"Invalid item ID in positive_history for user 'u1' must be a string, got int",
        ):
            validate_canonical_identity(ident, c_exp_pos)

        # 4. Corrupt nested settings sub-structures in actual and expected
        for nest_corrupt_fn, exp_regex in [
            (
                lambda obj: obj.resolved_settings["determinism"].__setitem__("rng_streams", []),
                r"determinism\.rng_streams cannot be empty",
            ),
            (
                lambda obj: obj.resolved_settings["execution"].__setitem__("worker_settings", {"bad": 1}),
                r"execution\.worker_settings fields mismatch",
            ),
            (
                lambda obj: obj.resolved_settings["execution"].__setitem__("cpu_threads", True),
                r"Field execution\.cpu_threads must be an integer \(not boolean\)",
            ),
            (
                lambda obj: obj.resolved_settings["determinism"].__setitem__("worker_derivation", "not_a_dict"),
                r"Field determinism\.worker_derivation must be a mapping",
            ),
        ]:
            c_act_nest = copy.deepcopy(ident)
            nest_corrupt_fn(c_act_nest)
            with self.assertRaisesRegex(CheckpointValidationError, exp_regex):
                validate_canonical_identity(c_act_nest, ident)

            c_exp_nest = copy.deepcopy(ident)
            nest_corrupt_fn(c_exp_nest)
            with self.assertRaisesRegex(CheckpointValidationError, exp_regex):
                validate_canonical_identity(ident, c_exp_nest)

    def test_oversized_worker_seed_mismatch_diagnostics(self):
        """Regression tests for oversized worker base_seed and determinism.seed mismatch diagnostics.

        Covers both uncoupled mismatch directions using 10**5000 (5001 decimal digits)
        from complete otherwise-valid positive-worker fixtures across:
        - compute_settings_fingerprint
        - build_canonical_identity
        - CanonicalResumeIdentity.from_dict (in-memory to_dict snapshot mutation)
        - validate_canonical_identity (actual role)
        - validate_canonical_identity (expected role)

        Asserts CheckpointValidationError and stable worker_derivation.base_seed/determinism.seed
        consistency guard text, ensuring _safe_repr formats oversized integers without leaking
        native ValueError prior to domain validation error construction.
        """
        huge_seed = 10**5000
        guard_regex = r"determinism\.worker_derivation\.base_seed.*must agree with determinism\.seed"
        pos_f = _make_valid_positive_worker_identity_fixture()
        base_pos_ident = build_canonical_identity(**pos_f)

        # Direction 1: worker_derivation.base_seed is 10**5000, determinism.seed remains 42
        with self.subTest(mismatch_direction="base_seed_oversized", target_api="compute_settings_fingerprint"):
            s1 = _make_valid_positive_worker_settings_fixture()
            s1["determinism"]["worker_derivation"]["base_seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                compute_settings_fingerprint(s1)

        with self.subTest(mismatch_direction="base_seed_oversized", target_api="build_canonical_identity"):
            f1 = copy.deepcopy(pos_f)
            f1["resolved_settings"]["determinism"]["worker_derivation"]["base_seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                build_canonical_identity(**f1)

        with self.subTest(mismatch_direction="base_seed_oversized", target_api="from_dict"):
            d1 = base_pos_ident.to_dict()
            d1["resolved_settings"]["determinism"]["worker_derivation"]["base_seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                CanonicalResumeIdentity.from_dict(d1)

        with self.subTest(mismatch_direction="base_seed_oversized", target_api="validate_canonical_identity_actual"):
            corrupt_act1 = copy.deepcopy(base_pos_ident)
            corrupt_act1.resolved_settings["determinism"]["worker_derivation"]["base_seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                validate_canonical_identity(corrupt_act1, base_pos_ident)

        with self.subTest(mismatch_direction="base_seed_oversized", target_api="validate_canonical_identity_expected"):
            corrupt_exp1 = copy.deepcopy(base_pos_ident)
            corrupt_exp1.resolved_settings["determinism"]["worker_derivation"]["base_seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                validate_canonical_identity(base_pos_ident, corrupt_exp1)

        # Direction 2: determinism.seed is 10**5000, worker_derivation.base_seed remains 42
        with self.subTest(mismatch_direction="determinism_seed_oversized", target_api="compute_settings_fingerprint"):
            s2 = _make_valid_positive_worker_settings_fixture()
            s2["determinism"]["seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                compute_settings_fingerprint(s2)

        with self.subTest(mismatch_direction="determinism_seed_oversized", target_api="build_canonical_identity"):
            f2 = copy.deepcopy(pos_f)
            f2["resolved_settings"]["determinism"]["seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                build_canonical_identity(**f2)

        with self.subTest(mismatch_direction="determinism_seed_oversized", target_api="from_dict"):
            d2 = base_pos_ident.to_dict()
            d2["resolved_settings"]["determinism"]["seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                CanonicalResumeIdentity.from_dict(d2)

        with self.subTest(mismatch_direction="determinism_seed_oversized", target_api="validate_canonical_identity_actual"):
            corrupt_act2 = copy.deepcopy(base_pos_ident)
            corrupt_act2.resolved_settings["determinism"]["seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                validate_canonical_identity(corrupt_act2, base_pos_ident)

        with self.subTest(mismatch_direction="determinism_seed_oversized", target_api="validate_canonical_identity_expected"):
            corrupt_exp2 = copy.deepcopy(base_pos_ident)
            corrupt_exp2.resolved_settings["determinism"]["seed"] = huge_seed
            with self.assertRaisesRegex(CheckpointValidationError, guard_regex):
                validate_canonical_identity(base_pos_ident, corrupt_exp2)


class SourceProvenanceContractTests(unittest.TestCase):
    """Contract tests for allowlisted source and environment provenance."""

    def test_literal_source_provenance_oracle(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        prov = build_source_provenance(src_files, git_ev, env_rec)

        expected_tree_hash = "1b86bce2e313d0a4588a457965e6dd076b91691a094170d0d66809c4b746d565"
        expected_total_bytes = 66

        self.assertEqual(prov["schema_version"], PROVENANCE_SCHEMA_VERSION)
        self.assertEqual(prov["source_tree_hash"], expected_tree_hash)
        self.assertEqual(prov["total_source_bytes"], expected_total_bytes)

        validate_source_provenance(prov, prov)

    def test_source_provenance_tampering_and_mismatch(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        prov = build_source_provenance(src_files, git_ev, env_rec)

        # Tamper content bytes
        bad_bytes_prov = copy.deepcopy(prov)
        bad_bytes_prov["source_files"]["scripts/checkpoint_identity.py"]["content_bytes"] = b"# mutated\n"
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_bytes_prov, prov)

        # Tamper sha256
        bad_sha_prov = copy.deepcopy(prov)
        bad_sha_prov["source_files"]["scripts/checkpoint_identity.py"]["sha256"] = "f" * 64
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_sha_prov, prov)

        # Tamper size
        bad_sz_prov = copy.deepcopy(prov)
        bad_sz_prov["source_files"]["scripts/checkpoint_identity.py"]["size_bytes"] = 999
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_sz_prov, prov)

        # Tamper total_source_bytes
        bad_total_prov = copy.deepcopy(prov)
        bad_total_prov["total_source_bytes"] = 999
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_total_prov, prov)

        # Tamper source_tree_hash
        bad_tree_prov = copy.deepcopy(prov)
        bad_tree_prov["source_tree_hash"] = "0" * 64
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_tree_prov, prov)

        # Omit file from actual
        missing_f_prov = copy.deepcopy(prov)
        del missing_f_prov["source_files"]["scripts/checkpoint_identity.py"]
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(missing_f_prov, prov)

        # Non-bytes content in source_files
        bad_content_type = {"scripts/checkpoint_identity.py": "not_bytes"}
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(bad_content_type, git_ev, env_rec)

    def test_git_evidence_adversarial_validation(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        prov = build_source_provenance(src_files, git_ev, env_rec)

        # Mismatch commit
        bad_commit_prov = copy.deepcopy(prov)
        bad_commit_prov["git"]["commit"] = "0" * 40
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_commit_prov, prov)

        # Invalid commit (non-hex)
        bad_git_ev = copy.deepcopy(git_ev)
        bad_git_ev["commit"] = "not-a-hex-commit"
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, bad_git_ev, env_rec)

        # Dirty True but empty diff and untracked
        bad_dirty_git = copy.deepcopy(git_ev)
        bad_dirty_git["dirty"] = True
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, bad_dirty_git, env_rec)

        # Dirty False but non-empty tracked_diff
        bad_clean_git = copy.deepcopy(git_ev)
        bad_clean_git["dirty"] = False
        bad_clean_git["tracked_diff"] = "diff --git a/file b/file"
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, bad_clean_git, env_rec)

        # Dirty flag not boolean
        bad_dirty_type = copy.deepcopy(git_ev)
        bad_dirty_type["dirty"] = 1
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, bad_dirty_type, env_rec)

        # Untracked source missing from source_files
        bad_untracked_missing_bytes = copy.deepcopy(git_ev)
        bad_untracked_missing_bytes["dirty"] = True
        bad_untracked_missing_bytes["untracked_sources"] = ["scripts/learned_models.py"]
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, bad_untracked_missing_bytes, env_rec)

        # Duplicate untracked source
        src_with_lm = dict(src_files, **{"scripts/learned_models.py": b"# lm\n"})
        bad_untracked_dup = copy.deepcopy(git_ev)
        bad_untracked_dup["dirty"] = True
        bad_untracked_dup["untracked_sources"] = ["scripts/learned_models.py", "scripts/learned_models.py"]
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_with_lm, bad_untracked_dup, env_rec)

        # Untracked sources non-allowlisted / path traversal
        bad_untracked_path = copy.deepcopy(git_ev)
        bad_untracked_path["dirty"] = True
        bad_untracked_path["untracked_sources"] = ["scripts/../outside.py"]
        with self.assertRaises(CheckpointSecurityError):
            build_source_provenance(src_files, bad_untracked_path, env_rec)

    def test_environment_record_tampering_and_adversarial_validation(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        prov = build_source_provenance(src_files, git_ev, env_rec)

        # Tamper python_version
        bad_env_prov = copy.deepcopy(prov)
        bad_env_prov["environment"]["python_version"] = "3.11.0"
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_env_prov, prov)

        # Empty python_version in builder
        bad_env = copy.deepcopy(env_rec)
        bad_env["python_version"] = "   "
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, git_ev, bad_env)

        # cpu_count non-positive or bool
        bad_cpu_env = copy.deepcopy(env_rec)
        bad_cpu_env["cpu_count"] = True
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, git_ev, bad_cpu_env)

        bad_cpu_zero = copy.deepcopy(env_rec)
        bad_cpu_zero["cpu_count"] = 0
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, git_ev, bad_cpu_zero)

        # Missing required environment key
        missing_env_key = copy.deepcopy(env_rec)
        del missing_env_key["torch_version"]
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, git_ev, missing_env_key)

        # Empty library_versions
        bad_libs = copy.deepcopy(env_rec)
        bad_libs["library_versions"] = {}
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, git_ev, bad_libs)

    def test_corrupt_actual_and_expected_provenance_objects(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        prov = build_source_provenance(src_files, git_ev, env_rec)

        # 1. Non-dict actual or expected
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance("not_dict", prov)
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(prov, None)
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(prov, [1, 2, 3])

        # 2. Missing and extra top-level keys in actual and expected
        for top_key in ("schema_version", "total_source_bytes", "source_tree_hash", "source_files", "git", "environment"):
            bad_act = copy.deepcopy(prov)
            del bad_act[top_key]
            with self.assertRaises(CheckpointProvenanceError, msg=f"Should reject actual missing {top_key}"):
                validate_source_provenance(bad_act, prov)

            bad_exp = copy.deepcopy(prov)
            del bad_exp[top_key]
            with self.assertRaises(CheckpointProvenanceError, msg=f"Should reject expected missing {top_key}"):
                validate_source_provenance(prov, bad_exp)

        bad_act_extra = copy.deepcopy(prov)
        bad_act_extra["extra_top_key"] = "bad"
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(bad_act_extra, prov)

        bad_exp_extra = copy.deepcopy(prov)
        bad_exp_extra["extra_top_key"] = "bad"
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(prov, bad_exp_extra)

        # 3. Corrupt inner sub-structures in actual and expected
        for corrupt_fn in [
            lambda p: p.__setitem__("git", None),
            lambda p: p.__setitem__("environment", "not_dict"),
            lambda p: p.__setitem__("source_files", {}),
            lambda p: p.__setitem__("total_source_bytes", True),
            lambda p: p.__setitem__("total_source_bytes", -1),
            lambda p: p["source_files"]["scripts/checkpoint_identity.py"].__setitem__("path", "scripts/other.py"),
            lambda p: p["source_files"]["scripts/checkpoint_identity.py"].__setitem__("content_bytes", "not_bytes"),
            lambda p: p["source_files"]["scripts/checkpoint_identity.py"].__setitem__("size_bytes", True),
            lambda p: p["source_files"]["scripts/checkpoint_identity.py"].__setitem__("sha256", "invalid_sha"),
            lambda p: p["source_files"]["scripts/checkpoint_identity.py"].pop("sha256"),
            lambda p: p["source_files"]["scripts/checkpoint_identity.py"].__setitem__("extra", 1),
            lambda p: p["git"].pop("commit"),
            lambda p: p["git"].__setitem__("extra", 1),
            lambda p: p["git"].__setitem__("commit", "not_a_40_hex_commit"),
            lambda p: p["git"].__setitem__("dirty", 1),
            lambda p: p["git"].__setitem__("tracked_diff", 123),
            lambda p: p["git"].__setitem__("untracked_sources", "not_a_list"),
            lambda p: p["git"].__setitem__("untracked_sources", ["scripts/checkpoint_identity.py", "scripts/checkpoint_identity.py"]),
            lambda p: p["git"].__setitem__("untracked_sources", ["configs/compute-plan.json"]),
            lambda p: p["environment"].pop("python_version"),
            lambda p: p["environment"].__setitem__("extra", 1),
            lambda p: p["environment"].__setitem__("cpu_count", True),
            lambda p: p["environment"].__setitem__("cpu_count", 0),
            lambda p: p["environment"].__setitem__("python_version", "   "),
            lambda p: p["environment"].__setitem__("library_versions", {}),
        ]:
            bad_act = copy.deepcopy(prov)
            corrupt_fn(bad_act)
            with self.assertRaises(CheckpointProvenanceError):
                validate_source_provenance(bad_act, prov)

            bad_exp = copy.deepcopy(prov)
            corrupt_fn(bad_exp)
            with self.assertRaises(CheckpointProvenanceError):
                validate_source_provenance(prov, bad_exp)

    def test_aggregate_byte_bound_rejection_before_hashing_mock_oracle(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        large_bytes = b"a" * (MAX_PROVENANCE_SOURCE_BYTES + 1)
        src_files_large = {"scripts/checkpoint_identity.py": large_bytes}

        def _mock_sha256(*args, **kwargs):
            raise AssertionError("hashlib.sha256 must NOT be called when aggregate bytes exceed limit!")

        # 1. build_source_provenance with oversized single file: zero hashing
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaises(CheckpointProvenanceError):
                build_source_provenance(src_files_large, git_ev, env_rec)

        # 2. build_source_provenance with small valid first file + oversized second file: zero hashing
        # Total bytes: 1024 + MAX_PROVENANCE_SOURCE_BYTES > MAX_PROVENANCE_SOURCE_BYTES
        src_split_overflow = {
            "scripts/checkpoint_identity.py": b"x" * 1024,
            "configs/training-semantics.json": b"y" * MAX_PROVENANCE_SOURCE_BYTES,
        }
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaises(CheckpointProvenanceError):
                build_source_provenance(src_split_overflow, git_ev, env_rec)

        # 3. validate_source_provenance with early-declared-total overflow: zero hashing
        valid_prov = build_source_provenance(src_files, git_ev, env_rec)
        corrupt_actual_declared_over = copy.deepcopy(valid_prov)
        corrupt_actual_declared_over["total_source_bytes"] = MAX_PROVENANCE_SOURCE_BYTES + 1
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaises(CheckpointProvenanceError):
                validate_source_provenance(corrupt_actual_declared_over, valid_prov)

        corrupt_expected_declared_over = copy.deepcopy(valid_prov)
        corrupt_expected_declared_over["total_source_bytes"] = MAX_PROVENANCE_SOURCE_BYTES + 1
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaises(CheckpointProvenanceError):
                validate_source_provenance(valid_prov, corrupt_expected_declared_over)

        # 4. validate_source_provenance with declared total WITHIN limit (2048 bytes),
        # but small first file (1024 bytes) + oversized second file (MAX_PROVENANCE_SOURCE_BYTES bytes)
        # reaching aggregate-content guard before hashing:
        # Test in ACTUAL role:
        corrupt_act_aggregate = copy.deepcopy(valid_prov)
        corrupt_act_aggregate["total_source_bytes"] = 2048
        corrupt_act_aggregate["source_files"] = {
            "scripts/checkpoint_identity.py": {
                "path": "scripts/checkpoint_identity.py",
                "size_bytes": 1024,
                "sha256": "0" * 64,
                "content_bytes": b"a" * 1024,
            },
            "configs/training-semantics.json": {
                "path": "configs/training-semantics.json",
                "size_bytes": MAX_PROVENANCE_SOURCE_BYTES,
                "sha256": "0" * 64,
                "content_bytes": b"b" * MAX_PROVENANCE_SOURCE_BYTES,
            },
        }
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaises(CheckpointProvenanceError):
                validate_source_provenance(corrupt_act_aggregate, valid_prov)

        # Test in EXPECTED role:
        corrupt_exp_aggregate = copy.deepcopy(valid_prov)
        corrupt_exp_aggregate["total_source_bytes"] = 2048
        corrupt_exp_aggregate["source_files"] = {
            "scripts/checkpoint_identity.py": {
                "path": "scripts/checkpoint_identity.py",
                "size_bytes": 1024,
                "sha256": "0" * 64,
                "content_bytes": b"a" * 1024,
            },
            "configs/training-semantics.json": {
                "path": "configs/training-semantics.json",
                "size_bytes": MAX_PROVENANCE_SOURCE_BYTES,
                "sha256": "0" * 64,
                "content_bytes": b"b" * MAX_PROVENANCE_SOURCE_BYTES,
            },
        }
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaises(CheckpointProvenanceError):
                validate_source_provenance(valid_prov, corrupt_exp_aggregate)

    def test_exact_two_mib_boundary_acceptance(self):
        _, git_ev, env_rec = _make_valid_provenance_fixture()

        # Exact 2 MiB boundary: 2,097,152 bytes must be ACCEPTED
        exact_bytes = b"x" * MAX_PROVENANCE_SOURCE_BYTES
        src_exact = {"scripts/checkpoint_identity.py": exact_bytes}
        prov_exact = build_source_provenance(src_exact, git_ev, env_rec)
        self.assertEqual(prov_exact["total_source_bytes"], MAX_PROVENANCE_SOURCE_BYTES)
        validate_source_provenance(prov_exact, prov_exact)

        # 2 MiB + 1 byte: 2,097,153 bytes must be REJECTED by builder
        overflow_bytes = b"x" * (MAX_PROVENANCE_SOURCE_BYTES + 1)
        src_overflow = {"scripts/checkpoint_identity.py": overflow_bytes}
        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_overflow, git_ev, env_rec)

    def test_provenance_untracked_members_and_aggregate_diagnostics(self):
        """Matrix testing untracked members, path security in both roles, aggregate byte preflight, and snapshot isolation."""
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        untracked_path = "scripts/learned_models.py"
        src_files_dirty = dict(src_files, **{untracked_path: b"# learned models code\n"})
        git_ev_dirty = copy.deepcopy(git_ev)
        git_ev_dirty["dirty"] = True
        git_ev_dirty["untracked_sources"] = [untracked_path]

        base_prov = build_source_provenance(src_files_dirty, git_ev_dirty, env_rec)
        validate_source_provenance(base_prov, base_prov)

        # 1. Untracked MEMBERS: bool, int, dict, list, blank, padded in BOTH validator roles
        # Start with dirty=True, valid byte coverage, and otherwise valid records
        # Assert CheckpointProvenanceError
        bad_untracked_members = [
            ("bool_true", True),
            ("bool_false", False),
            ("int_val", 12345),
            ("dict_val", {"path": untracked_path}),
            ("list_val", [untracked_path]),
            ("blank_str", ""),
            ("whitespace_str", "   "),
            ("padded_left", f" {untracked_path}"),
            ("padded_right", f"{untracked_path} "),
            ("padded_both", f"  {untracked_path}  "),
        ]

        for val_role in ("actual", "expected"):
            for label, bad_member in bad_untracked_members:
                with self.subTest(val_role=val_role, member_case=label):
                    mutated = copy.deepcopy(base_prov)
                    mutated["git"]["untracked_sources"] = [bad_member]
                    if val_role == "actual":
                        with self.assertRaisesRegex(
                            CheckpointProvenanceError,
                            rf"{val_role} invalid path in untracked_sources",
                        ):
                            validate_source_provenance(mutated, base_prov)
                    else:
                        with self.assertRaisesRegex(
                            CheckpointProvenanceError,
                            rf"{val_role} invalid path in untracked_sources",
                        ):
                            validate_source_provenance(base_prov, mutated)

        # 2. Forbidden and traversal paths in untracked_sources in BOTH validator roles
        # Must raise CheckpointSecurityError (not CheckpointProvenanceError)
        forbidden_untracked_paths = [
            ("path_traversal_dotdot", "scripts/../configs/training-semantics.json"),
            ("path_traversal_root", "scripts/../../etc/passwd"),
            ("path_relative_dot", "scripts/./checkpoint_identity.py"),
            ("absolute_path_root", "/etc/passwd"),
            ("absolute_path_project", "/home/sunf/Amazon4U/scripts/checkpoint_identity.py"),
            ("disallowed_top_level_data", "data/raw/benchmark.csv"),
            ("disallowed_top_level_tests", "tests/test_checkpoint_identity.py"),
            ("disallowed_substring_git", "scripts/.git/HEAD"),
            ("disallowed_substring_venv", "scripts/.venv/bin/python"),
            ("credential_pattern_secret", "configs/secret.key"),
            ("non_allowlisted_script", "scripts/unregistered_script.py"),
        ]

        security_patterns = {
            "path_traversal_dotdot": r"Path traversal component",
            "path_traversal_root": r"Path traversal component",
            "path_relative_dot": r"Path traversal component",
            "absolute_path_root": r"Absolute paths are forbidden",
            "absolute_path_project": r"Absolute paths are forbidden",
            "disallowed_top_level_data": r"Path must reside in approved directories",
            "disallowed_top_level_tests": r"Path must reside in approved directories",
            "disallowed_substring_git": r"Path contains forbidden substring",
            "disallowed_substring_venv": r"Path contains forbidden substring",
            "credential_pattern_secret": r"Potential credential path",
            "non_allowlisted_script": r"Path is not in allowlisted source/config set",
        }

        for val_role in ("actual", "expected"):
            for label, bad_path in forbidden_untracked_paths:
                with self.subTest(val_role=val_role, security_case=label):
                    mutated = copy.deepcopy(base_prov)
                    mutated["git"]["untracked_sources"] = [bad_path]
                    if val_role == "actual":
                        with self.assertRaisesRegex(CheckpointSecurityError, security_patterns[label]):
                            validate_source_provenance(mutated, base_prov)
                    else:
                        with self.assertRaisesRegex(CheckpointSecurityError, security_patterns[label]):
                            validate_source_provenance(base_prov, mutated)

        # 3. Builder AND BOTH validator roles: late oversized file assertion of SPECIFIC diagnostic AND ZERO hashlib.sha256 calls
        def _mock_sha256(*args, **kwargs):
            raise AssertionError("hashlib.sha256 must NOT be called during aggregate-byte preflight!")

        # Builder: valid small first file (1024 bytes) + oversized second file (MAX_PROVENANCE_SOURCE_BYTES)
        src_split_overflow = {
            "scripts/checkpoint_identity.py": b"x" * 1024,
            "configs/training-semantics.json": b"y" * MAX_PROVENANCE_SOURCE_BYTES,
        }
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaisesRegex(
                CheckpointProvenanceError,
                r"(?i)total source bytes exceed 2 mib limit",
            ):
                build_source_provenance(src_split_overflow, git_ev, env_rec)

        # Validators (actual and expected): declared total WITHIN limit (2048), but actual file content aggregate exceeds 2 MiB
        corrupt_act_aggregate = copy.deepcopy(base_prov)
        corrupt_act_aggregate["total_source_bytes"] = 2048
        corrupt_act_aggregate["source_files"] = {
            "scripts/checkpoint_identity.py": {
                "path": "scripts/checkpoint_identity.py",
                "size_bytes": 1024,
                "sha256": "0" * 64,
                "content_bytes": b"a" * 1024,
            },
            "configs/training-semantics.json": {
                "path": "configs/training-semantics.json",
                "size_bytes": MAX_PROVENANCE_SOURCE_BYTES,
                "sha256": "0" * 64,
                "content_bytes": b"b" * MAX_PROVENANCE_SOURCE_BYTES,
            },
        }
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaisesRegex(
                CheckpointProvenanceError,
                r"(?i)actual total actual source bytes exceed 2 mib limit",
            ):
                validate_source_provenance(corrupt_act_aggregate, base_prov)

        corrupt_exp_aggregate = copy.deepcopy(base_prov)
        corrupt_exp_aggregate["total_source_bytes"] = 2048
        corrupt_exp_aggregate["source_files"] = {
            "scripts/checkpoint_identity.py": {
                "path": "scripts/checkpoint_identity.py",
                "size_bytes": 1024,
                "sha256": "0" * 64,
                "content_bytes": b"a" * 1024,
            },
            "configs/training-semantics.json": {
                "path": "configs/training-semantics.json",
                "size_bytes": MAX_PROVENANCE_SOURCE_BYTES,
                "sha256": "0" * 64,
                "content_bytes": b"b" * MAX_PROVENANCE_SOURCE_BYTES,
            },
        }
        with patch("hashlib.sha256", side_effect=_mock_sha256):
            with self.assertRaisesRegex(
                CheckpointProvenanceError,
                r"(?i)expected total actual source bytes exceed 2 mib limit",
            ):
                validate_source_provenance(base_prov, corrupt_exp_aggregate)

        # 4. Deep isolation of nested library_versions and untracked_sources list compared against independently built expected
        prov_indep1 = build_source_provenance(src_files_dirty, git_ev_dirty, env_rec)
        prov_indep2 = build_source_provenance(src_files_dirty, git_ev_dirty, env_rec)
        self.assertIsNot(prov_indep1, prov_indep2)
        self.assertIsNot(prov_indep1["git"]["untracked_sources"], prov_indep2["git"]["untracked_sources"])
        self.assertIsNot(prov_indep1["environment"]["library_versions"], prov_indep2["environment"]["library_versions"])

        # Mutate nested untracked list on prov_indep1 and assert rejection against independent expected
        prov_indep1["git"]["untracked_sources"].append("scripts/build_kg.py")
        with self.assertRaisesRegex(
            CheckpointProvenanceError,
            r"actual untracked source 'scripts/build_kg\.py' must have corresponding reconstructable bytes in source_files",
        ):
            validate_source_provenance(prov_indep1, prov_indep2)

        # Re-verify prov_indep2 remained untouched
        self.assertEqual(prov_indep2["git"]["untracked_sources"], [untracked_path])
        validate_source_provenance(prov_indep2, prov_indep2)

        prov_indep3 = build_source_provenance(src_files_dirty, git_ev_dirty, env_rec)
        prov_indep3["environment"]["library_versions"]["new_isolated_lib"] = "1.0.0"
        with self.assertRaisesRegex(
            CheckpointProvenanceError,
            r"library_versions mismatch",
        ):
            validate_source_provenance(prov_indep3, prov_indep2)
        self.assertNotIn("new_isolated_lib", prov_indep2["environment"]["library_versions"])


class PathSecurityAndByteBoundTests(unittest.TestCase):
    """Security tests: path traversal, boundary limits, and credential patterns."""

    def test_byte_limit_enforced_before_processing(self):
        large_bytes = b"x" * (MAX_PROVENANCE_SOURCE_BYTES + 1)
        src_files = {"scripts/checkpoint_identity.py": large_bytes}
        _, git_ev, env_rec = _make_valid_provenance_fixture()

        with self.assertRaises(CheckpointProvenanceError):
            build_source_provenance(src_files, git_ev, env_rec)

    def test_path_traversal_rejection(self):
        for bad_path in (
            "scripts/../configs/training-semantics.json",
            "scripts/../../etc/passwd",
            "scripts/./checkpoint_identity.py",
        ):
            with self.assertRaises(CheckpointSecurityError, msg=f"Should reject traversal {bad_path}"):
                validate_allowlisted_relative_path(bad_path)

    def test_absolute_paths_rejected(self):
        for bad_path in ("/etc/passwd", "/home/sunf/Amazon4U/scripts/checkpoint_identity.py"):
            with self.assertRaises(CheckpointSecurityError, msg=f"Should reject absolute path {bad_path}"):
                validate_allowlisted_relative_path(bad_path)

    def test_disallowed_top_level_directories(self):
        for bad_path in (
            "data/raw/benchmark.csv",
            ".test-output/output.txt",
            "tests/test_checkpoint_identity.py",
        ):
            with self.assertRaises(CheckpointSecurityError, msg=f"Should reject top-level dir in {bad_path}"):
                validate_allowlisted_relative_path(bad_path)

    def test_disallowed_substrings(self):
        for bad_path in (
            "scripts/.git/HEAD",
            "scripts/.venv/bin/python",
            "scripts/Movie4U/README.md",
            "scripts/multiautoresearch/notes.txt",
            "scripts/.worktrees/checkout/file.py",
        ):
            with self.assertRaises(CheckpointSecurityError, msg=f"Should reject substring in {bad_path}"):
                validate_allowlisted_relative_path(bad_path)

    def test_credential_patterns_rejected(self):
        for bad_path in (
            "scripts/secret.key",
            "scripts/server.pem",
            "configs/token.json",
            "configs/app.secret",
            "configs/.env.production",
            "configs/user.credential",
            "configs/admin.password",
        ):
            with self.assertRaises(CheckpointSecurityError, msg=f"Should reject credential pattern in {bad_path}"):
                validate_allowlisted_relative_path(bad_path)

    def test_non_allowlisted_relative_paths(self):
        for bad_path in (
            "scripts/unregistered_script.py",
            "configs/unregistered_config.json",
        ):
            with self.assertRaises(CheckpointSecurityError):
                validate_allowlisted_relative_path(bad_path)


class InputOutputAliasingAndPostMutationValidationTests(unittest.TestCase):
    """Tests guaranteeing that snapshot isolation prevents caller aliasing and mutation tampering."""

    def test_input_aliasing_isolation(self):
        f = _make_valid_identity_fixture()
        ident = build_canonical_identity(**f)

        # Mutate caller input dicts and nested objects after building
        f["training_records"][0]["rating"] = 999.0
        f["warm_items"].append("p_new")
        f["resolved_settings"]["training"]["learning_rate"] = 999.0
        f["product_metadata"]["p1"]["categories"].append('c:["Electronics"]')
        f["validation_inputs"]["targets"]["u1"].append("p1")
        f["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = 999
        f["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = 999

        # Identity internal state must remain isolated and valid
        validate_canonical_identity(ident, ident)
        self.assertEqual(ident.training_records[0]["rating"], 4.0)
        self.assertEqual(len(ident.warm_items), 3)
        self.assertEqual(ident.resolved_settings["training"]["learning_rate"], 0.001)
        self.assertEqual(len(ident.product_metadata["p1"]["categories"]), 1)
        self.assertEqual(len(ident.validation_inputs["targets"]["u1"]), 1)
        self.assertEqual(ident.resolved_settings["determinism"]["rng_streams"][0]["seed"], 42)
        self.assertIsNone(ident.resolved_settings["execution"]["worker_settings"]["prefetch_factor"])

    def test_output_aliasing_isolation(self):
        f = _make_valid_identity_fixture()
        ident = build_canonical_identity(**f)

        d = ident.to_dict()
        d["training_records"][0]["rating"] = 999.0
        d["warm_items"].append("p_new")
        d["resolved_settings"]["training"]["learning_rate"] = 999.0
        d["product_metadata"]["p1"]["categories"].append('c:["Electronics"]')
        d["validation_inputs"]["targets"]["u1"].append("p1")
        d["resolved_settings"]["determinism"]["rng_streams"][0]["seed"] = 999
        d["resolved_settings"]["execution"]["worker_settings"]["prefetch_factor"] = 999

        # Identity internal state must remain isolated and valid
        validate_canonical_identity(ident, ident)
        self.assertEqual(ident.training_records[0]["rating"], 4.0)
        self.assertEqual(ident.resolved_settings["training"]["learning_rate"], 0.001)
        self.assertEqual(len(ident.product_metadata["p1"]["categories"]), 1)
        self.assertEqual(len(ident.validation_inputs["targets"]["u1"]), 1)
        self.assertEqual(ident.resolved_settings["determinism"]["rng_streams"][0]["seed"], 42)
        self.assertIsNone(ident.resolved_settings["execution"]["worker_settings"]["prefetch_factor"])

    def test_post_mutation_validation_detects_tampered_identity(self):
        f = _make_valid_identity_fixture()
        ident = build_canonical_identity(**f)

        # Mutate internal dictionary inside immutable tuple
        ident.training_records[0]["rating"] = 999.0

        with self.assertRaises(CheckpointIdentityMismatchError):
            validate_canonical_identity(ident, ident)

    def test_provenance_aliasing_isolation(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        prov = build_source_provenance(src_files, git_ev, env_rec)

        # Mutate input dicts and nested collections
        src_files["scripts/checkpoint_identity.py"] = b"mutated"
        git_ev["commit"] = "0" * 40
        git_ev["untracked_sources"].append("scripts/learned_models.py")
        env_rec["python_version"] = "9.9.9"
        env_rec["library_versions"]["new_lib"] = "1.0.0"

        # Stored provenance must remain isolated and valid
        validate_source_provenance(prov, prov)
        self.assertEqual(prov["git"]["commit"], "fd15dc3b5fe665f45d8787ba779b9e9541690bde")
        self.assertEqual(len(prov["git"]["untracked_sources"]), 0)
        self.assertNotIn("new_lib", prov["environment"]["library_versions"])

    def test_post_mutation_validation_detects_tampered_provenance(self):
        src_files, git_ev, env_rec = _make_valid_provenance_fixture()
        prov = build_source_provenance(src_files, git_ev, env_rec)

        prov["environment"]["cpu_count"] = 999
        with self.assertRaises(CheckpointProvenanceError):
            validate_source_provenance(prov, build_source_provenance(src_files, git_ev, env_rec))

    def test_bounded_synthetic_fixtures_invariants(self):
        f = _make_valid_identity_fixture()
        self.assertLessEqual(len(f["graph_nodes"]["User"]), 8)
        self.assertLessEqual(len(f["graph_nodes"]["Product"]), 12)
        for ntype, nodes in f["graph_nodes"].items():
            if ntype in ("CategoryPath", "Brand"):
                self.assertLessEqual(len(nodes), 8)


if __name__ == "__main__":
    unittest.main()
