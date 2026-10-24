"""Tests for Amazon4U source closure, hydration, paged collection, and artifact completeness (OP03, OP04, OP10, OP11).

NO numeric packages (torch, numpy) or learned model modules imported.
All tests run with fake transport, independent frozen oracles, and fresh tempdirs.
"""

from __future__ import annotations

import ast
import base64
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zlib

from multiautoresearch.adapters.amazon4u import (
    Amazon4UError,
    FakeKaggleTransport,
    MAX_AGGREGATE_BYTES,
    MAX_CHUNK_BYTES,
    MAX_FILE_BYTES,
    MAX_PAGES,
    SCHEMA_MANIFEST,
    SourcePacket,
    build_amazon4u_kernel_script,
    build_amazon4u_record,
    build_source_manifest,
    get_canonical_h_spec,
    collect_outputs,
    consume_kernel_logs,
    encode_bytes,
    hydrate_bundle,
    hydrate_packet,
    validate_collected_artifacts,
    verify_h_provenance,
    verify_source_manifest,
)

# Independent literal membership oracles (NOT imported from implementation under test)
LITERAL_H_28_PATHS = (
    "scripts/checkpoint_identity.py",
    "scripts/learned_models.py",
    "scripts/recommendation_core.py",
    "scripts/recommendation_evaluation.py",
    "scripts/differentiable_models.py",
    "scripts/build_kg.py",
    "scripts/audit_kg_inputs.py",
    "scripts/audit_candidates.py",
    "scripts/validate_kg_splits.py",
    "scripts/visualize_kg.py",
    "scripts/inspect_data.py",
    "scripts/download_5core.py",
    "configs/compute-plan.json",
    "configs/dataset-scope.json",
    "configs/evaluation-metrics.json",
    "configs/kg-v1.json",
    "configs/model-suite.json",
    "configs/protocol-lifecycle.json",
    "configs/reproducibility.json",
    "configs/training-semantics.json",
    "scripts/checkpoint_payload.py",
    "scripts/checkpoint_storage.py",
    "scripts/training_harness.py",
    "scripts/training_data.py",
    "scripts/training_entrypoint.py",
    "tests/test_training_harness.py",
    "tests/test_training_data.py",
    "docs/training-harness.md",
)

LITERAL_O_4_PATHS = (
    "multiautoresearch/post-training/scripts/kaggle_job.py",
    "multiautoresearch/adapters/__init__.py",
    "multiautoresearch/adapters/amazon4u.py",
    "multiautoresearch/adapters/amazon4u_kernel.py",
)

LITERAL_32_CLOSURE = tuple(sorted(set(LITERAL_H_28_PATHS) | set(LITERAL_O_4_PATHS)))


def fake_directory_promotion_seam(src: Path, dst: Path) -> bool:
    """Explicit test fixture helper modeling supported directory promotion seam.

    Solely models supported success in tests with fresh/private/owner checks;
    it is NOT proof of native atomic guarantees.
    """
    if not isinstance(src, Path):
        src = Path(src)
    if not isinstance(dst, Path):
        dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    if not src.is_dir() or src.is_symlink():
        raise ValueError(f"Source must be a regular directory: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    stat_src = src.stat()
    if stat_src.st_uid != os.getuid():
        raise PermissionError(f"Source directory {src} not owned by current user")
    stat_dst_parent = dst.parent.stat()
    if stat_dst_parent.st_uid != os.getuid():
        raise PermissionError(f"Destination parent directory {dst.parent} not owned by current user")
    src.rename(dst)
    return True


def _canonical_test_spec(
    run_id: str = "r1",
    approval_id: str = "a1",
    method: str = "LightGCN",
    epochs: int = 1,
    max_adam_steps: int = 1,
) -> dict:
    return {
        "schema_version": "amazon4u-H-spec-v1",
        "run_id": run_id,
        "approval_id": approval_id,
        "stage": "training_from_scratch",
        "mode": "fixture_verification",
        "data_kind": "owned_synthetic_packet",
        "method": method,
        "device": "cpu",
        "accelerator": "none",
        "profile_id": "H-REF-synthetic-cpu-v1",
        "epochs": epochs,
        "max_adam_steps": max_adam_steps,
    }


def _canonical_test_auth(
    run_id: str = "r1",
    approval_id: str = "a1",
    max_adam_steps: int = 1,
    scope: str = "tiny_synthetic_cpu_backward_and_private_restore",
) -> dict:
    return {
        "schema_version": "amazon4u-H-authorization-v1",
        "run_id": run_id,
        "approval_id": approval_id,
        "scope": scope,
        "max_adam_steps": max_adam_steps,
    }


def _canonical_test_env(
    run_id: str = "r1",
    approval_id: str = "a1",
    method: str = "LightGCN",
    identity_hash: str = "a" * 64,
    bundle_sha256: str = "b" * 64,
    initial_state_sha256: str = "c" * 64,
) -> dict:
    return {
        "schema_version": "amazon4u-H-run-v1",
        "run_id": run_id,
        "approval_id": approval_id,
        "profile_id": "H-REF-synthetic-cpu-v1",
        "method": method,
        "identity_hash": identity_hash,
        "device": "cpu",
        "accelerator": "none",
        "bundle_sha256": bundle_sha256,
        "initial_state_sha256": initial_state_sha256,
        "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
    }


def _canonical_test_manifest(manifest_digest: str = "m" * 64) -> dict:
    return {
        "schema_version": SCHEMA_MANIFEST,
        "manifest_digest": manifest_digest,
        "files": {},
        "file_count": 0,
        "aggregate_size": 0,
    }


def _canonical_test_inventory(
    plan_bytes: bytes,
    result_bytes: bytes,
    envelope_bytes: bytes,
    manifest_bytes: bytes,
    log_bytes: bytes,
    ckpt_bytes: bytes,
) -> tuple[dict, str]:
    ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()
    inv = {
        "plan": {"filename": "plan.json", "size": len(plan_bytes), "sha256": hashlib.sha256(plan_bytes).hexdigest()},
        "h_result": {"filename": "result.json", "size": len(result_bytes), "sha256": hashlib.sha256(result_bytes).hexdigest()},
        "run_envelope": {"filename": "run_envelope.json", "size": len(envelope_bytes), "sha256": hashlib.sha256(envelope_bytes).hexdigest()},
        "source_manifest": {"filename": "manifest.json", "size": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "log": {"filename": "run.log", "size": len(log_bytes), "sha256": hashlib.sha256(log_bytes).hexdigest()},
        "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_bytes), "physical_sha256": ckpt_sha},
        "checkpoint_physical_sha256": ckpt_sha,
    }
    return inv, ckpt_sha


class TestAmazon4UBundle(unittest.TestCase):
    """Test suite covering OP03, OP04, OP10, and OP11."""

    # -----------------------------------------------------------------------
    # OP03: Complete Source Closure & Actual Import Layout
    # -----------------------------------------------------------------------

    def test_op03_complete_outer_source_closure_positive(self) -> None:
        """Manifest incorporates all 28 canonical H paths plus 4 O paths matching literal oracle."""
        self.assertEqual(len(LITERAL_32_CLOSURE), 32)
        packets = [
            SourcePacket(rel_path, f"# Stub content for {rel_path}\n".encode("utf-8"))
            for rel_path in LITERAL_32_CLOSURE
        ]

        manifest = build_source_manifest(packets, require_full_closure=True)
        self.assertEqual(manifest["schema_version"], SCHEMA_MANIFEST)
        self.assertEqual(manifest["file_count"], 32)
        self.assertIn("manifest_digest", manifest)

        # Verify literal oracle membership
        files_map = manifest["files"]
        for p in LITERAL_32_CLOSURE:
            self.assertIn(p, files_map)

    def test_op03_outer_manifest_non_self_referential(self) -> None:
        """Manifest digest covers source file entries without self-referential cycle."""
        packets = [SourcePacket("scripts/learned_models.py", b"print(1)\n")]
        manifest = build_source_manifest(packets, require_full_closure=False)
        files_json = json.dumps(manifest["files"], sort_keys=True).encode("utf-8")
        expected_digest = hashlib.sha256(files_json).hexdigest()
        self.assertEqual(manifest["manifest_digest"], expected_digest)

    def test_op03_missing_closure_file_refused(self) -> None:
        """Missing a required H or O module raises Amazon4UError under full closure."""
        # Omit one file from closure
        incomplete = [
            SourcePacket(rel_path, b"# stub\n")
            for rel_path in LITERAL_32_CLOSURE[1:]
        ]
        with self.assertRaises(Amazon4UError) as ctx:
            build_source_manifest(incomplete, require_full_closure=True)
        self.assertIn("closure incomplete", str(ctx.exception))

    def test_op03_extra_closure_file_refused(self) -> None:
        """Extra or foreign module in manifest raises Amazon4UError under full closure."""
        extra_packets = [
            SourcePacket(rel_path, b"# stub\n")
            for rel_path in LITERAL_32_CLOSURE
        ] + [SourcePacket("scripts/foreign_script.py", b"# foreign\n")]
        with self.assertRaises(Amazon4UError) as ctx:
            build_source_manifest(extra_packets, require_full_closure=True)
        self.assertIn("closure incomplete", str(ctx.exception))

    def test_op03_path_safety_negative(self) -> None:
        """Absolute paths, backslashes, path traversal, and duplicate paths are refused."""
        # Absolute path
        with self.assertRaises(Amazon4UError):
            SourcePacket("/etc/passwd", b"root:x:0:0\n")

        # Backslash
        with self.assertRaises(Amazon4UError):
            SourcePacket("scripts\\helper.py", b"code\n")

        # Traversal
        with self.assertRaises(Amazon4UError):
            SourcePacket("scripts/../../secret.txt", b"secret\n")

        # Empty part or dot
        with self.assertRaises(Amazon4UError):
            SourcePacket("scripts/./helper.py", b"code\n")

        # Duplicate paths in manifest
        p1 = SourcePacket("scripts/learned_models.py", b"v1\n")
        p2 = SourcePacket("scripts/learned_models.py", b"v2\n")
        with self.assertRaises(Amazon4UError):
            build_source_manifest([p1, p2])

    def test_op03_scripts_namespace_no_init_py(self) -> None:
        """Scripts directory is a namespace package and must not contain __init__.py."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            files = {
                "scripts/learned_models.py": encode_bytes(b"def helper(): pass\n"),
            }
            manifest = {
                "files": {
                    "scripts/learned_models.py": {
                        "size": len(b"def helper(): pass\n"),
                        "sha256": hashlib.sha256(b"def helper(): pass\n").hexdigest(),
                    }
                }
            }
            hydrate_bundle(files, workdir, manifest=manifest)
            self.assertTrue((workdir / "scripts" / "learned_models.py").exists())
            self.assertFalse((workdir / "scripts" / "__init__.py").exists())

    def test_op03_verify_manifest_tampering_refused(self) -> None:
        """verify_source_manifest catches hash, size, and manifest digest tampering."""
        packets = [SourcePacket("scripts/learned_models.py", b"test\n")]
        actual = build_source_manifest(packets)
        expected = copy.deepcopy(actual)

        # Tampered hash
        tampered_hash = copy.deepcopy(expected)
        tampered_hash["files"]["scripts/learned_models.py"]["sha256"] = "0" * 64
        with self.assertRaises(Amazon4UError):
            verify_source_manifest(actual, tampered_hash)

        # Tampered size
        tampered_size = copy.deepcopy(expected)
        tampered_size["files"]["scripts/learned_models.py"]["size"] = 999
        with self.assertRaises(Amazon4UError):
            verify_source_manifest(actual, tampered_size)

        # Tampered digest
        tampered_digest = copy.deepcopy(expected)
        tampered_digest["manifest_digest"] = "1" * 64
        with self.assertRaises(Amazon4UError):
            verify_source_manifest(actual, tampered_digest)

    # -----------------------------------------------------------------------
    # OP04: Incremental Bounded Hydration
    # -----------------------------------------------------------------------

    def test_op04_incremental_bounded_hydration_positive(self) -> None:
        """Tiny fake files are restored byte-for-byte through streaming decompression."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            raw_content = b"Amazon4U deterministic hydration content for testing.\n" * 10
            encoded = encode_bytes(raw_content)

            written, sha = hydrate_packet(
                "scripts/test_hydrated.py",
                encoded,
                target_dir,
                expected_size=len(raw_content),
                expected_sha256=hashlib.sha256(raw_content).hexdigest(),
            )
            self.assertEqual(written, len(raw_content))
            self.assertEqual(sha, hashlib.sha256(raw_content).hexdigest())

            restored_file = target_dir / "scripts" / "test_hydrated.py"
            self.assertEqual(restored_file.read_bytes(), raw_content)

    def test_op04_hydration_one_over_file_limit_negative(self) -> None:
        """Hydration aborts before writing if decoded content exceeds file limit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            # Create 100 bytes of content, set limit to 50 bytes
            raw_content = b"x" * 100
            encoded = encode_bytes(raw_content)

            with self.assertRaises(Amazon4UError) as ctx:
                hydrate_packet(
                    "large.bin",
                    encoded,
                    target_dir,
                    expected_size=100,
                    expected_sha256=hashlib.sha256(raw_content).hexdigest(),
                    max_file_bytes=50,
                )
            self.assertIn("exceeded file limit", str(ctx.exception))
            # Verify file was cleaned up and not left over-limit
            self.assertFalse((target_dir / "large.bin").exists())

    def test_op04_hydration_aggregate_limit_negative(self) -> None:
        """Hydration aborts before writing if decoded content exceeds aggregate limit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            raw_content = b"x" * 100
            encoded = encode_bytes(raw_content)

            with self.assertRaises(Amazon4UError) as ctx:
                hydrate_packet(
                    "file.bin",
                    encoded,
                    target_dir,
                    expected_size=100,
                    expected_sha256=hashlib.sha256(raw_content).hexdigest(),
                    max_aggregate_bytes=80,
                    current_aggregate=50,
                )
            self.assertIn("exceeded aggregate limit", str(ctx.exception))
            self.assertFalse((target_dir / "file.bin").exists())

    def test_op04_hydration_size_lie_negative(self) -> None:
        """Declared size lies in metadata are caught and rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            raw_content = b"actual content is 26 bytes"
            encoded = encode_bytes(raw_content)

            with self.assertRaises(Amazon4UError) as ctx:
                hydrate_packet(
                    "lied.txt",
                    encoded,
                    target_dir,
                    expected_size=999,  # Size lie
                    expected_sha256=hashlib.sha256(raw_content).hexdigest(),
                )
            self.assertIn("Declared size lie", str(ctx.exception))

    def test_op04_hydration_truncated_stream_negative(self) -> None:
        """Truncated compressed gzip stream (EOF not reached) is detected and refused."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            raw_content = b"Hello world! This is a test for truncated stream.\n"
            full_gz = gzip.compress(raw_content, mtime=0)
            truncated_gz = full_gz[:len(full_gz) // 2]
            truncated_b64 = base64.b64encode(truncated_gz).decode("ascii")

            with self.assertRaises(Amazon4UError) as ctx:
                hydrate_packet(
                    "truncated.txt",
                    truncated_b64,
                    target_dir,
                    expected_size=len(raw_content),
                    expected_sha256=hashlib.sha256(raw_content).hexdigest(),
                )
            self.assertIn("Truncated compressed stream", str(ctx.exception))

    def test_op04_hydration_trailing_garbage_negative(self) -> None:
        """Trailing garbage data after compressed stream EOF is detected and refused."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            raw_content = b"Content with trailing garbage\n"
            full_gz = gzip.compress(raw_content, mtime=0) + b"GARBAGE_BYTES_AT_END"
            garbage_b64 = base64.b64encode(full_gz).decode("ascii")

            with self.assertRaises(Amazon4UError) as ctx:
                hydrate_packet(
                    "garbage.txt",
                    garbage_b64,
                    target_dir,
                    expected_size=len(raw_content),
                    expected_sha256=hashlib.sha256(raw_content).hexdigest(),
                )
            self.assertIn("Trailing garbage data", str(ctx.exception))

    def test_op04_hydration_overwrite_existing_file_refused(self) -> None:
        """Attempting to hydrate over an existing file is refused without overwrite."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            existing = target_dir / "already_exists.txt"
            existing.write_text("prior content", encoding="utf-8")

            raw_content = b"new content"
            encoded = encode_bytes(raw_content)

            with self.assertRaises(Amazon4UError) as ctx:
                hydrate_packet(
                    "already_exists.txt",
                    encoded,
                    target_dir,
                    expected_size=len(raw_content),
                    expected_sha256=hashlib.sha256(raw_content).hexdigest(),
                )
            self.assertIn("already exists", str(ctx.exception))
            self.assertEqual(existing.read_text(encoding="utf-8"), "prior content")

    def test_op04_hydration_ancestor_symlink_refused(self) -> None:
        """Attempting to hydrate into a directory with ancestor symlinks is refused."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            raw_content = b"content"
            encoded = encode_bytes(raw_content)
            with patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaises(Amazon4UError) as ctx:
                    hydrate_packet(
                        "mock_symlink_dir/file.txt",
                        encoded,
                        td,
                        expected_size=len(raw_content),
                        expected_sha256=hashlib.sha256(raw_content).hexdigest(),
                    )
            self.assertIn("Ancestor symlink forbidden", str(ctx.exception))

    # -----------------------------------------------------------------------
    # OP10: Paged Bounded Transport Collection
    # -----------------------------------------------------------------------

    def test_op10_paged_bounded_collection_positive(self) -> None:
        """Multi-page output collection downloads into fresh quarantine and promotes."""
        exp_spec = {
            "schema_version": "amazon4u-H-spec-v1",
            "run_id": "r1",
            "approval_id": "a1",
            "stage": "training_from_scratch",
            "mode": "fixture_verification",
            "data_kind": "owned_synthetic_packet",
            "method": "LightGCN",
            "device": "cpu",
            "accelerator": "none",
            "profile_id": "H-REF-synthetic-cpu-v1",
            "epochs": 1,
            "max_adam_steps": 1,
        }
        exp_auth = {
            "schema_version": "amazon4u-H-authorization-v1",
            "run_id": "r1",
            "approval_id": "a1",
            "scope": "tiny_synthetic_cpu_backward_and_private_restore",
            "max_adam_steps": 1,
        }
        exp_env = {
            "schema_version": "amazon4u-H-run-v1",
            "run_id": "r1",
            "approval_id": "a1",
            "profile_id": "H-REF-synthetic-cpu-v1",
            "method": "LightGCN",
            "identity_hash": "a" * 64,
            "device": "cpu",
            "accelerator": "none",
            "bundle_sha256": "b" * 64,
            "initial_state_sha256": "c" * 64,
            "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
        }
        exp_man = {
            "schema_version": SCHEMA_MANIFEST,
            "manifest_digest": hashlib.sha256(b"op10_manifest_data").hexdigest(),
            "files": {},
            "file_count": 0,
            "aggregate_size": 0,
        }

        plan_content = json.dumps({
            "schema_version": "amazon4u-O-plan-v1",
            "version": 1,
            "spec": exp_spec,
            "authorization": exp_auth,
        }).encode("utf-8")
        res_dict = {
            "schema_version": "amazon4u-H-result-v1",
            "status": "fixture_budget_complete",
            "acceptance_scope": "H-REF_CPU_reference_fixture_only",
            "deployment_decision": "NOT_AUTHORIZED",
            "run_spec": exp_spec,
            "run_envelope": exp_env,
            "device_capability": {
                "device": "cpu",
                "accelerator": "none",
                "supports_training": True,
                "state_transfer": "cpu_quiescent_epoch_v1",
            },
            "method": "LightGCN",
            "epoch": 1,
            "actual_Adam_steps": 1,
            "selection": {"status": "active"},
            "mixture": None,
            "validation_report": None,
            "remaining_readiness_gates": [
                "GPU-READY separately authorized and verified",
                "O-OFFLINE transport and packaging separately reviewed",
                "Production LightGCN initialization decision",
                "Approved environment and full-data feasibility profiling",
                "Comparative execution/search freeze and final-test authorization",
            ],
            "test_execution": "not_authorized",
        }
        res_content = json.dumps(res_dict).encode("utf-8")
        env_content = json.dumps(exp_env).encode("utf-8")
        manifest_content = json.dumps(exp_man).encode("utf-8")
        log_content = b"Execution log line 1\nExecution log line 2\n"
        ckpt_content = b"CHECKPOINT_BINARY_DATA_XYZ"
        ckpt_sha = hashlib.sha256(ckpt_content).hexdigest()

        exp_inv = {
            "plan": {"filename": "plan.json", "size": len(plan_content), "sha256": hashlib.sha256(plan_content).hexdigest()},
            "h_result": {"filename": "result.json", "size": len(res_content), "sha256": hashlib.sha256(res_content).hexdigest()},
            "run_envelope": {"filename": "run_envelope.json", "size": len(env_content), "sha256": hashlib.sha256(env_content).hexdigest()},
            "source_manifest": {"filename": "manifest.json", "size": len(manifest_content), "sha256": hashlib.sha256(manifest_content).hexdigest()},
            "log": {"filename": "run.log", "size": len(log_content), "sha256": hashlib.sha256(log_content).hexdigest()},
            "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_content), "physical_sha256": ckpt_sha},
            "checkpoint_physical_sha256": ckpt_sha,
        }

        paged_data = [
            # Page 0 (chunked items)
            [
                {"name": "plan.json", "chunks": [plan_content]},
                {"name": "result.json", "chunks": [res_content]},
            ],
            # Page 1
            [
                {"name": "run_envelope.json", "chunks": [env_content]},
                {"name": "manifest.json", "chunks": [manifest_content]},
                {"name": "run.log", "chunks": [log_content]},
                {"name": "checkpoint.bin", "chunks": [ckpt_content]},
            ],
        ]

        transport = FakeKaggleTransport(paged_outputs=paged_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected_output"

            with patch("multiautoresearch.adapters.amazon4u._rename_no_replace_linux", side_effect=fake_directory_promotion_seam):
                report = collect_outputs(
                    "amazon4u/test-slug", 1, transport, dest,
                    expected_spec=exp_spec,
                    expected_approval=exp_auth,
                    expected_manifest=exp_man,
                    expected_checkpoint_sha256=ckpt_sha,
                    expected_output_inventory=exp_inv,
                    expected_envelope=exp_env,
                )

            self.assertEqual(report["status"], "complete")
            self.assertTrue(dest.exists())
            self.assertTrue((dest / "plan.json").exists())
            self.assertTrue((dest / "result.json").exists())
            self.assertTrue((dest / "run_envelope.json").exists())
            self.assertTrue((dest / "manifest.json").exists())
            self.assertTrue((dest / "run.log").exists())
            self.assertTrue((dest / "checkpoint.bin").exists())

    def test_op10_paged_collection_cycle_detection_negative(self) -> None:
        """Cycle in pagination token is detected and aborted."""
        class CycleTransport(FakeKaggleTransport):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def list_output_page(self, kernel_ref, page_token=None):
                self.call_count += 1
                return [{"name": f"file_{self.call_count}.txt", "content": b"x"}], "token_repeat"

        transport = CycleTransport()
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            report = collect_outputs(
                "amazon4u/test-slug", 1, transport, dest,
                expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv, expected_envelope=exp_env,
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("Cycle detected in pagination token", report["reason"])
            self.assertGreaterEqual(transport.call_count, 2)

    def test_op10_paged_collection_duplicate_file_negative(self) -> None:
        """Duplicate file across pagination pages is detected and rejected."""
        paged_data = [
            [{"name": "file.txt", "content": b"page1"}],
            [{"name": "file.txt", "content": b"page2"}],
        ]
        transport = FakeKaggleTransport(paged_outputs=paged_data)
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            report = collect_outputs(
                "amazon4u/test-slug", 1, transport, dest,
                expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv, expected_envelope=exp_env,
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("Duplicate file in output collection", report["reason"])

    def test_op10_paged_collection_byte_caps_negative(self) -> None:
        """Oversized file during collection triggers byte cap and halts before writing."""
        paged_data = [
            [{"name": "oversized.bin", "content": b"x" * 2000}],
        ]
        transport = FakeKaggleTransport(paged_outputs=paged_data)
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            report = collect_outputs(
                "amazon4u/test-slug", 1, transport, dest,
                expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv, expected_envelope=exp_env,
                max_file_bytes=500,
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("exceeds max_file_bytes", report["reason"])

    def test_op10_destination_already_exists_refused(self) -> None:
        """collect_outputs refuses if destination already exists, even if empty."""
        transport = FakeKaggleTransport(paged_outputs=[])
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "existing_dest"
            dest.mkdir()
            with self.assertRaises(Amazon4UError) as ctx:
                collect_outputs(
                    "amazon4u/test-slug", 1, transport, dest,
                    expected_spec=exp_spec, expected_approval=exp_auth,
                    expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                    expected_output_inventory=exp_inv, expected_envelope=exp_env,
                )
            self.assertIn("already exists", str(ctx.exception))

    # -----------------------------------------------------------------------
    # OP11: Artifact Completeness & Independent Cross-Links
    # -----------------------------------------------------------------------

    def test_op11_artifact_completeness_positive(self) -> None:
        """Complete artifact inventory with matching crosslinks passes validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            plan_p = td / "plan.json"
            res_p = td / "result.json"
            env_p = td / "run_envelope.json"
            man_p = td / "manifest.json"
            log_p = td / "kernel.log"
            ckpt_p = td / "checkpoint.bin"

            exp_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "r1",
                "approval_id": "a1",
                "stage": "training_from_scratch",
                "mode": "fixture_verification",
                "data_kind": "owned_synthetic_packet",
                "method": "LightGCN",
                "device": "cpu",
                "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "epochs": 1,
                "max_adam_steps": 1,
            }
            exp_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "r1",
                "approval_id": "a1",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore",
                "max_adam_steps": 1,
            }
            exp_env = {
                "schema_version": "amazon4u-H-run-v1",
                "run_id": "r1",
                "approval_id": "a1",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "method": "LightGCN",
                "identity_hash": "a" * 64,
                "device": "cpu",
                "accelerator": "none",
                "bundle_sha256": "b" * 64,
                "initial_state_sha256": "c" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }

            plan_p.write_text(json.dumps({
                "schema_version": "amazon4u-O-plan-v1",
                "version": 1,
                "spec": exp_spec,
                "authorization": exp_auth,
            }))
            res_dict = {
                "schema_version": "amazon4u-H-result-v1",
                "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only",
                "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": exp_spec,
                "run_envelope": exp_env,
                "device_capability": {
                    "device": "cpu",
                    "accelerator": "none",
                    "supports_training": True,
                    "state_transfer": "cpu_quiescent_epoch_v1",
                },
                "method": "LightGCN",
                "epoch": 1,
                "actual_Adam_steps": 1,
                "selection": {"status": "active"},
                "mixture": None,
                "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p.write_text(json.dumps(res_dict))
            env_p.write_text(json.dumps(exp_env))
            exp_man = {
                "schema_version": SCHEMA_MANIFEST,
                "manifest_digest": hashlib.sha256(b"man_digest_op11").hexdigest(),
                "files": {},
                "file_count": 0,
                "aggregate_size": 0,
            }
            man_p.write_text(json.dumps(exp_man))
            log_p.write_text("All tests completed.")
            ckpt_p.write_bytes(b"CHECKPOINT_DATA_12345")

            files = {
                "plan.json": plan_p,
                "result.json": res_p,
                "run_envelope.json": env_p,
                "manifest.json": man_p,
                "kernel.log": log_p,
                "checkpoint.bin": ckpt_p,
            }

            ckpt_sha = hashlib.sha256(b"CHECKPOINT_DATA_12345").hexdigest()
            exp_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_p.read_bytes()), "sha256": hashlib.sha256(env_p.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "kernel.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_p.read_bytes()), "physical_sha256": ckpt_sha},
                "checkpoint_physical_sha256": ckpt_sha,
            }
            validation = validate_collected_artifacts(
                files,
                expected_version=1,
                expected_spec=exp_spec,
                expected_approval=exp_auth,
                expected_manifest=exp_man,
                expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv,
                expected_envelope=exp_env,
            )
            self.assertEqual(validation["status"], "complete")
            self.assertEqual(validation["roles"]["checkpoint_physical_sha256"], ckpt_sha)

            # Deterministic baseline logical checkpoint-N/A receipt control
            base_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "r_base",
                "approval_id": "a_base",
                "stage": "training_from_scratch",
                "mode": "fixture_verification",
                "data_kind": "owned_synthetic_packet",
                "method": "CorePopularity",
                "device": "cpu",
                "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "epochs": 0,
                "max_adam_steps": 0,
            }
            base_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "r_base",
                "approval_id": "a_base",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore",
                "max_adam_steps": 0,
            }
            base_env = {
                "schema_version": "amazon4u-H-run-v1",
                "run_id": "r_base",
                "approval_id": "a_base",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "method": "CorePopularity",
                "identity_hash": "a" * 64,
                "device": "cpu",
                "accelerator": "none",
                "bundle_sha256": "b" * 64,
                "initial_state_sha256": "c" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }
            plan_base = td / "plan_base.json"
            res_base = td / "result_base.json"
            env_base = td / "run_envelope_base.json"
            plan_base.write_text(json.dumps({
                "schema_version": "amazon4u-O-plan-v1",
                "version": 1,
                "spec": base_spec,
                "authorization": base_auth,
            }))
            res_base_dict = copy.deepcopy(res_dict)
            res_base_dict["run_spec"] = base_spec
            res_base_dict["run_envelope"] = base_env
            res_base_dict["method"] = "CorePopularity"
            res_base_dict["status"] = "deterministic_validation_complete"
            res_base_dict["epoch"] = 0
            res_base_dict["actual_Adam_steps"] = 0
            res_base_dict["selection"] = None
            res_base.write_text(json.dumps(res_base_dict))
            env_base.write_text(json.dumps(base_env))

            base_files = {
                "plan.json": plan_base,
                "result.json": res_base,
                "run_envelope.json": env_base,
                "manifest.json": man_p,
                "kernel.log": log_p,
            }
            base_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_base.read_bytes()), "sha256": hashlib.sha256(plan_base.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_base.read_bytes()), "sha256": hashlib.sha256(res_base.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_base.read_bytes()), "sha256": hashlib.sha256(env_base.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "kernel.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {
                    "status": "not_applicable_deterministic_baseline",
                    "applicable": False,
                    "method": "CorePopularity",
                    "epoch": 0,
                    "actual_Adam_steps": 0,
                    "selection": None,
                },
                "checkpoint_physical_sha256": "not_applicable_deterministic_baseline",
            }
            base_val = validate_collected_artifacts(
                base_files,
                expected_version=1,
                expected_spec=base_spec,
                expected_approval=base_auth,
                expected_manifest=exp_man,
                expected_checkpoint_sha256="not_applicable_deterministic_baseline",
                expected_output_inventory=base_inv,
                expected_envelope=base_env,
            )
            self.assertEqual(base_val["status"], "complete")
            self.assertEqual(base_val["roles"]["checkpoint"], "not_applicable_deterministic_baseline")
            self.assertEqual(base_val["roles"]["checkpoint_physical_sha256"], "not_applicable_deterministic_baseline")

    def test_op11_missing_required_role_marked_partial_negative(self) -> None:
        """Missing required artifact role marks collection unrecoverable (if checkpoint) or partial."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            exp_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "r1",
                "approval_id": "a1",
                "stage": "training_from_scratch",
                "mode": "fixture_verification",
                "data_kind": "owned_synthetic_packet",
                "method": "LightGCN",
                "device": "cpu",
                "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "epochs": 1,
                "max_adam_steps": 1,
            }
            exp_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "r1",
                "approval_id": "a1",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore",
                "max_adam_steps": 1,
            }
            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({
                "schema_version": "amazon4u-O-plan-v1",
                "version": 1,
                "spec": exp_spec,
                "authorization": exp_auth,
            }))
            exp_man = {
                "schema_version": SCHEMA_MANIFEST,
                "manifest_digest": hashlib.sha256(b"man_digest").hexdigest(),
                "files": {},
                "file_count": 0,
                "aggregate_size": 0,
            }
            exp_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": 100, "sha256": "1" * 64},
                "run_envelope": {"filename": "run_envelope.json", "size": 100, "sha256": "2" * 64},
                "source_manifest": {"filename": "manifest.json", "size": 100, "sha256": "3" * 64},
                "log": {"filename": "kernel.log", "size": 100, "sha256": "4" * 64},
                "checkpoint": {"filename": "checkpoint.bin", "size": 10, "physical_sha256": "5" * 64},
                "checkpoint_physical_sha256": "5" * 64,
            }

            # Missing checkpoint yields unrecoverable status
            files_no_ckpt = {"plan.json": plan_p}
            val_unrec = validate_collected_artifacts(
                files_no_ckpt,
                expected_version=1,
                expected_envelope=_canonical_test_env(),
                expected_checkpoint_sha256="5" * 64,
                expected_spec=exp_spec,
                expected_approval=exp_auth,
                expected_manifest=exp_man,
                expected_output_inventory=exp_inv,
            )
            self.assertEqual(val_unrec["status"], "unrecoverable")
            self.assertIn("Missing required artifact roles", val_unrec["reason"])

            # Checkpoint present but non-checkpoint role missing yields partial status
            ckpt_p = td / "checkpoint.bin"
            ckpt_p.write_bytes(b"dummy_ckpt")
            files_with_ckpt = {"plan.json": plan_p, "checkpoint.bin": ckpt_p}
            val_part = validate_collected_artifacts(
                files_with_ckpt,
                expected_version=1,
                expected_envelope=_canonical_test_env(),
                expected_checkpoint_sha256="5" * 64,
                expected_spec=exp_spec,
                expected_approval=exp_auth,
                expected_manifest=exp_man,
                expected_output_inventory=exp_inv,
            )
            self.assertEqual(val_part["status"], "partial")
            self.assertIn("Missing required artifact roles", val_part["reason"])

    def test_op11_mismatched_crosslink_negative(self) -> None:
        """Mismatched run_id crosslink across plan and envelope is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            plan_p = td / "plan.json"
            res_p = td / "result.json"
            env_p = td / "run_envelope.json"
            man_p = td / "manifest.json"
            log_p = td / "kernel.log"
            ckpt_p = td / "checkpoint.bin"

            exp_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "run_EXPECTED",
                "approval_id": "a1",
                "stage": "training_from_scratch",
                "mode": "fixture_verification",
                "data_kind": "owned_synthetic_packet",
                "method": "LightGCN",
                "device": "cpu",
                "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "epochs": 1,
                "max_adam_steps": 1,
            }
            exp_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "run_EXPECTED",
                "approval_id": "a1",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore",
                "max_adam_steps": 1,
            }
            exp_env = {
                "schema_version": "amazon4u-H-run-v1",
                "run_id": "run_EXPECTED",
                "approval_id": "a1",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "method": "LightGCN",
                "identity_hash": "a" * 64,
                "device": "cpu",
                "accelerator": "none",
                "bundle_sha256": "b" * 64,
                "initial_state_sha256": "c" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }

            plan_p.write_text(json.dumps({
                "schema_version": "amazon4u-O-plan-v1",
                "version": 1,
                "spec": exp_spec,
                "authorization": exp_auth,
            }))
            res_dict = {
                "schema_version": "amazon4u-H-result-v1",
                "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only",
                "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": exp_spec,
                "run_envelope": exp_env,
                "device_capability": {
                    "device": "cpu",
                    "accelerator": "none",
                    "supports_training": True,
                    "state_transfer": "cpu_quiescent_epoch_v1",
                },
                "method": "LightGCN",
                "epoch": 1,
                "actual_Adam_steps": 1,
                "selection": {"status": "active"},
                "mixture": None,
                "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p.write_text(json.dumps(res_dict))

            # Mismatched envelope run_id
            mismatched_env = dict(exp_env)
            mismatched_env["run_id"] = "run_WRONG"
            env_p.write_text(json.dumps(mismatched_env))

            exp_man = {
                "schema_version": SCHEMA_MANIFEST,
                "manifest_digest": hashlib.sha256(b"man_digest").hexdigest(),
                "files": {},
                "file_count": 0,
                "aggregate_size": 0,
            }
            man_p.write_text(json.dumps(exp_man))
            log_p.write_text("Done.")
            ckpt_p.write_bytes(b"DATA_CKPT")
            ckpt_sha = hashlib.sha256(b"DATA_CKPT").hexdigest()

            files = {
                "plan.json": plan_p,
                "result.json": res_p,
                "run_envelope.json": env_p,
                "manifest.json": man_p,
                "kernel.log": log_p,
                "checkpoint.bin": ckpt_p,
            }

            exp_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_p.read_bytes()), "sha256": hashlib.sha256(env_p.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "kernel.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_p.read_bytes()), "physical_sha256": ckpt_sha},
                "checkpoint_physical_sha256": ckpt_sha,
            }

            validation = validate_collected_artifacts(
                files,
                expected_version=1,
                expected_spec=exp_spec,
                expected_approval=exp_auth,
                expected_manifest=exp_man,
                expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv,
                expected_envelope=exp_env,
            )
            self.assertEqual(validation["status"], "error")
            self.assertIn("Crosslink mismatch", validation["reason"])

    def test_op11_internally_consistent_forged_set_refused(self) -> None:
        """Mutually consistent forged result set fails when mismatched with trusted expectations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            forged_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "forged_run",
                "approval_id": "forged_auth",
                "stage": "training_from_scratch",
                "mode": "fixture_verification",
                "data_kind": "owned_synthetic_packet",
                "method": "LightGCN",
                "device": "cpu",
                "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "epochs": 1,
                "max_adam_steps": 1,
            }
            forged_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "forged_run",
                "approval_id": "forged_auth",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore",
                "max_adam_steps": 1,
            }
            forged_env = {
                "schema_version": "amazon4u-H-run-v1",
                "run_id": "forged_run",
                "approval_id": "forged_auth",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "method": "LightGCN",
                "identity_hash": "f" * 64,
                "device": "cpu",
                "accelerator": "none",
                "bundle_sha256": "f" * 64,
                "initial_state_sha256": "f" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }

            plan_p = td / "plan.json"
            res_p = td / "result.json"
            env_p = td / "run_envelope.json"
            man_p = td / "manifest.json"
            log_p = td / "kernel.log"
            ckpt_p = td / "checkpoint.bin"

            plan_p.write_text(json.dumps({
                "schema_version": "amazon4u-O-plan-v1",
                "version": 1,
                "spec": forged_spec,
                "authorization": forged_auth,
            }))
            res_dict = {
                "schema_version": "amazon4u-H-result-v1",
                "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only",
                "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": forged_spec,
                "run_envelope": forged_env,
                "device_capability": {
                    "device": "cpu",
                    "accelerator": "none",
                    "supports_training": True,
                    "state_transfer": "cpu_quiescent_epoch_v1",
                },
                "method": "LightGCN",
                "epoch": 1,
                "actual_Adam_steps": 1,
                "selection": {"status": "active"},
                "mixture": None,
                "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p.write_text(json.dumps(res_dict))
            env_p.write_text(json.dumps(forged_env))
            exp_man = {
                "schema_version": SCHEMA_MANIFEST,
                "manifest_digest": hashlib.sha256(b"forged_man").hexdigest(),
                "files": {},
                "file_count": 0,
                "aggregate_size": 0,
            }
            man_p.write_text(json.dumps(exp_man))
            log_p.write_text("Forged success")
            ckpt_p.write_bytes(b"FORGED")
            ckpt_sha = hashlib.sha256(b"FORGED").hexdigest()

            files = {
                "plan.json": plan_p,
                "result.json": res_p,
                "run_envelope.json": env_p,
                "manifest.json": man_p,
                "kernel.log": log_p,
                "checkpoint.bin": ckpt_p,
            }

            # Validated against trusted expectation
            trusted_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "trusted_run",
                "approval_id": "trusted_auth",
                "stage": "training_from_scratch",
                "mode": "fixture_verification",
                "data_kind": "owned_synthetic_packet",
                "method": "LightGCN",
                "device": "cpu",
                "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "epochs": 1,
                "max_adam_steps": 1,
            }
            trusted_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "trusted_run",
                "approval_id": "trusted_auth",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore",
                "max_adam_steps": 1,
            }
            trusted_env = {
                "schema_version": "amazon4u-H-run-v1",
                "run_id": "trusted_run",
                "approval_id": "trusted_auth",
                "profile_id": "H-REF-synthetic-cpu-v1",
                "method": "LightGCN",
                "identity_hash": "a" * 64,
                "device": "cpu",
                "accelerator": "none",
                "bundle_sha256": "b" * 64,
                "initial_state_sha256": "c" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }
            exp_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_p.read_bytes()), "sha256": hashlib.sha256(env_p.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "kernel.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_p.read_bytes()), "physical_sha256": ckpt_sha},
                "checkpoint_physical_sha256": ckpt_sha,
            }
            validation = validate_collected_artifacts(
                files,
                expected_version=1,
                expected_spec=trusted_spec,
                expected_approval=trusted_auth,
                expected_manifest=exp_man,
                expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv,
                expected_envelope=trusted_env,
            )
            self.assertEqual(validation["status"], "error")
            self.assertIn("spec does not match expected_spec", validation["reason"])

    def test_op11_physical_vs_semantic_digest_distinction(self) -> None:
        """Physical file SHA-256 differs from semantic payload digest."""
        data_bytes = b'{"epoch": 1, "loss": 0.5}\n'
        physical_sha = hashlib.sha256(data_bytes).hexdigest()

        semantic_payload = {"epoch": 1, "loss": 0.5}
        semantic_sha = hashlib.sha256(json.dumps(semantic_payload, sort_keys=True).encode("utf-8")).hexdigest()

        self.assertNotEqual(physical_sha, semantic_sha)

    # -----------------------------------------------------------------------
    # Focused Collection Gaps: Targeted Verification Controls
    # -----------------------------------------------------------------------

    def test_op10_collection_deadline_parameter_validation_negative(self) -> None:
        """collect_outputs enforces finite positive deadline parameter before mutations."""
        transport = FakeKaggleTransport(paged_outputs=[])
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            for invalid_dl in (-10.0, 0, float("nan"), float("inf"), True):
                report = collect_outputs(
                    "amazon4u/test-slug", 1, transport, dest,
                    expected_spec=exp_spec, expected_approval=exp_auth,
                    expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                    expected_output_inventory=exp_inv, expected_envelope=exp_env,
                    deadline_seconds=invalid_dl,
                )
                self.assertEqual(report["status"], "error")
                self.assertIn("deadline_seconds must be a finite positive number", report["reason"])
                self.assertFalse(dest.exists())

    def test_op10_collection_deadline_exceeded_before_listing_negative(self) -> None:
        """Elapsed deadline before listing halts output collection immediately."""
        transport = FakeKaggleTransport(paged_outputs=[[{"name": "test.txt", "content": b"data"}]])
        fake_clock = [0.0, 50.0]  # Start 0.0, next check 50.0 >= 10.0
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            report = collect_outputs(
                "amazon4u/test-slug", 1, transport, dest,
                expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv, expected_envelope=exp_env,
                deadline_seconds=10.0,
                clock_fn=lambda: fake_clock.pop(0) if fake_clock else 100.0,
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("Deadline exceeded", report["reason"])
            self.assertFalse(dest.exists())

    def test_op10_collection_deadline_late_provider_return_aborts_negative(self) -> None:
        """Late provider returns must abort before file writes or completion."""
        time_state = {"now": 0.0}

        class LateTransport(FakeKaggleTransport):
            def list_output_page(self, kernel_ref, page_token=None):
                time_state["now"] = 500.0  # Provider returns late
                return [{"name": "result.json", "content": b"{}"}], None

        transport = LateTransport()
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            report = collect_outputs(
                "amazon4u/test-slug", 1, transport, dest,
                expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv, expected_envelope=exp_env,
                deadline_seconds=10.0,
                clock_fn=lambda: time_state["now"],
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("Deadline exceeded", report["reason"])
            self.assertFalse(dest.exists())

    def test_op10_collection_deadline_between_chunks_negative(self) -> None:
        """Deadline exceeded between chunks halts collection and cleans up quarantine."""
        time_state = {"now": 0.0}

        def chunk_generator():
            yield b"chunk_1"
            time_state["now"] = 300.0  # Time elapses between chunks
            yield b"chunk_2"

        transport = FakeKaggleTransport(paged_outputs=[[{"name": "large.bin", "chunks": chunk_generator()}]])
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        exp_inv, ckpt_sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            report = collect_outputs(
                "amazon4u/test-slug", 1, transport, dest,
                expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                expected_output_inventory=exp_inv, expected_envelope=exp_env,
                deadline_seconds=10.0,
                clock_fn=lambda: time_state["now"],
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("Deadline exceeded", report["reason"])
            self.assertFalse(dest.exists())

    def test_op10_collection_deadline_before_promotion_negative(self) -> None:
        """Deadline exceeded after artifact validation halts before promotion."""
        time_state = {"now": 0.0}
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()

        plan_content = json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}).encode("utf-8")
        res_dict = {
            "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
            "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
            "run_spec": exp_spec, "run_envelope": exp_env,
            "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
            "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
            "mixture": None, "validation_report": None,
            "remaining_readiness_gates": [
                "GPU-READY separately authorized and verified",
                "O-OFFLINE transport and packaging separately reviewed",
                "Production LightGCN initialization decision",
                "Approved environment and full-data feasibility profiling",
                "Comparative execution/search freeze and final-test authorization",
            ],
            "test_execution": "not_authorized",
        }
        res_content = json.dumps(res_dict).encode("utf-8")
        env_content = json.dumps(exp_env).encode("utf-8")
        man_content = json.dumps(exp_man).encode("utf-8")
        log_content = b"log"
        ckpt_content = b"ckpt_data"
        exp_inv, ckpt_sha = _canonical_test_inventory(
            plan_content, res_content, env_content, man_content, log_content, ckpt_content
        )
        paged_data = [[
            {"name": "plan.json", "chunks": [plan_content]},
            {"name": "result.json", "chunks": [res_content]},
            {"name": "run_envelope.json", "chunks": [env_content]},
            {"name": "manifest.json", "chunks": [man_content]},
            {"name": "run.log", "chunks": [log_content]},
            {"name": "checkpoint.bin", "chunks": [ckpt_content]},
        ]]
        transport = FakeKaggleTransport(paged_outputs=paged_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"

            orig_val = validate_collected_artifacts
            def validating_wrapper(*args, **kwargs):
                res = orig_val(*args, **kwargs)
                if res.get("status") == "complete":
                    time_state["now"] = 500.0  # Advance clock past deadline right before promotion check
                return res

            with patch("multiautoresearch.adapters.amazon4u.validate_collected_artifacts", side_effect=validating_wrapper):
                with patch("multiautoresearch.adapters.amazon4u._rename_no_replace_linux") as mock_rename:
                    report = collect_outputs(
                        "amazon4u/test-slug", 1, transport, dest,
                        expected_spec=exp_spec, expected_approval=exp_auth,
                        expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                        expected_output_inventory=exp_inv, expected_envelope=exp_env,
                        deadline_seconds=20.0,
                        clock_fn=lambda: time_state["now"],
                    )
                    self.assertEqual(report["status"], "error")
                    self.assertIn("Deadline exceeded (before_promotion)", report["reason"])
                    self.assertFalse(dest.exists())
                    mock_rename.assert_not_called()

    def test_op10_promotion_unsupported_native_fails_closed_negative(self) -> None:
        """When native directory no-replace is unsupported, collection fails closed with explicit error."""
        exp_spec = {
            "schema_version": "amazon4u-H-spec-v1",
            "run_id": "r1", "approval_id": "a1", "stage": "training_from_scratch",
            "mode": "fixture_verification", "data_kind": "owned_synthetic_packet",
            "method": "LightGCN", "device": "cpu", "accelerator": "none",
            "profile_id": "H-REF-synthetic-cpu-v1", "epochs": 1, "max_adam_steps": 1,
        }
        exp_auth = {
            "schema_version": "amazon4u-H-authorization-v1",
            "run_id": "r1", "approval_id": "a1",
            "scope": "tiny_synthetic_cpu_backward_and_private_restore", "max_adam_steps": 1,
        }
        exp_env = {
            "schema_version": "amazon4u-H-run-v1", "run_id": "r1", "approval_id": "a1",
            "profile_id": "H-REF-synthetic-cpu-v1", "method": "LightGCN",
            "identity_hash": "a" * 64, "device": "cpu", "accelerator": "none",
            "bundle_sha256": "b" * 64, "initial_state_sha256": "c" * 64,
            "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
        }
        exp_man = {
            "schema_version": SCHEMA_MANIFEST,
            "manifest_digest": hashlib.sha256(b"man").hexdigest(),
            "files": {}, "file_count": 0, "aggregate_size": 0,
        }
        plan_content = json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}).encode("utf-8")
        res_dict = {
            "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
            "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
            "run_spec": exp_spec, "run_envelope": exp_env,
            "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
            "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
            "mixture": None, "validation_report": None,
            "remaining_readiness_gates": [
                "GPU-READY separately authorized and verified",
                "O-OFFLINE transport and packaging separately reviewed",
                "Production LightGCN initialization decision",
                "Approved environment and full-data feasibility profiling",
                "Comparative execution/search freeze and final-test authorization",
            ],
            "test_execution": "not_authorized",
        }
        res_content = json.dumps(res_dict).encode("utf-8")
        env_content = json.dumps(exp_env).encode("utf-8")
        man_content = json.dumps(exp_man).encode("utf-8")
        log_content = b"log"
        ckpt_content = b"ckpt_data"
        ckpt_sha = hashlib.sha256(ckpt_content).hexdigest()

        exp_inv = {
            "plan": {"filename": "plan.json", "size": len(plan_content), "sha256": hashlib.sha256(plan_content).hexdigest()},
            "h_result": {"filename": "result.json", "size": len(res_content), "sha256": hashlib.sha256(res_content).hexdigest()},
            "run_envelope": {"filename": "run_envelope.json", "size": len(env_content), "sha256": hashlib.sha256(env_content).hexdigest()},
            "source_manifest": {"filename": "manifest.json", "size": len(man_content), "sha256": hashlib.sha256(man_content).hexdigest()},
            "log": {"filename": "run.log", "size": len(log_content), "sha256": hashlib.sha256(log_content).hexdigest()},
            "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_content), "physical_sha256": ckpt_sha},
            "checkpoint_physical_sha256": ckpt_sha,
        }
        paged_data = [[
            {"name": "plan.json", "chunks": [plan_content]},
            {"name": "result.json", "chunks": [res_content]},
            {"name": "run_envelope.json", "chunks": [env_content]},
            {"name": "manifest.json", "chunks": [man_content]},
            {"name": "run.log", "chunks": [log_content]},
            {"name": "checkpoint.bin", "chunks": [ckpt_content]},
        ]]
        transport = FakeKaggleTransport(paged_outputs=paged_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            # Simulating native renameat2 unsupported
            with patch("multiautoresearch.adapters.amazon4u._rename_no_replace_linux", return_value=False):
                report = collect_outputs(
                    "amazon4u/test-slug", 1, transport, dest,
                    expected_spec=exp_spec, expected_approval=exp_auth,
                    expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                    expected_output_inventory=exp_inv, expected_envelope=exp_env,
                )
                self.assertEqual(report["status"], "error")
                self.assertIn("Native directory no-replace is unsupported", report["reason"])
                self.assertFalse(dest.exists())

    def test_op10_promotion_collision_refused_negative(self) -> None:
        """Destination directory collision during promotion is refused without deleting existing content."""
        import errno
        # 1. Test helper fake seam refuses collision
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            src.mkdir()
            (src / "data.txt").write_text("source")
            dst = Path(tmp_dir) / "dst"
            dst.mkdir()
            (dst / "existing.txt").write_text("existing")

            with self.assertRaises(FileExistsError):
                fake_directory_promotion_seam(src, dst)

            self.assertTrue(dst.exists())
            self.assertTrue((dst / "existing.txt").exists())
            self.assertEqual((dst / "existing.txt").read_text(), "existing")

        # 2. Test full collect_outputs promotion refuses collision fail-closed
        exp_spec = _canonical_test_spec()
        exp_auth = _canonical_test_auth()
        exp_env = _canonical_test_env()
        exp_man = _canonical_test_manifest()
        plan_content = json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}).encode("utf-8")
        res_dict = {
            "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
            "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
            "run_spec": exp_spec, "run_envelope": exp_env,
            "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
            "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
            "mixture": None, "validation_report": None,
            "remaining_readiness_gates": [
                "GPU-READY separately authorized and verified",
                "O-OFFLINE transport and packaging separately reviewed",
                "Production LightGCN initialization decision",
                "Approved environment and full-data feasibility profiling",
                "Comparative execution/search freeze and final-test authorization",
            ],
            "test_execution": "not_authorized",
        }
        res_content = json.dumps(res_dict).encode("utf-8")
        env_content = json.dumps(exp_env).encode("utf-8")
        man_content = json.dumps(exp_man).encode("utf-8")
        log_content = b"log"
        ckpt_content = b"ckpt_data"
        exp_inv, ckpt_sha = _canonical_test_inventory(
            plan_content, res_content, env_content, man_content, log_content, ckpt_content
        )
        paged_data = [[
            {"name": "plan.json", "chunks": [plan_content]},
            {"name": "result.json", "chunks": [res_content]},
            {"name": "run_envelope.json", "chunks": [env_content]},
            {"name": "manifest.json", "chunks": [man_content]},
            {"name": "run.log", "chunks": [log_content]},
            {"name": "checkpoint.bin", "chunks": [ckpt_content]},
        ]]

        # (a) Modeled EEXIST collision seam reached once without rename fallback
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "eexist_dest"
            transport = FakeKaggleTransport(paged_outputs=paged_data)
            mock_rename = MagicMock(side_effect=FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dest)))
            with patch("multiautoresearch.adapters.amazon4u._rename_no_replace_linux", mock_rename):
                with patch.object(Path, "rename", side_effect=AssertionError("Path.rename fallback prohibited")) as mock_path_rename:
                    report = collect_outputs(
                        "amazon4u/test-slug", 1, transport, dest,
                        expected_spec=exp_spec, expected_approval=exp_auth,
                        expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                        expected_output_inventory=exp_inv, expected_envelope=exp_env,
                    )
                    self.assertEqual(report["status"], "error")
                    self.assertIn("Destination directory appeared concurrently", report["reason"])
                    mock_rename.assert_called_once()
                    mock_path_rename.assert_not_called()
                    self.assertFalse(dest.exists())

        # (b) Modeled False (unsupported) seam reached once without rename fallback
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "unsupported_dest"
            transport = FakeKaggleTransport(paged_outputs=paged_data)
            mock_rename = MagicMock(return_value=False)
            with patch("multiautoresearch.adapters.amazon4u._rename_no_replace_linux", mock_rename):
                with patch.object(Path, "rename", side_effect=AssertionError("Path.rename fallback prohibited")) as mock_path_rename:
                    report = collect_outputs(
                        "amazon4u/test-slug", 1, transport, dest,
                        expected_spec=exp_spec, expected_approval=exp_auth,
                        expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                        expected_output_inventory=exp_inv, expected_envelope=exp_env,
                    )
                    self.assertEqual(report["status"], "error")
                    self.assertIn("Native directory no-replace is unsupported on this platform", report["reason"])
                    mock_rename.assert_called_once()
                    mock_path_rename.assert_not_called()
                    self.assertFalse(dest.exists())

        # (c) Modeled other OSError seam reached once without rename fallback
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "oserror_dest"
            transport = FakeKaggleTransport(paged_outputs=paged_data)
            mock_rename = MagicMock(side_effect=OSError(errno.EIO, "Input/output error"))
            with patch("multiautoresearch.adapters.amazon4u._rename_no_replace_linux", mock_rename):
                with patch.object(Path, "rename", side_effect=AssertionError("Path.rename fallback prohibited")) as mock_path_rename:
                    report = collect_outputs(
                        "amazon4u/test-slug", 1, transport, dest,
                        expected_spec=exp_spec, expected_approval=exp_auth,
                        expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                        expected_output_inventory=exp_inv, expected_envelope=exp_env,
                    )
                    self.assertEqual(report["status"], "error")
                    self.assertIn("Native directory no-replace failed", report["reason"])
                    mock_rename.assert_called_once()
                    mock_path_rename.assert_not_called()
                    self.assertFalse(dest.exists())

        # (d) Precheck collision: destination directory appeared concurrently before promotion
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "concurrent_dest"

            class ConcurrentDestTransport(FakeKaggleTransport):
                def list_output_page(self, kernel_ref, page_token=None):
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / "pre_existing.txt").write_text("critical_user_data")
                    return super().list_output_page(kernel_ref, page_token)

            transport = ConcurrentDestTransport(paged_outputs=paged_data)
            with patch.object(Path, "rename", side_effect=AssertionError("Path.rename fallback prohibited")) as mock_path_rename:
                report = collect_outputs(
                    "amazon4u/test-slug", 1, transport, dest,
                    expected_spec=exp_spec, expected_approval=exp_auth,
                    expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                    expected_output_inventory=exp_inv, expected_envelope=exp_env,
                )
                self.assertEqual(report["status"], "error")
                self.assertIn("Destination directory appeared concurrently", report["reason"])
                mock_path_rename.assert_not_called()
                self.assertTrue(dest.exists())
                existing_file = dest / "pre_existing.txt"
                self.assertTrue(existing_file.exists())
                self.assertEqual(existing_file.read_text(), "critical_user_data")

    def test_op10_promotion_fake_seam_positive(self) -> None:
        """fake_directory_promotion_seam promotes fresh owned quarantine directory cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "quarantine"
            src.mkdir(mode=0o700)
            (src / "artifact.json").write_text('{"status": "ok"}')
            dst = Path(tmp_dir) / "promoted"

            res = fake_directory_promotion_seam(src, dst)
            self.assertTrue(res)
            self.assertTrue(dst.exists())
            self.assertTrue((dst / "artifact.json").exists())
            self.assertFalse(src.exists())

    def test_op11_result_h15_missing_or_extra_fields_negative(self) -> None:
        """result.json missing required H15 fields or containing extra fields is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            exp_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "r1", "approval_id": "a1", "stage": "training_from_scratch",
                "mode": "fixture_verification", "data_kind": "owned_synthetic_packet",
                "method": "LightGCN", "device": "cpu", "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1", "epochs": 1, "max_adam_steps": 1,
            }
            exp_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "r1", "approval_id": "a1",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore", "max_adam_steps": 1,
            }
            exp_env = {
                "schema_version": "amazon4u-H-run-v1", "run_id": "r1", "approval_id": "a1",
                "profile_id": "H-REF-synthetic-cpu-v1", "method": "LightGCN",
                "identity_hash": "a" * 64, "device": "cpu", "accelerator": "none",
                "bundle_sha256": "b" * 64, "initial_state_sha256": "c" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }
            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}))
            env_p = td / "run_envelope.json"
            env_p.write_text(json.dumps(exp_env))
            man_p = td / "manifest.json"
            exp_man = {"schema_version": SCHEMA_MANIFEST, "manifest_digest": "m" * 64, "files": {}, "file_count": 0, "aggregate_size": 0}
            man_p.write_text(json.dumps(exp_man))
            log_p = td / "run.log"
            log_p.write_text("log")
            ckpt_p = td / "checkpoint.bin"
            ckpt_p.write_bytes(b"ckpt")
            ckpt_sha = hashlib.sha256(b"ckpt").hexdigest()

            # Result missing device_capability
            res_bad = {
                "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": exp_spec, "run_envelope": exp_env,
                "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
                "mixture": None, "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p = td / "result.json"
            res_p.write_text(json.dumps(res_bad))

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            exp_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_p.read_bytes()), "sha256": hashlib.sha256(env_p.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "run.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_p.read_bytes()), "physical_sha256": ckpt_sha},
                "checkpoint_physical_sha256": ckpt_sha,
            }
            val = validate_collected_artifacts(
                files, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha, expected_output_inventory=exp_inv,
                expected_envelope=exp_env,
            )
            self.assertEqual(val["status"], "error")
            self.assertIn("15 canonical H fields", val["reason"])

            # Result with extra version field
            res_extra = dict(res_bad)
            res_extra["device_capability"] = {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"}
            res_extra["version"] = 1
            res_p.write_text(json.dumps(res_extra))
            exp_inv["h_result"] = {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()}
            val_extra = validate_collected_artifacts(
                files, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth,
                expected_envelope=exp_env,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha, expected_output_inventory=exp_inv,
            )
            self.assertEqual(val_extra["status"], "error")
            self.assertIn("Transport version must not be inserted", val_extra["reason"])

    def test_op11_result_device_capability_mismatch_negative(self) -> None:
        """result.json with invalid or non-CPU device capability is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            exp_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "r1", "approval_id": "a1", "stage": "training_from_scratch",
                "mode": "fixture_verification", "data_kind": "owned_synthetic_packet",
                "method": "LightGCN", "device": "cpu", "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1", "epochs": 1, "max_adam_steps": 1,
            }
            exp_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "r1", "approval_id": "a1",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore", "max_adam_steps": 1,
            }
            exp_env = {
                "schema_version": "amazon4u-H-run-v1", "run_id": "r1", "approval_id": "a1",
                "profile_id": "H-REF-synthetic-cpu-v1", "method": "LightGCN",
                "identity_hash": "a" * 64, "device": "cpu", "accelerator": "none",
                "bundle_sha256": "b" * 64, "initial_state_sha256": "c" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }
            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}))
            env_p = td / "run_envelope.json"
            env_p.write_text(json.dumps(exp_env))
            man_p = td / "manifest.json"
            exp_man = {"schema_version": SCHEMA_MANIFEST, "manifest_digest": "m" * 64, "files": {}, "file_count": 0, "aggregate_size": 0}
            man_p.write_text(json.dumps(exp_man))
            log_p = td / "run.log"
            log_p.write_text("log")
            ckpt_p = td / "checkpoint.bin"
            ckpt_p.write_bytes(b"ckpt")
            ckpt_sha = hashlib.sha256(b"ckpt").hexdigest()

            res_dict = {
                "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": exp_spec, "run_envelope": exp_env,
                "device_capability": {"device": "cuda", "accelerator": "gpu", "supports_training": True},  # Wrong
                "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
                "mixture": None, "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p = td / "result.json"
            res_p.write_text(json.dumps(res_dict))

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            exp_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_p.read_bytes()), "sha256": hashlib.sha256(env_p.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "run.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_p.read_bytes()), "physical_sha256": ckpt_sha},
                "checkpoint_physical_sha256": ckpt_sha,
            }
            val = validate_collected_artifacts(
                files, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha, expected_output_inventory=exp_inv,
                expected_envelope=exp_env,
            )
            self.assertEqual(val["status"], "error")
            self.assertIn("device_capability mismatch", val["reason"])

    def test_op11_h_authorization_scope_or_budget_negative(self) -> None:
        """H authorization with noncanonical scope or out-of-budget steps is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            exp_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "r1", "approval_id": "a1", "stage": "training_from_scratch",
                "mode": "fixture_verification", "data_kind": "owned_synthetic_packet",
                "method": "LightGCN", "device": "cpu", "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1", "epochs": 1, "max_adam_steps": 1,
            }
            exp_auth_bad = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "r1", "approval_id": "a1",
                "scope": "non_canonical_scope",  # Wrong scope
                "max_adam_steps": 1,
            }
            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth_bad}))
            res_p = td / "result.json"
            res_p.write_text("{}")
            env_p = td / "run_envelope.json"
            env_p.write_text("{}")
            man_p = td / "manifest.json"
            man_p.write_text("{}")
            log_p = td / "run.log"
            log_p.write_text("log")
            ckpt_p = td / "checkpoint.bin"
            ckpt_p.write_bytes(b"ckpt")

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            exp_inv = {
                "plan": {"filename": "plan.json", "size": 10, "sha256": "a" * 64},
                "h_result": {"filename": "result.json", "size": 10, "sha256": "b" * 64},
                "run_envelope": {"filename": "run_envelope.json", "size": 10, "sha256": "c" * 64},
                "source_manifest": {"filename": "manifest.json", "size": 10, "sha256": "d" * 64},
                "log": {"filename": "run.log", "size": 10, "sha256": "e" * 64},
                "checkpoint": {"filename": "checkpoint.bin", "size": 10, "physical_sha256": "f" * 64},
                "checkpoint_physical_sha256": "f" * 64,
            }
            val = validate_collected_artifacts(
                files, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth_bad,
                expected_envelope=_canonical_test_env(),
                expected_manifest={"files": {}}, expected_checkpoint_sha256="f" * 64, expected_output_inventory=exp_inv,
            )
            self.assertEqual(val["status"], "error")
            self.assertIn("scope must be 'tiny_synthetic_cpu_backward_and_private_restore'", val["reason"])

    def test_op11_physical_inventory_size_or_sha_missing_negative(self) -> None:
        """expected_output_inventory with missing size or sha256 is rejected upfront."""
        exp_spec = {
            "schema_version": "amazon4u-H-spec-v1",
            "run_id": "r1", "approval_id": "a1", "stage": "training_from_scratch",
            "mode": "fixture_verification", "data_kind": "owned_synthetic_packet",
            "method": "LightGCN", "device": "cpu", "accelerator": "none",
            "profile_id": "H-REF-synthetic-cpu-v1", "epochs": 1, "max_adam_steps": 1,
        }
        exp_auth = {
            "schema_version": "amazon4u-H-authorization-v1",
            "run_id": "r1", "approval_id": "a1",
            "scope": "tiny_synthetic_cpu_backward_and_private_restore", "max_adam_steps": 1,
        }
        # Missing size for role plan
        exp_inv_no_size = {
            "plan": {"filename": "plan.json", "sha256": "a" * 64},
            "h_result": {"filename": "result.json", "size": 10, "sha256": "b" * 64},
            "run_envelope": {"filename": "run_envelope.json", "size": 10, "sha256": "c" * 64},
            "source_manifest": {"filename": "manifest.json", "size": 10, "sha256": "d" * 64},
            "log": {"filename": "run.log", "size": 10, "sha256": "e" * 64},
            "checkpoint": {"filename": "checkpoint.bin", "size": 10, "physical_sha256": "f" * 64},
            "checkpoint_physical_sha256": "f" * 64,
        }
        val = validate_collected_artifacts(
            {}, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth,
            expected_manifest={"files": {}}, expected_checkpoint_sha256="f" * 64, expected_output_inventory=exp_inv_no_size,
            expected_envelope=_canonical_test_env(),
        )
        self.assertEqual(val["status"], "error")
        self.assertIn("missing required size field", val["reason"])

    def test_op11_checkpoint_physical_sha_confusion_rejected_negative(self) -> None:
        """Supplying wrong-role semantic digest as expected physical SHA fails with SHA mismatch."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            exp_spec = _canonical_test_spec()
            exp_auth = _canonical_test_auth()
            exp_env = _canonical_test_env()
            exp_man = _canonical_test_manifest()
            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}))
            res_dict = {
                "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": exp_spec, "run_envelope": exp_env,
                "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
                "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
                "mixture": None, "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p = td / "result.json"
            res_p.write_text(json.dumps(res_dict))
            env_p = td / "run_envelope.json"
            env_p.write_text(json.dumps(exp_env))
            man_p = td / "manifest.json"
            man_p.write_text(json.dumps(exp_man))
            log_p = td / "run.log"
            log_p.write_text("log")
            ckpt_p = td / "checkpoint.bin"
            ckpt_bytes = b"REAL_BINARY_CHECKPOINT_DATA_42"
            ckpt_p.write_bytes(ckpt_bytes)
            real_physical_sha = hashlib.sha256(ckpt_bytes).hexdigest()

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}

            # 1. Wrong role: semantic identity_hash supplied as expected checkpoint SHA fails
            wrong_semantic_digest = exp_env["identity_hash"]
            inv_wrong = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_p.read_bytes()), "sha256": hashlib.sha256(env_p.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "run.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_bytes), "physical_sha256": wrong_semantic_digest},
                "checkpoint_physical_sha256": wrong_semantic_digest,
            }
            val_fail = validate_collected_artifacts(
                files, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=wrong_semantic_digest, expected_output_inventory=inv_wrong,
                expected_envelope=exp_env,
            )
            self.assertEqual(val_fail["status"], "error")
            self.assertIn("does not match expected", val_fail["reason"])

            # 2. Correct role: physical checksum passes cleanly without error
            inv_correct = dict(inv_wrong)
            inv_correct["checkpoint"] = {"filename": "checkpoint.bin", "size": len(ckpt_bytes), "physical_sha256": real_physical_sha}
            inv_correct["checkpoint_physical_sha256"] = real_physical_sha
            val_ok = validate_collected_artifacts(
                files, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=real_physical_sha, expected_output_inventory=inv_correct,
                expected_envelope=exp_env,
            )
            self.assertEqual(val_ok["status"], "complete")
            self.assertEqual(val_ok["roles"]["checkpoint_physical_sha256"], real_physical_sha)

    def test_op11_forbidden_proxy_metric_in_validation_report_negative(self) -> None:
        """Presence of forbidden proxy metrics like 'eval_score' in validation_report is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            exp_spec = {
                "schema_version": "amazon4u-H-spec-v1",
                "run_id": "r1", "approval_id": "a1", "stage": "training_from_scratch",
                "mode": "fixture_verification", "data_kind": "owned_synthetic_packet",
                "method": "LightGCN", "device": "cpu", "accelerator": "none",
                "profile_id": "H-REF-synthetic-cpu-v1", "epochs": 1, "max_adam_steps": 1,
            }
            exp_auth = {
                "schema_version": "amazon4u-H-authorization-v1",
                "run_id": "r1", "approval_id": "a1",
                "scope": "tiny_synthetic_cpu_backward_and_private_restore", "max_adam_steps": 1,
            }
            exp_env = {
                "schema_version": "amazon4u-H-run-v1", "run_id": "r1", "approval_id": "a1",
                "profile_id": "H-REF-synthetic-cpu-v1", "method": "LightGCN",
                "identity_hash": "a" * 64, "device": "cpu", "accelerator": "none",
                "bundle_sha256": "b" * 64, "initial_state_sha256": "c" * 64,
                "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
            }
            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}))
            env_p = td / "run_envelope.json"
            env_p.write_text(json.dumps(exp_env))
            man_p = td / "manifest.json"
            exp_man = {"schema_version": SCHEMA_MANIFEST, "manifest_digest": "m" * 64, "files": {}, "file_count": 0, "aggregate_size": 0}
            man_p.write_text(json.dumps(exp_man))
            log_p = td / "run.log"
            log_p.write_text("log")
            ckpt_p = td / "checkpoint.bin"
            ckpt_p.write_bytes(b"ckpt")
            ckpt_sha = hashlib.sha256(b"ckpt").hexdigest()

            # Result with forbidden eval_score in validation_report
            res_dict = {
                "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": exp_spec, "run_envelope": exp_env,
                "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
                "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
                "mixture": None,
                "validation_report": {"overall_metrics": {"eval_score": 0.85}},  # Forbidden proxy metric!
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p = td / "result.json"
            res_p.write_text(json.dumps(res_dict))

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            exp_inv = {
                "plan": {"filename": "plan.json", "size": len(plan_p.read_bytes()), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
                "h_result": {"filename": "result.json", "size": len(res_p.read_bytes()), "sha256": hashlib.sha256(res_p.read_bytes()).hexdigest()},
                "run_envelope": {"filename": "run_envelope.json", "size": len(env_p.read_bytes()), "sha256": hashlib.sha256(env_p.read_bytes()).hexdigest()},
                "source_manifest": {"filename": "manifest.json", "size": len(man_p.read_bytes()), "sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()},
                "log": {"filename": "run.log", "size": len(log_p.read_bytes()), "sha256": hashlib.sha256(log_p.read_bytes()).hexdigest()},
                "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_p.read_bytes()), "physical_sha256": ckpt_sha},
                "checkpoint_physical_sha256": ckpt_sha,
            }
            val = validate_collected_artifacts(
                files, expected_version=1, expected_spec=exp_spec, expected_approval=exp_auth,
                expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha, expected_output_inventory=exp_inv,
                expected_envelope=exp_env,
            )
            self.assertEqual(val["status"], "error")
            self.assertIn("Forbidden proxy metric", val["reason"])

    def test_op10_callback_metadata_mutation_prevention_control(self) -> None:
        """Callback mutating expected_approval during transport listing cannot alter pre-callback snapshot."""
        exp_spec = _canonical_test_spec(run_id="run_1", approval_id="auth_1")
        exp_auth = _canonical_test_auth(run_id="run_1", approval_id="auth_1")
        exp_env = _canonical_test_env(run_id="run_1", approval_id="auth_1")
        exp_man = _canonical_test_manifest()
        plan_content = json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": exp_spec, "authorization": exp_auth}).encode("utf-8")
        res_dict = {
            "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
            "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
            "run_spec": exp_spec, "run_envelope": exp_env,
            "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
            "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
            "mixture": None, "validation_report": None,
            "remaining_readiness_gates": [
                "GPU-READY separately authorized and verified",
                "O-OFFLINE transport and packaging separately reviewed",
                "Production LightGCN initialization decision",
                "Approved environment and full-data feasibility profiling",
                "Comparative execution/search freeze and final-test authorization",
            ],
            "test_execution": "not_authorized",
        }
        res_content = json.dumps(res_dict).encode("utf-8")
        env_content = json.dumps(exp_env).encode("utf-8")
        man_content = json.dumps(exp_man).encode("utf-8")
        log_content = b"log"
        ckpt_content = b"ckpt_data"
        exp_inv, ckpt_sha = _canonical_test_inventory(
            plan_content, res_content, env_content, man_content, log_content, ckpt_content
        )
        paged_data = [[
            {"name": "plan.json", "chunks": [plan_content]},
            {"name": "result.json", "chunks": [res_content]},
            {"name": "run_envelope.json", "chunks": [env_content]},
            {"name": "manifest.json", "chunks": [man_content]},
            {"name": "run.log", "chunks": [log_content]},
            {"name": "checkpoint.bin", "chunks": [ckpt_content]},
        ]]

        class MaliciousTransport(FakeKaggleTransport):
            def list_output_page(self, kernel_ref, page_token=None):
                exp_auth["approval_id"] = "MUTATED_APPROVAL_ID"
                exp_auth["max_adam_steps"] = 9999
                return super().list_output_page(kernel_ref, page_token)

        transport = MaliciousTransport(paged_outputs=paged_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            with patch("multiautoresearch.adapters.amazon4u._rename_no_replace_linux", side_effect=fake_directory_promotion_seam):
                report = collect_outputs(
                    "amazon4u/test-slug", 1, transport, dest,
                    expected_spec=exp_spec, expected_approval=exp_auth,
                    expected_manifest=exp_man, expected_checkpoint_sha256=ckpt_sha,
                    expected_output_inventory=exp_inv, expected_envelope=exp_env,
                )
                self.assertEqual(report["status"], "complete")
                self.assertTrue(dest.exists())

    def test_op10_parameter_bounds_ceilings_negative(self) -> None:
        """Parameter values exceeding hard security ceilings fail before mutations."""
        transport = FakeKaggleTransport(paged_outputs=[])
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            spec = _canonical_test_spec()
            auth = _canonical_test_auth()
            env = _canonical_test_env()
            man = _canonical_test_manifest()
            inv, sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")

            # max_pages > 8
            rep = collect_outputs("s", 1, transport, dest, expected_spec=spec, expected_approval=auth, expected_manifest=man, expected_checkpoint_sha256=sha, expected_output_inventory=inv, expected_envelope=env, max_pages=9)
            self.assertEqual(rep["status"], "error")
            self.assertIn("must be an integer in 1..8", rep["reason"])

            # max_files > 128
            rep = collect_outputs("s", 1, transport, dest, expected_spec=spec, expected_approval=auth, expected_manifest=man, expected_checkpoint_sha256=sha, expected_output_inventory=inv, expected_envelope=env, max_files=129)
            self.assertEqual(rep["status"], "error")
            self.assertIn("must be an integer in 1..128", rep["reason"])

            # max_file_bytes > 2097152
            rep = collect_outputs("s", 1, transport, dest, expected_spec=spec, expected_approval=auth, expected_manifest=man, expected_checkpoint_sha256=sha, expected_output_inventory=inv, expected_envelope=env, max_file_bytes=2097153)
            self.assertEqual(rep["status"], "error")
            self.assertIn("must be an integer in 1..2097152", rep["reason"])

            # max_aggregate_bytes > 8388608
            rep = collect_outputs("s", 1, transport, dest, expected_spec=spec, expected_approval=auth, expected_manifest=man, expected_checkpoint_sha256=sha, expected_output_inventory=inv, expected_envelope=env, max_aggregate_bytes=8388609)
            self.assertEqual(rep["status"], "error")
            self.assertIn("must be an integer in 1..8388608", rep["reason"])
            self.assertFalse(dest.exists())

    def test_op10_invalid_clock_samples_fail_closed_negative(self) -> None:
        """Clock returning NaN, Inf, bool, or raising exception fails closed."""
        transport = FakeKaggleTransport(paged_outputs=[[{"name": "file.txt", "content": b"x"}]])
        spec = _canonical_test_spec()
        auth = _canonical_test_auth()
        env = _canonical_test_env()
        man = _canonical_test_manifest()
        inv, sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")

        invalid_clock_samples = [
            lambda: float("nan"),
            lambda: float("inf"),
            lambda: True,
            lambda: "100.0",
        ]
        for bad_clock in invalid_clock_samples:
            with tempfile.TemporaryDirectory() as tmp_dir:
                dest = Path(tmp_dir) / "collected"
                report = collect_outputs(
                    "amazon4u/test-slug", 1, transport, dest,
                    expected_spec=spec, expected_approval=auth,
                    expected_manifest=man, expected_checkpoint_sha256=sha,
                    expected_output_inventory=inv, expected_envelope=env,
                    clock_fn=bad_clock,
                )
                self.assertEqual(report["status"], "error")
                self.assertIn("Clock returned non-finite or invalid sample", report["reason"])
                self.assertFalse(dest.exists())

        def raising_clock():
            raise RuntimeError("Clock hardware failure")

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "collected"
            report = collect_outputs(
                "amazon4u/test-slug", 1, transport, dest,
                expected_spec=spec, expected_approval=auth,
                expected_manifest=man, expected_checkpoint_sha256=sha,
                expected_output_inventory=inv, expected_envelope=env,
                clock_fn=raising_clock,
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("Clock function raised exception", report["reason"])
            self.assertFalse(dest.exists())

    def test_op11_spec_or_auth_bool_budget_rejected_negative(self) -> None:
        """Booleans masquerading as integers in spec/auth versions or budgets are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            spec = _canonical_test_spec()
            auth = _canonical_test_auth()
            env = _canonical_test_env()
            man = _canonical_test_manifest()
            plan_p = td / "plan.json"
            res_p = td / "result.json"
            env_p = td / "run_envelope.json"
            man_p = td / "manifest.json"
            log_p = td / "run.log"
            ckpt_p = td / "checkpoint.bin"
            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            inv, sha = _canonical_test_inventory(b"p", b"r", b"e", b"m", b"l", b"c")

            # 1. Spec with epochs=True
            bad_spec = dict(spec)
            bad_spec["epochs"] = True
            val = validate_collected_artifacts(files, expected_version=1, expected_spec=bad_spec, expected_approval=auth, expected_envelope=env, expected_manifest=man, expected_output_inventory=inv, expected_checkpoint_sha256=sha)
            self.assertEqual(val["status"], "error")
            self.assertIn("must be an integer", val["reason"])

            # 2. Auth with max_adam_steps=True
            bad_auth = dict(auth)
            bad_auth["max_adam_steps"] = True
            val2 = validate_collected_artifacts(files, expected_version=1, expected_spec=spec, expected_approval=bad_auth, expected_envelope=env, expected_manifest=man, expected_output_inventory=inv, expected_checkpoint_sha256=sha)
            self.assertEqual(val2["status"], "error")
            self.assertIn("must be int in range 0..4", val2["reason"])

    def test_op11_device_capability_supports_training_int_rejected_negative(self) -> None:
        """Integer 1 masquerading as boolean True in device_capability supports_training is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            spec = _canonical_test_spec()
            auth = _canonical_test_auth()
            env = _canonical_test_env()
            man = _canonical_test_manifest()

            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": spec, "authorization": auth}))
            env_p = td / "run_envelope.json"
            env_p.write_text(json.dumps(env))
            man_p = td / "manifest.json"
            man_p.write_text(json.dumps(man))
            log_p = td / "run.log"
            log_p.write_text("log")
            ckpt_p = td / "checkpoint.bin"
            ckpt_p.write_bytes(b"ckpt")

            res_bad = {
                "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": spec, "run_envelope": env,
                "device_capability": {
                    "device": "cpu",
                    "accelerator": "none",
                    "supports_training": 1,
                    "state_transfer": "cpu_quiescent_epoch_v1",
                },
                "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
                "mixture": None, "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p = td / "result.json"
            res_p.write_text(json.dumps(res_bad))

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            inv, sha = _canonical_test_inventory(plan_p.read_bytes(), res_p.read_bytes(), env_p.read_bytes(), man_p.read_bytes(), log_p.read_bytes(), ckpt_p.read_bytes())

            val = validate_collected_artifacts(
                files, expected_version=1, expected_spec=spec, expected_approval=auth,
                expected_manifest=man, expected_checkpoint_sha256=sha, expected_output_inventory=inv,
                expected_envelope=env,
            )
            self.assertEqual(val["status"], "error")
            self.assertIn("device_capability mismatch", val["reason"])

    def test_op11_budget_exceeded_rejected_negative(self) -> None:
        """actual_Adam_steps in result exceeding approved budget is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            spec = _canonical_test_spec(max_adam_steps=1)
            auth = _canonical_test_auth(max_adam_steps=1)
            env = _canonical_test_env()
            man = _canonical_test_manifest()

            plan_p = td / "plan.json"
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": spec, "authorization": auth}))
            env_p = td / "run_envelope.json"
            env_p.write_text(json.dumps(env))
            man_p = td / "manifest.json"
            man_p.write_text(json.dumps(man))
            log_p = td / "run.log"
            log_p.write_text("log")
            ckpt_p = td / "checkpoint.bin"
            ckpt_p.write_bytes(b"ckpt")

            res_bad = {
                "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": spec, "run_envelope": env,
                "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
                "method": "LightGCN", "epoch": 1,
                "actual_Adam_steps": 5,
                "selection": {"status": "active"},
                "mixture": None, "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p = td / "result.json"
            res_p.write_text(json.dumps(res_bad))

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            inv, sha = _canonical_test_inventory(plan_p.read_bytes(), res_p.read_bytes(), env_p.read_bytes(), man_p.read_bytes(), log_p.read_bytes(), ckpt_p.read_bytes())

            val = validate_collected_artifacts(
                files, expected_version=1, expected_spec=spec, expected_approval=auth,
                expected_manifest=man, expected_checkpoint_sha256=sha, expected_output_inventory=inv,
                expected_envelope=env,
            )
            self.assertEqual(val["status"], "error")
            self.assertIn("actual_Adam_steps in 1..", val["reason"])

    def test_op11_json_root_non_dict_rejected_negative(self) -> None:
        """Non-dict JSON roots in plan, result, envelope, or manifest fail closed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            spec = _canonical_test_spec()
            auth = _canonical_test_auth()
            env = _canonical_test_env()
            man = _canonical_test_manifest()

            plan_p = td / "plan.json"
            res_p = td / "result.json"
            env_p = td / "run_envelope.json"
            man_p = td / "manifest.json"
            log_p = td / "run.log"
            ckpt_p = td / "checkpoint.bin"

            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": spec, "authorization": auth}))
            res_dict = {
                "schema_version": "amazon4u-H-result-v1", "status": "fixture_budget_complete",
                "acceptance_scope": "H-REF_CPU_reference_fixture_only", "deployment_decision": "NOT_AUTHORIZED",
                "run_spec": spec, "run_envelope": env,
                "device_capability": {"device": "cpu", "accelerator": "none", "supports_training": True, "state_transfer": "cpu_quiescent_epoch_v1"},
                "method": "LightGCN", "epoch": 1, "actual_Adam_steps": 1, "selection": {"status": "active"},
                "mixture": None, "validation_report": None,
                "remaining_readiness_gates": [
                    "GPU-READY separately authorized and verified",
                    "O-OFFLINE transport and packaging separately reviewed",
                    "Production LightGCN initialization decision",
                    "Approved environment and full-data feasibility profiling",
                    "Comparative execution/search freeze and final-test authorization",
                ],
                "test_execution": "not_authorized",
            }
            res_p.write_text(json.dumps(res_dict))
            env_p.write_text(json.dumps(env))
            man_p.write_text(json.dumps(man))
            log_p.write_text("log")
            ckpt_p.write_bytes(b"ckpt")

            files = {"plan.json": plan_p, "result.json": res_p, "run_envelope.json": env_p, "manifest.json": man_p, "run.log": log_p, "checkpoint.bin": ckpt_p}
            inv, sha = _canonical_test_inventory(plan_p.read_bytes(), res_p.read_bytes(), env_p.read_bytes(), man_p.read_bytes(), log_p.read_bytes(), ckpt_p.read_bytes())

            # 1. plan.json is a list [1, 2, 3]
            plan_p.write_text(json.dumps([1, 2, 3]))
            val = validate_collected_artifacts(files, expected_version=1, expected_spec=spec, expected_approval=auth, expected_envelope=env, expected_manifest=man, expected_output_inventory=inv, expected_checkpoint_sha256=sha)
            self.assertEqual(val["status"], "error")
            self.assertIn("must be a JSON dictionary", val["reason"])
            plan_p.write_text(json.dumps({"schema_version": "amazon4u-O-plan-v1", "version": 1, "spec": spec, "authorization": auth}))

            # 2. result.json is a string "invalid"
            res_p.write_text(json.dumps("invalid"))
            val2 = validate_collected_artifacts(files, expected_version=1, expected_spec=spec, expected_approval=auth, expected_envelope=env, expected_manifest=man, expected_output_inventory=inv, expected_checkpoint_sha256=sha)
            self.assertEqual(val2["status"], "error")
            self.assertIn("must be a JSON dictionary", val2["reason"])


class TestCollectionBlockerRegression(unittest.TestCase):
    def _bundle(self, directory, *, result_name="result.json", method="LightGCN"):
        baseline = method == "CorePopularity"
        steps = 0 if baseline else 1
        spec = _canonical_test_spec(method=method, epochs=steps, max_adam_steps=steps)
        auth = _canonical_test_auth(max_adam_steps=steps)
        envelope = _canonical_test_env(method=method)
        manifest = _canonical_test_manifest("a" * 64)
        plan = {"version": 1, "spec": copy.deepcopy(spec), "authorization": copy.deepcopy(auth)}
        result = {
            "schema_version": "amazon4u-H-result-v1",
            "status": "deterministic_validation_complete" if baseline else "fixture_budget_complete",
            "acceptance_scope": "H-REF_CPU_reference_fixture_only",
            "deployment_decision": "NOT_AUTHORIZED",
            "run_spec": copy.deepcopy(spec),
            "run_envelope": copy.deepcopy(envelope),
            "device_capability": {
                "device": "cpu", "accelerator": "none", "supports_training": True,
                "state_transfer": "cpu_quiescent_epoch_v1",
            },
            "method": method, "epoch": steps, "actual_Adam_steps": steps,
            "selection": None if baseline else {"status": "active"},
            "mixture": None, "validation_report": None,
            "remaining_readiness_gates": [
                "GPU-READY separately authorized and verified",
                "O-OFFLINE transport and packaging separately reviewed",
                "Production LightGCN initialization decision",
                "Approved environment and full-data feasibility profiling",
                "Comparative execution/search freeze and final-test authorization",
            ],
            "test_execution": "not_authorized",
        }
        content = {
            "plan.json": json.dumps(plan).encode(), result_name: json.dumps(result).encode(),
            "run_envelope.json": json.dumps(envelope).encode(),
            "manifest.json": json.dumps(manifest).encode(), "run.log": b"fixture log",
        }
        if not baseline:
            content["checkpoint.bin"] = b"fixture checkpoint"
        files = {}
        for name, data in content.items():
            files[name] = directory / name
            files[name].write_bytes(data)
        inventory = {}
        for role, name in (
            ("plan", "plan.json"), ("h_result", result_name),
            ("run_envelope", "run_envelope.json"), ("source_manifest", "manifest.json"),
            ("log", "run.log"),
        ):
            data = content[name]
            inventory[role] = {"filename": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if baseline:
            checkpoint_sha = "not_applicable_deterministic_baseline"
            inventory["checkpoint"] = {
                "status": checkpoint_sha, "applicable": False, "method": method,
                "epoch": 0, "actual_Adam_steps": 0, "selection": None,
            }
        else:
            data = content["checkpoint.bin"]
            checkpoint_sha = hashlib.sha256(data).hexdigest()
            inventory["checkpoint"] = {
                "filename": "checkpoint.bin", "size": len(data), "physical_sha256": checkpoint_sha,
            }
        expected = {
            "expected_version": 1, "expected_spec": spec, "expected_approval": auth,
            "expected_envelope": envelope, "expected_manifest": manifest,
            "expected_checkpoint_sha256": checkpoint_sha, "expected_output_inventory": inventory,
        }
        bundle = {"files": files, "expected": expected, "plan": plan, "result": result}
        self.assertEqual(validate_collected_artifacts(files, **expected)["status"], "complete")
        return bundle

    def _write_artifact(self, bundle, role, value):
        descriptor = bundle["expected"]["expected_output_inventory"][role]
        data = json.dumps(value).encode()
        self.assertLess(len(data), 8192)
        bundle["files"][descriptor["filename"]].write_bytes(data)
        descriptor["size"] = len(data)
        descriptor["sha256"] = hashlib.sha256(data).hexdigest()

    def _metrics(self):
        return {
            "NDCG@5": None, "NDCG@10": 0.5, "NDCG@20": "N/A",
            "HitRate@5": 0.5, "HitRate@10": 0.5, "HitRate@20": 0.5,
            "CatalogCoverage@10": 0.5,
        }

    def _assert_report_rejected(self, bundle):
        with self.subTest(boundary="collection"):
            outcome = validate_collected_artifacts(bundle["files"], **bundle["expected"])
            self.assertEqual(outcome["status"], "error")
            self.assertIn("validation report", outcome["reason"].lower().replace("validation_report", "validation report"))
        with self.subTest(boundary="provenance"):
            with self.assertRaises(Amazon4UError):
                verify_h_provenance(bundle["result"], expected_spec=bundle["expected"]["expected_spec"])
        with self.subTest(boundary="record"):
            with self.assertRaises(Amazon4UError):
                build_amazon4u_record(bundle["plan"], bundle["result"])

    def test_valid_bundle_and_approved_result_aliases(self):
        for result_name in ("result.json", "h_result.json"):
            with self.subTest(result_name=result_name), tempfile.TemporaryDirectory() as tmp:
                bundle = self._bundle(Path(tmp), result_name=result_name)
                self.assertEqual(build_amazon4u_record(bundle["plan"], bundle["result"])["status"], "simulated_offline_success")
                bundle["result"]["validation_report"] = {"overall_metrics": self._metrics()}
                self._write_artifact(bundle, "h_result", bundle["result"])
                self.assertEqual(validate_collected_artifacts(bundle["files"], **bundle["expected"])["status"], "complete")
                self.assertEqual(build_amazon4u_record(bundle["plan"], bundle["result"])["status"], "simulated_offline_success")

    def test_ambiguous_or_unapproved_result_alias_is_rejected(self):
        for keep_approved in (True, False):
            with self.subTest(keep_approved=keep_approved), tempfile.TemporaryDirectory() as tmp:
                bundle = self._bundle(Path(tmp), result_name="h_result.json")
                alias = Path(tmp) / "result.json"
                alias.write_bytes(b"unapproved alias")
                bundle["files"]["result.json"] = alias
                if not keep_approved:
                    del bundle["files"]["h_result.json"]
                outcome = validate_collected_artifacts(bundle["files"], **bundle["expected"])
                self.assertEqual(outcome["status"], "error")
                self.assertIn("alias" if keep_approved else "artifact mismatch", outcome["reason"].lower())

    def test_actual_nested_spec_bool_fields_are_rejected(self):
        for method in ("LightGCN", "CorePopularity"):
            for role, field in ((role, field) for role in ("plan", "h_result") for field in ("epochs", "max_adam_steps")):
                with self.subTest(method=method, role=role, field=field), tempfile.TemporaryDirectory() as tmp:
                    bundle = self._bundle(Path(tmp), method=method)
                    artifact = bundle["plan"] if role == "plan" else bundle["result"]
                    nested_spec = artifact["spec" if role == "plan" else "run_spec"]
                    nested_spec[field] = method == "LightGCN"
                    self._write_artifact(bundle, role, artifact)
                    outcome = validate_collected_artifacts(bundle["files"], **bundle["expected"])
                    self.assertEqual(outcome["status"], "error")
                    self.assertIn("integer", outcome["reason"])
                    if role == "h_result":
                        with self.assertRaises(Amazon4UError):
                            verify_h_provenance(artifact, expected_spec=bundle["expected"]["expected_spec"])
                    with self.assertRaises(Amazon4UError):
                        build_amazon4u_record(bundle["plan"], bundle["result"])

    def test_present_metric_sections_require_dictionaries(self):
        for section in ("overall_metrics", "metrics"):
            for value in (None, [], "invalid"):
                with self.subTest(section=section, value=value), tempfile.TemporaryDirectory() as tmp:
                    bundle = self._bundle(Path(tmp))
                    bundle["result"]["validation_report"] = {"overall_metrics": self._metrics(), section: value}
                    self._write_artifact(bundle, "h_result", bundle["result"])
                    self._assert_report_rejected(bundle)

    def test_unknown_or_retagged_expected_spec_is_rejected(self):
        canonical = _canonical_test_spec()
        for bad in (
            {"schema_version": "unregistered-wrapper", "h_spec": canonical},
            dict(canonical, schema_version="unregistered-spec", profile_id="unapproved-profile"),
        ):
            with self.subTest(schema=bad["schema_version"]):
                with self.assertRaises(Amazon4UError):
                    get_canonical_h_spec(bad)

    def test_invalid_expected_metadata_refused_before_callbacks(self):
        for invalid_role in ("spec", "approval"):
            with self.subTest(invalid_role=invalid_role), tempfile.TemporaryDirectory() as tmp:
                bundle = self._bundle(Path(tmp))
                expected = copy.deepcopy(bundle["expected"])
                if invalid_role == "spec":
                    expected["expected_spec"] = {"schema_version": "unregistered-wrapper", "h_spec": expected["expected_spec"]}
                else:
                    expected["expected_approval"]["schema_version"] = "amazon4u-O-authorization-v1"
                transport = MagicMock()
                transport.list_output_page.return_value = ([], None)
                clock = MagicMock(return_value=0.0)
                destination = Path(tmp) / "published"
                outcome = collect_outputs("fixture", 1, transport, destination, clock_fn=clock, **{
                    key: value for key, value in expected.items() if key != "expected_version"
                })
                transport.list_output_page.assert_not_called()
                clock.assert_not_called()
                self.assertEqual(outcome["status"], "error")
                self.assertFalse(destination.exists())
                self.assertFalse(any(path.name.startswith(".quarantine_") for path in Path(tmp).iterdir()))

    def test_training_proxy_keys_rejected_at_report_root(self):
        for field in ("train_loss", "learning_rate"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                bundle = self._bundle(Path(tmp))
                bundle["result"]["validation_report"] = {"overall_metrics": self._metrics(), field: 0.5}
                self._write_artifact(bundle, "h_result", bundle["result"])
                self._assert_report_rejected(bundle)


class TestGeneratedHydrationRegression(unittest.TestCase):
    def _generated_hydrate(self):
        source = build_amazon4u_kernel_script(_canonical_test_spec(), _canonical_test_auth(), "a" * 64)
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "streaming_hydrate"]
        self.assertEqual(len(functions), 1)
        self.assertFalse(functions[0].decorator_list)
        namespace = {
            "Path": Path, "os": os, "base64": base64, "zlib": zlib,
            "MAX_FILE_BYTES": 64, "MAX_AGGREGATE_BYTES": 128,
            "MAX_CHUNK_BYTES": 8, "MAX_ENCODED_BYTES": 256,
        }
        module = ast.Module(body=functions, type_ignores=[])
        exec(compile(module, "<generated-streaming-hydrate>", "exec"), namespace)
        return namespace["streaming_hydrate"]

    def test_generated_complete_stream_restores_exact_bytes(self):
        hydrate = self._generated_hydrate()
        raw = b"fixture"
        encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            written = hydrate("fixture.txt", encoded, directory, 0, len(raw), hashlib.sha256(raw).hexdigest())
            self.assertEqual(written, len(raw))
            self.assertEqual((directory / "fixture.txt").read_bytes(), raw)
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["fixture.txt"])

    def test_generated_truncated_stream_refused_without_retry_spin(self):
        hydrate = self._generated_hydrate()
        raw = b"fixture"
        compressed = gzip.compress(raw, mtime=0)[:-4]
        encoded = base64.b64encode(compressed).decode()
        real_factory = zlib.decompressobj
        no_progress = []

        class BoundedCodec:
            def __init__(self, *args, **kwargs):
                self.codec = real_factory(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self.codec, name)

            def decompress(self, data, max_length):
                output = self.codec.decompress(data, max_length)
                if not data and not output and not self.codec.eof:
                    no_progress.append(True)
                    if len(no_progress) >= 2:
                        raise AssertionError("Generated hydration retried an exhausted codec without progress")
                return output

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(zlib, "decompressobj", BoundedCodec):
                with self.assertRaisesRegex(ValueError, "Truncated compressed stream"):
                    hydrate("fixture.txt", encoded, directory, 0, len(raw), hashlib.sha256(raw).hexdigest())
            self.assertFalse((directory / "fixture.txt").exists())
            self.assertEqual(list(directory.iterdir()), [])

    def test_generated_trailing_data_refused_without_output(self):
        hydrate = self._generated_hydrate()
        raw = b"fixture"
        encoded = base64.b64encode(gzip.compress(raw, mtime=0) + b"extra").decode()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Trailing data after stream EOF"):
                hydrate("fixture.txt", encoded, directory, 0, len(raw), hashlib.sha256(raw).hexdigest())
            self.assertEqual(list(directory.iterdir()), [])


class TestLogOutputBudgetRegression(unittest.TestCase):
    def test_redaction_expansion_is_capped_in_bytes(self):
        transport = FakeKaggleTransport(log_stream_chunks=["token=a"])
        result = consume_kernel_logs("owner/fixture/v1", 1, transport, max_bytes=7)
        self.assertLessEqual(len(result["logs"].encode("utf-8")), 7)
        self.assertEqual(result["status"], "partial")
        self.assertNotIn("token=a", result["logs"])

    def test_unavailable_diagnostic_is_capped_in_utf8_bytes(self):
        transport = FakeKaggleTransport(log_availability_error=True)
        response = {"returncode": 1, "error": "é" * 8, "reader": None}
        with patch.object(transport, "open_log_reader", return_value=response):
            result = consume_kernel_logs("owner/fixture/v1", 1, transport, max_bytes=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertLessEqual(len(result["error"].encode("utf-8")), 7)


if __name__ == "__main__":
    unittest.main()
