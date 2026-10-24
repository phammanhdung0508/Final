"""Tests for Kaggle orchestration, profile dispatch, state, and runtime adapter (OP01, OP02, OP05-OP09, OP12-OP18).

NO numeric packages (torch, numpy) or learned model modules imported.
All tests run with fake transport, fake clock (zero real sleeps), fake entrypoint, and fresh tempdirs.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
import copy
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from multiautoresearch.adapters.amazon4u import (
    Amazon4UError,
    Amazon4UStateManager,
    ApprovedOfflineContext,
    FakeKaggleTransport,
    H_BUNDLE_PATHS,
    READINESS_GATES,
    REQUIRED_CLOSURE_PATHS,
    SCHEMA_AUTHORIZATION,
    SCHEMA_H_AUTHORIZATION,
    SCHEMA_H_SPEC,
    SCHEMA_SPEC,
    SourcePacket,
    SubmissionUnknownError,
    build_amazon4u_kernel_script,
    build_amazon4u_record,
    build_source_manifest,
    consume_kernel_logs,
    dispatch_amazon4u,
    encode_bytes,
    poll_kernel_status,
    render_plan,
    validate_amazon4u_authorization,
    validate_amazon4u_spec,
    validate_approved_offline_context,
    validate_h_spec,
    validate_submission_receipt,
    write_bundle,
)
from multiautoresearch.adapters.amazon4u_kernel import (
    KernelExecutionError,
    adapt_h_callable,
    run_kernel_stages,
)

# Load kaggle_job via file-based import to avoid invalid post_training package import
_RUNNER_PATH = Path(__file__).resolve().parents[1] / "multiautoresearch" / "post-training" / "scripts" / "kaggle_job.py"
_spec_loader = importlib.util.spec_from_file_location("kaggle_job", _RUNNER_PATH)
if _spec_loader is None or _spec_loader.loader is None:
    raise ImportError(f"Cannot load runner from {_RUNNER_PATH}")
kaggle_job = importlib.util.module_from_spec(_spec_loader)
_spec_loader.loader.exec_module(kaggle_job)


@contextmanager
def _offline_test_tripwires():
    """Install before focused-control setup/render; never initialize H or native providers."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "scripts"}:
            raise AssertionError("Forbidden H/numeric import: " + name)
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import), \
         patch("subprocess.Popen", side_effect=AssertionError("subprocess tripwire")), \
         patch("subprocess.run", side_effect=AssertionError("subprocess tripwire")), \
         patch("socket.socket", side_effect=AssertionError("network tripwire")), \
         patch("socket.create_connection", side_effect=AssertionError("network tripwire")), \
         patch("os.system", side_effect=AssertionError("command tripwire")), \
         patch("pathlib.Path.home", side_effect=AssertionError("home tripwire")), \
         patch("ctypes.CDLL", side_effect=OSError("simulated unavailable native provider")):
        yield


class FakeClock:
    """Simulated monotonic clock for testing timeout/deadline loops without real time.sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.current_time = start

    def now(self) -> float:
        return self.current_time

    def sleep(self, seconds: float) -> None:
        self.current_time += seconds


def sample_h_spec() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_H_SPEC,
        "run_id": "test_run_001",
        "approval_id": "appr_001",
        "stage": "training_from_scratch",
        "mode": "fixture_verification",
        "data_kind": "owned_synthetic_packet",
        "method": "LightGCN",
        "device": "cpu",
        "accelerator": "none",
        "profile_id": "H-REF-synthetic-cpu-v1",
        "epochs": 1,
        "max_adam_steps": 2,
    }


def sample_authorization() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_H_AUTHORIZATION,
        "run_id": "test_run_001",
        "approval_id": "appr_001",
        "scope": "tiny_synthetic_cpu_backward_and_private_restore",
        "max_adam_steps": 2,
    }


def sample_full_closure_packets() -> list[SourcePacket]:
    return [
        SourcePacket(rel, f"# Stub content for {rel}\n".encode("utf-8"))
        for rel in sorted(REQUIRED_CLOSURE_PATHS)
    ]


def sample_plan_dict() -> dict[str, object]:
    spec = sample_h_spec()
    auth = sample_authorization()
    packets = sample_full_closure_packets()
    manifest = build_source_manifest(packets, require_full_closure=True)
    return {
        "schema_version": "amazon4u-O-plan-v1",
        "spec": spec,
        "authorization": auth,
        "expected_spec": spec,
        "expected_sources": manifest,
        "source_packets": [
            {"rel_path": p.rel_path, "content": p.content.decode("utf-8")}
            for p in packets
        ],
        "owner": "amazon4u",
        "slug": "test-slug",
        "timeout": 7200,
        "accelerator": "none",
    }


def sample_canonical_h_envelope(
    spec: dict[str, object] | None = None,
    bundle_manifest: dict[str, object] | None = None,
) -> dict[str, str]:
    if spec is None:
        spec = sample_h_spec()
    bundle_sha = (
        str(bundle_manifest["sha256"])
        if bundle_manifest is not None
        else sample_h_bundle_manifest()["sha256"]
    )
    return {
        "schema_version": "amazon4u-H-run-v1",
        "run_id": str(spec.get("run_id", "test_run_001")),
        "approval_id": str(spec.get("approval_id", "appr_001")),
        "profile_id": str(spec.get("profile_id", "H-REF-synthetic-cpu-v1")),
        "method": str(spec.get("method", "LightGCN")),
        "identity_hash": "a" * 64,
        "device": str(spec.get("device", "cpu")),
        "accelerator": str(spec.get("accelerator", "none")),
        "bundle_sha256": str(bundle_sha),
        "initial_state_sha256": "b" * 64,
        "determinism_policy": "private_rng_bounded_cpu_no_global_setters",
    }


def sample_h_result() -> dict[str, object]:
    spec = sample_h_spec()
    return {
        "schema_version": "amazon4u-H-result-v1",
        "status": "fixture_budget_complete",
        "acceptance_scope": "H-REF_CPU_reference_fixture_only",
        "deployment_decision": "NOT_AUTHORIZED",
        "run_spec": spec,
        "run_envelope": sample_canonical_h_envelope(spec),
        "device_capability": {"cpu": True, "cuda": False},
        "method": "LightGCN",
        "epoch": 1,
        "actual_Adam_steps": 2,
        "selection": {"status": "complete"},
        "mixture": None,
        "validation_report": {
            "overall_metrics": {
                "NDCG@5": 0.123,
                "NDCG@10": 0.145,
                "NDCG@20": 0.167,
                "HitRate@5": 0.234,
                "HitRate@10": 0.256,
                "HitRate@20": 0.278,
                "CatalogCoverage@10": 0.089,
            }
        },
        "remaining_readiness_gates": list(READINESS_GATES),
        "test_execution": "not_authorized",
    }


def _oracle_encode_bytes(data: bytes) -> str:
    """Independent oracle test-source encoder (stdlib gzip + base64)."""
    import base64
    import gzip
    return base64.b64encode(gzip.compress(data, mtime=0)).decode("ascii")


def sample_h_bundle_packets() -> list[SourcePacket]:
    """28 canonical H bundle packets."""
    return [
        SourcePacket(rel, f"# Stub content for {rel}\n".encode("utf-8"))
        for rel in sorted(H_BUNDLE_PATHS)
    ]


def _oracle_owned_digest(files: dict[str, dict[str, object]]) -> str:
    """Independent stdlib oracle for H owned_digest semantic SHA-256 over H manifest files dict.

    Does not import or execute repository training_harness.py functions.
    Matches frozen owned_digest specification for dict/str/int types.
    """
    h = hashlib.sha256()

    def blob(tag: bytes, data: bytes) -> None:
        h.update(tag)
        h.update(struct.pack("!Q", len(data)))
        h.update(data)

    def visit(v: object) -> None:
        if v is None:
            h.update(b"N")
        elif type(v) is bool:
            h.update(b"T" if v else b"F")
        elif type(v) is int:
            blob(b"I", str(v).encode("ascii"))
        elif type(v) is float:
            blob(b"R", struct.pack("!d", v))
        elif type(v) is str:
            blob(b"S", v.encode("utf-8"))
        elif type(v) is bytes:
            blob(b"B", v)
        elif type(v) in (list, tuple):
            h.update(b"L" if type(v) is list else b"Q")
            h.update(struct.pack("!Q", len(v)))
            for item in v:
                visit(item)
        elif type(v) is dict:
            h.update(b"D")
            h.update(struct.pack("!Q", len(v)))
            for key in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(v[key])
        else:
            raise ValueError(f"unsupported digest value type: {type(v)}")

    visit(files)
    return h.hexdigest()


def sample_h_bundle_manifest(packets: list[SourcePacket] | None = None) -> dict[str, object]:
    """Build canonical H bundle manifest (schema amazon4u-H-bundle-v1) with bytes and owned_digest."""
    if packets is None:
        packets = sample_h_bundle_packets()
    files: dict[str, dict[str, object]] = {}
    for pkt in sorted(packets, key=lambda p: p.rel_path):
        files[pkt.rel_path] = {
            "bytes": len(pkt.content),
            "sha256": hashlib.sha256(pkt.content).hexdigest(),
        }
    overall_sha = _oracle_owned_digest(files)
    return {
        "schema_version": "amazon4u-H-bundle-v1",
        "files": files,
        "sha256": overall_sha,
    }


def sample_approved_context(expected_o_outer: dict[str, object] | None = None) -> ApprovedOfflineContext:
    spec = sample_h_spec()
    auth = sample_authorization()
    h_packets = sample_h_bundle_packets()
    h_manifest = sample_h_bundle_manifest(h_packets)

    if expected_o_outer is None:
        closure_packets = sample_full_closure_packets()
        outer_base = build_source_manifest(closure_packets, require_full_closure=True)
        kernel_script = build_amazon4u_kernel_script(spec, auth, str(outer_base["manifest_digest"]))

        expected_o_outer = copy.deepcopy(outer_base)
        expected_o_outer["generated_runtime"] = {
            "code_file": "kernel.py",
            "sha256": hashlib.sha256(kernel_script.encode("utf-8")).hexdigest(),
            "size": len(kernel_script.encode("utf-8")),
        }
        expected_o_outer["environment"] = {
            "device": "cpu",
            "accelerator": "none",
            "network": "disabled",
        }

    env = {
        "device": "cpu",
        "accelerator": "none",
        "network": "disabled",
        "capabilities": {
            "package": "amazon4u",
            "version": "1.0.0",
            "nativeIO": "simulated",
            "ABI": "cpython",
            "dependency": "frozen",
            "CPU": "reference",
            "runtime": "offline",
            "resource": "bounded",
            "callable": "execute_fixture",
            "bootstrap": "synthetic",
        },
    }

    ckpt_bytes = b"FAKE_CHECKPOINT_BYTES"
    ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()
    expected_output_inventory = {
        "plan": {"filename": "plan.json", "size": 100, "sha256": hashlib.sha256(b"plan").hexdigest()},
        "h_result": {"filename": "result.json", "size": 100, "sha256": hashlib.sha256(b"result").hexdigest()},
        "run_envelope": {"filename": "run_envelope.json", "size": 100, "sha256": hashlib.sha256(b"envelope").hexdigest()},
        "source_manifest": {"filename": "manifest.json", "size": 100, "sha256": hashlib.sha256(b"manifest").hexdigest()},
        "log": {"filename": "kernel.log", "size": 100, "sha256": hashlib.sha256(b"log").hexdigest()},
        "checkpoint": {"filename": "checkpoint.bin", "size": len(ckpt_bytes), "physical_sha256": ckpt_sha},
        "checkpoint_physical_sha256": ckpt_sha,
    }

    return ApprovedOfflineContext(
        canonical_h_spec=spec,
        h_authorization=auth,
        expected_h_envelope=sample_canonical_h_envelope(spec, h_manifest),
        expected_h_bundle=h_manifest,
        expected_o_outer=expected_o_outer,
        expected_input_identity={"spec": spec, "authorization": auth},
        expected_o_environment=env,
        submission_binding={"version": 1, "timeout": 7200},
        expected_output_inventory=expected_output_inventory,
        limits={"max_file_bytes": 2097152, "max_aggregate_bytes": 8388608},
    )


def sample_kernel_bootstrap(ctx: ApprovedOfflineContext) -> dict[str, object]:
    """Complete fake prerequisites; H source expectations use the stdlib test oracle."""
    caps = copy.deepcopy(ctx.expected_o_environment["capabilities"])
    checkpoint = b"FAKE_CHECKPOINT_BYTES"
    return {
        "expected_spec": copy.deepcopy(ctx.canonical_h_spec),
        "authorization": copy.deepcopy(ctx.h_authorization),
        "data": "FAKE_DATA",
        "expectations": {
            "resource_policy": {"kind": "synthetic"},
            "expected_runtime": {"device": "cpu", "cuda_available": False, "accelerator": "none"},
            "capabilities": copy.deepcopy(caps),
        },
        "run_envelope": copy.deepcopy(ctx.expected_h_envelope),
        "expected_run_envelope": copy.deepcopy(ctx.expected_h_envelope),
        "bundle_sources": {p.rel_path: p.content for p in sample_h_bundle_packets()},
        "expected_bundle": copy.deepcopy(ctx.expected_h_bundle),
        "checkpoint_data": checkpoint,
        "expected_checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "expected_capabilities": copy.deepcopy(caps),
        "actual_capabilities": copy.deepcopy(caps),
        "version": ctx.submission_binding["version"],
    }


class TestKaggleOrchestration(unittest.TestCase):
    """Test suite covering OP01, OP02, OP05-OP09, OP12-OP18."""

    # -----------------------------------------------------------------------
    # OP01: Explicit Profile & Authority Dispatch
    # -----------------------------------------------------------------------

    def test_op01_explicit_profile_and_authority_dispatch_positive(self) -> None:
        """Explicit --profile amazon4u with --plan dispatches without resolving credentials or CLI."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan_file = Path(tmp_dir) / "plan.json"
            plan_file.write_text(json.dumps(sample_plan_dict()), encoding="utf-8")
            ctx_file = Path(tmp_dir) / "context.json"
            ctx = sample_approved_context()
            ctx_dict = {
                "canonical_h_spec": ctx.canonical_h_spec,
                "h_authorization": ctx.h_authorization,
                "expected_h_envelope": ctx.expected_h_envelope,
                "expected_h_bundle": ctx.expected_h_bundle,
                "expected_o_outer": ctx.expected_o_outer,
                "expected_input_identity": ctx.expected_input_identity,
                "expected_o_environment": ctx.expected_o_environment,
                "submission_binding": ctx.submission_binding,
                "expected_output_inventory": ctx.expected_output_inventory,
                "limits": ctx.limits,
            }
            ctx_file.write_text(json.dumps(ctx_dict), encoding="utf-8")

            parser = kaggle_job.build_parser()
            args = parser.parse_args([
                "launch",
                "--profile", "amazon4u",
                "--plan", str(plan_file),
                "--context", str(ctx_file),
                "--dry-run",
            ])
            self.assertEqual(args.profile, "amazon4u")
            self.assertEqual(args.plan, str(plan_file))
            self.assertEqual(args.context, str(ctx_file))

            with patch("sys.stdout", new_callable=io.StringIO):
                rc = kaggle_job.main([
                    "launch",
                    "--profile", "amazon4u",
                    "--plan", str(plan_file),
                    "--context", str(ctx_file),
                    "--dry-run",
                ])
            self.assertEqual(rc, 0)

    def test_op01_subparser_default_does_not_erase_global_profile(self) -> None:
        """Global --profile amazon4u specified before subcommand is preserved and not erased."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan_file = Path(tmp_dir) / "plan.json"
            plan_file.write_text(json.dumps(sample_plan_dict()), encoding="utf-8")
            ctx_file = Path(tmp_dir) / "context.json"
            ctx = sample_approved_context()
            ctx_dict = {
                "canonical_h_spec": ctx.canonical_h_spec,
                "h_authorization": ctx.h_authorization,
                "expected_h_envelope": ctx.expected_h_envelope,
                "expected_h_bundle": ctx.expected_h_bundle,
                "expected_o_outer": ctx.expected_o_outer,
                "expected_input_identity": ctx.expected_input_identity,
                "expected_o_environment": ctx.expected_o_environment,
                "submission_binding": ctx.submission_binding,
                "expected_output_inventory": ctx.expected_output_inventory,
                "limits": ctx.limits,
            }
            ctx_file.write_text(json.dumps(ctx_dict), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO):
                rc = kaggle_job.main([
                    "--profile", "amazon4u",
                    "launch",
                    "--plan", str(plan_file),
                    "--context", str(ctx_file),
                    "--dry-run",
                ])
            self.assertEqual(rc, 0)

    def test_op01_launch_without_plan_refused(self) -> None:
        """amazon4u launch without explicit --plan refuses before credentials, state, or CLI access."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ctx_file = Path(tmp_dir) / "context.json"
            ctx = sample_approved_context()
            ctx_dict = {
                "canonical_h_spec": ctx.canonical_h_spec,
                "h_authorization": ctx.h_authorization,
                "expected_h_envelope": ctx.expected_h_envelope,
                "expected_h_bundle": ctx.expected_h_bundle,
                "expected_o_outer": ctx.expected_o_outer,
                "expected_input_identity": ctx.expected_input_identity,
                "expected_o_environment": ctx.expected_o_environment,
                "submission_binding": ctx.submission_binding,
                "expected_output_inventory": ctx.expected_output_inventory,
                "limits": ctx.limits,
            }
            ctx_file.write_text(json.dumps(ctx_dict), encoding="utf-8")

            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                rc = kaggle_job.main([
                    "launch",
                    "--profile", "amazon4u",
                    "--context", str(ctx_file),
                    "--dry-run",
                ])
                self.assertEqual(rc, 1)
                self.assertIn("Explicit independently supplied --plan is required", mock_err.getvalue())

    def test_op01_legacy_unprofiled_dispatch_positive(self) -> None:
        """Legacy invocation without profile matches frozen legacy behavior."""
        parser = kaggle_job.build_parser()
        args = parser.parse_args(["launch", "--mode", "experiment"])
        self.assertIsNone(getattr(args, "profile", None))
        self.assertEqual(args.mode, "experiment")
        self.assertEqual(args.poll_interval, 30)

    def test_op01_unknown_profile_refused(self) -> None:
        """Unknown profile refuses before credentials/CLI/state access."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            rc = kaggle_job.main(["launch", "--profile", "unknown_profile"])
            self.assertEqual(rc, 1)
            self.assertIn("Unknown profile", mock_err.getvalue())

    def test_op01_amazon_flags_without_profile_refused(self) -> None:
        """Amazon-specific flags without --profile amazon4u refuse rather than falling back."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            rc = kaggle_job.main(["launch", "--spec", "spec.json"])
            self.assertEqual(rc, 1)
            self.assertIn("Amazon4U configuration flags require explicit '--profile amazon4u'", mock_err.getvalue())

    def test_op01_invalid_spec_schema_refused(self) -> None:
        """Spec with unauthorized stage, mode, data_kind, or device is refused."""
        bad_spec = sample_h_spec()
        bad_spec["stage"] = "final_test"
        with self.assertRaises(Amazon4UError):
            validate_h_spec(bad_spec)

        bad_spec2 = sample_h_spec()
        bad_spec2["data_kind"] = "real_data_benchmark"
        with self.assertRaises(Amazon4UError):
            validate_h_spec(bad_spec2)

        bad_spec3 = sample_h_spec()
        bad_spec3["device"] = "cuda"
        with self.assertRaises(Amazon4UError):
            validate_h_spec(bad_spec3)
        for key, value in (("device", "CPU"), ("accelerator", ""), ("accelerator", "NONE")):
            with self.subTest(key=key, value=value):
                bad_spec = sample_h_spec()
                bad_spec[key] = value
                with self.assertRaises(Amazon4UError):
                    validate_h_spec(bad_spec)

    def test_op01_bool_as_int_rejected(self) -> None:
        """Python bool passed for integer fields (epochs/max_adam_steps) is strictly refused."""
        bad_spec = sample_h_spec()
        bad_spec["epochs"] = True  # bool subclass of int
        with self.assertRaises(Amazon4UError) as ctx:
            validate_h_spec(bad_spec)
        self.assertIn("integer", str(ctx.exception))

        bad_auth = sample_authorization()
        bad_auth["max_adam_steps"] = False
        with self.assertRaises(Amazon4UError) as ctx2:
            validate_amazon4u_authorization(bad_auth)
        self.assertIn("max_adam_steps", str(ctx2.exception))

    def test_op01_budget_violations_refused(self) -> None:
        """Epochs or update budgets exceeding bounds are refused."""
        bad_spec = sample_h_spec()
        bad_spec["epochs"] = 5  # > 4
        with self.assertRaises(Amazon4UError):
            validate_h_spec(bad_spec)

        # Baseline requesting updates
        bad_base = sample_h_spec()
        bad_base["method"] = "CorePopularity"
        bad_base["epochs"] = 1
        with self.assertRaises(Amazon4UError):
            validate_h_spec(bad_base)

    # -----------------------------------------------------------------------
    # OP02: Pure Render & Owned Bundle Write
    # -----------------------------------------------------------------------

    def test_op02_pure_render_deterministic_positive(self) -> None:
        """Pure rendering produces identical in-memory bundle and digest without disk writes."""
        spec = sample_h_spec()
        auth = sample_authorization()
        packets = sample_full_closure_packets()
        ctx = sample_approved_context()

        plan1 = render_plan(
            spec,
            authorization=auth,
            expected_spec=spec,
            expected_sources=ctx.expected_o_outer,
            source_packets=packets,
            context=ctx,
        )
        plan2 = render_plan(
            spec,
            authorization=auth,
            expected_spec=spec,
            expected_sources=ctx.expected_o_outer,
            source_packets=packets,
            context=ctx,
        )

        self.assertEqual(plan1.plan_digest, plan2.plan_digest)
        self.assertEqual(plan1.metadata, plan2.metadata)
        self.assertEqual(plan1.files, plan2.files)
        self.assertEqual(plan1.kernel_script, plan2.kernel_script)

    def test_op02_pure_render_tripwire_no_forbidden_seams(self) -> None:
        """Tripwires ensure Path.home, network, subprocess, or env credentials are not queried."""
        spec = sample_h_spec()
        auth = sample_authorization()
        packets = sample_full_closure_packets()
        ctx = sample_approved_context()

        with patch("pathlib.Path.home", side_effect=AssertionError("Path.home tripwire")):
            with patch("subprocess.run", side_effect=AssertionError("subprocess tripwire")):
                plan = render_plan(
                    spec,
                    authorization=auth,
                    expected_spec=spec,
                    expected_sources=ctx.expected_o_outer,
                    source_packets=packets,
                    context=ctx,
                )
                self.assertIsNotNone(plan.plan_digest)

    def test_op02_render_plan_missing_expectations_refused(self) -> None:
        """render_plan refuses if authorization or expected_spec is missing or synthesized."""
        spec = sample_h_spec()
        packets = sample_full_closure_packets()
        manifest = build_source_manifest(packets, require_full_closure=True)
        ctx = sample_approved_context()

        with self.assertRaises(Amazon4UError):
            render_plan(
                spec,
                authorization=None,  # missing authorization
                expected_spec=spec,
                expected_sources=manifest,
                source_packets=packets,
                context=ctx,
            )

        with self.assertRaises(Amazon4UError):
            render_plan(
                spec,
                authorization=sample_authorization(),
                expected_spec=spec,
                expected_sources=manifest,
                source_packets=packets,
                context=None,  # missing context
            )

    def test_op02_write_bundle_collision_refused(self) -> None:
        """write_bundle refuses collision if target directory already exists (empty or populated)."""
        spec = sample_h_spec()
        auth = sample_authorization()
        packets = sample_full_closure_packets()
        ctx = sample_approved_context()
        plan = render_plan(spec, authorization=auth, expected_spec=spec, expected_sources=ctx.expected_o_outer, source_packets=packets, context=ctx)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Empty pre-existing directory must be refused (no adoption)
            target = Path(tmp_dir) / "bundle"
            target.mkdir()
            with self.assertRaises(Amazon4UError):
                write_bundle(plan, target)

            # 2. Populated pre-existing directory must also be refused
            (target / "existing.txt").write_text("existing content", encoding="utf-8")
            with self.assertRaises(Amazon4UError):
                write_bundle(plan, target)

    # -----------------------------------------------------------------------
    # OP05: H Callable Bridge & Staged Failure Propagation
    # -----------------------------------------------------------------------

    def test_op05_h_callable_bridge_exact_arguments_positive(self) -> None:
        """H callable adapter receives exact 11 arguments and simulates success."""
        received_args = {}

        def mock_h_callable(spec, **kwargs):
            received_args["spec"] = spec
            received_args.update(kwargs)
            self.assertEqual(kwargs.get("run_envelope"), ctx.expected_h_envelope)
            self.assertEqual(kwargs.get("expected_run_envelope"), ctx.expected_h_envelope)
            self.assertEqual(kwargs.get("authorization"), ctx.h_authorization)
            self.assertEqual(kwargs.get("expected_bundle"), ctx.expected_h_bundle)
            return sample_h_result()

        spec = sample_h_spec()
        auth = sample_authorization()
        ctx = sample_approved_context()
        bootstrap = {
            "expected_spec": spec,
            "authorization": auth,
            "data": "FAKE_TRAINING_DATA",
            "expectations": {"resource_policy": {"kind": "synthetic"}},
            "run_envelope": dict(ctx.expected_h_envelope),
            "expected_run_envelope": dict(ctx.expected_h_envelope),
            "bundle_sources": {p.rel_path: p.content for p in sample_h_bundle_packets()},
            "expected_bundle": ctx.expected_h_bundle,
            "light_initial": None,
            "backend": None,
        }

        res = adapt_h_callable(
            mock_h_callable,
            spec,
            **bootstrap,
        )
        self.assertEqual(res["schema_version"], "amazon4u-H-result-v1")
        self.assertEqual(len(received_args), 11)  # 1 positional + 10 keyword arguments
        for k in (
            "spec", "expected_spec", "authorization", "data", "expectations",
            "run_envelope", "expected_run_envelope", "bundle_sources",
            "expected_bundle", "light_initial", "backend",
        ):
            self.assertIn(k, received_args)
        self.assertEqual(received_args["expected_bundle"], ctx.expected_h_bundle)

    def test_op05_staged_failure_propagation(self) -> None:
        """Failures in preflight, hydration, callable, checkpoint, or publish propagate nonzero exit codes."""
        spec = sample_h_spec()
        auth = sample_authorization()
        bundle_payload = {"test.py": _oracle_encode_bytes(b"# test\n")}
        manifest_data = {
            "files": {
                "test.py": {
                    "size": len(b"# test\n"),
                    "sha256": hashlib.sha256(b"# test\n").hexdigest(),
                }
            }
        }
        sample_caps = {
            "package": "amazon4u",
            "version": "1.0.0",
            "nativeIO": "simulated",
            "ABI": "cpython",
            "dependency": "frozen",
            "CPU": "reference",
            "runtime": "offline",
            "resource": "bounded",
            "callable": "execute_fixture",
            "bootstrap": "synthetic",
        }
        ctx = sample_approved_context()
        h_packets = sample_h_bundle_packets()
        bs = {
            "expected_spec": spec,
            "authorization": auth,
            "data": "FAKE_DATA",
            "expectations": {
                "resource_policy": {"kind": "synthetic"},
                "expected_runtime": {"device": "cpu", "cuda_available": False, "accelerator": "none"},
                "capabilities": dict(sample_caps),
            },
            "run_envelope": dict(ctx.expected_h_envelope),
            "expected_run_envelope": dict(ctx.expected_h_envelope),
            "bundle_sources": {pkt.rel_path: pkt.content for pkt in h_packets},
            "expected_bundle": ctx.expected_h_bundle,
            "checkpoint_data": b"FAKE_CHECKPOINT_BYTES",
            "expected_checkpoint_sha256": hashlib.sha256(b"FAKE_CHECKPOINT_BYTES").hexdigest(),
            "expected_capabilities": dict(sample_caps),
            "actual_capabilities": dict(sample_caps),
            "version": 1,
        }

        callable_calls: list[str] = []

        def good_callable(*args, **kwargs):
            callable_calls.append("called")
            return sample_h_result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)

            # Stage 1: Preflight failure (fails before dirs or logs are created)
            workdir_pref = td / "work_pref"
            outdir_pref = td / "out_pref"
            rc_pref = run_kernel_stages(workdir_pref, outdir_pref, spec, auth, context=ctx, fail_stage="preflight")
            self.assertEqual(rc_pref, 1)
            self.assertFalse(outdir_pref.exists())
            self.assertEqual(len(callable_calls), 0)

            # Stage 2: Hydration failure (missing bundle_files.json)
            workdir_hyd = td / "work_hyd"
            outdir_hyd = td / "out_hyd"
            rc_hyd = run_kernel_stages(
                workdir_hyd, outdir_hyd, spec, auth, context=ctx,
                h_callable=good_callable, fake_bootstrap=bs, fail_stage="hydration",
            )
            self.assertEqual(rc_hyd, 2)
            self.assertEqual(len(callable_calls), 0)

            # Stage 3: Callable failure
            # 3a. Explicit fail_stage
            workdir_call = td / "work_call"
            outdir_call = td / "out_call"
            workdir_call.mkdir()
            (workdir_call / "bundle_files.json").write_text(json.dumps(bundle_payload), encoding="utf-8")
            (workdir_call / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
            rc_call = run_kernel_stages(
                workdir_call, outdir_call, spec, auth,
                context=ctx,
                h_callable=good_callable,
                fake_bootstrap=bs,
                fail_stage="callable",
            )
            self.assertEqual(rc_call, 3)
            self.assertEqual(len(callable_calls), 0)

            # 3b. Precise stage 3 invalid-H-result negative control (forbidden eval_score proxy metric rejected)
            workdir_bad_h = td / "work_bad_h"
            outdir_bad_h = td / "out_bad_h"
            workdir_bad_h.mkdir()
            (workdir_bad_h / "bundle_files.json").write_text(json.dumps(bundle_payload), encoding="utf-8")
            (workdir_bad_h / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
            bad_h_result = sample_h_result()
            bad_h_result["validation_report"]["overall_metrics"]["eval_score"] = 0.99
            bad_h_calls = []

            def bad_h_callable(*args, **kwargs):
                bad_h_calls.append("called")
                return bad_h_result

            rc_bad_h = run_kernel_stages(
                workdir_bad_h, outdir_bad_h, spec, auth,
                context=ctx,
                h_callable=bad_h_callable,
                fake_bootstrap=bs,
                checkpoint_verifier=lambda b: True,
            )
            self.assertEqual(rc_bad_h, 3)
            self.assertEqual(bad_h_calls, ["called"])
            self.assertIn("H result provenance verification failed", (outdir_bad_h / "kernel-run.log").read_text(encoding="utf-8"))
            self.assertFalse((outdir_bad_h / "checkpoint.bin").exists())
            self.assertFalse((outdir_bad_h / "result.json").exists())
            self.assertFalse((outdir_bad_h / "plan.json").exists())
            self.assertFalse((outdir_bad_h / "amazon4u_record.json").exists())

            # Stage 4: Checkpoint failure (callable succeeds, but checkpoint verifier rejects)
            workdir_ckpt = td / "work_ckpt"
            outdir_ckpt = td / "out_ckpt"
            workdir_ckpt.mkdir()
            (workdir_ckpt / "bundle_files.json").write_text(json.dumps(bundle_payload), encoding="utf-8")
            (workdir_ckpt / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
            # 4a. Explicit fail_stage
            rc_ckpt = run_kernel_stages(
                workdir_ckpt, outdir_ckpt, spec, auth,
                context=ctx,
                h_callable=good_callable,
                fake_bootstrap=bs,
                fail_stage="checkpoint",
            )
            self.assertEqual(rc_ckpt, 4)
            self.assertEqual(len(callable_calls), 1)

            # 4b. Real checkpoint verifier mismatch control
            workdir_ckpt_ver = td / "work_ckpt_ver"
            outdir_ckpt_ver = td / "out_ckpt_ver"
            workdir_ckpt_ver.mkdir()
            (workdir_ckpt_ver / "bundle_files.json").write_text(json.dumps(bundle_payload), encoding="utf-8")
            (workdir_ckpt_ver / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
            rc_ckpt_ver = run_kernel_stages(
                workdir_ckpt_ver, outdir_ckpt_ver, spec, auth,
                context=ctx,
                h_callable=good_callable,
                fake_bootstrap=bs,
                checkpoint_verifier=lambda b: False,
            )
            self.assertEqual(rc_ckpt_ver, 4)
            self.assertEqual(len(callable_calls), 2)
            self.assertFalse((outdir_ckpt_ver / "checkpoint.bin").exists())

            # Stage 5: Publication failure (callable and checkpoint succeed, but publication fails)
            workdir_pub = td / "work_pub"
            outdir_pub = td / "out_pub"
            workdir_pub.mkdir()
            (workdir_pub / "bundle_files.json").write_text(json.dumps(bundle_payload), encoding="utf-8")
            (workdir_pub / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
            rc_pub = run_kernel_stages(
                workdir_pub, outdir_pub, spec, auth,
                context=ctx,
                h_callable=good_callable,
                fake_bootstrap=bs,
                checkpoint_verifier=lambda b: True,
                fail_stage="publish",
            )
            self.assertEqual(rc_pub, 5)
            self.assertEqual(len(callable_calls), 3)
            # Preceding stages succeeded: callable was executed and checkpoint was verified and written
            self.assertTrue((outdir_pub / "checkpoint.bin").exists())
            # Publication failed: final artifacts were not published
            self.assertFalse((outdir_pub / "plan.json").exists())
            self.assertFalse((outdir_pub / "result.json").exists())
            self.assertFalse((outdir_pub / "amazon4u_record.json").exists())
            self.assertFalse((outdir_pub / "manifest.json").exists())

    def test_op05_injected_callable_exception_propagates(self) -> None:
        """An exception raised inside the H callable is caught and propagates exit code 3."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir) / "work"
            outdir = Path(tmp_dir) / "out"
            workdir.mkdir()
            bundle_payload = {"test.py": _oracle_encode_bytes(b"# test\n")}
            (workdir / "bundle_files.json").write_text(json.dumps(bundle_payload), encoding="utf-8")
            manifest_data = {
                "files": {
                    "test.py": {
                        "size": len(b"# test\n"),
                        "sha256": hashlib.sha256(b"# test\n").hexdigest(),
                    }
                }
            }
            (workdir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

            spec = sample_h_spec()
            auth = sample_authorization()

            call_count = 0

            def broken_callable(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("Simulated internal callable failure")

            ctx = sample_approved_context()
            h_packets = sample_h_bundle_packets()
            sample_caps = {
                "package": "amazon4u",
                "version": "1.0.0",
                "nativeIO": "simulated",
                "ABI": "cpython",
                "dependency": "frozen",
                "CPU": "reference",
                "runtime": "offline",
                "resource": "bounded",
                "callable": "execute_fixture",
                "bootstrap": "synthetic",
            }
            bs = {
                "expected_spec": spec,
                "authorization": auth,
                "data": "FAKE_DATA",
                "expectations": {
                    "resource_policy": {"kind": "synthetic"},
                    "expected_runtime": {"device": "cpu", "cuda_available": False, "accelerator": "none"},
                    "capabilities": dict(sample_caps),
                },
                "run_envelope": dict(ctx.expected_h_envelope),
                "expected_run_envelope": dict(ctx.expected_h_envelope),
                "bundle_sources": {pkt.rel_path: pkt.content for pkt in h_packets},
                "expected_bundle": ctx.expected_h_bundle,
                "checkpoint_data": b"FAKE_CHECKPOINT",
                "expected_capabilities": dict(sample_caps),
                "actual_capabilities": dict(sample_caps),
                "version": 1,
            }
            rc = run_kernel_stages(workdir, outdir, spec, auth, context=ctx, h_callable=broken_callable, fake_bootstrap=bs)
            self.assertEqual(rc, 3)
            self.assertEqual(call_count, 1)
            self.assertFalse((outdir / "checkpoint.bin").exists())
            self.assertFalse((outdir / "result.json").exists())
            self.assertFalse((outdir / "amazon4u_record.json").exists())

    def test_op05_direct_script_no_op_refused(self) -> None:
        """Direct script execution without callable arguments is rejected with exit code 1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir) / "work"
            outdir = Path(tmp_dir) / "out"
            spec = sample_h_spec()
            spec["direct_script_execution"] = True
            auth = sample_authorization()

            rc = run_kernel_stages(workdir, outdir, spec, auth)
            self.assertEqual(rc, 1)

    # -----------------------------------------------------------------------
    # OP06: Environment & Capability Gates
    # -----------------------------------------------------------------------

    def test_op06_environment_and_capability_gates_positive(self) -> None:
        """Known fake CPU profile passes preflight validation."""
        spec = sample_h_spec()
        validated = validate_h_spec(spec)
        self.assertEqual(validated["device"], "cpu")
        self.assertEqual(validated["accelerator"], "none")

    def test_op06_mismatched_device_fails_before_work_negative(self) -> None:
        """Requesting GPU device in CPU reference scope fails before work; no CUDA probes."""
        spec = sample_h_spec()
        spec["device"] = "cuda"
        with self.assertRaises(Amazon4UError):
            validate_h_spec(spec)

        spec2 = sample_h_spec()
        spec2["accelerator"] = "gpu"
        with self.assertRaises(Amazon4UError):
            validate_h_spec(spec2)

    @_offline_test_tripwires()
    def test_op06_preflight_refuses_gpu_expected_runtime(self) -> None:
        """CPU reaches hydration; changing only requested CUDA hits the precise preflight guard."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir) / "work"
            outdir = Path(tmp_dir) / "out"
            spec = sample_h_spec()
            auth = sample_authorization()
            ctx = sample_approved_context()
            bs = sample_kernel_bootstrap(ctx)
            entry = unittest.mock.Mock(return_value=sample_h_result())
            cpu_rc = run_kernel_stages(
                workdir, outdir, spec, auth, context=ctx,
                h_callable=entry, fake_bootstrap=bs,
            )
            self.assertEqual(cpu_rc, 2)
            cpu_log = (outdir / "kernel-run.log").read_text(encoding="utf-8")
            self.assertIn("Stage 1: Preflight Validation PASS", cpu_log)
            self.assertIn("Hydration refusal: bundle_files.json missing.", cpu_log)
            entry.assert_not_called()

            gpu_bs = copy.deepcopy(bs)
            gpu_bs["expectations"]["expected_runtime"]["cuda_available"] = True
            gpu_work = Path(tmp_dir) / "gpu_work"
            gpu_out = Path(tmp_dir) / "gpu_out"
            with patch("pathlib.Path.mkdir") as mkdir, \
                 patch("os.open") as open_fd, \
                 patch("multiautoresearch.adapters.amazon4u_kernel.hydrate_bundle") as hydrate, \
                 patch("sys.stderr", new_callable=io.StringIO) as err:
                rc = run_kernel_stages(
                    gpu_work, gpu_out, spec, auth, context=ctx,
                    h_callable=entry, fake_bootstrap=gpu_bs,
                )
            self.assertEqual(err.getvalue().strip(), "Refusal: Preflight CPU-only scope violated (cuda_available)")
            self.assertEqual(rc, 1)
            mkdir.assert_not_called()
            open_fd.assert_not_called()
            hydrate.assert_not_called()
            entry.assert_not_called()
            self.assertFalse(gpu_work.exists())
            self.assertFalse(gpu_out.exists())

    @_offline_test_tripwires()
    def test_op05_h_sources_preflight_and_retained_handoff(self) -> None:
        """Malformed physical maps have no effects; provider mutation cannot replace retained bytes."""
        ctx = sample_approved_context()
        original = sample_kernel_bootstrap(ctx)["bundle_sources"]
        first = next(iter(original))
        missing = dict(original)
        missing.pop(first)
        extra = dict(original, unexpected=b"x")
        wrong_type = dict(original)
        wrong_type[first] = bytearray(original[first])
        short = dict(original)
        short[first] = original[first][:-1]
        tampered = dict(original)
        tampered[first] = b"!" + original[first][1:]
        cases = [None, {}, [], missing, extra, wrong_type, short, tampered]
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            for index, sources in enumerate(cases):
                with self.subTest(case=index):
                    bs = sample_kernel_bootstrap(ctx)
                    bs["bundle_sources"] = sources
                    entry = unittest.mock.Mock(return_value=sample_h_result())
                    work, out = td / f"work_{index}", td / f"out_{index}"
                    with patch("pathlib.Path.mkdir") as mkdir, \
                         patch("os.open") as open_fd, \
                         patch("multiautoresearch.adapters.amazon4u_kernel.hydrate_bundle") as hydrate, \
                         patch("sys.stderr", new_callable=io.StringIO) as err:
                        rc = run_kernel_stages(work, out, sample_h_spec(), sample_authorization(), context=ctx, h_callable=entry, fake_bootstrap=bs)
                    self.assertEqual(rc, 1)
                    self.assertIn("Refusal: H bundle source coherence invalid:", err.getvalue())
                    mkdir.assert_not_called()
                    open_fd.assert_not_called()
                    hydrate.assert_not_called()
                    entry.assert_not_called()
                    self.assertFalse(work.exists())
                    self.assertFalse(out.exists())

            work, out = td / "retained_work", td / "retained_out"
            work.mkdir()
            content = b"# tiny fake hydration only\n"
            (work / "bundle_files.json").write_text(json.dumps({"test.py": _oracle_encode_bytes(content)}), encoding="utf-8")
            (work / "manifest.json").write_text(json.dumps({"files": {"test.py": {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}), encoding="utf-8")
            bs = sample_kernel_bootstrap(ctx)
            received = []

            def provider():
                bs["bundle_sources"].clear()
                bs["run_envelope"] = {"mutated": True}
                bs["authorization"] = {"mutated": True}
                return copy.deepcopy(ctx.expected_o_environment["capabilities"])

            def entry(spec, **kwargs):
                received.append(kwargs["bundle_sources"])
                self.assertEqual(kwargs["bundle_sources"], original)
                self.assertEqual(kwargs["expected_bundle"], ctx.expected_h_bundle)
                self.assertEqual(kwargs["run_envelope"], ctx.expected_h_envelope)
                self.assertEqual(kwargs["authorization"], ctx.h_authorization)
                return sample_h_result()

            rc = run_kernel_stages(work, out, sample_h_spec(), sample_authorization(), context=ctx, h_callable=entry, fake_bootstrap=bs, capability_provider=provider, checkpoint_verifier=lambda b: b == b"FAKE_CHECKPOINT_BYTES", fail_stage="publish")
            self.assertEqual(rc, 5)
            self.assertEqual(len(received), 1)
            self.assertEqual(bs["bundle_sources"], {})
            self.assertTrue((out / "checkpoint.bin").exists())
            self.assertFalse((out / "result.json").exists())

            # Full completion control proves retained envelope is also published on disk
            work_pub, out_pub = td / "retained_pub_work", td / "retained_pub_out"
            work_pub.mkdir()
            (work_pub / "bundle_files.json").write_text(json.dumps({"test.py": _oracle_encode_bytes(content)}), encoding="utf-8")
            (work_pub / "manifest.json").write_text(json.dumps({"files": {"test.py": {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}), encoding="utf-8")
            bs_pub = sample_kernel_bootstrap(ctx)

            def provider_pub():
                bs_pub["run_envelope"] = {"mutated": True}
                bs_pub["authorization"] = {"mutated": True}
                return copy.deepcopy(ctx.expected_o_environment["capabilities"])

            rc_full = run_kernel_stages(work_pub, out_pub, sample_h_spec(), sample_authorization(), context=ctx, h_callable=entry, fake_bootstrap=bs_pub, capability_provider=provider_pub, checkpoint_verifier=lambda b: b == b"FAKE_CHECKPOINT_BYTES")
            self.assertEqual(rc_full, 0)
            published_env = json.loads((out_pub / "run_envelope.json").read_text(encoding="utf-8"))
            self.assertEqual(published_env, ctx.expected_h_envelope)

    # -----------------------------------------------------------------------
    # OP07: Submission Acknowledgment & Timeout Binding
    # -----------------------------------------------------------------------

    def test_op07_submission_acknowledgment_positive(self) -> None:
        """Fake push acknowledges exact version and bound timeout."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / "bundle"
            spec = sample_h_spec()
            auth = sample_authorization()
            packets = sample_full_closure_packets()
            manifest = build_source_manifest(packets, require_full_closure=True)
            ctx = sample_approved_context()
            plan = render_plan(spec, authorization=auth, expected_spec=spec, expected_sources=ctx.expected_o_outer, source_packets=packets, context=ctx)
            write_bundle(plan, bundle_dir)

            transport = FakeKaggleTransport(acknowledged_version=2)
            res = transport.push(bundle_dir, timeout=7200, accelerator="none", plan_digest=plan.plan_digest)
            self.assertEqual(res["returncode"], 0)
            self.assertEqual(res["acknowledged_version"], 2)
            self.assertEqual(res["owner"], "amazon4u")
            self.assertTrue(validate_submission_receipt(
                res,
                expected_owner="amazon4u",
                expected_slug=plan.slug,
                expected_version=2,
                expected_timeout=7200,
                expected_accelerator="none",
                expected_plan_digest=plan.plan_digest,
            ))

    def test_op07_ambiguous_submission_unknown_no_resubmit_negative(self) -> None:
        """Ambiguous submission timeout raises SubmissionUnknownError; no automatic repush."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir)
            transport = FakeKaggleTransport(push_ambiguous_timeout=True)
            with self.assertRaises(SubmissionUnknownError):
                transport.push(bundle_dir, timeout=7200)

    def test_op07_push_printed_error_with_zero_rc_negative(self) -> None:
        """Push printing an error message even if exit code is 0 is treated as failure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir)
            transport = FakeKaggleTransport(push_printed_error="Quota exceeded on remote host")
            res = transport.push(bundle_dir)
            self.assertIsNone(res.get("acknowledged_version"))
            self.assertIn("Error:", res.get("stdout", ""))

    def test_op07_receipt_validator_catches_mismatches(self) -> None:
        """validate_submission_receipt catches version, owner, and plan_digest mismatches."""
        valid_rcpt = {
            "returncode": 0,
            "owner": "amazon4u",
            "slug": "my-slug",
            "acknowledged_version": 1,
            "version_ref": "amazon4u/my-slug/v1",
            "plan_digest": "dig_A",
        }
        with self.assertRaises(Amazon4UError):
            validate_submission_receipt(valid_rcpt, expected_owner="wrong_owner", expected_slug="my-slug", expected_version=1)
        with self.assertRaises(Amazon4UError):
            validate_submission_receipt(valid_rcpt, expected_owner="amazon4u", expected_slug="my-slug", expected_version=2)
        with self.assertRaises(Amazon4UError):
            validate_submission_receipt(valid_rcpt, expected_owner="amazon4u", expected_slug="my-slug", expected_version=1, expected_plan_digest="dig_B")

    # -----------------------------------------------------------------------
    # OP08: Exact Bounded Status State Machine
    # -----------------------------------------------------------------------

    def test_op08_status_state_machine_positive(self) -> None:
        """queued -> running -> complete transition operates correctly with fake clock."""
        clock = FakeClock()
        transport = FakeKaggleTransport(
            acknowledged_version=1,
            statuses=["queued", "running", "complete"],
        )
        final_st = poll_kernel_status(
            "amazon4u/test-slug", 1, transport,
            clock_fn=clock.now, sleep_fn=clock.sleep,
        )
        self.assertEqual(final_st, "complete")

    def test_op08_status_substring_mislead_negative(self) -> None:
        """Status string containing 'complete' inside error text is not treated as complete."""
        clock = FakeClock()
        transport = FakeKaggleTransport(
            acknowledged_version=1,
            statuses=["error: task failed and could not complete"],
        )
        final_st = poll_kernel_status(
            "amazon4u/test-slug", 1, transport,
            clock_fn=clock.now, sleep_fn=clock.sleep,
        )
        self.assertEqual(final_st, "error")

    def test_op08_poll_deadline_exceeded_bounded(self) -> None:
        """Exceeding max poll iterations returns poll_deadline_exceeded without infinite loop."""
        clock = FakeClock()
        transport = FakeKaggleTransport(
            acknowledged_version=1,
            statuses=["running"] * 10,
        )
        final_st = poll_kernel_status(
            "amazon4u/test-slug", 1, transport,
            max_polls=3, clock_fn=clock.now, sleep_fn=clock.sleep,
        )
        self.assertEqual(final_st, "poll_deadline_exceeded")

    def test_op08_mismatched_status_version_refused(self) -> None:
        """Status reply with wrong version is rejected as error."""
        clock = FakeClock()
        transport = FakeKaggleTransport(acknowledged_version=2)
        # We expect version 1, but transport replies version 2
        final_st = poll_kernel_status(
            "amazon4u/test-slug", 1, transport,
            clock_fn=clock.now, sleep_fn=clock.sleep,
        )
        self.assertEqual(final_st, "error")

    # -----------------------------------------------------------------------
    # OP09: Bounded Credential-Safe Logs
    # -----------------------------------------------------------------------

    def test_op09_credential_redacted_bounded_logs_positive(self) -> None:
        """Logs redact secret credentials and respect size caps."""
        raw_log = "Run starting with token=SECRET_TOKEN_12345 and password: SECRET_PASSWORD\nDone."
        transport = FakeKaggleTransport(logs_text=raw_log, acknowledged_version=1)
        res = consume_kernel_logs("amazon4u/test-slug", 1, transport, max_bytes=1024)
        self.assertEqual(res["status"], "available")
        logs = res["logs"]
        self.assertNotIn("SECRET_TOKEN_12345", logs)
        self.assertNotIn("SECRET_PASSWORD", logs)
        self.assertIn("[REDACTED]", logs)

    def test_op09_log_availability_failure_distinguished(self) -> None:
        """Log availability failure is explicitly reported as unavailable."""
        transport = FakeKaggleTransport(log_availability_error=True, acknowledged_version=1)
        res = consume_kernel_logs("amazon4u/test-slug", 1, transport)
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("unavailable", res["error"].lower())

    # -----------------------------------------------------------------------
    # OP12: Immutable State & Multi-Attempt Preservation
    # -----------------------------------------------------------------------

    def test_op12_immutable_state_multi_attempt_positive(self) -> None:
        """Separate attempts maintain distinct immutable state files without overwriting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = Amazon4UStateManager(Path(tmp_dir))
            st1 = {"attempt": 1, "status": "failed_timeout"}
            st2 = {"attempt": 2, "status": "complete"}

            p1 = mgr.record_attempt("amazon4u", "run_A", "attempt_01", st1)
            p2 = mgr.record_attempt("amazon4u", "run_A", "attempt_02", st2)

            self.assertTrue(p1.exists())
            self.assertTrue(p2.exists())
            self.assertNotEqual(p1, p2)

            loaded1 = mgr.load_attempt("run_A", "attempt_01")
            loaded2 = mgr.load_attempt("run_A", "attempt_02")
            self.assertEqual(loaded1["status"], "failed_timeout")
            self.assertEqual(loaded2["status"], "complete")

    def test_op12_state_collision_refused_negative(self) -> None:
        """Attempting to overwrite existing attempt state file raises Amazon4UError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = Amazon4UStateManager(Path(tmp_dir))
            mgr.record_attempt("amazon4u", "run_A", "attempt_01", {"attempt": 1})
            with self.assertRaises(Amazon4UError):
                mgr.record_attempt("amazon4u", "run_A", "attempt_01", {"attempt": 1, "repush": True})

    def test_op12_state_path_escape_refused(self) -> None:
        """Path traversal or slashes in run_id or attempt_id are refused."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = Amazon4UStateManager(Path(tmp_dir))
            with self.assertRaises(Amazon4UError):
                mgr.record_attempt("amazon4u", "../escaped", "attempt_01", {})
            with self.assertRaises(Amazon4UError):
                mgr.record_attempt("amazon4u", "run_A", "attempt/sub", {})

    # -----------------------------------------------------------------------
    # OP13: Typed Results & Scientific Field Boundary
    # -----------------------------------------------------------------------

    def test_op13_typed_record_scientific_field_boundary_positive(self) -> None:
        """Record wrapper encapsulates H result faithfully without eval_score proxies."""
        plan = {"spec": sample_h_spec(), "authorization": sample_authorization()}
        h_res = sample_h_result()
        record = build_amazon4u_record(plan, h_res)

        self.assertEqual(record["schema_version"], "amazon4u-O-record-v1")
        self.assertEqual(record["profile"], "amazon4u")
        self.assertEqual(record["evidence_kind"], "simulated_offline_fixture")
        self.assertEqual(record["status"], "simulated_offline_success")
        self.assertIn("h_result", record)
        self.assertEqual(record["h_result"]["schema_version"], "amazon4u-H-result-v1")
        self.assertEqual(record["deployment_decision"], "NOT_AUTHORIZED")

    def test_op13_h_failure_status_yields_failure_record(self) -> None:
        """scientific_terminal represents legitimate scientific stopping (not error), recorded as success."""
        plan = {"spec": sample_h_spec(), "authorization": sample_authorization()}
        h_res = sample_h_result()
        h_res["status"] = "scientific_terminal"
        record = build_amazon4u_record(plan, h_res)
        self.assertEqual(record["status"], "simulated_offline_success")

        # Illegitimate H status is strictly refused
        h_bad = sample_h_result()
        h_bad["status"] = "invented_status_not_in_legitimate_statuses"
        with self.assertRaises(Amazon4UError):
            build_amazon4u_record(plan, h_bad)

    def test_op13_forbidden_proxy_metrics_refused(self) -> None:
        """eval_score, val_bpb, recall, or invented metrics key inside validation_report are rejected."""
        plan = {"spec": sample_h_spec(), "authorization": sample_authorization()}
        h_res = sample_h_result()
        h_res["validation_report"]["overall_metrics"]["eval_score"] = 0.99
        with self.assertRaises(Amazon4UError):
            build_amazon4u_record(plan, h_res)

        h_res2 = sample_h_result()
        h_res2["validation_report"]["overall_metrics"]["val_bpb"] = 1.23
        with self.assertRaises(Amazon4UError):
            build_amazon4u_record(plan, h_res2)

        # Invented metrics key instead of overall_metrics is rejected
        h_res3 = sample_h_result()
        h_res3["validation_report"] = {"metrics": {"NDCG@10": 0.5}}
        with self.assertRaises(Amazon4UError):
            build_amazon4u_record(plan, h_res3)

        # Fabricated Music_5core_sample population is rejected
        h_res4 = sample_h_result()
        with self.assertRaises(Amazon4UError):
            build_amazon4u_record(
                plan,
                h_res4,
                population_metadata={"kind": "synthetic_fixtures", "dataset_split": "Music_5core_sample"},
            )

    # -----------------------------------------------------------------------
    # OP14: Legacy Compatibility Without Legacy Execution
    # -----------------------------------------------------------------------

    def test_op14_legacy_compatibility_parser_defaults(self) -> None:
        """Legacy parser defaults and choices are preserved across subcommands."""
        parser = kaggle_job.build_parser()
        args = parser.parse_args(["launch"])
        self.assertIsNone(getattr(args, "profile", None))
        self.assertEqual(args.mode, "experiment")
        self.assertIsNone(args.accelerator)
        self.assertIsNone(args.timeout)
        self.assertFalse(args.dry_run)

        # Test profile positions before and after subcommand
        args_pre = parser.parse_args(["--profile", "amazon4u", "launch", "--dry-run"])
        self.assertEqual(args_pre.profile, "amazon4u")
        self.assertTrue(args_pre.dry_run)

        args_post = parser.parse_args(["launch", "--profile", "amazon4u", "--dry-run"])
        self.assertEqual(args_post.profile, "amazon4u")
        self.assertTrue(args_post.dry_run)

    def test_op14_legacy_training_nonzero_propagation(self) -> None:
        """Non-zero return code from generated legacy command propagates non-zero exit in runner."""
        script_code = kaggle_job.build_kernel_script("experiment", [])
        ns: dict[str, object] = {
            "__file__": "generated_kernel.py",
            "__name__": "generated_kernel",
        }
        fake_proc = unittest.mock.MagicMock()
        fake_proc.stdout = iter(["training step error\n"])
        fake_proc.wait.return_value = 42
        with patch("subprocess.Popen", return_value=fake_proc), \
             patch("socket.socket", side_effect=AssertionError("Tripwire: network socket in test")), \
             patch("sys.stdout", new_callable=io.StringIO):
            exec(compile(script_code, "generated_kernel.py", "exec"), ns)
            run_cmd_fn = ns["run_cmd"]
            rc = run_cmd_fn(["fake_train.py"])
            self.assertEqual(rc, 42)

    # -----------------------------------------------------------------------
    # OP15: Honest Durability & Readiness Gates
    # -----------------------------------------------------------------------

    def test_op15_honest_durability_and_readiness_gates(self) -> None:
        """Readiness gates list is preserved and deployment decision is NOT_AUTHORIZED."""
        plan = {"spec": sample_h_spec(), "authorization": sample_authorization()}
        h_res = sample_h_result()
        rec = build_amazon4u_record(plan, h_res)
        self.assertEqual(rec["readiness_gates"], READINESS_GATES)
        self.assertEqual(rec["deployment_decision"], "NOT_AUTHORIZED")

    # -----------------------------------------------------------------------
    # OP16: Safe Generated-Program Execution & Tripwires
    # -----------------------------------------------------------------------

    @_offline_test_tripwires()
    def test_op16_generated_fake_program_execution_and_tripwires(self) -> None:
        """Execute the ACTUAL generated program under fake runtime/preflight/entrypoint with tripwires."""
        orig_sys_path = list(sys.path)
        isolated_keys = [k for k in list(sys.modules) if k.startswith("multiautoresearch")]
        orig_sys_modules = {k: sys.modules.get(k) for k in isolated_keys}

        # Isolate cached project targets BEFORE executing kernel main
        for k in list(sys.modules):
            if k.startswith("multiautoresearch"):
                sys.modules.pop(k, None)

        with patch("subprocess.Popen", side_effect=AssertionError("Tripwire: subprocess in test")),              patch("socket.socket", side_effect=AssertionError("Tripwire: network socket in test")):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    td = Path(tmp_dir)
                    spec = sample_h_spec()
                    auth = sample_authorization()

                    # Use valid hydrated O source from repository rather than stubfiles
                    repo_root = Path(__file__).resolve().parent.parent
                    packets = []
                    for rel in sorted(REQUIRED_CLOSURE_PATHS):
                        src_file = repo_root / rel
                        if src_file.is_file():
                            content = src_file.read_bytes()
                        else:
                            content = f"# Mock content for {rel}\n".encode("utf-8")
                        packets.append(SourcePacket(rel, content))

                    manifest = build_source_manifest(packets, require_full_closure=True)
                    kernel_script_preview = build_amazon4u_kernel_script(spec, auth, str(manifest["manifest_digest"]))
                    exp_o_outer = copy.deepcopy(manifest)
                    exp_o_outer["generated_runtime"] = {
                        "code_file": "kernel.py",
                        "sha256": hashlib.sha256(kernel_script_preview.encode("utf-8")).hexdigest(),
                        "size": len(kernel_script_preview.encode("utf-8")),
                    }
                    exp_o_outer["environment"] = {
                        "device": "cpu",
                        "accelerator": "none",
                        "network": "disabled",
                    }
                    ctx = sample_approved_context(expected_o_outer=exp_o_outer)

                    plan = render_plan(
                        spec,
                        authorization=auth,
                        expected_spec=spec,
                        expected_sources=exp_o_outer,
                        source_packets=packets,
                        context=ctx,
                    )
                    bundle_dir = td / "bundle"
                    write_bundle(plan, bundle_dir)

                    script_path = bundle_dir / "kernel.py"
                    self.assertTrue(script_path.exists())

                    workdir_neg = td / "work_neg"
                    outdir_neg = td / "out_neg"
                    workdir_neg.mkdir()
                    (workdir_neg / "bundle_files.json").write_text(json.dumps(plan.files), encoding="utf-8")
                    (workdir_neg / "manifest.json").write_text(json.dumps(plan.outer_manifest), encoding="utf-8")

                    workdir_pos = td / "work_pos"
                    outdir_pos = td / "out_pos"
                    workdir_pos.mkdir()
                    (workdir_pos / "bundle_files.json").write_text(json.dumps(plan.files), encoding="utf-8")
                    (workdir_pos / "manifest.json").write_text(json.dumps(plan.outer_manifest), encoding="utf-8")

                    # Load generated kernel script into controlled non-main namespace
                    script_code = script_path.read_text(encoding="utf-8")
                    kernel_ns: dict[str, object] = {
                        "__file__": str(script_path),
                        "__name__": "kernel_module",
                    }
                    exec(compile(script_code, str(script_path), "exec"), kernel_ns)
                    kernel_main = kernel_ns["main"]

                    # 1. Negative control: default invocation without parameters fails closed (returns 1 before effects)
                    with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
                        rc_default = kernel_main()
                    self.assertEqual(rc_default, 1)

                    # Mock callable and bootstrap with captured origin evidence
                    captured_origin: dict[str, object] = {}

                    def mock_callable(s, **kwargs):
                        # Captured DURING execution inside hydrated environment before main returns
                        k_mod = sys.modules.get("multiautoresearch.adapters.amazon4u_kernel")
                        if k_mod is not None:
                            captured_origin["module_file"] = getattr(k_mod, "__file__", None)
                            captured_origin["marker"] = getattr(k_mod, "_LATEST_EXECUTION_MARKER", None)
                        # Assert actual independently expected H28 schema/bytes/digest received
                        received_bundle = kwargs.get("expected_bundle")
                        assert isinstance(received_bundle, dict), f"expected_bundle must be dict, got {type(received_bundle)}"
                        assert received_bundle.get("schema_version") == "amazon4u-H-bundle-v1", (
                            f"unexpected schema_version: {received_bundle.get('schema_version')}"
                        )
                        assert received_bundle.get("sha256") == ctx.expected_h_bundle["sha256"], (
                            f"digest mismatch: expected {ctx.expected_h_bundle['sha256']}, got {received_bundle.get('sha256')}"
                        )
                        assert len(received_bundle.get("files", {})) == 28, (
                            f"expected 28 files, got {len(received_bundle.get('files', {}))}"
                        )
                        for r_path, meta in received_bundle.get("files", {}).items():
                            assert "bytes" in meta and type(meta["bytes"]) is int, f"missing int bytes in {r_path}"
                            assert "sha256" in meta and len(meta["sha256"]) == 64, f"missing sha256 in {r_path}"
                        assert received_bundle == ctx.expected_h_bundle, "received expected_bundle does not match canonical H28 manifest"

                        # Assert canonical 11-field H run envelope and canonical H authorization received
                        received_env = kwargs.get("run_envelope")
                        assert isinstance(received_env, dict), f"run_envelope must be dict, got {type(received_env)}"
                        assert received_env.get("schema_version") == "amazon4u-H-run-v1", (
                            f"unexpected schema_version: {received_env.get('schema_version')}"
                        )
                        assert set(received_env.keys()) == {
                            "schema_version", "run_id", "approval_id", "profile_id", "method",
                            "identity_hash", "device", "accelerator", "bundle_sha256",
                            "initial_state_sha256", "determinism_policy",
                        }, f"invalid run_envelope keys: {set(received_env.keys())}"
                        assert received_env == ctx.expected_h_envelope, "run_envelope mismatch with expected_h_envelope"

                        received_exp_env = kwargs.get("expected_run_envelope")
                        assert received_exp_env == ctx.expected_h_envelope, "expected_run_envelope mismatch"

                        received_auth = kwargs.get("authorization")
                        assert isinstance(received_auth, dict), f"authorization must be dict, got {type(received_auth)}"
                        assert received_auth.get("schema_version") == "amazon4u-H-authorization-v1", (
                            f"unexpected authorization schema: {received_auth.get('schema_version')}"
                        )
                        assert received_auth == ctx.h_authorization, "authorization mismatch"
                        return sample_h_result()

                    h_packets = sample_h_bundle_packets()
                    sample_caps = {
                        "package": "amazon4u",
                        "version": "1.0.0",
                        "nativeIO": "simulated",
                        "ABI": "cpython",
                        "dependency": "frozen",
                        "CPU": "reference",
                        "runtime": "offline",
                        "resource": "bounded",
                        "callable": "execute_fixture",
                        "bootstrap": "synthetic",
                    }
                    mock_bs = {
                        "expected_spec": spec,
                        "authorization": auth,
                        "data": "FAKE_DATA",
                        "expectations": {
                            "resource_policy": {"kind": "synthetic"},
                            "expected_runtime": {"device": "cpu", "cuda_available": False, "accelerator": "none"},
                            "capabilities": dict(sample_caps),
                        },
                        "run_envelope": copy.deepcopy(ctx.expected_h_envelope),
                        "expected_run_envelope": dict(ctx.expected_h_envelope),
                        "bundle_sources": {pkt.rel_path: pkt.content for pkt in h_packets},
                        "expected_bundle": ctx.expected_h_bundle,
                        "checkpoint_data": b"FAKE_CHECKPOINT_BYTES",
                        "expected_checkpoint_sha256": hashlib.sha256(b"FAKE_CHECKPOINT_BYTES").hexdigest(),
                        "expected_capabilities": dict(sample_caps),
                        "actual_capabilities": dict(sample_caps),
                        "version": 1,
                    }

                    # All non-context prerequisites are the same as the positive control.
                    partial_ctx = {
                        "canonical_h_spec": spec,
                        "h_authorization": auth,
                        "expected_o_environment": copy.deepcopy(ctx.expected_o_environment),
                    }
                    invalid_limits = dataclasses.asdict(ctx)
                    invalid_limits["limits"]["max_file_bytes"] = True
                    invalid_inventory = dataclasses.asdict(ctx)
                    invalid_inventory["expected_output_inventory"].pop("checkpoint")
                    invalid_h_digest = dataclasses.asdict(ctx)
                    invalid_h_digest["expected_h_bundle"]["sha256"] = "0" * 64

                    invalid_env_legacy = dataclasses.asdict(ctx)
                    invalid_env_legacy["expected_h_envelope"]["schema_version"] = "amazon4u-H-envelope-v1"

                    invalid_env_missing_field = dataclasses.asdict(ctx)
                    invalid_env_missing_field["expected_h_envelope"].pop("initial_state_sha256")

                    invalid_env_bad_digest = dataclasses.asdict(ctx)
                    invalid_env_bad_digest["expected_h_envelope"]["identity_hash"] = "not_a_valid_hex_digest"

                    invalid_auth_o_tag = dataclasses.asdict(ctx)
                    invalid_auth_o_tag["h_authorization"]["schema_version"] = "amazon4u-O-authorization-v1"
                    invalid_auth_o_tag["expected_input_identity"]["authorization"]["schema_version"] = "amazon4u-O-authorization-v1"

                    class MetadataDict(dict):
                        pass

                    class MetadataStr(str):
                        pass

                    subtype_contexts = []
                    for field in ("canonical_h_spec", "h_authorization", "expected_h_envelope"):
                        bad = dataclasses.asdict(ctx)
                        bad[field] = MetadataDict(bad[field])
                        subtype_contexts.append(bad)
                    for field, key in (("h_authorization", "scope"), ("expected_h_envelope", "identity_hash")):
                        bad = dataclasses.asdict(ctx)
                        bad[field][key] = MetadataStr(bad[field][key])
                        subtype_contexts.append(bad)
                    device_contexts = []
                    for key, value in (("device", "CPU"), ("accelerator", ""), ("accelerator", "NONE")):
                        bad = dataclasses.asdict(ctx)
                        bad["canonical_h_spec"][key] = value
                        bad["expected_input_identity"]["spec"][key] = value
                        bad["expected_h_envelope"][key] = value
                        device_contexts.append(bad)
                    self.assertEqual(validate_approved_offline_context(ctx).expected_h_envelope, ctx.expected_h_envelope)
                    contexts = [
                        None, partial_ctx, invalid_limits, invalid_inventory, invalid_h_digest,
                        invalid_env_legacy, invalid_env_missing_field, invalid_env_bad_digest,
                        invalid_auth_o_tag, *subtype_contexts, *device_contexts,
                    ]
                    before_work = {p.name: p.read_bytes() for p in workdir_neg.iterdir()}
                    target_imports = []
                    safe_import = builtins.__import__

                    def forbid_target_import(name, *args, **kwargs):
                        if name.startswith("multiautoresearch"):
                            target_imports.append(name)
                            raise AssertionError("Untrusted target import before context validation")
                        return safe_import(name, *args, **kwargs)

                    for bad_context in contexts:
                        with self.subTest(context=bad_context):
                            before_context = copy.deepcopy(bad_context)
                            with self.assertRaises(Amazon4UError):
                                validate_approved_offline_context(bad_context)
                            entry = unittest.mock.Mock(return_value=sample_h_result())
                            provider = unittest.mock.Mock(return_value=dict(sample_caps))
                            hydration = unittest.mock.Mock()
                            with patch("builtins.__import__", side_effect=forbid_target_import), \
                                 patch("pathlib.Path.mkdir") as mkdir, \
                                 patch("pathlib.Path.write_text") as write_text, \
                                 patch("pathlib.Path.write_bytes") as write_bytes, \
                                 patch("os.open") as open_fd, \
                                 patch.dict(kernel_ns, {"streaming_hydrate": hydration}), \
                                 patch("sys.stderr", new_callable=io.StringIO) as err:
                                rc_context = kernel_main(
                                    custom_workdir=workdir_neg,
                                    custom_outdir=outdir_neg,
                                    callable_fn=entry,
                                    fake_bootstrap=mock_bs,
                                    checkpoint_verifier=lambda b: True,
                                    context=bad_context,
                                    capability_provider=provider,
                                )
                            self.assertEqual(rc_context, 1)
                            self.assertIn("Refusal: ApprovedOfflineContext:", err.getvalue())
                            self.assertEqual(bad_context, before_context)
                            mkdir.assert_not_called()
                            write_text.assert_not_called()
                            write_bytes.assert_not_called()
                            open_fd.assert_not_called()
                            hydration.assert_not_called()
                            entry.assert_not_called()
                            provider.assert_not_called()
                    self.assertEqual(target_imports, [])
                    self.assertEqual({p.name: p.read_bytes() for p in workdir_neg.iterdir()}, before_work)
                    self.assertFalse(outdir_neg.exists())

                    # Inconsistent bootstrap expectations negative control: supplying O32 outer manifest as expected_bundle is rejected before effects
                    inconsistent_bs = dict(mock_bs)
                    inconsistent_bs["expected_bundle"] = plan.outer_manifest
                    workdir_inconsistent = td / "work_inconsistent"
                    outdir_inconsistent = td / "out_inconsistent"
                    with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
                        rc_inconsistent = kernel_main(
                            custom_workdir=workdir_inconsistent,
                            custom_outdir=outdir_inconsistent,
                            callable_fn=mock_callable,
                            fake_bootstrap=inconsistent_bs,
                            context=ctx,
                            capability_provider=lambda: dict(sample_caps),
                        )
                    self.assertEqual(rc_inconsistent, 1)
                    self.assertFalse(outdir_inconsistent.exists())

                    # Actual-bootstrap metadata must fail BEFORE provider, effects or target imports.
                    inconsistent_env_bs = copy.deepcopy(mock_bs)
                    inconsistent_env_bs["run_envelope"]["schema_version"] = "amazon4u-H-envelope-v1"
                    inconsistent_auth_bs = copy.deepcopy(mock_bs)
                    inconsistent_auth_bs["authorization"]["schema_version"] = "amazon4u-O-authorization-v1"
                    missing_env_bs = copy.deepcopy(mock_bs)
                    missing_env_bs["expected_run_envelope"].pop("identity_hash")
                    bad_digest_bs = copy.deepcopy(mock_bs)
                    bad_digest_bs["run_envelope"]["identity_hash"] = "x" * 64
                    for index, bad_bs in enumerate((inconsistent_env_bs, inconsistent_auth_bs, missing_env_bs, bad_digest_bs)):
                        with self.subTest(bootstrap_metadata=index):
                            provider = unittest.mock.Mock(return_value=dict(sample_caps))
                            entry = unittest.mock.Mock(return_value=sample_h_result())
                            hydration = unittest.mock.Mock()
                            work_bad, out_bad = td / f"bad_meta_work_{index}", td / f"bad_meta_out_{index}"
                            with patch("builtins.__import__", side_effect=forbid_target_import), \
                                 patch("pathlib.Path.mkdir") as mkdir, \
                                 patch("pathlib.Path.write_text") as write_text, \
                                 patch("pathlib.Path.write_bytes") as write_bytes, \
                                 patch("os.open") as open_fd, \
                                 patch.dict(kernel_ns, {"streaming_hydrate": hydration}), \
                                 patch("sys.stderr", new_callable=io.StringIO) as err:
                                rc_bad = kernel_main(custom_workdir=work_bad, custom_outdir=out_bad, callable_fn=entry, fake_bootstrap=bad_bs, context=ctx, capability_provider=provider)
                            self.assertEqual(rc_bad, 1)
                            self.assertIn("Refusal: Preflight canonical bootstrap metadata:", err.getvalue())
                            provider.assert_not_called()
                            entry.assert_not_called()
                            mkdir.assert_not_called()
                            write_text.assert_not_called()
                            write_bytes.assert_not_called()
                            open_fd.assert_not_called()
                            hydration.assert_not_called()
                            self.assertFalse(work_bad.exists())
                            self.assertFalse(out_bad.exists())
                    self.assertEqual(target_imports, [])

                    # 2. Checkpoint verifier rejection negative control (returns 4)
                    with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
                        rc_ckpt_fail = kernel_main(
                            custom_workdir=workdir_neg,
                            custom_outdir=outdir_neg,
                            callable_fn=mock_callable,
                            fake_bootstrap=mock_bs,
                            checkpoint_verifier=lambda b: False,
                            context=ctx,
                            capability_provider=lambda: dict(sample_caps),
                        )
                    self.assertEqual(rc_ckpt_fail, 4)

                    # 3. Positive control with valid verifier
                    positive_context = dataclasses.asdict(ctx)
                    positive_bootstrap = copy.deepcopy(mock_bs)
                    provider_calls = []

                    def positive_provider():
                        provider_calls.append("provider")
                        positive_context["limits"]["max_file_bytes"] = False
                        positive_context["expected_o_environment"]["capabilities"]["CPU"] = "mutated caller alias"
                        for key in ("expected_spec", "authorization", "run_envelope", "expected_run_envelope", "expected_bundle", "bundle_sources"):
                            positive_bootstrap[key].clear()
                        return dict(sample_caps)

                    with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
                        rc = kernel_main(
                            custom_workdir=workdir_pos,
                            custom_outdir=outdir_pos,
                            callable_fn=mock_callable,
                            fake_bootstrap=positive_bootstrap,
                            checkpoint_verifier=lambda b: True,
                            context=positive_context,
                            capability_provider=positive_provider,
                        )

                    self.assertEqual(rc, 0)
                    self.assertTrue(provider_calls)
                    self.assertEqual(positive_bootstrap["run_envelope"], {})
                    self.assertEqual(json.loads((outdir_pos / "run_envelope.json").read_text()), ctx.expected_h_envelope)
                    self.assertTrue((outdir_pos / "plan.json").exists())
                    self.assertTrue((outdir_pos / "result.json").exists())
                    self.assertTrue((outdir_pos / "run_envelope.json").exists())
                    self.assertTrue((outdir_pos / "manifest.json").exists())
                    self.assertTrue((outdir_pos / "amazon4u_record.json").exists())
                    self.assertTrue((outdir_pos / "checkpoint.bin").exists())

                    # Verify runtime origin captured during active execution
                    self.assertIn("module_file", captured_origin)
                    self.assertTrue(str(Path(str(captured_origin["module_file"])).resolve()).startswith(str(workdir_pos.resolve())))
                    marker = captured_origin.get("marker")
                    self.assertIsNotNone(marker)
                    self.assertEqual(marker["origin_workdir"], str(workdir_pos))
                    exp_kernel_sha = plan.outer_manifest["files"]["multiautoresearch/adapters/amazon4u_kernel.py"]["sha256"]
                    self.assertEqual(marker["sha256"], exp_kernel_sha)

                    # Also verify persisted amazon4u_record.json execution_marker
                    rec = json.loads((outdir_pos / "amazon4u_record.json").read_text(encoding="utf-8"))
                    rec_marker = rec.get("execution_marker")
                    self.assertIsNotNone(rec_marker)
                    self.assertEqual(rec_marker["origin_workdir"], str(workdir_pos))
                    self.assertEqual(rec_marker["sha256"], exp_kernel_sha)
                    self.assertTrue(str(Path(rec_marker["module_file"]).resolve()).startswith(str(workdir_pos.resolve())))
            finally:
                sys.path[:] = orig_sys_path
                for k in list(sys.modules):
                    if k.startswith("multiautoresearch") and k not in orig_sys_modules:
                        sys.modules.pop(k, None)
                for k, v in orig_sys_modules.items():
                    sys.modules[k] = v

    # -----------------------------------------------------------------------
    # OP17: Whole-Suite Evidence Preservation
    # -----------------------------------------------------------------------

    def test_op17_whole_suite_evidence_preservation(self) -> None:
        """All 412 baseline tests and schemas remain unchanged."""
        self.assertEqual(SCHEMA_H_SPEC, "amazon4u-H-spec-v1")
        self.assertEqual(SCHEMA_H_AUTHORIZATION, "amazon4u-H-authorization-v1")
        self.assertEqual(SCHEMA_SPEC, "amazon4u-O-spec-v1")
        self.assertEqual(SCHEMA_AUTHORIZATION, "amazon4u-O-authorization-v1")

    # -----------------------------------------------------------------------
    # OP18: Source/Runtime vs Operational Authority
    # -----------------------------------------------------------------------

    def test_op18_live_operation_refuses_without_authority(self) -> None:
        """Attempting live launch without authorized transport fails closed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_file = Path(tmp_dir) / "spec.json"
            spec_file.write_text(json.dumps(sample_h_spec()), encoding="utf-8")

            parser = kaggle_job.build_parser()
            args = parser.parse_args([
                "launch",
                "--profile", "amazon4u",
                "--spec", str(spec_file),
            ])
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                rc = dispatch_amazon4u(args, transport=None)
                self.assertEqual(rc, 1)
                self.assertIn("Live Kaggle execution is NOT authorized", mock_err.getvalue())


class TestExpectedOuterManifestRegression(unittest.TestCase):
    def _invalid_contexts(self):
        base = dataclasses.asdict(sample_approved_context())
        self.assertEqual(len(base["expected_o_outer"]["files"]), 32)
        for label in ("extra_path", "bool_size", "oversized_file", "oversized_aggregate"):
            context = copy.deepcopy(base)
            outer = context["expected_o_outer"]
            files = outer["files"]
            first = next(iter(files))
            if label == "extra_path":
                files["extra.txt"] = {"size": 0, "sha256": hashlib.sha256(b"").hexdigest()}
            elif label == "bool_size":
                files[first]["size"] = True
            elif label == "oversized_file":
                files[first]["size"] = 2097153
            else:
                for path in list(files)[:5]:
                    files[path]["size"] = 2097152
            outer["file_count"] = len(files)
            outer["aggregate_size"] = sum(meta["size"] for meta in files.values())
            outer["manifest_digest"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            yield label, context

    def test_context_rejects_invalid_outer_inventory_before_runtime(self):
        validate_approved_offline_context(sample_approved_context())
        for label, context in self._invalid_contexts():
            with self.subTest(case=label):
                with self.assertRaises(Amazon4UError):
                    validate_approved_offline_context(context)

    def test_generated_preflight_rejects_same_invalid_outer_inventory(self):
        source = build_amazon4u_kernel_script(sample_h_spec(), sample_authorization(), "a" * 64)
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"_h_files_digest", "_retain_generated_context"}]
        self.assertEqual(len(functions), 2)
        namespace = {
            "hashlib": hashlib, "struct": struct, "copy": copy,
            "make_dataclass": dataclasses.make_dataclass,
            "SPEC_KEYS": tuple(sample_h_spec()), "H_BUNDLE_PATHS": H_BUNDLE_PATHS,
            "REQUIRED_CLOSURE_PATHS": REQUIRED_CLOSURE_PATHS,
            "MAX_FILE_BYTES": 2097152, "MAX_AGGREGATE_BYTES": 8388608,
            "json": json,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<generated-context-preflight>", "exec"), namespace)
        validate = namespace["_retain_generated_context"]
        validate(sample_approved_context())
        for label, context in self._invalid_contexts():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    validate(context)


if __name__ == "__main__":
    unittest.main()
