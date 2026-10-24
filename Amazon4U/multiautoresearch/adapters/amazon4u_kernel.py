"""Amazon4U kernel runtime adapter.

Thin bounded hydration, preflight verification, staged execution dispatch,
and exact H callable keyword argument handoff.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
import uuid

from multiautoresearch.adapters.amazon4u import (
    Amazon4UError,
    _retain_h_bundle_sources,
    atomic_move_no_replace,
    build_amazon4u_record,
    hydrate_bundle,
    validate_amazon4u_authorization,
    validate_amazon4u_spec,
    validate_approved_offline_context,
    validate_canonical_h_envelope,
    verify_h_provenance,
)


class KernelExecutionError(Amazon4UError):
    """Explicit stage, preflight, or callable failure inside kernel runner."""


_LATEST_EXECUTION_MARKER: dict[str, object] | None = None


def adapt_h_callable(
    callable_fn: Callable[..., dict[str, Any]],
    spec: dict[str, Any],
    *,
    expected_spec: dict[str, Any],
    authorization: dict[str, Any],
    data: Any,
    expectations: dict[str, Any],
    run_envelope: dict[str, Any],
    expected_run_envelope: dict[str, Any],
    bundle_sources: dict[str, bytes],
    expected_bundle: dict[str, Any],
    light_initial: Any = None,
    backend: Any = None,
) -> dict[str, Any]:
    """Invoke H callable with exact 11 arguments matching scripts.training_entrypoint:execute_fixture.

    Validates that real-data or production bootstrap requests fail closed before execution.
    """
    if spec.get("data_kind") != "owned_synthetic_packet":
        raise KernelExecutionError(
            f"Production bootstrap / real data ('{spec.get('data_kind')}') is NOT authorized; owned_synthetic_packet only."
        )

    # Pass exact arguments required by scripts.training_entrypoint:execute_fixture
    return callable_fn(
        spec,
        expected_spec=expected_spec,
        authorization=authorization,
        data=data,
        expectations=expectations,
        run_envelope=run_envelope,
        expected_run_envelope=expected_run_envelope,
        bundle_sources=bundle_sources,
        expected_bundle=expected_bundle,
        light_initial=light_initial,
        backend=backend,
    )


def run_kernel_stages(
    workdir: Path,
    outdir: Path,
    spec: dict[str, Any],
    auth: dict[str, Any],
    *,
    h_callable: Callable[..., dict[str, Any]] | None = None,
    fake_bootstrap: dict[str, Any] | None = None,
    fail_stage: str | None = None,
    checkpoint_verifier: Callable[[bytes], bool] | None = None,
    already_hydrated: bool = False,
    context: Any = None,
    capability_provider: Callable[[], dict[str, Any]] | None = None,
) -> int:
    """Execute declared kernel stages sequentially with staged failure propagation.

    Stages:
      1. preflight: validates spec, authorization, runtime/ABI expectations BEFORE creating dirs/logs.
      2. hydration: validates and unpacks bundle without symlinks, escapes, or silently skipping.
      3. callable_handoff: passes exact arguments to H callable; catches exceptions.
      4. checkpoint: handles checkpoint verification / artifact staging.
      5. publish: saves typed result records into outdir using atomic no-replace writes.

    All failures propagate nonzero; no success is claimed from missing or skipped stages.
    """
    global _LATEST_EXECUTION_MARKER
    _LATEST_EXECUTION_MARKER = {
        "origin_workdir": str(workdir),
        "origin_outdir": str(outdir),
        "module_file": str(Path(__file__).resolve()),
        "status": "executed_from_hydrated_bundle",
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    # STAGE 1: PREFLIGHT (must happen BEFORE directory or log creation)
    if fail_stage == "preflight":
        return 1

    # Check for direct script no-op attempt
    if spec.get("direct_script_execution"):
        return 1

    try:
        valid_spec = validate_amazon4u_spec(spec)
        valid_auth = validate_amazon4u_authorization(auth, actual_spec=valid_spec)
    except Exception:
        return 1

    # In preflight, require and validate independent ApprovedOfflineContext (no context=None bypass)
    if context is None:
        return 1

    try:
        valid_ctx = validate_approved_offline_context(context)
        if valid_ctx.canonical_h_spec != valid_spec:
            return 1
        if valid_ctx.h_authorization != valid_auth:
            return 1
    except Exception:
        return 1

    # Refusal of missing callable/bootstrap BEFORE effects
    if not callable(h_callable) or type(fake_bootstrap) is not dict:
        return 1

    # In preflight, if bootstrap is provided, require explicit expected_spec and expectations
    if fake_bootstrap is not None:
        if "expected_spec" not in fake_bootstrap:
            return 1
        if fake_bootstrap["expected_spec"] != valid_spec:
            return 1

        if "authorization" in fake_bootstrap and fake_bootstrap["authorization"] != valid_auth:
            return 1

        # Reject inconsistent bootstrap expectations before relevant effects/H call
        if "expected_bundle" in fake_bootstrap:
            if fake_bootstrap["expected_bundle"] != valid_ctx.expected_h_bundle:
                return 1

        # Validate run envelope and expected_run_envelope against independent canonical metadata
        run_env = fake_bootstrap.get("run_envelope")
        exp_run_env = fake_bootstrap.get("expected_run_envelope")
        if not isinstance(run_env, dict) or not isinstance(exp_run_env, dict):
            return 1
        if exp_run_env != valid_ctx.expected_h_envelope:
            return 1
        try:
            validate_canonical_h_envelope(
                run_env,
                expected_spec=valid_spec,
                expected_bundle=valid_ctx.expected_h_bundle,
                expected_envelope=valid_ctx.expected_h_envelope,
            )
        except Exception:
            return 1

        retained_run_envelope = copy.deepcopy(run_env)
        retained_expected_run_envelope = copy.deepcopy(valid_ctx.expected_h_envelope)
        retained_h_authorization = copy.deepcopy(valid_ctx.h_authorization)

        # Snapshot before capability_provider can mutate the caller's bootstrap.
        try:
            retained_h_sources = _retain_h_bundle_sources(
                fake_bootstrap.get("bundle_sources"),
                valid_ctx.expected_h_bundle,
            )
        except Exception as exc:
            print("Refusal: H bundle source coherence invalid: " + str(exc), file=sys.stderr)
            return 1

        expectations = fake_bootstrap.get("expectations")
        if not isinstance(expectations, dict) or "resource_policy" not in expectations:
            return 1

        # Capability provider and injected capability comparison (S05, OP06)
        actual_caps = fake_bootstrap.get("actual_capabilities")
        if capability_provider is not None:
            actual_caps = capability_provider()
        expected_caps = valid_ctx.expected_o_environment["capabilities"]
        for declared_caps in (
            fake_bootstrap.get("expected_capabilities"),
            expectations.get("capabilities"),
        ):
            if declared_caps is not None and declared_caps != expected_caps:
                return 1

        required_capability_keys = {
            "package", "version", "nativeIO", "ABI", "dependency",
            "CPU", "runtime", "resource", "callable", "bootstrap",
        }
        if fail_stage is None and "expected_spec" in fake_bootstrap and "authorization" in fake_bootstrap:
            if actual_caps is None or expected_caps is None:
                return 1
            if not isinstance(actual_caps, dict) or not isinstance(expected_caps, dict):
                return 1
            missing_caps = required_capability_keys - set(expected_caps.keys())
            if missing_caps:
                return 1
            for k, v in expected_caps.items():
                if actual_caps.get(k) != v:
                    return 1
        elif actual_caps is not None or expected_caps is not None:
            if actual_caps is None or expected_caps is None:
                return 1
            if not isinstance(actual_caps, dict) or not isinstance(expected_caps, dict):
                return 1
            for k, v in expected_caps.items():
                if actual_caps.get(k) != v:
                    return 1

        # Environment, dependency, and ABI capability guard (OP06)
        expected_runtime = expectations.get("expected_runtime", {})
        if isinstance(expected_runtime, dict):
            req_cuda = expected_runtime.get("cuda_available", False)
            if req_cuda:
                print("Refusal: Preflight CPU-only scope violated (cuda_available)", file=sys.stderr)
                return 1  # CPU reference scope only; CUDA/GPU forbidden
            req_device = expected_runtime.get("device", "cpu")
            if req_device != "cpu":
                print("Refusal: Preflight CPU-only scope violated (device)", file=sys.stderr)
                return 1
            req_accel = expected_runtime.get("accelerator", "none")
            if req_accel not in ("none", ""):
                print("Refusal: Preflight CPU-only scope violated (accelerator)", file=sys.stderr)
                return 1


        if fail_stage is None:
            required_bootstrap_keys = {
                "expected_spec", "authorization", "data", "expectations",
                "run_envelope", "expected_run_envelope", "bundle_sources", "expected_bundle",
            }
            missing_bs_keys = required_bootstrap_keys - set(fake_bootstrap.keys())
            if missing_bs_keys:
                return 1

    # Fresh output directory check: adoption of existing output directory is strictly forbidden (S05)
    if outdir.exists():
        return 1

    # Only after preflight passes, create directories and initialize logging
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=False)
    log_file = outdir / "kernel-run.log"

    def log(msg: str) -> None:
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass

    log("=== Stage 1: Preflight Validation PASS ===")

    # STAGE 2: HYDRATION
    log("=== Stage 2: Hydration Verification ===")
    if fail_stage == "hydration":
        log("Hydration failure triggered intentionally by test control.")
        return 2

    bundle_files_json = workdir / "bundle_files.json"
    manifest_json = workdir / "manifest.json"

    # Missing bundle_files must NOT silently skip hydration
    if not bundle_files_json.exists():
        log("Hydration refusal: bundle_files.json missing.")
        return 2

    if already_hydrated:
        try:
            files_dict = json.loads(bundle_files_json.read_text(encoding="utf-8"))
            expected_manifest = (
                valid_ctx.expected_o_outer
                or (fake_bootstrap.get("outer_manifest") or fake_bootstrap.get("expected_manifest")
                    if fake_bootstrap is not None else None)
            )

            if expected_manifest is not None:
                manifest_dict = expected_manifest
            elif manifest_json.exists():
                manifest_dict = json.loads(manifest_json.read_text(encoding="utf-8"))
            else:
                log("Hydration verification failure: manifest.json missing.")
                return 2

            manifest_files = manifest_dict.get("files", {})
            if not isinstance(manifest_files, dict) or set(files_dict.keys()) != set(manifest_files.keys()):
                log("Hydration verification failure: bundle files and manifest mismatch.")
                return 2

            for rel_path, exp_meta in manifest_files.items():
                target_file = workdir / rel_path
                if not target_file.exists() or not target_file.is_file() or target_file.is_symlink():
                    log(f"Hydration verification failure: file missing or invalid: {rel_path}")
                    return 2
                if any(parent.is_symlink() for parent in target_file.parents):
                    log(f"Hydration verification failure: ancestor symlink detected: {rel_path}")
                    return 2
                if target_file.stat().st_size != exp_meta.get("size"):
                    log(f"Hydration verification failure: size mismatch for {rel_path}")
                    return 2
                actual_sha = hashlib.sha256(target_file.read_bytes()).hexdigest()
                if actual_sha != exp_meta.get("sha256"):
                    log(f"Hydration verification failure: SHA mismatch for {rel_path}")
                    return 2

            scripts_init = workdir / "scripts" / "__init__.py"
            if scripts_init.exists():
                log("Hydration verification failure: scripts/__init__.py must not exist.")
                return 2

            log(f"Hydration verified: {len(files_dict)} files present, verified, and match manifest in workspace.")
        except Exception as exc:
            log(f"Hydration verification error: {exc}")
            return 2
    else:
        try:
            files_dict = json.loads(bundle_files_json.read_text(encoding="utf-8"))
            manifest_dict = json.loads(manifest_json.read_text(encoding="utf-8")) if manifest_json.exists() else None
            hydrate_bundle(files_dict, workdir, manifest=manifest_dict)
            log(f"Successfully hydrated {len(files_dict)} bundle files.")
        except Exception as exc:
            log(f"Hydration error: {exc}")
            return 2

    # STAGE 3: CALLABLE HANDOFF
    log("=== Stage 3: Callable Handoff ===")
    if fail_stage == "callable":
        log("Callable failure triggered intentionally by test control.")
        return 3

    if h_callable is None:
        log("Refusal: Production live bootstrap unavailable in O-OFFLINE scope; deployment NOT_AUTHORIZED.")
        return 3

    if fake_bootstrap is None:
        log("Refusal: Explicit trusted bootstrap expectations required; cannot synthesize.")
        return 3

    bs = fake_bootstrap
    # Require explicit trusted expectations; do not fabricate defaults
    required_bootstrap_keys = {
        "expected_spec", "authorization", "data", "expectations",
        "run_envelope", "expected_run_envelope", "bundle_sources", "expected_bundle",
    }
    missing_bs_keys = required_bootstrap_keys - set(bs.keys())
    if missing_bs_keys:
        log(f"Callable handoff refusal: missing bootstrap expectations: {missing_bs_keys}")
        return 3

    try:
        h_result = adapt_h_callable(
            h_callable,
            valid_spec,
            expected_spec=valid_ctx.canonical_h_spec,
            authorization=retained_h_authorization,
            data=bs["data"],
            expectations=bs["expectations"],
            run_envelope=retained_run_envelope,
            expected_run_envelope=retained_expected_run_envelope,
            bundle_sources=retained_h_sources,
            expected_bundle=valid_ctx.expected_h_bundle,
            light_initial=bs.get("light_initial"),
            backend=bs.get("backend"),
        )
    except Exception as exc:
        log(f"H callable execution failed: {exc}")
        return 3

    if not isinstance(h_result, dict) or h_result.get("schema_version") != "amazon4u-H-result-v1":
        log("Callable returned invalid result schema")
        return 3

    try:
        verify_h_provenance(h_result, expected_spec=valid_spec, expected_run_envelope=valid_ctx.expected_h_envelope)
    except Exception as exc:
        log(f"H result provenance verification failed: {exc}")
        return 3

    # STAGE 4: CHECKPOINT VERIFICATION
    log("=== Stage 4: Checkpoint Verification ===")
    if fail_stage == "checkpoint":
        log("Checkpoint failure triggered intentionally by test control.")
        return 4

    is_baseline = valid_spec.get("method") in ("CorePopularity", "KG-only")
    checkpoint_data = bs.get("checkpoint_data")
    if is_baseline:
        if checkpoint_data is not None:
            log("Deterministic baseline cannot have physical checkpoint data.")
            return 4
        # Deterministic baseline: logical receipt, no physical checkpoint written
        log("Deterministic baseline: physical checkpoint not applicable.")
    else:
        if checkpoint_data is None:
            log("Refusal: Learned method missing checkpoint data.")
            return 4
        raw_ckpt = checkpoint_data if isinstance(checkpoint_data, bytes) else str(checkpoint_data).encode("utf-8")
        if checkpoint_verifier is None:
            log("Refusal: Mandatory checkpoint verifier is absent.")
            return 4
        if not checkpoint_verifier(raw_ckpt):
            log("Checkpoint verifier rejected checkpoint data")
            return 4
        if "expected_checkpoint_sha256" not in bs:
            log("Refusal: expected_checkpoint_sha256 is mandatory for learned methods.")
            return 4
        actual_sha = hashlib.sha256(raw_ckpt).hexdigest()
        if actual_sha != bs["expected_checkpoint_sha256"]:
            log(f"Checkpoint SHA mismatch: expected {bs['expected_checkpoint_sha256']}, got {actual_sha}")
            return 4
        ckpt_out = outdir / "checkpoint.bin"
        temp_ckpt = outdir / f".checkpoint.bin.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        created_temp = False
        try:
            fd = os.open(str(temp_ckpt), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created_temp = True
            with os.fdopen(fd, "wb") as f:
                f.write(raw_ckpt)
            atomic_move_no_replace(temp_ckpt, ckpt_out)
            created_temp = False
        finally:
            if created_temp and temp_ckpt.exists():
                try:
                    temp_ckpt.unlink()
                except OSError:
                    pass
        log("Checkpoint written successfully.")

    # STAGE 5: PUBLICATION
    log("=== Stage 5: Artifact Publication ===")
    if fail_stage == "publish":
        log("Publish failure triggered intentionally by test control.")
        return 5

    try:
        ver = valid_ctx.submission_binding.get("version") or bs.get("version")
        if ver is None or type(ver) is not int or isinstance(ver, bool) or ver < 1:
            log(f"Refusal: Invalid or missing version {ver}; defaulting is strictly forbidden")
            return 5
        plan_summary = {
            "schema_version": "amazon4u-O-plan-v1",
            "spec": valid_spec,
            "authorization": valid_auth,
            "version": ver,
        }
        o_record = build_amazon4u_record(plan_summary, h_result)
        o_record["version"] = ver
        if _LATEST_EXECUTION_MARKER is not None:
            o_record["execution_marker"] = dict(_LATEST_EXECUTION_MARKER)

        out_manifest = (
            valid_ctx.expected_o_outer
            or bs.get("expected_manifest")
            or bs.get("outer_manifest")
        )
        if out_manifest is None:
            log("Refusal: Missing O outer manifest expectation; fallback to H expected_bundle is strictly forbidden.")
            return 5

        def _publish_json_atomic(dst: Path, data: Any) -> None:
            tmp = dst.with_name(f".{dst.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
            payload = (json.dumps(data, indent=2) + "\n").encode("utf-8")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            atomic_move_no_replace(tmp, dst)

        _publish_json_atomic(outdir / "plan.json", plan_summary)
        _publish_json_atomic(outdir / "result.json", h_result)
        _publish_json_atomic(outdir / "run_envelope.json", retained_run_envelope)
        _publish_json_atomic(outdir / "manifest.json", out_manifest)
        _publish_json_atomic(outdir / "amazon4u_record.json", o_record)
        log("Artifacts published successfully.")
    except Exception as exc:
        log(f"Publication error: {exc}")
        return 5

    log("=== Kernel stages finished successfully ===")
    return 0


def hydrate_bundle_in_kernel(
    bundle_files: dict[str, str],
    workdir: Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hydrate bundle files inside Kaggle kernel environment with namespace preservation."""
    return hydrate_bundle(bundle_files, workdir, manifest=manifest)
