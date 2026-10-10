#!/usr/bin/env python3
"""Manage Autolab benchmark and training jobs on Kaggle Kernels.

Generic, domain-agnostic replacement for hf_job.py.
Uses Kaggle Kernels for compute and Kaggle Datasets/Models for storage.

Usage:
    uv run scripts/kaggle_job.py preflight
    uv run scripts/kaggle_job.py launch --mode prepare
    uv run scripts/kaggle_job.py launch --mode experiment
    uv run scripts/kaggle_job.py launch --dry-run
    uv run scripts/kaggle_job.py status [kernel_slug]
    uv run scripts/kaggle_job.py logs [kernel_slug]
    uv run scripts/kaggle_job.py output [kernel_slug]
"""

from __future__ import annotations

import argparse
import base64
import difflib
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
DEFAULT_BUNDLE_DIR = RUNTIME_DIR / "kaggle-kernel"
LAST_JOB_PATH = RUNTIME_DIR / "kaggle-job-last.json"
KAGGLE_JOB_STATE_DIR = RUNTIME_DIR / "kaggle-jobs"
KAGGLE_JOB_LOG_DIR = RUNTIME_DIR / "kaggle-logs"

DEFAULT_POLL_INTERVAL = 30  # seconds
TERMINAL_KERNEL_STAGES = {"complete", "error", "cancelacknowledged"}

SUMMARY_KEYS = {
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
}

KNOWN_CHANGE_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scheduler", ("FINAL_LR_FRAC", "WARMDOWN_RATIO", "get_lr_multiplier")),
    ("lm_head_weight_decay", ("lm_head_params", "weight_decay")),
    ("value_embeds_weight_decay", ("value_embeds_params", "weight_decay")),
    ("muon_beta2", ("momentum=0.95", "beta2=")),
    ("window_pattern", ("WINDOW_PATTERN",)),
    ("gqa_kv_heads", ("n_kv_head",)),
    ("value_embed_stride", ("has_ve", "layer_idx %")),
    ("attention_branch_scale", ("1.3 *", "attn_out")),
    ("attention_temperature", ("temperature",)),
)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_file(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_kaggle_cli() -> str:
    """Resolve the kaggle CLI executable path."""
    explicit = os.environ.get("KAGGLE_CLI_PATH")
    if explicit and os.path.exists(explicit):
        return explicit

    venv_path = ROOT / ".venv" / "bin" / "kaggle"
    if venv_path.exists():
        return str(venv_path)

    sibling_venv = ROOT.parent / ".venv" / "bin" / "kaggle"
    if sibling_venv.exists():
        return str(sibling_venv)

    which_kaggle = shutil.which("kaggle")
    if which_kaggle:
        return which_kaggle

    raise SystemExit(
        "Could not find `kaggle` CLI executable. "
        "Install it via `pip install kaggle` or set KAGGLE_CLI_PATH."
    )


def resolve_kaggle_username(explicit: str | None = None) -> str:
    """Resolve Kaggle username from explicit argument, env vars, or credentials."""
    if explicit:
        return explicit

    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_user:
        return env_user

    # Try parsing ~/.kaggle/kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
            user = data.get("username")
            if isinstance(user, str) and user:
                return user
        except Exception:
            pass

    # Try running `kaggle config view`
    try:
        res = subprocess.run(
            [resolve_kaggle_cli(), "config", "view"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in res.stdout.splitlines():
            if "username:" in line.lower():
                val = line.split(":", 1)[1].strip()
                if val and val.lower() != "none":
                    return val
    except Exception:
        pass

    raise SystemExit(
        "Unable to determine Kaggle username. "
        "Please set KAGGLE_USERNAME environment variable or configure ~/.kaggle/kaggle.json"
    )


def run_command(argv: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=capture_output, check=False)


def slugify_label_value(value: str) -> str:
    lowered = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return cleaned[:48]


def coerce_value(raw: str) -> int | float | str:
    raw = raw.strip()
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def parse_metrics(text: str) -> dict[str, int | float | str] | None:
    """Extract metrics from training stdout or log."""
    metrics: dict[str, int | float | str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_]+):\s+(.+)$", line.strip())
        if not match:
            continue
        key, raw = match.groups()
        if key in SUMMARY_KEYS:
            metrics[key] = coerce_value(raw)
    return metrics if "val_bpb" in metrics else None


def load_pyproject_dependencies() -> list[str]:
    """Read dependencies from pyproject.toml."""
    pyproject_path = ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        deps = data.get("project", {}).get("dependencies", [])
        return [str(d) for d in deps] if isinstance(deps, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Preflight & Experiment Validation
# ---------------------------------------------------------------------------

def train_diff_preview() -> tuple[list[str], int, int]:
    train_path = ROOT / "train.py"
    orig_path = ROOT / "train_orig.py"
    if not train_path.exists() or not orig_path.exists():
        return [], 0, 0
    orig_lines = orig_path.read_text(encoding="utf-8").splitlines(keepends=True)
    curr_lines = train_path.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            orig_lines,
            curr_lines,
            fromfile="train_orig.py",
            tofile="train.py",
        )
    )
    hunk_count = sum(1 for line in diff if line.startswith("@@"))
    changed_line_count = sum(
        1
        for line in diff
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    return [line.rstrip("\n") for line in diff], hunk_count, changed_line_count


def detect_known_change_categories(preview: list[str]) -> list[str]:
    changed_lines = [
        line[1:]
        for line in preview
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    categories: list[str] = []
    for name, needles in KNOWN_CHANGE_CATEGORY_PATTERNS:
        if all(any(needle in line for line in changed_lines) for needle in needles):
            categories.append(name)
    return categories


def collect_launch_context() -> dict[str, object]:
    master_path = ROOT / "research" / "live" / "master.json"
    master = load_json_file(master_path) if master_path.exists() else None
    return {
        "campaign": os.environ.get("AUTOLAB_CAMPAIGN"),
        "experiment_id": os.environ.get("AUTOLAB_EXPERIMENT_ID"),
        "worker_id": os.environ.get("AUTOLAB_WORKER_ID"),
        "hypothesis": os.environ.get("AUTOLAB_HYPOTHESIS"),
        "master_hash": master.get("promoted_hash") if isinstance(master, dict) else None,
        "master_val_bpb": master.get("val_bpb") if isinstance(master, dict) else None,
    }


def build_preflight_report(context: dict[str, object]) -> dict[str, object]:
    report: dict[str, object] = {
        "context": context,
        "errors": [],
        "warnings": [],
    }
    errors = report["errors"]
    warnings = report["warnings"]
    assert isinstance(errors, list)
    assert isinstance(warnings, list)

    train_path = ROOT / "train.py"
    orig_path = ROOT / "train_orig.py"
    report["train_exists"] = train_path.exists()
    report["train_orig_exists"] = orig_path.exists()
    if not train_path.exists():
        errors.append("missing train.py")
    if not orig_path.exists():
        errors.append("missing train_orig.py; run refresh_master first")

    preview, hunk_count, changed_line_count = train_diff_preview()
    report["diff_preview"] = preview
    report["diff_hunks"] = hunk_count
    report["diff_changed_lines"] = changed_line_count
    report["known_change_categories"] = detect_known_change_categories(preview)

    if orig_path.exists() and train_path.exists():
        same = train_path.read_text(encoding="utf-8") == orig_path.read_text(encoding="utf-8")
        report["train_matches_orig"] = same
        is_baseline = "baseline" in (context.get("experiment_id") or "").lower()
        if same and not is_baseline:
            errors.append("train.py matches train_orig.py; no experiment change present (use 'baseline' ID to override)")
        elif same and is_baseline:
            warnings.append("baseline run: train.py matches train_orig.py")
        elif hunk_count == 0:
            errors.append("unable to compute a diff between train.py and train_orig.py")
        if changed_line_count > 30:
            warnings.append("train.py differs from train_orig.py by many lines; review for multi-change drift")

    return report


def print_preflight_report(report: dict[str, object]) -> None:
    print("Preflight Check:")
    context = report.get("context")
    if isinstance(context, dict):
        parts: list[str] = [
            f"{k}={v}" for k, v in context.items() if v
        ]
        if parts:
            print("  " + " | ".join(parts))

    print(
        "  "
        + f"diff_hunks={report.get('diff_hunks', 0)}"
        + f" changed_lines={report.get('diff_changed_lines', 0)}"
        + f" categories={report.get('known_change_categories', [])}"
    )
    for warning in report.get("warnings", []):
        print(f"  [WARNING] {warning}")
    for error in report.get("errors", []):
        print(f"  [ERROR] {error}")


# ---------------------------------------------------------------------------
# Code Bundling & Kernel Rendering
# ---------------------------------------------------------------------------

def encode_file_bytes(path: Path) -> str:
    """Compress and base64-encode a file for bundle hydration."""
    raw = path.read_bytes()
    compressed = gzip.compress(raw, mtime=0)
    return base64.b64encode(compressed).decode("ascii")


def collect_project_files() -> dict[str, str]:
    """Collect source files in root to bundle into the kernel."""
    files: dict[str, str] = {}
    candidate_names = ["prepare.py", "train.py", "model.py", "pyproject.toml"]
    for name in candidate_names:
        p = ROOT / name
        if p.is_file():
            files[name] = encode_file_bytes(p)

    # Include any supporting .py files in root
    for p in ROOT.glob("*.py"):
        if p.name not in files and not p.name.startswith((".", "sitecustomize")):
            files[p.name] = encode_file_bytes(p)
            
    # Include any models
    models_dir = ROOT / "models"
    if models_dir.is_dir():
        for p in models_dir.rglob("*.py"):
            rel_path = f"models/{p.name}" # Assumes flat models/ folder
            files[rel_path] = encode_file_bytes(p)

    return files


def build_kernel_script(
    mode: str,
    pip_dependencies: list[str],
    user_train_args: str = "",
) -> str:
    """Generate the autonomous standalone Python script executed inside Kaggle."""
    files_payload = collect_project_files()
    files_json = json.dumps(files_payload)
    pip_deps_json = json.dumps(pip_dependencies)

    return f'''#!/usr/bin/env python3
"""Autonomous Autolab Kaggle Kernel Runner — Mode: {mode}"""
import base64
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MODE = {mode!r}
FILES = {files_json}
PIP_DEPENDENCIES = {pip_deps_json}
USER_TRAIN_ARGS = {user_train_args!r}

SUMMARY_KEYS = {sorted(SUMMARY_KEYS)!r}

def run_cmd(argv, cwd=None):
    print(">> Executing: " + " ".join(argv))
    sys.stdout.flush()
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return proc.wait()

def install_dependencies():
    if not PIP_DEPENDENCIES:
        return
    print("Installing dependencies: " + ", ".join(PIP_DEPENDENCIES))
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + PIP_DEPENDENCIES
    rc = run_cmd(cmd)
    if rc != 0:
        print("Warning: pip install returned non-zero code " + str(rc), file=sys.stderr)

def hydrate_files(workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    for rel_path, payload in FILES.items():
        dst = workdir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = gzip.decompress(base64.b64decode(payload))
        dst.write_bytes(raw)
        print("Hydrated: " + rel_path)

def parse_metrics_from_text(text: str):
    metrics = {{}}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]+):\\s+(.+)$", line.strip())
        if not m:
            continue
        k, raw = m.groups()
        if k in SUMMARY_KEYS:
            v = raw.strip()
            for caster in (int, float):
                try:
                    v = caster(v)
                    break
                except ValueError:
                    pass
            metrics[k] = v
    return metrics if "val_bpb" in metrics else None

def main():
    print("=== Kaggle Execution Environment ===")
    print("Python:", sys.version)
    if Path("/kaggle/input").exists():
        print("Mounted /kaggle/input datasets:")
        for item in sorted(Path("/kaggle/input").iterdir()):
            print("  " + item.name)
    sys.stdout.flush()

    install_dependencies()

    workdir = Path("/kaggle/working/experiment")
    outdir = Path("/kaggle/working")
    hydrate_files(workdir)

    log_path = outdir / "autolab-run.log"
    target_script = "prepare.py" if MODE == "prepare" else "train.py"
    target_file = workdir / target_script

    if not target_file.exists():
        print(f"Error: Target file {{target_script}} does not exist in bundle!", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(target_file)]
    if USER_TRAIN_ARGS:
        cmd.extend(USER_TRAIN_ARGS.split())

    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
        rc = proc.wait()

    # Parse and write metrics.json
    try:
        log_text = log_path.read_text(encoding="utf-8")
        metrics = parse_metrics_from_text(log_text)
        if metrics:
            metrics_file = outdir / "metrics.json"
            metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            print("Parsed metrics successfully: " + json.dumps(metrics))
        else:
            print("No val_bpb metric found in run log", file=sys.stderr)
    except Exception as e:
        print("Error parsing metrics: " + str(e), file=sys.stderr)

    return rc

if __name__ == "__main__":
    sys.exit(main())
'''


def build_kernel_metadata(
    username: str,
    slug: str,
    code_file: str,
    accelerator: str,
    timeout: int,
    dataset_sources: list[str],
    kernel_sources: list[str],
    model_sources: list[str],
    enable_internet: bool = True,
) -> dict[str, object]:
    """Render the metadata JSON required by Kaggle CLI."""
    # Ensure title has at least 5 characters to pass Kaggle validation
    title = slug.replace("-", " ").title()
    if len(title) < 5:
        title = f"Autolab {title}"

    is_gpu = accelerator.lower() in ("gpu", "nvidiateslat4", "nvidiateslap100")
    is_tpu = accelerator.lower() in ("tpu", "tpu1vmv38", "tpu1vmv5e")

    return {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if is_gpu else "false",
        "enable_tpu": "true" if is_tpu else "false",
        "machine_shape": accelerator if accelerator not in ("none", "", "gpu") else ("NvidiaTeslaT4" if is_gpu else ""),
        "accelerator": accelerator if accelerator not in ("none", "") else "",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": model_sources,
    }


def render_bundle(
    output_dir: Path,
    username: str,
    slug: str,
    mode: str,
    accelerator: str,
    timeout: int,
    dataset_sources: list[str],
    kernel_sources: list[str],
    model_sources: list[str],
    pip_dependencies: list[str],
    train_args: str = "",
) -> Path:
    """Render bundle directory with script and kernel-metadata.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    code_filename = "kernel.py"
    script_path = output_dir / code_filename

    script_content = build_kernel_script(
        mode=mode,
        pip_dependencies=pip_dependencies,
        user_train_args=train_args,
    )
    script_path.write_text(script_content, encoding="utf-8")

    metadata = build_kernel_metadata(
        username=username,
        slug=slug,
        code_file=code_filename,
        accelerator=accelerator,
        timeout=timeout,
        dataset_sources=dataset_sources,
        kernel_sources=kernel_sources,
        model_sources=model_sources,
    )
    meta_path = output_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_dir


# ---------------------------------------------------------------------------
# CLI Command Implementations
# ---------------------------------------------------------------------------

def resolve_kernel_slug(mode: str, explicit: str | None, context: dict[str, object]) -> str:
    if explicit:
        return slugify_label_value(explicit)
    if mode == "prepare":
        return "autolab-prepare"
    exp_id = context.get("experiment_id")
    if isinstance(exp_id, str) and exp_id:
        return f"autolab-exp-{slugify_label_value(exp_id)}"
    # Fallback with timestamp
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"autolab-exp-{ts}"


def persist_job_state(state: dict[str, object]) -> None:
    slug = state.get("kernel_slug")
    if not isinstance(slug, str) or not slug:
        return
    KAGGLE_JOB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = KAGGLE_JOB_STATE_DIR / f"{slug}.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def launch_job(args: argparse.Namespace) -> int:
    context = collect_launch_context()

    if args.mode == "experiment" and not args.skip_preflight:
        report = build_preflight_report(context)
        print_preflight_report(report)
        errors = report.get("errors", [])
        if errors and not args.allow_preflight_fail:
            raise SystemExit(
                "Preflight check failed. Pass --allow-preflight-fail to override."
            )

    username = resolve_kaggle_username(args.username)
    slug = resolve_kernel_slug(args.mode, args.slug, context)
    kernel_ref = f"{username}/{slug}"

    # Accelerator: prepare defaults to none (CPU), experiment defaults to gpu (T4)
    accelerator = args.accelerator
    if not accelerator:
        accelerator = "none" if args.mode == "prepare" else "gpu"

    timeout = args.timeout or (7200 if args.mode == "prepare" else 5400)

    # Datasets, kernels, models
    datasets = [d.strip() for d in (args.datasets or "").split(",") if d.strip()]
    kernel_sources = [k.strip() for k in (args.kernel_sources or "").split(",") if k.strip()]
    model_sources = [m.strip() for m in (args.model_sources or "").split(",") if m.strip()]

    # Collect pip dependencies from pyproject or args
    pip_deps = [d.strip() for d in (args.pip_dependencies or "").split(",") if d.strip()]
    if not pip_deps:
        # Default essential dependencies for standard autoresearch/transformer scripts
        pip_deps = ["torch-geometric>=2.5.0", "pandas>=2.0", "numpy>=1.24", "pyyaml>=6.0", "torch>=2.2"]

    bundle_dir = render_bundle(
        output_dir=args.bundle_dir or DEFAULT_BUNDLE_DIR,
        username=username,
        slug=slug,
        mode=args.mode,
        accelerator=accelerator,
        timeout=timeout,
        dataset_sources=datasets,
        kernel_sources=kernel_sources,
        model_sources=model_sources,
        pip_dependencies=pip_deps,
        train_args=args.train_args or "",
    )
    print(f"Kernel bundle generated at: {bundle_dir}")

    state: dict[str, object] = {
        "kernel_slug": slug,
        "kernel_ref": kernel_ref,
        "username": username,
        "mode": args.mode,
        "accelerator": accelerator,
        "timeout": timeout,
        "bundle_dir": str(bundle_dir),
        "launched_at": now_utc_iso(),
    }
    state.update(context)

    LAST_JOB_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_JOB_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    persist_job_state(state)

    if args.dry_run:
        print("[DRY-RUN] Kernel metadata and script generated. Skipping push.")
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    kaggle_cli = resolve_kaggle_cli()
    push_cmd = [kaggle_cli, "kernels", "push", "-p", str(bundle_dir)]
    print(f"Pushing Kaggle kernel: {kernel_ref}")
    print("  " + " ".join(push_cmd))

    res = run_command(push_cmd, capture_output=True)
    if res.stdout:
        print(res.stdout, end="")
    if res.stderr:
        print(res.stderr, end="", file=sys.stderr)
    if res.returncode != 0:
        return res.returncode

    if not args.detach:
        return poll_and_fetch(kernel_ref, args)
    return 0


def resolve_kernel_ref(explicit: str | None, username: str | None = None) -> str:
    if explicit:
        return explicit if "/" in explicit else f"{resolve_kaggle_username(username)}/{explicit}"
    if LAST_JOB_PATH.exists():
        data = load_json_file(LAST_JOB_PATH)
        if isinstance(data, dict):
            ref = data.get("kernel_ref")
            if isinstance(ref, str) and ref:
                return ref
    raise SystemExit("Kernel reference required; pass one explicitly or launch a job first.")


def poll_and_fetch(kernel_ref: str, args: argparse.Namespace) -> int:
    """Poll kernel status until completion, then fetch output."""
    kaggle_cli = resolve_kaggle_cli()
    interval = getattr(args, "poll_interval", DEFAULT_POLL_INTERVAL)
    print(f"Polling status for {kernel_ref} every {interval}s ...")
    while True:
        res = run_command([kaggle_cli, "kernels", "status", kernel_ref], capture_output=True)
        status_text = (res.stdout or "").strip()
        print(f"  [{time.strftime('%H:%M:%S')}] {status_text}")
        if any(stage in status_text.lower() for stage in TERMINAL_KERNEL_STAGES):
            break
        time.sleep(interval)

    if "complete" in status_text.lower():
        print(f"Kernel {kernel_ref} completed successfully. Fetching output...")
        return fetch_output_impl(kernel_ref, args)

    print(f"Kernel finished with error/status: {status_text}", file=sys.stderr)
    return 1


def fetch_output_impl(kernel_ref: str, args: argparse.Namespace) -> int:
    """Download kernel outputs and update local experiment metrics."""
    kaggle_cli = resolve_kaggle_cli()
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        slug_clean = kernel_ref.replace("/", "_")
        output_dir = KAGGLE_JOB_LOG_DIR / slug_clean
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading output for {kernel_ref} to {output_dir} ...")
    cmd = [kaggle_cli, "kernels", "output", kernel_ref, "-p", str(output_dir), "-o"]
    res = run_command(cmd, capture_output=True)
    if res.stdout:
        print(res.stdout, end="")
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr, end="", file=sys.stderr)
        return res.returncode

    # Extract metrics
    metrics = None
    metrics_path = output_dir / "metrics.json"
    log_path = output_dir / "autolab-run.log"
    if metrics_path.exists():
        metrics = load_json_file(metrics_path)
    elif log_path.exists():
        try:
            metrics = parse_metrics(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if metrics:
        print("Experiment Metrics:")
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print(f"Warning: No val_bpb metric found in {output_dir}", file=sys.stderr)

    # Update state record
    slug = kernel_ref.split("/")[-1]
    state = load_json_file(KAGGLE_JOB_STATE_DIR / f"{slug}.json") or {}
    state["output_dir"] = str(output_dir)
    state["cached_log_path"] = str(log_path) if log_path.exists() else None
    if metrics:
        state["metrics"] = metrics
        state["val_bpb"] = metrics.get("val_bpb")
    persist_job_state(state)

    if LAST_JOB_PATH.exists():
        last = load_json_file(LAST_JOB_PATH) or {}
        if last.get("kernel_ref") == kernel_ref:
            last.update(state)
            LAST_JOB_PATH.write_text(json.dumps(last, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


def status_command(args: argparse.Namespace) -> int:
    kernel_ref = resolve_kernel_ref(args.kernel, args.username)
    res = run_command([resolve_kaggle_cli(), "kernels", "status", kernel_ref])
    return res.returncode


def logs_command(args: argparse.Namespace) -> int:
    kernel_ref = resolve_kernel_ref(args.kernel, args.username)
    res = run_command([resolve_kaggle_cli(), "kernels", "logs", kernel_ref], capture_output=True)
    if res.returncode == 0 and res.stdout:
        print(res.stdout, end="")
        return 0

    # If 'logs' subcommand is not supported in the installed kaggle CLI (e.g. v2.0.0),
    # fall back to checking downloaded output or instruct to run output
    slug_clean = kernel_ref.replace("/", "_")
    cached_log = KAGGLE_JOB_LOG_DIR / slug_clean / "autolab-run.log"
    if cached_log.exists():
        print(f"(Displaying downloaded log from {cached_log})")
        print(cached_log.read_text(encoding="utf-8"))
        return 0

    if "invalid choice: 'logs'" in (res.stderr or ""):
        print(
            f"Note: Your installed Kaggle CLI does not support live 'kernels logs'.\n"
            f"Use 'python scripts/kaggle_job.py output {kernel_ref}' to download the logs and artifacts when finished.",
            file=sys.stderr,
        )
        return 0

    if res.stderr:
        print(res.stderr, end="", file=sys.stderr)
    return res.returncode


def output_command(args: argparse.Namespace) -> int:
    kernel_ref = resolve_kernel_ref(args.kernel, args.username)
    return fetch_output_impl(kernel_ref, args)


def preflight_command(args: argparse.Namespace) -> int:
    context = collect_launch_context()
    report = build_preflight_report(context)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_preflight_report(report)
    errors = report.get("errors", [])
    return 1 if (isinstance(errors, list) and errors) else 0


# ---------------------------------------------------------------------------
# Argument Parser & Entrypoint
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Autolab runs on Kaggle Kernels.")
    parser.add_argument("--username", help="Kaggle username (defaults to KAGGLE_USERNAME or ~/.kaggle/kaggle.json)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # preflight
    p_preflight = subparsers.add_parser("preflight", help="Run local preflight experiment checks")
    p_preflight.add_argument("--json", action="store_true", help="Output report as JSON")

    # launch
    p_launch = subparsers.add_parser("launch", help="Package and push a Kaggle kernel")
    p_launch.add_argument("--mode", choices=["experiment", "prepare"], default="experiment", help="Run mode")
    p_launch.add_argument("--slug", help="Custom kernel slug")
    p_launch.add_argument("--accelerator", default=None, help="Accelerator: gpu, tpu, none (default: gpu for experiment, none for prepare)")
    p_launch.add_argument("--timeout", type=int, help="Maximum execution time in seconds")
    p_launch.add_argument("--datasets", help="Comma-separated Kaggle dataset sources (e.g. owner/dataset)")
    p_launch.add_argument("--kernel-sources", help="Comma-separated parent kernel sources")
    p_launch.add_argument("--model-sources", help="Comma-separated Kaggle model sources")
    p_launch.add_argument("--pip-dependencies", help="Comma-separated extra PyPI packages to install")
    p_launch.add_argument("--train-args", help="Arguments to pass to train.py")
    p_launch.add_argument("--bundle-dir", type=Path, help="Local directory to render kernel bundle")
    p_launch.add_argument("--dry-run", action="store_true", help="Render bundle locally without pushing to Kaggle")
    p_launch.add_argument("--detach", action="store_true", help="Push and exit without polling for completion")
    p_launch.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Status polling interval in seconds")
    p_launch.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    p_launch.add_argument("--allow-preflight-fail", action="store_true", help="Proceed even if preflight has errors")

    # status
    p_status = subparsers.add_parser("status", help="Check status of a Kaggle kernel")
    p_status.add_argument("kernel", nargs="?", help="Kernel slug or owner/slug")

    # logs
    p_logs = subparsers.add_parser("logs", help="Fetch logs of a Kaggle kernel")
    p_logs.add_argument("kernel", nargs="?", help="Kernel slug or owner/slug")

    # output
    p_output = subparsers.add_parser("output", help="Download output files and metrics of a Kaggle kernel")
    p_output.add_argument("kernel", nargs="?", help="Kernel slug or owner/slug")
    p_output.add_argument("--output-dir", type=Path, help="Local directory to save downloaded output")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preflight":
        return preflight_command(args)
    if args.command == "launch":
        return launch_job(args)
    if args.command == "status":
        return status_command(args)
    if args.command == "logs":
        return logs_command(args)
    if args.command == "output":
        return output_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
