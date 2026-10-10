#!/usr/bin/env python3
"""Manage post-training runs on Kaggle Kernels.

Generic, domain-agnostic replacement for hf_job.py.
Uses Kaggle Kernels for compute and Kaggle Datasets/Models for storage.

Usage:
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
    "eval_score",
    "ndcg_10",
    "hit_rate_10",
    "recall_10",
    "precision_10",
    "coverage",
    "raw_accuracy",
    "mean_f1",
    "num_correct",
    "num_examples",
    "train_loss",
    "training_seconds",
    "best_step",
    "best_limited_score",
}

ROOT_SOURCE_FILES = (
    "AGENTS.md",
    "README.md",
    "evaluate.py",
    "model.py",
    "prepare.py",
    "program.md",
    "pyproject.toml",
    "train.py",
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
    """Extract metrics from training/eval log."""
    metrics: dict[str, int | float | str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_]+):\s+(.+)$", line.strip())
        if not match:
            continue
        key, raw = match.groups()
        if key in SUMMARY_KEYS:
            metrics[key] = coerce_value(raw)
    return metrics if "eval_score" in metrics else None


# ---------------------------------------------------------------------------
# Code Bundling & Kernel Rendering
# ---------------------------------------------------------------------------

def encode_file_bytes(path: Path) -> str:
    """Compress and base64-encode a file for bundle hydration."""
    raw = path.read_bytes()
    compressed = gzip.compress(raw, mtime=0)
    return base64.b64encode(compressed).decode("ascii")


def collect_project_files() -> dict[str, str]:
    """Collect source files in root and src/ to bundle into the kernel."""
    files: dict[str, str] = {}
    for rel in ROOT_SOURCE_FILES:
        p = ROOT / rel
        if p.is_file():
            files[rel] = encode_file_bytes(p)

    # Any other .py files in root
    for p in ROOT.glob("*.py"):
        rel = p.name
        if rel not in files and not rel.startswith((".", "sitecustomize")):
            files[rel] = encode_file_bytes(p)

    # All files under src/ and models/
    for d in ["src", "models"]:
        d_path = ROOT / d
        if d_path.is_dir():
            for p in d_path.rglob("*"):
                if p.is_file() and not p.name.startswith("."):
                    rel = p.relative_to(ROOT).as_posix()
                    files[rel] = encode_file_bytes(p)

    return files


def build_kernel_script(
    mode: str,
    pip_dependencies: list[str],
    train_args: str = "",
    eval_args: str = "",
) -> str:
    """Generate the autonomous standalone Python script executed inside Kaggle."""
    files_payload = collect_project_files()
    files_json = json.dumps(files_payload)
    pip_deps_json = json.dumps(pip_dependencies)

    return f'''#!/usr/bin/env python3
"""Autonomous Post-Training Kaggle Kernel Runner — Mode: {mode}"""
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
TRAIN_ARGS = {train_args!r}
EVAL_ARGS = {eval_args!r}

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
    return metrics if "eval_score" in metrics else None

def main():
    print("=== Kaggle Post-Training Environment ===")
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

    log_path = outdir / "posttrain-run.log"

    # 1. Prepare mode
    if MODE == "prepare":
        prep_file = workdir / "prepare.py"
        if not prep_file.exists():
            print("Error: prepare.py not found in bundle", file=sys.stderr)
            return 1
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen([sys.executable, str(prep_file)], cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                log_fh.write(line)
            return proc.wait()

    # 2. Experiment mode: Train then Evaluate
    train_file = workdir / "train.py"
    eval_file = workdir / "evaluate.py"

    if not train_file.exists():
        print("Error: train.py not found in bundle", file=sys.stderr)
        return 1

    train_cmd = [sys.executable, str(train_file)]
    if TRAIN_ARGS:
        train_cmd.extend(TRAIN_ARGS.split())

    with log_path.open("w", encoding="utf-8") as log_fh:
        # Run train
        print(">> Running train.py...")
        proc = subprocess.Popen(train_cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        rc = proc.wait()
        if rc != 0:
            print("train.py failed with code " + str(rc), file=sys.stderr)
            return rc

        # Run evaluate if present and final_model exists
        final_model_dir = workdir / "final_model"
        if eval_file.exists() and final_model_dir.is_dir():
            eval_cmd = [sys.executable, str(eval_file), "--model-path", "final_model"]
            if EVAL_ARGS:
                eval_cmd.extend(EVAL_ARGS.split())
            print(">> Running evaluate.py...")
            proc_eval = subprocess.Popen(eval_cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc_eval.stdout:
                sys.stdout.write(line)
                log_fh.write(line)
            rc_eval = proc_eval.wait()
            if rc_eval != 0:
                print("evaluate.py failed with code " + str(rc_eval), file=sys.stderr)

    # Copy final_model to /kaggle/working/ so it is persisted as output
    final_model_dir = workdir / "final_model"
    if final_model_dir.is_dir():
        dst_model = outdir / "final_model"
        if dst_model.exists():
            shutil.rmtree(dst_model)
        shutil.copytree(final_model_dir, dst_model)
        print("Saved final_model to output directory.")

    # Parse and write metrics.json
    try:
        log_text = log_path.read_text(encoding="utf-8")
        metrics = parse_metrics_from_text(log_text)
        if metrics:
            metrics_file = outdir / "metrics.json"
            metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            print("Parsed metrics successfully: " + json.dumps(metrics))
        else:
            print("Warning: No eval_score found in log", file=sys.stderr)
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
    title = slug.replace("-", " ").title()
    if len(title) < 5:
        title = f"Posttrain {title}"

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
    eval_args: str = "",
) -> Path:
    """Render bundle directory with script and kernel-metadata.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    code_filename = "kernel.py"
    script_path = output_dir / code_filename

    script_content = build_kernel_script(
        mode=mode,
        pip_dependencies=pip_dependencies,
        train_args=train_args,
        eval_args=eval_args,
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

def resolve_kernel_slug(mode: str, explicit: str | None) -> str:
    if explicit:
        return slugify_label_value(explicit)
    if mode == "prepare":
        return "posttrain-prepare"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"posttrain-exp-{ts}"


def persist_job_state(state: dict[str, object]) -> None:
    slug = state.get("kernel_slug")
    if not isinstance(slug, str) or not slug:
        return
    KAGGLE_JOB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = KAGGLE_JOB_STATE_DIR / f"{slug}.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def launch_job(args: argparse.Namespace) -> int:
    username = resolve_kaggle_username(args.username)
    slug = resolve_kernel_slug(args.mode, args.slug)
    kernel_ref = f"{username}/{slug}"

    accelerator = args.accelerator
    if not accelerator:
        accelerator = "none" if args.mode == "prepare" else "gpu"

    timeout = args.timeout or 7200

    datasets = [d.strip() for d in (args.datasets or "").split(",") if d.strip()]
    kernel_sources = [k.strip() for k in (args.kernel_sources or "").split(",") if k.strip()]
    model_sources = [m.strip() for m in (args.model_sources or "").split(",") if m.strip()]

    pip_deps = [d.strip() for d in (args.pip_dependencies or "").split(",") if d.strip()]
    if not pip_deps:
        pyproject_path = ROOT / "pyproject.toml"
        if pyproject_path.exists():
            try:
                data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
                proj_deps = data.get("project", {}).get("dependencies", [])
                pip_deps = [d for d in proj_deps if not d.strip().startswith("torch>")]
            except Exception:
                pass
        if not pip_deps:
            pip_deps = ["torch-geometric>=2.5.0", "pandas>=2.0", "numpy>=1.24", "pyyaml>=6.0"]

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
        eval_args=args.eval_args or "",
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
    log_path = output_dir / "posttrain-run.log"
    if metrics_path.exists():
        metrics = load_json_file(metrics_path)
    elif log_path.exists():
        try:
            metrics = parse_metrics(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if metrics:
        print("Post-Training Metrics:")
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print(f"Warning: No eval_score found in {output_dir}", file=sys.stderr)

    # Update state record
    slug = kernel_ref.split("/")[-1]
    state = load_json_file(KAGGLE_JOB_STATE_DIR / f"{slug}.json") or {}
    state["output_dir"] = str(output_dir)
    state["cached_log_path"] = str(log_path) if log_path.exists() else None
    if metrics:
        state["metrics"] = metrics
        state["eval_score"] = metrics.get("eval_score")
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

    slug_clean = kernel_ref.replace("/", "_")
    cached_log = KAGGLE_JOB_LOG_DIR / slug_clean / "posttrain-run.log"
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


# ---------------------------------------------------------------------------
# Argument Parser & Entrypoint
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage post-training runs on Kaggle Kernels.")
    parser.add_argument("--username", help="Kaggle username (defaults to KAGGLE_USERNAME or ~/.kaggle/kaggle.json)")

    subparsers = parser.add_subparsers(dest="command", required=True)

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
    p_launch.add_argument("--eval-args", help="Arguments to pass to evaluate.py")
    p_launch.add_argument("--bundle-dir", type=Path, help="Local directory to render kernel bundle")
    p_launch.add_argument("--dry-run", action="store_true", help="Render bundle locally without pushing to Kaggle")
    p_launch.add_argument("--detach", action="store_true", help="Push and exit without polling for completion")
    p_launch.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Status polling interval in seconds")

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
