"""Freeze a new protocol version: copy its baseline files and write their manifest.

The file list is the predecessor manifest's, plus any `--add` paths, so a version can
never silently drop a baseline document. Sources must be under configs/ or docs/; the
lifecycle record and compute plan are mutable and never frozen. Refuses to overwrite an
existing version directory. See docs/protocol-lifecycle.md.

Usage: python -m scripts.protocol_snapshot --version v2.1.0 --predecessor v2.0.0 \\
           --amendment docs/<amendment>.md --approval <slug> [--add docs/<file>.md ...]
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / 'configs/protocols'
NEVER_FROZEN = {'configs/protocol-lifecycle.json', 'configs/compute-plan.json'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(version, predecessor, *, amendment, approval, add=(), root=ROOT):
    protocols = root / 'configs/protocols'
    old_path = protocols / predecessor / 'baseline-manifest.json'
    old = json.loads(old_path.read_text())
    sources = [e['source_path'] for e in old['files']]
    for path in add:
        if path not in sources:
            sources.append(path)
    for path in sources:
        parts = Path(path).parts
        if Path(path).is_absolute() or '..' in parts or parts[0] not in ('configs', 'docs') or path in NEVER_FROZEN:
            raise ValueError(f'{path} cannot be part of a protocol baseline')
    out = protocols / version
    out.mkdir(exist_ok=False)
    files = []
    for path in sources:
        data = (root / path).read_bytes()
        target = out / 'snapshot' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'xb') as f:
            f.write(data)
        files.append({'source_path': path, 'snapshot_path': f'snapshot/{path}', 'bytes': len(data),
                      'sha256': _sha(data)})
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    manifest = {
        'manifest_schema_version': 'baseline-manifest-v1',
        'protocol_version': f'amazon4u-protocol-{version}',
        'lifecycle_at_freeze': old['lifecycle_at_freeze'],
        'execution_revision': None,
        'source_git_revision': revision,
        'source_state': 'worktree_snapshot_of_user_approved_amendment',
        'profiling_authorized_by_freeze': False,
        'training_authorized_by_freeze': False,
        'deployment_decision': 'NOT_AUTHORIZED',
        'heldout_model_results_exposed': False,
        'prior_split_metadata_audits_already_performed': True,
        'hash_algorithm': 'SHA-256',
        'predecessor_manifest': f'configs/protocols/{predecessor}/baseline-manifest.json',
        'predecessor_manifest_sha256': _sha(old_path.read_bytes()),
        'amendment_document': amendment,
        'approval': approval,
        'files': files,
    }
    (out / 'baseline-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--predecessor', required=True)
    ap.add_argument('--amendment', required=True)
    ap.add_argument('--approval', required=True)
    ap.add_argument('--add', action='append', default=[])
    args = ap.parse_args(argv)
    manifest = snapshot(args.version, args.predecessor, amendment=args.amendment,
                        approval=args.approval, add=args.add)
    print(json.dumps({'protocol_version': manifest['protocol_version'], 'files': len(manifest['files'])}))


if __name__ == '__main__':
    main()
