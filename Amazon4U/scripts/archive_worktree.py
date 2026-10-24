#!/usr/bin/env python3
"""Archive a finished `.worktrees/<name>` checkout into the main Git repository.

The working tree (tracked and untracked, non-ignored files) becomes a commit at
`refs/archive/worktrees/<name>`. A clone's own refs are kept under that prefix.
Ignored files and empty directories go to a zstd tarball outside the repository.
Every file on disk is verified against the archive before `--remove` deletes it.

Restore: git worktree add Amazon4U/.worktrees/<name> refs/archive/worktrees/<name>
then extract `<name>.ignored.tar.zst` into it.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKTREES = REPO / 'Amazon4U' / '.worktrees'
DEFAULT_ARCHIVE = Path.home() / 'FSB-archive' / 'amazon4u-worktrees'


def git(*args, env=None, cwd=REPO, data=None):
    return subprocess.run(['git', *args], cwd=cwd, env=env, input=data,
                          capture_output=True, check=True).stdout


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def disk_entries(root):
    files, empty_dirs = [], []
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if not (Path(current) == root and d == '.git')]
        rel = Path(current).relative_to(root)
        links = [d for d in dirs if os.path.islink(os.path.join(current, d))]
        files.extend(str(rel / d) for d in links)
        dirs[:] = [d for d in dirs if d not in links]
        if current != str(root) and not dirs and not names:
            empty_dirs.append(str(rel))
        for name in names:
            if Path(current) == root and name == '.git':
                continue
            files.append(str(rel / name))
    return sorted(files), sorted(empty_dirs)


def snapshot(name, worktree, archive_dir):
    ref = f'refs/archive/worktrees/{name}'
    kind = 'copy'
    base = None
    if (worktree / '.git').is_dir():
        kind = 'clone'
        git('fetch', '--quiet', '--no-tags', str(worktree),
            f'+refs/*:{ref}-refs/*', 'HEAD')
        base = git('rev-parse', 'FETCH_HEAD').decode().strip()
    elif (worktree / '.git').is_file():
        kind = 'linked_worktree'
        base = git('rev-parse', 'HEAD', cwd=worktree).decode().strip()

    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, 'GIT_DIR': str(REPO / '.git'), 'GIT_WORK_TREE': str(worktree),
               'GIT_INDEX_FILE': str(Path(tmp) / 'index')}
        if base:
            git('read-tree', base, env=env)
        git('add', '--all', '.', env=env, cwd=worktree)
        # Nested repositories become empty gitlinks; keep their real files in the tarball.
        gitlinks = [line.split('\t', 1)[1] for line in
                    git('ls-files', '-s', '-z', env=env, cwd=worktree).decode().split('\0')
                    if line.startswith('160000 ')]
        for path in gitlinks:
            git('rm', '--cached', '--quiet', path, env=env, cwd=worktree)
        tree = git('write-tree', env=env).decode().strip()
        listed = [p for p in git('ls-files', '-o', '-i', '--exclude-standard', '-z',
                                 env=env, cwd=worktree).decode().split('\0') if p]
    ignored = set()
    for path in listed + gitlinks:
        target = worktree / path
        if target.is_dir() and not target.is_symlink():
            ignored.update(str(Path(current, n).relative_to(worktree))
                           for current, _, names in os.walk(target) for n in names)
        else:
            ignored.add(path.rstrip('/'))
    ignored = sorted(ignored)
    existing = subprocess.run(['git', 'rev-parse', '--verify', '--quiet', f'{ref}^{{tree}}'],
                              cwd=REPO, capture_output=True, text=True).stdout.strip()
    if existing == tree:
        commit = git('rev-parse', ref).decode().strip()
    else:
        message = f'Archive snapshot of Amazon4U/.worktrees/{name} ({kind})\n'
        commit = git('commit-tree', tree, *(['-p', base] if base else []),
                     '-m', message).decode().strip()
        git('update-ref', ref, commit)

    files, empty_dirs = disk_entries(worktree)
    tarball = None
    if ignored or empty_dirs:
        tarball = archive_dir / f'{name}.ignored.tar.zst'
        listing = '\0'.join(ignored + empty_dirs).encode()
        subprocess.run(['tar', '--zstd', '-cf', str(tarball), '-C', str(worktree),
                        '--null', '--no-recursion', '-T', '-'], input=listing, check=True)
    manifest = {
        'name': name, 'kind': kind, 'ref': ref, 'commit': commit, 'base': base,
        'tracked_files': len(files) - len(ignored), 'ignored_files': len(ignored),
        'empty_directories': empty_dirs,
        'ignored_tarball': str(tarball) if tarball else None,
        'ignored_tarball_sha256': sha256(tarball) if tarball else None,
        'ignored_sha256': {p: sha256(worktree / p) for p in ignored
                           if (worktree / p).is_file() and not (worktree / p).is_symlink()},
        'ignored_symlinks': {p: os.readlink(worktree / p) for p in ignored
                             if (worktree / p).is_symlink()},
    }
    (archive_dir / f'{name}.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(manifest, worktree):
    """Every disk file must match the archived commit blob or the ignored manifest."""
    files, empty_dirs = disk_entries(worktree)
    tree = {}
    for line in git('ls-tree', '-r', '-z', manifest['commit']).decode().split('\0'):
        if line:
            meta, path = line.split('\t', 1)
            tree[path] = meta.split()[2]
    ignored = manifest['ignored_sha256']
    in_tarball = set(ignored) | set(manifest['ignored_symlinks'])
    archived = [f for f in files if f not in in_tarball]
    problems = [f for f in archived if f not in tree]
    problems += [p for p in tree if p not in set(archived)]
    if archived:
        hashed = git('hash-object', '--no-filters', '--stdin-paths',
                     data='\n'.join(str(worktree / f) for f in archived).encode()).decode().split()
        problems += [f for f, h in zip(archived, hashed) if tree.get(f) != h
                     and not (worktree / f).is_symlink()]
    problems += [f for f, h in ignored.items() if sha256(worktree / f) != h]
    if manifest['ignored_tarball']:
        members = set(subprocess.run(['tar', '--zstd', '-tf', manifest['ignored_tarball']],
                                     capture_output=True, check=True, text=True).stdout.split('\n'))
        members = {m.rstrip('/') for m in members}
        problems += [p for p in [*in_tarball, *empty_dirs] if p not in members]
    problems += sorted(set(empty_dirs) - set(manifest['empty_directories']))
    return problems


def remove(manifest, worktree):
    if manifest['kind'] == 'linked_worktree':
        git('worktree', 'remove', '--force', '--force', str(worktree))
    else:
        shutil.rmtree(worktree)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('names', nargs='+', help='folder names under Amazon4U/.worktrees')
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--remove', action='store_true', help='delete each folder after it verifies')
    args = parser.parse_args()
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in args.names:
        worktree = WORKTREES / name
        if not worktree.is_dir():
            print(f'SKIP {name}: not found')
            continue
        manifest = snapshot(name, worktree, args.archive_dir)
        problems = verify(manifest, worktree)
        if problems:
            failed = True
            print(f'FAIL {name}: {len(problems)} mismatches, e.g. {problems[:3]}; kept on disk')
            continue
        if args.remove:
            remove(manifest, worktree)
        print(f"OK {name}: {manifest['commit'][:12]} tracked={manifest['tracked_files']} "
              f"ignored={manifest['ignored_files']}{' removed' if args.remove else ''}")
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
