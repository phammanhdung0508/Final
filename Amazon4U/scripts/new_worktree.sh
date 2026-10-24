#!/usr/bin/env bash
# Create Amazon4U/.worktrees/<name> without AGENTS.md copies.
# The files stay tracked, committed and pushable; they are only not written to disk,
# so agents in the worktree load the current main Amazon4U/AGENTS.md instead.
#
# Usage: scripts/new_worktree.sh <name> [<commit-ish>]      # detached at a commit (default HEAD)
#        scripts/new_worktree.sh <name> -b <new-branch> [<commit-ish>]
set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
name=${1:?usage: new_worktree.sh <name> [-b <branch>] [<commit-ish>]}
shift
path="$repo/Amazon4U/.worktrees/$name"
[ -e "$path" ] && { echo "refusing: $path already exists" >&2; exit 1; }

if [ "${1:-}" = "-b" ]; then
  git -C "$repo" worktree add --no-checkout -b "${2:?branch name required}" "$path" "${3:-HEAD}"
else
  git -C "$repo" worktree add --no-checkout --detach "$path" "${1:-HEAD}"
fi
git -C "$path" sparse-checkout set --no-cone '/*' '!AGENTS.md' '!**/AGENTS.md'
git -C "$path" checkout --quiet

left=$(find "$path" -name AGENTS.md -not -path '*/.git/*' | wc -l)
echo "created $path ($left AGENTS.md files on disk)"
