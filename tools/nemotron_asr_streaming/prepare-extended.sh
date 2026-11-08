#!/usr/bin/env bash
# Make a local integration worktree; never update or publish Extended itself.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 NEW_WORKTREE_DIRECTORY" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
candidate=$(git -C "$repo" rev-parse HEAD)
# Promoted SM61 backend from the user's successful run 39846015167.
extended=6b29b7f43a574fae32984b34b46de7c1d9b145eb
destination=$(realpath -m -- "$1")
if [[ -e "$destination" ]]; then
  echo "Choose a new worktree path: $destination" >&2
  exit 2
fi
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
if ! git -C "$repo" cat-file -e "$extended^{commit}" 2>/dev/null; then
  git -C "$repo" fetch origin "$extended"
fi
git -C "$repo" worktree add --detach "$destination" "$extended"
git -C "$destination" merge --squash "$candidate"
echo "Local Extended + Nemotron source ready: $destination"
echo "The integration is staged locally; the promoted branch has not moved."
echo "Run tools/nemotron_asr_streaming/qualify-local.sh from this worktree."
