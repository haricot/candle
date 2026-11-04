#!/usr/bin/env bash
# Verification gate for the mathematically specified standalone contracts.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
cargo test --manifest-path Cargo.toml
if ! command -v lake >/dev/null 2>&1; then
  echo "LEAN4_CHECK=FAIL reason=lake_missing (install elan/Lean toolchain)" >&2
  exit 1
fi
# Never accept admitted proofs in the maintained model.
if grep -RnE '(^|[[:space:]])(sorry|sorryAx|axiom)([[:space:]]|$)' lean/HaricotLeanContracts --include='*.lean'; then
  echo "LEAN4_CHECK=FAIL reason=admitted_proof" >&2
  exit 1
fi
(cd lean && lake build)
echo "LEAN4_CHECK=PASS"
