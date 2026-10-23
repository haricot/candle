#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENDPOINT_PYTHON:-python3}"
CACHE="${ENDPOINT_REFERENCE_CACHE:-$ROOT/reference-data/cache}"
REPORT="${ENDPOINT_REPORT_ROOT:-$ROOT/reports/qualification-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$CACHE" "$REPORT"
export HF_HUB_DISABLE_TELEMETRY=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_*.py'
cargo fmt --manifest-path "$ROOT/Cargo.toml" -- --check
cargo test --release --locked --manifest-path "$ROOT/Cargo.toml" --lib
cargo build --release --locked --manifest-path "$ROOT/Cargo.toml" --bins
"$PYTHON" "$ROOT/tools/convert.py" --cache "$CACHE" --output "$REPORT/models" > "$REPORT/conversion.log"
"$PYTHON" "$ROOT/tools/qualify.py" --models "$REPORT/models" --cache "$CACHE" \
  --binary "${CARGO_TARGET_DIR:-$ROOT/target}/release/probe" --output "$REPORT/cases"
cp "$REPORT/cases/qualification.json" "$REPORT/qualification.json"
echo "ENDPOINT_CPU_REPORT=$REPORT/qualification.json"
