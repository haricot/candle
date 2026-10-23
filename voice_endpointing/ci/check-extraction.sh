#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENDPOINT_PYTHON:-python3}"
CACHE="${ENDPOINT_REFERENCE_CACHE:-$ROOT/reference-data/cache}"
REPORT="${ENDPOINT_REPORT_ROOT:-$ROOT/reports/extraction-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$CACHE" "$REPORT"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_*.py'
cargo fmt --manifest-path "$ROOT/Cargo.toml" -- --check
rustfmt --edition 2021 --check "$ROOT/../candle-transformers/src/models/fireredvad.rs" \
  "$ROOT/../candle-transformers/src/models/smart_turn.rs" \
  "$ROOT/../candle-transformers/src/models/fireredvad/audio.rs" \
  "$ROOT/../candle-transformers/src/models/smart_turn/audio.rs"
cargo test --release --locked --manifest-path "$ROOT/Cargo.toml" --lib
cargo build --release --locked --manifest-path "$ROOT/Cargo.toml" --bins
"$PYTHON" "$ROOT/tools/prepare_baseline.py" --candle-root "$ROOT/.." --output "$REPORT/baseline-app"
CARGO_TARGET_DIR="$REPORT/baseline-target" cargo build --release --locked \
  --manifest-path "$REPORT/baseline-app/Cargo.toml" --bins
"$PYTHON" "$ROOT/tools/convert.py" --cache "$CACHE" --output "$REPORT/models" > "$REPORT/conversion.log"
"$PYTHON" "$ROOT/tools/check_extraction.py" \
  --baseline-binary "$REPORT/baseline-target/release/probe" \
  --candidate-binary "${CARGO_TARGET_DIR:-$ROOT/target}/release/probe" \
  --models "$REPORT/models" --cache "$CACHE" --output "$REPORT/extraction"
cp "$REPORT/extraction/extraction-compatibility.json" "$REPORT/extraction-compatibility.json"
