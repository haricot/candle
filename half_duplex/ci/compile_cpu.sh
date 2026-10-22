#!/usr/bin/env bash
# Real Rust/Candle C2 compile and unit tests; NO checkpoint inference claim.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RUNNER_TEMP:?Requires CI runner temp}"
export HDB2A_RUNTIME_DIR="$RUNNER_TEMP/halfduplex-pinned-parakeet"
export HDB2A_MODEL_DIR="$RUNNER_TEMP/halfduplex-buildonly-config"
mkdir -p "$HDB2A_MODEL_DIR"
printf '{"preprocessor":{"normalize":"NA","n_fft":512,"features":128}}\n' > "$HDB2A_MODEL_DIR/config.json"
python3 scripts/prepare_candle_runtime.py --runtime-dir "$HDB2A_RUNTIME_DIR"
echo 'HALF_DUPLEX_C2_CPU=BUILDING_TRUE_RUST_SOURCES_NO_REAL_CHECKPOINT'
python3 -m hdb2c.c2 build --variant cpu --candidate "$RUNNER_TEMP/halfduplex-rust-c2"
echo 'HALF_DUPLEX_C2_CPU=RUST_BUILD_AND_TEST_PASS_NO_ASR_INFERENCE'
