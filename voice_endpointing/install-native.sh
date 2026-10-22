#!/usr/bin/env bash
# One-time build/conversion. run-native.sh subsequently needs no Python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HDB_PROMOTED_ROOT:?Set HDB_PROMOTED_ROOT to the validated Extended checkout}"
[[ -f "$HDB_PROMOTED_ROOT/half_duplex/ci/extended_build.py" ]] || exit 2
EXPECTED_CANDLE=6b29b7f43a574fae32984b34b46de7c1d9b145eb
[[ "$(git -C "$HDB_PROMOTED_ROOT" rev-parse HEAD)" == "$EXPECTED_CANDLE" ]] || {
  echo "Expected the validated Extended commit $EXPECTED_CANDLE" >&2; exit 2;
}
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}" CUDA_ROOT="${CUDA_ROOT:-/opt/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-14}"
export CUDA_COMPUTE_CAP=61 RUSTUP_TOOLCHAIN=1.99.0
rustup toolchain install 1.99.0 --profile minimal
TOOLS="$HOME/tmp/half-duplex-tools"
python3 "$HDB_PROMOTED_ROOT/half_duplex/ci/bootstrap_local_uv.py" "$TOOLS"
if [[ ! -x "$ROOT/.venv-convert/bin/python" ]]; then
  "$TOOLS/uv" venv --python 3.13 "$ROOT/.venv-convert"
fi
"$TOOLS/uv" pip install --python "$ROOT/.venv-convert/bin/python" -r "$ROOT/requirements-convert.txt"
MODELS="${ENDPOINT_MODELS_DIR:-$HOME/tmp/haricot-half-duplex/models/native-endpointing-v1}"
if [[ ! -d "$MODELS" ]]; then
  "$ROOT/.venv-convert/bin/python" "$ROOT/tools/convert.py" \
    --cache "$HOME/tmp/haricot-half-duplex/models/endpoint-conversion-cache" --output "$MODELS"
fi
mkdir -p "$ROOT/integration"
BUILD_PARENT="$(mktemp -d "$ROOT/integration/resident-XXXXXX")"
BUILD="$BUILD_PARENT/native"
"$ROOT/.venv-convert/bin/python" "$ROOT/tools/prepare_half_duplex.py" \
  --half-root "$HDB_PROMOTED_ROOT/half_duplex" --candle-root "$HDB_PROMOTED_ROOT" --output "$BUILD"
export CARGO_TARGET_DIR="$ROOT/integration/cargo-target"
cargo build --release --locked --manifest-path "$BUILD/r9/Cargo.toml" \
  --bin hdb1_native --features "cuda cudnn"
printf '%s\n' "$CARGO_TARGET_DIR/release/hdb1_native" > "$ROOT/integration/current-binary"
printf '%s\n' "$MODELS" > "$ROOT/integration/current-models"
echo 'NATIVE_HALF_DUPLEX_BUILD=READY'
echo 'Launch: bash voice_endpointing/run-native.sh --mic --max-turns 8'
