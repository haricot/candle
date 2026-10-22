#!/usr/bin/env bash
# Runtime launcher: invokes only the resident Rust executable, no interpreter.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/integration/current-binary" && -f "$ROOT/integration/current-models" ]] || {
  echo 'Build first with voice_endpointing/install-native.sh' >&2; exit 2;
}
IFS= read -r BINARY < "$ROOT/integration/current-binary"
IFS= read -r MODELS < "$ROOT/integration/current-models"
[[ -x "$BINARY" ]] || { echo "Missing native binary: $BINARY" >&2; exit 2; }
export HDB2C6_MASK_MODE=both
exec "$BINARY" --models "$MODELS" \
  --asr-model-dir "${HDB_MODEL_DIR:-$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle}" \
  --asr-device cuda-cudnn \
  --report-root "${HDB_NATIVE_REPORT_ROOT:-$HOME/tmp/haricot-half-duplex/reports/native}" "$@"
