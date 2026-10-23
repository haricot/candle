#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODEL_DIR="${HDB2A_MODEL_DIR:-$ROOT/models/transcribepsy-eu-sub-candle}"
RUNTIME_DIR="${HDB2A_RUNTIME_DIR:-$ROOT/integration/parakeet-candle/runtime}"
REPORT_ROOT="${HDB2A_REPORT_ROOT:-$HOME/tmp/haricot-half-duplex/reports}"
usage() {
cat <<EOF
HDB2-A real Candle TranscribePsy EU-sub (experimental source port)
Usage:
  ./run-hdb2a.sh preflight
  ./run-hdb2a.sh bootstrap [--candle-root /path/to/haricot/candle]
  ./run-hdb2a.sh repair-runtime
  ./run-hdb2a.sh prepare-model /path/to/TranscribePsy-250M-EU-sub.nemo
  ./run-hdb2a.sh prepare-official-model /path/to/QVAC/model.safetensors /path/to/old/converted-model
  ./run-hdb2a.sh repair-model-config
  ./run-hdb2a.sh check-model-config
  ./run-hdb2a.sh build cpu|cuda
  ./run-hdb2a.sh transcribe /absolute/path/to/voice.wav [cpu|cuda]
  ./run-hdb2a.sh selftest

Uses pinned Candle 0.11 and a patched LOCAL checkout of pinned parakeet-candle.
Never invokes a C++ ASR or substitutes the NVIDIA teacher checkpoint.
Requires: Python 3; torch, safetensors, PyYAML, sentencepiece ONLY for NeMo conversion.
Official SafeTensors preparation uses the standard library and an existing HDB2-A config;
          Git, Rust/Cargo and ffmpeg; network access on the LOCAL machine.
EOF
}
case "${1:-help}" in
  preflight)
    echo 'HDB2A=PREPARED_NOT_QUALIFIED'
    echo 'CANDLE_PIN=f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3'
    echo 'PARAKEET_PIN=53c0d3065f9dd8de0356b25391e60ed3a471192b'
    for tool in git cargo python3 ffmpeg hf; do
      if command -v "$tool" >/dev/null 2>&1; then echo "$tool=FOUND"; else echo "$tool=MISSING"; fi
    done
    echo "MODEL_DIR=$MODEL_DIR"; echo "RUNTIME_DIR=$RUNTIME_DIR"
    ;;
  bootstrap)
    shift
    python3 scripts/prepare_candle_runtime.py --runtime-dir "$RUNTIME_DIR" "$@"
    ;;
  repair-runtime)
    python3 scripts/prepare_candle_runtime.py --runtime-dir "$RUNTIME_DIR" --repair-existing
    ;;
  prepare-model)
    if [[ $# -ne 2 ]];then usage;exit 2;fi
    python3 scripts/convert_nemo_hdb2a.py --nemo "$2" --output-dir "$MODEL_DIR"
    ;;
  prepare-official-model)
    if [[ $# -ne 3 ]];then usage;exit 2;fi
    python3 scripts/prepare_qvac_official.py --weights "$2" \
      --converted-model-dir "$3" --output-dir "$MODEL_DIR"
    ;;
  repair-model-config)
    python3 scripts/hdb2a_model_config.py repair --model-dir "$MODEL_DIR"
    ;;
  check-model-config)
    python3 scripts/hdb2a_model_config.py check --model-dir "$MODEL_DIR"
    ;;
  build)
    device="${2:-cpu}"
    if [[ "$device" != cpu && "$device" != cuda ]];then usage;exit 2;fi
    if [[ ! -f "$RUNTIME_DIR/Cargo.toml" ]];then echo 'HDB2A=NO_RUNTIME_RUN_BOOTSTRAP' >&2;exit 2;fi
    # Upstream parakeet-candle pins libc 0.2.184; Candle's zip/nix needs
    # libc >= 0.2.186. Update *only the standalone runtime lockfile*.
    python3 scripts/hdb2a_cargo_lock.py --manifest-path "$RUNTIME_DIR/Cargo.toml"
    args=(--release --manifest-path "$RUNTIME_DIR/Cargo.toml")
    if [[ "$device" == cuda ]];then args+=(--features cuda);fi
    cargo build "${args[@]}"
    echo "HDB2A=RUNTIME_BUILT_$device"
    ;;
  transcribe)
    if [[ $# -lt 2 || $# -gt 3 ]];then usage;exit 2;fi
    python3 scripts/run_candle_hdb2a.py --input "$2" --device "${3:-cpu}" \
      --model-dir "$MODEL_DIR" --runtime-dir "$RUNTIME_DIR" --report-root "$REPORT_ROOT"
    ;;
  selftest)
    python3 -m unittest discover -s tests/hdb2a -v
    ;;
  help|--help|-h) usage;;
  *) usage;exit 2;;
esac
