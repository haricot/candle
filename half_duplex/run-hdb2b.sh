#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUNTIME_DIR="${HDB2A_RUNTIME_DIR:-$ROOT/integration/parakeet-candle/runtime}"
MODEL_DIR="${HDB2A_MODEL_DIR:-$ROOT/models/transcribepsy-eu-sub-candle}"
REPORT_ROOT="${HDB2A_REPORT_ROOT:-$HOME/tmp/haricot-half-duplex/reports}"
usage(){
cat <<EOF
HDB2-B / Candle Pascal sm_61 — first measurable CUDA FP32 qualification.
  ./run-hdb2b.sh selftest
  ./run-hdb2b.sh preflight
  ./run-hdb2b.sh repair-runtime
  ./run-hdb2b.sh build cpu|cuda|cuda-cudnn
  ./run-hdb2b.sh bench /absolute/path/to/audio.wav cpu|cuda|cuda-cudnn [runs=3] [warmup=1]
  ./run-hdb2b.sh compare /path/to/first-report.json /path/to/second-report.json
  # Supports cpu vs cuda, cpu vs cuda-cudnn, and cuda vs cuda-cudnn
  ./run-hdb2a.sh transcribe /path/to/audio.wav cpu     # existing HDB2A baseline untouched

Requires an already bootstrapped and working HDB2-A runtime and converted model.
All variants build into separate target directories. CUDA_COMPUTE_CAP=61 forced.
No FP16/TF32, no speculative timings, no fallback-to-CPU accepted as GPU.
EOF
}
cmd="${1:-help}"
case "$cmd" in
  preflight)
    echo "HDB2B_MODEL_DIR=$MODEL_DIR"
    echo "HDB2B_RUNTIME_DIR=$RUNTIME_DIR"
    echo 'HDB2B_CUDA_TARGET=sm_61'
    echo 'HDB2B_DTYPE=f32'
    for tool in cargo rustc nvcc nvidia-smi ffmpeg python3; do
      if command -v "$tool" >/dev/null 2>&1; then echo "$tool=FOUND"; else echo "$tool=MISSING";fi
    done
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader -i 0 || true
    fi
    ;;
  repair-runtime)
    python3 scripts/hdb2b_runtime_patch.py --runtime-dir "$RUNTIME_DIR"
    ;;
  build)
    [[ $# == 2 ]] || { usage;exit 2;}
    python3 scripts/hdb2b_driver.py build --variant "$2" --runtime-dir "$RUNTIME_DIR" --project-root "$ROOT" --report-root "$REPORT_ROOT"
    ;;
  bench)
    [[ $# -ge 3 && $# -le 5 ]] || { usage;exit 2;}
    python3 scripts/hdb2b_driver.py benchmark --input "$2" --variant "$3" \
      --runs "${4:-3}" --warmup "${5:-1}" --model-dir "$MODEL_DIR" \
      --runtime-dir "$RUNTIME_DIR" --report-root "$REPORT_ROOT"
    ;;
  compare)
    [[ $# == 3 ]] || { usage;exit 2;}
    python3 scripts/hdb2b_driver.py compare --first "$2" --second "$3" --report-root "$REPORT_ROOT"
    ;;
  selftest)
    python3 -m unittest discover -s tests/hdb2b -v
    ;;
  help|--help|-h) usage;;
  *) usage; exit 2;;
esac
