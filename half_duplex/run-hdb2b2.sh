#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUNTIME_DIR="${HDB2A_RUNTIME_DIR:-$ROOT/integration/parakeet-candle/runtime}"
MODEL_DIR="${HDB2A_MODEL_DIR:-$ROOT/models/transcribepsy-eu-sub-candle}"
REPORT_ROOT="${HDB2A_REPORT_ROOT:-$HOME/tmp/haricot-half-duplex/reports}"
usage(){
cat <<'EOF'
HDB2-B2: final relative latency qualification on GTX 1080 Max-Q sm_61.
  ./run-hdb2b2.sh selftest
  ./run-hdb2b2.sh preflight
  ./run-hdb2b2.sh repair-runtime
  ./run-hdb2b2.sh build cpu|cuda|cuda-cudnn
  ./run-hdb2b2.sh bench /absolute/path/audio.wav cpu|cuda|cuda-cudnn [runs=10] [warmup=3] [profile_runs=3]
  ./run-hdb2b2.sh compare3 /path/cpu/report.json /path/cuda/report.json /path/cuda-cudnn/report.json
  ./run-hdb2b2.sh qualify /absolute/path/audio.wav

Non-destructive reuse: model files and patched standalone runtime from HDB2-A/B.
Separate binaries: runtime/target/hdb2b2-{cpu,cuda,cuda-cudnn}/release/parakeet-candle.
No CPU fallback for GPU, no synthetic transcript, no kernel-only timing masquerading as end-to-end.
EOF
}
case "${1:-help}" in
  selftest)
    python3 -m unittest discover -s tests/hdb2a -v
    python3 -m unittest discover -s tests/hdb2b -v
    python3 -m unittest discover -s tests/hdb2b2 -v
    ;;
  preflight)
    echo "HDB2B2_RUNTIME_DIR=$RUNTIME_DIR"
    echo "HDB2B2_MODEL_DIR=$MODEL_DIR"
    echo "HDB2B2_REPORT_ROOT=$REPORT_ROOT"
    for tool in cargo rustc nvcc nvidia-smi ffmpeg python3; do
      if command -v "$tool" >/dev/null 2>&1;then echo "$tool=FOUND";else echo "$tool=MISSING";fi
    done
    if command -v nvidia-smi >/dev/null 2>&1;then
      nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader -i 0 || true
    fi
    [[ -f "$MODEL_DIR/model.safetensors" ]] && echo "MODEL=LOCAL_SAFETENSORS_PRESENT" || echo "MODEL=MISSING"
    ;;
  repair-runtime)
    python3 scripts/hdb2b_runtime_patch.py --runtime-dir "$RUNTIME_DIR"
    python3 scripts/hdb2b2_runtime_patch.py --runtime-dir "$RUNTIME_DIR"
    ;;
  build)
    [[ $# -eq 2 ]] || { usage;exit 2;}
    python3 scripts/hdb2b2_driver.py build --variant "$2" --runtime-dir "$RUNTIME_DIR" \
      --project-root "$ROOT" --report-root "$REPORT_ROOT"
    ;;
  bench)
    [[ $# -ge 3 && $# -le 6 ]] || { usage;exit 2;}
    python3 scripts/hdb2b2_driver.py bench --input "$2" --variant "$3" \
      --runs "${4:-10}" --warmup "${5:-3}" --profile-runs "${6:-3}" \
      --model-dir "$MODEL_DIR" --runtime-dir "$RUNTIME_DIR" --report-root "$REPORT_ROOT"
    ;;
  compare3)
    [[ $# -eq 4 ]] || { usage;exit 2;}
    python3 scripts/hdb2b2_driver.py compare3 \
      --cpu "$2" --cuda "$3" --cuda-cudnn "$4" --report-root "$REPORT_ROOT"
    ;;
  qualify)
    [[ $# -eq 2 ]] || { usage;exit 2;}
    INPUT="$2"
    [[ -f "$INPUT" ]] || { echo "HDB2B2=NO_INPUT_AUDIO $INPUT" >&2;exit 2;}
    ./run-hdb2b2.sh selftest
    ./run-hdb2b2.sh repair-runtime
    declare -A REPORTS
    for variant in cpu cuda cuda-cudnn; do
      ./run-hdb2b2.sh build "$variant"
      logfile="$(mktemp)"
      # With pipefail a failed benchmark never turns into a comparison PASS.
      ./run-hdb2b2.sh bench "$INPUT" "$variant" 10 3 3 | tee "$logfile"
      report="$(sed -n 's/^HDB2B2_REPORT=//p' "$logfile" | tail -n 1)"
      rm -f "$logfile"
      [[ -n "$report" && -f "$report" ]] || { echo 'HDB2B2=REPORT_MISSING' >&2;exit 1;}
      REPORTS[$variant]="$report"
    done
    ./run-hdb2b2.sh compare3 "${REPORTS[cpu]}" "${REPORTS[cuda]}" "${REPORTS[cuda-cudnn]}"
    ;;
  help|--help|-h) usage;;
  *)usage;exit 2;;
esac
