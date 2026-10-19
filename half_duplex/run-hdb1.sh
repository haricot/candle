#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
usage(){
cat <<'TXT'
HDB1 real half-duplex FireRedVAD ONNX + Smart Turn v3.2 ONNX + Candle
  ./run-hdb1.sh selftest
  ./run-hdb1.sh install-python   # isolated .venv-hdb1 (not PyTorch)
  ./run-hdb1.sh download-models  # ~10MB, FireRed Git + SmartTurn HF
  ./run-hdb1.sh preflight
  ./run-hdb1.sh file /home/np/test_micro_2.wav [--variant cuda-cudnn]
  ./run-hdb1.sh mic [--max-turns 3] [--endpoint-hold-ms 1200]
  ./run-hdb1.sh file /home/np/test_micro_2.wav --no-asr

Env: HDB2A_RUNTIME_DIR, HDB2A_MODEL_DIR, HDB2A_REPORT_ROOT, HDB1_MODEL_ROOT
No simulation in production. --no-asr evaluates real VAD+SmartTurn only.
TXT
}
case "${1:-help}" in
  install-python)
    python3 -m venv .venv-hdb1
    .venv-hdb1/bin/python -m pip install -r requirements-hdb1.txt
    ;;
  download-models)
    PYTHON="${HDB1_PYTHON:-$ROOT/.venv-hdb1/bin/python}"
    [[ -x "$PYTHON" ]] || PYTHON=python3
    "$PYTHON" -m hdb1.bootstrap --model-root "${HDB1_MODEL_ROOT:-$HOME/tmp/haricot-half-duplex/models/hdb1}"
    ;;
  selftest)
    python3 -m unittest discover -s tests/hdb1 -v
    ;;
  preflight)
    PYTHON="${HDB1_PYTHON:-$ROOT/.venv-hdb1/bin/python}"
    [[ -x "$PYTHON" ]] || PYTHON=python3
    "$PYTHON" -m hdb1.cli preflight --model-root "${HDB1_MODEL_ROOT:-$HOME/tmp/haricot-half-duplex/models/hdb1}" "${@:2}"
    ;;
  file)
    [[ $# -ge 2 ]] || { usage;exit 2; }
    PYTHON="${HDB1_PYTHON:-$ROOT/.venv-hdb1/bin/python}"
    [[ -x "$PYTHON" ]] || PYTHON=python3
    FILE="$2";shift 2
    "$PYTHON" -m hdb1.cli file --input "$FILE" --model-root "${HDB1_MODEL_ROOT:-$HOME/tmp/haricot-half-duplex/models/hdb1}" "$@"
    ;;
  mic)
    PYTHON="${HDB1_PYTHON:-$ROOT/.venv-hdb1/bin/python}"
    [[ -x "$PYTHON" ]] || PYTHON=python3
    shift
    "$PYTHON" -m hdb1.cli mic --model-root "${HDB1_MODEL_ROOT:-$HOME/tmp/haricot-half-duplex/models/hdb1}" "$@"
    ;;
  help|--help|-h) usage;;
  *) usage;exit 2;;
esac
