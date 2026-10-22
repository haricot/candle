#!/usr/bin/env bash
# HDB2-C2 reproducible installer: complete isolated Rust source + explicit build.
# Never writes to the production runtime or installs Python or models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
usage() {
cat <<'TXT'
HDB2-C2 v0.6.6 — install_update.sh
  ./install_update.sh                                  # selftest + isolated CPU Rust build
  ./install_update.sh --report /path/to/C1_PROBES_DIR   # build + real same-WAV A/B + numerical probes
  ./install_update.sh --prepare-only                   # prepare isolated source, no build
  ./install_update.sh --cuda-cudnn --report /path/...  # optional Pascal CUDA qualification
  ./install_update.sh --skip-tests                     # only skip package-level Python selftests
  ./install_update.sh --help

Requires: pinned local HDB2A parakeet-candle runtime source, Rust cargo,
model config+weights, existing Python 3.13 NeMo environment with numpy.
Does NOT download any model, reinstall NeMo/PyTorch, or overwrite the live ASR.
TXT
}
REPORT=""
MODE=build
VARIANT=cpu
SKIP_TESTS=0
while (($#));do
  case "$1" in
    --report) [[ $# -ge 2 ]] || { echo 'Missing --report value' >&2; exit 2; }; REPORT="$2"; MODE=compare; shift 2;;
    --prepare-only) MODE=prepare;shift;;
    --cuda-cudnn) VARIANT=cuda-cudnn;shift;;
    --skip-tests) SKIP_TESTS=1;shift;;
    --help|-h) usage;exit 0;;
    *) echo "Unknown argument: $1" >&2;usage;exit 2;;
  esac
done
if [[ "$MODE" == prepare && -n "$REPORT" ]];then echo 'Incompatible --prepare-only and --report' >&2;exit 2;fi
for cmd in python3 cargo git; do
  command -v "$cmd" >/dev/null || { echo "HDB2C2_MISSING_TOOL=$cmd" >&2;exit 3; }
done
PY="${HDB2C_PYTHON:-${HDB2C_NEMO_PYTHON:-}}"
if [[ -z "$PY" ]]; then
  for path in \
    "$ROOT/.venv-hdb2c-py313/bin/python" \
    "$HOME/tmp/haricot-half-duplex/.venv-hdb2c-py313/bin/python" \
    "$(dirname "$ROOT")/haricot-half-duplex-hdb2c-v0.6.1/.venv-hdb2c-py313/bin/python"; do
    if [[ -x "$path" ]];then PY="$path";break;fi
  done
fi
[[ -n "$PY" && -x "$PY" ]] || { echo 'HDB2C2_REQUIRES_EXISTING_PY313: export HDB2C_PYTHON=/path/to/.venv-hdb2c-py313/bin/python' >&2;exit 3; }
"$PY" -c 'import sys,numpy; assert sys.version_info[:2] == (3,13), sys.version'
RUNTIME="${HDB2A_RUNTIME_DIR:-$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime}"
MODEL="${HDB2A_MODEL_DIR:-$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle}"
[[ -f "$RUNTIME/Cargo.toml" && -f "$RUNTIME/src/parakeet/audio.rs" ]] || { echo "HDB2C2_RUNTIME_MISSING=$RUNTIME" >&2;exit 3; }
[[ -f "$MODEL/config.json" && -f "$MODEL/model.safetensors" ]] || { echo "HDB2C2_MODEL_MISSING=$MODEL" >&2;exit 3; }
export HDB2A_RUNTIME_DIR="$RUNTIME" HDB2A_MODEL_DIR="$MODEL"
if [[ "$SKIP_TESTS" == 0 ]]; then
  echo 'HDB2C2_PHASE=PYTHON_SELFTESTS'
  "$PY" -m unittest discover -s tests/hdb2c -v
fi
ARGS=("$MODE" --variant "$VARIANT")
if [[ "$MODE" == compare ]];then
  [[ -f "$REPORT/nemo_report.json" && -f "$REPORT/nemo_mel.npz" ]] || { echo "HDB2C2_NOT_A_PROBE_REPORT=$REPORT" >&2;exit 3; }
  ARGS+=(--report-dir "$REPORT")
fi
mkdir -p "$ROOT/.hdb2c2-logs"
LOG="$ROOT/.hdb2c2-logs/build-${VARIANT}.log"
echo "HDB2C2_PYTHON=$PY"
echo "HDB2C2_MODE=$MODE HDB2C2_VARIANT=$VARIANT"
echo "HDB2C2_LOG=$LOG"
set +e
"$PY" -m hdb2c.c2 "${ARGS[@]}" > >(tee "$LOG") 2> >(tee -a "$LOG" >&2)
status=$?
set -e
if [[ "$status" != 0 ]];then
  echo "HDB2C2_BUILD_OR_AB=FAILED rc=$status logfile=$LOG" >&2
  exit "$status"
fi
if [[ "$MODE" == compare ]];then
  echo 'HDB2C2_INSTALL_UPDATE=AB_EXECUTED_CHECK_NUMERIC_REPORT'
elif [[ "$MODE" == build ]];then
  echo 'HDB2C2_INSTALL_UPDATE=BUILT_RUNTIME_NOT_YET_QUALIFIED'
else
  echo 'HDB2C2_INSTALL_UPDATE=SOURCE_PREPARED_NOT_YET_COMPILED'
fi
