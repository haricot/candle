#!/usr/bin/env bash
# Invoke on the laptop to record one real intervention, retained locally.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${HDB2C_PYTHON:-}" ]];then
  HDB2C_PYTHON="${HDB2C_NEMO_PYTHON:-}"
  if [[ -z "$HDB2C_PYTHON" && -n "${HDB2C_ENV_DIR:-}" ]];then HDB2C_PYTHON="$HDB2C_ENV_DIR/bin/python";fi
  if [[ -z "$HDB2C_PYTHON" ]];then
    for candidate in "$PWD/.venv-hdb2c-py313/bin/python" \
      "$HOME/tmp/haricot-half-duplex/.venv-hdb2c-py313/bin/python" \
      "$HOME/tmp/haricot-half-duplex-hdb2c-v0.6.1/.venv-hdb2c-py313/bin/python";do
      if [[ -x "$candidate" ]];then HDB2C_PYTHON="$candidate";break;fi
    done
  fi
fi
export HDB2C_PYTHON="${HDB2C_PYTHON:-$PWD/.venv-hdb2c-py313/bin/python}"
export HDB2C_NEMO_PYTHON="$HDB2C_PYTHON"
variant="${1:-cuda-cudnn}"
case "$variant" in cpu|cuda-cudnn) ;; *) echo 'usage: bash ci/r8_microphone.sh [cpu|cuda-cudnn]' >&2;exit 2;; esac
module="${HDB2C_MICROPHONE_MODULE:-hdb2c.c3_r8}"
case "$module" in hdb2c.c3_r8|hdb2c.c3_r9_text) ;; *) echo 'unsupported microphone qualification module' >&2;exit 2;; esac
test -x "$HDB2C_PYTHON" || { echo "NeMo Python missing: $HDB2C_PYTHON; configure HDB2C_PYTHON or run ./run-hdb2c.sh install-nemo" >&2;exit 3; }
export HDB2A_REPORT_ROOT="${HDB2A_REPORT_ROOT:-$("$HDB2C_PYTHON" -c 'from hdb2c.cli import settings; print(settings()["reports"])')}"
"$HDB2C_PYTHON" - <<'PY'
import json
from hdb2c.cli import settings
s=settings()
for p in (s['nemo'],s['model']/'config.json',s['model']/'model.safetensors',s['runtime']/'src/parakeet/model.rs'):
    if not p.is_file():raise SystemExit('Microphone prerequisite missing: '+str(p))
import torch,nemo.collections.asr
assert torch.__version__=='2.12.1+cpu' and torch.version.cuda is None, 'Original NeMo reference must use the qualified CPU environment'
print('MICROPHONE_MODEL_AND_NEMO_PREFLIGHT=PASS',flush=True)
PY
command -v pw-record
command -v ffmpeg
if [[ "$variant" == cuda-cudnn ]];then
  command -v nvcc
  state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/haricot-sm61-runner"
  mkdir -p "$state_dir"
  exec {stt_gpu_lock}>"$state_dir/gpu.lock"
  flock -n "$stt_gpu_lock" || { echo 'GPU qualification is already running; wait for its terminal to finish.' >&2;exit 75; }
fi
mkdir -p "$HDB2A_REPORT_ROOT"
recording_dir="$(mktemp -d "$HDB2A_REPORT_ROOT/r8-microphone-XXXXXX")"
echo 'Speak for 12 seconds. PCM stays in your local report directory.'
if timeout --signal=INT 12s pw-record --rate 16000 --channels 1 --format s16 "$recording_dir/input.wav";then
  :
else
  status=$?
  [[ "$status" == 124 || "$status" == 130 ]] || exit "$status"
fi
"$HDB2C_PYTHON" -m "$module" --wav "$recording_dir/input.wav" --variant "$variant" --keep-raw --require-numeric
