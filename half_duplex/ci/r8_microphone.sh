#!/usr/bin/env bash
# Invoke on the laptop to record one real intervention, retained locally.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HDB2C_PYTHON:?existing NeMo environment required}"
: "${HDB2A_REPORT_ROOT:?local report directory required}"
variant="${1:-cuda-cudnn}"
case "$variant" in cpu|cuda-cudnn) ;; *) echo 'usage: bash ci/r8_microphone.sh [cpu|cuda-cudnn]' >&2;exit 2;; esac
command -v pw-record
mkdir -p "$HDB2A_REPORT_ROOT"
recording_dir="$(mktemp -d "$HDB2A_REPORT_ROOT/r8-microphone-XXXXXX")"
echo 'Speak for 12 seconds. PCM stays in your local report directory.'
if timeout --signal=INT 12s pw-record --rate 16000 --channels 1 --format s16 "$recording_dir/input.wav";then
  :
else
  status=$?
  [[ "$status" == 124 || "$status" == 130 ]] || exit "$status"
fi
export HDB2C_NEMO_PYTHON="$HDB2C_PYTHON"
"$HDB2C_PYTHON" -m hdb2c.c3_r8 --wav "$recording_dir/input.wav" --variant "$variant" --keep-raw --require-numeric
