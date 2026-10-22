#!/usr/bin/env bash
# Shared local/hosted harness for the isolated first LayerNorm, never full STT.
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-cpu}"
case "$backend" in cpu|cuda) ;; *) echo 'usage: r12_layernorm.sh cpu|cuda' >&2;exit 2;; esac
export RUNNER_TEMP="${RUNNER_TEMP:-${XDG_CACHE_HOME:-$HOME/.cache}/haricot-stt-layernorm}"
mkdir -p "$RUNNER_TEMP"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNNER_TEMP/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RUNNER_TEMP/python}"
export HF_HOME="${HF_HOME:-$RUNNER_TEMP/huggingface}"
if ! command -v uv >/dev/null; then
  python3 ci/bootstrap_local_uv.py "$RUNNER_TEMP/tools"
  export PATH="$RUNNER_TEMP/tools:$PATH"
fi
export HDB2C_ENV_DIR="${HDB2C_ENV_DIR:-$RUNNER_TEMP/nemo-py313}"
export HDB2C_PYTHON="${HDB2C_PYTHON:-$HDB2C_ENV_DIR/bin/python}"
export HDB2C_NEMO_PYTHON="$HDB2C_PYTHON"
./run-hdb2c.sh install-nemo
uv pip install --no-config --python "$HDB2C_PYTHON" --default-index https://pypi.org/simple \
  'datasets==3.6.0' soundfile huggingface_hub
"$HDB2C_PYTHON" ci/cameo_subset.py --out "$RUNNER_TEMP/cameo" --per-emotion 1
if [[ -z "${HDB2C_NEMO:-}" ]];then
  mkdir -p "$RUNNER_TEMP/checkpoint"
  "$HDB2C_PYTHON" - "$RUNNER_TEMP/checkpoint" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download('qvac/TranscribePsy-250M-EU-sub',allow_patterns=['*.nemo'],local_dir=sys.argv[1])
PY
  mapfile -d '' checkpoints < <(find "$RUNNER_TEMP/checkpoint" -type f -name '*.nemo' -print0)
  [[ "${#checkpoints[@]}" == 1 ]] || { echo 'R12_NO_UNIQUE_ORIGINAL_CHECKPOINT' >&2;exit 3; }
  export HDB2C_NEMO="${checkpoints[0]}"
fi
run_dir="${HDB2C_R12_RUN_DIR:-$RUNNER_TEMP/runs/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$run_dir"
# Publish metrics even when qualification fails. Raw operands are local-only.
status=0
"$HDB2C_PYTHON" -m hdb2c.c3_r12_layernorm --corpus "$RUNNER_TEMP/cameo/manifest.json" \
  --nemo "$HDB2C_NEMO" --backend "$backend" --output "$run_dir/result" || status=$?
if [[ -n "${GITHUB_WORKSPACE:-}" && -s "$run_dir/result/report.json" ]];then
  mkdir -p "$GITHUB_WORKSPACE/halfduplex-ci-published"
  cp "$run_dir/result/report.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/r12-layernorm-$backend.json"
fi
echo "R12_LOCAL_REPORT=$run_dir/result/report.json"
exit "$status"
