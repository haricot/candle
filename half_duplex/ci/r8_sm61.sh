#!/usr/bin/env bash
# Run on the actual Pascal host; no automatic CPU fallback or model download.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HDB2A_RUNTIME_DIR:?existing bootstrapped runtime required}"
: "${HDB2A_MODEL_DIR:?R7B converted model required}"
: "${HDB2C_NEMO:?original pinned .nemo checkpoint required}"
: "${HDB2C_PYTHON:?existing isolated NeMo Python required}"
: "${RUNNER_TEMP:?temporary campaign directory required}"
command -v nvcc
command -v nvidia-smi
compute="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d '[:space:]')"
test "$compute" = '6.1' || { echo "SM61_REQUIRED_FOUND=$compute" >&2; exit 3; }
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv
export CUDA_COMPUTE_CAP=61
export HDB2C_NEMO_PYTHON="$HDB2C_PYTHON"
export HDB2A_REPORT_ROOT="$RUNNER_TEMP/halfduplex-r8-cuda-reports"
"$HDB2C_PYTHON" -c 'import torch,nemo,datasets,soundfile; assert torch.version.cuda is None, "NeMo reference must stay CPU"'
"$HDB2C_PYTHON" ci/cameo_subset.py --out "$RUNNER_TEMP/r8-cameo" --per-emotion 1
"$HDB2C_PYTHON" -m hdb2c.c3_r8 --corpus "$RUNNER_TEMP/r8-cameo/manifest.json" --variant cuda-cudnn --require-numeric
