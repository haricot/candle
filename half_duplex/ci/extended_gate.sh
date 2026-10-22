#!/usr/bin/env bash
# Extended owns queuing; its run-specific unified runner owns physical GPU access.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RUNNER_TEMP:?}"
stage="${1:?contracts|build|cpu|gpu}"
export HDB2A_CANDLE_ROOT="$(cd .. && pwd)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4
case "$stage" in
  contracts)
    for suite in hdb1 hdb2a hdb2b hdb2b2 hdb2c;do
      python3 -m unittest discover -s "tests/$suite" -v
    done
    python3 ci/test_cameo_subset.py
    ;;
  build)
    python3 ci/extended_build.py --backend cpu --candle-root "$HDB2A_CANDLE_ROOT" \
      --output "$RUNNER_TEMP/halfduplex-extended-cpu-build"
    ;;
  cpu|gpu)
    if ! command -v uv >/dev/null;then
      python3 ci/bootstrap_local_uv.py "$RUNNER_TEMP/halfduplex-tools"
      export PATH="$RUNNER_TEMP/halfduplex-tools:$PATH"
    fi
    if [[ "$stage" == cpu ]];then
      export HDB2C_R13_CPU_FULL=1 HDB2C_R8_VARIANT=cpu
      export HDB2C_R14_FP64_REFERENCE=1 HDB2C_R14_CUDA=0
    else
      test "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d '[:space:]')" = 6.1
      export CUDA_COMPUTE_CAP=61 HDB2C_R8_CORPUS=1 HDB2C_R9_TEXT=1
      export HDB2C_R8_VARIANT=cuda-cudnn HDB2C_R13_CPU_FULL=0
      export HDB2C_R14_CUDA=1 HDB2C_R14_FP64_REFERENCE=0
      echo 'HALF_DUPLEX_GPU=REAL_NATIVE_CUDA_CUDNN_CPU_NEMO_REFERENCE'
      echo 'HALF_DUPLEX_GPU_SCOPE=R9_NATIVE_GPU_NOT_THE_CPU_ONLY_R13_KERNELS'
      echo 'HALF_DUPLEX_GPU_AUDIT=FP64_OPERATORS_14_REGRESSION_28_ADDITIONAL_CLIPS'
    fi
    # Original full numerical gates remain mandatory; R14 surveys relative
    # tolerances without approving them from candidate outputs.
    # The installed Torch wheel stays CPU-only on Pascal in both stages.
    bash ci/full_cpu.sh
    ;;
  *) echo 'usage: extended_gate.sh contracts|build|cpu|gpu' >&2;exit 2 ;;
esac
