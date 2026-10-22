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
    policy=../candle-integration/half-duplex-ci.json
    minimum=short
    if [[ -f "$policy" ]];then
      minimum="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$policy")"
    fi
    profile="${HDB_CI_PROFILE:-$minimum}"
    case "$minimum/$profile" in
      short/short|short/full|short/audit|full/full|full/audit|audit/audit) ;;
      *) echo "HALF_DUPLEX_INVALID_OR_DOWNGRADED_PROFILE=$minimum/$profile" >&2;exit 3 ;;
    esac
    export HDB_CI_PROFILE="$profile"
    # Legacy flags must never select an additional implicit campaign.
    export HDB2C_R13_CPU_FULL=0 HDB2C_R14_CUDA=0 HDB2C_R14_FP64_REFERENCE=0
    echo "HALF_DUPLEX_CI_PROFILE=$profile"
    if ! command -v uv >/dev/null;then
      python3 ci/bootstrap_local_uv.py "$RUNNER_TEMP/halfduplex-tools"
      export PATH="$RUNNER_TEMP/halfduplex-tools:$PATH"
    fi
    if [[ "$stage" == cpu ]];then
      export HDB2C_R8_VARIANT=cpu
    else
      test "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d '[:space:]')" = 6.1
      export CUDA_COMPUTE_CAP=61 HDB2C_R8_CORPUS=1 HDB2C_R9_TEXT=1
      export HDB2C_R8_VARIANT=cuda-cudnn HDB2C_R13_CPU_FULL=0
      echo 'HALF_DUPLEX_GPU=REAL_NATIVE_CUDA_CUDNN_CPU_NEMO_REFERENCE'
      echo 'HALF_DUPLEX_GPU_SCOPE=R9_NATIVE_GPU_NOT_THE_CPU_ONLY_R13_KERNELS'
      if [[ "$profile" == audit ]];then
        export HDB2C_R14_CUDA=1 HDB2C_R14_REQUIRE_FUNCTIONAL=1
        echo 'HALF_DUPLEX_GPU_AUDIT=EXPLICIT_FP64_42_CLIPS'
      fi
    fi
    # Both stages keep the original CPU NeMo reference. Detailed FP64 replay
    # runs only for an explicitly requested audit; no legacy threshold is fitted.
    bash ci/full_cpu.sh
    ;;
  *) echo 'usage: extended_gate.sh contracts|build|cpu|gpu' >&2;exit 2 ;;
esac
