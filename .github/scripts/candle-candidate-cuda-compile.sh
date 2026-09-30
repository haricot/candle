#!/usr/bin/env bash
set -uo pipefail
report="$RUNNER_TEMP/candle-candidate-cuda"
mkdir -p "$report"
# Hosted CUDA containers run as root over a runner-owned workspace.
# The checkout action's host safe.directory setting does not cross
# the container HOME boundary; configure it before reading HEAD.
git config --global --add safe.directory "$GITHUB_WORKSPACE"
test "$(dpkg-query -W -f='${Version}' cuda-toolkit-12-9)" = '12.9.2-1'
test "$(dpkg-query -W -f='${Version}' libcublas-12-9)" = '12.9.2.10-1'
test "$(dpkg-query -W -f='${Version}' libcudnn9-cuda-12)" = '9.10.2.21-1'
grep -Fq 'VERSION_ID="26.04"' /etc/os-release
recipe_sha="$(sha256sum "$GITHUB_WORKSPACE/.ci-tools/.github/container/Containerfile.cuda1292-ubuntu2604" | cut -d' ' -f1)"
[[ "$(git rev-parse HEAD)" == "$CANDIDATE_SHA" ]] || exit 3
cp "$RUNNER_TEMP/candidate-cpu-proof/Cargo.lock" Cargo.lock
lock="$(sha256sum Cargo.lock | cut -d' ' -f1)"
[[ -n "$LOCK_SHA256" && "$lock" == "$LOCK_SHA256" ]] || {
  echo "::error::CPU and CUDA lockfile differ"; exit 3;
}
case "$CHECK_BRANCH" in
  bf16_candle) features="candle-core/cuda-legacy-bf16,candle-nn/cuda" ;;
  fp8_candle) features="candle-core/cuda-legacy-fp8,candle-nn/cuda" ;;
  fp4_candle|moe_simt_f16_candle) features="candle-core/cuda,candle-nn/cuda" ;;
  standalone_fp48_integration)
    features="candle-core/cuda-legacy-fp8,candle-core/cuda-legacy-fp4,candle-nn/cuda" ;;
  standalone_six_integration)
    features="candle-core/cuda-legacy-bf16,candle-core/cuda-legacy-fp8,candle-core/cuda-legacy-fp4,candle-nn/cudnn" ;;
  cudnn_fallback_candle|asd_core)
    features="candle-core/cudnn,candle-nn/cudnn" ;;
  cuda_asd_runner_v2)
    features="candle-core/cuda-legacy-bf16,candle-core/cuda-legacy-fp8,candle-nn/cudnn" ;;
  *) echo "::error::Unknown branch"; exit 2 ;;
esac
{
  nvcc --version
  rustc --version
  echo "target: sm_61"
  echo "features: $features"
} > "$report/environment.txt"
build_rc=0
cargo check -p candle-core -p candle-nn --lib --tests \
  --no-default-features --features "$features" --locked \
  > "$report/cuda-check.log" 2>&1 || build_rc=$?

if [[ "$CHECK_BRANCH" == standalone_six_integration && "$build_rc" -eq 0 ]]; then
  cuda_tests_rc=0
  cargo check -p candle-core --features cuda --tests --locked \
    > "$report/cuda-tests-default.log" 2>&1 || cuda_tests_rc=$?
  if ((cuda_tests_rc != 0)); then
    echo "::error::Integrated candle-core CUDA --tests check failed"
    tail -n 100 "$report/cuda-tests-default.log"
    build_rc=$cuda_tests_rc
  fi
fi
if [[ "$CHECK_BRANCH" == standalone_six_integration && "$build_rc" -eq 0 ]]; then
  fp4_tests_rc=0
  cargo check -p candle-core --features "cuda cuda-legacy-fp4" --tests --locked \
    > "$report/cuda-tests-legacy-fp4.log" 2>&1 || fp4_tests_rc=$?
  if ((fp4_tests_rc != 0)); then
    echo "::error::Integrated candle-core CUDA+legacy-FP4 --tests check failed"
    tail -n 100 "$report/cuda-tests-legacy-fp4.log"
    build_rc=$fp4_tests_rc
  fi
fi


# Isolated FP4 deliberately leaves the normal GGUF MXFP4 paths enabled.
# Also compile the explicitly opted-in NVFP4 software kernels/tests under SM61
# before any physical GPU testing; failing this extra gate fails Candidate CI.
if [[ "${CANDIDATE_REF:-}" == fp4_candle_standalone && "$build_rc" -eq 0 ]]; then
  fp4_rc=0
  cargo check -p candle-core -p candle-nn --lib --tests --locked \
    --no-default-features \
    --features "candle-core/cuda-legacy-fp4,candle-nn/cuda" \
    > "$report/fp4-legacy-optin-check.log" 2>&1 || fp4_rc=$?
  if ((fp4_rc != 0)); then
    echo "::error::Standalone FP4 software NVFP4 opt-in fails SM61 compilation"
    tail -n 100 "$report/fp4-legacy-optin-check.log"
    build_rc=$fp4_rc
  else
    echo "Standalone FP4 MXFP4 default + optional NVFP4 software compile PASS" >> "$report/environment.txt"
  fi
fi

# The regular --tests check above uses the integrated feature selection.
# It may compile candle-nn without its own cuda feature, which excludes
# moe_simt_sm61.rs via #![cfg(feature = "cuda")]. Explicitly compile this
# test with BOTH candle-nn features before assigning a physical GPU job.
if [[ ( "$CHECK_BRANCH" == moe_simt_f16_candle || "$CHECK_BRANCH" == cuda_asd_runner_v2 || "$CHECK_BRANCH" == standalone_six_integration ) && "$build_rc" -eq 0 ]]; then
  moe_test_rc=0
  cargo check -p candle-nn --locked --no-default-features \
    --features "cuda cudnn" --test moe_simt_sm61 \
    > "$report/moe-test-compile.log" 2>&1 || moe_test_rc=$?
  if ((moe_test_rc != 0)); then
    echo "::error::SM61 MoE test fails explicit CUDA compilation (rc=$moe_test_rc)."
    tail -n 70 "$report/moe-test-compile.log"
    build_rc="$moe_test_rc"
  else
    echo "SM61 MoE CUDA test compilation passed" >> "$report/environment.txt"
  fi
fi

if [[ ( "$CHECK_BRANCH" == moe_simt_f16_candle || "$CHECK_BRANCH" == cuda_asd_runner_v2 || "$CHECK_BRANCH" == standalone_six_integration ) && "$build_rc" -eq 0 ]]; then
  # cargo check cannot prove static-library linkability. For Pascal,
  # the SIMT translation unit must define both the real SIMT entry
  # and the WMMA compatibility ABI stub omitted with WMMA objects.
  archive="$(find target/debug/build -path '*/out/libmoe.a' -type f | sort | head -n 1)"
  if [[ -z "$archive" ]]; then
    echo "ERROR: libmoe.a absent after CUDA compilation" > "$report/moe-symbols.log"
    build_rc=1
  elif ! nm -g --defined-only "$archive" > "$report/moe-symbols.log" 2>&1; then
    build_rc=1
  else
    echo "archive=$archive" >> "$report/moe-symbols.log"
    for symbol in moe_gemm_simt_f16 moe_gemm_wmma; do
      if ! grep -Eq "[[:space:]](T|W)[[:space:]]${symbol}$" "$report/moe-symbols.log"; then
        echo "ERROR: missing SM61 MoE ABI symbol $symbol" >> "$report/moe-symbols.log"
        build_rc=1
      fi
    done
  fi
fi
status=CUDA_COMPILE_PASSED
((build_rc == 0)) || status=CUDA_COMPILE_FAILED
jq -n --arg branch "$CHECK_BRANCH" --arg campaign "$CHECK_CAMPAIGN" \
  --arg sha "$CANDIDATE_SHA" --arg lock "$lock" --arg status "$status" \
  --arg features "$features" --arg recipe "$recipe_sha" --argjson rc "$build_rc" \
  '{branch:$branch,campaign:$campaign,candidate_sha:$sha,lock_sha256:$lock,
    status:$status,cuda_version:"12.9.2",ubuntu_version:"26.04",
      cuda_containerfile_sha256:$recipe,target:"sm_61",
    features:$features,exit_code:$rc,gpu_executed:false}' > "$report/report.json"
echo "CUDA: $status / $CHECK_BRANCH / $CANDIDATE_SHA (GPU not executed)" >> "$GITHUB_STEP_SUMMARY"
[[ "$status" == CUDA_COMPILE_PASSED ]]
