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

if [[ ( "$CHECK_BRANCH" == moe_simt_f16_candle || "$CHECK_BRANCH" == cuda_asd_runner_v2 ) && "$build_rc" -eq 0 ]]; then
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
