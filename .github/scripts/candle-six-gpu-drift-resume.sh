#!/usr/bin/env bash
# Resume ONLY the ASD DW5x5 physical SM61 measurement, preserving the exact
# successful per-feature GPU evidence from the previous partial run.
set -euo pipefail
umask 077
cfg="$GITHUB_WORKSPACE/.ci-tools/.github/six-asd-drift-resume.json"
report="$RUNNER_TEMP/six-gpu"
candidate=e44869d909dc536b9a1f37cf89bfba3d46fc1985
campaign=six-sm61-v3-r3-20260929
prior_run=36556360212
expected_lock=653c6c9c59bf8789a2f44af9b4d2cf4057d51a99836ac7ab0d5c02d3e0257d6b
expected_manifest=47176fd95cf3e2e2182b42263dae2b11124b3c4ebac250c483d5774060322f86
expected_policy=912c8b0c88fc8b3e07a25a79f5b9f0d10eef1b8193271ee8fafb4db1f9e9615f
mkdir -p "$report/logs"
[[ "$(git rev-parse HEAD)" == "$candidate" ]] || { echo "::error::Candidate SHA mismatch"; exit 3; }
jq -e --arg sha "$candidate" --arg c "$campaign" --arg lock "$expected_lock" \
  --arg manifest "$expected_manifest" --argjson prior "$prior_run" '
  .schema_version==1 and .candidate_sha==$sha and .campaign==$c and
  .lock_sha256==$lock and .manifest_sha256==$manifest and
  .original_run_id==$prior and .original_gpu_job_id==109372414549 and
  .original_gpu_artifact_id==11032135466 and
  .warmup_ms==1000 and .iters==40 and .inner==32 and .max_drift_pct==20 and
  (.prior_files|length)==14
' "$cfg" >/dev/null || { echo "::error::Drift-resume input mismatch"; exit 3; }
[[ "$(sha256sum candle-integration/standalone-six.json | cut -f1 -d' ')" == "$expected_manifest" ]] || exit 3
jq -e --arg c "$campaign" '.schema_version==2 and .kind=="standalone-six" and
  .campaign==$c and ([.features[].feature]==["bf16_candle","fp8_candle",
    "fp4_candle","cudnn_fallback_candle","moe_simt_f16_candle","asd_core"])' \
  candle-integration/standalone-six.json >/dev/null || exit 3
[[ "$(jq -r .candidate_sha "$RUNNER_TEMP/six-integrated-cpu/report.json")" == "$candidate" ]] || exit 3
[[ "$(jq -r .lock_sha256 "$RUNNER_TEMP/six-integrated-cpu/report.json")" == "$expected_lock" ]] || exit 3
[[ "$(jq -r .status "$RUNNER_TEMP/six-integrated-cpu/report.json")" == CPU_PASSED ]] || exit 3
test -s "$RUNNER_TEMP/six-integrated-cpu/Cargo.lock"
cp "$RUNNER_TEMP/six-integrated-cpu/Cargo.lock" Cargo.lock
[[ "$(sha256sum Cargo.lock | cut -f1 -d' ')" == "$expected_lock" ]] || exit 3
jq -e --arg sha "$candidate" --arg lock "$expected_lock" '
  .status=="CUDA_COMPILE_PASSED" and .candidate_sha==$sha and
  .lock_sha256==$lock and .target=="sm_61"
' "$RUNNER_TEMP/six-cuda/report.json" >/dev/null || exit 3
# Match EVERY file from the original completed GPU job, including the
# original failed DW5x5 benchmark. Reject missing, extra, or changed evidence.
while IFS=$'\t' read -r relative checksum; do
  [[ "$relative" =~ ^(logs/[a-z0-9-]+\.log|device\.txt|nvcc\.txt)$ ]] || exit 3
  [[ -s "$report/$relative" ]] || { echo "::error::Missing old evidence: $relative"; exit 3; }
  [[ "$(sha256sum "$report/$relative" | cut -f1 -d' ')" == "$checksum" ]] || {
    echo "::error::Previous GPU evidence altered: $relative"; exit 3;
  }
done < <(jq -r '.prior_files|to_entries[]|[.key,.value]|@tsv' "$cfg")
for name in bf16 fp8 fp4 conv-native conv-cudnn cudnn-quarantine moe \
  nvfp4-lut nvfp4-parity asd-core asd-transpose; do
  grep -Eq 'test result: ok\. [1-9][0-9]* passed' "$report/logs/$name.log" || {
    if [[ "$name" == asd-transpose ]]; then
      grep -Fxq 'STATUS=PASS' "$report/logs/$name.log" || exit 3
    else
      echo "::error::No retained GPU test PASS evidence for $name"; exit 3
    fi
  }
done
grep -Fxq 'STATUS=PASS' "$report/logs/asd-transpose.log" || exit 3
grep -Fq 'GATE parity=true drift=true integrated_speedup=true p90_non_regression=true' \
  "$report/logs/asd-transpose.log" || exit 3
grep -Fq 'STATUS=HOLD' "$report/logs/asd-conv2d.log" || exit 3
mv "$report/logs/asd-conv2d.log" "$report/logs/asd-conv2d-original-hold.log"
# Re-attest this SAME physical GPU; never reuse the old policy check as a substitute.
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  root="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
  root="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)"
else
  echo "::error::CUDA_HOME/nvcc missing"; exit 3
fi
export PATH="$root/bin:$PATH"
cuda_lib="$root/lib64"; [[ -d "$cuda_lib" ]] || cuda_lib="$root/lib"
libs="$cuda_lib"
if [[ -n "${CUDNN_HOME:-}" ]]; then
  cudnn_lib="$CUDNN_HOME/lib64"; [[ -d "$cudnn_lib" ]] || cudnn_lib="$CUDNN_HOME/lib"
  [[ -d "$cudnn_lib" ]] || exit 3
  libs="$cudnn_lib:$libs"
fi
export LD_LIBRARY_PATH="$libs:${LD_LIBRARY_PATH:-}"
nvcc --version | grep -F 'release 12.9'
if [[ -n "${CUDA_VERSION:-}" ]]; then
  [[ "$CUDA_VERSION" == 12.9.2 ]] || exit 3
elif command -v pacman >/dev/null 2>&1; then
  [[ "$(pacman -Q cuda-12.9)" == 'cuda-12.9 12.9.2-1' ]] || exit 3
else
  [[ "${CANDLE_CUDA_TOOLKIT_RELEASE:-}" == 12.9.2 ]] || exit 3
fi
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')"
[[ "$cc" == 6.1 ]] || exit 3
policy="$RUNNER_TEMP/stage2f-production.v2.asd"
if [[ -n "${POLICY_B64:-}" ]]; then
  printf '%s' "$POLICY_B64" | base64 --decode > "$policy"
elif [[ -n "${CANDLE_ASD_EXACT_POLICY:-}" && -r "$CANDLE_ASD_EXACT_POLICY" ]]; then
  cp -- "$CANDLE_ASD_EXACT_POLICY" "$policy"
else
  echo "::error::Missing pinned Stage2F policy"; exit 3
fi
[[ "$(sha256sum "$policy" | cut -f1 -d' ')" == "$expected_policy" ]] || exit 3
[[ "$(head -n1 "$policy")" == ASD-EXACT-POLICY-V2 ]] || exit 3
[[ "$(awk -F'|' '$1=="decision" && $3=="promoted"{n++} END{print n+0}' "$policy")" == 12 ]] || exit 3
[[ "$(awk -F'|' '$1=="decision"{n++} END{print n+0}' "$policy")" == 12 ]] || exit 3
actual_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n1 | tr -d '[:space:]\r')"
policy_uuid="$(sed -n 's/^target.gpu_uuid=//p' "$policy")"
[[ "$actual_uuid" == GPU-* && "$actual_uuid" == "$policy_uuid" ]] || exit 3
export CUDA_COMPUTE_CAP=61 CARGO_INCREMENTAL=0 CANDLE_ASD_VALIDATION=1
export CANDLE_ASD_EXACT_POLICY="$policy"
export CANDLE_ASD_TARGET_GPU_UUID="$actual_uuid"
export CANDLE_SM61_EXACT_GROUPED_EVIDENCE_GPU_UUID="$actual_uuid"
# No relaxed drift threshold and no cherry-picking successful individual cases:
# repeat all four promoted decisions with the source example's own default
# 1000-ms warmup, 40 timed samples and 32 launches per sample.
features="cuda cuda-legacy-bf16 cuda-legacy-fp8 cuda-legacy-fp4 cudnn"
if ! cargo run -p candle-core --locked --release --no-default-features \
    --features "$features" --example grouped_conv2d_asd_real_dispatch_validate -- \
    --warmup-ms 1000 --iters 40 --inner 32 --max-drift-pct 20 \
    --min-integrated-speedup-x 1.01 > "$report/logs/asd-conv2d-resume.log" 2>&1; then
  echo "::error::Repeat ASD Conv2D is not stable; no promotion"
  tail -n 90 "$report/logs/asd-conv2d-resume.log"
  exit 1
fi
grep -Fxq 'STATUS=PASS' "$report/logs/asd-conv2d-resume.log"
for channels in 48 96 192 384; do
  grep -E "^GATE c=$channels parity=true drift=true integrated_speedup=true p90_non_regression=true .* pass=true$" \
    "$report/logs/asd-conv2d-resume.log" >/dev/null || exit 3
done
grep -Fq 'DOMAIN_MISS current_vs_candidate' "$report/logs/asd-conv2d-resume.log"
grep -Fq 'parity=true expected_backend=cudnn pass=true' "$report/logs/asd-conv2d-resume.log"
grep -Fq 'REAL_DISPATCH_GATE cases=4 domain_miss=true production_activation=false pass=true' \
  "$report/logs/asd-conv2d-resume.log"
cp candle-integration/standalone-six.json "$report/sources.json"
cp "$cfg" "$report/resume-manifest.json"
jq -n --arg sha "$candidate" --arg c "$campaign" \
  --arg lock "$expected_lock" --arg manifest "$expected_manifest" \
  --arg cc "$cc" --arg policy "$expected_policy" --argjson prior "$prior_run" \
  --argjson current "$GITHUB_RUN_ID" --slurpfile sources "$report/sources.json" \
  '{schema_version:2,branch:"standalone_six_integration",
    candidate_sha:$sha,campaign:$c,ci_run_id:$current,
    original_gpu_run_id:$prior,prior_gpu_evidence_reused:true,
    resume_scope:"ASD DW5x5 four-case A/B/A only",
    resume_sampling:{warmup_ms:1000,iters:40,inner:32,max_drift_pct:20},
    upstream_sha:$sources[0].main_sha,lock_sha256:$lock,
    manifest_sha256:$manifest,gpu_compute_cap:$cc,gpu_executed:true,
    status:"GPU_PASSED",auto_promote:true,suite_coverage:"all_six",
    asd_policy_sha256:$policy,asd_policy_promoted_decisions:12,
    asd_policy_gpu_uuid_matches:true,
    sources:$sources[0].features}' > "$report/report.json"
echo "Reused exact-hash prior feature GPU proofs; fresh ASD DW5x5 4/4 stable PASS" \
  >> "$GITHUB_STEP_SUMMARY"
