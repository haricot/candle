#!/usr/bin/env bash
set -Eeuo pipefail

: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${CUDA_COMPUTE_CAP:=61}"
: "${NVCC:=/opt/cuda/bin/nvcc}"
: "${NVCC_CCBIN:=gcc-14}"

REPORT_DIR="${RUNNER_TEMP:-/tmp}/legacy-sm61-proof"
mkdir -p "$REPORT_DIR"

fail() {
  echo "LEGACY_SM61_VALIDATION=FAIL reason=$*" | tee -a "$REPORT_DIR/summary.txt" >&2
  exit 1
}

actual_sha="$(git rev-parse HEAD)"
[[ "$actual_sha" == "$CANDIDATE_SHA" ]] || fail "candidate_sha_mismatch expected=$CANDIDATE_SHA actual=$actual_sha"

[[ -f candle-integration/legacy-cuda.json ]] || fail "missing candle-integration/legacy-cuda.json"
jq -e '
  .schema_version == 1 and
  .kind == "legacy-cuda-five" and
  .integration_target == "legacy-cuda" and
  (.features | length == 5)
' candle-integration/legacy-cuda.json >/dev/null || fail "invalid legacy-cuda manifest"

campaign="$(jq -r '.campaign' candle-integration/legacy-cuda.json)"
[[ -n "$campaign" && "$campaign" != null ]] || fail "missing campaign"

for var in CANDLE_ASD_HOME CANDLE_ASD_RUNTIME_PROFILE CANDLE_ASD_MODULE_DIR CANDLE_ASD_FALLBACKS; do
  [[ -z "${!var:-}" ]] || fail "unexpected ASD runtime environment: $var"
done

cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')"
[[ "$cc" == "6.1" ]] || fail "expected compute capability 6.1, got $cc"

gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n1 | tr -d '[:space:]')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '[:space:]')"

"$NVCC" --version | tee "$REPORT_DIR/nvcc.txt"
gcc-14 --version | head -n1 | tee "$REPORT_DIR/gcc.txt"
rustc --version | tee "$REPORT_DIR/rustc.txt"
cargo --version | tee "$REPORT_DIR/cargo.txt"

{
  echo "candidate_sha=$actual_sha"
  echo "campaign=$campaign"
  echo "gpu_name=$gpu_name"
  echo "gpu_uuid=$gpu_uuid"
  echo "compute_cap=$cc"
  echo "driver=$driver"
  echo "cuda_compute_cap=$CUDA_COMPUTE_CAP"
} | tee "$REPORT_DIR/environment.txt"

export CUDA_COMPUTE_CAP=61
export NVCC
export NVCC_CCBIN
export CARGO_INCREMENTAL=0

CORE_FEATURES="cuda,cudnn,cuda-legacy-bf16,cuda-legacy-fp8,cuda-legacy-fp4"

run_gate() {
  local name="$1"
  shift
  echo "=== GATE $name ===" | tee -a "$REPORT_DIR/summary.txt"
  "$@" 2>&1 | tee "$REPORT_DIR/$name.log"
  echo "GATE_$name=PASS" | tee -a "$REPORT_DIR/summary.txt"
}

run_gate combined_compile \
  cargo check -p candle-core -p candle-nn --release --features "$CORE_FEATURES"

run_gate bf16 \
  cargo test -p candle-core --release --features "$CORE_FEATURES" \
    --test cuda_legacy_bf16_tests -- --nocapture

run_gate fp8 \
  cargo test -p candle-core --release --features "$CORE_FEATURES" \
    --test fp8_legacy_tests -- --nocapture

run_gate fp4_mxfp4 \
  cargo test -p candle-core --release --features "$CORE_FEATURES" \
    --test mxfp4_tests -- --nocapture

run_gate cudnn_fallback \
  cargo test -p candle-core --release --features "$CORE_FEATURES" --lib \
    cuda_backend::cudnn_fallback_classifier_tests::synthetic_quarantine_routes_generic_conv1d_conv2d \
    -- --exact --nocapture

run_gate moe_simt_f16 \
  cargo test -p candle-nn --release --features cuda \
    --test moe_simt_sm61 -- --nocapture

echo "LEGACY_SM61_CANDIDATE=$actual_sha" | tee -a "$REPORT_DIR/summary.txt"
echo "LEGACY_SM61_CAMPAIGN=$campaign" | tee -a "$REPORT_DIR/summary.txt"
echo "LEGACY_SM61_VALIDATION=PASS" | tee -a "$REPORT_DIR/summary.txt"
