#!/usr/bin/env bash
set -Eeuo pipefail

: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${CUDA_COMPUTE_CAP:=61}"
: "${NVCC:=/opt/cuda/bin/nvcc}"
: "${NVCC_CCBIN:=gcc-14}"
: "${CANDLE_ASD_HOME:=/run/haricot/asd}"

REPORT_DIR="${RUNNER_TEMP:-/tmp}/extended-sm61-proof"
mkdir -p "$REPORT_DIR"

fail() {
  echo "EXTENDED_SM61_VALIDATION=FAIL reason=$*" | tee -a "$REPORT_DIR/summary.txt" >&2
  exit 1
}

actual_sha="$(git rev-parse HEAD)"
[[ "$actual_sha" == "$CANDIDATE_SHA" ]] || fail "candidate_sha_mismatch expected=$CANDIDATE_SHA actual=$actual_sha"

manifest=candle-integration/extended.json
[[ -f "$manifest" ]] || fail "missing $manifest"
jq -e '
  .schema_version == 1 and
  .kind == "extended-six" and
  .integration_target == "extended" and
  [.features[].feature] == ["bf16_candle","fp8_candle","fp4_candle",
    "cudnn_fallback_candle","moe_simt_f16_candle","asd_core_v3_standalone"]
' "$manifest" >/dev/null || fail "invalid Extended manifest"
campaign="$(jq -r '.campaign' "$manifest")"

cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')"
[[ "$cc" == "6.1" ]] || fail "expected compute capability 6.1, got $cc"

test -r "$CANDLE_ASD_HOME/fallbacks/current.json" || fail "missing ASD fallback store"
test -r "$CANDLE_ASD_HOME/profiles/current.asd" || fail "missing ASD runtime profile"
export CANDLE_ASD_MODULE_DIR="${CANDLE_ASD_MODULE_DIR:-$CANDLE_ASD_HOME/artifacts/sm61}"

G2_IMPL='candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256'
test -r "$CANDLE_ASD_MODULE_DIR/$G2_IMPL.cubin" || fail "missing G2 ASD CUBIN"
test -r "$CANDLE_ASD_MODULE_DIR/$G2_IMPL.manifest" || fail "missing G2 ASD manifest"

"$NVCC" --version | tee "$REPORT_DIR/nvcc.txt"
gcc-14 --version | head -n1 | tee "$REPORT_DIR/gcc.txt"
rustc --version | tee "$REPORT_DIR/rustc.txt"
cargo --version | tee "$REPORT_DIR/cargo.txt"
nvidia-smi --query-gpu=name,compute_cap,driver_version,uuid --format=csv,noheader   | tee "$REPORT_DIR/gpu.txt"

export CUDA_COMPUTE_CAP=61
export NVCC
export NVCC_CCBIN
export CARGO_INCREMENTAL=0
export CANDLE_ASD_HOME
export CANDLE_ASD_MODULE_DIR

CORE_FEATURES="cuda,cudnn,cuda-legacy-bf16,cuda-legacy-fp8,cuda-legacy-fp4"

run_gate() {
  local name="$1"
  shift
  echo "=== GATE $name ===" | tee -a "$REPORT_DIR/summary.txt"
  "$@" 2>&1 | tee "$REPORT_DIR/$name.log"
  echo "GATE_$name=PASS" | tee -a "$REPORT_DIR/summary.txt"
}

# Composition gate: Legacy CUDA and ASD-Core V3 must coexist in one build.
run_gate combined_compile   cargo check -p candle-core -p candle-nn --release --features "$CORE_FEATURES"

# Downstream closure: the qualified composition must compile the transformer stack
# and the concrete quantized Qwen3 CUDA consumer that exposed the previous gap.
run_gate downstream_transformers_compile \
  cargo check -p candle-transformers --lib --release --locked
run_gate downstream_qwen3_cuda_compile \
  cargo check -p candle-examples --example quantized-qwen3 --features cuda --release --locked

# Preserve the already-qualified Legacy runtime behavior.
run_gate bf16   cargo test -p candle-core --release --features "$CORE_FEATURES"     --test cuda_legacy_bf16_tests -- --nocapture

run_gate fp8   cargo test -p candle-core --release --features "$CORE_FEATURES"     --test fp8_legacy_tests -- --nocapture

run_gate fp4_mxfp4   cargo test -p candle-core --release --features "$CORE_FEATURES"     --test mxfp4_tests -- --nocapture

run_gate cudnn_fallback   cargo test -p candle-core --release --features "$CORE_FEATURES" --lib     cuda_backend::cudnn_fallback_classifier_tests::synthetic_quarantine_routes_generic_conv1d_conv2d     -- --exact --nocapture

run_gate moe_simt_f16   cargo test -p candle-nn --release --features cuda     --test moe_simt_sm61 -- --nocapture

# ASD-Core V3 must remain runtime-data driven inside the Legacy composition.
run_gate asd_module_provider   cargo run -p candle-core --release --example asd_v3_module_provider_validate     --features "$CORE_FEATURES"

run_gate asd_runtime_profile_swap   cargo run -p candle-core --release --example asd_v3_phase_d_runtime_profile_swap_validate     --features "$CORE_FEATURES"

echo "EXTENDED_SM61_CANDIDATE=$actual_sha" | tee -a "$REPORT_DIR/summary.txt"
echo "EXTENDED_SM61_CAMPAIGN=$campaign" | tee -a "$REPORT_DIR/summary.txt"
echo "EXTENDED_SM61_VALIDATION=PASS" | tee -a "$REPORT_DIR/summary.txt"
