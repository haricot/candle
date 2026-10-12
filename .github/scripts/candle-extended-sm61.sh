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
  .kind == "extended-multi" and
  .integration_target == "extended" and
  (.features | type == "array" and length > 0) and
  .feature_count == (.features | length) and
  ([.features[].feature] | unique | length) == .feature_count and
  all(.features[]; (.cuda_features | type) == "array") and
  ([.features[].cuda_features[]] | length > 0)
' "$manifest" >/dev/null || fail "invalid Extended manifest"
has_feature() {
  jq -e --arg feature "$1" '.features | any(.feature == $feature)' "$manifest" >/dev/null
}
campaign="$(jq -r '.campaign' "$manifest")"

cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')"
[[ "$cc" == "6.1" ]] || fail "expected compute capability 6.1, got $cc"

if has_feature asd_core_v3_standalone; then
  test -r "$CANDLE_ASD_HOME/fallbacks/current.json" || fail "missing ASD fallback store"
  test -r "$CANDLE_ASD_HOME/profiles/current.asd" || fail "missing ASD runtime profile"
  export CANDLE_ASD_MODULE_DIR="${CANDLE_ASD_MODULE_DIR:-$CANDLE_ASD_HOME/artifacts/sm61}"

  G2_IMPL='candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256'
  test -r "$CANDLE_ASD_MODULE_DIR/$G2_IMPL.cubin" || fail "missing G2 ASD CUBIN"
  test -r "$CANDLE_ASD_MODULE_DIR/$G2_IMPL.manifest" || fail "missing G2 ASD manifest"
fi

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

CORE_FEATURES="$(jq -r '[.features[].cuda_features[]] | unique | join(",")' "$manifest")"
[[ -n "$CORE_FEATURES" ]] || fail "no CUDA features in this physical SM61 campaign"

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

# Feature-specific smoke gates; do not assume seven/eight sources.
if has_feature embeddinggemma2; then
  run_gate embeddinggemma2_cuda_compile \
    cargo check -p candle-examples --example embedding-gemma2 --features cuda --release --locked
fi
if has_feature lean4_standalone; then
  # Only CUDA adapter compilation; Lean mathematical proof ran in CPU gate.
  run_gate lean4_cuda_adapter_compile \
    cargo check --manifest-path tools/lean4-standalone/Cargo.toml --features cuda --release
fi

# Preserve the already-qualified Legacy runtime behavior.
if has_feature bf16_candle; then
  run_gate bf16 cargo test -p candle-core --release --features "$CORE_FEATURES" --test cuda_legacy_bf16_tests -- --nocapture
fi

if has_feature fp8_candle; then
  run_gate fp8 cargo test -p candle-core --release --features "$CORE_FEATURES" --test fp8_legacy_tests -- --nocapture
fi

if has_feature fp4_candle; then
  run_gate fp4_mxfp4 cargo test -p candle-core --release --features "$CORE_FEATURES" --test mxfp4_tests -- --nocapture
fi

if has_feature cudnn_fallback_candle; then
  run_gate cudnn_fallback cargo test -p candle-core --release --features "$CORE_FEATURES" --lib \
    cuda_backend::cudnn_fallback_classifier_tests::synthetic_quarantine_routes_generic_conv1d_conv2d -- --exact --nocapture
fi

if has_feature moe_simt_f16_candle; then
  run_gate moe_simt_f16 cargo test -p candle-nn --release --features cuda --test moe_simt_sm61 -- --nocapture
fi

# ASD-Core V3 runtime invariants apply only when ASD is composed.
if has_feature asd_core_v3_standalone; then
  run_gate asd_module_provider cargo run -p candle-core --release --example asd_v3_module_provider_validate --features "$CORE_FEATURES"
  run_gate asd_runtime_profile_swap cargo run -p candle-core --release --example asd_v3_phase_d_runtime_profile_swap_validate --features "$CORE_FEATURES"
fi

echo "EXTENDED_SM61_CANDIDATE=$actual_sha" | tee -a "$REPORT_DIR/summary.txt"
echo "EXTENDED_SM61_CAMPAIGN=$campaign" | tee -a "$REPORT_DIR/summary.txt"
echo "EXTENDED_SM61_VALIDATION=PASS" | tee -a "$REPORT_DIR/summary.txt"
