#!/usr/bin/env bash
# Two modes, same reproducible host toolchain. No GPU is executed here.
set -euo pipefail
: "${MODE:?}" "${SHA:?}" "${RUNNER_TEMP:?}"
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || exit 3
if [[ "$MODE" == standalone ]]; then
  : "${FEATURE:?}" "${SOURCE:?}"
  report="$RUNNER_TEMP/six-standalone-cpu"
elif [[ "$MODE" == integrated ]]; then
  : "${MANIFEST_SHA:?}" "${CAMPAIGN:?}"
  report="$RUNNER_TEMP/six-integrated-cpu"
  manifest=candle-integration/standalone-six.json
  [[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$MANIFEST_SHA" ]] || exit 3
  jq -e --arg c "$CAMPAIGN" '
    .schema_version==2 and .kind=="standalone-six" and .campaign==$c and
    [.features[].feature]==["bf16_candle","fp8_candle","fp4_candle",
      "cudnn_fallback_candle","moe_simt_f16_candle","asd_core"]
  ' "$manifest" >/dev/null || exit 3
  while IFS=$'\t' read -r feature prepared; do
    git merge-base --is-ancestor "$prepared" HEAD || {
      echo "::error::Integrated HEAD lacks prepared $feature=$prepared"; exit 3
    }
  done < <(jq -r '.features[]|[.feature,.prepared_sha]|@tsv' "$manifest")
else
  echo "::error::Unknown Candle standalone CPU mode"; exit 2
fi
mkdir -p "$report"
cargo fmt --all -- --check > "$report/fmt.log" 2>&1 || {
  tail -n 70 "$report/fmt.log"; exit 1
}
cargo generate-lockfile > "$report/lock.log" 2>&1 || {
  tail -n 80 "$report/lock.log"; exit 1
}
cp Cargo.lock "$report/Cargo.lock"
lock="$(sha256sum Cargo.lock | cut -d' ' -f1)"
cargo check -p candle-core -p candle-nn --no-default-features --locked \
  > "$report/cpu-check.log" 2>&1 || {
    tail -n 110 "$report/cpu-check.log"; exit 1
}
if [[ "$MODE" == integrated ]]; then
  cargo test -p candle-core --no-default-features --locked \
    --test mxfp4_tests --test nvfp4_experiment \
    --test grouped_conv_core_tests --test grouped_conv_transpose_core_tests \
    -- --test-threads=1 > "$report/cpu-tests.log" 2>&1 || {
      tail -n 110 "$report/cpu-tests.log"; exit 1
    }
  jq -n --arg sha "$SHA" --arg campaign "$CAMPAIGN" \
    --arg manifest "$MANIFEST_SHA" --arg lock "$lock" \
    '{status:"CPU_PASSED",candidate_sha:$sha,campaign:$campaign,
      manifest_sha256:$manifest,lock_sha256:$lock}' > "$report/report.json"
  echo "lock_sha256=$lock" >> "$GITHUB_OUTPUT"
  echo "Combined six-feature CPU PASS at $SHA, Cargo.lock $lock" >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi
case "$FEATURE" in
  fp4_candle)
    cargo test -p candle-core --no-default-features --locked \
      --test mxfp4_tests --test nvfp4_experiment > "$report/cpu-tests.log" 2>&1 ;;
  asd_core)
    cargo test -p candle-core --no-default-features --locked \
      --test grouped_conv_core_tests --test grouped_conv_transpose_core_tests \
      -- --test-threads=1 > "$report/cpu-tests.log" 2>&1 ;;
  moe_simt_f16_candle)
    rustc --edition=2021 --test candle-kernels/src/moe_selection.rs \
      -o "$RUNNER_TEMP/standalone-moe-selection" > "$report/cpu-tests.log" 2>&1 &&
    "$RUNNER_TEMP/standalone-moe-selection" --nocapture >> "$report/cpu-tests.log" 2>&1 ;;
  bf16_candle|fp8_candle|cudnn_fallback_candle)
    cargo test -p candle-core --no-default-features --locked --lib \
      > "$report/cpu-tests.log" 2>&1 ;;
  *) echo "::error::Unsupported standalone CPU suite $FEATURE"; exit 3 ;;
esac || {
  tail -n 110 "$report/cpu-tests.log"; exit 1
}
jq -n --arg feature "$FEATURE" --arg src "$SOURCE" --arg sha "$SHA" \
  --arg lock "$lock" \
  '{feature:$feature,source_sha:$src,prepared_sha:$sha,
    lock_sha256:$lock,status:"CPU_PASSED"}' > "$report/report.json"
echo "Independent $FEATURE CPU PASS at $SHA" >> "$GITHUB_STEP_SUMMARY"
