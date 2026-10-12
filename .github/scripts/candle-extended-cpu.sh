#!/usr/bin/env bash
# Hosted CPU gate for a manifest-driven Extended candidate.
set -euo pipefail
: "${MODE:?}" "${SHA:?}" "${RUNNER_TEMP:?}"
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || exit 3

if [[ "$MODE" == standalone ]]; then
  : "${FEATURE:?}" "${SOURCE:?}"
  report="$RUNNER_TEMP/extended-standalone-cpu"
elif [[ "$MODE" == integrated ]]; then
  : "${MANIFEST_SHA:?}" "${CAMPAIGN:?}"
  report="$RUNNER_TEMP/extended-integrated-cpu"
  manifest=candle-integration/extended.json
  [[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$MANIFEST_SHA" ]] || exit 3
  jq -e --arg c "$CAMPAIGN" --slurpfile sources ".ci-tools/.github/extended-sources.json" '
    .schema_version == 1 and .kind == "extended-multi" and .campaign == $c and
    .integration_target == "extended" and
    .feature_count == (.features | length) and
    .feature_count > 0 and
    [.features[] | {feature,cuda_features}] ==
      [$sources[0].features[] | {feature,cuda_features}] and
    all(.features[]; (.cuda_features | type) == "array") and
    ([.features[].cuda_features[]] | length > 0)
  ' "$manifest" >/dev/null || exit 3
  while IFS=$'\t' read -r feature prepared; do
    git merge-base --is-ancestor "$prepared" HEAD || {
      echo "::error::Integrated HEAD lacks prepared $feature=$prepared"; exit 3;
    }
  done < <(jq -r '.features[]|[.feature,.prepared_sha]|@tsv' "$manifest")
else
  echo "::error::Unknown Extended CPU mode"; exit 2
fi

mkdir -p "$report"
if cargo fmt --all -- --check > "$report/fmt.log" 2>&1; then
  echo "FMT_CHECK=PASS" > "$report/fmt-status.txt"
else
  echo "FMT_CHECK=ADVISORY_DIFF" > "$report/fmt-status.txt"
  echo "::warning::Rustfmt differences recorded; functional validation continues"
fi

cargo generate-lockfile > "$report/lock.log" 2>&1 || {
  tail -n 80 "$report/lock.log"; exit 1;
}
cp Cargo.lock "$report/Cargo.lock"
lock="$(sha256sum Cargo.lock | cut -d' ' -f1)"

cargo check -p candle-core -p candle-nn --no-default-features --locked   > "$report/cpu-check.log" 2>&1 || {
    tail -n 120 "$report/cpu-check.log"; exit 1;
  }

has_feature() {
  jq -e --arg feature "$1" '.features | any(.feature == $feature)' "$manifest" >/dev/null
}

if [[ "$MODE" == integrated ]]; then
  cargo check -p candle-transformers --lib --locked > "$report/transformers-check.log" 2>&1 || {
    tail -n 120 "$report/transformers-check.log"; exit 1;
  }
  if has_feature embeddinggemma2; then
    cargo check -p candle-examples --example embedding-gemma2 --locked > "$report/embedding-gemma2-check.log" 2>&1 || {
      tail -n 120 "$report/embedding-gemma2-check.log"; exit 1;
    }
    cargo test -p candle-transformers embedding_gemma2::tests --lib --locked > "$report/embedding-gemma2-tests.log" 2>&1 || {
      tail -n 120 "$report/embedding-gemma2-tests.log"; exit 1;
    }
  fi
  cargo test -p candle-core --no-default-features --locked --lib > "$report/core-lib-tests.log" 2>&1 || {
    tail -n 120 "$report/core-lib-tests.log"; exit 1;
  }
  if has_feature fp4_candle; then
    cargo test -p candle-core --no-default-features --locked --test mxfp4_tests --test nvfp4_experiment > "$report/fp4-reference-tests.log" 2>&1 || {
      tail -n 120 "$report/fp4-reference-tests.log"; exit 1;
    }
  fi
  if has_feature asd_core_v3_standalone; then
    cargo test -p candle-core --no-default-features --locked --test grouped_conv_core_tests --test grouped_conv_transpose_core_tests -- --test-threads=1 > "$report/grouped-tests.log" 2>&1 || {
      tail -n 120 "$report/grouped-tests.log"; exit 1;
    }
  fi
  cargo test -p candle-nn --no-default-features --locked --lib > "$report/nn-lib-tests.log" 2>&1 || {
    tail -n 120 "$report/nn-lib-tests.log"; exit 1;
  }
  if has_feature lean4_standalone; then
    bash tools/lean4-standalone/scripts/check.sh > "$report/lean4-contracts.log" 2>&1 || {
      tail -n 120 "$report/lean4-contracts.log"; exit 1;
    }
  fi
  jq -n --arg sha "$SHA" --arg campaign "$CAMPAIGN" --arg manifest "$MANIFEST_SHA" \
    --arg lock "$lock" --argjson features "$(jq '[.features[].feature]' "$manifest")" \
    '{status:"CPU_PASSED",candidate_sha:$sha,campaign:$campaign,
      manifest_sha256:$manifest,lock_sha256:$lock,
      feature_scope:$features,gpu_executed:false}' > "$report/report.json"
  echo "lock_sha256=$lock" >> "$GITHUB_OUTPUT"
  echo "Extended combined CPU PASS at $SHA; GPU not executed" >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi

case "$FEATURE" in
  lean4_standalone)
    bash tools/lean4-standalone/scripts/check.sh > "$report/cpu-tests.log" 2>&1 ;;
  fp4_candle)
    cargo test -p candle-core --no-default-features --locked       --test mxfp4_tests --test nvfp4_experiment > "$report/cpu-tests.log" 2>&1 ;;
  asd_core_v3_standalone)
    cargo test -p candle-core --no-default-features --locked       --test grouped_conv_core_tests --test grouped_conv_transpose_core_tests       -- --test-threads=1 > "$report/cpu-tests.log" 2>&1 &&
    cargo test -p candle-nn --no-default-features --locked --lib       >> "$report/cpu-tests.log" 2>&1 ;;
  embeddinggemma2)
    cargo check -p candle-transformers --lib --locked > "$report/embedding-gemma2-transformers.log" 2>&1 &&
    cargo check -p candle-examples --example embedding-gemma2 --locked > "$report/embedding-gemma2-cli.log" 2>&1 &&
    cargo test -p candle-transformers embedding_gemma2::tests --lib --locked > "$report/cpu-tests.log" 2>&1 ;;
  moe_simt_f16_candle)
    rustc --edition=2021 --test candle-kernels/src/moe_selection.rs       -o "$RUNNER_TEMP/extended-moe-selection" > "$report/cpu-tests.log" 2>&1 &&
    "$RUNNER_TEMP/extended-moe-selection" --nocapture >> "$report/cpu-tests.log" 2>&1 ;;
  bf16_candle|fp8_candle|cudnn_fallback_candle)
    cargo test -p candle-core --no-default-features --locked --lib       > "$report/cpu-tests.log" 2>&1 ;;
  *) echo "::error::Unsupported Extended standalone CPU suite $FEATURE"; exit 3 ;;
esac || {
  tail -n 120 "$report/cpu-tests.log"; exit 1;
}

jq -n --arg feature "$FEATURE" --arg src "$SOURCE" --arg sha "$SHA"   --arg lock "$lock"   '{feature:$feature,source_sha:$src,prepared_sha:$sha,
    lock_sha256:$lock,status:"CPU_PASSED"}' > "$report/report.json"
echo "Independent $FEATURE CPU PASS at $SHA" >> "$GITHUB_STEP_SUMMARY"
