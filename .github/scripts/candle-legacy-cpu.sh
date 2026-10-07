#!/usr/bin/env bash
# Hosted CPU gate for the combined five-feature legacy-cuda candidate.
set -euo pipefail
: "${SHA:?}" "${MANIFEST_SHA:?}" "${CAMPAIGN:?}" "${RUNNER_TEMP:?}"
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || exit 3
report="$RUNNER_TEMP/legacy-integrated-cpu"
mkdir -p "$report"
manifest=candle-integration/legacy-cuda.json
[[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$MANIFEST_SHA" ]] || exit 3
jq -e --arg c "$CAMPAIGN" '
  .schema_version==1 and .kind=="legacy-cuda-five" and .campaign==$c and
  .integration_target=="legacy-cuda" and
  [.features[].feature]==["bf16_candle","fp8_candle","fp4_candle",
    "cudnn_fallback_candle","moe_simt_f16_candle"]
' "$manifest" >/dev/null || exit 3
while IFS=$'\t' read -r feature prepared; do
  git merge-base --is-ancestor "$prepared" HEAD || {
    echo "::error::Integrated HEAD lacks prepared $feature=$prepared"; exit 3;
  }
done < <(jq -r '.features[]|[.feature,.prepared_sha]|@tsv' "$manifest")

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

cargo test -p candle-core --no-default-features --locked --lib   > "$report/core-lib-tests.log" 2>&1 || {
    tail -n 120 "$report/core-lib-tests.log"; exit 1;
  }

cargo test -p candle-core --no-default-features --locked   --test mxfp4_tests --test nvfp4_experiment   > "$report/fp4-reference-tests.log" 2>&1 || {
    tail -n 120 "$report/fp4-reference-tests.log"; exit 1;
  }

rustc --edition=2021 --test candle-kernels/src/moe_selection.rs   -o "$RUNNER_TEMP/legacy-moe-selection" > "$report/moe-selection-build.log" 2>&1
"$RUNNER_TEMP/legacy-moe-selection" --nocapture > "$report/moe-selection-tests.log" 2>&1

jq -n --arg sha "$SHA" --arg campaign "$CAMPAIGN"   --arg manifest "$MANIFEST_SHA" --arg lock "$lock"   '{status:"CPU_PASSED",candidate_sha:$sha,campaign:$campaign,
    manifest_sha256:$manifest,lock_sha256:$lock,
    feature_scope:["bf16","fp8","fp4","cudnn-fallback","moe"],
    gpu_executed:false}' > "$report/report.json"
echo "lock_sha256=$lock" >> "$GITHUB_OUTPUT"
echo "legacy-cuda combined CPU PASS at $SHA; GPU not executed" >> "$GITHUB_STEP_SUMMARY"
