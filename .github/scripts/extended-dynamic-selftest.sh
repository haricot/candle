#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
config=.github/extended-sources.json
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

for script in \
  .github/scripts/candle-extended-assemble.sh \
  .github/scripts/candle-extended-cpu.sh \
  .github/scripts/candle-extended-sm61.sh \
  .github/scripts/candle-candidate-cuda-compile.sh; do
  bash -n "$script"
done

# Exercise the earlier compositions and the current thirteen-source voice set.
# production assembler. Validate source uniqueness and candidate parity.
for count in 6 7 8 9 12 13; do
  jq --argjson count "$count" '.features |= .[:$count]' "$config" > "$scratch/config.json"
  jq -e --argjson count "$count" '
    .schema_version == 1 and .kind == "extended-main-exact-delta" and
    .integration_target == "extended" and
    (.features | length) == $count and
    ([.features[].feature] | unique | length) == $count and
    ([.features[].target_ref] | unique | length) == $count and
    all(.features[];
      (.feature | test("^[a-z][a-z0-9_]{0,63}$")) and
      (.review_source_sha | test("^[a-f0-9]{40}$")) and
      (.cpu_tests as $tests | ["default","fp4-reference","grouped-core",
        "embedding-gemma2-native","moe-selection","lean4-contracts","half-duplex-native",
        "fireredvad-native","smart-turn-native","transcribepsy-native",
        "nemotron-asr-native","pockettts-native"] | index($tests) != null))
  ' "$scratch/config.json" >/dev/null
  jq --argjson count "$count" '
    {schema_version:1,kind:"extended-multi",integration_target:"extended",
     feature_count:$count,features:[.features[]|{feature,source_ref,target_ref,review_source_sha,cuda_features}]}
  ' "$scratch/config.json" > "$scratch/integration.json"
  jq -e --slurpfile cfg "$scratch/config.json" '
    .feature_count == (.features | length) and
    [.features[]|{feature,cuda_features}] ==
      [$cfg[0].features[]|{feature,cuda_features}] and
    all(.features[]; (.cuda_features|type)=="array") and
    ([.features[].cuda_features[]] | length > 0)
  ' "$scratch/integration.json" >/dev/null
  # Regression: CUDA consumers must never receive a null capabilities field.
  jq 'del(.features[0].cuda_features)' "$scratch/integration.json" > "$scratch/missing.json"
  if jq -e 'all(.features[]; (.cuda_features | type) == "array")' \
      "$scratch/missing.json" >/dev/null; then
    echo "DYNAMIC_EXTENDED_MISSING_CUDA_METADATA=FAIL" >&2
    exit 1
  fi
  echo "DYNAMIC_EXTENDED_COUNT_$count=PASS"
done

# A duplicated component is rejected by manifest uniqueness requirements.
jq '.features += [.features[0]]' "$config" > "$scratch/duplicated.json"
if jq -e '([.features[].feature] | unique | length) == (.features | length)' \
    "$scratch/duplicated.json" >/dev/null; then
  echo "DYNAMIC_EXTENDED_DUPLICATE=FAIL" >&2
  exit 1
fi
echo "DYNAMIC_EXTENDED_DUPLICATE=REJECTED"
echo "DYNAMIC_EXTENDED_STATIC=PASS"
