#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
config=.github/extended-sources.json
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

for script in \
  .github/scripts/candle-extended-assemble.sh \
  .github/scripts/candle-extended-pin-equivalence.sh \
  .github/scripts/candle-promote-validated.sh \
  .github/scripts/candle-extended-cpu.sh \
  .github/scripts/candle-extended-sm61.sh \
  .github/scripts/candle-candidate-cuda-compile.sh; do
  bash -n "$script"
done

# Exercise the earlier compositions and the current twelve-source voice set.
# production assembler. Validate source uniqueness and candidate parity.
for count in 6 7 8 9 12; do
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
        "fireredvad-native","smart-turn-native","transcribepsy-native","nemotron-asr-native"] | index($tests) != null))
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

# A previously promoted standalone may have a different commit SHA but must
# carry the identical, reviewed exact-delta patch. No arbitrary moved head passes.
jq -e '.publication.update_canonical_feature_refs == false and
  ([.features[] | select(.allow_promoted_equivalent_source == true) | .feature] | length) == 6 and
  ([.features[] | select(.allow_promoted_equivalent_source == true) | .pin_source_sha] | all)' "$config" >/dev/null

pin_repo="$scratch/pin-equivalence"
mkdir -p "$pin_repo"
git -C "$pin_repo" init -q
git -C "$pin_repo" config user.name "Extended source pin selftest"
git -C "$pin_repo" config user.email "source-pin-selftest@example.invalid"
printf 'baseline\n' > "$pin_repo/sample.txt"
git -C "$pin_repo" add sample.txt
git -C "$pin_repo" commit -qm baseline
base="$(git -C "$pin_repo" rev-parse HEAD)"
git -C "$pin_repo" checkout -qb reviewed
printf 'reviewed delta\n' > "$pin_repo/sample.txt"
git -C "$pin_repo" commit -qam reviewed
review="$(git -C "$pin_repo" rev-parse HEAD)"
git -C "$pin_repo" checkout -qb prepared "$base"
printf 'reviewed delta\n' > "$pin_repo/sample.txt"
git -C "$pin_repo" commit -qam "prepare(test_source): exact source delta $review on main $base"
prepared="$(git -C "$pin_repo" rev-parse HEAD)"

if ! (cd "$pin_repo" &&
  FEATURE=test_source REVIEW_SOURCE_SHA="$review" OBSERVED_SOURCE_SHA="$prepared" \
  CURRENT_MAIN_SHA="$base" ALLOW_PROMOTED_EQUIVALENT_SOURCE=true \
  bash "$OLDPWD/.github/scripts/candle-extended-pin-equivalence.sh" \
    > "$scratch/pin-valid.log" 2>&1); then
  echo "DYNAMIC_EXTENDED_REVIEWED_DELTA=FAIL" >&2
  sed -n '1,20p' "$scratch/pin-valid.log" >&2
  exit 1
fi

if (cd "$pin_repo" &&
  FEATURE=test_source REVIEW_SOURCE_SHA="$review" OBSERVED_SOURCE_SHA="$prepared" \
  CURRENT_MAIN_SHA="$base" ALLOW_PROMOTED_EQUIVALENT_SOURCE=false \
  bash "$OLDPWD/.github/scripts/candle-extended-pin-equivalence.sh" \
    > "$scratch/pin-strict.log" 2>&1); then
  echo "DYNAMIC_EXTENDED_UNAPPROVED_PIN=FAIL" >&2
  exit 1
fi

git -C "$pin_repo" checkout -qb altered "$base"
printf 'unreviewed mutation\n' > "$pin_repo/sample.txt"
git -C "$pin_repo" commit -qam "prepare(test_source): exact source delta $review on main $base"
altered="$(git -C "$pin_repo" rev-parse HEAD)"
if (cd "$pin_repo" &&
  FEATURE=test_source REVIEW_SOURCE_SHA="$review" OBSERVED_SOURCE_SHA="$altered" \
  CURRENT_MAIN_SHA="$base" ALLOW_PROMOTED_EQUIVALENT_SOURCE=true \
  bash "$OLDPWD/.github/scripts/candle-extended-pin-equivalence.sh" \
    > "$scratch/pin-altered.log" 2>&1); then
  echo "DYNAMIC_EXTENDED_ALTERED_DELTA=FAIL" >&2
  exit 1
fi
echo "DYNAMIC_EXTENDED_REVIEWED_DELTA=PASS"
echo "DYNAMIC_EXTENDED_UNAPPROVED_PIN=REJECTED"
echo "DYNAMIC_EXTENDED_ALTERED_DELTA=REJECTED"
echo "DYNAMIC_EXTENDED_STATIC=PASS"
