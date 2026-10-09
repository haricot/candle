#!/usr/bin/env bash
# Build the five-feature legacy-cuda candidate from exact standalone deltas.
# This stage may publish immutable temporary refs only. Stable refs are promoted separately.
set -euo pipefail
umask 077
: "${CAMPAIGN:?}" "${STRATEGY:?}" "${PUBLISH:?}" "${GITHUB_REPOSITORY:?}"
[[ "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$ ]] || exit 2
[[ "$STRATEGY" == exact-delta || "$STRATEGY" == merge ]] || exit 2
[[ "$PUBLISH" == true || "$PUBLISH" == false ]] || exit 2

cfg=.github/legacy-cuda-sources.json
jq -e '
  .schema_version==1 and .kind=="legacy-cuda-main-exact-delta" and
  .integration_target=="legacy-cuda" and
  [.features[].feature]==["bf16_candle","fp8_candle","fp4_candle",
    "cudnn_fallback_candle","moe_simt_f16_candle"] and
  all(.features[];
    (.source_ref|type)=="string" and
    (.target_ref==.feature) and
    (.review_source_sha|test("^[a-f0-9]{40}$")))
' "$cfg" >/dev/null

report="$RUNNER_TEMP/candle-legacy-cuda"
scratch="$RUNNER_TEMP/candle-legacy-worktrees"
mkdir -p "$report/logs" "$report/conflicts" "$report/diffs" "$scratch"
echo "campaign=$CAMPAIGN" >> "$GITHUB_OUTPUT"

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
gh auth setup-git
git fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/legacy-main'
base="$(git rev-parse refs/remotes/origin/legacy-main)"
if [[ -n "${BASE_SHA_INPUT:-}" && "${BASE_SHA_INPUT}" != "$base" ]]; then
  echo "::error::Fork main moved since requested SHA"; exit 3
fi
echo "base_sha=$base" >> "$GITHUB_OUTPUT"

target="integration/legacy-cuda/candidate/$CAMPAIGN"
if git ls-remote --exit-code --heads origin "refs/heads/$target" >/dev/null 2>&1; then
  echo "::error::Candidate ref already exists; choose a new campaign"; exit 3
fi

mapfile -t features < <(jq -r '.features[].feature' "$cfg")
declare -A pinned oldtarget prepared review source_ref

for feature in "${features[@]}"; do
  source_ref[$feature]="$(jq -r --arg f "$feature" '.features[]|select(.feature==$f)|.source_ref' "$cfg")"
  review[$feature]="$(jq -r --arg f "$feature" '.features[]|select(.feature==$f)|.review_source_sha' "$cfg")"
  git fetch --no-tags origin "+refs/heads/${source_ref[$feature]}:refs/remotes/origin/legacy-$feature"
  pinned[$feature]="$(git rev-parse "refs/remotes/origin/legacy-$feature")"
  oldtarget[$feature]="${pinned[$feature]}"
  git merge-base "$base" "${pinned[$feature]}" >/dev/null || exit 3
done

oldintegration="$(git ls-remote --heads origin refs/heads/cuda_legacy | cut -f1 || true)"
: > "$report/sources.ndjson"

for feature in "${features[@]}"; do
  sha="${pinned[$feature]}"
  work="$scratch/$feature"
  common="$(git merge-base "$base" "$sha")"
  source_patch="$report/diffs/$feature-source.patch"
  prepared_patch="$report/diffs/$feature-prepared.patch"
  git diff --binary --full-index "$common" "$sha" > "$source_patch"
  source_patch_sha="$(sha256sum "$source_patch" | cut -d' ' -f1)"

  if [[ "$STRATEGY" == exact-delta ]]; then
    git worktree add --detach "$work" "$base" > "$report/logs/$feature-worktree.log" 2>&1
    git -C "$work" config rerere.enabled false
    if ! git -C "$work" apply --3way --index "$source_patch" > "$report/logs/$feature-update.log" 2>&1; then
      git -C "$work" ls-files -u > "$report/conflicts/$feature-index.txt"
      git -C "$work" diff --name-only --diff-filter=U > "$report/conflicts/$feature-files.txt"
      git -C "$work" status --short > "$report/conflicts/$feature-status.txt"
      echo "::error::Exact feature delta conflicts with main for $feature"; exit 1
    fi
    git -C "$work" diff --cached --check
    git -C "$work" diff --cached --quiet && {
      echo "::error::Exact feature delta unexpectedly empty for $feature"; exit 3;
    }
    git -C "$work" commit -m "prepare($feature): exact source delta $sha on main $base"       >> "$report/logs/$feature-update.log" 2>&1
  else
    git worktree add --detach "$work" "$sha" > "$report/logs/$feature-worktree.log" 2>&1
    if ! git -C "$work" merge --no-ff --no-edit -m "prepare($feature): main $base" "$base"        > "$report/logs/$feature-update.log" 2>&1; then
      git -C "$work" ls-files -u > "$report/conflicts/$feature-index.txt"
      git -C "$work" diff --name-only --diff-filter=U > "$report/conflicts/$feature-files.txt"
      echo "::error::Feature merge conflict in $feature"; exit 1
    fi
  fi

  prepared[$feature]="$(git -C "$work" rev-parse HEAD)"
  git merge-base --is-ancestor "$base" "${prepared[$feature]}" || exit 3
  git diff --binary --full-index "$base" "${prepared[$feature]}" > "$prepared_patch"
  prepared_patch_sha="$(sha256sum "$prepared_patch" | cut -d' ' -f1)"
  if [[ "$STRATEGY" == exact-delta ]] && ! cmp -s "$source_patch" "$prepared_patch"; then
    diff -u "$source_patch" "$prepared_patch" > "$report/diffs/$feature-delta-mismatch.diff" || :
    echo "::error::Prepared $feature changed its exact standalone delta"; exit 3
  fi

  jq -cn --arg feature "$feature"     --arg source_ref "${source_ref[$feature]}" --arg target_ref "$feature"     --arg source_sha "$sha" --arg review_source_sha "${review[$feature]}"     --arg old_target_sha "${oldtarget[$feature]}"     --arg prepared_sha "${prepared[$feature]}"     --arg preparation_ref "prepare/legacy/$CAMPAIGN/$feature"     --arg common_base_sha "$common"     --arg source_patch_sha256 "$source_patch_sha"     --arg prepared_patch_sha256 "$prepared_patch_sha"     '{feature:$feature,source_ref:$source_ref,target_ref:$target_ref,
      source_sha:$source_sha,review_source_sha:$review_source_sha,
      old_target_sha:$old_target_sha,prepared_sha:$prepared_sha,
      preparation_ref:$preparation_ref,common_base_sha:$common_base_sha,
      source_patch_sha256:$source_patch_sha256,
      prepared_patch_sha256:$prepared_patch_sha256}' >> "$report/sources.ndjson"
done

aggregate="$scratch/integration"
git worktree add --detach "$aggregate" "$base" > "$report/logs/integration-worktree.log" 2>&1
git -C "$aggregate" config rerere.enabled false

for feature in "${features[@]}"; do
  if ! git -C "$aggregate" merge --no-ff --no-edit     -m "integrate($feature): prepared ${prepared[$feature]}"     "${prepared[$feature]}" > "$report/logs/integrate-$feature.log" 2>&1; then
    resolved=false
    if [[ "$feature" == fp8_candle ]] &&
      bash .github/scripts/candle-resolve-bf16-fp8.sh "$aggregate"         "${review[bf16_candle]}" "${review[fp8_candle]}" "$report"; then
      resolved=true
    elif [[ "$feature" == fp4_candle ]] &&
      bash .github/scripts/candle-resolve-bf16-fp8-fp4.sh "$aggregate"         "${review[bf16_candle]}" "${review[fp8_candle]}"         "${review[fp4_candle]}" "$report"; then
      resolved=true
    elif [[ "$feature" == cudnn_fallback_candle ]] &&
      bash .github/scripts/candle-resolve-bf16-fp8-fp4-cudnn.sh "$aggregate"         "${review[bf16_candle]}" "${review[fp8_candle]}"         "${review[fp4_candle]}" "${review[cudnn_fallback_candle]}" "$report"; then
      resolved=true
    elif [[ "$feature" == moe_simt_f16_candle ]] &&
      bash .github/scripts/candle-resolve-bf16-fp8-fp4-moe.sh "$aggregate"         "${review[bf16_candle]}" "${review[fp8_candle]}"         "${review[fp4_candle]}" "${review[cudnn_fallback_candle]}"         "${review[moe_simt_f16_candle]}" "$report"; then
      resolved=true
    fi

    if [[ "$resolved" == true ]]; then
      echo "Replayed reviewed exact-stage resolution for $feature" >> "$GITHUB_STEP_SUMMARY"
    else
      git -C "$aggregate" ls-files -u > "$report/conflicts/integrate-$feature-index.txt"
      git -C "$aggregate" diff --name-only --diff-filter=U > "$report/conflicts/integrate-$feature-files.txt"
      git -C "$aggregate" status --short > "$report/conflicts/integrate-$feature-status.txt"
      while IFS= read -r -d '' conflict; do
        for stage in 1 2 3; do
          dest="$report/conflicts/integrate-$feature/stage-$stage/$conflict"
          mkdir -p "$(dirname "$dest")"
          git -C "$aggregate" show ":$stage:$conflict" > "$dest" 2>/dev/null || rm -f "$dest"
        done
      done < <(git -C "$aggregate" diff --name-only --diff-filter=U -z)
      echo "::error::Novel legacy-cuda integration conflict at $feature"; exit 1
    fi
  fi
done

for feature in "${features[@]}"; do
  git -C "$aggregate" merge-base --is-ancestor "${prepared[$feature]}" HEAD || exit 3
done

mkdir -p "$aggregate/candle-integration"
jq -s --arg main_sha "$base" --arg campaign "$CAMPAIGN"   --arg strategy "$STRATEGY" --arg old_integration_sha "$oldintegration"   --arg config_sha256 "$(sha256sum "$cfg" | cut -d' ' -f1)"   '{schema_version:1,kind:"legacy-cuda-five",main_sha:$main_sha,
    campaign:$campaign,strategy:$strategy,integration_target:"legacy-cuda",
    old_integration_sha:(if $old_integration_sha=="" then null else $old_integration_sha end),
    source_config_sha256:$config_sha256,
    validation_state:"UNVALIDATED_CANDIDATE",features:.}'   "$report/sources.ndjson" > "$aggregate/candle-integration/legacy-cuda.json"
cp "$aggregate/candle-integration/legacy-cuda.json" "$report/legacy-cuda.json"
git -C "$aggregate" add candle-integration/legacy-cuda.json
git -C "$aggregate" diff --cached --check
git -C "$aggregate" commit -m "integrate: lock five legacy CUDA source and prepared SHAs"   > "$report/logs/manifest-commit.log" 2>&1

sha="$(git -C "$aggregate" rev-parse HEAD)"
echo "candidate_sha=$sha" >> "$GITHUB_OUTPUT"
echo "candidate_ref=$target" >> "$GITHUB_OUTPUT"
echo "manifest_sha256=$(sha256sum "$report/legacy-cuda.json" | cut -d' ' -f1)" >> "$GITHUB_OUTPUT"

jq -cn '[inputs | {feature,source_sha,prepared_sha,preparation_ref}] | {include:.}'   < "$report/sources.ndjson" > "$report/matrix.json"
jq -e '
  (.include|length)==5 and
  [.include[].feature]==["bf16_candle","fp8_candle","fp4_candle",
    "cudnn_fallback_candle","moe_simt_f16_candle"]
' "$report/matrix.json" >/dev/null
echo "matrix=$(cat "$report/matrix.json")" >> "$GITHUB_OUTPUT"

if [[ "$PUBLISH" != true ]]; then
  echo "published=false" >> "$GITHUB_OUTPUT"
  echo "PREVIEW ONLY: legacy-cuda candidate prepared; no refs modified" >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi

[[ "$(git ls-remote --heads origin refs/heads/main | cut -f1)" == "$base" ]] || exit 3
refs=()
for feature in "${features[@]}"; do
  [[ "$(git ls-remote --heads origin "refs/heads/${source_ref[$feature]}" | cut -f1)" == "${pinned[$feature]}" ]] || exit 3
  temp="prepare/legacy/$CAMPAIGN/$feature"
  if git ls-remote --exit-code --heads origin "refs/heads/$temp" >/dev/null 2>&1; then
    echo "::error::Temporary ref already exists: $temp"; exit 3
  fi
  refs+=("${prepared[$feature]}:refs/heads/$temp")
done
refs+=("$sha:refs/heads/$target")
git push --atomic origin "${refs[@]}" > "$report/logs/publish.log" 2>&1 || {
  echo "::error::Atomic temporary publish failed; stable refs untouched"; exit 3
}
echo "published=true" >> "$GITHUB_OUTPUT"
echo "Published five prepared refs plus one immutable legacy-cuda candidate; stable refs untouched"   >> "$GITHUB_STEP_SUMMARY"
