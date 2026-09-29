#!/usr/bin/env bash
# Rebase six standalone sources in detached worktrees, then merge into a seventh.
# Only temporary refs may be published here. Permanent refs are GPU-gated.
set -euo pipefail
umask 077
: "${CAMPAIGN:?}" "${STRATEGY:?}" "${PUBLISH:?}" "${GITHUB_REPOSITORY:?}"
[[ "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$ ]] || exit 2
[[ "$STRATEGY" == rebase || "$STRATEGY" == merge ]] || exit 2
[[ "$PUBLISH" == true || "$PUBLISH" == false ]] || exit 2
cfg=.github/standalone-six-sources.json
jq -e '.schema_version==2 and .kind=="six-standalone-main-rebase" and
  .integration_target=="cuda_asd_runner_v3" and
  [.features[].feature]==["bf16_candle","fp8_candle","fp4_candle",
    "cudnn_fallback_candle","moe_simt_f16_candle","asd_core"] and
  all(.features[]; .source_ref==(.feature+"_standalone") and
    .target_ref==.feature and (.initial_source_sha|test("^[a-f0-9]{40}$")))' "$cfg" >/dev/null
report="$RUNNER_TEMP/candle-standalone-six"
scratch="$RUNNER_TEMP/candle-six-worktrees"
mkdir -p "$report/logs" "$report/conflicts" "$report/diffs" "$scratch"
echo "campaign=$CAMPAIGN" >> "$GITHUB_OUTPUT"
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
gh auth setup-git
git fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/six-main'
base="$(git rev-parse refs/remotes/origin/six-main)"
if [[ -n "${BASE_SHA_INPUT:-}" && "$BASE_SHA_INPUT" != "$base" ]]; then
  echo "::error::Fork main moved since requested SHA"; exit 3
fi
echo "base_sha=$base" >> "$GITHUB_OUTPUT"
target="integration/candidate/$CAMPAIGN"
if git ls-remote --exit-code --heads origin "refs/heads/$target" >/dev/null 2>&1; then
  echo "::error::Candidate ref already exists; use a new campaign"; exit 3
fi
mapfile -t features < <(jq -r '.features[].feature' "$cfg")
if [[ "${PUBLISH}" == true && "${AUTO_PROMOTE:-false}" == true ]]; then
  # Fail before allocating a GPU if the future atomic promotion could only
  # succeed by rewriting an open PR's head (including BF16 test-only PR #8).
  open="$(gh api --paginate "repos/$GITHUB_REPOSITORY/pulls?state=open&per_page=100" | jq -s 'add')"
  for feature in bf16_candle fp8_candle fp4_candle cudnn_fallback_candle moe_simt_f16_candle asd_core; do
    if jq -e --arg f "$feature" 'any(.[]; .head.ref==$f)' <<<"$open" >/dev/null; then
      echo "::error::Open PR on $feature blocks post-GPU atomic promotion; close test-only PR #8 when ready, without merging it to main."
      exit 3
    fi
  done
fi
declare -A pinned oldtarget prepared
for feature in "${features[@]}"; do
  source="${feature}_standalone"
  git fetch --no-tags origin "+refs/heads/$source:refs/remotes/origin/six-$feature"
  pinned[$feature]="$(git rev-parse "refs/remotes/origin/six-$feature")"
  oldtarget[$feature]="$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)"
  [[ "${oldtarget[$feature]}" =~ ^[a-f0-9]{40}$ ]] || exit 3
  git merge-base "$base" "${pinned[$feature]}" >/dev/null || exit 3
  if git merge-base --is-ancestor ac983d16750a136a8563def01926ee83a7831d22 "${pinned[$feature]}"; then
    echo "::error::$source already contains the previous six-way aggregate"; exit 3
  fi
done
oldintegration="$(git ls-remote --heads origin refs/heads/cuda_asd_runner_v3 | cut -f1)"
: > "$report/sources.ndjson"
for feature in "${features[@]}"; do
  sha="${pinned[$feature]}"
  work="$scratch/$feature"
  git worktree add --detach "$work" "$sha" > "$report/logs/$feature-worktree.log" 2>&1
  git -C "$work" config rerere.enabled true
  git -C "$work" config rerere.autoupdate false
  common="$(git merge-base "$base" "$sha")"
  if [[ "$STRATEGY" == rebase ]]; then
    if ! git -C "$work" rebase --rebase-merges --onto "$base" "$common" \
       > "$report/logs/$feature-update.log" 2>&1; then
      git -C "$work" ls-files -u > "$report/conflicts/$feature-index.txt"
      git -C "$work" diff --name-only --diff-filter=U > "$report/conflicts/$feature-files.txt"
      git -C "$work" status --short > "$report/conflicts/$feature-status.txt"
      echo "::error::Standalone rebase conflict in $feature"; exit 1
    fi
  elif ! git -C "$work" merge --no-ff --no-edit -m "prepare($feature): main $base" "$base" \
       > "$report/logs/$feature-update.log" 2>&1; then
    git -C "$work" ls-files -u > "$report/conflicts/$feature-index.txt"
    git -C "$work" diff --name-only --diff-filter=U > "$report/conflicts/$feature-files.txt"
    git -C "$work" status --short > "$report/conflicts/$feature-status.txt"
    echo "::error::Standalone merge conflict in $feature"; exit 1
  fi
  prepared[$feature]="$(git -C "$work" rev-parse HEAD)"
  git merge-base --is-ancestor "$base" "${prepared[$feature]}" || exit 3
  git diff --binary "$base" "${prepared[$feature]}" > "$report/diffs/$feature.patch"
  if [[ "$STRATEGY" == rebase ]]; then
    git range-diff "$common..$sha" "$base..${prepared[$feature]}" \
      > "$report/diffs/$feature-range-diff.txt" || :
  fi
  jq -cn --arg feature "$feature" --arg source_ref "${feature}_standalone" \
    --arg target_ref "$feature" --arg source_sha "$sha" \
    --arg old_target_sha "${oldtarget[$feature]}" \
    --arg prepared_sha "${prepared[$feature]}" \
    --arg preparation_ref "prepare/$CAMPAIGN/$feature" \
    --arg common_base_sha "$common" \
    '{feature:$feature,source_ref:$source_ref,target_ref:$target_ref,
      source_sha:$source_sha,old_target_sha:$old_target_sha,
      prepared_sha:$prepared_sha,preparation_ref:$preparation_ref,
      common_base_sha:$common_base_sha}' >> "$report/sources.ndjson"
done
aggregate="$scratch/integration"
git worktree add --detach "$aggregate" "$base" > "$report/logs/integration-worktree.log" 2>&1
git -C "$aggregate" config rerere.enabled true
git -C "$aggregate" config rerere.autoupdate false
for feature in "${features[@]}"; do
  if ! git -C "$aggregate" merge --no-ff --no-edit \
    -m "integrate($feature): prepared ${prepared[$feature]}" \
    "${prepared[$feature]}" > "$report/logs/integrate-$feature.log" 2>&1; then
    # A reviewed BF16→FP8 four-file union exists ONLY for the exact Git
    # stage blobs recorded by the earlier six-source preview. Unknown paths
    # or changed preimages must never inherit this resolution.
    if [[ "$feature" == fp8_candle ]] &&
      bash .github/scripts/candle-resolve-bf16-fp8.sh "$aggregate" \
        "${pinned[bf16_candle]}" "${pinned[fp8_candle]}" "$report"; then
      echo "Replayed exact-stage BF16+FP8 resolution from this reviewed PR" \
        >> "$GITHUB_STEP_SUMMARY"
    else
      git -C "$aggregate" ls-files -u > "$report/conflicts/integrate-$feature-index.txt"
      git -C "$aggregate" diff --name-only --diff-filter=U > "$report/conflicts/integrate-$feature-files.txt"
      git -C "$aggregate" status --short > "$report/conflicts/integrate-$feature-status.txt"
      # Preserve all three Git conflict stages for a single auditable review.
      while IFS= read -r -d '' conflict; do
        for stage in 1 2 3; do
          dest="$report/conflicts/integrate-$feature/stage-$stage/$conflict"
          mkdir -p "$(dirname "$dest")"
          git -C "$aggregate" show ":$stage:$conflict" > "$dest" 2>/dev/null || rm -f "$dest"
        done
      done < <(git -C "$aggregate" diff --name-only --diff-filter=U -z)
      echo "::error::Novel integration conflict at $feature; no refs published"; exit 1
    fi
  fi
done
for feature in "${features[@]}"; do
  git -C "$aggregate" merge-base --is-ancestor "${prepared[$feature]}" HEAD || exit 3
done
mkdir -p "$aggregate/candle-integration"
jq -s --arg main_sha "$base" --arg campaign "$CAMPAIGN" \
  --arg strategy "$STRATEGY" --arg old_integration_sha "$oldintegration" \
  --arg config_sha256 "$(sha256sum "$cfg" | cut -d' ' -f1)" \
  '{schema_version:2,kind:"standalone-six",main_sha:$main_sha,
    campaign:$campaign,strategy:$strategy,
    integration_target:"cuda_asd_runner_v3",
    old_integration_sha:$old_integration_sha,
    source_config_sha256:$config_sha256,features:.}' "$report/sources.ndjson" \
    > "$aggregate/candle-integration/standalone-six.json"
cp "$aggregate/candle-integration/standalone-six.json" "$report/standalone-six.json"
git -C "$aggregate" add candle-integration/standalone-six.json
git -C "$aggregate" diff --cached --check
git -C "$aggregate" commit -m "integrate: lock six distinct source and prepared SHAs" \
  > "$report/logs/manifest-commit.log" 2>&1
sha="$(git -C "$aggregate" rev-parse HEAD)"
echo "candidate_sha=$sha" >> "$GITHUB_OUTPUT"
echo "candidate_ref=$target" >> "$GITHUB_OUTPUT"
echo "manifest_sha256=$(sha256sum "$report/standalone-six.json" | cut -d' ' -f1)" >> "$GITHUB_OUTPUT"
jq -cns '[inputs | {feature,source_sha,prepared_sha,preparation_ref}] | {include:.}' \
  < "$report/sources.ndjson" > "$report/matrix.json"
echo "matrix=$(cat "$report/matrix.json")" >> "$GITHUB_OUTPUT"
if [[ "$PUBLISH" != true ]]; then
  echo "published=false" >> "$GITHUB_OUTPUT"
  echo "PREVIEW ONLY: six rebases and one aggregate prepared; no refs modified" >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi
[[ "$(git ls-remote --heads origin refs/heads/main | cut -f1)" == "$base" ]] || exit 3
refs=()
for feature in "${features[@]}"; do
  [[ "$(git ls-remote --heads origin "refs/heads/${feature}_standalone" | cut -f1)" == "${pinned[$feature]}" ]] || exit 3
  [[ "$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)" == "${oldtarget[$feature]}" ]] || exit 3
  temp="prepare/$CAMPAIGN/$feature"
  if git ls-remote --exit-code --heads origin "refs/heads/$temp" >/dev/null 2>&1; then
    echo "::error::Temporary ref already exists: $temp"; exit 3
  fi
  refs+=("${prepared[$feature]}:refs/heads/$temp")
done
[[ "$(git ls-remote --heads origin refs/heads/cuda_asd_runner_v3 | cut -f1)" == "$oldintegration" ]] || exit 3
refs+=("$sha:refs/heads/$target")
git push --atomic origin "${refs[@]}" > "$report/logs/publish.log" 2>&1 || {
  echo "::error::Atomic temporary publish failed, no permanent refs touched"; exit 3
}
echo "published=true" >> "$GITHUB_OUTPUT"
echo "Published only seven temporary refs; all standalone/public refs untouched" >> "$GITHUB_STEP_SUMMARY"
