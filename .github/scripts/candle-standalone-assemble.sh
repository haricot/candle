#!/usr/bin/env bash
# Prepare exact pinned FP8/FP4 standalone sources in detached worktrees.
# Never push or reset a standalone source branch.
set -euo pipefail
umask 077
: "${CAMPAIGN:?}" "${STRATEGY:?}" "${PUBLISH:?}" "${GITHUB_REPOSITORY:?}"
[[ "$CAMPAIGN" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}$ ]] || exit 2
[[ "$STRATEGY" == merge || "$STRATEGY" == rebase ]] || exit 2
report="$RUNNER_TEMP/candle-standalone-prepare"
scratch="$RUNNER_TEMP/candle-standalone-worktrees"
mkdir -p "$report/logs" "$report/conflicts" "$report/diffs" "$scratch"
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
gh auth setup-git
git fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/pinned-main'
base="$(git rev-parse refs/remotes/origin/pinned-main)"
if [[ -n "${BASE_SHA_INPUT:-}" ]]; then
  [[ "$BASE_SHA_INPUT" =~ ^[a-f0-9]{40}$ && "$BASE_SHA_INPUT" == "$base" ]] || {
    echo "::error::main advanced; start a new campaign with current base_sha"; exit 3;
  }
fi
echo "base_sha=$base" >> "$GITHUB_OUTPUT"
echo "campaign=$CAMPAIGN" >> "$GITHUB_OUTPUT"
target="integration/standalone-$CAMPAIGN-fp8-fp4"
if git ls-remote --exit-code --heads origin "refs/heads/$target" >/dev/null 2>&1; then
  echo "::error::Candidate ref already exists; use a new campaign ID"; exit 3;
fi
declare -A pinned prepared
for feature in fp8_candle fp4_candle; do
  source="${feature}_standalone"
  git fetch --no-tags origin \
    "+refs/heads/$source:refs/remotes/origin/pinned-$feature"
  pinned[$feature]="$(git rev-parse "refs/remotes/origin/pinned-$feature")"
  git merge-base "$base" "${pinned[$feature]}" >/dev/null || {
    echo "::error::No common ancestry with pinned fork main"; exit 3;
  }
done
: > "$report/sources.ndjson"
for feature in fp8_candle fp4_candle; do
  source="${feature}_standalone"
  source_sha="${pinned[$feature]}"
  work="$scratch/$feature"
  git worktree add --detach "$work" "$source_sha" > "$report/logs/$feature-worktree.log" 2>&1
  git -C "$work" config rerere.enabled true
  git -C "$work" config rerere.autoupdate false
  common="$(git merge-base "$base" "$source_sha")"
  if [[ "$STRATEGY" == rebase ]]; then
    if ! git -C "$work" rebase --rebase-merges --onto "$base" "$common" \
        > "$report/logs/$feature-update.log" 2>&1; then
      git -C "$work" ls-files -u > "$report/conflicts/$feature-index.txt"
      git -C "$work" diff --name-only --diff-filter=U > "$report/conflicts/$feature-files.txt"
      git -C "$work" status --short > "$report/conflicts/$feature-status.txt"
      echo "::error::$feature rebase conflict; original standalone ref unchanged"; exit 1
    fi
  elif ! git -C "$work" merge --no-ff --no-edit -m \
       "prepare($feature): merge exact main $base into detached $source_sha" "$base" \
       > "$report/logs/$feature-update.log" 2>&1; then
    git -C "$work" ls-files -u > "$report/conflicts/$feature-index.txt"
    git -C "$work" diff --name-only --diff-filter=U > "$report/conflicts/$feature-files.txt"
    git -C "$work" status --short > "$report/conflicts/$feature-status.txt"
    echo "::error::$feature merge conflict; original standalone ref unchanged"; exit 1
  fi
  prepared[$feature]="$(git -C "$work" rev-parse HEAD)"
  git merge-base --is-ancestor "$base" "${prepared[$feature]}" || exit 3
  git diff --binary "$base" "${prepared[$feature]}" > "$report/diffs/$feature.patch"
  if [[ "$STRATEGY" == rebase ]]; then
    git range-diff "$common..$source_sha" "$base..${prepared[$feature]}" \
      > "$report/diffs/$feature-range-diff.txt" || :
  fi
  jq -cn --arg feature "$feature" --arg source_ref "$source" \
     --arg source_sha "$source_sha" --arg prepared_sha "${prepared[$feature]}" \
     --arg common_base_sha "$common" --arg strategy "$STRATEGY" \
     '{feature:$feature,source_ref:$source_ref,source_sha:$source_sha,
       prepared_sha:$prepared_sha,common_base_sha:$common_base_sha,strategy:$strategy}' \
     >> "$report/sources.ndjson"
done
assembled="$scratch/integration"
git worktree add --detach "$assembled" "$base" > "$report/logs/integration-worktree.log" 2>&1
git -C "$assembled" config rerere.enabled true
git -C "$assembled" config rerere.autoupdate false
for feature in fp8_candle fp4_candle; do
  if ! git -C "$assembled" merge --no-ff --no-edit \
      -m "integrate($feature): detached prepared ${prepared[$feature]}" \
      "${prepared[$feature]}" > "$report/logs/integrate-$feature.log" 2>&1; then
    git -C "$assembled" ls-files -u > "$report/conflicts/integration-$feature-index.txt"
    git -C "$assembled" diff --name-only --diff-filter=U \
      > "$report/conflicts/integration-$feature-files.txt"
    git -C "$assembled" status --short > "$report/conflicts/integration-$feature-status.txt"
    echo "::error::Aggregate conflict in $feature; candidate not published"; exit 1
  fi
done
for feature in fp8_candle fp4_candle; do
  git -C "$assembled" merge-base --is-ancestor "${prepared[$feature]}" HEAD || exit 3
done
mkdir -p "$assembled/candle-integration"
jq -s --arg base_sha "$base" --arg strategy "$STRATEGY" --arg campaign "$CAMPAIGN" \
  '{schema_version:1,kind:"standalone-fp8-fp4",base_sha:$base_sha,
    campaign:$campaign,strategy:$strategy,features:.}' \
  "$report/sources.ndjson" > "$assembled/candle-integration/standalone-sources.json"
cp "$assembled/candle-integration/standalone-sources.json" "$report/standalone-sources.json"
git -C "$assembled" add candle-integration/standalone-sources.json
git -C "$assembled" diff --cached --check
git -C "$assembled" commit -m 'integrate: pin isolated FP8/FP4 prepared and source SHAs' \
  > "$report/logs/manifest-commit.log" 2>&1
sha="$(git -C "$assembled" rev-parse HEAD)"
printf 'base=%s\nfp8=%s -> %s\nfp4=%s -> %s\nintegrated=%s\n' \
  "$base" "${pinned[fp8_candle]}" "${prepared[fp8_candle]}" \
  "${pinned[fp4_candle]}" "${prepared[fp4_candle]}" "$sha" > "$report/summary.txt"
echo "candidate_sha=$sha" >> "$GITHUB_OUTPUT"
echo "candidate_ref=$target" >> "$GITHUB_OUTPUT"
if [[ "$PUBLISH" != true ]]; then
  echo "published=false" >> "$GITHUB_OUTPUT"
  echo "PREVIEW ONLY: $target would point to $sha; source refs untouched" >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi
# Recheck immutable source/base before creating the ONLY remote ref.
[[ "$(git ls-remote --heads origin refs/heads/main | cut -f1)" == "$base" ]] || exit 3
for feature in fp8_candle fp4_candle; do
  [[ "$(git ls-remote --heads origin "refs/heads/${feature}_standalone" | cut -f1)" == "${pinned[$feature]}" ]] || exit 3
done
git push origin "$sha:refs/heads/$target" > "$report/logs/publish.log" 2>&1
[[ "$(git ls-remote --heads origin "refs/heads/$target" | cut -f1)" == "$sha" ]] || exit 3
echo "published=true" >> "$GITHUB_OUTPUT"
echo "Published ONLY $target at $sha; source branches unchanged" >> "$GITHUB_STEP_SUMMARY"
