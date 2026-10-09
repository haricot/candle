#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${CANDIDATE_REF:?CANDIDATE_REF is required}"
: "${CAMPAIGN:?CAMPAIGN is required}"
: "${BASE_SHA:?BASE_SHA is required}"
: "${TARGET_BRANCH:?TARGET_BRANCH is required}"
: "${MANIFEST_PATH:?MANIFEST_PATH is required}"
: "${PROMOTION_KIND:?PROMOTION_KIND is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${RUN_ATTEMPT:?RUN_ATTEMPT is required}"

report="${RUNNER_TEMP:-/tmp}/candle-promotion"
mkdir -p "$report"
manifest="$report/manifest.json"
summary="$report/summary.txt"
: > "$summary"

fail() {
  echo "::error::$*" >&2
  echo "PROMOTION=FAIL reason=$*" >> "$summary"
  exit 1
}

remote_head() {
  git ls-remote --heads origin "refs/heads/$1" | awk 'NR==1 {print $1}'
}

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
gh auth setup-git

git fetch --no-tags origin   "+refs/heads/${CANDIDATE_REF}:refs/remotes/origin/promotion-candidate"

actual_candidate="$(git rev-parse refs/remotes/origin/promotion-candidate)"
[[ "$actual_candidate" == "$CANDIDATE_SHA" ]] ||
  fail "candidate ref moved: expected $CANDIDATE_SHA, got $actual_candidate"

git show "${CANDIDATE_SHA}:${MANIFEST_PATH}" > "$manifest" ||
  fail "cannot read validated candidate manifest"

jq -e   --arg campaign "$CAMPAIGN"   --arg main "$BASE_SHA"   '.campaign == $campaign and .main_sha == $main and (.features | type == "array" and length > 0)'   "$manifest" >/dev/null ||
  fail "candidate manifest does not match campaign/base"

manifest_target="$(jq -r '.integration_target' "$manifest")"
[[ "$manifest_target" == "$TARGET_BRANCH" ]] ||
  fail "manifest integration_target=$manifest_target, expected $TARGET_BRANCH"

declare -a archive_specs=()
declare -a promote_specs=()
declare -a lease_args=()
declare -a verify_targets=()
declare -a verify_prepared=()

ensure_archive() {
  local ref="$1" sha="$2" current
  git check-ref-format "refs/heads/$ref" >/dev/null ||
    fail "invalid archive ref $ref"
  current="$(remote_head "$ref")"
  if [[ -n "$current" ]]; then
    [[ "$current" == "$sha" ]] ||
      fail "archive ref $ref already exists at unexpected SHA $current"
    echo "ARCHIVE_REUSED=$ref@$sha" >> "$summary"
  else
    archive_specs+=("$sha:refs/heads/$ref")
    echo "ARCHIVE_PLANNED=$ref@$sha" >> "$summary"
  fi
}

idx=0
while IFS=$'\t' read -r feature target source prepared; do
  [[ -n "$feature" && -n "$target" && -n "$source" && -n "$prepared" ]] ||
    fail "malformed feature entry in manifest"

  current="$(remote_head "$target")"
  [[ -n "$current" ]] ||
    fail "canonical feature ref $target is missing"

  git fetch --no-tags origin     "+refs/heads/$target:refs/remotes/origin/promotion-feature-$idx"

  ensure_archive "archive/promotion/$CAMPAIGN/before/$target" "$current"

  if [[ "$current" == "$prepared" ]]; then
    echo "FEATURE_ALREADY_PROMOTED=$target@$current" >> "$summary"
  elif [[ "$current" == "$source" ]]; then
    promote_specs+=("$prepared:refs/heads/$target")
    lease_args+=("--force-with-lease=refs/heads/$target:$source")
    echo "FEATURE_PROMOTE=$target:$source->$prepared" >> "$summary"
  else
    current_tree="$(git rev-parse "$current^{tree}")"
    prepared_tree="$(git rev-parse "$prepared^{tree}")"
    if [[ "$current_tree" == "$prepared_tree" ]]; then
      echo "FEATURE_EQUIVALENT_ALREADY_PROMOTED=$target@$current" >> "$summary"
    else
      fail "canonical feature ref $target moved unexpectedly: $current (expected $source)"
    fi
  fi

  verify_targets+=("$target")
  verify_prepared+=("$prepared")
  idx=$((idx + 1))
done < <(jq -r '.features[] | [.feature,.target_ref,.source_sha,.prepared_sha] | @tsv' "$manifest")

old_integration="$(jq -r '.old_integration_sha // empty' "$manifest")"
current_integration="$(remote_head "$TARGET_BRANCH")"

if [[ -n "$current_integration" ]]; then
  git fetch --no-tags origin     "+refs/heads/$TARGET_BRANCH:refs/remotes/origin/promotion-integration"
  ensure_archive "archive/promotion/$CAMPAIGN/before/$TARGET_BRANCH" "$current_integration"
fi

ensure_archive   "archive/$PROMOTION_KIND/sm61-validated-$RUN_ID-$RUN_ATTEMPT"   "$CANDIDATE_SHA"

if [[ "$current_integration" == "$CANDIDATE_SHA" ]]; then
  echo "INTEGRATION_ALREADY_PROMOTED=$TARGET_BRANCH@$CANDIDATE_SHA" >> "$summary"
elif [[ -n "$old_integration" && "$current_integration" == "$old_integration" ]]; then
  promote_specs+=("$CANDIDATE_SHA:refs/heads/$TARGET_BRANCH")
  lease_args+=("--force-with-lease=refs/heads/$TARGET_BRANCH:$old_integration")
  echo "INTEGRATION_PROMOTE=$TARGET_BRANCH:$old_integration->$CANDIDATE_SHA" >> "$summary"
elif [[ -z "$old_integration" && -z "$current_integration" ]]; then
  promote_specs+=("$CANDIDATE_SHA:refs/heads/$TARGET_BRANCH")
  lease_args+=("--force-with-lease=refs/heads/$TARGET_BRANCH:")
  echo "INTEGRATION_CREATE=$TARGET_BRANCH@$CANDIDATE_SHA" >> "$summary"
else
  fail "integration ref $TARGET_BRANCH moved unexpectedly: current=${current_integration:-MISSING} expected=${old_integration:-MISSING}"
fi

if (("${#archive_specs[@]}" > 0)); then
  git push --atomic origin "${archive_specs[@]}"
fi

if (("${#promote_specs[@]}" > 0)); then
  git push --atomic "${lease_args[@]}" origin "${promote_specs[@]}"
fi

for i in "${!verify_targets[@]}"; do
  target="${verify_targets[$i]}"
  prepared="${verify_prepared[$i]}"
  current="$(remote_head "$target")"
  [[ -n "$current" ]] || fail "promoted feature ref $target disappeared"
  if [[ "$current" != "$prepared" ]]; then
    git fetch --no-tags origin       "+refs/heads/$target:refs/remotes/origin/promotion-verify-$i"
    current_tree="$(git rev-parse "$current^{tree}")"
    prepared_tree="$(git rev-parse "$prepared^{tree}")"
    [[ "$current_tree" == "$prepared_tree" ]] ||
      fail "feature ref $target does not match validated prepared tree after promotion"
  fi
done

[[ "$(remote_head "$TARGET_BRANCH")" == "$CANDIDATE_SHA" ]] ||
  fail "integration ref $TARGET_BRANCH is not the validated candidate after promotion"

{
  echo "PROMOTION=PASS"
  echo "CAMPAIGN=$CAMPAIGN"
  echo "BASE_SHA=$BASE_SHA"
  echo "CANDIDATE_SHA=$CANDIDATE_SHA"
  echo "TARGET_BRANCH=$TARGET_BRANCH"
} >> "$summary"

cat "$summary"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "## Validated ref promotion"
    echo
    echo "<pre>"
    cat "$summary"
    echo "</pre>"
  } >> "$GITHUB_STEP_SUMMARY"
fi
