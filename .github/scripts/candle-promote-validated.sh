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

# Unified mode: one Extended CUDA/SM61 archive proves both compositions.
# Fetch and bind the independently assembled Legacy candidate BEFORE any push.
if [[ -n "${LEGACY_CANDIDATE_SHA:-}" || -n "${LEGACY_CANDIDATE_REF:-}" ]]; then
  [[ "${PROMOTION_KIND}" == extended && "${TARGET_BRANCH}" == extended ]] ||
    fail "unified promotion must originate from the Extended candidate"
  [[ "${LEGACY_CANDIDATE_SHA:-}" =~ ^[a-f0-9]{40}$ && -n "${LEGACY_CANDIDATE_REF:-}" ]] ||
    fail "unified promotion needs both Legacy candidate SHA and ref"
  git fetch --no-tags origin \
    "+refs/heads/${LEGACY_CANDIDATE_REF}:refs/remotes/origin/promotion-legacy-candidate"
  [[ "$(git rev-parse refs/remotes/origin/promotion-legacy-candidate)" == "$LEGACY_CANDIDATE_SHA" ]] ||
    fail "Legacy candidate ref moved"
  legacy_manifest="$report/legacy-manifest.json"
  git show "${LEGACY_CANDIDATE_SHA}:candle-integration/legacy-cuda.json" > "$legacy_manifest" ||
    fail "cannot read Legacy candidate manifest"
  python3 -B .github/scripts/candle-unified-pair-verify.py \
    --extended "$manifest" --legacy "$legacy_manifest" ||
    fail "Legacy and Extended candidates do not share exact standalone preparations"
  while IFS= read -r prepared; do
    git merge-base --is-ancestor "$prepared" "$LEGACY_CANDIDATE_SHA" ||
      fail "Legacy candidate does not contain shared prepared standalone $prepared"
  done < <(jq -r '.features[].prepared_sha' "$legacy_manifest")
fi

if [[ "$PROMOTION_KIND" == extended ]]; then
  jq -e '(.update_canonical_feature_refs | type) == "boolean"' "$manifest" >/dev/null ||
    fail "Extended candidate has no explicit standalone publication policy"
  update_feature_refs="$(jq -r '.update_canonical_feature_refs' "$manifest")"
else
  update_feature_refs=true
fi

manifest_target="$(jq -r '.promotion_target // .integration_target' "$manifest")"
[[ "$manifest_target" == "$TARGET_BRANCH" ]] ||
  fail "manifest promotion target=$manifest_target, expected $TARGET_BRANCH"

declare -a archive_specs=()
declare -a promote_specs=()
declare -a lease_args=()
declare -a verify_targets=()
declare -a verify_prepared=()
declare -a verify_exact=()

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

  ensure_archive "archive/promotion/$CAMPAIGN/before/$target" "$source"

  if [[ "$update_feature_refs" == false ]]; then
    [[ "$current" == "$source" ]] ||
      fail "standalone ref $target moved during qualification: current=$current expected=$source"
    echo "STANDALONE_PRESERVED=$target@$source" >> "$summary"
    verify_prepared+=("$source")
    verify_exact+=("true")
  else
    git merge-base --is-ancestor "$BASE_SHA" "$prepared" ||
      fail "prepared standalone $target does not include qualified main"
    git merge-base --is-ancestor "$prepared" "$CANDIDATE_SHA" ||
      fail "prepared standalone $target is not in the qualified integration"
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
    verify_prepared+=("$prepared")
    verify_exact+=("false")
  fi
  verify_targets+=("$target")
  idx=$((idx + 1))
done < <(jq -r '.features[] | [.feature,.target_ref,.source_sha,.prepared_sha] | @tsv' "$manifest")

old_integration="$(jq -r '.old_integration_sha // empty' "$manifest")"
current_integration="$(remote_head "$TARGET_BRANCH")"

if [[ -n "$current_integration" ]]; then
  git fetch --no-tags origin     "+refs/heads/$TARGET_BRANCH:refs/remotes/origin/promotion-integration"
  if [[ -n "$old_integration" ]]; then
    ensure_archive "archive/promotion/$CAMPAIGN/before/$TARGET_BRANCH" "$old_integration"
  fi
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

# Archive and promote both final integration refs in the SAME atomic push.
if [[ -n "${LEGACY_CANDIDATE_SHA:-}" ]]; then
  old_legacy="$(jq -r '.old_integration_sha // empty' "$legacy_manifest")"
  current_legacy="$(remote_head cuda_legacy)"
  if [[ -n "$current_legacy" && -n "$old_legacy" ]]; then
    ensure_archive "archive/promotion/$CAMPAIGN/before/cuda_legacy" "$old_legacy"
  fi
  ensure_archive "archive/legacy/sm61-validated-$RUN_ID-$RUN_ATTEMPT" "$LEGACY_CANDIDATE_SHA"
  if [[ "$current_legacy" == "$LEGACY_CANDIDATE_SHA" ]]; then
    echo "LEGACY_ALREADY_PROMOTED=cuda_legacy@$LEGACY_CANDIDATE_SHA" >> "$summary"
  elif [[ -n "$old_legacy" && "$current_legacy" == "$old_legacy" ]]; then
    promote_specs+=("$LEGACY_CANDIDATE_SHA:refs/heads/cuda_legacy")
    lease_args+=("--force-with-lease=refs/heads/cuda_legacy:$old_legacy")
    echo "LEGACY_PROMOTE=cuda_legacy:$old_legacy->$LEGACY_CANDIDATE_SHA" >> "$summary"
  elif [[ -z "$old_legacy" && -z "$current_legacy" ]]; then
    promote_specs+=("$LEGACY_CANDIDATE_SHA:refs/heads/cuda_legacy")
    lease_args+=("--force-with-lease=refs/heads/cuda_legacy:")
    echo "LEGACY_CREATE=cuda_legacy@$LEGACY_CANDIDATE_SHA" >> "$summary"
  else
    fail "Legacy integration ref moved unexpectedly: current=${current_legacy:-MISSING} expected=${old_legacy:-MISSING}"
  fi
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
  if [[ "${verify_exact[$i]}" == true ]]; then
    [[ "$current" == "$prepared" ]] ||
      fail "preserved standalone $target changed during promotion: current=$current expected=$prepared"
  elif [[ "$current" != "$prepared" ]]; then
    git fetch --no-tags origin       "+refs/heads/$target:refs/remotes/origin/promotion-verify-$i"
    current_tree="$(git rev-parse "$current^{tree}")"
    prepared_tree="$(git rev-parse "$prepared^{tree}")"
    [[ "$current_tree" == "$prepared_tree" ]] ||
      fail "feature ref $target does not match validated prepared tree after promotion"
  fi
done

[[ "$(remote_head "$TARGET_BRANCH")" == "$CANDIDATE_SHA" ]] ||
  fail "integration ref $TARGET_BRANCH is not the validated candidate after promotion"
if [[ -n "${LEGACY_CANDIDATE_SHA:-}" ]]; then
  [[ "$(remote_head cuda_legacy)" == "$LEGACY_CANDIDATE_SHA" ]] ||
    fail "Legacy integration ref does not match the Extended-qualified candidate"
fi

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
