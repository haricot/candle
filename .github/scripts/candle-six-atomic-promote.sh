#!/usr/bin/env bash
# Promote only after six CPU, combined CPU/CUDA and one physical GPU PASS.
# Atomic CAS updates six canonical feature branches, integrated v3 and the stable candle_asd_cuda alias.
set -euo pipefail
umask 077
: "${CAMPAIGN:?}" "${SHA:?}" "${LOCK:?}" "${MANIFEST_SHA:?}" "${GITHUB_REPOSITORY:?}"
report="$RUNNER_TEMP/candle-six-promotion"
mkdir -p "$report"
manifest=candle-integration/standalone-six.json
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || exit 3
[[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$MANIFEST_SHA" ]] || exit 3
jq -e --arg c "$CAMPAIGN" '
  .schema_version==3 and .kind=="canonical-feature-six" and
  .campaign==$c and .integration_target=="cuda_asd_runner_v3" and
  ([.features[].feature]==["bf16_candle","fp8_candle","fp4_candle",
    "cudnn_fallback_candle","moe_simt_f16_candle","asd_core"]) and
  all(.features[]; .source_ref==.feature and .target_ref==.feature and
    .source_sha==.old_target_sha and
    (.source_sha|test("^[a-f0-9]{40}$")) and
    (.prepared_sha|test("^[a-f0-9]{40}$")))
' "$manifest" >/dev/null || exit 3
main="$(jq -r .main_sha "$manifest")"
oldintegration="$(jq -r .old_integration_sha "$manifest")"
integration_alias="candle_asd_cuda"
oldalias="$(git ls-remote --heads origin "refs/heads/${integration_alias}" | cut -f1)"
[[ "$oldalias" == "$oldintegration" ]] || {
  echo "::error::${integration_alias} must match cuda_asd_runner_v3 before promotion"; exit 3
}
[[ "$(git ls-remote --heads origin refs/heads/main | cut -f1)" == "$main" ]] || {
  echo "::error::main advanced; discard this campaign rather than force-promote"; exit 3
}
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
gh auth setup-git
# Verify all independent CPU artifacts from this SAME Actions run.
for feature in bf16_candle fp8_candle fp4_candle cudnn_fallback_candle moe_simt_f16_candle asd_core; do
  entry="$(jq -c --arg f "$feature" '.features[]|select(.feature==$f)' "$manifest")"
  source="$(jq -r .source_sha <<<"$entry")"
  candidate="$(jq -r .prepared_sha <<<"$entry")"
  ref="$(jq -r .preparation_ref <<<"$entry")"
  old="$(jq -r .old_target_sha <<<"$entry")"
  jq -e --arg f "$feature" --arg s "$source" --arg sha "$candidate" '
    .feature==$f and .source_sha==$s and .prepared_sha==$sha and
    .status=="CPU_PASSED"
  ' "$RUNNER_TEMP/six-cpu/standalone-$CAMPAIGN-$feature-cpu/report.json" >/dev/null || {
    echo "::error::Missing matching standalone CPU PASS for $feature"; exit 3
  }
  git merge-base --is-ancestor "$candidate" "$SHA" || exit 3
  [[ "$(git ls-remote --heads origin "refs/heads/$ref" | cut -f1)" == "$candidate" ]] || exit 3
  [[ "$source" == "$old" ]] || exit 3
  [[ "$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)" == "$source" ]] || exit 3
done
jq -e --arg sha "$SHA" --arg manifest "$MANIFEST_SHA" --arg lock "$LOCK" '
  .status=="CPU_PASSED" and .candidate_sha==$sha and
  .manifest_sha256==$manifest and .lock_sha256==$lock
' "$RUNNER_TEMP/six-integrated-cpu/report.json" >/dev/null || exit 3
jq -e --arg sha "$SHA" --arg lock "$LOCK" '
  .status=="CUDA_COMPILE_PASSED" and .candidate_sha==$sha and
  .lock_sha256==$lock and .target=="sm_61"
' "$RUNNER_TEMP/six-cuda/report.json" >/dev/null || exit 3
jq -e --arg sha "$SHA" --arg manifest "$MANIFEST_SHA" --arg lock "$LOCK" '
  .status=="GPU_PASSED" and .gpu_executed==true and
  .candidate_sha==$sha and .manifest_sha256==$manifest and
  .lock_sha256==$lock and .gpu_compute_cap=="6.1" and
  .suite_coverage=="all_six" and
  .asd_policy_promoted_decisions==12 and
  .asd_policy_gpu_uuid_matches==true
' "$RUNNER_TEMP/six-gpu/report.json" >/dev/null || exit 3
# PR #8 is a BF16 test vehicle. Moving its HEAD during atomic promotion
# would silently rewrite the open PR's test target. Do not do so: the owner
# may close the test-only PR (never merge into main) when tests are complete.
open_prs="$(gh api --paginate "repos/$GITHUB_REPOSITORY/pulls?state=open&per_page=100" \
  | jq -s 'add')"
for feature in bf16_candle fp8_candle fp4_candle cudnn_fallback_candle moe_simt_f16_candle asd_core; do
  jq -e --arg f "$feature" '[.[]|select(.head.ref==$f)]|length==0' \
    <<<"$open_prs" >/dev/null || {
      echo "::error::Open PR on $feature; close the test-only PR #8 when ready. Never rewrite an open PR head."; exit 3;
    }
done
[[ "$(git ls-remote --heads origin refs/heads/cuda_asd_runner_v3 | cut -f1)" == "$oldintegration" ]] || exit 3
# Archive the PREVIOUS canonical feature refs under uniquely campaign-scoped
# annotated tags; if this push fails, no branch is rewritten.
archive_refs=()
while IFS=$'\t' read -r feature src old prepared; do
  [[ "$src" == "$old" ]] || exit 3
  tag="archive/six-feature/$CAMPAIGN/$feature"
  if git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null 2>&1; then
    echo "::error::Backup tag already exists: $tag"; exit 3
  fi
  git tag -a "$tag" "$src" -m "Pre-promotion canonical $feature=$src before verified $CAMPAIGN"
  archive_refs+=("refs/tags/$tag:refs/tags/$tag")
done < <(jq -r '.features[]|[.feature,.source_sha,.old_target_sha,.prepared_sha]|@tsv' "$manifest")
if [[ -n "$oldintegration" ]]; then
  tag="archive/six-feature/$CAMPAIGN/cuda_asd_runner_v3"
  git tag -a "$tag" "$oldintegration" -m "Previous integrated v3 before $CAMPAIGN"
  archive_refs+=("refs/tags/$tag:refs/tags/$tag")
fi
git push --atomic origin "${archive_refs[@]}" > "$report/backup-tags.log" 2>&1 || exit 3
# Check all leases again *after* pushing backups. If any ref moved,
# fail closed. The following single atomic push changes all eight permanent
# refs (six canonical features + v3 + stable alias) + promotion tags, or none.
args=()
leases=()
promotion_tags=()
while IFS=$'\t' read -r feature src old prepared; do
  [[ "$src" == "$old" ]] || exit 3
  [[ "$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)" == "$src" ]] || exit 3
  leases+=("--force-with-lease=refs/heads/$feature:$src")
  args+=("$prepared:refs/heads/$feature")
  tag="promoted/six-feature/$CAMPAIGN/$feature"
  git tag -a "$tag" "$prepared" -m "Canonical $feature, six-way GPU PASS on $SHA"
  promotion_tags+=("refs/tags/$tag:refs/tags/$tag")
done < <(jq -r '.features[]|[.feature,.source_sha,.old_target_sha,.prepared_sha]|@tsv' "$manifest")
[[ "$(git ls-remote --heads origin refs/heads/cuda_asd_runner_v3 | cut -f1)" == "$oldintegration" ]] || exit 3
leases+=("--force-with-lease=refs/heads/cuda_asd_runner_v3:$oldintegration")
args+=("$SHA:refs/heads/cuda_asd_runner_v3")
[[ "$(git ls-remote --heads origin "refs/heads/${integration_alias}" | cut -f1)" == "$oldalias" ]] || exit 3
leases+=("--force-with-lease=refs/heads/${integration_alias}:$oldalias")
args+=("$SHA:refs/heads/${integration_alias}")
tag="promoted/six-feature/$CAMPAIGN/cuda_asd_runner_v3"
git tag -a "$tag" "$SHA" -m "Integrated all-six GPU PASS; manifest $MANIFEST_SHA"
promotion_tags+=("refs/tags/$tag:refs/tags/$tag")
git push --atomic "${leases[@]}" origin "${args[@]}" "${promotion_tags[@]}" \
  > "$report/atomic-promotion.log" 2>&1 || {
    echo "::error::Atomic ref transaction rejected; original branches left unchanged"; exit 3
  }
for feature in bf16_candle fp8_candle fp4_candle cudnn_fallback_candle moe_simt_f16_candle asd_core; do
  sha="$(jq -r --arg f "$feature" '.features[]|select(.feature==$f)|.prepared_sha' "$manifest")"
  [[ "$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)" == "$sha" ]] || exit 3
done
[[ "$(git ls-remote --heads origin refs/heads/cuda_asd_runner_v3 | cut -f1)" == "$SHA" ]] || exit 3
[[ "$(git ls-remote --heads origin "refs/heads/${integration_alias}" | cut -f1)" == "$SHA" ]] || exit 3
jq -n --arg c "$CAMPAIGN" --arg sha "$SHA" --arg manifest "$MANIFEST_SHA" \
  --arg gpu_run "$GITHUB_RUN_ID" --arg alias "$integration_alias" --slurpfile sources "$manifest" \
  '{status:"ATOMIC_PROMOTED",campaign:$c,integration_sha:$sha,integration_alias:$alias,
    manifest_sha256:$manifest,gpu_run_id:$gpu_run,
    sources:$sources[0].features}' > "$report/report.json"
echo "ATOMIC PROMOTION complete: six canonical feature refs + integrated v3 + ${integration_alias}" \
  >> "$GITHUB_STEP_SUMMARY"

# Cleanup is deliberately post-promotion. Temporary refs are deleted only
# after report.json already records ATOMIC_PROMOTED and only if every live
# ref still points at the immutable SHA recorded by this campaign.
cleanup_refs=("integration/candidate/$CAMPAIGN")
cleanup_expected=("$SHA")
while IFS=\t' read -r ref prepared; do
  [[ "$ref" == "prepare/$CAMPAIGN/"* ]] || {
    echo "::warning::Unexpected preparation ref namespace; cleanup skipped: $ref"
    cleanup_refs=()
    break
  }
  cleanup_refs+=("$ref")
  cleanup_expected+=("$prepared")
done < <(jq -r '.features[]|[.preparation_ref,.prepared_sha]|@tsv' "$manifest")

cleanup_status="NONE"
delete_refs=()
if (("${#cleanup_refs[@]}" > 0)); then
  cleanup_status="DELETED"
  for i in "${!cleanup_refs[@]}"; do
    ref="${cleanup_refs[$i]}"
    expected="${cleanup_expected[$i]}"
    live="$(git ls-remote --heads origin "refs/heads/$ref" | cut -f1)"
    if [[ -z "$live" ]]; then
      continue
    fi
    if [[ "$live" != "$expected" ]]; then
      echo "::warning::Temporary ref moved; refusing cleanup: $ref=$live expected=$expected"
      cleanup_status="PENDING"
      delete_refs=()
      break
    fi
    delete_refs+=("refs/heads/$ref")
  done
  if (("${#delete_refs[@]}" > 0)); then
    if ! git push --atomic origin --delete "${delete_refs[@]}" > "$report/cleanup-refs.log" 2>&1; then
      echo "::warning::Promotion succeeded but temporary-ref cleanup failed; refs retained"
      cleanup_status="PENDING"
    fi
  fi
else
  cleanup_status="PENDING"
fi

tmp_report="$(mktemp "$report/report.json.XXXXXX")"
jq --arg cleanup_status "$cleanup_status" \
  '. + {temporary_ref_cleanup:$cleanup_status}' "$report/report.json" > "$tmp_report"
mv "$tmp_report" "$report/report.json"
echo "Temporary campaign refs cleanup: $cleanup_status" >> "$GITHUB_STEP_SUMMARY"
