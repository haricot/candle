#!/usr/bin/env bash
# Promote only after six CPU, combined CPU/CUDA and one physical GPU PASS.
# Atomic CAS updates public aliases, canonical standalone sources and v3.
set -euo pipefail
umask 077
: "${CAMPAIGN:?}" "${SHA:?}" "${LOCK:?}" "${MANIFEST_SHA:?}" "${GITHUB_REPOSITORY:?}"
report="$RUNNER_TEMP/candle-six-promotion"
mkdir -p "$report"
manifest=candle-integration/standalone-six.json
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || exit 3
[[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$MANIFEST_SHA" ]] || exit 3
jq -e --arg c "$CAMPAIGN" '
  .schema_version==2 and .kind=="standalone-six" and
  .campaign==$c and .integration_target=="cuda_asd_runner_v3" and
  ([.features[].feature]==["bf16_candle","fp8_candle","fp4_candle",
    "cudnn_fallback_candle","moe_simt_f16_candle","asd_core"]) and
  all(.features[]; .source_ref==(.feature+"_standalone") and
    .target_ref==.feature and
    (.source_sha|test("^[a-f0-9]{40}$")) and
    (.old_target_sha|test("^[a-f0-9]{40}$")) and
    (.prepared_sha|test("^[a-f0-9]{40}$")))
' "$manifest" >/dev/null || exit 3
main="$(jq -r .main_sha "$manifest")"
oldintegration="$(jq -r .old_integration_sha "$manifest")"
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
  [[ "$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)" == "$old" ]] || exit 3
  [[ "$(git ls-remote --heads origin "refs/heads/${feature}_standalone" | cut -f1)" == "$source" ]] || exit 3
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
# Archive the PREVIOUS public and standalone refs under uniquely campaign-scoped
# annotated tags; if this push fails, no branch is rewritten.
archive_refs=()
while IFS=$'\t' read -r feature src old prepared; do
  for spec in "$feature:$old" "${feature}_standalone:$src"; do
    name="${spec%%:*}"
    rev="${spec#*:}"
    tag="archive/standalone-six/$CAMPAIGN/$name"
    if git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null 2>&1; then
      echo "::error::Backup tag already exists: $tag"; exit 3
    fi
    git tag -a "$tag" "$rev" -m "Pre-promotion $name=$rev before verified $CAMPAIGN"
    archive_refs+=("refs/tags/$tag:refs/tags/$tag")
  done
done < <(jq -r '.features[]|[.feature,.source_sha,.old_target_sha,.prepared_sha]|@tsv' "$manifest")
if [[ -n "$oldintegration" ]]; then
  tag="archive/standalone-six/$CAMPAIGN/cuda_asd_runner_v3"
  git tag -a "$tag" "$oldintegration" -m "Previous integrated v3 before $CAMPAIGN"
  archive_refs+=("refs/tags/$tag:refs/tags/$tag")
fi
git push --atomic origin "${archive_refs[@]}" > "$report/backup-tags.log" 2>&1 || exit 3
# Check all leases again *after* pushing backups. If any ref moved,
# fail closed. The following single atomic push changes either all 13
# permanent refs + promotion tags, or none.
args=()
leases=()
promotion_tags=()
while IFS=$'\t' read -r feature src old prepared; do
  [[ "$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)" == "$old" ]] || exit 3
  [[ "$(git ls-remote --heads origin "refs/heads/${feature}_standalone" | cut -f1)" == "$src" ]] || exit 3
  leases+=("--force-with-lease=refs/heads/$feature:$old")
  leases+=("--force-with-lease=refs/heads/${feature}_standalone:$src")
  args+=("$prepared:refs/heads/$feature")
  args+=("$prepared:refs/heads/${feature}_standalone")
  tag="promoted/standalone-six/$CAMPAIGN/$feature"
  git tag -a "$tag" "$prepared" -m "Standalone $feature, six-way GPU PASS on $SHA"
  promotion_tags+=("refs/tags/$tag:refs/tags/$tag")
done < <(jq -r '.features[]|[.feature,.source_sha,.old_target_sha,.prepared_sha]|@tsv' "$manifest")
[[ "$(git ls-remote --heads origin refs/heads/cuda_asd_runner_v3 | cut -f1)" == "$oldintegration" ]] || exit 3
leases+=("--force-with-lease=refs/heads/cuda_asd_runner_v3:$oldintegration")
args+=("$SHA:refs/heads/cuda_asd_runner_v3")
tag="promoted/standalone-six/$CAMPAIGN/cuda_asd_runner_v3"
git tag -a "$tag" "$SHA" -m "Integrated all-six GPU PASS; manifest $MANIFEST_SHA"
promotion_tags+=("refs/tags/$tag:refs/tags/$tag")
git push --atomic "${leases[@]}" origin "${args[@]}" "${promotion_tags[@]}" \
  > "$report/atomic-promotion.log" 2>&1 || {
    echo "::error::Atomic ref transaction rejected; original branches left unchanged"; exit 3
  }
for feature in bf16_candle fp8_candle fp4_candle cudnn_fallback_candle moe_simt_f16_candle asd_core; do
  sha="$(jq -r --arg f "$feature" '.features[]|select(.feature==$f)|.prepared_sha' "$manifest")"
  [[ "$(git ls-remote --heads origin "refs/heads/$feature" | cut -f1)" == "$sha" ]] || exit 3
  [[ "$(git ls-remote --heads origin "refs/heads/${feature}_standalone" | cut -f1)" == "$sha" ]] || exit 3
done
[[ "$(git ls-remote --heads origin refs/heads/cuda_asd_runner_v3 | cut -f1)" == "$SHA" ]] || exit 3
jq -n --arg c "$CAMPAIGN" --arg sha "$SHA" --arg manifest "$MANIFEST_SHA" \
  --arg gpu_run "$GITHUB_RUN_ID" --slurpfile sources "$manifest" \
  '{status:"ATOMIC_PROMOTED",campaign:$c,integration_sha:$sha,
    manifest_sha256:$manifest,gpu_run_id:$gpu_run,
    sources:$sources[0].features}' > "$report/report.json"
echo "ATOMIC PROMOTION complete: six pure standalone refs + six aliases + integrated v3" \
  >> "$GITHUB_STEP_SUMMARY"
