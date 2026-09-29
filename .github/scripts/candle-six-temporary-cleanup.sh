#!/usr/bin/env bash
# One-shot, pinned cleanup after the six-standalone V3 atomic GPU promotion.
# Archives r1/r2 candidate commits BEFORE deleting precisely 23 known refs.
set -euo pipefail
umask 077
[[ "$GITHUB_REPOSITORY" == "haricot/candle" ]] || exit 3
[[ "$GITHUB_REF" == "refs/heads/mirror_orch" ]] || exit 3
cfg=".github/six-temporary-cleanup-launch.json"
report="$RUNNER_TEMP/six-temporary-cleanup"
mkdir -p "$report"
jq -e '
  .schema_version==1 and .operation=="archive-r1-r2-and-prune-temporary-six-source-refs" and
  .approved_promotion_run_id==36571847341 and
  (.deletions|type)=="object" and (.deletions|length)==23 and
  (.protected|type)=="object" and (.protected|length)==15 and
  (.new_archive_tags|length)==2 and
  (.existing_archive_tag_count)==12 and (.existing_promotion_tag_count)==7
' "$cfg" >/dev/null || { echo "::error::Unexpected cleanup manifest"; exit 3; }
gh auth setup-git
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git fetch --no-tags origin "+refs/heads/main:refs/remotes/origin/cleanup-main"
live_mirror="$(git ls-remote --heads origin refs/heads/mirror_orch | cut -f1)"
[[ "$live_mirror" == "$GITHUB_SHA" ]] || { echo "::error::mirror_orch moved"; exit 3; }
check_head() {
  local name="$1" expected="$2" live
  live="$(git ls-remote --heads origin "refs/heads/$name" | cut -f1)"
  [[ "$live" == "$expected" ]] || {
    echo "::error::Expected $name=$expected but remote has $live"; exit 3;
  }
}
# Pin and preserve ALL permanent refs, including v2 and v3.
while IFS=$'\t' read -r name sha; do
  [[ "$name" =~ ^(main|cuda_asd_runner_v2|cuda_asd_runner_v3|(bf16_candle|fp8_candle|fp4_candle|cudnn_fallback_candle|moe_simt_f16_candle|asd_core)(_standalone)?)$ ]] || exit 3
  [[ "$sha" =~ ^[a-f0-9]{40}$ ]] || exit 3
  check_head "$name" "$sha"
done < <(jq -r '.protected|to_entries[]|[.key,.value]|@tsv' "$cfg")
count_prep=0; count_candidate=0; count_pr=0
while IFS=$'\t' read -r name sha; do
  [[ "$sha" =~ ^[a-f0-9]{40}$ ]] || exit 3
  if [[ "$name" =~ ^prepare/six-sm61-v3-r[123]-20260929/(bf16_candle|fp8_candle|fp4_candle|cudnn_fallback_candle|moe_simt_f16_candle|asd_core)$ ]]; then
    ((count_prep+=1))
  elif [[ "$name" =~ ^integration/candidate/six-sm61-v3-r[123]-20260929$ ]]; then
    ((count_candidate+=1))
  elif [[ "$name" == ci/standalone-worktree-candidate-v1 || "$name" == ci/six-asd-drift-resume-v1 ]]; then
    ((count_pr+=1))
    # Both PR commits must already be reachable via mirror_orch merges.
    git merge-base --is-ancestor "$sha" "$GITHUB_SHA" || {
      echo "::error::Unmerged PR-head commit $name=$sha"; exit 3;
    }
  else
    echo "::error::Ref outside the 23 explicitly authorized deletions: $name"; exit 3
  fi
  check_head "$name" "$sha"
done < <(jq -r '.deletions|to_entries[]|[.key,.value]|@tsv' "$cfg")
[[ "$count_prep" == 18 && "$count_candidate" == 3 && "$count_pr" == 2 ]] || exit 3
# Confirm the successful physical SM61 resume and the successful atomic promotion.
gh api "repos/$GITHUB_REPOSITORY/actions/runs/36571847341" \
  --jq '.status=="completed" and .conclusion=="success"' | grep -Fxq true
gh api "repos/$GITHUB_REPOSITORY/actions/runs/36571847341/jobs?per_page=100" \
  --jq '([.jobs[]|select(.name=="resume_gpu" and .conclusion=="success")]|length)==1 and
        ([.jobs[]|select(.name=="promote" and .conclusion=="success")]|length)==1' | grep -Fxq true
for number in 14 15; do
  gh api "repos/$GITHUB_REPOSITORY/pulls/$number" \
    --jq '.state=="closed" and .merged_at!=null and .base.ref=="mirror_orch"' | grep -Fxq true
done
check_open_prs() {
  local open_prs
  open_prs="$(gh api --paginate "repos/$GITHUB_REPOSITORY/pulls?state=open&per_page=100" | jq -s 'add // []')"
  while IFS= read -r name; do
    if jq -e --arg b "$name" 'any(.[]; .head.ref==$b or .base.ref==$b)' <<<"$open_prs" >/dev/null; then
      echo "::error::Open PR references $name"; exit 3
    fi
  done < <(jq -r '.deletions|keys[]' "$cfg")
}
check_open_prs
# Verify all 12 old-source archive tags and seven V3 promotion tags.
v3="$(jq -r '.protected.cuda_asd_runner_v3' "$cfg")"
git cat-file -e "$v3^{commit}" || git fetch --no-tags origin "$v3"
git show "$v3:candle-integration/standalone-six.json" > "$report/standalone-six.json"
jq -e '.schema_version==2 and .campaign=="six-sm61-v3-r3-20260929"
 and (.features|length)==6' "$report/standalone-six.json" >/dev/null || exit 3
remote_tag_tip() {
  local tag="$1"
  git ls-remote --tags origin "refs/tags/$tag^{}" | awk -v ref="refs/tags/$tag^{}" '$2==ref {print $1}'
}
count_old=0; count_promoted=0
while IFS=$'\t' read -r feature src old prepared; do
  for entry in "$feature:$old" "${feature}_standalone:$src"; do
    name="${entry%%:*}"; commit="${entry#*:}"
    tag="archive/standalone-six/six-sm61-v3-r3-20260929/$name"
    [[ "$(remote_tag_tip "$tag")" == "$commit" ]] || {
      echo "::error::Missing or mismatched old-branch archive $tag"; exit 3;
    }
    ((count_old+=1))
  done
  tag="promoted/standalone-six/six-sm61-v3-r3-20260929/$feature"
  [[ "$(remote_tag_tip "$tag")" == "$prepared" ]] || {
    echo "::error::Missing or mismatched promoted tag $tag"; exit 3;
  }
  ((count_promoted+=1))
done < <(jq -r '.features[]|[.feature,.source_sha,.old_target_sha,.prepared_sha]|@tsv' "$report/standalone-six.json")
[[ "$(remote_tag_tip "promoted/standalone-six/six-sm61-v3-r3-20260929/cuda_asd_runner_v3")" == "$v3" ]] || exit 3
((count_promoted+=1))
[[ "$count_old" == 12 && "$count_promoted" == 7 ]] || exit 3
# Preserve exactly the r1/r2 integrated tips as annotated tags on GitHub
# first. Deletions cannot proceed unless BOTH archives are publicly verified.
archive_specs=()
while IFS=$'\t' read -r tag sha; do
  [[ "$tag" =~ ^archive/standalone-six/six-sm61-v3-r[12]-20260929/integration-candidate$ ]] || exit 3
  [[ "$sha" =~ ^[a-f0-9]{40}$ ]] || exit 3
  campaign="${tag#archive/standalone-six/}"
  campaign="${campaign%/integration-candidate}"
  [[ "$(git ls-remote --heads origin "refs/heads/integration/candidate/$campaign" | cut -f1)" == "$sha" ]] || exit 3
  [[ -z "$(git ls-remote --tags origin "refs/tags/$tag")" ]] || {
    echo "::error::Archive tag already exists: $tag"; exit 3;
  }
  git cat-file -e "$sha^{commit}" || git fetch --no-tags origin "$sha"
  git tag -a "$tag" "$sha" -m "Preserve exact failed $campaign integration candidate before deleting temporary refs"
  archive_specs+=("refs/tags/$tag:refs/tags/$tag")
done < <(jq -r '.new_archive_tags|to_entries[]|[.key,.value]|@tsv' "$cfg")
[[ "${#archive_specs[@]}" -eq 2 ]] || exit 3
git push --atomic origin "${archive_specs[@]}" > "$report/archive-push.log" 2>&1 || {
  echo "::error::Archiving r1/r2 candidate tips failed. No branch deleted."
  tail -n 50 "$report/archive-push.log"; exit 3;
}
while IFS=$'\t' read -r tag sha; do
  [[ "$(remote_tag_tip "$tag")" == "$sha" ]] || {
    echo "::error::Cannot verify archived $tag. No branch deleted."; exit 3;
  }
done < <(jq -r '.new_archive_tags|to_entries[]|[.key,.value]|@tsv' "$cfg")
echo "Two r1/r2 candidate commits archived as verified annotated tags" >> "$GITHUB_STEP_SUMMARY"
# Compare-and-delete all 23 authorized refs in ONE atomic transaction.
# force-with-lease prevents concurrent edits from being silently discarded.
[[ "$(git ls-remote --heads origin refs/heads/mirror_orch | cut -f1)" == "$GITHUB_SHA" ]] || exit 3
while IFS=$'\t' read -r name sha; do
  check_head "$name" "$sha"
done < <(jq -r '.protected|to_entries[]|[.key,.value]|@tsv' "$cfg")
check_open_prs
leases=(); deletes=()
while IFS=$'\t' read -r name sha; do
  check_head "$name" "$sha"
  leases+=("--force-with-lease=refs/heads/$name:$sha")
  deletes+=(":refs/heads/$name")
done < <(jq -r '.deletions|to_entries[]|[.key,.value]|@tsv' "$cfg")
[[ "${#deletes[@]}" == 23 ]] || exit 3
git push --atomic "${leases[@]}" origin "${deletes[@]}" \
  > "$report/atomic-delete.log" 2>&1 || {
  echo "::error::Atomic deletion rejected; archived tags kept, no partial cleanup."
  tail -n 80 "$report/atomic-delete.log"; exit 3;
}
while IFS= read -r name; do
  [[ -z "$(git ls-remote --heads origin "refs/heads/$name")" ]] || {
    echo "::error::Unexpected remaining temporary ref: $name"; exit 3;
  }
done < <(jq -r '.deletions|keys[]' "$cfg")
while IFS=$'\t' read -r name sha; do
  check_head "$name" "$sha"
done < <(jq -r '.protected|to_entries[]|[.key,.value]|@tsv' "$cfg")
[[ "$(git ls-remote --heads origin refs/heads/mirror_orch | cut -f1)" == "$GITHUB_SHA" ]] || exit 3
while IFS=$'\t' read -r tag sha; do
  [[ "$(remote_tag_tip "$tag")" == "$sha" ]] || exit 3
done < <(jq -r '.new_archive_tags|to_entries[]|[.key,.value]|@tsv' "$cfg")
jq -n --slurpfile c "$cfg" --arg sha "$GITHUB_SHA" --arg run "$GITHUB_RUN_ID" \
 '{status:"CLEANUP_VERIFIED",cleanup_workflow_sha:$sha,run_id:$run,
  deletion_count:($c[0].deletions|length),
  deleted_refs:$c[0].deletions,
  preserved_permanent_refs:$c[0].protected,
  archived_r1_r2_tags:$c[0].new_archive_tags,
  preserved_preexisting_archive_tags:12,preserved_preexisting_promotion_tags:7}' \
 > "$report/report.json"
echo "CLEANUP_VERIFIED: two archival tags created, 23 temp refs removed atomically, 15 permanent refs unchanged" \
  >> "$GITHUB_STEP_SUMMARY"
