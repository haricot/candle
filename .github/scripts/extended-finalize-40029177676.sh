#!/usr/bin/env bash
# One reviewed successful twelve-source campaign, with archival tags before writes.
set -Eeuo pipefail
umask 077
[[ "$GITHUB_REPOSITORY" == haricot/candle ]]
[[ "$GITHUB_REF" == refs/heads/ops/extended-finalize-40029177676 ]]
cfg=.github/extended-finalize-40029177676.json
report="$RUNNER_TEMP/extended-finalization"
mkdir -p "$report"
trap 'echo "FINALIZATION=FAILED line=$LINENO" | tee -a "$report/summary.txt" >&2' ERR
jq -e '
  .schema==1 and .operation=="promote-qualified-extended-and-retire-temporary-refs" and
  .repository=="haricot/candle" and .gpu_run_id==40029177676 and .gpu_attempt==2 and
  .gpu_job_id==120837415140 and
  .candidate_sha=="40b5a8acc43b5f8ef925370ae12daddd929f39f2" and
  .orchestrator_sha=="9a0c9348249bfd76cd8d1d92addb7cd724fe6385" and
  .campaign=="voice-full-40029177676" and
  (.feature_targets|length)==12 and (.deletions|length)==95 and
  .retire_workflow_paths==[".github/workflows/voice-models-extended.yml"]
' "$cfg" >/dev/null
cp "$cfg" "$report/request.json"
repo="$GITHUB_REPOSITORY"
run=40029177676
candidate=40b5a8acc43b5f8ef925370ae12daddd929f39f2
campaign=voice-full-40029177676
gh api "repos/$repo/actions/runs/$run" > "$report/source-run.json"
jq -e '.status=="completed" and .conclusion=="success" and .run_attempt==2 and
  .head_sha=="9a0c9348249bfd76cd8d1d92addb7cd724fe6385" and
  .head_branch=="orch/extended-full-voice-run-20261023"' "$report/source-run.json" >/dev/null
gh api "repos/$repo/actions/runs/$run/jobs?per_page=100" > "$report/source-jobs.json"
jq -e '
  any(.jobs[]; .id==120837415140 and .name=="SM61 Extended runtime" and .conclusion=="success") and
  all(.jobs[]; .conclusion=="success") and
  ([.jobs[]|select(.name|startswith("standalone_cpu ("))]|length)==12
' "$report/source-jobs.json" >/dev/null
for name in prepare integrated_cpu integrated_cuda_compile; do
  jq -e --arg name "$name" 'any(.jobs[]; .name==$name and .conclusion=="success")' \
    "$report/source-jobs.json" >/dev/null
done
while IFS=$'\t' read -r kind id; do
  gh api "repos/$repo/actions/artifacts/$id" > "$report/$kind-artifact.json"
  jq -e --argjson run "$run" '.expired==false and .workflow_run.id==$run and
    .workflow_run.head_sha=="9a0c9348249bfd76cd8d1d92addb7cd724fe6385"' \
    "$report/$kind-artifact.json" >/dev/null
  # Exact IDs and digests matter: attempt 1 has a failed artifact with the same name.
  gh api "repos/$repo/actions/artifacts/$id/zip" > "$report/$kind.zip"
done < <(jq -r '.artifacts[]|[.kind,.id]|@tsv' "$cfg")
gh auth setup-git
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git fetch --no-tags origin
git show "$candidate:candle-integration/extended.json" > "$report/manifest.json"
python3 .github/scripts/verify-extended-proof-40029177676.py "$report" "$cfg" \
  | tee "$report/evidence-summary.txt"

head_sha() { git ls-remote --heads origin "refs/heads/$1" | awk 'NR==1 {print $1}'; }
remote_check() {
  local name=$1 expected=$2 current
  current="$(head_sha "$name")"
  [[ "$current" == "$expected" ]] || {
    echo "::error::Ref moved: $name expected=$expected actual=$current"; return 3;
  }
}
git show "$(jq -r .canonical_orchestrator.new_sha "$cfg"):.github/scripts/extended-dynamic-selftest.sh" \
  > "$report/canonical-selftest.sh"
mkdir -p "$report/canonical-check"
git archive "$(jq -r .canonical_orchestrator.new_sha "$cfg")" .github \
  | tar -x -C "$report/canonical-check"
(cd "$report/canonical-check"; bash .github/scripts/extended-dynamic-selftest.sh) \
  | tee "$report/canonical-contracts.txt"
git merge-base --is-ancestor "$(jq -r .canonical_orchestrator.old_sha "$cfg")" \
  "$(jq -r .canonical_orchestrator.new_sha "$cfg")"

# Explicit snapshot only. Keep every canonical model/application/D7/PR branch.
while IFS=$'\t' read -r name sha; do
  [[ "$sha" =~ ^[a-f0-9]{40}$ ]]
  [[ "$name" =~ ^prepare/extended/(extended-(39738701484|39781921938|39815786458|39846015167)-1|voice-full-(40018990330|40019534399|40029177676))/[a-z0-9_]+$ ||
     "$name" =~ ^integration/extended/candidate/(extended-(39738701484|39781921938|39815786458|39846015167)-1|voice-full-(40018990330|40019534399|40029177676))$ ||
     "$name" =~ ^archive/(promotion/extended-39846015167-1/before/[a-z0-9_]+|extended/sm61-validated-39846015167-1)$ ||
     "$name" =~ ^orch/extended-full-voice-(no-half|run)-20261023$ ||
     "$name" == nemotron_asr_extended_candidate || "$name" == voice_models_extended_candidate ||
     "$name" == ops/extended-canonical-40029177676 ]]
  current="$(head_sha "$name")"
  [[ -z "$current" || "$current" == "$sha" ]]
done < <(jq -r '.deletions|to_entries[]|[.key,.value]|@tsv' "$cfg")
gh api --paginate "repos/$repo/pulls?state=open&per_page=100" | jq -s 'add // []' > "$report/open-prs.json"
jq -e --slurpfile cfg "$cfg" '
  all(.[]; . as $pr | ($cfg[0].deletions|has($pr.head.ref)|not) and
                       ($cfg[0].deletions|has($pr.base.ref)|not))
' "$report/open-prs.json" >/dev/null
gh api "repos/$repo/actions/runs?per_page=100" > "$report/recent-runs.json"
jq -e --slurpfile cfg "$cfg" '
  all(.workflow_runs[]; .status=="completed" or
      (.head_branch as $b | $cfg[0].deletions|has($b)|not))
' "$report/recent-runs.json" >/dev/null

declare -a promote=() leases=() archive=() remove=() remove_leases=()
add_promotion() {
  local name=$1 old=$2 new=$3 current
  current="$(head_sha "$name")"
  [[ "$current" == "$old" || "$current" == "$new" ]]
  if [[ "$current" != "$new" ]]; then
    promote+=("$new:refs/heads/$name")
    leases+=("--force-with-lease=refs/heads/$name:$old")
  fi
}
while IFS=$'\t' read -r name old new; do add_promotion "$name" "$old" "$new"; done \
  < <(jq -r '.feature_targets[]|[.branch,.old_sha,.new_sha]|@tsv' "$cfg")
add_promotion extended "$(jq -r .old_extended_sha "$cfg")" "$candidate"
add_promotion orch/extended-v1 "$(jq -r .canonical_orchestrator.old_sha "$cfg")" \
  "$(jq -r .canonical_orchestrator.new_sha "$cfg")"

archive_tag() {
  local tag=$1 sha=$2 current
  current="$(git ls-remote --tags origin "refs/tags/$tag" | awk 'NR==1 {print $1}')"
  if [[ -n "$current" ]]; then
    git fetch --no-tags origin "refs/tags/$tag:refs/tags/$tag"
    [[ "$(git rev-parse "refs/tags/$tag^{commit}")" == "$sha" ]]
  else
    archive+=("$sha:refs/tags/$tag")
  fi
  jq -nc --arg tag "$tag" --arg sha "$sha" '{tag:$tag,sha:$sha}' >> "$report/archives.ndjson"
}
while IFS=$'\t' read -r name old new; do
  archive_tag "archive/promotion/$campaign/before/$name" "$old"
  archive_tag "promoted/extended/$campaign/$name" "$new"
done < <(jq -r '.feature_targets[]|[.branch,.old_sha,.new_sha]|@tsv' "$cfg")
archive_tag "archive/promotion/$campaign/before/extended" "$(jq -r .old_extended_sha "$cfg")"
archive_tag "archive/promotion/$campaign/before/orch/extended-v1" "$(jq -r .canonical_orchestrator.old_sha "$cfg")"
archive_tag archive/extended/sm61-validated-40029177676-2 "$candidate"
while IFS=$'\t' read -r name sha; do
  if [[ "$name" == archive/* ]]; then tag=$name; else tag="archive/cleanup/$campaign/$name"; fi
  archive_tag "$tag" "$sha"
done < <(jq -r '.deletions|to_entries[]|[.key,.value]|@tsv' "$cfg")
archive_tag "archive/cleanup/$campaign/ops/extended-finalize-40029177676" "$GITHUB_SHA"
if (( ${#archive[@]} )); then git push --atomic origin "${archive[@]}"; fi
if (( ${#promote[@]} )); then git push --atomic "${leases[@]}" origin "${promote[@]}"; fi
while IFS=$'\t' read -r name old new; do remote_check "$name" "$new"; done \
  < <(jq -r '.feature_targets[]|[.branch,.old_sha,.new_sha]|@tsv' "$cfg")
remote_check extended "$candidate"
remote_check orch/extended-v1 "$(jq -r .canonical_orchestrator.new_sha "$cfg")"
echo 'PROMOTION=PASS twelve isolated source refs plus Extended and canonical orchestration' \
  | tee -a "$report/summary.txt"

# Promotion is now concrete and verified. Ref-specific leases prevent stale deletions.
while IFS=$'\t' read -r name sha; do
  current="$(head_sha "$name")"
  if [[ -n "$current" ]]; then
    [[ "$current" == "$sha" ]]
    remove+=(":refs/heads/$name")
    remove_leases+=("--force-with-lease=refs/heads/$name:$sha")
  fi
done < <(jq -r '.deletions|to_entries[]|[.key,.value]|@tsv' "$cfg")
if (( ${#remove[@]} )); then git push --atomic "${remove_leases[@]}" origin "${remove[@]}"; fi
while read -r name; do [[ -z "$(head_sha "$name")" ]]; done < <(jq -r '.deletions|keys[]' "$cfg")
echo "CLEANUP=PASS removed=${#remove[@]} temporary heads retained as tags" | tee -a "$report/summary.txt"

# Retire only the superseded voice-only aggregate, never the common Extended path.
git fetch --no-tags --prune origin '+refs/heads/*:refs/remotes/origin/*'
orphan=true
while read -r ref; do
  if git cat-file -e "$ref:.github/workflows/voice-models-extended.yml" 2>/dev/null; then
    orphan=false
    echo "WORKFLOW_RETAINED live_definition=$ref" | tee -a "$report/summary.txt"
  fi
done < <(git for-each-ref --format='%(refname)' refs/remotes/origin/)
if [[ "$orphan" == true ]]; then
  gh api "repos/$repo/actions/workflows?per_page=100" > "$report/workflows.json"
  while read -r id; do
    gh api --method PUT "repos/$repo/actions/workflows/$id/disable"
    echo "WORKFLOW_DISABLED id=$id path=.github/workflows/voice-models-extended.yml" | tee -a "$report/summary.txt"
  done < <(jq -r '.workflows[]|select(.path==".github/workflows/voice-models-extended.yml")|.id' "$report/workflows.json")
fi
cat "$report/summary.txt" >> "$GITHUB_STEP_SUMMARY"
