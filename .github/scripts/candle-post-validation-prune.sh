#!/usr/bin/env bash
# Audited 2026-10-16 campaign pruning: preview by default, archive first, atomic delete.
set -euo pipefail
umask 077
[[ "$GITHUB_REPOSITORY" == "haricot/candle" && "$GITHUB_REF" == "refs/heads/mirror_orch" ]] || exit 3
[[ "$(git rev-parse HEAD)" == "$GITHUB_SHA" ]] || exit 3
m=.github/post-validation-prune-20261016.json
out="${RUNNER_TEMP:-/tmp}/candle-prune"
mkdir -p "$out"
jq -e '
 .schema_version==1 and .repository=="haricot/candle" and
 .validation.legacy_run_id==38898658386 and .validation.extended_run_id==38898680839 and
 (.targets|length)==99 and (.protected_heads|length)==13 and
 ([.targets[].ref]|unique|length)==99 and
 all(.targets[]; (.sha|test("^[0-9a-f]{40}$")) and
 (.ref|test("^(archive/|prepare/(legacy|extended)/|integration/(legacy-cuda|extended)/candidate/)"))) and
 all(.protected_heads[]; test("^[0-9a-f]{40}$"))
' "$m" >/dev/null
dry="${PRUNE_DRY_RUN:-true}"
[[ "$dry" == true || "$dry" == false ]] || exit 3
if [[ "$dry" == false && "${PRUNE_CONFIRM:-}" != ARCHIVE_AND_DELETE_99 ]]; then
 echo "::error::Apply requires ARCHIVE_AND_DELETE_99"; exit 3
fi
for kind in legacy extended; do
 run="$(jq -r ".validation.${kind}_run_id" "$m")"
 job="$(jq -r ".validation.${kind}_job_id" "$m")"
 gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run" |
   jq -e '.status=="completed" and .conclusion=="success"' >/dev/null
 gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run/jobs?per_page=100" |
   jq -e --argjson job "$job" --arg kind "$kind" '
    any(.jobs[]; .id==$job and .conclusion=="success") and
    any(.jobs[]; .name==("Promote validated "+(if $kind=="legacy" then "Legacy" else "Extended" end)+" refs") and .conclusion=="success")
   ' >/dev/null
done
git ls-remote --heads origin > "$out/heads"
head_sha() { awk -v n="refs/heads/$1" '$2==n {print $1}' "$out/heads"; }
check_protected() {
 while IFS=$'\t' read -r name sha; do
  [[ "$(head_sha "$name")" == "$sha" ]] ||
    { echo "::error::Protected head moved: $name"; exit 3; }
 done < <(jq -r '.protected_heads|to_entries[]|[.key,.value]|@tsv' "$m")
}
check_candidates() {
 gh api --paginate "repos/$GITHUB_REPOSITORY/pulls?state=open&per_page=100" |
   jq -s 'add // []' > "$out/pulls.json"
 while IFS=$'\t' read -r name sha; do
  [[ "$(head_sha "$name")" == "$sha" ]] ||
    { echo "::error::Candidate head moved: $name"; exit 3; }
  [[ "$(jq -r --arg n "$name" '.protected_heads|has($n)' "$m")" == false ]] ||
    { echo "::error::Protected target: $name"; exit 3; }
  jq -e --arg n "$name" 'any(.[]; .head.ref==$n or .base.ref==$n)' "$out/pulls.json" >/dev/null &&
    { echo "::error::Open PR uses $name"; exit 3; }
  git check-ref-format "refs/tags/archive/candle-pruned-20261016/$name" >/dev/null
 done < <(jq -r '.targets[]|[.ref,.sha]|@tsv' "$m")
}
check_protected
check_candidates
echo "PRUNE_PREFLIGHT=PASS targets=99 dry_run=$dry"
if [[ "$dry" == true ]]; then
 echo "99 pinned refs eligible; no tags or branches altered." >> "$GITHUB_STEP_SUMMARY"
 exit 0
fi
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
gh auth setup-git
git fetch --no-tags origin '+refs/heads/*:refs/remotes/origin/*'
git ls-remote --tags origin 'refs/tags/archive/candle-pruned-20261016/*' > "$out/tags"
tagsha() { awk -v n="refs/tags/$1^{}" '$2==n {print $1}' "$out/tags"; }
specs=()
while IFS=$'\t' read -r name sha; do
 tag="archive/candle-pruned-20261016/$name"
 existing="$(tagsha "$tag")"
 if [[ -n "$existing" ]]; then
  [[ "$existing" == "$sha" ]] || { echo "::error::Archive tag conflict: $tag"; exit 3; }
 else
  [[ -z "$(awk -v n="refs/tags/$tag" '$2==n {print $1}' "$out/tags")" ]] ||
    { echo "::error::Unexpected existing tag: $tag"; exit 3; }
  git cat-file -e "$sha^{commit}"
  git tag -a "$tag" "$sha" -m "Archive $name at $sha before post-SM61 cleanup"
  specs+=("refs/tags/$tag:refs/tags/$tag")
 fi
done < <(jq -r '.targets[]|[.ref,.sha]|@tsv' "$m")
if (( ${#specs[@]} > 0 )); then git push --atomic origin "${specs[@]}"; fi
git ls-remote --tags origin 'refs/tags/archive/candle-pruned-20261016/*' > "$out/tags"
while IFS=$'\t' read -r name sha; do
 [[ "$(tagsha "archive/candle-pruned-20261016/$name")" == "$sha" ]] ||
   { echo "::error::Archive unverified: $name"; exit 3; }
done < <(jq -r '.targets[]|[.ref,.sha]|@tsv' "$m")
git ls-remote --heads origin > "$out/heads"
check_protected
check_candidates
leases=(); deletions=()
while IFS=$'\t' read -r name sha; do
 leases+=("--force-with-lease=refs/heads/$name:$sha")
 deletions+=(":refs/heads/$name")
done < <(jq -r '.targets[]|[.ref,.sha]|@tsv' "$m")
[[ "${#deletions[@]}" == 99 ]] || exit 3
git push --atomic "${leases[@]}" origin "${deletions[@]}"
git ls-remote --heads origin > "$out/heads"
while IFS=$'\t' read -r name sha; do
 [[ -z "$(head_sha "$name")" ]] || { echo "::error::Still present: $name"; exit 3; }
done < <(jq -r '.targets[]|[.ref,.sha]|@tsv' "$m")
check_protected
echo "PRUNE=PASS archive_tags=99 deleted_branches=99" | tee -a "$GITHUB_STEP_SUMMARY"
