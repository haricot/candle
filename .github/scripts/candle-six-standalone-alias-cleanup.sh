#!/usr/bin/env bash
set -euo pipefail
umask 077
cfg=.github/six-standalone-alias-cleanup-launch.json
report="$RUNNER_TEMP/six-standalone-alias-cleanup"
mkdir -p "$report"
jq -e '.schema_version==1 and .operation=="delete-six-duplicate-standalone-aliases" and
  (.canonical_feature_refs|length)==6 and (.duplicate_refs|length)==6 and
  .require_no_open_pr_references==true and .atomic_delete==true' "$cfg" >/dev/null || exit 3
gh auth setup-git
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
open_prs="$(gh api --paginate "repos/$GITHUB_REPOSITORY/pulls?state=open&per_page=100" | jq -s 'add // []')"
leases=(); deletes=()
while IFS=$'\t' read -r canonical sha duplicate dsha; do
  [[ "$sha" == "$dsha" ]] || { echo "::error::Manifest mismatch for $canonical"; exit 3; }
  live_c="$(git ls-remote --heads origin "refs/heads/$canonical" | cut -f1)"
  live_d="$(git ls-remote --heads origin "refs/heads/$duplicate" | cut -f1)"
  [[ "$live_c" == "$sha" && "$live_d" == "$sha" ]] || {
    echo "::error::Ref moved: $canonical=$live_c $duplicate=$live_d expected=$sha"; exit 3;
  }
  if jq -e --arg b "$duplicate" 'any(.[]; .head.ref==$b or .base.ref==$b)' <<<"$open_prs" >/dev/null; then
    echo "::error::Open PR still references $duplicate"; exit 3
  fi
  leases+=("--force-with-lease=refs/heads/$duplicate:$sha")
  deletes+=(":refs/heads/$duplicate")
done < <(jq -r '.canonical_feature_refs|to_entries[] as $c |
  [$c.key,$c.value,($c.key+"_standalone"),(.duplicate_refs[$c.key+"_standalone"])]|@tsv' "$cfg")
[[ "${#deletes[@]}" -eq 6 ]] || exit 3
git push --atomic "${leases[@]}" origin "${deletes[@]}" > "$report/delete.log" 2>&1 || {
  tail -n 80 "$report/delete.log"; exit 1;
}
while IFS=$'\t' read -r canonical sha duplicate dsha; do
  [[ "$(git ls-remote --heads origin "refs/heads/$canonical" | cut -f1)" == "$sha" ]] || exit 3
  [[ -z "$(git ls-remote --heads origin "refs/heads/$duplicate")" ]] || exit 3
done < <(jq -r '.canonical_feature_refs|to_entries[] as $c |
  [$c.key,$c.value,($c.key+"_standalone"),(.duplicate_refs[$c.key+"_standalone"])]|@tsv' "$cfg")
jq -n --slurpfile c "$cfg" --arg run "$GITHUB_RUN_ID"   '{status:"STANDALONE_ALIASES_DELETED",run_id:$run,
    deleted_refs:($c[0].duplicate_refs|keys),
    preserved_canonical_refs:$c[0].canonical_feature_refs}' > "$report/report.json"
echo "Deleted six duplicate *_standalone refs atomically; canonical refs unchanged" >> "$GITHUB_STEP_SUMMARY"
