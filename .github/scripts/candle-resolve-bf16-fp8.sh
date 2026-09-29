#!/usr/bin/env bash
# Replay ONE reviewed BF16→FP8 four-file union ONLY when all twelve
# exact Git conflict-stage blob IDs match the frozen review evidence.
set -euo pipefail
work="$1"; source_bf16="$2"; source_fp8="$3"; report="$4"
root=".github/resolutions/bf16-fp8-exact-v1"
manifest="$root/stage-manifest.json"
[[ "$source_bf16" == 07980b71e52cfccc4c95c0c747043c9fb985dcab &&
   "$source_fp8" == 121f14beb269754ff9edcb3a33e77c38844d5c26 ]] || exit 3
jq -e --arg b "$source_bf16" --arg f "$source_fp8" \
  '.schema_version==1 and .source_bf16==$b and .source_fp8==$f and
  (.files|length)==4' "$manifest" >/dev/null
expected="$(mktemp)"; actual="$(mktemp)"
trap 'rm -f "$expected" "$actual"' EXIT
jq -r '.files|keys[]' "$manifest" | LC_ALL=C sort > "$expected"
git -C "$work" diff --name-only --diff-filter=U | LC_ALL=C sort > "$actual"
diff -u "$expected" "$actual" > "$report/conflicts/fp8-reviewed-scope.diff" || {
  echo "::error::Unreviewed conflict path detected; BF16+FP8 replay refused"; exit 3;
}
while IFS=$'\t' read -r path base ours theirs; do
  [[ -s "$root/$path" ]] || exit 3
  mapfile -t stages < <(git -C "$work" ls-files -u -- "$path" | awk '{print $2}')
  [[ "${#stages[@]}" -eq 3 &&
     "${stages[0]}" == "$base" &&
     "${stages[1]}" == "$ours" &&
     "${stages[2]}" == "$theirs" ]] || {
    echo "::error::Unreviewed conflict-stage SHA for $path"; exit 3;
  }
done < <(jq -r '.files|to_entries[]|[.key,.value.base,.value.ours,.value.theirs]|@tsv' "$manifest")
while IFS= read -r path; do
  install -D -m 644 "$root/$path" "$work/$path"
  git -C "$work" add -- "$path"
done < "$expected"
[[ -z "$(git -C "$work" ls-files -u)" ]] || exit 3
git -C "$work" diff --cached --check || exit 3
git -C "$work" commit -m "integrate(fp8): exact reviewed BF16+FP8 source union" \
  > "$report/logs/fp8-reviewed-commit.log" 2>&1
printf 'EXACT_REVIEWED_BF16_FP8_REPLAY=%s\n' "$(git -C "$work" rev-parse HEAD)" \
  > "$report/logs/fp8-reviewed-resolution.txt"
