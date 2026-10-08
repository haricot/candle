#!/usr/bin/env bash
set -euo pipefail
work="$1"; source_bf16="$2"; source_fp8="$3"; source_fp4="$4"; source_cudnn="$5"; source_moe="$6"; source_asd="$7"; report="$8"
root=".github/resolutions/extended-asd-v3-exact-v1"
manifest="$root/stage-manifest.json"
[[ "$source_bf16" == 07980b71e52cfccc4c95c0c747043c9fb985dcab &&
   "$source_fp8" == 121f14beb269754ff9edcb3a33e77c38844d5c26 &&
   "$source_fp4" == 2976961e97f0aff7a353e45b209653c392f2c657 &&
   "$source_cudnn" == 84a5a3694ac4c03b4d7e7d20d349ae58d69d7c1f &&
   "$source_moe" == 1c62e23cf6c986826ef12c9d544ee9356972d59b &&
   "$source_asd" == aef4738f8ef513ef425dbaa345358cdd3ae8cbb7 ]] || exit 3
jq -e --arg b "$source_bf16" --arg f "$source_fp8" --arg p "$source_fp4" --arg c "$source_cudnn" --arg m "$source_moe" --arg a "$source_asd" '
  .schema_version==1 and .scope=="exact-extended-asd-v3-final-merge" and
  .source_bf16==$b and .source_fp8==$f and .source_fp4==$p and
  .source_cudnn==$c and .source_moe==$m and .source_asd_v3==$a and (.files|length)==4
' "$manifest" >/dev/null
expected="$(mktemp)"; actual="$(mktemp)"
trap 'rm -f "$expected" "$actual"' EXIT
jq -r '.files|keys[]' "$manifest" | LC_ALL=C sort > "$expected"
git -C "$work" diff --name-only --diff-filter=U | LC_ALL=C sort > "$actual"
diff -u "$expected" "$actual" > "$report/conflicts/asd-v3-reviewed-scope.diff" || {
  echo "::error::Unreviewed conflict path detected; Extended ASD V3 replay refused"; exit 3;
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
git -C "$work" commit -m "integrate(asd-v3): reviewed Legacy CUDA plus runtime-only ASD union" > "$report/logs/asd-v3-reviewed-commit.log" 2>&1
printf 'EXACT_REVIEWED_EXTENDED_ASD_V3_REPLAY=%s\n' "$(git -C "$work" rev-parse HEAD)" > "$report/logs/asd-v3-reviewed-resolution.txt"
