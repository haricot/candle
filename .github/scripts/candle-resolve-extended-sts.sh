#!/usr/bin/env bash
# Audited, fail-closed resolution of the single STS/main-vs-ASD-V3 Cargo merge.
set -euo pipefail
work="${1:?aggregate worktree required}"
report="${2:?report required}"
file='candle-core/Cargo.toml'
fixture='.github/resolutions/sts-main-v1/Cargo.toml'

[[ "$(git -C "$work" diff --name-only --diff-filter=U)" == "$file" ]] || {
  echo 'STS_RESOLUTION=FAIL unexpected conflict scope' >&2
  exit 3
}
mapfile -t stages < <(git -C "$work" ls-files -u -- "$file" | awk '{print $2}')
[[ "${#stages[@]}" -eq 3 &&
   "${stages[0]}" == b04c0b4e4dc158bccc6b72ab9d70e3f17914c673 &&
   "${stages[1]}" == dd3681ae670fe37ea4495f863554326e1660c259 &&
   "${stages[2]}" == 6acf079d54dfaec88f46d051c8695166c3a38438 ]] || {
  echo 'STS_RESOLUTION=FAIL unreviewed merge stages' >&2
  exit 3
}
[[ "$(git hash-object "$fixture")" == 52220b3076fa97cc4a5697b51c076631a295f9c3 ]] || exit 3
install -m 644 "$fixture" "$work/$file"
git -C "$work" add -- "$file"
[[ -z "$(git -C "$work" ls-files -u)" ]] || exit 3
git -C "$work" diff --cached --check || exit 3
git -C "$work" commit -m 'integrate(sts): reviewed ASD-V3 Cargo feature union' > "$report/logs/sts-reviewed-merge.log" 2>&1
[[ "$(git -C "$work" rev-parse "HEAD:$file")" == 52220b3076fa97cc4a5697b51c076631a295f9c3 ]] || exit 3
echo 'STS_RESOLUTION=PASS'
