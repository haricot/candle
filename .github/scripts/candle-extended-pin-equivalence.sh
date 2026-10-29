#!/usr/bin/env bash
# Fail-closed source pin validation for previously promoted standalone heads.
# A relocated source is acceptable only when its exact Git delta equals the
# explicitly reviewed source delta (the old reviewed SHA never changes).
set -euo pipefail
: "${FEATURE:?}" "${REVIEW_SOURCE_SHA:?}" "${OBSERVED_SOURCE_SHA:?}" "${CURRENT_MAIN_SHA:?}"
: "${ALLOW_PROMOTED_EQUIVALENT_SOURCE:=false}"

reject() {
  echo "::error::Pinned Extended source $FEATURE: $*" >&2
  exit 3
}

if ! git cat-file -e "${REVIEW_SOURCE_SHA}^{commit}" 2>/dev/null; then
  git fetch --no-tags origin "$REVIEW_SOURCE_SHA" >/dev/null 2>&1 ||
    reject "reviewed source commit $REVIEW_SOURCE_SHA unavailable"
fi
git cat-file -e "${OBSERVED_SOURCE_SHA}^{commit}" ||
  reject "observed source commit $OBSERVED_SOURCE_SHA unavailable"

if [[ "$OBSERVED_SOURCE_SHA" == "$REVIEW_SOURCE_SHA" ]]; then
  echo "EXTENDED_SOURCE_PIN=EXACT feature=$FEATURE sha=$OBSERVED_SOURCE_SHA"
  exit 0
fi

[[ "$ALLOW_PROMOTED_EQUIVALENT_SOURCE" == true ]] ||
  reject "source head $OBSERVED_SOURCE_SHA differs from reviewed SHA $REVIEW_SOURCE_SHA"
read -r -a ancestry <<< "$(git rev-list --parents -n 1 "$OBSERVED_SOURCE_SHA")"
(( ${#ancestry[@]} == 2 )) ||
  reject "relocated source must be a single-parent prepared commit"
parent="${ancestry[1]}"
git merge-base --is-ancestor "$parent" "$CURRENT_MAIN_SHA" ||
  reject "prepared source parent is not an ancestor of current main"
subject="$(git log -1 --format=%s "$OBSERVED_SOURCE_SHA")"
expected_subject="prepare($FEATURE): exact source delta $REVIEW_SOURCE_SHA on main $parent"
[[ "$subject" == "$expected_subject" ]] ||
  reject "relocated source does not carry the reviewed exact-delta preparation identity"

base="$(git merge-base "$parent" "$REVIEW_SOURCE_SHA")" ||
  reject "no common base for source-delta proof"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
git diff --binary --full-index "$base" "$REVIEW_SOURCE_SHA" > "$scratch/reviewed.patch"
git diff --binary --full-index "$parent" "$OBSERVED_SOURCE_SHA" > "$scratch/observed.patch"
[[ -s "$scratch/reviewed.patch" ]] ||
  reject "reviewed source delta is unexpectedly empty"
cmp -s "$scratch/reviewed.patch" "$scratch/observed.patch" ||
  reject "prepared source delta differs byte-for-byte from reviewed source"

echo "EXTENDED_SOURCE_PIN=REVIEWED_EQUIVALENT feature=$FEATURE reviewed=$REVIEW_SOURCE_SHA observed=$OBSERVED_SOURCE_SHA parent=$parent"
