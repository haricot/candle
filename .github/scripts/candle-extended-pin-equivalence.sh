#!/usr/bin/env bash
set -euo pipefail
: "${FEATURE:?}" "${REVIEW_SOURCE_SHA:?}" "${OBSERVED_SOURCE_SHA:?}" "${CURRENT_MAIN_SHA:?}"
python3 "$(dirname -- "${BASH_SOURCE[0]}")/candle-standalone-source-pin.py"
