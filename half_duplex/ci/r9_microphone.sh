#!/usr/bin/env bash
# Local original-tokenizer candidate; share R8's recorder and physical GPU lock.
set -euo pipefail
export HDB2C_MICROPHONE_MODULE=hdb2c.c3_r9_text
exec bash "$(dirname "$0")/r8_microphone.sh" "$@"
