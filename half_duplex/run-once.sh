#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
report_dir="$ROOT/reports/hdb0-$run_id"
usage() {
  cat <<USAGE
Usage: ./run-once.sh {preflight|check|selftest|inspect|simulate} [WAV]
  preflight  Check Cargo and optional local Candle revision; read-only
  check      cargo test --workspace --locked --offline
  selftest   check + synthetic turn report in reports/hdb0-<id>/
  inspect    inspect provided WAV or bundled synthetic WAV
  simulate   synthetic turn decisions applied to provided WAV (NO VAD/ASR)
Environment: CANDLE_ROOT=/path/to/local/haricot/candle (optional)
USAGE
}
preflight() {
  # HDB0 records an external reference, not an enforced Cargo dependency.
  local pin
  pin="$(sed -n 's/^CANDLE_COMMIT=//p' "$ROOT/config/candle-extended.lock" | head -n 1)"
  if [[ ! "$pin" =~ ^[0-9a-f]{40}$ ]]; then
    echo "CANDLE_PIN=INVALID_LOCK_FILE"
    return 2
  fi
  echo "HDB0=SOFTWARE_ONLY_NO_ASR"
  echo "CANDLE_REPO=https://github.com/haricot/candle.git"
  echo "CANDLE_EXPECTED_COMMIT=$pin"
  echo "CANDLE_INFERENCE=NOT_IMPLEMENTED_HDB0"
  echo "CANDLE_PIN=REFERENCE_ONLY_NOT_A_CARGO_DEPENDENCY"
  if command -v cargo >/dev/null 2>&1; then
    echo "CARGO=$(cargo --version)"
  else
    echo "CARGO=MISSING (install Rust 1.99 or use rustup)"
    return 2
  fi
  local candle_root="${CANDLE_ROOT:-$HOME/tmp/candle_extented/candle}"
  if git -C "$candle_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    local sha
    sha="$(git -C "$candle_root" rev-parse HEAD)"
    echo "CANDLE_ROOT=$candle_root"
    echo "CANDLE_LOCAL_HEAD=$sha"
    if [[ "$sha" == "$pin" ]]; then
      echo "CANDLE_LOCAL_MATCH=YES"
    else
      echo "CANDLE_LOCAL_MATCH=NO_NON_BLOCKING"
      echo "CANDLE_LOCAL_NOTE=Local_checkout_differs_from_HDB0_reference"
    fi
  else
    echo "CANDLE_LOCAL=NOT_FOUND_NOT_REQUIRED_FOR_HDB0"
  fi
}
run_cargo() {
  cargo test --workspace --locked --offline
}
case "${1:---help}" in
  preflight)
    preflight
    ;;
  check)
    run_cargo
    echo 'HDB0=PASS_CARGO_TESTS_NO_ASR'
    ;;
  selftest)
    run_cargo
    mkdir -p "$report_dir"
    cargo run --quiet --locked --offline -p half-duplex-cli --bin half-duplex -- selftest "$report_dir/selftest.json"
    echo "HDB0_REPORT_DIR=$report_dir"
    ;;
  inspect)
    input="${2:-$ROOT/tests/fixtures/hdb0-tone-16k.wav}"
    mkdir -p "$report_dir"
    cargo run --quiet --locked --offline -p half-duplex-cli --bin half-duplex -- inspect-wav "$input" "$report_dir/inspect.json"
    echo "HDB0_REPORT_DIR=$report_dir"
    ;;
  simulate)
    input="${2:-$ROOT/tests/fixtures/hdb0-tone-16k.wav}"
    mkdir -p "$report_dir"
    cargo run --quiet --locked --offline -p half-duplex-cli --bin half-duplex -- simulate-turn "$input" "$report_dir/simulate.json"
    echo "HDB0_REPORT_DIR=$report_dir"
    ;;
  *) usage; exit 2 ;;
esac
