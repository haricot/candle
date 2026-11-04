#!/usr/bin/env bash
# Deliberately opt-in full CPU F32 campaign. Python is reference tooling only.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: NEMOTRON_REFERENCE_PYTHON=/path/to/python bash $0 MODEL_DIRECTORY NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
model=$(realpath -- "$1")
output=$(realpath -m -- "$2")
python=${NEMOTRON_REFERENCE_PYTHON:-python3}
if [[ -e "$output" ]]; then
  echo "Choose a new qualification output directory: $output" >&2
  exit 2
fi
mkdir -p -- "$output"
cd -- "$repo"
target=$(realpath -m -- "${CARGO_TARGET_DIR:-"$repo/target/nemotron-reference"}")
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export HF_HUB_DISABLE_TELEMETRY=1
export OTEL_SDK_DISABLED=true
cargo build --release --target-dir "$target" -p candle-examples --example nemotron_asr
"$python" "$script_dir/fixture.py" --output "$output/fixture"
binary="$target/release/examples/nemotron_asr"
run_case() {
  local name=$1 mode=$2 language=$3 right=$4 wav=$5
  "$python" "$script_dir/reference.py" --model-dir "$model" --input "$wav" \
    --mode "$mode" --language "$language" --lookahead "$right" --output "$output/reference-$name" \
    > "$output/reference-$name.log" 2>&1
  "$binary" --model-dir "$model" --input "$wav" --device cpu --dtype f32 \
    --mode "$mode" --language "$language" --lookahead "$right" --dump-activations \
    --expect-result "$output/reference-$name/result.json" --output "$output/candle-$name" \
    > "$output/candle-$name.log" 2>&1
  "$python" "$script_dir/compare.py" "$output/reference-$name" "$output/candle-$name" \
    --output "$output/metrics-$name.json" > "$output/compare-$name.log"
  echo "PASS: $name"
}
for mode in offline streaming; do
  for language in fr-FR auto; do
    run_case "$mode-$language" "$mode" "$language" 3 "$output/fixture/public-fr-16k.wav"
  done
done
for right in 0 3 6 13; do
  run_case "long-r$right" streaming fr-FR "$right" "$output/fixture/public-fr-repeat-16k.wav"
done
echo "PASS: 8 official-reference CPU F32 cases. Metrics: $output/metrics-*.json"
