#!/usr/bin/env bash
# Native F32 CPU baseline, then exact CUDA decisions for the same user-supplied WAV.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash $0 MODEL_DIRECTORY INPUT_WAV OUTPUT_DIRECTORY [cuda-cudnn|cuda|cpu]" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
model=$(realpath -- "$1")
input=$(realpath -- "$2")
output=$(realpath -m -- "$3")
backend=${4:-cuda-cudnn}
features=()
case "$backend" in
  cuda-cudnn) features=(--features cuda,cudnn) ;;
  cuda) features=(--features cuda) ;;
  cpu) ;;
  *) echo "Unsupported device: $backend" >&2; exit 2 ;;
esac
if [[ -e "$output" ]]; then
  echo "Choose a new output directory to keep each qualification separate: $output" >&2
  exit 2
fi
mkdir -p -- "$output"
cd -- "$repo"
target=${CARGO_TARGET_DIR:-"$repo/target/nemotron-$backend"}
target=$(realpath -m -- "$target")
git rev-parse HEAD > "$output/source-commit.txt"
git write-tree > "$output/source-index-tree.txt"
cargo build --release --target-dir "$target" -p candle-examples --example nemotron_asr "${features[@]}"
binary="$target/release/examples/nemotron_asr"
for language in fr-FR auto; do
  "$binary" --model-dir "$model" --input "$input" --mode streaming \
    --language "$language" --lookahead 3 --device cpu --dtype f32 \
    --output "$output/cpu-$language" > "$output/cpu-$language.log"
  "$binary" --model-dir "$model" --input "$input" --mode streaming \
    --language "$language" --lookahead 3 --device "$backend" --dtype f32 \
    --expect-result "$output/cpu-$language/result.json" \
    --output "$output/$backend-$language" > "$output/$backend-$language.log"
done
echo "PASS: $backend matches the native CPU tokens, text and all frame decisions (fr-FR and auto)."
echo "Results: $output"
