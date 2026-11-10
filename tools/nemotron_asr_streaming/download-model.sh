#!/usr/bin/env bash
# Download the unchanged official HF files. No Python or conversion step.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: bash $0 MODEL_DIRECTORY" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$1"
model_dir=$(cd -- "$1" && pwd)
revision=ea30d66debe3740a08b573244286791d423d6b3e
while read -r expected file; do
  if [[ -f "$model_dir/$file" ]] && [[ $(sha256sum "$model_dir/$file" | cut -d ' ' -f 1) == "$expected" ]]; then
    echo "Verified $file"
    continue
  fi
  curl --fail --location --retry 3 --continue-at - \
    "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/$revision/$file" \
    --output "$model_dir/$file.part"
  actual=$(sha256sum "$model_dir/$file.part" | cut -d ' ' -f 1)
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $file; existing complete file was not replaced" >&2
    exit 1
  fi
  mv -- "$model_dir/$file.part" "$model_dir/$file"
done < "$script_dir/model.sha256"
echo "Model ready: $model_dir (revision $revision)"
