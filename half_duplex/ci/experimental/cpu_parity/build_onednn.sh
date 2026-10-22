#!/usr/bin/env bash
# CPU-only dependency for a local numerical experiment; no runner registration.
set -euo pipefail
source_dir="${1:?absolute source directory required}"
build_dir="${2:?absolute build directory required}"
revision=03c022d3ffdcee958cfacbe720048e725fdf644c
if [[ ! -d "$source_dir/.git" ]];then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/uxlfoundation/oneDNN.git "$source_dir"
  git -C "$source_dir" fetch --depth 1 origin "$revision"
  git -C "$source_dir" checkout --detach FETCH_HEAD
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
git -C "$source_dir" diff --quiet
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF \
  -DONEDNN_BUILD_GRAPH=OFF -DDNNL_CPU_RUNTIME=OMP -DDNNL_GPU_RUNTIME=NONE \
  -DDNNL_ENABLE_WORKLOAD=INFERENCE '-DDNNL_ENABLE_PRIMITIVE=CONVOLUTION;REORDER' \
  -DDNNL_LIBRARY_TYPE=SHARED
cmake --build "$build_dir" --parallel "${HDB2C_ONEDNN_JOBS:-4}"
