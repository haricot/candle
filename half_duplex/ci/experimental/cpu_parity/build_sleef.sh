#!/usr/bin/env bash
# Standalone SLEEF revision used by PyTorch 2.12.1; never links ATen/libtorch.
set -euo pipefail
source_dir="${1:?absolute source directory required}"
build_dir="${2:?absolute build directory required}"
revision=5a1d179df9cf652951b59010a2d2075372d67f68
if [[ ! -d "$source_dir/.git" ]];then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/shibatch/sleef.git "$source_dir"
  git -C "$source_dir" fetch --depth 1 origin "$revision"
  git -C "$source_dir" checkout --detach FETCH_HEAD
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
git -C "$source_dir" diff --quiet
cmake -S "$source_dir" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DSLEEF_BUILD_TESTS=OFF -DSLEEF_BUILD_GNUABI_LIBS=OFF \
  -DSLEEF_BUILD_DFT=OFF -DSLEEF_BUILD_QUAD=OFF -DSLEEF_ENABLE_CUDA=OFF \
  -DSLEEF_ENFORCE_AVX2=ON -DSLEEF_DISABLE_AVX512F=OFF -DSLEEF_DISABLE_AVX=OFF \
  -DSLEEF_DISABLE_SSE4=OFF -DSLEEF_DISABLE_FMA4=OFF \
  -DSLEEF_DISABLE_FLOAT128=ON -DSLEEF_DISABLE_LONG_DOUBLE=ON
cmake --build "$build_dir" --parallel "${HDB2C_SLEEF_JOBS:-4}"
