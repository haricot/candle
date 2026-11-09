#!/usr/bin/env bash
# Install the exact source-owned bridge ONLY after the 13-source ASD V3 aggregate.
set -euo pipefail
work="${1:?aggregate worktree required}"
report="${2:?report required}"
patch1='.github/sts-asd-v3-exact.patch'
patch2='.github/sts-asd-v3-activate.patch'

[[ "$(git -C "$work" hash-object "$patch1")" == 80d4206e67b0fda7aacd2db2c3dbf8e2c4065bb9 &&
   "$(git -C "$work" hash-object "$patch2")" == d46c0b49f4df72a7c440a6585ee6d594f08fd975 ]] || {
  echo 'STS_COMPOSE=FAIL unreviewed source-owned bridge patch' >&2
  exit 3
}
[[ -z "$(git -C "$work" status --porcelain)" ]] || {
  echo 'STS_COMPOSE=FAIL aggregate not clean after merge' >&2
  exit 3
}

# A new upstream ASD / Legacy resolution must receive its OWN reviewed bridge.
declare -A original=(
  [candle-core/src/asd_runtime_extensions.rs]=cf3157e827620a294c96c703d192c625d640bc15
  [candle-core/src/conv.rs]=d6dbe36067e3083bf5cc30e011759575dac43f78
  [candle-core/src/conv/asd_exact_cuda.rs]=0d3c49837a65d263be458277715bacd217e5ffc1
  [candle-core/src/conv/grouped_transpose_dispatch.rs]=2015117548f780dd1b14909b47571ec27fcf95a1
  [candle-core/src/cuda_backend/device/asd_modules.rs]=991dcff00aad2cde6a3dd598f0839a75133c7d09
  [candle-kernels/src/asd_exact.rs]=39b4c6f9cb02066347e0b539c5fabc47f79044d8
  [candle-core/src/stateful.rs]=6d7ace29ab0f4fda512c504b5a682b22ef07866f
  [candle-core/examples/d7_shortconv_f16_validate.rs]=bff490f60880c69ec0f3e8d80c48b32dbfd51669
)
declare -A composed=(
  [candle-core/src/asd_runtime_extensions.rs]=beb67b7b07f04384d1e5d6d9bca8e42f6b843148
  [candle-core/src/conv.rs]=f60df2a64d49e31379a774351c8cb6af740aaf02
  [candle-core/src/conv/asd_exact_cuda.rs]=26fbf64ed3fd317193df8980c98d2e7ebdddd061
  [candle-core/src/conv/grouped_transpose_dispatch.rs]=5150a2a108971f9d2f47db599de4de8711227057
  [candle-core/src/cuda_backend/device/asd_modules.rs]=8d873b938c862572a9942b352240d52f7162b655
  [candle-kernels/src/asd_exact.rs]=66a9ef41750973a09fb42b04d98591867b5cea40
  [candle-core/src/stateful.rs]=aed38799eeca3aaf212426acd2136d4f67f85a29
  [candle-core/examples/d7_shortconv_f16_validate.rs]=88ea530a614b605299eedb7d9d82030cceaaf664
)
for path in "${!original[@]}"; do
  [[ "$(git -C "$work" rev-parse "HEAD:$path")" == "${original[$path]}" ]] || {
    echo "STS_COMPOSE=FAIL unreviewed bridge input $path" >&2
    exit 3
  }
done

git -C "$work" apply --index "$patch1"
git -C "$work" apply --index "$patch2"
git -C "$work" diff --cached --check
for path in "${!composed[@]}"; do
  [[ "$(git -C "$work" hash-object "$path")" == "${composed[$path]}" ]] || {
    echo "STS_COMPOSE=FAIL unreviewed bridge output $path" >&2
    exit 3
  }
done
[[ "$(git -C "$work" hash-object candle-core/src/diagnostics.rs)" == f653ad42750ffb575640c6b59080dbb5f25d3034 &&
   "$(git -C "$work" hash-object candle-core/src/storage.rs)" == b861ffbb3408c22ce8172a414b41436f617f54f9 &&
   "$(git -C "$work" hash-object candle-core/src/stateful/cuda.rs)" == bd4ea4d07dec71e71961c180cc853b97095276b1 ]] || {
  echo 'STS_COMPOSE=FAIL semantic observer or F16 CUDA identity mismatch' >&2
  exit 3
}
git -C "$work" commit -m 'integrate(sts): exact stateful F16 ASD V3 bridge after 13 sources' > "$report/logs/sts-exact-composition.log" 2>&1
echo 'STS_COMPOSE=PASS source=main_asd_after_13 qualifier=SM61_UNQUALIFIED'
