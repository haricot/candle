# Hosted Candle release CPU targets

Both Legacy and Extended accept `cpu_profile: portable` (the default) or
`cpu_profile: avx2` through workflow_dispatch.

- `portable` selects `-C target-cpu=x86-64`.
- `avx2` selects `-C target-cpu=x86-64-v3`, including AVX2, FMA and F16C.
  These features are present in Nicolas's supplied native rustc configuration.
  This is an explicit instruction-set baseline, not a guess at his CPU model.

Do not use `target-cpu=native` on a hosted runner: it describes that runner.
Other cfg values such as debug_assertions are not CPU flags to copy.
The native hardware's complete feature list can include additional instructions
that v3 does not enable. No claim of identical tuning or measured speedup is made.

The selected profile partitions Cargo caches, is recorded in the SHA256-attested
executable manifest and is checked before any archived executable runs. An AVX2
archive requires all visible CPUs to advertise the v3 capabilities. Existing
explicitly portable archives remain valid. Extra flags and mismatched profiles
are rejected, including flags that only start with target-cpu=x86-64.

The integrated CUDA prechecks and final executables now use the same release
directory and CPU flags. This avoids a redundant debug build of CUDA build
scripts/kernels. Cargo still has distinct units for different packages/features;
CPU-only standalone validation is also distinct. Multiple Cargo invocations do
not imply recompilation when their inputs are identical. CPU/GPU execution uses
the already archived release executables, without Cargo on the GPU runner.

CUDA SM61, immutable compiler image resolution, ASD, model evidence, archive
hashes, campaign/attempt binding and source promotion rules remain unchanged.
No new model weights or local container image is needed for these changes.
