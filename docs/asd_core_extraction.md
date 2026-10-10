# ASD Core V2 — standalone extraction

- Parent: `haricot/candle:main` `66a8cf184a5a519671454066b1b9efd446ec9f5c`.
- Historical source: `cuda_asd_runner` `0835f4a7d57c6c190305ed9c41713c867ffb34d7`.
- Immutable preservation: `archive/cuda_asd_runner-0835f4a-20260925`.

## Ownership

This branch holds the grouped-convolution contracts and CPU reference,
native grouped CUDA/cuDNN/Metal backends, V2-only ASD policy parser, seven
SM61 exact grouped kernels, DW5x5 and ASD-only fused activation kernels,
V2 production routing and validation harnesses.

Intentionally excluded: legacy BF16, FP8, MXFP4/NVFP4, general cuDNN
execution-failure fallback, MoE SIMT F16, RTMPose application, LFM2 and
unrelated quantization/CPU performance experiments. Those belong in their
own feature branches. A grouped cuDNN native operation is distinct from
the general pre-Volta execution-failure recovery policy.

## Eight independently inspectable commits

1. `feat(conv): establish grouped convolution CPU contracts`
2. `feat(conv): add native CUDA and cuDNN grouped backends`
3. `feat(conv): add Metal native grouped-transpose backend`
4. `feat(asd): introduce V2-only device-scoped policy contracts`
5. `feat(cuda-asd): register exact SM61 grouped kernel catalogue`
6. `feat(cuda-asd): isolate exact DW5x5 and fused Conv2D kernels`
7. `feat(asd): connect V2 production dispatch and strict runtime guards`
8. `test(asd): consolidate V2 evidence and cross-backend CI`

## Evidence and limitations

- `cargo fmt --all -- --check`
- CPU: `cargo check -p candle-core -p candle-nn --no-default-features`
- CPU: grouped convolution / transpose suites, candle-nn library tests
- macOS: `cargo check -p candle-core -p candle-nn --features metal`
- SM61: exact PTX and grouped cuDNN tests, device identity + thermal
  evidence and all 12 V2 promoted decisions must still be validated on
  a real GTX 1080 before production promotion.

The original historical benchmark numbers are provenance, not evidence
that new code or revised compilation settings produce the same timings.
The separate ASD CUDA fusion module requires fresh GPU parity testing.
