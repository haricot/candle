# R1.24 S5.4 — SM61 CUBIN feasibility: measured result (2026-11-21)

Source of the numbers: user-run headless-relogin
`target/dkr-r124-s54/headless-20261121T152124421926Z/result.json`.
Qualification markers: `DKR_R124_S54_DRIVER=PASS`,
`DKR_R124_S54_VRAM=PASS sample_count=93 sampled_peak_mib=343.19`,
`DKR_R124_S54_PASS cubin_pilot_only=1`, exit 0.

## Direct driver module loading

Three rounds on NVIDIA GTX 1080 Max-Q (SM61), CUDA driver 580.196.07,
`CUDA_MODULE_LOADING=EAGER`. Values below are median `cuModuleLoadData`
times in milliseconds, **not end-to-end inference or steady-state kernel
execution**. Each observation used a new process and context. Cold PTX
used its own empty `CUDA_CACHE_PATH`, warm PTX reused that cache, and
CUBIN loading used the identical architecture and selected symbol.

| Family | Cold PTX (ms) | Warm PTX (ms) | SM61 CUBIN (ms) | Cold PTX / CUBIN |
|---|---:|---:|---:|---:|
| fill | 111.107 | 0.417 | 0.255 | 435.714× |
| mmq_instance_q4_k | 3202.206 | 10.095 | 5.878 | 544.778× |
| mmvq_gguf | 7101.482 | 22.244 | 5.663 | 1254.014× |

The module lookup itself took approximately 5–9 microseconds. Module VRAM
deltas observed within an existing driver context were 0 MiB for fill,
10 MiB for Q4_K MMQ and 2 MiB for MMVQ **in each tested image format**.
They are driver accounting deltas, not total running-process VRAM.

The `fill_f32` kernel produced exactly 1.25 across 4096 elements in
both PTX and CUBIN mode (three of three pairs). The MMQ and MMVQ CUBIN
symbols were successfully resolved on the driver, but **their CUBIN
entrypoints were not launched for numerical comparison** in S5.4.

## Concurrent workload VRAM observation

S5.3 Nextest benchmark replay, NVML sampling at 25ms:
- 93 samples;
- sampled system GPU memory minimum 108,068,864 bytes (103.06 MiB);
- sampled maximum 359,858,176 bytes (343.19 MiB);
- sampled range about 240.13 MiB.

These samples cannot establish peak allocator usage or attribute every
byte to Candle. They may miss short peaks. S5.3's original timings
remain authoritative because VRAM sampling may perturb latency.

## What is proved, and the next acceptance gate

**Proved for this pilot:**
- Existing S5.1 PTX packages/manifest SHA-256 were not modified;
- offline SM61 CUBIN artifacts were generated and load on the tested GPU;
- driver cold PTX module compilation dominates the first-module load and
  can be bypassed with CUBIN by factors of about 436× to 1254× here;
- 4096-element fill parity passes;
- no increase in the *observed per-module driver VRAM delta* between PTX and
  CUBIN for these three families.

**Not yet proved:**
- production Candle runtime uses any CUBIN (it still embeds PTX);
- numerical equivalence for CUBIN MMVQ/MMQ/MoE real launch signatures;
- cold **application** startup gain or full model performance;
- complete 25-family CUBIN packaging, validation and runtime selection;
- absence of transient memory peaks or compatibility beyond SM61.

**Recommended S5.4b acceptance before any production promotion:**
1. Test actual CUBIN MMVQ/MMQ launches against the same S5.2 numerical
   reference on the device, not just module symbol lookup.
2. Add a generic optional native-code image record keyed by *target CC,
   PTX provenance SHA, CUBIN SHA, producer toolchain and ABI*. Do not
   hardcode Pascal runtime policy in Candle.
3. In an isolated experimental feature, use build-time-verified embedded
   CUBIN for exact-target matches; reject unverified file overrides and
   retain the sealed PTX baseline.
4. Require output parity, loaded-image provenance and headless cold/warm
   application benchmark before selecting CUBIN by default.
5. Only then extend to the remaining 22 package families, without
   recompiling existing PTX or running `cargo clean`.

Do not interpret the 1254× figure as a steady-state throughput gain.
