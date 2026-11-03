# ASD v1 exact-domain promotion closure

The first production promotion is deliberately limited to the exact CT1D g2 signature validated by `grouped_conv_transpose_asd_validate`.

Authoritative integrated validation was performed headless with 16 warmup samples, 40 timed samples, and 32 launches per sample. Two consecutive replications passed numerical parity, selected/submitted backend identity, synchronized CUDA completion, out-of-domain CURRENT fallback, drift <= 5%, integrated speedup >= 1.10x, and p90 non-regression.

Conservative promotion evidence uses the worse observed integrated speedup: 4.847072x. The second replication measured 4.903093x. No generalized `groups >= N` rule is implied by this promotion.

A promoted `ASD-EXACT-POLICY-V1` is accepted in normal builds without `CANDLE_ASD_VALIDATION=1`; candidate and integrated-only policies still require validation mode. Device-scoped policies still require the matching `CANDLE_ASD_TARGET_GPU_UUID` at build time.
