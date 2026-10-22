# Reserved for HDB3 benchmarks

No ASR WER, RTF, p50/p95, or GPU throughput data are available in HDB0.
The synthetic event test is not a benchmark of FireRedVAD, Smart Turn, or TranscribePsy.

### HDB2-B CUDA sm_61 in-process comparative bench

Use `./run-hdb2b.sh bench INPUT cpu 3 1` and `./run-hdb2b.sh bench INPUT cuda 3 1`.
The generated report's `hot_ms_median` excludes model loading, unlike the
HDB2-A one-shot measurement. To compare, use `./run-hdb2b.sh compare ...`.
No source-only synthetic pass is an inference qualification.
