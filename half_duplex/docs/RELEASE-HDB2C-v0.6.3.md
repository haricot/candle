
## HDB2-C v0.6.3 — isolated NA single-change A/B

The confirmed C1 same-PCM experiment showed NeMo recognizes the user's French
while baseline Candle degrades it. In the source code of `parakeet-candle`,
`normalize="NA"` currently runs the global normalization `else` branch.
The NeMo ASR reference implementation regards `NA` as **no normalization**.
This archive adds the `na-ab` candidate: **only** changes `} else {` to
`} else if args.normalize == "all_features" {` at the global normalization
branch inside `src/parakeet/audio.rs::get_logmel`. Other source paths remain identical.

**First** run the baseline numerical probe using the *real* report path:

```bash
./run-hdb2c.sh probe \
 "$HOME/tmp/haricot-half-duplex/reports/hdb2c-20261018T195644Z-87019-000" \
 --variant cpu
```

Then **create a complete isolated Rust runtime copy**, compile and compare candidate
transcripts and real log-Mel/encoder probes, without changing the live binary:

```bash
./run-hdb2c.sh na-ab \
 "$HOME/tmp/haricot-half-duplex/reports/hdb2c-20261018T195644Z-87019-000" \
 --variant cpu
```

The candidate copy lives beside the original runtime at
`runtime-hdb2c-na-v0.6.3`; compilation artifacts are in the candidate itself.
The code fails closed if the preprocessor is not `NA`, the source has changed,
or the candidate was edited after creation. It writes `na_ab_cpu.json`,
`c2c3_na_cpu.json`, and a candidate directory in the same report.

If CPU numeric parity and transcription improve, re-run with
`--variant cuda-cudnn` to ensure Pascal `sm_61` parity with the same source.
Do **not** replace the production ASR executable or promote this candidate
until the source and numeric evidence are reviewed. C4 TDT remains unqualified.

Other potential NeMo differences remain uncorrected here: periodic Hann
vs symmetric Hann, zero placement in 512-point STFT for a 400-sample window,
continuous-Hz librosa Slaney filters vs rounded-bin filters, and logarithm
guard. Follow up by first divergence after the NA ablation; no simultaneous
multi-change optimizations.
