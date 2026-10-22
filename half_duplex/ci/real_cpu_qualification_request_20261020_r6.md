# HDB2-C3-A-R6 — Encoder Padding Mask Parity

Run the existing `half-duplex-cpu.yml` workflow on `half_duplex`, with the
`[real-asr]` push marker. This extends the verified R5 CPU experiment on the
same two public CaFE clips and the genuine TranscribePsy checkpoint.

## Question

R5 final encoder output was numerically aligned for clip 0 (50 valid / 50
total frames, max_abs=3.8743e-7), while clip 1 (51 / 52 frames) remained
different (max_abs=0.22318, cosine=0.97509). Does omitting the final encoder's
padding masks cause that remaining discrepancy?

Pinned NeMo 2.7.3 code applies two relevant masks absent from the native R5
runtime: (1) blocked attention keys AND queries, with masked scores replaced
by -10000 and masked softmax probabilities set to zero; (2) padded convolution
input zeroed after GLU, before depthwise convolution. R6 implements those
operations in a new isolated native runtime copy; R5 positional math, weights,
pre-encoder, activation thresholds and original R5 control stay unchanged.

## Four real native forward passes per clip

| Mode | Attention padding | Convolution padding |
|---|---|---|
| none | off | off |
| attention | on | off |
| convolution | off | on |
| both | on | on |

There is one R6 executable with an explicit experimental mode per process.
Its source and entire Cargo target directory are separate from R5. Its build
must leave the R5 control executable SHA256 unchanged. Three native Rust unit
tests exercise mask construction, masked softmax (including fully blocked
queries, huge blocked scores and head broadcasting), and post-GLU zeroing.

## Required controls and evidence

- Fresh recursively eval-mode C1/C3 NeMo references, exact reference self-check.
- Observe actual boolean masks passed to NeMo layer 0 and its actual
  post-mask depthwise-convolution input; no reconstruction of that activation.
- Verify native padding and attention mask bits against those real NeMo
  masks, including the full padded frame and all blocked query rows.
- Native mask-off must reproduce **all frames** of all 26 R5 encoder stages
  bit-for-bit. For the fully valid clip, all four modes must reproduce R5.
- All modes consume identical NeMo Mel bytes and exact valid lengths. No
  internal encoder activations or positional tables are injected.
- Pre-encoder, positional output and positional table remain bitwise equal
  to the R5 control across all modes.
- Compare the full 26-stage trace, 14 block-0 stages, final output metrics,
  and padded convolution values before/after masking.

Only the known offline, regular global `rel_pos` architecture is qualified;
unexpected context style, cache use, mask semantics or source anchors stop
explicitly. This is not a streaming/cache qualification.

Expected extra artifact reports: `c3_padding_0.json`, `c3_padding_1.json`, in
the existing JSON-only `half-duplex-real-cpu-cafe-reports` artifact.

Keep final-output numerical parity, 24-block numerical parity, all-stage
numerical parity and whole-encoder bitwise equality separate. R5's small
pre-encoder absolute errors and non-bitwise positional table remain known
limits. TDT, GPU and live microphone quality are outside R6.

Local checks: 6 new R6 gate tests + 19 C3 + 7 R5 tests pass. Both strict source
transforms pass against the genuine pinned runtime. Torch hook tests, native
Rust tests, compilation and real inference are delegated to GitHub Actions.

Primary sources:
- https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/parts/submodules/conformer_modules.py
- https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/parts/submodules/multi_head_attention.py
- https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/modules/conformer_encoder.py
- https://github.com/n0madic/parakeet-candle/tree/53c0d3065f9dd8de0356b25391e60ed3a471192b
