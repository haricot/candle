# TP3 — QVAC official model, one public French clip

Narrow parity harness for Candle's standalone TranscribePsy branch. The
model code in candle-transformers/src/models/transcribepsy/ remains free of
downloads, NeMo, tokenizers, WAV readers, probes and machine-specific changes.

Official unchanged model artifacts (pinned at
ccf70de9ff9848f5cd5975adda1370ce4addf6f1):

- model.safetensors — SHA-256 481fec004f2074909a5d253cb674fe676a49ce856bca2ce0156136ea920c3168.
- TranscribePsy-250M-EU-sub.nemo — SHA-256
  0acdaf1e1ebf09094400263006d134d33b0ebb871ebb064253cfd3a7c528903c.

The original NeMo .nemo supplies the authoritative model config and the
*original* SentencePiece tokenizer. All float tensor weights are checked
byte-for-byte against the official Safetensors before inferring anything.
No converted checkpoint or generated model variant is allowed.

The sole clip is selected deterministically from
amu-cai/CAMEO revision c7f2fc9f0538ca871d38e183ba8ad6c329120ecc,
French CaFE split, cafe_12 neutral, smallest file_id. The CAMEO
collection/CaFE clip is licensed CC BY-NC-SA 4.0; it is downloaded and
decoded locally and **not committed or uploaded**.

The main comparison uses **one real, unmodified NeMo forward** and **one
independent Candle forward** from the same PCM:

1. F32 log-Mel output valid frames, shape, max absolute error, RMSE, cosine;
2. final 24-layer FastConformer output (same numerical gates);
3. full production NeMo TDT decisions (frame/token/duration/advance/forced);
4. token IDs and exact text via the original QVAC SentencePiece asset.

Numerical limits for each tensor: max_abs <= 1e-3 and
cosine >= 0.999. TDT decisions, token sequence and decoded text require
*exact* equality. Failures must be reported as REJECT_OR_NOT_QUALIFIED,
not hidden by instrumentation or aggregate WER.

## TP3-R1 — independent stages

The Rust driver also accepts `--reference <NeMo-output-directory>` to run two
separate handoffs after the independent forward:

- NeMo Mel -> Candle FastConformer -> Candle TDT;
- NeMo encoder -> Candle TDT.

`compare.py --require-isolation` requires these diagnostics **in addition to**
the independent PCM comparison. An injected tensor can never replace a failing
independent forward in the acceptance gate. Trace inputs are labeled explicitly;
the WAV SHA-256 binds the NeMo reference to the selected public fixture.

The first failure was caused by missing implicit NeMo preprocessing defaults
(`preemph=0.97`, `pad_to=16`), incorrect ReLU placement before the subsampler's
pointwise convolutions, and missing mask propagation. The original checkpoint's
window and filterbank remain unchanged; the centered constant-padding STFT and
normalization formulas did not require adjustment for this `normalize=NA` model.
The config exporter now materializes the actual reference frontend settings.
Dither remains inactive in evaluation, as in NeMo.

Local TP3-R1 result, same 55,649-sample excerpt and 913 identical F32 tensors:

| Comparison | Max absolute error | Outcome |
| --- | ---: | --- |
| Independent Mel, 347 valid / 352 padded frames | 0.000140190125 | PASS |
| Independent encoder, `[1,44,640]` | 0.000000357628 | PASS |
| Encoder given NeMo Mel | 0.000000357628 | PASS |
| TDT, independent and both handoffs | 16/16 decisions exact | PASS |
| Original tokenizer text, all three paths | `Un cheval fou dans mon jardin.` | Exact |

These results use the existing thresholds without relaxation. Bitwise tensor
equality remains a diagnostic and is false. This qualifies only the selected
utterance on CPU F32; it makes no CUDA, streaming or general-corpus claim.

## GitHub

.github/workflows/transcribepsy-tp3.yml compiles and tests the CPU
driver on each TP3 code change. The approximately 2-GB official-model
validation is manual (workflow_dispatch), to avoid repeated large downloads
on every commit. It uses a temporary folder and uploads **only**
tp3-public-report.json (a small public-source summary). No WAV,
weights, tensor dumps or local paths enter the repository or artifact.

## Local requirements

A Python 3.12 environment with NeMo ASR 2.7.3, PyTorch CPU,
safetensors, pyarrow, sentencepiece, numpy, huggingface_hub;
ffmpeg; Rust 1.99 and Candle dependencies.

The workflow documents exact download commands and invocation order:
fixture.py, reference.py, Rust example transcribepsy_tp3,
compare.py. Include `--reference` on the Rust command and
`--require-isolation` on comparison for TP3-R1. A run is qualified only when
the real report passes the independent and isolated-stage gates.

This is an isolated one-utterance CPU/F32 qualification, **not** WER,
streaming, CUDA, mobile or generalized-corpus qualification.
