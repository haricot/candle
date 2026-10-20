# HDB2-C3-A-R5 — Positional Encoding Exact Parity

Run the existing `.github/workflows/half-duplex-cpu.yml` on `half_duplex`,
including `real-cpu-asr` via the `[real-asr]` commit marker. Use the same public
CaFE neutral/sadness utterances and original TranscribePsy checkpoint as R4.

R4 run 39429711430 completed, but **did not establish numerical parity**:
positional `max_abs=1`, cosine 0.94806/0.94839; block-0 input already diverged.
R4's 26-stage instrumentation self-control passed on both utterances.

## R5 reference correction

NeMo 2.7.3's ASR transcription teardown first restores the parent training
state, then unfreezes the encoder/decoder/joint. Therefore parent `.training`
and `torch.inference_mode()` alone cannot prove dropout is inactive.

Record the actual module tree immediately after transcription, recursively
call `model.eval()`, and require zero training modules before **both** fresh C1
probes and the C3 reference forward. Old C1 probes without this proof are
rejected. Compare C1/C3 valid-frame outputs again. Runtime evidence must confirm
whether training children were active; source inspection is not that measurement.

## Native positional experiment

- Keep R3C masks, R4 tracing and all weights unchanged.
- Separate R5 source copy; change only frequency construction from f64
  multiply/exp then f32 cast to NeMo's f32 multiply then f32 exp.
- Keep native Rust sin/cos and scaling; never inject a NeMo positional table
  or activation into the full encoder.
- Compare the genuine NeMo class and actual Candle class on 17 common-input
  operator cases: live input plus lengths 1/2/7/513, both input scaling modes,
  and cache offsets 0/3. Exercise table growth from initial length 2.
- Check exact consumed input bytes, relative-position shapes, output scaling,
  numerical error, and f32 byte equality (including signed zero).
- Run the complete R5 encoder on the original shared NeMo Mel, verifying PCM,
  checkpoint, lengths, masking, unchanged R4/R5 pre-encoder, all 26 encoder
  stages and 14 block-0 stages. Operator parity does not promote encoder/TDT.

## Deliverables and interpretation

Add `c3_positional_0.json` and `c3_positional_1.json` to the existing
`half-duplex-real-cpu-cafe-reports` artifact. Upload JSON summaries only.

`POSITIONAL_NUMERIC_PARITY` and `PASS_BITWISE` are independent gates. The
workflow may complete a valid diagnostic while exact parity remains
`NOT_ESTABLISHED`. Record full encoder, TDT and GPU qualification separately.

Sources used to audit the pinned implementations:

- https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/parts/mixins/transcription.py
- https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/parts/submodules/multi_head_attention.py
- https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/modules/conformer_encoder.py
- https://github.com/n0madic/parakeet-candle/blob/53c0d3065f9dd8de0356b25391e60ed3a471192b/src/parakeet/attention.rs

Local validation: 6 new numerical/source-integrity tests and 19 existing C3
tests pass. R4/R5 transforms also pass against the actual pinned upstream
source. Rust compilation, Torch module tests and real NeMo inference are
delegated to the existing GitHub Actions jobs; no local qualification claimed.
