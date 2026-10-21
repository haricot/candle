# R9 — original SentencePiece unknown-token rendering

R8B identified the two final-text mismatches: actual emitted unknown token ID 0
uses the original SentencePiece surface ` ⁇ `, while the native vocabulary join
emits `<unk>`. This isolated candidate changes TDT token text and returned text
using metadata read from the original `.nemo` asset. It does not modify weights,
frontend, encoder, predictor, joint, decisions, frames, durations or thresholds.

The implementation accepts only the verified SentencePiece subset: normal,
unknown and user-defined pieces, original dummy-prefix settings, no control or
byte pieces, no denormalizer. It verifies the entire converted vocabulary and
reads the unknown ID/surface instead of guessing them. Unsupported semantics
fail before inference.

The complete R8B campaign remains the independent baseline. R9 runs both native
PCM and shared-Mel paths and requires byte-identical PCM, all eleven frontend
primitive tensors, every encoder/operand trace and every TDT activation and
decision. Only returned `text` is compared separately in this additional
R8B/R9 control; the original R7/R8B result-byte control stays strict. R9 text
must then match actual NeMo independently. All original neural numerical and
global bitwise gates remain in force, so fixing text alone cannot qualify the
whole pipeline.

On an existing prepared environment, from `half_duplex`:

```bash
"$HDB2C_PYTHON" -m hdb2c.c3_r9_text \
  --corpus /absolute/path/to/cameo/manifest.json --require-numeric
```

CI uses dispatch input `r9_text=true`, or commit markers `[real-asr]`,
`[r8-corpus]` and `[r9-text]`. All 14 public clips are required; raw tensors,
token sequences and transcripts are deleted per clip. Artifacts include the
actual R8B baseline reports and separate `r9-text-cpu-*.json` results. A failed
strict qualification produces a failing ASR job even when all measurements
complete. This is an isolated candidate; no production promotion, microphone,
streaming, beam-search or GPU qualification is inferred from the CPU campaign.

Local validation: 161 Python tests, 12 native Rust tests, R9 probe compilation,
source generation against the qualified R7 chain and metadata from the original
tokenizer SHA256 `eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e`.
Real-model results are pending at implementation time.
