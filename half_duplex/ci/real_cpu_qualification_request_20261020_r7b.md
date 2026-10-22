# R7B — materialize NeMo's missing padding default

Full R7 succeeded in run 39456066299 at source d98e8120c8f8371576c37de7c52c2619e57b56c4.
Actual production TDT agrees on all 16/24 decisions, tokens, timestamps,
durations and text in both autonomous PCM and shared-Mel runs. Predictor,
states and raw logits pass the predeclared numerical gates. Strict global
f32 equality fails, including an earlier Mel shape mismatch: 395/404 native
frames versus 400/416 NeMo frames. PCM samples agree bitwise.

The checkpoint omits `pad_to`. Official NeMo 2.7.3
`AudioToMelSpectrogramPreprocessor.__init__` defaults to 16:
https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/modules/audio_preprocessing.py
The pinned native `PreprocessArgs.pad_to` uses serde's zero default. Export
the omitted NeMo default as explicit `pad_to=16`; preserve every explicit
nonnegative integer, including zero. Invalid unsupported values fail.

Requalify through the same workflow with `[real-asr] [r7-focused]`. Before
compiling, the focused profile verifies six native source hashes against
the successful full R7 artifact and the exact checkpoint SHA. It compiles
separate R6/R7 targets and runs a real R6 both-mask control per clip, actual
NeMo references and all R7 common-input/autonomous executions. Historical
C2/R3/R4/R5 and four-mode R6 native ablations are not repeated in this
config-only profile. No historical or synthetic activation is used as a
new inference result. Normal `[real-asr]` still executes the full chain.

Record actual C1/fresh NeMo frontend padding state. Require matching Mel
shapes after the fix and TDT numerical/discrete parity. Continue to report
strict f32 differences independently; shape equality does not imply byte
equality. The zero-duration blank/max-symbol edge cases remain unit-tested;
neither was triggered by the two real R7 clips.
