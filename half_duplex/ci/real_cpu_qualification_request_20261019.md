# Real CPU NeMo ↔ Candle C2 — 2026-10-19

Requested execution of the real-model evaluation on the public French CaFE subset of CAMEO. This intentionally activates `real-cpu-asr` via the `[real-asr]` commit flag. Uses the original QVAC TranscribePsy-250M-EU-sub `.nemo` checkpoint, its converted Candle weights, the same PCM16 inputs, and the C2 frontend. The run must record actual transcripts and tensor RMSE; a passing build alone is never evidence of ASR fidelity. No personal microphone recordings are used, uploaded or committed.

Success criteria: real NeMo and Candle CPU inference on the same two CaFE WAVs (neutral, sadness); original and C2 transcript comparison; log-Mel and FastConformer numeric metrics; saved JSON artifacts. The CI run should fail closed for missing model, missing probes, compiler errors or dependency problems.
