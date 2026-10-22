# R9 text correction — completed CPU measurement

Source: `a19c13257582b06b49c1c7784810952cbab31d21`.
[CPU run 39602129424](https://github.com/haricot/candle/actions/runs/39602129424)
completed all 14 real CaFE/Oreau clips without execution errors. The ASR job
correctly fails the unchanged internal numerical qualification gates.

Artifact: `half-duplex-real-cpu-cafe-reports`, ID `12348401210`, 653582 bytes.
ZIP SHA256: `9f92f6495f54b2406b9ff5280f98a398cf8871a193c902283266d1b52e2bd7ca`.
The artifact contains independent R8B baseline and R9 per-clip reports.

| Check | Observed result |
| --- | --- |
| Actual NeMo final text, native and shared-Mel | exact, 14/14 clips |
| TDT IDs, frames, durations, blank/advance decisions | exact, all 283 decisions |
| Predictor states and raw logits numerical gates | pass, 14/14 |
| R8B/R9 mathematical and non-text result byte controls | pass, 28/28 executions |
| All 11 R8B/R9 frontend primitive byte controls | pass, 14/14 clips |
| Valid final encoder numerical gate | pass, 14/14; worst max_abs 9.238719940185547e-7 |
| Common-input final subsampling linear numerical gate | fail, 6/14; worst 0.002197265625 |
| Complete internal-stage numerical qualification | not established |
| Global FP32 bit equality against NeMo | fail, 14/14 |

The two R8B text mismatches are corrected solely by using the original
SentencePiece unknown-token surface. The mathematical controls verify that
this change does not affect any observed frontend, encoder, predictor, joint
or decision tensor. It does not repair the remaining numerical discrepancies.

The remaining native failures include the initial convolutions/projection,
their scaled error entering block 0 (worst 0.24609375), block 0 internals and
one layer-12 output (Oreau surprise, 0.001312255859375). Shared-Mel still fails
the projection on 12/14 and scaled/block-0 internal gates on all 14 clips.
Thus frontend changes alone cannot be assumed to solve all CPU gates.

Python contracts, C2 Rust checks and real corpus preparation pass. No new GPU
run or microphone measurement follows this result: CPU numerical qualification
must be completed first, per the user's instruction.
