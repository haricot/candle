# HDB2-C3-A-R4 — real CPU GitHub Actions request (2026-10-20)

Requested from remote: execute the existing `half-duplex-cpu.yml` pipeline
on branch `half_duplex`, including its guarded `real-cpu-asr` job.

Trigger: `[real-asr]` in this commit message, which is checked by the
workflow's push-event expression. No new workflow or separate branch.

## Experiment and scope

- Use the public CaFE/Oréau CAMEO corpus (selected neutral + sadness utterances),
  not any private microphone recordings.
- Restore the authentic QVAC TranscribePsy-250M-EU-sub NeMo checkpoint.
- Compare real CPU NeMo and Candle on identical PCM and identical injected
  NeMo log-Mel input; enforce checkpoint/hash, shapes, valid lengths and
  actual consumed-Mel checks.
- Execute C2, C3 R3, R3C and existing R4 block-0 internal probe.
- Preserve 26-stage R3C-vs-R4 instrumentation consistency and report the
  first differing internal stage; do not promote diagnostic localization to
  numeric or bitwise parity.
- Do not modify GPU/CUDA, TDT inference, production runtime or model weights.

## Output / qualification gate

Expected artifacts from existing workflow:
`half-duplex-real-cpu-cafe-reports` including `c2_ab_0/1.json`,
`c2c3_0/1.json`, `c3_layers_0/1.json`, `c3_dw_0/1.json`,
`c3_mask_ab_0/1.json` and `c3_block0_0/1.json`.

Only JSON summaries can be uploaded. Private recordings, NeMo model weights,
raw activations and local probe traces must never be included as artifacts.

Run status remains NOT_VERIFIED until actual GitHub Actions job results and
artifacts have been inspected. A green workflow alone is not a claim of
full FastConformer or TDT numeric parity.
