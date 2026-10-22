# HDB0 v0.1.2 — WAV 24k microphone normalization

Input evidence: `hdb0-20261017T115.zip`, microphone WAV inspection 24 kHz, 353211 samples, 14.717125 s. Original v0.1.1 simulation failed at its explicit 16 kHz requirement.

Correction: `half_duplex_core::to_16khz` provides a standalone, zero-dependency **offline** Blackman-windowed sinc resampler with 32 samples on either side, anti-alias cutoff for decimation, input-rate guard 8–192 kHz and rounded output length. `simulate-turn` records original and normalized clocks in JSON.

Qualification: ZIP integrity, Bash syntax, fixture readability and source sanity can be checked at packaging time; Rust/Cargo compilation and all tests require user's Rust machine. Do not report a successful Rust test before it runs.

```bash
./run-once.sh check
./run-once.sh simulate /home/np/test_micro_2.wav
```

Expected normalized 235474 mono samples at 16k; synthetic-only turn event logic; no VAD/ASR.
