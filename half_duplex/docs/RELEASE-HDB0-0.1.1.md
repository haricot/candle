# HDB0 v0.1.1 — WAV fixture regression correction

## User-reported results on v0.1.0

`cargo test --workspace --locked --offline` on Arch / Rust 1.99: 13 passing tests and one failing test, `half_duplex_wav::tests::float32_and_rejection`.

Expected `UnsupportedFormat`, received `InvalidHeader`.

## Root cause

The 24-bit PCM test fixture contained an odd-length `data` chunk of three bytes and the test builder failed to add the RIFF mandatory pad byte. The total RIFF size was likewise one byte too small. The WAV parser correctly treated this as invalid; it had not reached the unsupported-codec check.

## Correction

`make_wav()` now emits a pad byte for odd chunk sizes and includes it in the RIFF container size. The `data` chunk size remains the true sample-data length. The test asserts that well-formed PCM24 returns `UnsupportedFormat` while removing the pad makes RIFF truncated.

Preflight distinguishes the recorded Candle reference commit from a Cargo dependency. HDB0 depends on neither Candle nor Mistral.rs. A local HEAD different from the reference is reported as `CANDLE_LOCAL_MATCH=NO_NON_BLOCKING`, not a failure.

## Test on the user's machine

```bash
./run-once.sh preflight
./run-once.sh check
./run-once.sh selftest
```

The release generation container did not have `rustc` or `cargo`; the Rust test suite could not be rerun there. Independent RIFF-length and padding checks, shell syntax, and preflight mismatch behavior were exercised. This is a complete source ZIP, not a patch.
