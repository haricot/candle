# HDB2-C v0.6.4 — Rust E0596 probe repair, isolated NA A/B

## Observed on target (2026-10-18)

- Original NeMo C1 vs Candle shows important ASR difference; numerical parity unqualified.
- NeMo C1 `--probes` correctly emits reference tensors in
  `hdb2c-20261018T195644Z-87019-000`.
- `probe` and `na-ab` compilation failed with Rust `E0596`: `.encoder.forward(&mel, None)` requires `&mut self`, but the `match` used `&model`.

## Isolated source-level correction

`rust/hdb2c_probe.rs`: `let mut model=...; match &mut model {...}`.
The probe is additional tooling only; the **production ASR binary, Candle core, weights and NeMo environment are unchanged**. `hdb2c/probe.py` only replaces the source of the known buggy v0.6.3 probe based on SHA256; any unknown user modifications cause a hard refusal. No blind source overwrite. `na-ab` creates a *new* `runtime-hdb2c-na-v0.6.4` copy, preserving `runtime-hdb2c-na-v0.6.3` and all existing logs.

## Run

Extract the full ZIP into `~/tmp` and export existing HDB2A runtime, model and reports. No Python installation is needed.

```bash
REPORT="$HOME/tmp/haricot-half-duplex/reports/hdb2c-20261018T195644Z-87019-000"
./run-hdb2c.sh selftest
./run-hdb2c.sh probe "$REPORT" --variant cpu
./run-hdb2c.sh na-ab "$REPORT" --variant cpu
```

`probe` first is baseline, `na-ab` second is isolated change `normalize="NA"` only; outputs `c2c3_cpu.json`, `na_ab_cpu.json`, and `c2c3_na_cpu.json` in the same C1 report. Do not rerun C1 or `install-nemo` unless new input evidence is needed. C4 TDT is not qualified. If the Rust build still fails, share exact stderr for the new attempt; this ZIP was not compiled against the real checkout or CUDA in the release environment.

## Reference

Rust official E0596 documentation: https://doc.rust-lang.org/error_codes/E0596.html
Source model `Conformer::forward(&mut self,...)`: https://github.com/n0madic/parakeet-candle
