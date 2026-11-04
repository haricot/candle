# Candle Extended executable

The Extended campaign builds one native release application, `candle-extended`.
Its model entrypoints are linked into that executable and run in the same
process. It does not unpack or launch the campaign's test executables.

The application is installed before the candidate is committed. Its source
receipt, Cargo.lock, candidate SHA, SDK digest, CPU profile and binary SHA-256
are recorded in the immutable build manifest. The final application is linked
once. Hosted CPU checks and the physical Nemotron and PocketTTS gates execute
those same bytes; neither executor invokes Cargo to rebuild the application.

The user-facing artifact is `candle-extended-<campaign>-<cpu_profile>` and contains:

- `candle-extended`: the single native ELF program;
- `candle-extended.sha256`: its checksum;
- `release.json`: provenance and runtime requirements.

The default target is `-C target-cpu=x86-64-v3` (AVX2/FMA/F16C), not the hosted
machine's `native` CPU. The CUDA target is SM61. CUDA and cuDNN shared runtime
libraries are required. After downloading and extracting the Actions artifact:

```sh
sha256sum -c candle-extended.sha256
chmod +x candle-extended
./candle-extended --help
./candle-extended capabilities
./candle-extended nemotron-asr --help
./candle-extended qwen3 --help
```

The artifact becomes available after hosted CPU validation. Physical CUDA
qualification and branch promotion retain their existing gates. The metadata
does not claim that GPU validation has happened before the GPU job finishes.

`pockettts-check` is an operator check, not full speech synthesis. FireRedVAD
and Smart Turn accept feature tensors already prepared for their native models.
Model files remain external. The application adds no model downloads to CI.

Internal SHA-addressed test executables remain in the separate build-once
proof artifact. They are not the downloadable Candle Extended application.
