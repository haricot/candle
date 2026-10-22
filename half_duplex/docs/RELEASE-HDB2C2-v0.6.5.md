# HDB2-C2 v0.6.5 — NeMo-exact audio frontend: isolated Rust A/B

## Objective

Test the hypothesis that the unreliable Candle transcript on the user's soft
French Bluetooth speech is caused by `parakeet-candle`'s audio frontend. Preserve
NeMo checkpoint, asr runtime, Pascal CUDA, HDB1, and the original WAV, unchanged.

The original NeMo C1 reference (same PCM, SHA256 `a075850b69b66591ca8611cdc29750c33e53a9f59f23910acf23897fbb526f7e`)
correctly recognized almost the whole utterance, unlike Candle.

## Independently verified NeMo frontend formula

On this real 16 kHz 216,378-sample WAV, an **independently run PyTorch CPU**
implementation using `torch.stft`, `torch.hann_window(periodic=False)`,
`librosa.filters.mel(norm="slaney")`, and `torch.log(x+2**-24)` compared to
NeMo's *real saved `nemo_mel.npz` tensor*, for its 1,352 valid frames:

- RMSE 1.3690088e-6
- MAE 4.218403e-7
- Max absolute difference 5.531311e-5
- Cosine approximately 0.9999999

The reference cross-check was executed in Python, NOT in the new Rust runtime.
It does not qualify the Rust port until the real probe runs.

## Rust frontend changes

`rust/hdb2c_nemo.rs` is complete, independently authored Rust source. It uses the
same pinned library FFT implementation (`audio::fft`), not a new framework CUDA
kernel. The isolated candidate implements:

- Resolved preemphasis with first sample unchanged;
- Hann symmetric (`periodic=False`) with 400 taps centered in a 512-point FFT;
- Zero waveform padding (not reflection), unnormalized FFT;
- 128 full, continuous-frequency Slaney mel triangles (no bin quantization);
- Default log guard `2**-24` and `add` vs `clamp`;
- `normalize="NA"` no normalization; per_feature/all_features supported with
  unbiased variance for testing;
- STFT raw length 1,353, **valid length 1,352**, padded Mel 1,360;
- Zeroing invalid frames, 16-frame temporal alignment;
- Explicit propagation of 1,352 **valid frames** into FastConformer, which
  should return 169 valid encoder frames while keeping the full 170-frame tensor.

C2 checks numeric parity on valid frames. C3 checks the encoder only after Mel
parity is qualified. C4/TDT logits are NOT numerically qualified.

## Installation and qualification

No install of PyTorch, NeMo, HF checkpoint, or Rust/Candle fork.

```bash
cd ~/tmp
unzip haricot-half-duplex-hdb2c-c2-standalone-v0.6.5.zip
cd haricot-half-duplex-hdb2c-v0.6.5

export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
export HDB2A_REPORT_ROOT="$HOME/tmp/haricot-half-duplex/reports"

# Phase 1: tests + prepare complete isolated source + CPU cargo build.
./install_update.sh

# Phase 2: real NeMo/Candle A/B, same existing C1 report and WAV.
REPORT="$HDB2A_REPORT_ROOT/hdb2c-20261018T195644Z-87019-000"
./install_update.sh --report "$REPORT"
```

The isolated candidate is created at
`~/tmp/haricot-half-duplex/integration/parakeet-candle/runtime-hdb2c-c2-v0.6.5`.
It has a source manifest plus full, compiled Rust files. The live runtime is
never written. `install_update.sh` is idempotent and refuses a dirty candidate.

Generated report outputs:
- `c2_ab_cpu.json` — original NeMo / baseline Candle / exact-frontend Candle transcription;
- `c2c3_exact_cpu.json` — numerical comparison on valid Mel and encoder frames;
- `candidate_c2_cpu/{asr.*,probe.*,candle_mel.f32le,candle_encoder.f32le,candle_probe.json}`.

After CPU qualification, optional `--cuda-cudnn --report "$REPORT"`
compiles another isolated binary on GTX 1080 Max-Q sm_61. This is an independent
later qualification, not the default action.

### What has not been verified

The archive authoring environment has no `cargo`, `rustc`, CUDA or NVIDIA GPU.
The release validates Python source transforms, compatibility gates, packaging,
and the independent Python NeMo frontend formula only. **It does NOT claim that
Rust compiled or that real Candle ASR improved.** The user's build and real
numeric reports will decide promotion.

### Sources

- NVIDIA-NeMo/Speech `nemo/collections/asr/parts/preprocessing/features.py`
- NVIDIA-NeMo/Speech `nemo/collections/asr/modules/audio_preprocessing.py`
- librosa `filters.mel` Slaney, HTK false
- pinned `n0madic/parakeet-candle` SHA `53c0d3065f9dd8de0356b25391e60ed3a471192b`
- torch STFT centered-window, pad_mode=constant
