# HDB2-C v0.6.0 — audit NeMo/Candle (NOUVEAU)

**Lire d’abord [README-HDB2C.md](README-HDB2C.md)** pour lancer C1 sans recompiler, puis C2/C3 avec un nouveau binaire de sondage. La campagne ne déclare aucune parité tant que les modèles réels ne sont pas exécutés.

---

# HDB1-R1 v0.5.1 — Annulable endpointing / FireRedVAD + Smart Turn + Candle

**Important change:** `speech_end` no longer completes immediately. HDB1
waits 1200ms of continuous post-VAD silence (`--endpoint-hold-ms 1200`) before
calling the real Smart Turn ONNX; resumed speech cancels the candidate and
keeps audio intact. The mic capture is restarted after every cold ASR turn,
so stale queued PCM is discarded. This is still **blocking half-duplex**,
not a resident ASR worker, and will drop speech during the cold ASR interval.

**Verified locally:** 15 HDB1 software tests, including replay of the user's
observed 3.4s / 8.1s / 11.4s candidates, plus inherited HDB2 tests.
**Not yet verified locally:** actual PyTorch-free ONNX mic inference or Pascal
GPU runtime for this modified release. Prior v0.5.0 real model successes do
not establish v0.5.1 runtime quality.

```bash
cd ~/tmp
unzip haricot-half-duplex-hdb1-r1-standalone-v0.5.1.zip
cd haricot-half-duplex-hdb1-r1-v0.5.1
export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
export HDB2A_REPORT_ROOT="$HOME/tmp/haricot-half-duplex/reports"
./run-hdb1.sh selftest
./run-hdb1.sh mic --max-turns 3 --variant cuda-cudnn --endpoint-hold-ms 1200
```

The pre-existing model downloads in `~/tmp/haricot-half-duplex/models/hdb1`
are reused. See `docs/RELEASE-HDB1-R1-v0.5.1.md` for the defects corrected,
precise acceptance limits, and instrumentation.

---

# HDB1 v0.5.0 — Preserved baseline (historical documentation)

**Qualification status:** software control tests pass; **real end-to-end ONNX + mic inference not tested in this ZIP-building environment**. The preceding HDB2-A CPU and HDB2-B CUDA+cuDNN runs were successful on the user's computer. HDB2-B2 can be deferred; its tools are preserved here.

## Architecture and boundaries

```
WAV / PipeWire pw-record (s16, 16 kHz mono)
  -> 100-ms PCM chunks
  -> FireRedVAD Stream ONNX (Kaldi FBank + CMVN; 8x128x19 model cache)
  -> speech_start / speech_end candidate
  -> Smart Turn v3.2 INT8 ONNX (Whisper-feature extraction; last 8 sec)
  -> if complete: finalize buffered WAV (pre-roll + active speech)
  -> existing haricot/Candle FastConformer/TDT binary (TranscribePsy-250M-EU-sub)
  -> transcription.txt, turn-report.json, events.jsonl, report.json
```

- FireRedVAD uses the **real ONNX inference**, not a dummy RMS threshold. ONNX inference defaults to CPU; the **Candle ASR can remain CUDA+cuDNN on Pascal**.
- Smart Turn is invoked **only after** a VAD speech-end candidate. A low probability retains the audio and continues the same turn. After 3.2 s of silence with no resumption, force-finish with `HESITATION_TIMEOUT` (never report as semantic approval). On file EOF, likewise mark `EOF_FORCED_UNCONFIRMED` or `EOF_WHILE_SPEAKING`.
- `--no-asr` is a strictly labelled endpoint-only diagnostic and does **not** claim any ASR result.
- Important: the **Candle binary is launched as a fresh process per completed turn**. The measured HDB2-B `324 ms` warm inference does **not** equal HDB1 mouth-to-text latency with this initial architecture. A persistent ASR worker and microphone pausing would be later work.
- This is deliberately a **Python ONNX adapter / controller and Rust/Candle ASR hybrid**, **not a pure Rust ONNX implementation**. It does not require compiling HDB2-B2 first.
- Stores recorded voice clips **locally** inside the chosen `reports` directory. Delete these clips manually if not needed. No audio upload by the HDB1 pipeline; model bootstrap downloads the published model weights.

## First-time installation on Arch Linux

```bash
cd ~/tmp
unzip haricot-half-duplex-hdb1-standalone-v0.5.0.zip
cd haricot-half-duplex-hdb1-v0.5.0

# Reuse already compiled HDB2-B CUDA+cuDNN binary and the converted QVAC checkpoint.
export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
export HDB2A_REPORT_ROOT="$HOME/tmp/haricot-half-duplex/reports"

./run-hdb1.sh selftest
./run-hdb1.sh install-python
./run-hdb1.sh download-models
./run-hdb1.sh preflight

# Offline first: use already recorded WAV (24 kHz input accepted via FFmpeg).
./run-hdb1.sh file /home/np/test_micro_2.wav --variant cuda-cudnn

# Actual PipeWire microphone: detect a SINGLE turn and transcribe.
./run-hdb1.sh mic --max-turns 1 --variant cuda-cudnn

# Optionally process three consecutive turns (micro is restarted after each ASR).
./run-hdb1.sh mic --max-turns 3 --variant cuda-cudnn
```

`pw-record` needs a valid default audio source; `--pw-target DEVICE` can override it. For diagnosing endpoint decisions alone (without any ASR binary): `./run-hdb1.sh file /home/np/test_micro_2.wav --no-asr`. No artificial transcript is ever produced.

If the previous HDB2-B compiled binary is in a non-default location, set `HDB2A_RUNTIME_DIR` accordingly. Required:

`$HDB2A_RUNTIME_DIR/target/hdb2b-cuda-cudnn/release/parakeet-candle`

`$HDB2A_MODEL_DIR/{config.json,conversion.json,model.safetensors}`

The original `.nemo` download is not needed for this step, and there are no changes to `haricot/candle` or `mistral.rs`.

## Model provenance

FireRedVAD ONNX models / CMVN: `https://github.com/eschmidbauer/fireredvad-onnx` at pinned revision `faa757dafa0058d0b231ec82e084ec910a3a46d5`, based on [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD). The bootstrap installs the streamed model with caches and `cmvn.ark`. FireRedVAD is Apache-2.0; the independent integration source is attributed to its public streaming reference (MIT).

Smart Turn v3.2 INT8 ONNX model: `pipecat-ai/smart-turn-v3`, pinned HF revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`; mandatory SHA256 `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`; BSD-2-Clause. It expects 16 kHz mono, 8-second window padded at the start; model score is the **probability of a completed turn**, not a speech probability.

A separate CPU ONNX environment avoids PyTorch CUDA binary dependencies on sm_61. In the event of a missing ONNX library, no fallback VAD is substituted.

## HDB1 acceptance criteria

1. `selftest`: 9 software tests (mock endpoints and mocked Candle output); **NOT** real ONNX inference.
2. `preflight`: actual model files present and Smart Turn digest matches pinned release.
3. `file`: `events.jsonl` captures actual FireRedVAD speech boundaries, each Smart Turn probability, and either a Smart Turn confirmation or an explicit forced decision.
4. A positive turn outputs `turn-000/turn.wav` and `transcription.txt` **only when the genuine Candle binary returns nonempty text**.
5. `mic`: PipeWire captures PCM16 mono; turns can be detected from an actual microphone, and no speech means no transcript.
6. Validated end-to-end times and false-positive rate on user recordings remain outstanding. Compare pauses, question intonation, hesitation, noise, short fillers, long speech and overlap before calling endpointing qualified.

## Performance notes

- `run-hdb2b2.sh qualify` and the HDB2-B benchmark remain available unchanged.
- No hot-path latency improvement is claimed: audio capture, FireRed VAD and Smart Turn are separate inferences, and ASR process creation is cold.
- `HDB1_STATUS=COMPLETE` means the driver finished, not necessarily that speech was detected or a transcript obtained. Examine `turns[]` and each `asr_status`.


## HDB2-C v0.6.2 Python compatibility correction

The old installer selected Python 3.14 and backtracked NeMo to 2.5.0, then source-built ONNX with Protobuf 7.36.1-generated code but Protobuf 5.29.6 runtime. Run `./run-hdb2c.sh install-nemo` with `uv` to use separate Python 3.13, and `./run-hdb2c.sh verify-nemo` before C1. No changes to your existing Candle model, HDB1 pipeline or binaries.


### Dépannage installation NeMo — v0.6.2

Le premier installateur v0.6.1 a laissé l'index PyTorch CPU prioritaire
pour les dépendances NeMo, cachant `setuptools>=79` disponible sur PyPI.
La commande `./run-hdb2c.sh install-nemo` de v0.6.2 conserve automatiquement
le Python 3.13 et le PyTorch CPU déjà présents, installe `setuptools` sur
PyPI et résout NeMo uniquement sur PyPI avec une contrainte de roue CPU.
Ne pas utiliser `--index-strategy unsafe-best-match` ni recréer l'environnement.
Exécuter ensuite `./run-hdb2c.sh verify-nemo` et, en cas de succès, C1.
Si l'import échoue à nouveau, enregistrer l'erreur sans modifier Candle.

Lancien environnement `.venv-hdb2c-py313` est redécouvert automatiquement entre les archives versionnées.


## HDB2-C v0.6.3 — isolated NA single-change A/B

The confirmed C1 same-PCM experiment showed NeMo recognizes the user's French
while baseline Candle degrades it. In the source code of `parakeet-candle`,
`normalize="NA"` currently runs the global normalization `else` branch.
The NeMo ASR reference implementation regards `NA` as **no normalization**.
This archive adds the `na-ab` candidate: **only** changes `} else {` to
`} else if args.normalize == "all_features" {` at the global normalization
branch inside `src/parakeet/audio.rs::get_logmel`. Other source paths remain identical.

**First** run the baseline numerical probe using the *real* report path:

```bash
./run-hdb2c.sh probe \
 "$HOME/tmp/haricot-half-duplex/reports/hdb2c-20261018T195644Z-87019-000" \
 --variant cpu
```

Then **create a complete isolated Rust runtime copy**, compile and compare candidate
transcripts and real log-Mel/encoder probes, without changing the live binary:

```bash
./run-hdb2c.sh na-ab \
 "$HOME/tmp/haricot-half-duplex/reports/hdb2c-20261018T195644Z-87019-000" \
 --variant cpu
```

The candidate copy lives beside the original runtime at
`runtime-hdb2c-na-v0.6.3`; compilation artifacts are in the candidate itself.
The code fails closed if the preprocessor is not `NA`, the source has changed,
or the candidate was edited after creation. It writes `na_ab_cpu.json`,
`c2c3_na_cpu.json`, and a candidate directory in the same report.

If CPU numeric parity and transcription improve, re-run with
`--variant cuda-cudnn` to ensure Pascal `sm_61` parity with the same source.
Do **not** replace the production ASR executable or promote this candidate
until the source and numeric evidence are reviewed. C4 TDT remains unqualified.

Other potential NeMo differences remain uncorrected here: periodic Hann
vs symmetric Hann, zero placement in 512-point STFT for a 400-sample window,
continuous-Hz librosa Slaney filters vs rounded-bin filters, and logarithm
guard. Follow up by first divergence after the NA ablation; no simultaneous
multi-change optimizations.


## HDB2-C v0.6.4 — Rust probe E0596

Le modèle NeMo original passe C1 et C1 `--probes`. Les sondes Candle échouent à la compilation E0596,
car Conformer::forward exige `&mut self` mais le probe utilisait `match &model`.
La v0.6.4 corrige les DEUX emprunts, revalide le SHA256 de l’ancienne sonde avant substitution,
et isole le candidat A/B dans `runtime-hdb2c-na-v0.6.4`; aucune nouvelle installation NeMo.
Exécuter `probe` puis `na-ab` sur le rapport `hdb2c-20261018T195644Z-87019-000`; détails dans
`docs/RELEASE-HDB2C-v0.6.4.md`. Le correctif Rust reste à compiler sur la machine réelle.

## HDB2-C2 v0.6.6 — Explicit build and update entry point

`./install_update.sh` (tests, isolated source copying, `cargo build --release`
CPU) then `./install_update.sh --report /path/to/C1_PROBES_DIR` (real NeMo/Candle
same-WAV A/B + C2/C3 numeric probes). Original runtime/binaries untouched.
Detailed procedures and limitations: `docs/RELEASE-HDB2C2-v0.6.6.md`.

C2 `install_update.sh` also runs `cargo test --release --lib hdb2c_nemo::tests` on the isolated candidate after building. Compilation and runtime unit tests remain pending on the user machine.
