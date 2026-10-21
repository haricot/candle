# HDB2-C v0.6.0 — STATE-MEMORY / 2026-10-18

Décision : priorité à la parité **NeMo original .nemo vs Candle QVAC**. Le test de voix avec microphone direct fournit un texte imparfait même sans VAD ; +12 dB ne suffit pas. Ne pas imposer une voix plus forte ni modifier HDB1/R1 en parallèle.

C1 : *même* fichier PCM16 16 kHz mono, même sha256 payload audio, NeMo CPU dans `.venv-hdb2c` et Candle CUDA+cuDNN déjà compilé. Vérifier le `source_nemo_sha256` de `conversion.json` contre l'archive `.nemo`. Capturer les deux transcriptions. La distance de leurs sorties **n'est pas** un vrai WER sans référence humaine vérifiée.

C2 et C3 : le binaire additionnel `src/bin/hdb2c_probe.rs` (inclus dans cette archive, installé sans modifier `src/main.rs`) accède directement à `get_logmel` et à l'encodeur TDT Candle. NeMo donne mel/encoder via `model.preprocessor` et `model.encoder`. Comparaison stricte avec formes, valid_frames, NaN, RMSE, cosine et max abs; une divergence C2 invalide l'attribution de C3 à l'encodeur. `C4` logits/jetons non implémenté et explicitement `NOT_RUN`. `C5` toute correction ciblée est bloquée avant preuves.

Aucune nouvelle compilation Rust ni NeMo réelle n'a été testée au moment de l'archive. Code de sondage à qualifier. Modèles `.nemo` et safetensors externes non inclus. `haricot/candle` PIN inchangé, GTX 1080 Max-Q sm61 inchangée. Protocole `docs/RELEASE-HDB2C-v0.6.0.md`; commandes `README-HDB2C.md`.

---

# HDB1-R1 v0.5.1 — State memory / 2026-10-18

Most recent standalone is HDB1-R1 v0.5.1. Retain pinned TranscribePsy
checkpoint, Candle extended, Pascal sm_61, FireRedVAD ONNX and Smart Turn
v3.2 ONNX. No model weights included. No patches required to precompiled
Rust binaries for this endpoint policy change.

Evidence from user's 2026-10-18 mic test: three VAD candidate ends at
3.4 / 8.1 / 11.4s all SmartTurn-accepted (0.98232 / 0.91398 / 0.988007)
within the FIRST intended spoken intervention. `--max-turns 3` stopped
capture after this first intervention and the short transcriptions were
low-quality. Model runs succeeded: architecture/endpointing bug, not CUDA
failure.

Decision: HDB1-R1 defers Smart Turn until 1200 ms of uninterrupted silence
after the VAD candidate; new speech cancels the candidate without losing
buffered PCM. Await successful SmartTurn decision before ASR; forced timeout
and EOF retain explicit reasons. Fix mic capture restart and clear old pre-roll.

**Known limits**: no native ONNX Rust runtime yet; Candle ASR is cold per turn
and blocks microphone consumption. B2 performance campaign remains deferred;
CPU/CUDA/CUDA+cuDNN HDB2-B validation exists. HDB1-R1 tests are synthetic
control-flow only; real endpoint fidelity **NOT QUALIFIED** until user's test.

Commands: see README.md / docs/RELEASE-HDB1-R1-v0.5.1.md.

---

# STATE-MEMORY — Haricot Half-Duplex v0.4.0 HDB2-B2

## Architecture et historique conservés

- Projet Rust standalone, source complète, modèles et runtimes préexistants réutilisés.
- HDB0 : WAV mono/rééchantillonnage, ring buffer, simulateur sans ASR.
- HDB2-A : véritable TranscribePsy-250M-EU-sub FastConformer/TDT avec
  Candle, non synthétique, sur CPU ; `test_micro_2.wav` (24k) → PCM16 16k.
- Runtime `n0madic/parakeet-candle` à `53c0d3065f9dd8de0356b25391e60ed3a471192b`,
  adaptation locale du chargeur QVAC et de `layer_norm`.
- `haricot/candle` épinglé à `f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3`.
- GTX 1080 Max-Q Pascal sm_61, CUDA 12.9 / cuDNN 9 ; pas de FP8/FP4 natif.
- HDB2-B v0.3.1 : CPU FP32 6402,94 ms, CUDA FP32 727,87 ms,
  CUDA+cuDNN FP32 324,294 ms médians à chaud sur 14,717125 s audio.
  CPU/CUDA transcript identique et CUDA+cuDNN même texte observé.
  Attention : trois échantillons de l'ancienne campagne, non conclusifs
  sur le p95 ni sur le gain statistique à grande échelle.

## HDB2-B2 v0.4.0 — modifications

- `scripts/hdb2b2_runtime_patch.py` : ajoute un chemin d'inférence à chaud
  et de profil segmenté **sans toucher aux chemins legacy** HDB2-A/B.
  Patch idempotent du checkout local uniquement, verrou Git obligatoire.
- `scripts/hdb2b2_driver.py` : compilation en `target/hdb2b2-*`, exécution
  de 3 échauffements, 10 mesures complètes, 3 mesures segmentées,
  synchronisation périphérique, p50/p95/RTF/dispersion, VRAM GPU globale
  et optionnelle PID, rapports JSON avec preuve runtime.
- `run-hdb2b2.sh` : `selftest`, `preflight`, `repair-runtime`, `build`,
  `bench`, `compare3`, `qualify` automatisé.
- `tests/hdb2b2/` : tests purement logiciels, aucun GPU simulé présenté comme réel.

## Qualification attendue

- Résultat par variante `REAL_CANDLE_HDB2B2_QUALIFIED_ON_SINGLE_AUDIO_UNQUALIFIED_WER`.
- 3 rapports non vides avec exactement même modèle, checkpoint, PCM SHA,
  révisions et même texte NFC.
- `PASS_IDENTICAL_TEXT_ALL_VARIANTS` dans `hdb2b2-compare3/.../comparison.json`.
- CPU, CUDA et CUDA+cuDNN mesurés sur leur appareil effectif, sans fallback.
- Ne PAS extrapoler du seul test micro à un WER français, des p95 de production,
  un gain FP16 ou une performance sur toute la distribution audio.

## Dernière qualification disponible au moment de l'archive

- Archive Python: tests HDB2-A et HDB2-B hérités + tests HDB2-B2 :
  disponibles sous `./run-hdb2b2.sh selftest`.
- **Rust/CUDA/cuDNN B2 non exécuté dans l'environnement de création**.
  La validation doit être produite par l'hôte Pascal réel.
- Prochaine cible après validation : HDB1 — FireRedVAD + Smart Turn,
  avec profiling réel de l'endpoint à la transcription.

## Commandes

```bash
export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
export HDB2A_REPORT_ROOT="$HOME/tmp/haricot-half-duplex/reports"
./run-hdb2b2.sh selftest
./run-hdb2b2.sh preflight
./run-hdb2b2.sh qualify /home/np/test_micro_2.wav
```

## HDB1 v0.5.0 — 2026-10-17: FireRedVAD Stream ONNX + Smart Turn v3.2

- HDB2-A CPU real transcription confirmed locally by user (14.717s, approx. 6.68s cold).
- HDB2-B CUDA+cuDNN real transcription also confirmed; hot median 324.294 ms. HDB2-B2 v0.4.0 qualification not yet compiled by user; may be deferred.
- New `hdb1/` pure-Python ONNX sidecar orchestrates preexisting Rust/Candle FastConformer/TDT; NOT pure Rust ONNX and NOT a simulated production model.
- FireRed ONNX models from eschmidbauer GitHub pinned `faa757dafa0058d0b231ec82e084ec910a3a46d5`, uses Kaldi FBank, CMVN, ONNX caches. Smart Turn v3.2 CPU INT8 from HF pinned `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`, sha256 strict.
- Input 16kHz PCM16 mono, PipeWire `pw-record` or FFmpeg WAV, 100ms input chunks, speech_end candidate -> semantic Smart decision; incomplete retains audio; 3.2s forced hesitation limit; 25s turn max.
- Same QVAC Candle checkpoint and existing `target/hdb2b-cuda-cudnn/release/parakeet-candle` binary reused; per-turn process creation cold, no claim of 324ms complete turn-to-transcript latency.
- Events in `reports/hdb1-*/events.jsonl`, accepted or forced turn WAVs and text, JSON reports, raw voice retained locally.
- Commands: `./run-hdb1.sh selftest`, `install-python`, `download-models`, `preflight`, `file /home/np/test_micro_2.wav --variant cuda-cudnn`, `mic --max-turns 1`.
- Software tests only, no real ONNX or microphone validation conducted in artifact environment; next step user runs file then mic and examines `HDB1_REPORT` and `events.jsonl`.

## HDB2-C v0.6.1 — 18 Oct 2026
- User Python 3.14 pip resolver backtracked `nemo_toolkit[asr]` to 2.5.0 and built ONNX from source: Protobuf gencode 7.36.1 vs runtime 5.29.6.
- Corrective separate `.venv-hdb2c-py313` via `uv`; pin NeMo 2.7.3, ONNX 1.18.0 (prebuilt), Protobuf 5.29.6, torch 2.12.1+cpu.
- Actual NeMo import/restoration still NOT RUN in ChatGPT environment. Validate via `verify-nemo` and C1 on user machine.
- Existing environment `.venv-hdb2c` preserved; do not change Candle, CUDA, HDB1.


## HDB2-C v0.6.2 — 18 Oct 2026
- Résolution de la panne v0.6.1 observée après succès de Torch CPU 2.12.1+cpu : `uv` a privilégié l'index pytorch.org pour setuptools (max 78.1.0), mais le logger NeMo 2.7.3 exige setuptools>=79.
- Le venv `.venv-hdb2c-py313` et Torch CPU sont **réutilisés** sans téléchargement.
- NeMo, ONNX 1.18.0 et Protobuf 5.29.6 résolus sur PyPI **seul** ; `setuptools>=79,<82` installé d'abord sur PyPI.
- Contrôle `torch.__version__ == 2.12.1+cpu`, `torch.version.cuda is None`, `pip check`.
- Pas de changement de l'exécution Candle ni des poids, C1/C2/C3 toujours à qualifier matériellement.
- Les tests logiciels prouvent la structure du lanceur, pas que l'ensemble des dépendances externes soit installable dans toute combinaison de plateformes.

- Environnement inter-archives : autodécouverte de l'ancienne `.venv-hdb2c-py313` dans les dossiers partagés/versionnés et variable `HDB2C_ENV_DIR` pour réutilisation explicite. Le CLI C1 et `verify-nemo` utilisent le même choix.

## HDB2-C v0.6.3 — 2026-10-18

C1 same-PCM original NeMo significantly more faithful than Candle. No numeric parity yet. Confirmed source mismatch: `normalize=NA` erroneously enters global normalization branch in pinned n0madic/parakeet-candle. New NA-only full source-copy A/B candidate in `hdb2c/na_ab.py`, do not alter production binary; baseline report `hdb2c-20261018T195644Z-87019-000` contains real NeMo probes. Run `probe <real-report> --variant cpu` then `na-ab <real-report> --variant cpu`. Await real hardware evidence. Web research: huggingface/candle PR #1560 Whisper mel and #3341 Parakeet separate concerns; NVIDIA-NeMo/Speech normalize_batch NA bypass.


## HDB2-C v0.6.4 — Rust probe E0596

Le modèle NeMo original passe C1 et C1 `--probes`. Les sondes Candle échouent à la compilation E0596,
car Conformer::forward exige `&mut self` mais le probe utilisait `match &model`.
La v0.6.4 corrige les DEUX emprunts, revalide le SHA256 de l’ancienne sonde avant substitution,
et isole le candidat A/B dans `runtime-hdb2c-na-v0.6.4`; aucune nouvelle installation NeMo.
Exécuter `probe` puis `na-ab` sur le rapport `hdb2c-20261018T195644Z-87019-000`; détails dans
`docs/RELEASE-HDB2C-v0.6.4.md`. Le correctif Rust reste à compiler sur la machine réelle.

## HDB2-C2 v0.6.6 — Exact NeMo audio frontend and install_update.sh

- Primary goal: correct poor French ASR by matching original NeMo frontend in an isolated Rust `parakeet-candle` candidate, without modifying production CUDA/cuDNN or HDB1.
- NeMo original vs Candle C1 on real Bluetooth WAV proves fidelity gap. NumPy/PyTorch independent STFT+Mel reproduces real NeMo log-Mel RMSE=1.3690088e-6 (1352 valid frames, 128 bins), confirming exact conventions, **but no Rust runtime parity proven yet**.
- New authored full Rust source `rust/hdb2c_nemo.rs`: NeMo symmetric Hann, continuous-frequency Slaney, zero padding and centered window, no NA normalization, log guard 2^-24, temporal masking + padding.
- `src/parakeet/model.rs` candidate direct path passes the valid 1352 frames to `encoder.forward` and `decode`, avoiding 1360 padding interpreted as valid.
- New `hdb2c/c2.py` copies pinned external runtime source (excluding huge target and .git) into `runtime-hdb2c-c2-v0.6.6`, auditable source SHA/manifest, idempotent only when unchanged; no production source or model weights touched.
- `install_update.sh`: `./install_update.sh` runs tests + builds CPU, `./install_update.sh --report "$HDB2A_REPORT_ROOT/hdb2c-20261018T195644Z-87019-000"` builds + runs A/B, optional later `--cuda-cudnn`. Existing NeMo 3.13 venv reused; no Python installs or HF checkpoint downloads. `run-hdb2c.sh c2` delegates to it.
- NeMo C2 targets: full padded Mel [1,1360,128], valid_frames=1352, full encoded [1,170,640], encoder_valid_frames=169. Numeric parity gate (RMSE/MAE/max/cos), compare only valid frames. C4 TDT logits unqualified.
- Actual Rust cargo / Pascal sm61 CUDA compilation NOT RUN in archive environment. Await build errors or `c2_ab_cpu.json` and `c2c3_exact_cpu.json` before promotion.

C2 `install_update.sh` also runs `cargo test --release --lib hdb2c_nemo::tests` on the isolated candidate after building. Compilation and runtime unit tests remain pending on the user machine.

## HDB2-C3-A-R5 — 2026-10-20

- User requests implementation and GitHub Actions on existing `half_duplex`.
- R4 run 39429711430 completed; two CaFE clips show positional max_abs=1,
  cosine ~0.948, with block0 input already different. R4 is a completed
  diagnostic, not encoder parity; 26-stage instrumentation control passed.
- Audited NeMo 2.7.3 transcription teardown: parent mode restoration precedes
  encoder unfreeze. R5 records actual child training state after warmup and
  recursively enforces eval before fresh C1 and C3 probes. Old references
  lacking eval evidence are rejected; do not compare their numbers as a
  direct native-kernel A/B with the new inference reference.
- Isolated R5 candidate preserves R4 source and weights; f32 frequency
  multiply/exp order matches NeMo. Native sin/cos remain native. A separate
  actual Candle positional binary checks 17 identical-input cases, including
  scaling on/off, offsets, short sequences and table growth.
- Publish only `c3_positional_0/1.json` summaries alongside existing reports.
  Numeric tolerance, strict f32 byte equality, whole-encoder parity and
  TDT/GPU parity are separate fields. Do not promote a green diagnostic run.
- Request and scope: `ci/real_cpu_qualification_request_20261020_r5.md`.
  Local source/gate tests pass; real Rust/NeMo qualification awaits CI results.

R5 first CI attempt 39436600013: 124 Python tests and real R5 Rust builds pass.
First clip confirms 700 training children after transcribe, reduced to 0 by
recursive eval; C1/C3 reference exact. Scaling is bitwise equal in 17/17 cases;
native positional values are not bitwise exact, long T=513 cases reach
3.0517578e-5 and fail the fixed 1e-5 gate. Second clip stopped on strict R4
instrumentation self-control after R5 used the R4 Cargo target directory.
R5 correction isolates its entire Cargo target and checks R4 executable SHA
before/after build. Keep all numerical and exact gates unchanged; rerun both
clips. First-run report is partial, not an encoder or positional qualification.

R5 corrected run **39440057205 completed successfully** on tested source
`d15fc800a68355cca0e8089fc3b1e5343c956dcb`: 125 Python tests, Rust and both real
CaFE clips. Full independent Cargo target fixes the observed second-clip
control failure; strict R3C/R4 equality passes all 26 stages on both clips.
NeMo 700 training children after transcribe confirmed twice; recursive eval
gives exact fresh C1/C3 reference. Scaling is bitwise exact in 34/34 operator
cases. Live positional max_abs=3.8072467e-6 on both clips, but long T=513
max_abs=3.0517578e-5 fails unchanged 1e-5 gate: strict positional parity still
NOT_ESTABLISHED. Last encoder layer passes clip 0 (50/50 frames, max3.87e-7),
fails clip 1 (51/52 frames, max0.22318, cosine0.97509). Padding-mask parity is
a next hypothesis, not proven attribution. Both clips still exceed the
absolute pre_encode gate slightly (0.00146/0.00122); no all-stage promotion.
Details and artifact links: `ci/RESULTS-HDB2-C3-A-R5.md`.

## HDB2-C3-A-R6 — Encoder Padding Mask Parity

User approves verifying the extra frame/masking hypothesis after R5. Implement
an isolated native R6 candidate and 2x2 attention/convolution mask ablation on
the same two real CaFE clips, using the existing branch/workflow. Observe
actual NeMo masks and actual post-mask depthwise input. Native masks must
match those bits exactly; mask-OFF must reproduce all 26 full-frame R5 traces.
The no-padding clip must be bitwise invariant across all four modes. Separate
R6 Cargo target; unchanged R5 executable SHA across build. Native unit tests
cover padded queries/keys, masked scores/probabilities, and GLU-boundary zeroing.
No internal activation injection, no threshold relaxation or GPU/TDT claim.
Request: `ci/real_cpu_qualification_request_20261020_r6.md`. Local gate tests
and real pinned-source transforms pass; real compilation/inference await CI.

R6 run **39449511991 completed successfully**, source
`ea021b7b8df8634e6620498798baa57ffb7e5410`: 133 Python tests, 3 native Rust
mask tests, all 8 real forwards. Mask-OFF reproduces R5 bitwise on all 26 full
stage tensors for both clips; no-padding clip is bitwise invariant in all four
modes. Native mask bits exactly match real NeMo masks. Padded clip (51/52):
no masks max_abs=0.2231819704 cosine=0.97509012; attention only 0.2014631703 /
0.96945573; convolution only 0.02134536952 / 0.99827817; BOTH 5.1036477e-7 /
0.99999999999953. Both masks are necessary for the fixed gate in this test.
All 24 encoder blocks and final output now pass numerically on BOTH clips;
pre_encode/post-scaling strict absolute errors and non-bitwise positional math
remain unchanged. Normal ASR/TDT, cache streaming and GPU remain unqualified.
Detailed evidence: `ci/RESULTS-HDB2-C3-A-R6.md`. Padding-mask hypothesis is
confirmed in the controlled two-clip experiment; no threshold relaxation.

## HDB2-C3-A-R8 — 2026-10-21

User explicitly authorizes frontend primitive decomposition, common-input
subsampling comparisons, all 14 existing CaFE/Oreau excerpts, then actual
CUDA/cuDNN sm61 and laptop microphone. Parent source is R7B `45878246`.
R7/R7B establish numerical TDT and exact discrete decisions on two real CPU
clips; global f32 bit equality still fails. Do not equate CI execution with
complete numerical or bitwise parity.

R8 adds separate instrumented sources, independently compiled R7 baseline,
actual uninstrumented/observed NeMo controls, echo-verified common inputs and
weights, and per-clip/corpus JSON summaries. Original absolute/cosine gates
are unchanged. CI uses `--require-numeric`, so completed diagnostics with
failed numerical gates produce a failed qualification job. No production
promotion. Raw public-corpus tensors are deleted per clip; microphone audio
remains local. Instructions and exact scope: `ci/README-R8.md`.

R8 CPU run **39566816910** on source **ce3461ff** completed all 14 clips with
zero execution errors; the real ASR job correctly fails unchanged numerical
gates. Frontend stages pass 14/14; final encoder output passes 14/14 (worst
9.2387199e-7). All 283 TDT decisions are identical. Five convolutions and three
ReLUs on common inputs pass 112/112. Common-input linear `out` fails 6/14
(worst 2.197265625e-3). Exact final text is 12/14: Oréau neutral/happiness differ
after identical token selection. Global FP32 bits fail 14/14. All 24 block
outputs pass 13/14; Oréau surprise layer 12 fails. Full initial/internal stage
gates remain unqualified. Detailed evidence and artifact digest:
`ci/RESULTS-HDB2-C3-A-R8.md`.

R8B observes actual masked native operands, avoiding R8's reconstruction of
unmasked previous outputs. It adds independent FP64 linear diagnostics and
actual tokenizer category/codepoint diagnostics. Autonomous FP32 math and all
numerical/bitwise gates stay unchanged. R8B CPU run **39597750497** on
**b1d6a027** completes all 14 clips without errors, with bitwise observation
controls and 283 exact TDT decisions. Detailed evidence and artifact digest:
`ci/RESULTS-HDB2-C3-A-R8B.md`. Both text mismatches emit one unknown token;
NeMo renders its original surface ` ⁇ ` while native concatenation uses
`<unk>`. Native text has not yet been corrected. Independent FP64 linear
results rounded to FP32 are bitwise identical 14/14; comparing that result to
Torch FP32 still fails 1/14 (CaFE fear 1.220703125e-3). Do not claim that higher
precision alone solves all gates. Actual post-mask input observations replace
R8's unmasked reconstruction artefacts.
Actual sm61 CUDA/cuDNN and microphone results are still pending.

The unified V6.3 archive was inspected: its four profile resolvers do not
support the STT workflow, and its ephemeral container does not inherit host
model paths. `ci/stt_sm61_runner.py` reuses its existing image and GPU lock,
binds exact run/job/commit and dedicated campaign labels, and prompts for the
usual ephemeral registration token only on the physical host. The STT CUDA
workflow now bootstraps temporary public assets/CPU NeMo itself. No change to
the user's archive or image; no assumption of a live or remotely accessible
runner. Usage: `ci/README-R8.md`.

Physical GTX 1080 Max-Q preflight from the user: sm61, driver580.178.04,
CUDA12.9.2, cuDNN9.10.2.21 pass. CUDA run39598582574 is queued at candidate
8e827fadddfb6b5e555289836728aee52c6b9bb1. Launcher commitd0a322d fixes
non-seekable /dev/tty r+ via a writable prompt stream/getpass read descriptor;
a real pseudo-terminal regression passes. GPU inference/mic not yet measured.

## HDB2-C3-A-R9-TEXT — isolated candidate

R9 prepares the concrete text fix while the original R8B CUDA campaign runs.
It extracts the original SentencePiece unknown ID/piece/surface from the pinned
.nemo, verifies the complete converted vocabulary, and refuses unsupported
control/byte/denormalizer settings. TDT token text and final full text use this
metadata; neural arithmetic and decision logic stay unchanged. No promotion.
An independent real R8B baseline is rerun on all 14 clips. Additional R8B/R9
controls require all frontend/encoder/operand/TDT bytes and non-text result
fields to match; the original R7/R8B complete result-byte control is retained.
Actual NeMo independently evaluates corrected text on both native/shared paths.
Original numerical/bitwise gates remain strict. 161 Python and 12 native Rust
tests pass locally; the actual R9 probe compiles. Real R9 results pending.
Scope/usage: `ci/README-R9-TEXT.md`.

R9 microphone mode is prepared separately (`ci/r9_microphone.sh`). It checks
local model/environment prerequisites before recording, detects usual existing
NeMo Python locations, reuses the physical unified runner GPU lock, and routes
all microphone audio/tensors/summaries to local directories even with
GITHUB_WORKSPACE set. The new publication-path regression passes (162 Python
tests including earlier checks). Native Rust rendering matches original
SentencePiece on 13195 isolated token-sequence cases. Actual microphone capture
and inference remain pending; the running corpus jobs use their immutable
original candidates.

## CPU-first constraint and completed R9/GPU measurements — 2026-10-21

User explicitly requires **no new GPU test until CPU works at all declared
numerical gates**. Do not rerun/retry GPU run39598582574, enqueue another sm61
job, or run the CUDA microphone while CPU remains unqualified. The CUDA
workflow is manual-only; push markers cannot queue a GPU job. Keep the
microphone prepared and defer its GPU execution.

GPU run39598582574/job119216990721 on exact source8e827fa is COMPLETE: all 14
clips and 283 identical TDT decisions, zero execution errors. Strict numerical
qualification FAILS. Common-input `out` fails14/14, worst0.0068359375; valid
final encoder passes14/14, worst9.462237358093262e-7. Exact text12/14 retains
the known unknown-token issue because this is the immutable R8B candidate.
Artifact12346604616 SHA256
7d2651eb672a2c84ee179b44cdc62f79f23481fd8b86f2b7d3254c0b1541fc18.
Full evidence: `ci/RESULTS-HDB2-C3-A-R8B-GPU.md`. Read completed evidence only.

R9 CPU run39602129424 on sourcea19c132 is COMPLETE, all14 clips, zero execution
errors. Exact text, full discrete TDT output and TDT numerical gates pass14/14
on BOTH native/shared-Mel paths. All28 R8B/R9 mathematical controls pass
bitwise; all11 frontend primitive byte controls pass. Final valid encoder
passes14/14, worst9.238719940185547e-7. Original absolute internal gates still
FAIL: common-input projection6/14, scaled/block0 errors and layer12 on one
clip. This is a validated text correction, not complete CPU qualification.
Artifact12348401210 SHA256
9f92f6495f54b2406b9ff5280f98a398cf8871a193c902283266d1b52e2bd7ca.
Full evidence: `ci/RESULTS-HDB2-C3-A-R9-TEXT.md`.

CPU investigation: original converted weights preserve fixed
`preprocessor.featurizer.window` [400] and `.fb` [1,128,257]. The native
frontend currently recomputes these with rounding differences. Loading original
checkpoint buffers is a hypothesis under investigation, not a measured fix;
any candidate must verify their bytes against actual NeMo and retain the
independent baseline and all original gates. Shared-Mel failures prove that
frontend changes alone cannot be assumed sufficient.

## R10 native CPU MKL profile — local 14-clip result

CPU-first instruction remains absolute: no GPU rerun/queue or CUDA microphone
until all CPU numerical gates pass. R10 is a separate CPU-only qualification
binary; default Candle runtime is not promoted. `ci/README-R10-CPU.md` describes
the optional standalone MKL2024.2.2 SDK and CPU workflow input `r10_cpu`.

Original checkpoint buffers match actual NeMo exactly: window SHA256
4009f7a2105ca4d3cb8098fe62d296a08e0418d845255135a8a99164ed399795,
Mel bank SHA256729949d567af3397191d22901d9661709c40a875782941181b545b75d7a4bb73.
The currently recomputed window differs even from a window recreated by the
current Torch; use checkpoint buffers, not a phase-only numerical guess.

Local AMD EPYC9V74/9-thread measurement on all14 real PCM files: frontend all11
stages bit-identical154/154; common frontend7 stages exact98/98; common-input
projection exact14/14 using native MKL SGEMM with bias included. No libtorch
link or activation injection. Text/TDT numerical/discrete pass14/14 on both
paths, all283 decisions exact. Final valid encoder max4.805624485015869e-7.
Remaining strict failures: pre_encode1/14 max0.001220703125; post-scaling and
block0 internals14/14, post_pos_enc max0.03125; one layer12 max0.00146484375.
All gates unchanged. Internal values reach90822, with FP32 ULP0.0078125,
larger than the original absolute gate0.001. Never infer full qualification
from final encoder or text success. Detailed local source/SDK hashes and
redacted metrics: `ci/RESULTS-HDB2-C3-A-R10-CPU.md` and its JSON link.

164 Python and3 native SDK tests pass; actual new binary compiles and runs14
clips. This local measurement reuses actual per-clip original NeMo references;
independent R7/R8B executable controls are NOT rerun locally. The hosted R10
entry point retains/reruns all original baselines and controls. Hosted profile
results must be checked independently. No production/GPU/mic qualification.

Next CPU experiment: actual Torch dispatch chooses oneDNN for all5 subsampling
convolutions. Pinned oneDNN3.11.2 commit03c022d3ffdcee958cfacbe720048e725fdf644c
matches the Torch build. A standalone native build is prepared for exact-input
comparisons. It must include both CONVOLUTION and REORDER primitives; the first
convolution-only build could not execute required reorders. No successful
oneDNN convolution measurement has been claimed yet.

## R11 local CPU result and next normalization operator — 2026-10-21

R10 code/results committed378977308a24a7c2ef8dc41f413422bdfd720356 with CPU
marker[r10-cpu]. The branch ref and public git remote reflect this commit,
but Actions API still showed no new run at the last read; do not claim that
the hosted campaign has run/passed. Workflow dispatch input r10_cpu is ready.
GPU workflow remains manual-only and no GPU test was launched.

Standalone oneDNN3.11.2 at03c022d3ffdcee958cfacbe720048e725fdf644c now builds
with BOTH CONVOLUTION and REORDER, CPU OMP/GNU, GPU_RUNTIME=NONE. All5 common
convolutions are bit-identical on all14 actual NeMo fixtures:70/70. A native
R11 prototype changes only subsampling convolution execution/replay after R10.
Actual full PCM and shared-Mel runs complete14/14: all11 frontend stages,
pre_encode and post_pos_enc are bit-identical14/14. TDT/text pass14/14,283
exact decisions; final valid encoder max7.897615432739258e-7. Strict internal
gates still FAIL: block0FF1 max0.01367187514/14, attention max0.16263961791992188
14/14, layer12 max0.00183105468751/14. Some ReLU differences are signed zeros;
do not remove them to claim global equality. Full evidence/source hashes:
`ci/RESULTS-HDB2-C3-A-R11-CPU.md`, redacted JSON, experimental CPU components.

Do not interpret encoder_audit's array order as chronological execution:
whole-layer outputs are listed before block0 internals. R11's temporary
first_failure summary showed layer12 for one clip, but block0FF1 actually
fails14/14 and is earlier. Use the full per-stage measurements above.

Next first arithmetic gap after exact block0 input: norm_feed_forward1.
CaFE fear max1.33514404296875e-5, inherited/amplified into FF1. Current Candle
CPU uses sequential sum and E[x²]-E[x]² variance; Torch uses vectorized
Welford/cascade. Standalone ordered norm candidate reproduces actual NeMo
norm bits14/14. Important: Torch global CPU reports AVX512 but ordinary
LayerNorm REGISTER_DISPATCH disables AVX512; this kernel uses AVX2 Vec8.
Vec8 + FMA contraction fast matches14/14, Vec16/off variants fail. Independent
scalar-lane C++ implementation and common-input checker are archived under
ci/experimental/cpu_parity. This norm is NOT integrated into the encoder;
no full-pipeline qualification follows. FF1 linears/SiLU also remain to be
separately replayed on identical operands before attributing their full error.

CPU-first constraint unchanged: no CUDA/cuDNN job or CUDA microphone until
all declared CPU numerical gates pass. The default backend and physical
laptop CPU have not been qualified by these local SDK-specific experiments.

Portable experimental R11 builder has been compiled/tested against the actual
R10 sources and CPU SDK. Rebuilt probe vs measured R11 prototype on real CaFE
fear matches75 encoder/operand trace tensors,28 TDT decisions/activations,
returned result and11 frontend traces bit-for-bit. The reused comparator's
literal status string names R7/R8, but the actual scope of this check is only
R11 prototype vs R11 portable rebuild; never present it as a fresh R7/R8 run.
Experimental ordered norm also matches14/14 when compiled for AVX2/FMA with
the documented flags. No integration into the full encoder was performed.

## R12 — local isolated CUDA LayerNorm command and manual CPU CI

The user explicitly requested a local GPU test outside CI for the exact
ordered LayerNorm candidate, followed by CPU CI. The existing candidate was
only an isolated CPU operator; no CUDA activation command existed. R12 adds
an explicitly isolated first-norm experiment, without replacing any operator
in the full Candle model. Do not describe this as an integrated STT candidate.

`python3 half_duplex/ci/local_layernorm.py --device cuda` reuses the existing
rootless unified image, the shared GPU lock and a persistent local cache. It
overrides the registration entrypoint and never calls Actions or asks for a
token. It prepares the actual 14 public clips/original checkpoint, captures
real unmodified NeMo CPU common inputs and weights, replays the archived CPU
source, then runs an independent diagnostic CUDA port through NVRTC12.9 and
the CUDA driver. It verifies copied GPU operand bytes and keeps CPU/NeMo,
CUDA/NeMo and CUDA/CPU verdicts distinct. No automatic CPU fallback exists.

Fresh local CPU measurement with Torch2.12.1+cpu/NeMo2.7.3, AVX2/four threads:
first norm640 features exact14/14, observation-vs-original encoder output
exact14/14. JSON/source hashes recorded in the R12 results. All170 Python
tests pass. Offline NVRTC12.9.86 compilation emits compute_61 PTX; no GPU
kernel was executed here and no native Podman launch was available to test.
An actual local GPU result is still required. The original C++ norm source
remains unchanged and the full R11 internal-stage failures remain unresolved.

CPU workflow adds manual input `r12_layernorm=true`, which selects a hosted
CPU-only operator job and uploads only its JSON metrics. This new heavy job
does not run on push. Prepare it now; dispatch after reviewing the local GPU
result as the user requested. The CUDA workflow remains manual-only; no new
full-model GPU campaign or microphone test has been authorized by this
isolated-operator request. See `ci/README-R12-LAYERNORM.md` for both commands.

## R12 GPU CI request after hosted CPU qualification

User supplied CPU run39649356383 and a failed local Podman `diagnose` command.
Verified through Actions jobs/logs and the downloaded artifact: CPU LayerNorm
14/14 bit-identical, final PASS_BITWISE_14_COMMON_INPUTS, source core unchanged.
The run's real-cpu-asr job was skipped; full STT CPU qualification is not implied.
The `/etc/hosts: permission denied` was a launcher preflight bug: local diagnose
omitted --no-hosts, unlike the actual local execution and the CI runner. Fixed
through a shared diagnose_command helper and a regression test. No CUDA kernel
ran during that failed local attempt.

The user now explicitly authorizes GPU CI for the isolated R12 operator. Add
manual r12_layernorm mode to the existing CUDA workflow; exclude both full-model
paths whenever it is selected. Download CPU artifact from39649356383 and verify
all14 exact controls, model/reference pins and current core source hashes before
GPU execution. Keep GPU workflow dispatch-only and the run-specific runner
labels/commit checks. R12 result uploads as a separate JSON artifact. CPU artifact
expires after14days; future reuse needs fresh CPU evidence and its updated pin.

New local command: python3 half_duplex/ci/stt_sm61_runner.py layernorm-ci.
Uses the laptop's authenticated gh CLI to dispatch R12 only, selects its exact
run by a random request identifier/display title, then starts the existing
unified one-job runner. The temporary registration token stays in the terminal.
Candidate must match local HEAD and current branch. Completed CPU or stale GPU
run IDs cannot be reused. Generic check/start commands retain their --run-id
requirement. No physical GPU was started from the assistant environment.

Validation:13 targeted Python tests pass (new routing/preflight/CPU-evidence
checks, existing runner binding/PTY test and CPU-first boundary). Actual CPU
artifact from the user's run passes verify_r12_cpu.py against current core
sources. No model math or CUDA kernel source changes; no new full CPU/GPU STT
measurement is claimed. The local GPU CI command still needs execution on the
user's physical laptop; this environment has no dispatch API or runner access.
