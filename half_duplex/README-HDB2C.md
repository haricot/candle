# HDB2-C v0.6.2 — Numerical Parity & ASR Fidelity

## Pourquoi cette campagne ?

Le checkpoint `qvac/TranscribePsy-250M-EU-sub` est chargé par Candle CUDA+cuDNN, mais le test avec le microphone direct donne encore des substitutions importantes (« mon champ de l'homme » à la place de « mon microphone Bluetooth »). Ajouter 12 dB n'a pas corrigé ces substitutions ; les erreurs existent aussi sans FireRedVAD ni Smart Turn.

**Question falsifiable : le véritable NeMo original reconnaît-il mieux les mêmes échantillons que notre portage Candle ?**

La première étape C1 ne nécessite pas de recompiler Candle. Les étapes C2/C3 fournissent ensuite des sondes numériques séparées. C4 (logits TDT) est réservé à une étape ultérieure, **pas qualifié artificiellement**.

## Installer (une seule fois)

```bash
cd ~/tmp
unzip haricot-half-duplex-hdb2c-standalone-v0.6.2.zip
cd haricot-half-duplex-hdb2c-v0.6.2

export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
export HDB2C_NEMO="$HOME/tmp/haricot-half-duplex/models/source/TranscribePsy-250M-EU-sub.nemo"
export HDB2A_REPORT_ROOT="$HOME/tmp/haricot-half-duplex/reports"

./run-hdb2c.sh selftest
./run-hdb2c.sh preflight --wav "$HOME/tmp/hdb1-direct-micro.wav"
./run-hdb2c.sh install-nemo
./run-hdb2c.sh verify-nemo
```

L'environnement NeMo CPU est créé dans `.venv-hdb2c-py313` (Python 3.13 via `uv`), isolé de l'ancien `.venv-hdb2c` Python 3.14 et de HDB1. Sur Arch : `sudo pacman -S --needed uv` si nécessaire. NeMo 2.7.3 / ONNX 1.18.0 / Protobuf 5.29.6 / Torch CPU 2.12.1 sont épinglés pour limiter la résolution. Ces combinaisons restent à qualifier sur la machine cible. L'installation peut demander plusieurs gigaoctets et nécessite Python >=3.12 ; ne remplace ni CUDA ni votre environnement PocketTTS. Les guides NeMo officiels prévoient explicitement l'inférence CPU : https://docs.nvidia.com/nemo/speech/nightly/starthere/install.html

## C1 — Confrontation des transcriptions (sans nouvelle compilation Rust)

```bash
./run-hdb2c.sh c1 "$HOME/tmp/hdb1-direct-micro.wav" --variant cuda-cudnn
```

Le programme imprimera `HDB2C_REPORT_DIR=/.../hdb2c-...` et y placera :

- `input-16k.wav`, `audio.json` avec le SHA256 du **PCM**.
- `candle_report.json`, `candle.stdout`, `candle.stderr`.
- `nemo_report.json`, `nemo.stdout`, `nemo.stderr`.
- `c1_comparison.json` avec les textes des deux runtimes et une distance entre hypothèses.

Attention : la différence entre les deux textes **n'est pas un WER de fidélité**. Pour obtenir un véritable WER, fournissez un transcript corrigé manuellement :

```bash
./run-hdb2c.sh c1 "$HOME/tmp/hdb1-direct-micro.wav" \
  --variant cuda-cudnn \
  --reference-file /chemin/vers/transcription-humaine-verifiee.txt
```

## C2/C3 — Comparaison numérique des tenseurs réels

Une fois C1 réussi, relancer C1 avec `--probes` pour obtenir les activations NeMo :

```bash
./run-hdb2c.sh c1 "$HOME/tmp/hdb1-direct-micro.wav" \
  --variant cuda-cudnn --probes
```

Copier le chemin `HDB2C_REPORT_DIR`, puis compiler et lancer **uniquement** le binaire additionnel CPU :

```bash
./run-hdb2c.sh probe '/chemin/vers/hdb2c-REPORT_DIR' --variant cpu
```

Le rapport `c2c3_cpu.json` contient RMSE, cosine, max_abs et contrôles de formes pour log-Mel et encodeur. Pour vérifier CUDA également :

```bash
./run-hdb2c.sh probe '/chemin/vers/hdb2c-REPORT_DIR' --variant cuda-cudnn
```

**Ne pas remplacer les binaires ou poids existants.** Le probe produit `candle_mel.f32le` / `candle_encoder.f32le` et des métadonnées d'axes ; les tenseurs NeMo sont en `.npz`.

Si NeMo échoue à restaurer le checkpoint, transmettre d'abord `nemo.stderr` et `nemo_report.json`. S'il transcrit mieux, ne modifier Candle qu'après avoir localisé la première divergence numérique.

## Reproduire / prolonger

- Rapport sur les divergences source : `./run-hdb2c.sh preflight`
- Tests `./run-hdb2c.sh selftest`
- Architecture et protocole : `docs/RELEASE-HDB2C-v0.6.2.md`
- État persistant du projet : `STATE-MEMORY.md`
- Les sources et scripts historiques HDB0 → HDB1-R1 sont conservés. Ne pas les exécuter pour C1.


### Dépannage installation NeMo — v0.6.2

Le premier installateur v0.6.1 a laissé l'index PyTorch CPU prioritaire
pour les dépendances NeMo, cachant `setuptools>=79` disponible sur PyPI.
La commande `./run-hdb2c.sh install-nemo` de v0.6.2 conserve automatiquement
le Python 3.13 et le PyTorch CPU déjà présents, installe `setuptools` sur
PyPI et résout NeMo uniquement sur PyPI avec une contrainte de roue CPU.
Ne pas utiliser `--index-strategy unsafe-best-match` ni recréer l'environnement.
Exécuter ensuite `./run-hdb2c.sh verify-nemo` et, en cas de succès, C1.
Si l'import échoue à nouveau, enregistrer l'erreur sans modifier Candle.

Le lanceur détecte également l'environnement installé sous
`~/tmp/haricot-half-duplex/.venv-hdb2c-py313`, même si vous
exécutez la v0.6.2 depuis son propre dossier versionné. Pour tout autre
chemin, définir `HDB2C_ENV_DIR=/chemin/vers/.venv-hdb2c-py313`
(ou `HDB2C_NEMO_PYTHON=.../bin/python`).


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

## HDB2-C2 v0.6.6 — Full Rust NeMo frontend + one-step installer

Use **`./install_update.sh`** after extracting this ZIP. It checks the existing
HDB2A runtime and checkpoint, executes HDB2-C2 Python tests, prepares a full
independent Rust source tree, and compiles both ASR and probe binaries on CPU.
For comparison with the existing NeMo C1 report, run
`./install_update.sh --report "$HDB2A_REPORT_ROOT/hdb2c-20261018T195644Z-87019-000"`.
See `docs/RELEASE-HDB2C2-v0.6.6.md` for the source contract and reports.

This intentionally does **not** download/reinstall NeMo, PyTorch or models.
No patch file, production runtime mutation or promotion occurs; a copied full
Rust code tree under `runtime-hdb2c-c2-v0.6.6` is compiled and SHA-verified.

C2 `install_update.sh` also runs `cargo test --release --lib hdb2c_nemo::tests` on the isolated candidate after building. Compilation and runtime unit tests remain pending on the user machine.
