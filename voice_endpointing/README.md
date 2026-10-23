# Half-duplex natif Rust / Candle

FireRedVAD Stream et Smart Turn v3.2 sont implémentés ici comme deux modules
Rust indépendants. L'inférence emploie Candle ; le traitement audio emploie
Rust et rustfft. Aucun interpréteur Python, ONNX Runtime, Transformers ou
Kaldi natif n'est chargé par les exécutables.

La branche `voice_endpointing_standalone`, basée sur `main`
`c68b24997319b8d76e5a4e775dab197c7f26322a`, regroupe ces deux ports et leur
contrôleur commun. Ce choix évite de dupliquer les buffers, le traitement audio,
la politique de fin de tour et l'intégration R9 entre deux branches.
Les deux modèles restent utilisables séparément depuis la bibliothèque.

| Composant | Exécution native |
| --- | --- |
| FireRedVAD Stream | DFSMN Candle CPU ; 8 caches `[1,128,19]` |
| Prétraitement FireRed | PCM16 non normalisé, FBank Kaldi 80 bandes, FFT 512, fenêtre Povey 400, pas 160, CMVN |
| Smart Turn v3.2 | Encodeur Whisper Tiny entraîné + pooling par attention + classifieur Candle CPU |
| Prétraitement Smart Turn | Dernières 8 secondes, padding à gauche, normalisation, FFT 400, Mel Slaney 80 bandes, 800 trames |
| R9 dans `hdb1_native` | Modèle ASR résident, Candle CUDA/cuDNN ou CPU explicitement choisi |
| Microphone desktop | `pw-record` PipeWire, mono PCM16 16 kHz ; le contrôleur est Rust |

Les deux prétraitements audio sont différents. Le Mel NeMo du STT ne les
remplace pas. Les positions entraînées de Smart Turn sont chargées depuis les
poids ; elles ne sont pas reconstruites depuis un Whisper générique.

## Première installation sur le laptop Pascal

Le checkout Extended déjà validé est réutilisé, de même que le modèle QVAC
converti. L'installateur produit un nouvel exécutable et ne remplace aucun
ancien runtime. Il demande les outils déjà utilisés : Rust 1.99, CUDA 12.9,
cuDNN, `/usr/bin/g++-14`, Python 3 pour la préparation et un accès réseau
pour les dépendances et poids. Pas de Podman.

Pour créer le checkout depuis le clone de qualification, utiliser un refspec
explicite. Il crée aussi la référence distante lorsque le clone ne suit
qu'une autre branche :

```bash
bash <<'BASH'
set -euo pipefail
cd "$HOME/tmp"

git -C "$HOME/tmp/candle-stt-qualification" fetch origin \
  refs/heads/voice_endpointing_standalone:refs/remotes/origin/voice_endpointing_standalone

git -C "$HOME/tmp/candle-stt-qualification" worktree add --detach \
  "$HOME/tmp/candle-voice-endpointing" \
  refs/remotes/origin/voice_endpointing_standalone
BASH
```

Un dossier de destination déjà présent et vide peut être réutilisé. La commande
`fetch origin voice_endpointing_standalone` seule peut ne renseigner que
`FETCH_HEAD`, sans créer `origin/voice_endpointing_standalone`.

Depuis la racine du checkout créé :

```bash
cd "$HOME/tmp/candle-voice-endpointing"
```

```bash
HDB_PROMOTED_ROOT="$HOME/tmp/candle-hdb-extended-6b29b7f" \
  bash voice_endpointing/install-native.sh
```

Python est utilisé **une seule fois pour convertir les poids et préparer le
build R9 existant**, dans `.venv-convert` isolé, avec versions fixées.
Ni PyTorch ni ONNX Runtime ne sont nécessaires à cette préparation.
L'application distribuée lit uniquement les paquets `safetensors + config +
manifest`. Les poids du modèle et de CMVN sont vérifiés par SHA256.
Le cache de conversion et les modèles restent dans `~/tmp/haricot-half-duplex/models`.

## Essai microphone avec R9 CUDA résident

```bash
bash voice_endpointing/run-native.sh --mic --max-turns 8
```

Sorties attendues :

```text
HDB_NATIVE_ASR=R9_RESIDENT_LOADED_ONCE device=Cuda(...)
HDB_NATIVE_READY=LISTENING turn=0 endpoint_device=Cpu
HDB_NATIVE_STATE=TRANSCRIBING microphone=PAUSED
HDB_NATIVE_TURN={... "reason":"SMART_TURN_CONFIRMED", ... "asr":{"device":"Cuda(...)"...}}
HDB_NATIVE_READY=LISTENING turn=1 endpoint_device=Cpu
```

Parler lorsque `LISTENING` apparaît ; attendre le prochain `LISTENING` après
chaque transcription. `--max-turns` compte les interventions, pas les secondes.
Le microphone est arrêté pendant la transcription ; l'audio mis en attente
pendant cette pause n'est pas rejoué comme une nouvelle intervention.
Les trois modèles restent chargés dans le même processus. Le décodeur TDT est
appelé avec un état initial par tour ; le cache VAD et le buffer de pré-roll
sont réinitialisés après chaque tour accepté.

La politique conserve 1200 ms de silence après le candidat VAD, annule ce
candidat si la parole reprend, puis consulte Smart Turn au seuil 0.5. Les
clôtures forcées `HESITATION_TIMEOUT`, `MAX_TURN_TIMEOUT` et EOF sont distinctes
d'une confirmation Smart Turn. Durée maximale configurée : 25 s par tour.

Les WAV, transcriptions et rapports sont conservés localement dans
`~/tmp/haricot-half-duplex/reports/native`. Aucun envoi audio n'est effectué.
`--pw-target ID_OU_NOM` sélectionne un microphone PipeWire précis.

Un WAV mono PCM16 16 kHz peut être utilisé à la place du micro :

```bash
bash voice_endpointing/run-native.sh --input /chemin/essai-16k.wav --max-turns 8
```

## Bibliothèque et exécutable de détection seuls

```bash
cargo build --release --locked --manifest-path voice_endpointing/Cargo.toml
voice_endpointing/target/release/voice-endpointing \
  --models "$HOME/tmp/haricot-half-duplex/models/native-endpointing-v1" \
  --mic --max-turns 3
```

Ce dernier exécutable indique explicitement `ENDPOINT_ONLY_NO_ASR_REQUESTED`.
`--asr-binary CHEMIN --asr-model-dir CHEMIN --asr-device cuda-cudnn` permet
aussi de raccorder un ancien binaire **R9** explicitement, avec un processus
ASR froid par tour. `hdb1_native`, utilisé par `run-native.sh`, est la version
résidente recommandée ; elle ne lance pas HDB2-B.

Les API `FireRedModel`, `FireRedVad`, `SmartTurn` et `Pipeline::feed` permettent
un raccordement à un autre producteur PCM, notamment mobile, sans PipeWire.
L'exécution Android n'est pas encore qualifiée.

## Référence FP32 et ancienne variante INT8

Le port Smart Turn charge le fichier officiel `smart-turn-v3.2-gpu.onnx`
**pour en extraire les poids FP32 lors de la conversion**. Le nom « gpu » ne
force pas une exécution GPU : le modèle converti est exécuté sur CPU par Candle.
Le graphe ONNX ne sert jamais à l'inférence native.

L'ancien fichier `smart-turn-v3.2-cpu.onnx` est quantifié INT8. Ces deux
variantes n'ont pas toujours les mêmes scores ou décisions : sur les dix
cas initiaux, le préfixe Oréau/fear et le silence diffèrent au seuil 0.5.
Le silence seul n'autorise pas un tour dans la chaîne, car FireRed doit d'abord
détecter de la parole. Les scores ne constituent ni une confiance ASR ni une
mesure d'émotion.

Les statistiques des LayerNorm Smart Turn sont calculées avec une variance
centrée en FP64, puis la sortie est reconvertie en FP32. Cela évite
l'annulation numérique de `E[x²]-E[x]²` sur les trames de faible variance.
Ce choix est local au nouveau port ; aucun kernel Candle ou calcul R9 n'est changé.

## Qualification courte et reproductible

La CI `Native voice endpointing CPU` réalise les contrôles Rust, puis compare
les deux modèles aux sources ONNX CPU, sans NeMo et sans campagne ASR complète.
La télémétrie ONNX Runtime est désactivée avant toute session de référence.
Ces dépendances Python sont isolées et réservées à la qualification.

```bash
python3 -m venv voice_endpointing/.venv-reference
voice_endpointing/.venv-reference/bin/python -m pip install \
  -r voice_endpointing/requirements-reference.txt
ENDPOINT_PYTHON="$PWD/voice_endpointing/.venv-reference/bin/python" \
  bash voice_endpointing/ci/check.sh
```

Corpus fixé : quatre extraits CAMEO français déjà ancrés par empreinte dans la
qualification STT, leurs quatre préfixes, un silence et une tonalité synthétiques.
Chaque cas est exécuté depuis le PCM natif et depuis les mêmes features que la
référence. Les formes, la finitude, les limites absolues **et** relatives, les
événements VAD et les décisions Smart Turn sont vérifiés. Les bits restent un
diagnostic. Les bornes sont déclarées dans `tools/qualify.py` avant les mesures.
Les fichiers audio et les activations ne sont pas publiés par la CI.

Un second contrôle rejoue quatre interventions consécutives devant le
contrôleur HDB1 existant avec la référence Smart Turn **FP32** : PCM découpés,
raisons, événements et horloges d'échantillons doivent être identiques.
La CI compile aussi le binaire combiné contre le checkout Extended promu
`6b29b7f43a574fae32984b34b46de7c1d9b145eb`.

Ces tests qualifient ce port sur ce corpus. Ils n'établissent pas une qualité
générale de fin de tour, une performance microphone sur le laptop, une parité
bit à bit entre backends ou la qualité d'une future diarisation/émotion.

La branche n'est pas ajoutée automatiquement au manifeste de promotion
Extended : sa CI CPU et l'essai micro CUDA résident précèdent cette intégration.
Voir `STATE-MEMORY.md` et `results/` pour les preuves initiales.
