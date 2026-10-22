# HDB2-B2 v0.4.0 — Qualification de latence sm_61

## Objectif

Qualifier le fonctionnement réel de QVAC TranscribePsy-250M-EU-sub sur trois
binaires indépendants CPU FP32, CUDA FP32 et CUDA+cuDNN FP32, en préservant
le checkpoint QVAC, le runtime Parakeet épinglé et `haricot/candle` à la
révision `f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3`.

Ceci n'établit ni la précision WER, ni la supériorité sur d'autres fichiers
ou matériels, ni une optimalité GPU universelle.

## Mesure de la latence

Chaque `bench` effectue **3 warmups puis 10 répétitions chronométrées**
avec les poids déjà chargés, dans un seul processus. Chaque appel inclut
lecture WAV 16 kHz, log-Mel, encodeur FastConformer, décodage TDT et
alignement des tokens. Synchronisation du périphérique avant et après les
appels pour exclure un sous-comptage des kernels asynchrones CUDA.

Les métriques sont l'ensemble des valeurs brutes, p50 (médiane), moyenne,
minimum, maximum, écart-type, coefficient de variation, p95 nearest-rank
et RTF basé sur la durée réelle du PCM. **À n=10, le p95 nearest-rank
correspond au maximum observé** ; ce n'est pas un p95 de production
avec intervalle de confiance robuste.

`model_ready_ms` inclut le chargement du binaire, de la configuration,
des poids et du périphérique — il ne mesure PAS uniquement le chargement
CUDA. `process_wall_ms` comprend warmups + mesures + profil segmenté.

## Profil segmenté : instrument d'analyse, pas somme de performances

**Après** les 10 appels totaux, 3 appels de profil indépendants mesurent :

- `wav_decode_ms` : lecture/décodage d'un WAV 16 kHz.
- `logmel_ms` : prétraitement des caractéristiques (avec synchro GPU).
- `fastconformer_ms` : `encoder.forward`, avec synchronisation GPU.
- `tdt_decode_ms` : `decode` du réseau TDT, avec synchronisation GPU.
- `text_alignment_ms` : assemblage et alignement des tokens.
- `profile_wall_ms` : temps total de l'appel instrumenté.

Ces mesures CPU murales **ne sont pas des temps GPU kernel-only**. La
synchronisation explicite de chaque frontière peut perturber les caches,
la planification et le recouvrement CPU/GPU. Le `profile_wall_ms` peut
ainsi différer du `hot_ms_median` et il ne faut pas utiliser la somme des
étapes pour prédire la latence totale non instrumentée.

Le texte extrait dans le profil segmenté doit être strictement égal au
texte extrait dans les dix mesures à chaud.

## VRAM et portée

Un relevé `nvidia-smi` est effectué pendant le processus, environ toutes
les 250 ms : mémoire globale du GPU et, si l'option est prise en charge,
mémoire du PID (`--query-compute-apps=pid,used_gpu_memory`). Les champs
`global_peak_mib` et `process_peak_mib` sont des **pics échantillonnés** ;
ils ne constituent pas le maximum matériel ni l'empreinte d'un allocateur.
Le global inclut éventuellement d'autres processus. `null` indique une
mesure indisponible, pas zéro VRAM.

## Garde-fous

- Baseline CPU et variantes CUDA construites dans des `target/` distincts
  des binaires HDB2-B existants.
- SHA256 du binaire/point d'entrée, runtime et Candle épinglés.
- `CUDA_COMPUTE_CAP=61` et contrôle `nvidia-smi` de la capacité 6.1.
- Refus d'un runtime GPU qui sélectionne un périphérique CPU.
- Conversion audio déterministe et égalité des SHA256 du PCM normalisé.
- Égalité exacte des textes après normalisation Unicode NFC.
- Refus de mélanger les variantes, modèles et checkpoints.
- Échec d'une seule variante => aucune qualification comparative globale.

## Commandes

```bash
export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
export HDB2A_REPORT_ROOT="$HOME/tmp/haricot-half-duplex/reports"

./run-hdb2b2.sh selftest
./run-hdb2b2.sh preflight
./run-hdb2b2.sh repair-runtime

# Reconstitution reproductible des trois variantes (3 + 10 + 3 mesures)
./run-hdb2b2.sh qualify /home/np/test_micro_2.wav

# Alternative incrémentale
./run-hdb2b2.sh build cpu
./run-hdb2b2.sh bench /home/np/test_micro_2.wav cpu 10 3 3
./run-hdb2b2.sh build cuda
./run-hdb2b2.sh bench /home/np/test_micro_2.wav cuda 10 3 3
./run-hdb2b2.sh build cuda-cudnn
./run-hdb2b2.sh bench /home/np/test_micro_2.wav cuda-cudnn 10 3 3
./run-hdb2b2.sh compare3 /path/to/cpu/report.json /path/to/cuda/report.json /path/to/cuda-cudnn/report.json
```

## Critère de conclusion

`PASS_IDENTICAL_TEXT_ALL_VARIANTS` signifie seulement que **ces trois
exécutions**, sur ce fichier, avec le même checkpoint et le même PCM, ont
abouti au même texte. Un WER certifié demande une référence humaine et un
corpus diversifié ; une généralisation des performances demande plusieurs
longueurs/durées d'enregistrement et plusieurs sessions indépendantes.

La préqualification HDB2-B précédente a observé : CPU 6402,94 ms,
CUDA 727,87 ms et CUDA+cuDNN 324,29 ms de médiane à chaud pour 14,717125 s
d'audio. Ces nombres **ne sont pas les mesures HDB2-B2** et ne doivent pas
être injectés dans de nouveaux rapports automatiquement.
