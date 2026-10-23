# HDB2-A — charger les poids QVAC officiels sans conversion

Objectif limité : réutiliser **directement** le fichier original
[qvac/TranscribePsy-250M-EU-sub/model.safetensors](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub/blob/main/model.safetensors).
Aucune modification des poids, aucun dépôt Hugging Face `haricot` et aucune campagne
de parité complète dans ce changement.

## Préparation locale (Arch Linux ou autre)

1. Utiliser l'ancien HDB2-A uniquement comme source de `config.json`
   normalisé par NeMo (vocabulaire, TDT, fenêtre, dither et padding). L'export
   HDB2-A possède déjà un `conversion.json` identifiant QVAC. **Ne pas**
   réutiliser les poids convertis pour cette nouvelle exécution.
2. Télécharger une seule fois le modèle officiel dans le cache Hugging Face
   ou vers un emplacement local. Le script contrôle les **913 tenseurs**, la
   disposition native des quatre paramètres LSTM par couche et le SHA256
   de la révision QVAC `ccf70de9ff9848f5cd5975adda1370ce4addf6f1`.
3. Choisir un **nouveau** `HDB2A_MODEL_DIR`. Le script refuse d'écraser le
   modèle converti précédent et crée seulement un lien symbolique vers les
   poids officiels, un `config.json` et un manifeste sans chemin personnel.

```bash
cd /chemin/vers/candle/half_duplex
export QVAC_WEIGHTS="$(hf download qvac/TranscribePsy-250M-EU-sub model.safetensors \
  --revision ccf70de9ff9848f5cd5975adda1370ce4addf6f1)"

export HDB2A_MODEL_DIR="$PWD/models/transcribepsy-eu-sub-qvac-official"
./run-hdb2a.sh prepare-official-model \
  "$QVAC_WEIGHTS" \
  /chemin/vers/ancien-hdb2a/models/transcribepsy-eu-sub-candle

./run-hdb2a.sh repair-runtime
./run-hdb2a.sh build cpu
./run-hdb2a.sh check-model-config
./run-hdb2a.sh transcribe /chemin/vers/extrait-public-fr.wav cpu
```

`repair-runtime` n'altère que le checkout local **épinglé** de
`n0madic/parakeet-candle`, en ajoutant la recherche des poids natifs
`decoder.prediction.dec_rnn.lstm.{weight,bias}_{ih,hh}_lN`. Il continue
à accepter les alias de l'ancienne conversion. Si le runtime local est
déjà en cours d'utilisation, arrêter le processus avant la recompilation.

Les bibliothèques PyTorch, PyYAML, SentencePiece et NeMo ne sont pas
nécessaires pour `prepare-official-model` : Python standard suffit.
La provenance est enregistrée dans `conversion.json` avec
`stage=HDB2A_OFFICIAL_SAFETENSORS` et
`source_safetensors_sha256`, sans `model_dir` absolu.

## Portée et vérification

```bash
python3 -m unittest discover -s tests/hdb2a -p 'test_qvac_official.py' -v
```

Ce test logiciel valide des petits poids synthétiques, le patch RNNT
idempotent, la compatibilité des deux manifestes, le refus des corruptions
et l'absence de copie des poids. Il **ne démontre pas** la parité Mel,
encodeur, TDT ou la qualité ASR du modèle réel. La qualification numérique
sur extrait public français ainsi que l'extraction propre du module Rust
vers `candle-transformers/src/models` demeurent des étapes distinctes.

Si `config.json` de la conversion historique n'est plus disponible,
regénérer *uniquement cette configuration* depuis les métadonnées du NeMo
original est une évolution séparée : ne pas inventer un tokenizer ou une
configuration à partir du seul fichier de poids.
