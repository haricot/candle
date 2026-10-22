# HDB2-A v0.2.3 — Préprocesseur NeMo / dither

## Diagnostic sur la machine de l'utilisateur

- `runtime.stderr`: `Error: missing field `dither`` après `HDB2A_CANDLE_LOCAL_MODEL=...`.
- `report.json`: statut `NOT_QUALIFIED`, `runtime_exit_code=1`, sans transcription.
- `ffmpeg.log`: enregistrement source 24 kHz mono 14.717125 s, normalisé avec succès à 16 kHz / 235474 frames.
- Inférence Candle CPU **non démarrée** : échec de désérialisation d'un champ de configuration.

## Correctifs

- `convert_nemo_hdb2a.py`: impose `preprocessor.dither=0.0` lors de l'export déterministe. La valeur d'origine (ou son absence) est conservée dans `conversion.json` (`original_dither` / `inference_dither`).
- `hdb2a_model_config.py`: contrôle des clés obligatoires pour le runtime Rust épinglé, avec erreur agrégée `CANDLE_CONFIG_MISSING_FIELDS=...`; réparation locale idempotente et atomique, sous contrôle de provenance QVAC EU-sub. **Aucun poids n'est modifié**.
- `run-hdb2a.sh`: deux sous-commandes `repair-model-config` et `check-model-config`.
- `run_candle_hdb2a.py`: contrôle de configuration avant invocation du runtime ; refus explicite des modèles incomplets.
- Validation logicielle : 8 tests Python, y compris conversion synthétique sans champ dither, réparation sans modification répétée, refus des champs inconnus ; aucune compilation Rust/Candle dans l'environnement de fabrication.

## Reprise sans nouveau téléchargement / recompilation

```bash
cd "$HOME/tmp"
unzip haricot-half-duplex-hdb2a-standalone-v0.2.3.zip
cd haricot-half-duplex-hdb2a-v0.2.3
source "$HOME/tmp/haricot-half-duplex/.venv/bin/activate"
export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
./run-hdb2a.sh selftest && \
./run-hdb2a.sh repair-model-config && \
./run-hdb2a.sh check-model-config && \
./run-hdb2a.sh transcribe /home/np/test_micro_2.wav cpu
```

Ne pas relancer le `bootstrap`, la conversion Safetensors, ni `cargo build` pour ce correctif JSON, car le runtime a déjà été compilé et son chargeur démarré sur la machine cible. Si le schéma signale d'autres champs manquants, vérifier la source NeMo avant de les renseigner ; ne pas deviner des valeurs de modèle.

**Seuil de réussite**: `transcription.txt` non vide issu du runtime Candle, `runtime_exit_code=0`, statut `REAL_CANDLE_TRANSCRIPTION_OBSERVED_UNQUALIFIED_WER`, puis vérification humaine du texte.
