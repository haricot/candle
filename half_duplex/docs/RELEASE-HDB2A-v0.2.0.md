# HDB2-A — version source expérimentale

- Base HDB0 v0.1.2 (archive précédente), aucune suppression de fonctionnalité.
- Checkpoint cible : `qvac/TranscribePsy-250M-EU-sub.nemo` **sans modèles de remplacement**.
- Conversion NeMo -> Safetensors pour un runtime Rust FastConformer/TDT déjà existant utilisant Candle.
- Intégration dans un répertoire propre : pas de push GitHub ni de modification de `haricot/candle`.
- Inférence via le binaire Rust/Candle, `--model-dir`; non simulée et refus explicite en cas de texte vide.
- Tests de conversion et de patch réussis dans l'environnement de fabrication.
- **Limitations** : ni binaire compilé ni modèle chargé ici ; c'est un candidat local à qualifier, pas une certification ASR CPU/sm_61.
