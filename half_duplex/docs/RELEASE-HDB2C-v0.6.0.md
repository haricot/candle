# HDB2-C v0.6.0 — Qualifier la fidélité NeMo/Candle

## Objectifs

**C1** : inférence du vrai checkpoint original `.nemo` sur CPU PyTorch, face au runtime Rust/Candle existant, à partir du **même fichier PCM16 mono 16 kHz normalisé une seule fois**. Capturer les deux transcriptions et leurs modèles/hachages dans un rapport reproductible.

**C2** : extraire de vrais tenseurs Mel depuis `model.preprocessor` NeMo et depuis le `get_logmel` Rust/Candle via un **nouveau binaire cargo indépendant**, `hdb2c_probe`; comparer strictement valeurs, taille et trames valides sans masquer le padding.

**C3** : même chose pour les activations de sortie FastConformer. Les valeurs de l'encodeur ne peuvent être déclarées équivalentes que si C2 est lui-même validé ; un écart en C3 ne prouve pas que l'encodeur est fautif quand C2 diffère déjà.

**C4** : **non implémenté** dans cette première archive : pas de capture des logits joints TDT et pas de parité des séquences de jetons. Une égalité de texte ne remplace pas ce test.

**C5** : aucune correction numérique appliquée ni intégration HDB1 automatique avant que C1-C3 localisent l'écart. Éviter de changer plusieurs étapes simultanément.

## Isolation

- Réutilise sans modification le binaire `target/hdb2b-cuda-cudnn/release/parakeet-candle` et le checkpoint Safetensors existants.
- Environnement Python **séparé** `.venv-hdb2c` pour NeMo CPU et PyTorch. Aucun PyTorch GPU compatible sm_61 exigé.
- Ne touche ni à `haricot/candle` ni au binaire `parakeet-candle`; le nouveau fichier `src/bin/hdb2c_probe.rs` est seulement ajouté au checkout déjà épinglé à `53c0d3065...` après contrôle du SHA Git.
- Compile le probe dans un répertoire Cargo distinct `target/hdb2c-probe-cpu` ou `target/hdb2c-probe-cuda-cudnn` si demandé.
- Conserve les WAV et les hypothèses uniquement dans `~/tmp/haricot-half-duplex/reports/hdb2c-*/`.
- Ne contient ni checkpoint privé ni enregistrement de voix.

## Critères de qualification

| Sujet | Condition de succès |
|---|---|
| C1 | Les deux modèles exécutent réellement l'inférence ; même sha256 PCM ; origine du checkpoint NeMo cohérente |
| C1 référence | Référence humaine vérifiée explicitement, WER séparés |
| C2 | Forme/canaux et nombre de trames valides identiques, pas de NaN/Inf, max abs <= 1e-3, cos >= 0.999 |
| C3 | Mêmes critères, et C2 validé préalablement |
| C4 | **NOT_RUN** : logits/tokens non exportés |
| Fin de campagne | Décision localisée, pas de promesse de WER sans texte de référence |

Le seuil de parité est un **point de départ** pour diagnostiquer la divergence. En cas de petites différences, demander un examen couche par couche avant d'ajuster les seuils.

## Limites connues et hypothèses à tester

Les implémentations ne sont pas intrinsèquement identiques : NeMo peut utiliser une fenêtre STFT symétrique (périodicité différente), des filtres Mel `librosa`, une constante `log_zero_guard` du checkpoint et un padding temporel. L'adaptateur Candle utilise une autre FFT/banque Mel et `ln(x+1e-5)` dans la source épinglée. Ces différences doivent être **mesurées numériquement**, sans inférer leur amplitude à partir des sources seules.

NeMo 3.x peut faire évoluer ses signatures : en cas d'échec Python, conserver `nemo.stderr` et `nemo_report.json`, ne pas forcer `weights_only=False` sur une source non fiable. Le checkpoint `.nemo` ici est celui que l'utilisateur a explicitement téléchargé.

Aucune compilation Rust, inférence NeMo 1 Go ni comparaison GPU n'a été réalisée dans l'environnement d'assemblage ; les tests sont des tests de logique sur petits tenseurs synthétiques.
