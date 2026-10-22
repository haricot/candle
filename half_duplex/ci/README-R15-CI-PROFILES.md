# R15 — CI fonctionnelle courte, campagne complète et audit à la demande

Politique acceptée le 22 octobre 2026, après lecture du run
[39815786458](https://github.com/haricot/candle/actions/runs/39815786458).
Le moteur Rust R9 et ses calculs restent inchangés.

| Profil Extended | CPU | CUDA/cuDNN physique | Audit local FP64 |
|---|---|---|---|
| `auto` | 4 extraits si le moteur est inchangé ; 42 sinon | Même choix que le CPU | Aucun |
| `short` | 4 extraits ; escalade à 42 si le moteur a changé | Même choix | Aucun |
| `full` | 42 extraits | 42 extraits | Aucun |
| `audit` | 42 extraits | 42 extraits avec R14 et ses replays d’opérateurs | Explicite |

Chaque extrait est exécuté par les deux chemins natifs : PCM autonome et Mel
NeMo commun. Le second est un contrôle séparé ; il ne remplace jamais le premier.
La référence est le checkpoint original NeMo en FP32 sur CPU, y compris sur
la GTX 1080 sm_61. NeMo est chargé une seule fois pour le corpus R15.

La CI courte utilise les quatre entrées originales d’indice zéro : CaFE colère,
CaFE neutre, Oréau peur et Oréau surprise. Leur PCM est vérifié contre les SHA256
archivés de R13. Le profil complet utilise les 14 entrées originales et les 28
entrées supplémentaires de R14, avec la même révision CAMEO immuable. Les WAV
et transcriptions ne sont pas ajoutés au dépôt ou aux métriques publiées.

## Contrôles bloquants

- Identité du checkpoint et de la conversion ; corpus et PCM fixés.
- Tokens, texte, trames, durées et toutes les décisions TDT exactement identiques
  à la référence originale, dans les deux chemins.
- Longueurs valides, configuration du décodeur et périphérique demandé exacts ;
  un repli CPU pendant le test CUDA échoue.
- Absence de NaN/Inf dans tous les tenseurs FP32 observés, y compris les zones
  paddées et les traces TDT ; une trace absente/corrompue échoue.
- Sortie finale de l’encodeur sur les trames valides : seuils antérieurs conservés,
  `max_abs <= 0.001` et `cosine >= 0.999`.
- Tous les extraits attendus doivent terminer ; une erreur ou un seul échec suffit.

Le profil `full` observe aussi les couches intermédiaires de l’encodeur, avec
contrôle de non-altération du forward NeMo. Leurs métriques et les différences
de bits sont diagnostiques. Aucun nouveau seuil relatif n’est ajusté aux sorties.
Un NaN/Inf reste bloquant même dans une couche dont l’écart fini est diagnostique.

Le profil `audit` exige la complétude du FP64 et ces mêmes contrôles fonctionnels.
Il conserve séparément l’ancien verdict numérique R14 : un audit peut donc
rapporter `PASS_FUNCTIONAL_REGRESSION` et `NUMERIC_PARITY_NOT_ESTABLISHED`.
Les commandes historiques R13 et R14 avec `--require-numeric` gardent leurs règles.

## Sélection automatique dans Extended

L’assembleur compare les arbres du candidat et du dernier Extended promu avec
la politique R15. Pour la migration initiale, la référence revue est le candidat
`50ed7c1636e93fa5c8bf0a16e90bcce6a5ad7cc1`, mesuré par le run R14 ci-dessus.
Son échec historique n’est pas réécrit et ce candidat n’est pas promu rétroactivement.

Les sources Rust du moteur, ses générateurs, la conversion, les dépendances
et les sources partagées Candle core/nn/kernels déclenchent `full`. Les suppressions
et déplacements sont pris en compte. La liste conservatrice est explicite dans
`resolve_profile.py`. Sans référence qualifiée disponible, `full` est obligatoire.
Les changements de documentation ou d’organisation de CI seuls permettent `short`.

Le choix, sa raison, le SHA de référence et les chemins concernés sont enregistrés
dans `candle-integration/half-duplex-ci.json`, à l’intérieur du candidat immuable.
Les étapes CPU et GPU utilisent ce fichier. La promotion Extended conserve ses
autres contrôles et attend toujours le succès du GPU physique.

## Commandes

Le push de l’orchestrateur utilise `auto`. Pour forcer une campagne complète :

```bash
gh workflow run candle-extended-composite.yml \
  --repo haricot/candle --ref orch/extended-v1 \
  -f campaign="hdb-full-$(date -u +%Y%m%dT%H%M%S)" \
  -f publish=true -f half_duplex_profile=full
```

Pour l’audit détaillé, utiliser la même commande avec une campagne unique
`hdb-audit-...` et `-f half_duplex_profile=audit`.
Le runner physique reste lancé par l’utilisateur avec l’identifiant du nouveau run :

```bash
cd "$HOME/tmp/haricot-sm61-runner-v6.3"
./run-once.sh check candle/extended --run-id RUN_ID
./run-once.sh start candle/extended --run-id RUN_ID
```

Dans un checkout local compatible, les mêmes profils s’exécutent sans Podman :

```bash
RUNNER_TEMP="$(mktemp -d)" HDB_CI_PROFILE=short \
  bash half_duplex/ci/extended_gate.sh cpu
```

Remplacer `short` par `full` ou `audit`, et `cpu` par `gpu` pour le test natif
CUDA sur la configuration sm_61 préparée. Un profil demandé ne peut pas abaisser
celui inscrit dans un candidat Extended.

Python/NeMo restent nécessaires à la référence de CI et à la conversion initiale.
L’inférence native Rust déjà validée au microphone ne reçoit aucune dépendance
Python supplémentaire. R15 ne qualifie pas le streaming, la fin de tour, la
diarisation ou un WER général. La durée de cette nouvelle CI reste à mesurer.
