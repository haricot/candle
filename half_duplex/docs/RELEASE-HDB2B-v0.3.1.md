# HDB2-B v0.3.1 — comparateur universel de variantes

**Cause corrigée** : en v0.3.0, la commande `compare` supposait que son premier
rapport était systématiquement `cpu` et le deuxième `cuda` / `cuda-cudnn`.
Elle rejetait donc une comparaison légitime `cuda` vs `cuda-cudnn` :
`ValueError: Expected CPU report and CUDA variant report`.

**Changement** : la CLI accepte désormais deux rapports de variantes **distinctes**
parmi `cpu`, `cuda`, `cuda-cudnn`, quel que soit leur ordre. Aucune re-compilation,
reconversion de poids ni exécution CUDA n'est nécessaire pour comparer des
rapports existants. On garde le format `PASS_IDENTICAL_TEXT` et les anciens
champs `speedup_cpu_over_gpu` pour une comparaison `cpu -> GPU`.

```bash
./run-hdb2b.sh compare \
  /home/np/tmp/haricot-half-duplex/reports/hdb2b-cuda-20261017T133508Z-389260-836c97/report.json \
  /home/np/tmp/haricot-half-duplex/reports/hdb2b-cuda-cudnn-20261017T134424Z-395835-87ff4a/report.json
```

**Convention** : `speedup_first_over_second = median(first) / median(second)`.
Si ce ratio > 1, **la deuxième variante est plus rapide**. Si < 1, la première
est plus rapide. `second_latency_reduction_pct` suit la même orientation
(positif = la deuxième variante est plus rapide). Les variantes et appareils
mesurés sont conservés explicitement dans `comparison.json`.

**Conformité** : refus de comparer un rapport non qualifié, un même rapport, deux
variantes identiques, du matériel déclaré incohérent avec la variante, un
checkpoint, un audio PCM ou des pins distincts, une autre précision que f32,
des mesures invalides et des transcriptions divergentes. La comparaison est
strictement identique après normalisation Unicode NFC (sans supprimer les mots
ou la ponctuation).

**Résultat observé transmis par l'utilisateur** : `cuda-cudnn` sur Pascal sm_61,
texte identique à la baseline CPU, médiane à chaud **324.294 ms** pour **14.717125 s**
d'audio, RTF **0.02204**, runtime `Cuda(CudaDevice(DeviceId(1)))`.
Le rapport `cuda` comparé n'a **pas** été joint au message ; son temps et son
gain face à cuDNN restent inconnus tant que les deux rapports locaux n'ont pas
été comparés. 3 répétitions chronométrées + 1 échauffement ne donnent pas une
estimation robuste des queues de latence ; répéter ensuite à 10+ mesures.

**Tests locaux de fabrication** : 14 tests `tests/hdb2b`, dont CPU/GPU, GPU/GPU,
inversion de l'ordre, rejet d'appareil déclaré invalide, contenu divergent,
source audio différente et répétition d'une même variante. Aucun matériel CUDA
accessible dans l'environnement de fabrication ; les performances proviennent
uniquement du retour utilisateur.
