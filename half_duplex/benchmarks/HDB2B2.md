# HDB2-B2 — Analyse et reproductibilité

Aucun résultat GPU n'est précalculé ni incorporé aux nouveaux rapports.

Pour chaque variante, conserver :

- `report.json` : identité des sources, timing brut, p50, p95, dispersion,
  profil segmenté et VRAM.
- `runtime.stderr` : ligne d'évidence `HDB2B2_BENCH_JSON=` et erreurs.
- `runtime.stdout` : JSON transcription réelle.
- `transcription.txt` : sortie reconnue, uniquement en cas de succès.
- `input-16k.wav` : PCM canonique pour vérification et répétition.
- `ffmpeg.log` : commande de prétraitement.

Commande de lecture rapide des temps de profil :

```bash
python3 - "$REPORT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print('variant=',r['variant'],'p50=',r['hot_ms_median'],'p95=',r['hot_ms_p95'])
for k,v in r['phase_profile'].items(): print(k,round(v['median_ms'],3),'ms')
print('GPU VRAM=',r.get('vram'))
PY
```

Avec dix échantillons, l'estimation p95 est essentiellement un maximum
empirique. Les temps segmentés comprennent des barrières GPU et ne se
substituent pas au chronométrage total. Le relevé GPU est échantillonné.
