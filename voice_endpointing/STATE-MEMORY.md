# Port natif initial — 2026-10-22

Demande : porter FireRedVAD et Smart Turn en Rust/Candle avant diarisation,
émotion et débruitage. Préserver le moteur R9 qualifié et sa route CUDA/cuDNN
sur GTX 1080 Max-Q SM61. Python n'est admis que pour préparation/qualification.

## Organisation

Une seule source `voice_endpointing_standalone`, basée sur le fork `main`
`c68b24997319b8d76e5a4e775dab197c7f26322a`, contient la bibliothèque et le
contrôleur commun. Les modules `firered` et `smart_turn` sont indépendants de
R9. L'adaptateur R9 dépend de cette bibliothèque, sans dépendance inverse.
Ne pas ajouter deux branches qui dupliqueraient les mêmes frontends/buffers.

La compatibilité ZIP9 est exactement celle des autres sources standalone :
version `=9.0.3`, adaptations d'API dans npy/pickle. Aucun kernel ou chemin
numérique Candle n'est changé. Le manifeste de promotion Extended n'est pas
modifié par ce port ; qualification GPU/micro de la nouvelle composition à suivre.

## Implémentation

- FireRed : 567 937 paramètres, huit blocs DFSMN, convolution causale groupée,
  cache `[8,1,128,19]`, FBank Kaldi/Povey et CMVN reproduits en Rust.
- Smart Turn : 8 000 386 paramètres FP32, quatre couches Whisper Tiny,
  positions entraînées chargées, pooling par attention, tête de classification.
  Prétraitement Whisper 8 s complet en Rust, sans réutiliser le Mel NeMo STT.
- Statistiques LayerNorm centrées en FP64 puis sortie FP32, uniquement dans
  Smart Turn, pour éviter l'annulation sur les trames de faible variance.
  Les tolérances initiales n'ont pas été desserrées après mesure.
- Contrôleur : politique HDB1-R1 portée en Rust ; 1200 ms de silence,
  annulation par reprise, vrais motifs de clôture forcée et limites PCM.
- `prepare_half_duplex.py` ajoute le binaire `hdb1_native` au R9 généré par
  le build existant. Tous les anciens fichiers Rust R9 sont vérifiés inchangés.
  R9 et les deux modèles endpointing restent résidents dans un seul processus.
  Le module de téléchargement HF de l'ancien CLI est rendu optionnel ; le
  binaire natif n'a besoin ni du hub ni d'OpenSSL.
- `run-native.sh` lance directement ce binaire ; aucune exécution Python,
  ONNX, Transformers ou Kaldi natif dans le parcours microphone/transcription.

## Preuves locales disponibles

- Rust 1.99.0 : build release, rustfmt et six tests Rust réussis.
- Dix cas réels/synthétiques fixés : quatre extraits français CAMEO ancrés
  par SHA256, quatre préfixes, silence et tonalité. Deux chemins par cas,
  PCM autonome et features de référence communes. Tout passe contre les
  références ONNX CPU, avec finitude, formes, tolérances abs/rel et décisions.
- Même campagne réussie contre Candle Extended exact
  `6b29b7f43a574fae32984b34b46de7c1d9b145eb`.
- Erreur maximale de probabilité VAD autonome : 4.589557647705078e-6 ;
  sur features communes : 2.086162567138672e-7.
- Erreur maximale de score Smart Turn FP32 : 2.2649765014648438e-6.
- Quatre tours successifs avec le contrôleur Python de référence, exécutant
  les mêmes modèles FireRed et Smart Turn FP32 : PCM, raisons, événements
  et horloges d'échantillons exactement identiques.
- Binaire combiné CPU compilé contre Extended et aide CLI exécutée.
  Dépendances dynamiques CPU : libc/libm/libgcc ; aucune bibliothèque ONNX,
  Python, hub ou OpenSSL. L'arbre de dépendances normales confirme cela.

Les JSON publiables sont dans `results/`. Aucun WAV, transcript de test,
poids, activation brute ou chemin local n'y est conservé.

## Différence FP32 / INT8 à conserver explicitement

Le port ne reproduit pas le graphe INT8 historique. Les poids viennent du
fichier officiel FP32 `smart-turn-v3.2-gpu.onnx`, exécuté ici sur CPU.
Deux des dix décisions FP32/INT8 diffèrent au seuil 0.5 : préfixe Oréau/fear
et silence seul. Ce ne sont pas des divergences Rust/référence FP32.
Le VAD interdit une clôture de tour lorsque seule du silence est présent.

## Limites et suite

Le nouveau binaire combiné n'a pas encore exécuté le modèle ASR ni le
microphone du laptop dans cet environnement. La compilation CPU et les
preuves du R9 antérieur ne constituent pas une mesure de latence CUDA du
nouveau processus résident. Utiliser install-native.sh puis run-native.sh.

La CI dédiée contrôle les nouveaux modèles sur CPU et compile le raccordement
R9 contre Extended. Elle ne relance pas NeMo, une campagne FP64 ASR ou le
runner physique. Après l'essai micro natif, intégrer cette source au manifeste
Extended avec ses contrôles ciblés. Android reste une cible non qualifiée.

La première référence a été bloquée par le contrôle automatique pour une
connexion de télémétrie ONNX Runtime. Toutes les références actives appellent
disable_telemetry_events avant création de session ; la validation reprise a
réussi. Cette dépendance n'existe pas dans l'application native.

## Correction de préparation locale — 2026-10-23

L'installateur plaçait son output sous `voice_endpointing/integration/`, puis
`prepare_half_duplex.py` recopiait tout `voice_endpointing/` sans exclure ce
dossier. La sortie se recopiait dans elle-même jusqu'à l'erreur de chemin.
La CI ne reproduisait pas ce problème : sa sortie était hors du checkout.
La copie globale embarquait également `.venv-convert/` sur le laptop.

La préparation copie désormais seulement les entrées du crate : `src/`,
`Cargo.toml`, `Cargo.lock` et `NOTICE.md`. Elle accepte les sorties sous
`integration/` sans parcourir les builds existants et refuse une destination
dans les sources Rust. Les anciens fichiers Rust R9 restent vérifiés inchangés.
Les modèles déjà convertis et les anciens dossiers de travail sont conservés.

Deux tests Python stdlib passent : sorties imbriquée/externe avec un ancien
build et un environnement de conversion présents, puis refus d'une sortie
dans `src/`. Le même cas reproduit l'erreur avec l'ancienne copie globale.
Ces tests sont exécutés avant les builds dans `ci/check.sh`. La CI prépare
maintenant réellement le binaire résident sous `voice_endpointing/integration/`
avant sa compilation, comme l'installateur. Aucun calcul Rust n'est modifié.

La première correction de copie (`08134113`) a ensuite échoué en CI et sur
le laptop parce qu'elle demandait `NOTICE` au lieu du vrai `NOTICE.md`.
Le test sur données synthétiques reproduisait ce mauvais nom. Le nom est
corrigé et un troisième test copie le vrai checkout dans un sous-dossier
`integration/`, puis compare les manifestes, la notice et les empreintes de
tous les fichiers Rust. Ce test reproduit l'erreur avant correction et passe
après, en complément des deux contrôles de disposition des fichiers.
