# Local-only recordings — Haricot Half-Duplex

Personal microphone recordings (including Bluetooth audio) **must stay local**.
The repository is public. Only the synthetic fixture
`tests/fixtures/hdb0-tone-16k.wav` may be checked in.

Place personal files in one of:

- `half_duplex/tests/local/`
- `half_duplex/tests/private/`
- `half_duplex/tests/recordings/`

Existing local recordings under `~/tmp/haricot-half-duplex/tests/` may
remain there; moving them into the public Git checkout is not required.

Other `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, `.aif`, `.aiff` and
`.aac` files are ignored throughout `half_duplex/`, including under
`tests/fixtures/`. Local-only test directories ignore all filenames and
extensions. **Note:** `git add -f` can override `.gitignore`, so the CI
also rejects newly tracked audio other than the synthetic fixture.

To check before committing:

```bash
git check-ignore -v half_duplex/tests/local/bluetooth.wav
git check-ignore -v half_duplex/tests/fixtures/private.wav
git ls-files 'half_duplex/**/*.wav' 'half_duplex/**/*.flac'
```

A local-only ASR experiment can explicitly supply its private input path
without storing the audio in Git. Example (run from `half_duplex/`, using
your existing local model/runtime):

```bash
./run-hdb2c.sh c1 ./tests/local/bluetooth.wav --variant cpu
```

The public GitHub Actions workflow uses **CaFE/Oréau (CAMEO)**. It downloads
samples onto ephemeral GitHub runners and never publishes their audio as
artifacts. Only aggregate reports/metrics are candidates for publication.
Do not upload full transcripts of private audio in public Actions artifacts.
