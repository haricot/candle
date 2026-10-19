#!/usr/bin/env bash
# Heavy real-model validation on two CaFE/CAMEO excerpts; no personal recordings.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RUNNER_TEMP:?}"
export HDB2A_RUNTIME_DIR="$RUNNER_TEMP/halfduplex-real-parakeet"
export HDB2A_MODEL_DIR="$RUNNER_TEMP/halfduplex-transcribepsy"
export HDB2A_REPORT_ROOT="$RUNNER_TEMP/halfduplex-reports"
export HDB2C_ENV_DIR="$RUNNER_TEMP/hdb2c-py313"
export HDB2C_PYTHON="$HDB2C_ENV_DIR/bin/python"
export HDB2C_NEMO_PYTHON="$HDB2C_PYTHON"
export HDB2C_NEMO="$RUNNER_TEMP/checkpoint/original.nemo"
mkdir -p "$HDB2A_REPORT_ROOT" "$RUNNER_TEMP/checkpoint"
./run-hdb2c.sh install-nemo
./run-hdb2c.sh verify-nemo
export PATH="$HDB2C_ENV_DIR/bin:$PATH"
uv pip install --python "$HDB2C_PYTHON" --default-index https://pypi.org/simple 'datasets==3.6.0' 'soundfile' 'huggingface_hub'
hf download qvac/TranscribePsy-250M-EU-sub --include '*.nemo' --local-dir "$RUNNER_TEMP/checkpoint"
mapfile -d '' archives < <(find "$RUNNER_TEMP/checkpoint" -type f -name '*.nemo' -print0)
if [[ ${#archives[@]} != 1 ]];then echo 'NO_UNIQUE_PUBLIC_NEMO_CHECKPOINT' >&2;exit 8;fi
ln -sf "${archives[0]}" "$HDB2C_NEMO"
./run-hdb2a.sh bootstrap
./run-hdb2a.sh prepare-model "$HDB2C_NEMO"
./run-hdb2a.sh check-model-config
# HDB2-B's in-process measurement hooks must be installed in the freshly
# bootstrapped pinned runtime before its build gate checks the source marker.
# This is idempotent and only edits $HDB2A_RUNTIME_DIR (the CI temp checkout).
./run-hdb2b.sh repair-runtime
./run-hdb2b.sh build cpu
"$HDB2C_PYTHON" ci/cameo_subset.py --out "$RUNNER_TEMP/cameo" --per-emotion 1
mapfile -t clips < <("$HDB2C_PYTHON" - "$RUNNER_TEMP/cameo/manifest.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))["rows"]
for x in rows:
    if x["split"]=="cafe" and x["emotion"] in ("neutral","sadness"):
        print(x["wav"]+"|"+x["reference_file"])
PY
)
test "${#clips[@]}" -eq 2
for i in "${!clips[@]}";do
  IFS='|' read -r wav reference <<< "${clips[$i]}"
  ./run-hdb2c.sh c1 "$wav" --variant cpu --reference-file "$reference" --probes | tee "$RUNNER_TEMP/c1-$i.log"
  report="$(sed -n 's/^HDB2C_REPORT_DIR=//p' "$RUNNER_TEMP/c1-$i.log" | tail -1)"
  test -n "$report" && test -s "$report/nemo_report.json"
  ./install_update.sh --report "$report"
  test -s "$report/c2_ab_cpu.json"
  test -s "$report/c2c3_exact_cpu.json"
  mkdir -p "$GITHUB_WORKSPACE/halfduplex-ci-published"
  cp "$report/c2_ab_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c2_ab_$i.json"
  cp "$report/c2c3_exact_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c2c3_$i.json"
done
echo 'HALF_DUPLEX_REAL_CPU=EXECUTED_REVIEW_NUMERIC_VALUES'
