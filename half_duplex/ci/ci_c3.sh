#!/usr/bin/env bash
# Run after the true C2 A/B of a C1 report. No private audio.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
report="${1:?Expected real --probes C1 report path}"
python3 -m unittest discover -s tests/hdb2c -p 'test_c3.py' -v
"${HDB2C_PYTHON:?Missing isolated NeMo venv}" -m hdb2c.c3 "$report"
test -s "$report/c3_layers_cpu.json"
"$HDB2C_PYTHON" - "$report/c3_layers_cpu.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert len(r['stages'])==26
assert r['mel_input']=='EXACT_SHARED_NEMO_F32LE_VERIFIED'
assert r['status'] in ('FIRST_DIVERGENCE_LOCALIZED','ALL_STAGES_NUMERIC_PARITY')
print('HDB2C3_DIAGNOSTIC=COMPLETE_NOT_AUTOMATIC_NUMERIC_PARITY')
print('HDB2C3_FIRST_DIVERGENCE='+str(r['first_divergence']))
PY