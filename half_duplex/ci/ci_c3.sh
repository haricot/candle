#!/usr/bin/env bash
# Run after the true C2 A/B of a C1 report. No private audio.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
report="${1:?Expected real --probes C1 report path}"
python3 -m unittest discover -s tests/hdb2c -p 'test_c3.py' -v
"${HDB2C_PYTHON:?Missing isolated NeMo venv}" -m hdb2c.c3 "$report"
test -s "$report/c3_layers_cpu.json"
test -s "$report/c3_dw_cpu.json"
test -s "$report/c3_mask_ab_cpu.json"
"$HDB2C_PYTHON" - "$report/c3_layers_cpu.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert len(r['stages'])==26
assert r['mel_input']=='EXACT_SHARED_NEMO_F32LE_VERIFIED'
assert r['status'] in ('FIRST_DIVERGENCE_LOCALIZED','ALL_STAGES_NUMERIC_PARITY')
print('HDB2C3_DIAGNOSTIC=COMPLETE_NOT_AUTOMATIC_NUMERIC_PARITY')
print('HDB2C3_FIRST_DIVERGENCE='+str(r['first_divergence']))
d=json.load(open(sys.argv[1].replace('c3_layers_cpu.json','c3_dw_cpu.json')))
assert len(d['weights'])==12 and len(d['operations'])==9
assert d['status'] in ('ALL_DW_STAGES_NUMERIC_PARITY','FIRST_DW_DIVERGENCE_LOCALIZED')
print('HDB2C3_R3_FIRST_WEIGHT='+str(d['first_weight_mismatch']))
print('HDB2C3_R3_FIRST_OPERATION='+str(d['first_operation_divergence']))
a=json.load(open(sys.argv[1].replace('c3_layers_cpu.json','c3_mask_ab_cpu.json')))
assert len(a['operations_ab'])==9
assert a['status'] in ('ALL_DW_NUMERIC_PARITY','DW_NUMERIC_PARITY_NOT_ESTABLISHED')
assert a['dw_mask_propagation']=='NEMO_MASKED_CONV_SEQUENTIAL_ISOLATED_CPU'
print('HDB2C3_R3C='+a['status'])
print('HDB2C3_R3C_FIRST_MASKED='+str(a['first_operation_masked']))
print('HDB2C3_R3C_PRE_ENCODE='+a['pre_encode_status'])
print('HDB2C3_R3C_FULL_ENCODER='+a['full_encoder_status'])
PY