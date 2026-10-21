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
if [[ "${HDB2C_R7_FOCUSED:-0}" != 1 && "${HDB2C_R8_CORPUS:-0}" != 1 ]];then
  ./run-hdb2b.sh build cpu
fi
"$HDB2C_PYTHON" ci/cameo_subset.py --out "$RUNNER_TEMP/cameo" --per-emotion 1
if [[ "${HDB2C_R8_CORPUS:-0}" == 1 ]];then
  "$HDB2C_PYTHON" -m hdb2c.c3_r8 --corpus "$RUNNER_TEMP/cameo/manifest.json" --require-numeric
  exit
fi
mapfile -t clips < <("$HDB2C_PYTHON" - "$RUNNER_TEMP/cameo/manifest.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))["rows"]
for x in rows:
    if x["split"]=="cafe" and x["emotion"] in ("neutral","sadness"):
        print(x["wav"]+"|"+x["reference_file"])
PY
)
test "${#clips[@]}" -eq 2
if [[ "${HDB2C_R7_FOCUSED:-0}" == 1 ]];then
  "$HDB2C_PYTHON" -m hdb2c.c3_r7_focused --corpus "$RUNNER_TEMP/cameo/manifest.json"
  exit
fi
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
  # C3-A: the real NeMo log-Mel is injected bit-identically into the Rust encoder.
  # Detect the FIRST mismatching encoder stage; keep raw activations local.
  bash ci/ci_c3.sh "$report"
  cp "$report/c3_layers_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c3_layers_$i.json"
  cp "$report/c3_dw_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c3_dw_$i.json"
  cp "$report/c3_mask_ab_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c3_mask_ab_$i.json"
  cp "$report/c3_block0_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c3_block0_$i.json"
  # R5: recursively verified NeMo eval + native f32 positional encoding A/B.
  # Operator fixtures stay separate from uninjected full-encoder inference.
  "$HDB2C_PYTHON" -m hdb2c.c3_r5 "$report"
  cp "$report/c3_positional_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c3_positional_$i.json"
  "$HDB2C_PYTHON" - "$report/c3_positional_cpu.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['revision']=='HDB2-C3-A-R5'
assert len(r['operator_cases'])==17
assert r['status'] in ('POSITIONAL_NUMERIC_PARITY','POSITIONAL_NUMERIC_MISMATCH')
assert r['positional_exact_parity'] in ('PASS_BITWISE','NOT_ESTABLISHED')
for key in ('c1_eval_state','nemo_eval_state'):
    assert r[key]['status']=='ALL_MODULES_EVAL_VERIFIED'
    assert r[key]['after_eval']['training_module_count']==0
assert r['r4_r5_pre_encode_control']['bitwise_equal']
assert r['reference_bit_reproducible']
assert r['encoder_positional_injection']=='NONE_NATIVE_COMPUTATION_ONLY'
assert len(r['encoder_stages'])==26 and len(r['block0_stages'])==14
print('HDB2C3_R5_DIAGNOSTIC=COMPLETE_REVIEW_NUMERIC_AND_BITWISE_GATES')
PY
  "$HDB2C_PYTHON" -m hdb2c.c3_r6 "$report"
  cp "$report/c3_padding_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c3_padding_$i.json"
  "$HDB2C_PYTHON" - "$report/c3_padding_cpu.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['revision']=='HDB2-C3-A-R6'
assert [x['mode'] for x in r['modes']]==['none','attention','convolution','both']
assert r['modes'][0]['off_control']['status']=='PASS_BITWISE_ALL_FRAMES'
assert r['modes'][0]['off_control']['stages_checked']==26
for row in r['modes']:
    assert row['mask_bits']=='EXACT_NEMO_PAD_AND_ATTENTION_MASKS'
    assert len(row['encoder_stages'])==26 and len(row['block0_stages'])==14
    if r['padded_encoder_frames']==0:
        assert row['no_padding_control']['status']=='PASS_BITWISE_ALL_FRAMES'
assert r['encoder_internal_activation_or_positional_injection']=='NONE'
print('HDB2C3_R6_DIAGNOSTIC=COMPLETE_2X2_MASK_ABLATION')
PY
  # R7: actual production NeMo TDT vs an independent native PCM path.
  # Byte equality is reported separately from numerical/discrete parity.
  "$HDB2C_PYTHON" -m hdb2c.c3_r7 "$report"
  cp "$report/c3_tdt_cpu.json" "$GITHUB_WORKSPACE/halfduplex-ci-published/c3_tdt_$i.json"
  "$HDB2C_PYTHON" - "$report/c3_tdt_cpu.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['revision']=='HDB2-C3-A-R7'
assert r['reference']['decoder_replaced'] is False
assert r['encoder_r6_control']['status']=='PASS_BITWISE_ALL_FRAMES'
assert r['native_pipeline_internal_tensor_injection']=='NONE'
assert r['tdt']['native_pcm']['scope']=='NATIVE_PCM_FRONTEND_ENCODER_TDT'
assert r['common_input_operators']['shared_inputs']=='BIT_IDENTICAL_STATE_ENCODER_AND_PREDICTOR_INPUTS'
assert r['global_bitwise_status'] == ('PASS_BITWISE' if r['global_bitwise_equal'] else 'FAIL_DIFFERENT_F32_BITS')
print('HDB2C3_R7_DIAGNOSTIC=COMPLETE_REVIEW_TDT_AND_STRICT_BITS_GATES')
PY
done
echo 'HALF_DUPLEX_REAL_CPU=EXECUTED_REVIEW_NUMERIC_VALUES'
