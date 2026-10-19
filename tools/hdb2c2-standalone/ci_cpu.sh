#!/usr/bin/env bash
# CPU only. Never upload private WAVs, model weights, binaries or raw tensors.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PIN_PARAKEET=53c0d3065f9dd8de0356b25391e60ed3a471192b
EXPECTED_ZIP_SHA256=9eac6aa6ca7f1ed567849aceb9a27baaebc4a7d78d98f235ca101f05076bedb9
BUNDLE="$ROOT/tools/hdb2c2-standalone/haricot-half-duplex-hdb2c-c2-standalone-v0.6.6.zip"
OUT="$ROOT/hdb2c2-ci-output"
mkdir -p "$OUT"
case "${1:-smoke}" in
  smoke)
    # Isolate the nested Cargo project from the haricot/candle workspace.
    OUT="${RUNNER_TEMP:?RUNNER_TEMP required on GitHub Actions}/hdb2c2-upstream"
    mkdir -p "$OUT"
    python3 -m unittest discover -s "$ROOT/tools/hdb2c2-standalone" -p 'test_cameo_subset.py' -v
    echo 'HDB2C2_STAGE=UPSTREAM_BASELINE_SMOKE_NOT_C2_QUALIFIED'
    git clone -q https://github.com/n0madic/parakeet-candle.git "$OUT/pinned-parakeet"
    git -C "$OUT/pinned-parakeet" checkout -q --detach "$PIN_PARAKEET"
    test "$(git -C "$OUT/pinned-parakeet" rev-parse HEAD)" = "$PIN_PARAKEET"
    cargo build --locked --release --manifest-path "$OUT/pinned-parakeet/Cargo.toml" --bin parakeet-candle
    cargo test --locked --release --manifest-path "$OUT/pinned-parakeet/Cargo.toml" --lib
    echo 'HDB2C2_STAGE=UPSTREAM_SMOKE_PASS_NOT_C2_QUALIFIED'
    ;;
  qualify)
    test -f "$BUNDLE" || { echo 'HDB2C2_NOT_QUALIFIED=MISSING_SOURCE_ZIP' >&2; exit 5; }
    echo "$EXPECTED_ZIP_SHA256  $BUNDLE" | sha256sum -c -
    mkdir -p "$OUT/package" "$OUT/models/source" "$OUT/reports"
    unzip -q "$BUNDLE" -d "$OUT/package"
    PROJECT="$OUT/package/haricot-half-duplex-hdb2c-v0.6.6"
    test -x "$PROJECT/install_update.sh"
    export HDB2A_RUNTIME_DIR="$OUT/runtime-parakeet"
    export HDB2A_MODEL_DIR="$OUT/models/transcribepsy-candle"
    export HDB2A_REPORT_ROOT="$OUT/reports"
    export HDB2C_ENV_DIR="$OUT/venv-hdb2c-py313"
    export HDB2C_PYTHON="$HDB2C_ENV_DIR/bin/python"
    export HDB2C_NEMO_PYTHON="$HDB2C_PYTHON"
    export HDB2C_NEMO="$OUT/models/source/original.nemo"
    cd "$PROJECT"
    python3 -m unittest discover -s tests/hdb2c -v
    ./run-hdb2c.sh install-nemo
    export PATH="$HDB2C_ENV_DIR/bin:$PATH"
    hf download qvac/TranscribePsy-250M-EU-sub --include '*.nemo' --local-dir "$OUT/models/source"
    mapfile -d '' checkpoints < <(find "$OUT/models/source" -type f -name '*.nemo' -print0)
    test "${#checkpoints[@]}" -eq 1 || { echo 'HDB2C2_NOT_QUALIFIED=CHECKPOINT_DISCOVERY' >&2; exit 6; }
    ln -s "${checkpoints[0]}" "$HDB2C_NEMO"
    ./run-hdb2a.sh bootstrap
    ./run-hdb2a.sh prepare-model "$HDB2C_NEMO"
    ./run-hdb2a.sh check-model-config
    python3 scripts/hdb2a_cargo_lock.py --manifest-path "$HDB2A_RUNTIME_DIR/Cargo.toml"
    CARGO_TARGET_DIR="$HDB2A_RUNTIME_DIR/target/hdb2b-cpu" cargo build --release --manifest-path "$HDB2A_RUNTIME_DIR/Cargo.toml" --bin parakeet-candle
    # Dataset requires no user recordings; each sample has a real transcript.
    uv pip install --python "$HDB2C_PYTHON" 'datasets==3.6.0' 'soundfile'
    "$HDB2C_PYTHON" "$ROOT/tools/hdb2c2-standalone/cameo_subset.py" --out "$OUT/cameo" --per-emotion 1
    "$HDB2C_PYTHON" - "$OUT/cameo/manifest.json" "$OUT" "$PROJECT" <<'PY'
import json, pathlib, subprocess, sys
manifest=pathlib.Path(sys.argv[1])
out=pathlib.Path(sys.argv[2])
project=pathlib.Path(sys.argv[3])
records=json.loads(manifest.read_text())["rows"]
reports=[]
for i,sample in enumerate(records):
    cmd=[str(project/"run-hdb2c.sh"),"c1",sample["wav"],"--variant","cpu",
         "--reference-file",sample["reference_file"]]
    if sample["emotion"]=="neutral":cmd.append("--probes")
    run=subprocess.run(cmd,check=True,text=True,capture_output=True,cwd=project)
    print(run.stdout,flush=True)
    marker=[line.split("=",1)[1] for line in run.stdout.splitlines() if line.startswith("HDB2C_REPORT_DIR=")]
    if len(marker)!=1:raise RuntimeError("C1 did not return one report directory")
    report=pathlib.Path(marker[0])
    c1=json.loads((report/"c1_comparison.json").read_text())
    result={"split":sample["split"],"emotion":sample["emotion"],
            "source_pcm_sha256":sample["pcm_sha256"],
            "dataset_id":sample["id"],"c1":c1}
    if sample["emotion"]=="neutral":
        c2=subprocess.run([str(project/"install_update.sh"),"--report",str(report)],check=True,cwd=project,
                          text=True,capture_output=True)
        print(c2.stdout,flush=True)
        result["c2"]=json.loads((report/"c2_ab_cpu.json").read_text())
        result["c2c3"]=json.loads((report/"c2c3_exact_cpu.json").read_text())
    reports.append(result)
(out/"published").mkdir(exist_ok=True)
(out/"published"/"cameo_results.json").write_text(json.dumps(
    {"status":"REAL_OBSERVATIONS_PARITY_NOT_ASSUMED","dataset":"amu-cai/CAMEO","results":reports},
    indent=2,ensure_ascii=False)+"\\n")
PY
    test -s "$OUT/published/cameo_results.json"
    echo 'HDB2C2_STAGE=REAL_CPU_COMPARISON_REVIEW_NUMERIC_STATUS'
    ;;
  *) echo "Usage: $0 smoke|qualify" >&2; exit 2 ;;
esac
