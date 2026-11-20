#!/usr/bin/env python3
"""R1.24 S4.4: archive once, then test without the source catalogue.

Uses a private copy, never renames or deletes the supplied catalogue.
Requires cargo-nextest with archive support, Python 3, CUDA SM61.
Reports and archive are retained in target/dkr-r124-s44-*.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent

def require(condition, message):
    if not condition:
        raise SystemExit("DKR_R124_S44_FAIL: " + message)

def run(args, env, log, expect_failure=None):
    print("+ " + " ".join(map(str, args)), flush=True)
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            out.write(line)
        rc = proc.wait()
    result = log.read_text(encoding="utf-8")
    if expect_failure:
        require(rc != 0 and expect_failure in result, f"wrong failure or unexpected success: {log}")
    else:
        require(rc == 0, f"exit {rc}: {log}")
    return result

def inventory(root):
    rows = {}
    for manifest in sorted(root.glob("*/manifest.json")):
        family = manifest.parent.name
        data = json.loads(manifest.read_text())
        require(data["family"] == family, f"wrong manifest family {family}")
        require(data["target_cc"] == 61, f"wrong target {family}")
        ptx = manifest.parent / (family + ".ptx")
        digest = hashlib.sha256(ptx.read_bytes()).hexdigest()
        require(digest == data["sha256"], f"manifest digest mismatch {family}")
        rows[family] = digest
    require(len(rows) == 25, "expected 25 source packages")
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--skip-filesystem-sample", action="store_true")
    args = parser.parse_args()
    source = args.catalog.resolve(strict=True)
    require(source.is_dir(), "catalogue must be a directory")
    expected = inventory(source)
    target = ROOT / "target"
    target.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dkr-r124-s44-", dir=target))
    print(f"DKR_R124_S44_REPORT_DIR={work}", flush=True)
    stage = work / "native-input"
    shutil.copytree(source, stage, symlinks=False)
    require(inventory(stage) == expected, "private copy differs from original")
    (work / "expected-digests.json").write_text(json.dumps(expected, indent=2, sort_keys=True))
    env = os.environ.copy()
    for key in ("CUDA_COMPUTE_CAP", "DKR_EMBED_NATIVE_DIR",
                "CANDLE_DKR_S31_DIR", "MISTRALRS_DKR_ARTIFACT_SOURCE"):
        env.pop(key, None)
    env["CANDLE_DKR_TRACE"] = "1"
    run(["cargo", "check", "-p", "candle-core", "--no-default-features",
         "--features", "cuda-dkr-embedded"], env, work / "missing-build-input.log",
        expect_failure="cuda-dkr-embedded requires DKR_EMBED_NATIVE_DIR at build time")
    print("DKR_R124_S44_MISSING_BUILD_INPUT=PASS", flush=True)
    if not args.skip_filesystem_sample:
        fsenv = dict(env, CANDLE_DKR_S31_DIR=str(stage),
                     MISTRALRS_DKR_ARTIFACT_SOURCE="filesystem")
        run(["cargo", "nextest", "run", "-p", "candle-core", "--no-default-features",
             "--features", "cuda-dkr-catalog", "--test", "dkr_catalog_mode",
             "--run-ignored", "all", "-E", "test(dkr_catalog_device_sample)",
             "--no-capture"], fsenv, work / "filesystem-sample.log")
    archive = work / "tests.tar.zst"
    buildenv = dict(env, DKR_EMBED_NATIVE_DIR=str(stage))
    run(["cargo", "nextest", "archive", "-p", "candle-core", "-p", "candle-nn",
         "--no-default-features", "--features", "cuda-dkr-embedded", "--lib",
         "--test", "dkr_catalog_mode", "--test", "cuda_legacy_bf16_tests",
         "--test", "fp8_sm61_f8_tests", "--test", "fp4_e7c_contract",
         "--test", "quantized_tests", "--test", "moe_simt_sm61",
         "--archive-file", str(archive)], buildenv, work / "archive-build.log")
    # Only remove the private copy; no Cargo command that compiles after this.
    shutil.rmtree(stage)
    require(not stage.exists(), "embedded build inputs still present")
    print("DKR_R124_S44_PRIVATE_INPUT_REMOVED=1", flush=True)
    require(archive.is_file(), "archive missing")
    def archived(label, expr, runtime_env):
        return run(["cargo", "nextest", "run", "--archive-file", str(archive),
                    "--workspace-remap", str(ROOT), "--run-ignored", "all",
                    "--no-fail-fast", "--no-capture", "--test-threads", "1", "-E", expr],
                   runtime_env, work / (label + ".log"))
    result = archived("embedded-digests",
            "test(dkr_embedded_catalog_digest_inventory)", env)
    entries = re.findall(r"DKR_R124_S44_DIGEST family=([a-z0-9_]+) sha256=([a-f0-9]{64})", result)
    require(len(entries) == 25 and dict(entries) == expected,
            "embedded payload hashes differ from source")
    print("DKR_R124_S44_DIGEST_PARITY=PASS packages=25", flush=True)
    tests = [
        "dkr_catalog_builtin_ptx_omitted", "dkr_catalog_static_ffi_omitted",
        "dkr_catalog_device_sample", "dkr_routed_recipe_embedded_table",
        "dkr_bf16_recipe_embedded_table", "dkr_fp4_recipe_embedded_table",
        "dkr_fp8_recipe_embedded_table",
        "cuda_legacy_bf16_matmul_exponent_range_and_views",
        "f8_sm61_linear_simt_dp4a_cuda", "expanded_f16_preserves_f32_output_range",
        "dkr_fp4_auto_matches_selected_route_cuda", "dkr_routed_mmvq_cuda_batch_and_dtype",
        "dkr_catalog_mmq_cuda_batch_and_dtype", "dkr_routed_gguf_moe_cuda",
        "sm61_simt_decode_weighted_fp16", "sm61_simt_prefill_topk2_fp16",
    ]
    gpu = archived("embedded-tests", " | ".join(f"test({t})" for t in tests), env)
    # Unit test names contain Rust module paths; filter by unique suffix and
    # require evidence of every selected test actually completing.
    for name in tests:
        require(re.search(r"test (?:[^\\s]+::)?" + re.escape(name) + r" \\.\\.\\. ok", gpu)
                is not None, f"missing successful test: {name}")
    require("CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25" in gpu,
            "missing embedded runtime provenance")
    require("CANDLE_DKR_NATIVE_CATALOG source=filesystem" not in gpu,
            "filesystem runtime unexpectedly selected")
    require("DKR_R124_S43_STATIC_FFI=0" in gpu and "DKR_R124_S42_BUILTIN_PTX=0" in gpu,
            "static producer markers missing")
    for family in ("cast", "quantized", "mmvq_gguf", "mmq_quantize",
                   "mmq_instance_q8_0", "moe_gguf", "moe_simt_f16"):
        require(f"CANDLE_DKR_MODULE_LOADED family={family} target=sm61" in gpu,
                f"missing GPU module provenance: {family}")
    for name, override in [
        ("dkr_embedded_rejects_filesystem", {"MISTRALRS_DKR_ARTIFACT_SOURCE": "filesystem"}),
        ("dkr_embedded_rejects_directory_override", {"CANDLE_DKR_S31_DIR": str(source)}),
    ]:
        result = archived(name, f"test(={name})", dict(env, **override))
        require(f"test {name} ... ok" in result, f"missing runtime rejection test: {name}")
    print("DKR_R124_S44_RUNTIME_OVERRIDES=REJECTED", flush=True)
    require(inventory(source) == expected, "original catalogue changed")
    print("DKR_R124_S44_ORIGINAL_CATALOG_UNCHANGED=1", flush=True)
    print("DKR_R124_S44_PASS", flush=True)

if __name__ == "__main__":
    main()
