#!/usr/bin/env python3
"""Reproducible source, CPU, hosted-CUDA and operator-run GPU checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "legacy-dispatch/sources.json").read_text())

def run(args):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)

def source_check():
    run(["git", "merge-base", "--is-ancestor", META["base"], "HEAD"])
    changed = subprocess.check_output(["git", "diff", "--name-only", META["base"]], cwd=ROOT, text=True).splitlines()
    assert set(changed) <= set(META["payload_sha256"]) | {"legacy-dispatch/sources.json"}, "unexpected source delta"
    for name, digest in META["payload_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print("CANDLE_LEGACY_DISPATCH_SOURCES=PASS", flush=True)

def ptx_check():
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    bindings = sorted(target.glob("**/build/candle-kernels-*/out/ptx.rs"))
    if not bindings:
        raise RuntimeError("No generated Candle PTX found")
    required = {"unary": ["ucopy_bf16"]}
    if "bf16" in META["formats"]:
        required.update({"cast": ["cast_bf16_f32", "cast_f32_bf16"], "binary": ["badd_bf16"], "reduce": ["rmsnorm_bf16"]})
    if "fp8" in META["formats"]:
        required.setdefault("quantized", []).extend(["fp8_sm61_w8a16_gemv", "fp8_sm61_w8a16_prefill4", "fp8_sm61_w8abf16_gemv", "fp8_sm61_w8abf16_prefill4"])
    if "fp4" in META["formats"]:
        required.setdefault("quantized", []).extend(["nvfp4_arch_e6_w4a16_scalar", "indexed_moe_forward_mxfp4_q8_1"])
    for binding in bindings:
        for module, symbols in required.items():
            path = binding.parent / (module + ".ptx")
            text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
            for symbol in symbols:
                if not re.search(r"\.entry\s+" + re.escape(symbol) + r"\s*\(", text):
                    raise RuntimeError(f"Missing CUDA entry {symbol} in {path}")
        print(f"CANDLE_LEGACY_DISPATCH_PTX=PASS: {binding.parent}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["sources", "cpu", "cuda-compile", "gpu", "ptx"])
    phase = parser.parse_args().phase
    source_check()
    if phase == "sources": return
    if phase == "ptx": ptx_check(); return
    shutil.copyfile(ROOT / "legacy-dispatch/Cargo.lock", ROOT / "Cargo.lock")
    run(["rustc", "--version"])
    cargo = ["cargo", "test", "--locked", "-p", "candle-core", "--no-default-features"]
    targets = [arg for target in META["tests"] for arg in ["--test", target]]
    if phase == "cpu":
        run(["cargo", "check", "--locked", "-p", "candle-core", "-p", "candle-nn", "--no-default-features"])
        run(cargo + ["--lib"] + targets)
    else:
        features = ",".join(META["features"])
        if phase == "cuda-compile":
            run(["nvcc", "--version"])
            run(["cargo", "check", "--locked", "-p", "candle-core", "-p", "candle-nn", "--no-default-features", "--features", features])
            run(cargo + ["--features", features] + targets + ["--no-run"])
            ptx_check()
        else:
            # Explicit operator-run phase. Hosted CI never takes this branch.
            run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv"])
            run(cargo + ["--release", "--features", features] + targets + ["--", "--nocapture", "--test-threads=1"])
            for target, name in META["physical_ignored_tests"]:
                run(cargo + ["--release", "--features", features, "--test", target, name, "--", "--ignored", "--nocapture", "--test-threads=1"])
    print("CANDLE_LEGACY_DISPATCH_" + phase.upper().replace("-", "_") + "=PASS", flush=True)

if __name__ == "__main__": main()
