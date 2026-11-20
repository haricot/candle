#!/usr/bin/env python3
"""Verify resolved Cargo feature closure for the decoupled and hybrid modes."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEGACY = {"cuda-legacy-bf16", "cuda-legacy-fp8", "cuda-legacy-fp4"}

def resolved(package, feature):
    result = subprocess.run(
        ["cargo", "tree", "-p", package, "--no-default-features",
         "--features", feature, "-e", "normal", "--prefix", "none",
         "--format", "{p}|{f}"],
        cwd=ROOT, text=True, check=True, stdout=subprocess.PIPE,
    )
    found = {}
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        package_text, fields = line.split("|", 1)
        name = package_text.split()[0]
        fields = fields.removesuffix(" (*)")
        found.setdefault(name, set()).update(f for f in fields.split(",") if f)
    return found

def require(ok, message):
    if not ok:
        raise SystemExit("DKR_R124_FEATURE_AUDIT_FAIL: " + message)

for package in ("candle-core", "candle-nn"):
    for mode in ("cuda-dkr-catalog", "cuda-dkr-s31"):
        graph = resolved(package, mode)
        for name in ("candle-core", "candle-kernels", package):
            require(name in graph, f"{package}/{mode}: {name} missing")
        core = graph["candle-core"]
        kernels = graph["candle-kernels"]
        require({"cuda", "cuda-dkr-catalog"} <= core,
                f"{package}/{mode}: catalogue runtime missing")
        if mode == "cuda-dkr-catalog":
            require(not (LEGACY & core or LEGACY & kernels),
                    f"{package}/{mode}: legacy flags leaked")
            require("cuda-dkr-s31" not in core,
                    f"{package}/{mode}: hybrid feature leaked")
        else:
            require(LEGACY <= core and LEGACY <= kernels and mode in core,
                    f"{package}/{mode}: hybrid compatibility changed")
        print(f"DKR_R124_FEATURE_AUDIT_PASS package={package} mode={mode} "
              f"core={','.join(sorted(core))} kernels={','.join(sorted(kernels))}")
print("DKR_R124_S41_FEATURE_GRAPH=PASS")
