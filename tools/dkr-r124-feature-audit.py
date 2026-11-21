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
    for mode in ("cuda-dkr-catalog", "cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq", "cuda-dkr-s31", "cuda"):
        graph = resolved(package, mode)
        for name in ("candle-core", "candle-kernels", package):
            require(name in graph, f"{package}/{mode}: {name} missing")
        core = graph["candle-core"]
        kernels = graph["candle-kernels"]
        require("cuda" in core, f"{package}/{mode}: CUDA runtime missing")
        if mode == "cuda":
            require("cuda-dkr-catalog" not in core and "cuda-dkr-catalog" not in kernels
                    and not (LEGACY & core or LEGACY & kernels),
                    f"{package}/{mode}: plain CUDA unexpectedly selects DKR or legacy flags")
        else:
            require("cuda-dkr-catalog" in core and "cuda-dkr-catalog" in kernels,
                    f"{package}/{mode}: catalogue runtime/build selection missing")
        if mode in ("cuda-dkr-catalog", "cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq"):
            require(("cuda-dkr-embedded" in core) == (mode in ("cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq"))
                    and ("cuda-dkr-embedded" in kernels) == (mode in ("cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq")),
                    f"{package}/{mode}: embedded feature propagation mismatch")
            require(not (LEGACY & core or LEGACY & kernels),
                    f"{package}/{mode}: legacy flags leaked")
            require("cuda-dkr-s31" not in core,
                    f"{package}/{mode}: hybrid feature leaked")
            require(("cuda-dkr-cubin-embedded" in core) == (mode == "cuda-dkr-cubin-embedded"),
                    f"{package}/{mode}: experimental CUBIN feature propagation mismatch")
            require(("cuda-dkr-neutral-mmq" in core) == (mode == "cuda-dkr-neutral-mmq"),
                    f"{package}/{mode}: neutral launch table feature propagation mismatch")
            require("cuda-dkr-neutral-mmq" not in kernels,
                    f"{package}/{mode}: neutral recipe must not activate static producer")
            require("cuda-dkr-cubin-embedded" not in kernels,
                    f"{package}/{mode}: legacy/static CUDA producer must not gain CUBIN feature")
        elif mode == "cuda-dkr-s31":
            require(LEGACY <= core and LEGACY <= kernels and mode in core,
                    f"{package}/{mode}: hybrid compatibility changed")
        print(f"DKR_R124_FEATURE_AUDIT_PASS package={package} mode={mode} "
              f"core={','.join(sorted(core))} kernels={','.join(sorted(kernels))}")
print("DKR_R124_S41_FEATURE_GRAPH=PASS")
print("DKR_R124_S42_FEATURE_GRAPH=PASS")
print("DKR_R124_S44_FEATURE_GRAPH=PASS")
print("DKR_R124_S54B_FEATURE_GRAPH=PASS")
print("DKR_R124_S55A_FEATURE_GRAPH=PASS")
