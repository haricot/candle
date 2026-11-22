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
    for mode in ("cuda-dkr-catalog", "cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq", "cuda-dkr-neutral-recipes", "cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default", "cuda-dkr-s31", "cuda"):
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
        if mode in ("cuda-dkr-catalog", "cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq", "cuda-dkr-neutral-recipes", "cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default"):
            require(("cuda-dkr-embedded" in core) == (mode in ("cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq", "cuda-dkr-neutral-recipes", "cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default"))
                    and ("cuda-dkr-embedded" in kernels) == (mode in ("cuda-dkr-embedded", "cuda-dkr-cubin-embedded", "cuda-dkr-neutral-mmq", "cuda-dkr-neutral-recipes", "cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default")),
                    f"{package}/{mode}: embedded feature propagation mismatch")
            require(not (LEGACY & core or LEGACY & kernels),
                    f"{package}/{mode}: legacy flags leaked")
            require("cuda-dkr-s31" not in core,
                    f"{package}/{mode}: hybrid feature leaked")
            require(("cuda-dkr-cubin-embedded" in core) == (mode in ("cuda-dkr-cubin-embedded", "cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default")),
                    f"{package}/{mode}: experimental CUBIN feature propagation mismatch")
            require(("cuda-dkr-neutral-mmq" in core) == (mode in ("cuda-dkr-neutral-mmq", "cuda-dkr-neutral-recipes", "cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default")),
                    f"{package}/{mode}: neutral launch table feature propagation mismatch")
            require(("cuda-dkr-neutral-recipes" in core) == (mode in ("cuda-dkr-neutral-recipes", "cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default")),
                    f"{package}/{mode}: typed recipe feature propagation mismatch")
            require(("cuda-dkr-cudaforge-catalog" in core) == (mode in ("cuda-dkr-cudaforge-catalog", "cuda-dkr-qualified-default")),
                    f"{package}/{mode}: Cudaforge catalog feature closure mismatch")
            require(("cuda-dkr-qualified-default" in core) == (mode == "cuda-dkr-qualified-default"),
                    f"{package}/{mode}: qualified-default feature closure mismatch")
            require("cuda-dkr-qualified-default" not in kernels,
                    f"{package}/{mode}: qualified-default must not compile kernels")
            require("cuda-dkr-cudaforge-catalog" not in kernels,
                    f"{package}/{mode}: Cudaforge producer must not become a kernel dependency")
            require("cuda-dkr-neutral-recipes" not in kernels,
                    f"{package}/{mode}: producer recipe must not activate static kernels")
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
print("DKR_R124_S55B_FEATURE_GRAPH=PASS")
print("DKR_R124_S55C_FEATURE_GRAPH=PASS")

# S5.5d: the model probe links candle-transformers -> candle-nn.
# Propagate DKR features explicitly; otherwise "cuda" alone reactivates
# candle-nn's static-FFI imports while candle-kernels deliberately skips FFI.
for probe_mode,nn_feature in (
    ("ptx","cuda-dkr-embedded"),
    ("cubin","cuda-dkr-cudaforge-catalog"),
    ("promoted","cuda-dkr-qualified-default"),
):
    graph=resolved("dkr-s55d-model-probe",probe_mode)
    core=graph.get("candle-core",set())
    nn=graph.get("candle-nn",set())
    kernels=graph.get("candle-kernels",set())
    require(nn_feature in core and nn_feature in nn,
            f"S5.5d model probe {probe_mode}: candle-core/candle-nn feature mismatch")
    require("cuda-dkr-s31" not in core and "cuda-dkr-s31" not in nn,
            f"S5.5d model probe {probe_mode}: static-FFI hybrid feature leaked")
    require(not (LEGACY & core or LEGACY & kernels),
            f"S5.5d model probe {probe_mode}: static CUDA flags leaked")
    print(f"DKR_R124_S55D_MODEL_FEATURE_AUDIT_PASS mode={probe_mode} nn={nn_feature}")
print("DKR_R124_S55D_FEATURE_GRAPH=PASS")

print("DKR_R125_P1_FEATURE_GRAPH=PASS")
