#!/usr/bin/env python3
"""S5.1: seal qualified SM61 input and replay archived evidence headlessly.
No Cargo build, no NVCC, no mutation of source catalogue or S4.4 archive.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent
BASE = "d1929bea2a57beb82d466c9654c1d5f5d6ff616c"
PTX_INVENTORY_SHA = "619c8fc36501aa3e5268ebcde5976b8b4a50b7bbdd7b651cb8877ac4e7399805"
PRODUCTION = ["candle-core", "candle-kernels", "candle-nn",
              "dkr-kernel-registry", "tools/dkr-s31-builder"]
AUDIT = [
    ("active_recipe", "candle-core/src/cuda_backend/dkr_native_catalog.rs",
     "S31_FP4", "Fixed recipe selection; index validates descriptors against Rust."),
    ("active_recipe", "candle-core/src/cuda_backend/dkr_native_catalog.rs",
     "contract::mmq_tile", "MMQ launch geometry still delegates to Pascal formula."),
    ("active_recipe", "candle-core/src/cuda_backend/dkr_s31.rs",
     "recipe.fp4()", "Embedded descriptor is checked against compiled recipe."),
    ("active_recipe", "candle-kernels/dkr_fp8_recipe.rs",
     "pub(crate) const S31_FP8", "FP8 symbols/geometry remain in Rust."),
    ("active_recipe", "candle-kernels/dkr_fp4_recipe.rs",
     "auto_cc: 61", "NVFP4 Auto threshold and DP4A floor remain in Rust."),
    ("active_recipe", "candle-kernels/dkr_bf16_recipe.rs",
     "pub(crate) const S31_BF16", "BF16 widened SGEMM policy fixed in Rust."),
    ("active_recipe", "candle-kernels/dkr_routed_recipe.rs",
     "pub(crate) const S31_ROUTED", "MMVQ/MoE geometry fixed in Rust."),
    ("active_recipe", "candle-kernels/dkr_s31_contract.rs",
     "pub fn mmq_smem_bytes", "Pascal-specific shared-memory formula called at runtime."),
    ("build_adapter", "candle-core/src/cuda_backend/dkr_s31_validation.rs",
     "contract::is_sm61_ptx", "Qualified legacy package validation; filesystem path excluded in strict mode."),
    ("build_adapter", "candle-core/src/cuda_backend/dkr_native_catalog.rs",
     "cc != contract::CC", "Schema admits only qualified S31 exact target."),
    ("static_compatibility", "candle-core/src/cuda_backend/device.rs",
     "crate::quantized::fp4_recipe::S31_FP4", "Static recipe fallback remains in source; strict device requires catalogue."),
    ("static_compatibility", "candle-core/src/cuda_backend/device.rs",
     "legacy_static_matmul", "Static BF16 policy remains in source; unreachable when strict provider admitted."),
    ("static_compatibility", "candle-core/src/quantized/fast_mmq.rs",
     "di.cc >= 700", "Static FFI block excluded by strict feature."),
    ("optional_runtime", "candle-core/src/cuda_backend/cudnn.rs",
     "major >= 7", "Separate cuDNN failure policy; not qualified with cuDNN in S4.4."),
    ("historical_label", "candle-core/src/quantized/fp8_linear.rs",
     "haricot.fp8.sm61.f8e", "Historical operation label; not dispatch policy."),
    ("generic_admission", "candle-core/src/cuda_backend/dkr_s31.rs",
     "artifact.admit_exact_target", "Compare verified artifact target to device; retain."),
]
PATTERN = re.compile(r"sm_?61|pascal|\b(?:S31_[A-Z_]+|49152)\b|"
                     r"compute_capability|COMPUTE_CAPABILITY|"
                     r"(?:\bmajor|\bcc)\s*(?:[<>]=?|==|!=)\s*\d+|"
                     r"\b(?:auto_cc|min_dp4a_cc)\b", re.I)

def require(ok, message):
    if not ok:
        raise SystemExit("DKR_R124_S51_FAIL: " + message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def check_source():
    require(git("status", "--porcelain", "--untracked-files=no", "--", *PRODUCTION) == "",
            "tracked production changes")
    require(subprocess.run(["git", "diff", "--quiet", BASE, "HEAD", "--", *PRODUCTION],
                           cwd=ROOT).returncode == 0, "production differs from qualified S4.4")
    return git("rev-parse", "HEAD")

def entry_maps(family, data, entries):
    # S31 serializes Rust Option::None as JSON null for unrelated families.
    # The qualified validator requires maps only for their own family prefix.
    maps = {}
    symbols = set(entries)
    for prefix, key in (("mmq_", "mmq_entries"), ("moe_", "moe_entries")):
        value = data.get(key)
        if family.startswith(prefix):
            require(isinstance(value, dict) and bool(value)
                    and all(isinstance(symbol, str) and symbol in symbols
                            for symbol in value.values()),
                    "entry map " + family + "/" + key)
            maps[key] = value
        else:
            require(value is None, "unexpected entry map " + family + "/" + key)
    return maps

def self_test():
    entries = ["symbol"]
    require(entry_maps("affine", {"mmq_entries": None, "moe_entries": None}, entries) == {},
            "null map regression")
    require(entry_maps("affine", {}, entries) == {}, "absent map regression")
    for family, key in (("mmq_quantize", "mmq_entries"), ("moe_gguf", "moe_entries")):
        require(entry_maps(family, {key: {"entry": "symbol"}}, entries)
                == {key: {"entry": "symbol"}}, "valid map regression")
        for invalid in (None, {}, [], {"entry": "missing"}, {"entry": []}):
            try:
                entry_maps(family, {key: invalid}, entries)
            except SystemExit:
                pass
            else:
                require(False, "invalid map accepted")
    try:
        entry_maps("affine", {"mmq_entries": {"entry": "symbol"}}, entries)
    except SystemExit:
        pass
    else:
        require(False, "unrelated map accepted")
    print("DKR_R124_S51_SELF_TEST=PASS")

def catalogue(root):
    rows = {}
    for mf in sorted(root.glob("*/manifest.json")):
        family = mf.parent.name
        data = json.loads(mf.read_text())
        ptx = mf.parent / (family + ".ptx")
        text = ptx.read_text()
        entries = sorted(set(re.findall(r"\.entry\s+([^\s(]+)", text)))
        require(data["family"] == family and data["target_cc"] == 61, "manifest identity " + family)
        require(data["schema"] == "CANDLE-DKR-S31-NATIVE-MODULE-V1", "manifest schema " + family)
        require(data["base_revision"] == "d99f6d77e507688bd9e4b654fbe20ac9e18238fc", "manifest base " + family)
        require(re.findall(r"(?m)^\s*\.target\s+([^\s,]+)", text) == ["sm_61"], "PTX target " + family)
        digest = sha(ptx)
        require(digest == data["sha256"] and entries, "PTX digest/entries " + family)
        maps = entry_maps(family, data, entries)
        rows[family] = {"ptx_sha256": digest, "manifest_sha256": sha(mf),
                        "ptx_bytes": ptx.stat().st_size, "entry_count": len(entries),
                        "entry_names_sha256": canonical_sha(entries), "flags": data["flags"],
                        "entry_maps": maps}
    require(len(rows) == 25, "package count")
    require(canonical_sha({k:v["ptx_sha256"] for k,v in rows.items()}) == PTX_INVENTORY_SHA,
            "PTX inventory differs from S4.4 qualified log")
    index = root / "native-catalog.json"
    return {"packages": rows, "index_sha256": sha(index) if index.is_file() else None,
            "ptx_inventory_sha256": PTX_INVENTORY_SHA}

def audit():
    findings = []
    for kind, path, anchor, explanation in AUDIT:
        lines = (ROOT / path).read_text().splitlines()
        hits = [i for i,l in enumerate(lines,1) if anchor in l]
        require(hits, "audit anchor changed: " + path + " / " + anchor)
        findings.append({"kind": kind, "path": path, "lines": hits,
                         "anchor": anchor, "explanation": explanation})
    candidates = []
    paths = git("ls-files", "--", "candle-core/src", "candle-nn/src", "candle-kernels",
                "dkr-kernel-registry/src", "candle-core/build.rs").splitlines()
    for path in paths:
        if not path.endswith(".rs"):
            continue
        for i,line in enumerate((ROOT/path).read_text().splitlines(),1):
            if PATTERN.search(line):
                candidates.append({"path": path, "line": i, "text": line.strip()})
    return {"scope": "tracked Rust source; lexical candidates are not reachability proof",
            "neutrality": "OPEN", "findings": findings, "lexical_candidates": candidates}

def evidence(report):
    archive = report / "tests.tar.zst"
    require(archive.is_file() and not (report/"native-input").exists(), "archive/source-input state")
    files = {
        "missing-build-input.log": ["cuda-dkr-embedded requires DKR_EMBED_NATIVE_DIR at build time"],
        "embedded-digests.log": ["1 passed", "DKR_R124_S44_DIGEST family="],
        "embedded-tests.log": ["16 passed", "source=embedded manifests=25",
                               "DKR_R124_S42_BUILTIN_PTX=0", "DKR_R124_S43_STATIC_FFI=0"],
        "dkr_embedded_rejects_filesystem.log": ["test dkr_embedded_rejects_filesystem ... ok"],
        "dkr_embedded_rejects_directory_override.log": ["test dkr_embedded_rejects_directory_override ... ok"],
        "archive-build.log": ["CANDLE_DKR_EMBEDDED_REQUIRED=1 packages=25", "CANDLE_DKR_CUDA_BUILD=skipped"],
    }
    logs = {}
    for name, markers in files.items():
        p = report/name
        text = p.read_text(errors="replace")
        require(all(marker in text for marker in markers), "missing marker in " + name)
        logs[name] = sha(p)
    digests = dict(re.findall(r"DKR_R124_S44_DIGEST family=([a-z0-9_]+) sha256=([a-f0-9]{64})",
                             (report/"embedded-digests.log").read_text()))
    require(len(digests)==25 and canonical_sha(digests)==PTX_INVENTORY_SHA, "archived digest evidence")
    require(json.loads((report/"expected-digests.json").read_text())==digests, "stored expected inventory")
    return {"archive": str(archive), "archive_sha256": sha(archive), "logs": logs}

def seal(args):
    head = check_source()
    root = args.catalog.resolve(strict=True)
    report = args.s44_report.resolve(strict=True)
    out = args.out.resolve()
    require(not out.exists(), "seal directory already exists; choose a new one")
    cat = catalogue(root)
    ev = evidence(report)
    audit_data = audit()
    out.mkdir(parents=True, exist_ok=False)
    (out/"hardware-audit.json").write_text(json.dumps(audit_data, indent=2)+"\n")
    seal_data = {"schema":"CANDLE-DKR-R124-S51-SEAL-V1", "source_baseline":BASE,
                 "head":head, "catalogue":str(root), "s44_report":str(report),
                 "catalogue_inventory":cat, "evidence":ev,
                 "audit_sha256":sha(out/"hardware-audit.json"),
                 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/"seal.json").write_text(json.dumps(seal_data,indent=2,sort_keys=True)+"\n")
    print("DKR_R124_S51_PTX_SEAL=PASS packages=25")
    print("DKR_R124_S51_ARCHIVE_SHA256="+ev["archive_sha256"])
    print("DKR_R124_S51_HARDWARE_AUDIT=RECORDED neutrality=OPEN")
    print("DKR_R124_S51_SEAL="+str(out/"seal.json"))
    print("DKR_R124_S51_PREPARE=PASS")

def run_log(args, env, path):
    with path.open("w") as f:
        p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in p.stdout:
            print(line,end="",flush=True)
            f.write(line)
        require(p.wait()==0, "command failed: "+str(path))
    return path.read_text()

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "run through headless-relogin without display variables")
    check_source()
    seal_path=args.seal.resolve(strict=True)
    data=json.loads(seal_path.read_text())
    require(data["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1" and data["source_baseline"]==BASE,"seal schema")
    require(sha(seal_path.parent/"hardware-audit.json")==data["audit_sha256"],"audit changed")
    require(catalogue(Path(data["catalogue"]))==data["catalogue_inventory"],"catalogue changed")
    require(evidence(Path(data["s44_report"]))==data["evidence"],"S4.4 archive/evidence changed")
    env=os.environ.copy()
    for key in ("CUDA_COMPUTE_CAP","DKR_EMBED_NATIVE_DIR","CANDLE_DKR_S31_DIR",
                "MISTRALRS_DKR_ARTIFACT_SOURCE"):
        env.pop(key,None)
    env["CANDLE_DKR_TRACE"]="1"
    out=seal_path.parent/("headless-"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    gpu=run_log(["nvidia-smi","--query-gpu=index,name,compute_cap,driver_version,temperature.gpu,pstate,clocks.gr,clocks.mem,memory.used","--format=csv"],env,out/"gpu-before.log")
    require(any(", 6.1," in line for line in gpu.splitlines()), "SM61 device missing")
    base=["cargo","nextest","run","--archive-file",data["evidence"]["archive"],
          "--workspace-remap",str(ROOT),"--run-ignored","all","--no-capture","--no-fail-fast","-E"]
    samples=[]
    for i in range(3):
        text=run_log(base+["test(=dkr_catalog_device_sample)"],env,out/f"sample-{i+1}.log")
        require("test dkr_catalog_device_sample ... ok" in text and
                "source=embedded manifests=25" in text,"embedded replay failed")
        row={}
        for key,pattern in [
            ("admission_us",r"CANDLE_DKR_CATALOG_ADMISSION source=embedded elapsed_us=(\d+)"),
            ("device_init_us",r"DKR_R124_S44_DEVICE_INIT_US=(\d+)"),
            ("first_operation_us",r"DKR_R124_S44_FIRST_OPERATION_US=(\d+)")]:
            hits=re.findall(pattern,text)
            require(len(hits)==1,"missing timing "+key)
            row[key]=int(hits[0])
        samples.append(row)
    run_log(["nvidia-smi","--query-gpu=index,name,temperature.gpu,pstate,clocks.gr,clocks.mem,memory.used","--format=csv"],env,out/"gpu-after.log")
    result={"kind":"headless_smoke_not_cold_jit_benchmark","samples":samples,
            "median_us":{k:statistics.median(s[k] for s in samples) for k in samples[0]}}
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    print("DKR_R124_S51_HEADLESS_SMOKE=PASS")
    print("DKR_R124_S51_REPORT="+str(out/"result.json"))
    print("DKR_R124_S51_PASS neutrality=OPEN")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    s=sub.add_parser("seal")
    for name in ("catalog","s44-report","out"):
        s.add_argument("--"+name,type=Path,required=True)
    sub.add_parser("self-test")
    h=sub.add_parser("headless")
    h.add_argument("--seal",type=Path,required=True)
    args=p.parse_args()
    if args.mode == "self-test":
        self_test()
    else:
        (seal if args.mode=="seal" else headless)(args)

if __name__=="__main__":
    main()
