#!/usr/bin/env python3
"""S5.2 sealed native SM61 numeric qualification (no NVCC)."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent.parent
SOURCE_BASE="d1929bea2a57beb82d466c9654c1d5f5d6ff616c"
PRODUCTION=["candle-core/src","candle-core/build.rs","candle-core/Cargo.toml",
            "candle-kernels","candle-nn/src","candle-nn/Cargo.toml",
            "dkr-kernel-registry","tools/dkr-s31-builder"]
CODECS=("Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q2K","Q3K","Q4K","Q5K","Q6K")
MOE_CODECS=("Q8_0","Q4K","Q2K","Q3K","Q5K","Q6K")
CORE=("affine","binary","cast","conv","fill","indexing","reduce",
      "sort","ternary","unary","quantized")
FAMILIES=set(CORE)|{"mmvq_gguf","mmq_quantize","moe_gguf","moe_simt_f16"}
FAMILIES.update("mmq_instance_"+codec.replace("K","_K").lower() for codec in CODECS)

def require(ok,message):
    if not ok: raise SystemExit("DKR_R124_S52_FAIL: "+message)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

def source_gate():
    require(subprocess.run(["git","diff","--quiet",SOURCE_BASE,"HEAD","--",*PRODUCTION],
                           cwd=ROOT).returncode==0,"production sources differ from S4.4")
    require(not subprocess.check_output(
        ["git","status","--porcelain","--untracked-files=no","--",*PRODUCTION],
        cwd=ROOT,text=True).strip(),"dirty production source")

def seal_gate(seal_file):
    seal_file=seal_file.resolve(strict=True)
    data=json.loads(seal_file.read_text())
    require(data["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1"
            and data["source_baseline"]==SOURCE_BASE,"invalid S5.1 seal")
    require(sha(seal_file.parent/"hardware-audit.json")==data["audit_sha256"],
            "S5.1 hardware audit changed")
    cat=Path(data["catalogue"])
    items=data["catalogue_inventory"]["packages"]
    require(set(items)==FAMILIES and len(items)==25,"25-family catalogue mismatch")
    for name,item in items.items():
        folder=cat/name
        require(sha(folder/"manifest.json")==item["manifest_sha256"]
                and sha(folder/(name+".ptx"))==item["ptx_sha256"],
                "sealed manifest/PTX changed: "+name)
    ev=data["evidence"]
    require(sha(Path(ev["archive"]))==ev["archive_sha256"],"S4.4 archive changed")
    return data

def environment():
    env=os.environ.copy()
    for name in ["DKR_EMBED_NATIVE_DIR","CUDA_COMPUTE_CAP",
                 "CANDLE_DKR_S31_DIR","MISTRALRS_DKR_ARTIFACT_SOURCE"]:
        env.pop(name,None)
    env["CANDLE_DKR_TRACE"]="1"
    return env

def run(cmd,env,path):
    print("+ "+" ".join(map(str,cmd)),flush=True)
    with path.open("w") as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in p.stdout:
            print(line,end="",flush=True)
            log.write(line)
        rc=p.wait()
    require(rc==0,f"exit {rc}: {path}")
    return path.read_text(errors="replace")

def build(args):
    source_gate()
    seal=seal_gate(args.seal)
    out=args.out.resolve()
    require(not out.exists(),"output exists; choose a new directory")
    out.mkdir(parents=True)
    archive=out/"s52-tests.tar.zst"
    env=environment()
    env["DKR_EMBED_NATIVE_DIR"]=seal["catalogue"]
    run(["cargo","nextest","archive","-p","candle-core","-p","candle-nn",
         "--no-default-features","--features","cuda-dkr-embedded",
         "--test","dkr_s52_numeric","--test","moe_simt_sm61",
         "--test","cuda_legacy_bf16_tests","--archive-file",str(archive)],
        env,out/"archive-build.log")
    require(archive.is_file(),"Nextest archive missing")
    receipt={"schema":"CANDLE-DKR-R124-S52-ARCHIVE-V1",
             "seal":str(args.seal.resolve(strict=True)),
             "seal_sha256":sha(args.seal.resolve(strict=True)),
             "archive":str(archive),"archive_sha256":sha(archive)}
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("DKR_R124_S52_BUILD=PASS")
    print("DKR_R124_S52_RECEIPT="+str(out/"receipt.json"))

def passed_tests(log):
    return {line.strip()[5:-7].rsplit("::",1)[-1]
            for line in log.splitlines()
            if line.strip().startswith("test ") and line.strip().endswith(" ... ok")}

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "run headless, without DISPLAY/WAYLAND_DISPLAY")
    source_gate()
    receipt_path=args.receipt.resolve(strict=True)
    receipt=json.loads(receipt_path.read_text())
    require(receipt["schema"]=="CANDLE-DKR-R124-S52-ARCHIVE-V1","receipt schema")
    sf=Path(receipt["seal"])
    require(sha(sf)==receipt["seal_sha256"],"seal modified")
    seal_gate(sf)
    archive=Path(receipt["archive"])
    require(sha(archive)==receipt["archive_sha256"],"S5.2 archive modified")
    env=environment()
    out=receipt_path.parent/("headless-"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    before=run(["nvidia-smi","--query-gpu=index,name,compute_cap,pstate,temperature.gpu,clocks.gr,clocks.mem,memory.used",
                "--format=csv"],env,out/"gpu-before.log")
    require(", 6.1," in before,"SM61 device missing")
    output=run(["cargo","nextest","run","--archive-file",str(archive),
                "--workspace-remap",str(ROOT),"--run-ignored","all",
                "--no-fail-fast","--no-capture","-E",
                "test(dkr_s52_) | test(sm61_simt_) | test(cuda_legacy_bf16_)"],
               env,out/"numeric.log")
    passed=passed_tests(output)
    expected={"dkr_s52_ten_codecs_mmvq_mmq","dkr_s52_core_families_cpu_parity",
              "dkr_s52_gguf_moe_six_codecs","sm61_simt_prefill_topk2_fp16",
              "sm61_simt_decode_weighted_fp16"}
    require(expected<=passed,"missing tests: "+",".join(sorted(expected-passed)))
    require("DKR_R124_S52_CORE=PASS" in output
            and "DKR_R124_S52_QUANT=PASS cases=120" in output
            and "DKR_R124_S52_GGUF_MOE=PASS codecs=6" in output,
            "numeric suite incomplete")
    cases=re.findall(r"DKR_R124_S52_QUANT_CASE codec=(\w+) dtype=(\w+) batch=(\d+) route=(\w+)",output)
    required={(c,d,str(b),"MMVQ" if b<=8 else "MMQ")
              for c in CODECS for d in ("F16","BF16","F32") for b in (1,8,9,17)}
    require(len(cases)==120 and set(cases)==required,"120 numeric cases incomplete")
    moe=re.findall(r"DKR_R124_S52_GGUF_MOE codec=(\w+) rows=4 n=5 k=256",output)
    require(len(moe)==6 and set(moe)==set(MOE_CODECS),"six GGUF MoE codecs incomplete")
    require("CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25" in output
            and "CANDLE_DKR_NATIVE_CATALOG source=filesystem" not in output,
            "invalid embedded provenance")
    loaded=set(re.findall(r"CANDLE_DKR_MODULE_LOADED family=([a-z0-9_]+) target=sm61",output))
    require(loaded<=FAMILIES,"unexpected native family")
    missing=sorted(FAMILIES-loaded)
    run(["nvidia-smi","--query-gpu=index,name,pstate,temperature.gpu,clocks.gr,clocks.mem,memory.used",
         "--format=csv"],env,out/"gpu-after.log")
    result={"schema":"CANDLE-DKR-R124-S52-RESULT-V1",
            "archive_sha256":receipt["archive_sha256"],"passed_tests":sorted(passed),
            "numeric_cases":len(cases),"gguf_moe_codecs":len(moe),
            "loaded_families":sorted(loaded),"unseen_families":missing,
            "scope":"selected numerical paths and first module loads; not all PTX entrypoints"}
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"DKR_R124_S52_NUMERIC=PASS cases={len(cases)} gguf_moe={len(moe)}")
    print(f"DKR_R124_S52_FAMILIES={len(loaded)}/25")
    print("DKR_R124_S52_REPORT="+str(out/"result.json"))
    require(not missing,"native family unobserved: "+",".join(missing))
    print("DKR_R124_S52_PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("build")
    b.add_argument("--seal",required=True,type=Path)
    b.add_argument("--out",required=True,type=Path)
    h=sub.add_parser("headless")
    h.add_argument("--receipt",required=True,type=Path)
    args=p.parse_args()
    (build if args.mode=="build" else headless)(args)

if __name__=="__main__":
    main()
