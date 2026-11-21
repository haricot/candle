#!/usr/bin/env python3
"""R1.24 S5.3: sealed headless CUDA_CACHE_PATH paired performance qualification.

Use 'build' before graphical logout; only Nextest archive replay in 'headless'.
Paired cold/warm labels mean fresh CUDA disk cache directory versus reuse of the
same directory by another process. They do NOT promise a cold driver, cold
filesystem page cache, fresh GPU state, or isolated hardware kernel timings.
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

ROOT=Path(__file__).resolve().parent.parent
SOURCE_BASE="d1929bea2a57beb82d466c9654c1d5f5d6ff616c"
PRODUCTION=["candle-core/src","candle-core/build.rs","candle-core/Cargo.toml",
            "candle-kernels","candle-nn/src","candle-nn/Cargo.toml",
            "dkr-kernel-registry","tools/dkr-s31-builder"]
CASES=("mmvq_q4k_b1","mmvq_q8_0_b8","mmq_q4k_b16","mmq_q8_0_b32",
       "moe_gguf_q4k_t8","moe_simt_f16_decode_t3","moe_simt_f16_prefill_t8")
RE_CASE=re.compile(
    r"^DKR_R124_S53_CASE phase=(cold|warm) round=(\d+) "
    r"name=([a-z0-9_]+) first_us=(\d+) p50_us=(\d+) p95_us=(\d+) "
    r"min_us=(\d+) max_us=(\d+) tokens=(\d+) samples=(\d+)$",re.M)
RE_DEVICE=re.compile(r"^DKR_R124_S53_DEVICE phase=(cold|warm) round=(\d+) init_us=(\d+)$",re.M)

def require(ok,message):
    if not ok: raise SystemExit("DKR_R124_S53_FAIL: "+message)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

def source_gate():
    require(subprocess.run(["git","diff","--quiet",SOURCE_BASE,"HEAD","--",*PRODUCTION],
                           cwd=ROOT).returncode==0,"production differs from qualified S4.4")
    require(not subprocess.check_output(
        ["git","status","--porcelain","--untracked-files=no","--",*PRODUCTION],
        cwd=ROOT,text=True).strip(),"tracked production changes present")

def seal_gate(sf):
    sf=sf.resolve(strict=True)
    seal=json.loads(sf.read_text())
    require(seal["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1" and
            seal["source_baseline"]==SOURCE_BASE,"bad S5.1 seal")
    require(sha(sf.parent/"hardware-audit.json")==seal["audit_sha256"],"hardware audit changed")
    root=Path(seal["catalogue"])
    for family,item in seal["catalogue_inventory"]["packages"].items():
        require(sha(root/family/"manifest.json")==item["manifest_sha256"] and
                sha(root/family/(family+".ptx"))==item["ptx_sha256"],
                "sealed package changed: "+family)
    ev=seal["evidence"]
    require(sha(Path(ev["archive"]))==ev["archive_sha256"],"S4.4 archive changed")
    return seal

def numeric_gate(result_path):
    result_path=result_path.resolve(strict=True)
    result=json.loads(result_path.read_text())
    require(result["schema"]=="CANDLE-DKR-R124-S52-RESULT-V1" and
            result["numeric_cases"]==120 and result["gguf_moe_codecs"]==6 and
            len(result["loaded_families"])==25 and
            result["unseen_families"]==[],
            "S5.2 qualification incomplete")
    source_receipt=ROOT/"target/dkr-r124-s52/receipt.json"
    r=json.loads(source_receipt.read_text())
    require(result["archive_sha256"]==r["archive_sha256"] and
            sha(Path(r["archive"]))==r["archive_sha256"],
            "S5.2 archive evidence changed")
    return result_path

def environment():
    env=os.environ.copy()
    for key in ("CUDA_COMPUTE_CAP","CANDLE_DKR_S31_DIR",
                "MISTRALRS_DKR_ARTIFACT_SOURCE","DKR_EMBED_NATIVE_DIR",
                "DKR_R124_S53_PHASE","DKR_R124_S53_ROUND","CUDA_CACHE_PATH"):
        env.pop(key,None)
    env["CANDLE_DKR_TRACE"]="1"
    env["CUDA_CACHE_DISABLE"]="0"
    env["CUDA_CACHE_MAXSIZE"]=str(512*1024*1024)
    return env

def run(args,env,output):
    print("+ "+" ".join(map(str,args)),flush=True)
    with output.open("w") as log:
        process=subprocess.Popen(args,env=env,cwd=ROOT,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in process.stdout:
            print(line,end="",flush=True)
            log.write(line)
        rc=process.wait()
    require(rc==0,"exit "+str(rc)+": "+str(output))
    return output.read_text(errors="replace")

def build(args):
    source_gate()
    seal=seal_gate(args.seal)
    source_numeric=numeric_gate(args.s52_result)
    out=args.out.resolve()
    require(not out.exists(),"S5.3 output already exists; choose a new directory")
    out.mkdir(parents=True)
    env=environment()
    env["DKR_EMBED_NATIVE_DIR"]=seal["catalogue"]
    archive=out/"s53-tests.tar.zst"
    run(["cargo","nextest","archive","-p","candle-nn",
         "--no-default-features","--features","cuda-dkr-embedded",
         "--test","dkr_s53_perf","--archive-file",str(archive)],
        env,out/"archive-build.log")
    require(archive.is_file(),"S5.3 Nextest archive absent")
    receipt={
        "schema":"CANDLE-DKR-R124-S53-ARCHIVE-V1",
        "s51_seal":str(args.seal.resolve(strict=True)),
        "s51_seal_sha256":sha(args.seal.resolve(strict=True)),
        "s52_result":str(source_numeric),
        "s52_result_sha256":sha(source_numeric),
        "archive":str(archive),
        "archive_sha256":sha(archive),
        "workload_cases":list(CASES),
        "warmups":24,"samples":80,"rounds":3,
        "source_head":subprocess.check_output(["git","rev-parse","HEAD"],
                                             cwd=ROOT,text=True).strip(),
    }
    (out/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S53_BUILD=PASS")
    print("DKR_R124_S53_RECEIPT="+str(out/"receipt.json"))

def parse_cases(output,phase,round_id):
    matches=RE_CASE.findall(output)
    require(len(matches)==len(CASES),"expected exactly seven benchmark rows")
    rows={}
    for phase_s,round_s,name,first,p50,p95,mini,maxi,tokens,samples in matches:
        require(phase_s==phase and int(round_s)==round_id,"case phase/round mismatch")
        require(name in CASES and name not in rows,"invalid or repeated case name: "+name)
        row={"first_us":int(first),"p50_us":int(p50),"p95_us":int(p95),
             "min_us":int(mini),"max_us":int(maxi),"tokens":int(tokens),
             "samples":int(samples)}
        require(row["samples"]==80 and row["tokens"]>0 and
                0<row["min_us"]<=row["p50_us"]<=row["p95_us"]<=row["max_us"] and
                row["first_us"]>0, "invalid latency sample: "+name)
        rows[name]=row
    require(set(rows)==set(CASES),"benchmark case inventory incomplete")
    device=RE_DEVICE.findall(output)
    require(len(device)==1 and device[0][0]==phase and
            int(device[0][1])==round_id and int(device[0][2])>0,
            "device timing missing")
    require("DKR_R124_S53_TEST=PASS phase="+phase+" round="+str(round_id)+" cases=7 samples=80" in output,
            "benchmark did not finish")
    require("CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25" in output and
            "CANDLE_DKR_NATIVE_CATALOG source=filesystem" not in output,
            "runtime was not sealed embedded")
    for family in ("mmvq_gguf","mmq_quantize","mmq_instance_q4_k",
                   "mmq_instance_q8_0","moe_gguf","moe_simt_f16"):
        require("CANDLE_DKR_MODULE_LOADED family="+family+" target=sm61" in output,
                "expected native module not loaded: "+family)
    return {"cases":rows,"device_init_us":int(device[0][2])}

def cache_files(path):
    return sum(1 for item in path.rglob("*") if item.is_file())

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "run inside headless-relogin, no graphical session")
    source_gate()
    rp=args.receipt.resolve(strict=True)
    receipt=json.loads(rp.read_text())
    require(receipt["schema"]=="CANDLE-DKR-R124-S53-ARCHIVE-V1"
            and receipt["workload_cases"]==list(CASES),"S5.3 receipt schema/inventory")
    seal=Path(receipt["s51_seal"])
    require(sha(seal)==receipt["s51_seal_sha256"],"S5.1 seal modified")
    seal_gate(seal)
    nr=Path(receipt["s52_result"])
    require(sha(nr)==receipt["s52_result_sha256"],"S5.2 evidence modified")
    numeric_gate(nr)
    archive=Path(receipt["archive"])
    require(sha(archive)==receipt["archive_sha256"],"S5.3 archive modified")
    out=rp.parent/("headless-"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    env=environment()
    gpu=run(["nvidia-smi","--query-gpu=index,name,compute_cap,driver_version,pstate,temperature.gpu,clocks.gr,clocks.mem,memory.used",
             "--format=csv"],env,out/"gpu-before.log")
    require(", 6.1," in gpu,"SM61 GPU missing")
    records=[]
    for round_id in range(1,4):
        cache=out/f"jit-cache-r{round_id}"
        cache.mkdir()
        require(cache_files(cache)==0,"cold cache is not empty")
        for phase in ("cold","warm"):
            phase_env=dict(env, CUDA_CACHE_PATH=str(cache),
                           DKR_R124_S53_PHASE=phase,
                           DKR_R124_S53_ROUND=str(round_id))
            gpu_pre=run(["nvidia-smi","--query-gpu=pstate,temperature.gpu,clocks.gr,clocks.mem,memory.used",
                         "--format=csv"],phase_env,out/f"gpu-r{round_id}-{phase}-before.log")
            output=run(["cargo","nextest","run","--archive-file",str(archive),
                        "--workspace-remap",str(ROOT),"--run-ignored","all",
                        "--no-fail-fast","--no-capture","-E","test(=dkr_s53_bench)"],
                       phase_env,out/f"benchmark-r{round_id}-{phase}.log")
            measured=parse_cases(output,phase,round_id)
            gpu_post=run(["nvidia-smi","--query-gpu=pstate,temperature.gpu,clocks.gr,clocks.mem,memory.used",
                          "--format=csv"],phase_env,out/f"gpu-r{round_id}-{phase}-after.log")
            records.append({"round":round_id,"phase":phase,"cache_path":str(cache),
                            "cache_files_after":cache_files(cache),
                            "gpu_pre":gpu_pre.strip(),"gpu_post":gpu_post.strip(),**measured})
    summary={}
    for name in CASES:
        summary[name]={}
        for phase in ("cold","warm"):
            group=[r["cases"][name] for r in records if r["phase"]==phase]
            summary[name][phase]={
                field+"_median_us":statistics.median(row[field] for row in group)
                for field in ("first_us","p50_us","p95_us")}
            summary[name][phase]["tokens_per_sec_median_estimate"] = round(
                group[0]["tokens"] * 1_000_000 / summary[name][phase]["p50_us_median_us"],2)
        first_cold=summary[name]["cold"]["first_us_median_us"]
        first_warm=summary[name]["warm"]["first_us_median_us"]
        summary[name]["paired_first_touch_ratio_cold_over_warm"]=round(first_cold/first_warm,4)
    result={"schema":"CANDLE-DKR-R124-S53-RESULT-V1","status":"PASS",
            "s51_seal_sha256":receipt["s51_seal_sha256"],
            "s52_result_sha256":receipt["s52_result_sha256"],
            "archive_sha256":receipt["archive_sha256"],
            "rounds":records,"summary":summary,
            "timing":"synchronized host wall time including launch/allocator overhead",
            "cold_definition":"fresh isolated CUDA_CACHE_PATH, new process; not guaranteed fully cold driver",
            "warm_definition":"same cache path, new process after cold phase",
            "not_measured":"device event kernel-only latency, cold OS page cache, model inference tokens/s"}
    (out/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S53_BENCH=PASS rounds=3 phases=2 cases=7 samples_per_case=80")
    print("DKR_R124_S53_REPORT="+str(out/"result.json"))
    print("DKR_R124_S53_PASS")

def self_test():
    sample=("DKR_R124_S53_DEVICE phase=cold round=1 init_us=74000\n"+
            "\n".join("DKR_R124_S53_CASE phase=cold round=1 name="+name+
                      " first_us=2000 p50_us=100 p95_us=150 min_us=80 max_us=200 tokens=8 samples=80"
                      for name in CASES)+"\n"+
            "DKR_R124_S53_TEST=PASS phase=cold round=1 cases=7 samples=80\n"+
            "CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25\n"+
            "".join("CANDLE_DKR_MODULE_LOADED family="+f+" target=sm61\n"
                    for f in ("mmvq_gguf","mmq_quantize","mmq_instance_q4_k",
                              "mmq_instance_q8_0","moe_gguf","moe_simt_f16")))
    assert len(parse_cases(sample,"cold",1)["cases"])==7
    print("DKR_R124_S53_SELF_TEST=PASS")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("build")
    b.add_argument("--seal",type=Path,required=True)
    b.add_argument("--s52-result",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True)
    h=sub.add_parser("headless")
    h.add_argument("--receipt",type=Path,required=True)
    sub.add_parser("self-test")
    args=parser.parse_args()
    if args.mode=="build": build(args)
    elif args.mode=="headless": headless(args)
    else: self_test()

if __name__=="__main__":
    main()
