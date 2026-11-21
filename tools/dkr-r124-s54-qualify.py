#!/usr/bin/env python3
"""R1.24 S5.4: sealed PTX -> SM61 CUBIN pilot; JIT cache and live VRAM.

Prepare CUBIN files outside headless with CUDA 12.x ptxas; nothing in the
qualified PTX directory or production Candle sources is modified. Headless
runs fresh-process PTX cold/warm and CUBIN driver probes, plus independent
S5.3 workload VRAM sampling. No NVCC or Cargo compilation occurs headlessly.
"""
import argparse
import ctypes as C
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent.parent
SOURCE_BASE="d1929bea2a57beb82d466c9654c1d5f5d6ff616c"
PRODUCTION=["candle-core/src","candle-core/build.rs","candle-core/Cargo.toml",
            "candle-kernels","candle-nn/src","candle-nn/Cargo.toml",
            "dkr-kernel-registry","tools/dkr-s31-builder"]
DEFAULT_FAMILIES=("fill","mmvq_gguf","mmq_instance_q4_k")
S53_CASES=("mmvq_q4k_b1","mmvq_q8_0_b8","mmq_q4k_b16",
           "mmq_q8_0_b32","moe_gguf_q4k_t8",
           "moe_simt_f16_decode_t3","moe_simt_f16_prefill_t8")

def require(ok,message):
    if not ok:
        raise SystemExit("DKR_R124_S54_FAIL: "+message)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda:f.read(1048576),b""): h.update(piece)
    return h.hexdigest()

def source_gate():
    require(subprocess.run(["git","diff","--quiet",SOURCE_BASE,"HEAD","--",*PRODUCTION],
                           cwd=ROOT).returncode==0,"production differs from S4.4")
    require(not subprocess.check_output(
        ["git","status","--porcelain","--untracked-files=no","--",*PRODUCTION],
        cwd=ROOT,text=True).strip(),"tracked production files modified")

def seal_gate(path):
    sf=path.resolve(strict=True)
    seal=json.loads(sf.read_text())
    require(seal["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1"
            and seal["source_baseline"]==SOURCE_BASE,"bad S5.1 seal")
    require(sha(sf.parent/"hardware-audit.json")==seal["audit_sha256"],
            "S5.1 hardware audit changed")
    root=Path(seal["catalogue"])
    packages=seal["catalogue_inventory"]["packages"]
    require(len(packages)==25,"S5.1 inventory cardinality")
    for family,item in packages.items():
        require(sha(root/family/"manifest.json")==item["manifest_sha256"]
                and sha(root/family/(family+".ptx"))==item["ptx_sha256"],
                "sealed family changed: "+family)
    require(sha(Path(seal["evidence"]["archive"]))==seal["evidence"]["archive_sha256"],
            "S4.4 archived evidence changed")
    return seal

def s53_gate(result_path):
    path=result_path.resolve(strict=True)
    result=json.loads(path.read_text())
    require(result["schema"]=="CANDLE-DKR-R124-S53-RESULT-V1"
            and result["status"]=="PASS","S5.3 result not qualified")
    require(len(result["rounds"])==6 and set(result["summary"])==set(S53_CASES),
            "S5.3 performance inventory incomplete")
    receipt_path=ROOT/"target/dkr-r124-s53/receipt.json"
    receipt=json.loads(receipt_path.read_text())
    require(sha(Path(receipt["archive"]))==receipt["archive_sha256"]
            and result["archive_sha256"]==receipt["archive_sha256"],
            "S5.3 benchmark archive changed")
    require(result["s51_seal_sha256"]==receipt["s51_seal_sha256"],
            "S5.3 seal mismatch")
    return result,receipt

def parse_symbols(text):
    return sorted(set(re.findall(r"\.entry\s+([^\s(]+)",text)))

def select_symbol(family,entries):
    require(bool(entries),"no PTX entry for "+family)
    if family=="fill":
        require("fill_f32" in entries,"fill_f32 symbol missing")
        return "fill_f32"
    if family=="mmvq_gguf":
        symbol="mmvq_gguf_q4_k_f16_plain_cuda1"
        require(symbol in entries,"expected MMVQ Q4K symbol absent")
        return symbol
    return entries[0]

def clean_env():
    env=os.environ.copy()
    for key in ("DKR_EMBED_NATIVE_DIR","CANDLE_DKR_S31_DIR",
                "MISTRALRS_DKR_ARTIFACT_SOURCE","CUDA_COMPUTE_CAP",
                "CUDA_CACHE_PATH","DKR_R124_S53_PHASE","DKR_R124_S53_ROUND"):
        env.pop(key,None)
    env["CUDA_CACHE_DISABLE"]="0"
    env["CUDA_CACHE_MAXSIZE"]=str(512*1024*1024)
    env["CUDA_MODULE_LOADING"]="EAGER"
    env["CANDLE_DKR_TRACE"]="1"
    return env

def execute(cmd,env,log_path):
    print("+ "+" ".join(map(str,cmd)),flush=True)
    with log_path.open("w") as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in p.stdout:
            print(line,end="",flush=True)
            log.write(line)
        code=p.wait()
    require(code==0,f"exit={code}, log={log_path}")
    return log_path.read_text(errors="replace")

def prepare(args):
    source_gate()
    seal=seal_gate(args.seal)
    s53_gate(args.s53_result)
    ptxas=shutil.which(args.ptxas) if "/" not in args.ptxas else args.ptxas
    require(ptxas is not None and Path(ptxas).is_file(),
            "ptxas not found; pass --ptxas /path/to/CUDA-12.x/bin/ptxas")
    ptxas=str(Path(ptxas).resolve())
    version=subprocess.run([ptxas,"--version"],text=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    require(version.returncode==0,"cannot execute ptxas --version")
    match=re.search(r"release\s+(\d+)\.(\d+)",version.stdout)
    require(match and int(match.group(1))==12,
            "SM61 offline compilation requires CUDA 12.x ptxas, NOT CUDA 13.x; got: "+version.stdout.strip())
    families=tuple(x.strip() for x in args.families.split(",") if x.strip())
    require(families and len(set(families))==len(families)
            and set(families)<=set(seal["catalogue_inventory"]["packages"]),
            "invalid S5.1 family selection")
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/"receipt.json").exists(),
            "receipt exists; do not overwrite an already prepared qualification")
    cubindir=out/"cubins"
    cubindir.mkdir(exist_ok=True)
    rows={}
    for family in families:
        ptx=Path(seal["catalogue"])/family/(family+".ptx")
        lines=ptx.read_text()
        symbol=select_symbol(family,parse_symbols(lines))
        cubin=cubindir/(family+".cubin")
        note=cubindir/(family+".json")
        if cubin.exists() or note.exists():
            require(cubin.is_file() and note.is_file(),
                    "incomplete prior CUBIN; inspect "+str(cubin))
            existing=json.loads(note.read_text())
            require(existing["ptx_sha256"]==sha(ptx)
                    and existing["cubin_sha256"]==sha(cubin)
                    and existing["ptxas_version"]==version.stdout.strip()
                    and existing["symbol"]==symbol,
                    "CUBIN cache metadata conflict for "+family)
            print("DKR_R124_S54_CUBIN_REUSED="+family,flush=True)
            rows[family]=existing
            continue
        intermediate=cubindir/(family+".cubin.partial")
        log=cubindir/(family+".ptxas.log")
        print("DKR_R124_S54_COMPILING_CUBIN="+family,flush=True)
        try:
            execute([ptxas,"-arch=sm_61","-O3",str(ptx),"-o",str(intermediate)],
                    clean_env(),log)
            require(intermediate.is_file() and intermediate.stat().st_size>0,
                    "ptxas produced no CUBIN for "+family)
            intermediate.replace(cubin)
        finally:
            if intermediate.exists(): intermediate.unlink()
        details={"family":family,"ptx":str(ptx),"ptx_sha256":sha(ptx),
                 "cubin":str(cubin),"cubin_sha256":sha(cubin),
                 "cubin_bytes":cubin.stat().st_size,"symbol":symbol,
                 "ptxas":ptxas,"ptxas_version":version.stdout.strip()}
        note.write_text(json.dumps(details,sort_keys=True,indent=2)+"\n")
        rows[family]=details
        print("DKR_R124_S54_CUBIN_BUILT="+family+" bytes="+str(cubin.stat().st_size),flush=True)
    sf=args.seal.resolve(strict=True)
    sr=args.s53_result.resolve(strict=True)
    receipt={"schema":"CANDLE-DKR-R124-S54-PILOT-V1",
             "seal":str(sf),"seal_sha256":sha(sf),
             "s53_result":str(sr),"s53_result_sha256":sha(sr),
             "families":rows,"ptxas_version":version.stdout.strip(),
             "target_cc":61,
             "scope":"external offline CUBIN pilot; production Candle remains PTX-only"}
    (out/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S54_PREPARE=PASS families="+str(len(rows)),flush=True)
    print("DKR_R124_S54_RECEIPT="+str(out/"receipt.json"),flush=True)

def cache_count(path):
    return sum(1 for p in path.rglob("*") if p.is_file())

class NvmlSampler:
    def __init__(self):
        self.records=[]
        self.failure=None
        self.stop=threading.Event()
        self.thread=None
        self.lib=None
        self.handle=C.c_void_p()
    def start(self):
        lib=C.CDLL("libnvidia-ml.so.1")
        def fn(name,argtypes):
            f=getattr(lib,name)
            f.argtypes=argtypes
            f.restype=C.c_int
            return f
        self.shutdown=fn("nvmlShutdown",[])
        init=fn("nvmlInit_v2",[])
        get_device=fn("nvmlDeviceGetHandleByIndex_v2",
                      [C.c_uint,C.POINTER(C.c_void_p)])
        self.mem_fn=fn("nvmlDeviceGetMemoryInfo",
                       [C.c_void_p,C.c_void_p])
        require(init()==0,"NVML initialization failed")
        self.lib=lib
        require(get_device(0,C.byref(self.handle))==0,"NVML device index 0 unavailable")
        class Memory(C.Structure):
            _fields_=[("total",C.c_ulonglong),
                      ("free",C.c_ulonglong),
                      ("used",C.c_ulonglong)]
        self.Memory=Memory
        def sample():
            while not self.stop.is_set():
                memory=self.Memory()
                code=self.mem_fn(self.handle,C.byref(memory))
                if code==0:
                    self.records.append((time.monotonic_ns(),int(memory.used)))
                else:
                    self.failure="NVML memory query returned "+str(code)
                    return
                self.stop.wait(0.025)
        self.thread=threading.Thread(target=sample,daemon=True)
        self.thread.start()
    def finish(self):
        self.stop.set()
        if self.thread is not None: self.thread.join(timeout=5)
        if self.lib is not None: self.shutdown()
        require(self.failure is None,"VRAM sampler: "+str(self.failure))
        require(len(self.records)>0,"VRAM sampler did not capture data")
        values=[b for _,b in self.records]
        return {"samples":len(values),"used_min_bytes":min(values),
                "used_max_bytes":max(values),"used_last_bytes":values[-1],
                "poll_interval_ms":25}

def probe(path,symbol,cache,log,fill):
    env=clean_env()
    env["CUDA_CACHE_PATH"]=str(cache)
    command=[sys.executable,str(ROOT/"tools/dkr-r124-s54-driver-probe.py"),
             "--file",str(path),"--symbol",symbol]
    if fill: command.append("--fill-parity")
    log_text=execute(command,env,log)
    markers=[line[len("DKR_R124_S54_PROBE="):] for line in log_text.splitlines()
             if line.startswith("DKR_R124_S54_PROBE=")]
    require(len(markers)==1,"missing CUDA driver probe result")
    result=json.loads(markers[0])
    require(result["sm"]=="6.1" and result["symbol"]==symbol,
            "driver did not load expected symbol on SM61")
    if fill:
        require(result["fill_parity"] and result["fill_parity"]["passed"],
                "CUBIN/PTX fill numerical parity failed")
    require(result["module_load_us"]>0 and result["vram_after_module_bytes"]>0,
            "invalid module load/memory sample")
    return result

def live_s53_vram(s53_result,output_dir,log):
    _,rec=s53_gate(s53_result)
    archive=Path(rec["archive"])
    require(archive.is_file(),"S5.3 archived binary missing")
    # Reuse a qualified warm cache, avoiding another expensive cold JIT.
    original=json.loads(s53_result.read_text())
    cache=Path(original["rounds"][-1]["cache_path"])
    require(cache.is_dir() and cache_count(cache)>0,
            "prior S5.3 warm cache unavailable")
    env=clean_env()
    env["CUDA_CACHE_PATH"]=str(cache)
    env["DKR_R124_S53_PHASE"]="warm"
    env["DKR_R124_S53_ROUND"]="4"
    cmd=["cargo","nextest","run","--archive-file",str(archive),
         "--workspace-remap",str(ROOT),"--run-ignored","all",
         "--no-fail-fast","--no-capture","-E","test(=dkr_s53_bench)"]
    sampler=NvmlSampler()
    sampler.start()
    try:
        output=execute(cmd,env,log)
    finally:
        measurement=sampler.finish()
    require("DKR_R124_S53_TEST=PASS phase=warm round=4 cases=7 samples=80"
            in output,"S5.3 VRAM replay did not complete")
    (output_dir/"vram-sampling.json").write_text(
        json.dumps(measurement,sort_keys=True,indent=2)+"\n")
    return measurement

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "launch through headless-relogin")
    source_gate()
    receipt_path=args.receipt.resolve(strict=True)
    receipt=json.loads(receipt_path.read_text())
    require(receipt["schema"]=="CANDLE-DKR-R124-S54-PILOT-V1"
            and receipt["target_cc"]==61,"S5.4 receipt invalid")
    seal_file=Path(receipt["seal"])
    s53_result=Path(receipt["s53_result"])
    require(sha(seal_file)==receipt["seal_sha256"]
            and sha(s53_result)==receipt["s53_result_sha256"],
            "qualified S5.1/S5.3 evidence changed")
    seal_gate(seal_file)
    s53_gate(s53_result)
    for family,item in receipt["families"].items():
        require(sha(Path(item["ptx"]))==item["ptx_sha256"]
                and sha(Path(item["cubin"]))==item["cubin_sha256"],
                "CUBIN or sealed PTX modified for "+family)
    env=clean_env()
    out=receipt_path.parent/("headless-"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    gpu=execute(["nvidia-smi","--query-gpu=index,name,compute_cap,driver_version,temperature.gpu,pstate,clocks.gr,clocks.mem,memory.used",
                 "--format=csv"],env,out/"gpu-before.log")
    require(", 6.1," in gpu,"SM61 GPU missing")
    results={}
    for family,item in receipt["families"].items():
        values=[]
        for round_id in range(1,4):
            cache=out/f"jit-{family}-r{round_id}"
            cache.mkdir()
            require(cache_count(cache)==0,"PTX cold cache not empty")
            for label,path in (("ptx_cold",Path(item["ptx"])),
                               ("ptx_warm",Path(item["ptx"])),
                               ("cubin",Path(item["cubin"]))):
                record=probe(path,item["symbol"],cache,
                             out/f"{family}-{round_id}-{label}.log",family=="fill")
                record.update({"round":round_id,"label":label,
                               "cache_files_after":cache_count(cache)})
                values.append(record)
                print(f"DKR_R124_S54_SAMPLE family={family} round={round_id} mode={label} load_us={record['module_load_us']} lookup_us={record['function_lookup_us']} vram_module_delta={record['vram_module_delta_bytes']}",flush=True)
        summary={}
        for label in ("ptx_cold","ptx_warm","cubin"):
            rows=[v for v in values if v["label"]==label]
            summary[label]={
                "module_load_us_median":statistics.median(v["module_load_us"] for v in rows),
                "function_lookup_us_median":statistics.median(v["function_lookup_us"] for v in rows),
                "vram_module_delta_bytes_median":statistics.median(v["vram_module_delta_bytes"] for v in rows),
                "cuda_cache_files_after_median":statistics.median(v["cache_files_after"] for v in rows)}
        cold=summary["ptx_cold"]["module_load_us_median"]
        cubin=summary["cubin"]["module_load_us_median"]
        summary["ptx_cold_over_cubin_load_ratio"]=round(cold/max(1,cubin),3)
        results[family]={"samples":values,"summary":summary}
        print(f"DKR_R124_S54_CUBIN family={family} ptx_cold_us={cold} cubin_us={cubin} ratio={summary['ptx_cold_over_cubin_load_ratio']}",flush=True)
    vram=live_s53_vram(s53_result,out,out/"s53-live-vram.log")
    execute(["nvidia-smi","--query-gpu=index,name,temperature.gpu,pstate,clocks.gr,clocks.mem,memory.used",
             "--format=csv"],env,out/"gpu-after.log")
    report={"schema":"CANDLE-DKR-R124-S54-RESULT-V1","status":"PASS",
            "evidence":"sealed PTX, offline CUBIN and direct CUDA driver loading",
            "families":results,"s53_live_vram":vram,
            "measurement_note":"cuModuleLoadData with EAGER loading; new CUDA context for each probe",
            "caveats":["No production Candle CUBIN dispatch",
                       "Only fill_f32 is launch-parity checked; MMVQ/MMQ CUBIN symbol loads are verified",
                       "SM61 CUBIN is architecture-specific",
                       "VRAM sampled with NVML at 25ms; short-lived peaks can be missed",
                       "A fast driver module load is not guaranteed to eliminate all first kernel invocation overhead"]}
    (out/"result.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S54_DRIVER=PASS")
    print("DKR_R124_S54_VRAM=PASS sample_count="+str(vram["samples"])+" sampled_peak_mib="+str(round(vram["used_max_bytes"]/1048576,2)))
    print("DKR_R124_S54_REPORT="+str(out/"result.json"))
    print("DKR_R124_S54_PASS cubin_pilot_only=1")

def self_test():
    assert parse_symbols(".visible .entry fill_f32( .param .u64 x ) {\nret;\n}") == ["fill_f32"]
    assert select_symbol("fill",["fill_f32"])=="fill_f32"
    assert select_symbol("mmvq_gguf",["mmvq_gguf_q4_k_f16_plain_cuda1"])=="mmvq_gguf_q4_k_f16_plain_cuda1"
    assert len(DEFAULT_FAMILIES)==3
    print("DKR_R124_S54_SELF_TEST=PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--seal",type=Path,required=True)
    a.add_argument("--s53-result",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--ptxas",default="ptxas")
    a.add_argument("--families",default=",".join(DEFAULT_FAMILIES))
    b=sub.add_parser("headless")
    b.add_argument("--receipt",type=Path,required=True)
    sub.add_parser("self-test")
    args=p.parse_args()
    if args.mode=="prepare": prepare(args)
    elif args.mode=="headless": headless(args)
    else: self_test()

if __name__=="__main__":
    main()
