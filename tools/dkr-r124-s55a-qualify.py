#!/usr/bin/env python3
"""R1.24 S5.5a: producer-owned MMQ launch table and embedded CUBIN qualification.

'build' compiles Rust tests only; 'headless' replays an immutable Nextest
archive without Cargo build, NVCC, ptxas, or runtime access to CUBIN files.
Production PTX and all legacy S5.1-5.4 reference artifacts remain unchanged.
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
SCHEMA="CANDLE-DKR-R124-S54B-CUBIN-V1"
CODECS=("Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q2K","Q3K","Q4K","Q5K","Q6K")
MOE_CODECS=("Q8_0","Q4K","Q2K","Q3K","Q5K","Q6K")
CORE=("affine","binary","cast","conv","fill","indexing","reduce",
      "sort","ternary","unary","quantized")
FAMILIES=set(CORE)|{"mmvq_gguf","mmq_quantize","moe_gguf","moe_simt_f16"}
FAMILIES.update("mmq_instance_"+c.replace("K","_K").lower() for c in CODECS)
CASES=("mmvq_q4k_b1","mmvq_q8_0_b8","mmq_q4k_b16",
       "mmq_q8_0_b32","moe_gguf_q4k_t8",
       "moe_simt_f16_decode_t3","moe_simt_f16_prefill_t8")
RE_LOAD=re.compile(
    r"CANDLE_DKR_IMAGE_LOADED family=([a-z0-9_]+) target=sm61 format=cubin "
    r"cubin_sha256=([a-f0-9]{64}) ptx_sha256=([a-f0-9]{64})")
RE_CASE=re.compile(
    r"^DKR_R124_S53_CASE phase=cold round=(\d+) name=([a-z0-9_]+) "
    r"first_us=(\d+) p50_us=(\d+) p95_us=(\d+) min_us=(\d+) "
    r"max_us=(\d+) tokens=(\d+) samples=80$",re.M)

def require(ok,msg):
    if not ok: raise SystemExit("DKR_R124_S54B_FAIL: "+msg)

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

def source_head():
    require(not subprocess.check_output(
        ["git","status","--porcelain","--untracked-files=no",
         "--","candle-core/src","candle-core/build.rs","candle-core/Cargo.toml",
         "candle-nn/src","candle-nn/Cargo.toml","dkr-kernel-registry",
         "candle-kernels"],cwd=ROOT,text=True).strip(),"dirty production tree")
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

def seal_gate(path):
    sf=path.resolve(strict=True)
    data=json.loads(sf.read_text())
    require(data["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1"
            and data["source_baseline"]=="d1929bea2a57beb82d466c9654c1d5f5d6ff616c",
            "invalid S5.1 reference")
    require(digest(sf.parent/"hardware-audit.json")==data["audit_sha256"],
            "historical audit changed")
    orig=Path(data["catalogue"])
    packages=data["catalogue_inventory"]["packages"]
    require(len(packages)==25 and set(packages)==FAMILIES,"S5.1 package set changed")
    for name,item in packages.items():
        require(digest(orig/name/"manifest.json")==item["manifest_sha256"]
                and digest(orig/name/(name+".ptx"))==item["ptx_sha256"],
                "historical PTX package changed: "+name)
    require(digest(Path(data["evidence"]["archive"]))==
            data["evidence"]["archive_sha256"],"S4.4 archive changed")
    return data

def images_gate(root,seal):
    root=root.resolve(strict=True)
    mf=root/"manifest.json"
    manifest=json.loads(mf.read_text())
    require(manifest["schema"]==SCHEMA and manifest["target_cc"]==61,
            "unrecognized CUBIN manifest")
    rows=manifest["families"]
    require(set(rows)==FAMILIES and len(rows)==25,"missing or extra CUBIN family")
    for name,entry in rows.items():
        ptx_sha=seal["catalogue_inventory"]["packages"][name]["ptx_sha256"]
        image=root/"cubins"/(name+".cubin")
        require(entry["target_cc"]==61 and entry["ptx_sha256"]==ptx_sha
                and digest(image)==entry["cubin_sha256"]
                and image.stat().st_size==entry["cubin_bytes"],
                "invalid derived image identity: "+name)
    return manifest

def neutral_gate(recipe_dir,seal):
    folder=recipe_dir.resolve(strict=True)
    path=folder/"manifest.json"
    obj=json.loads(path.read_text())
    require(obj["schema"]=="CANDLE-DKR-R124-S55A-MMQ-LAUNCH-V1",
            "S5.5a MMQ manifest schema")
    require(obj["target_cc"]==61 and
            obj["source_index_sha256"]==digest(
                Path(seal["catalogue"])/"native-catalog.json"),
            "S5.5a source index mismatch")
    expected={c.replace("K","_k").lower() for c in CODECS}
    require(set(obj["codecs"])==expected and len(expected)==10,
            "S5.5a 10 codecs missing")
    for codec,entry in obj["codecs"].items():
        family="mmq_instance_"+codec
        require(entry["family"]==family and
                entry["ptx_sha256"]==seal["catalogue_inventory"]["packages"][family]["ptx_sha256"]
                and len(entry["tiles"])==8,
                "S5.5a MMQ recipe package mismatch: "+codec)
    return obj

def historical_gate(s52_path,s53_path,seal):
    s52=json.loads(s52_path.read_text())
    s53=json.loads(s53_path.read_text())
    require(s52["schema"]=="CANDLE-DKR-R124-S52-RESULT-V1"
            and s52["numeric_cases"]==120 and s52["gguf_moe_codecs"]==6
            and len(s52["loaded_families"])==25 and not s52["unseen_families"],
            "S5.2 original qualification not PASS")
    require(s53["schema"]=="CANDLE-DKR-R124-S53-RESULT-V1"
            and s53["status"]=="PASS" and set(s53["summary"])==set(CASES),
            "S5.3 original performance baseline invalid")
    receipt=ROOT/"target/dkr-r124-s53/receipt.json"
    old=json.loads(receipt.read_text())
    require(s53["s51_seal_sha256"]==old["s51_seal_sha256"] and
            digest(Path(old["archive"]))==old["archive_sha256"] and
            s53["archive_sha256"]==old["archive_sha256"],"S5.3 provenance failed")
    return s52,s53

def clean_env():
    env=os.environ.copy()
    for key in ("DKR_EMBED_NATIVE_DIR","DKR_EMBED_CUBIN_DIR","CANDLE_DKR_S31_DIR",
                "MISTRALRS_DKR_ARTIFACT_SOURCE","CUDA_COMPUTE_CAP",
                "CUDA_CACHE_PATH","DKR_R124_S53_PHASE","DKR_R124_S53_ROUND",
                "DKR_EMBED_S55_MMQ_DIR"):
        env.pop(key,None)
    env["CANDLE_DKR_TRACE"]="1"
    env["CUDA_CACHE_DISABLE"]="0"
    env["CUDA_CACHE_MAXSIZE"]=str(512*1024*1024)
    return env

def execute(cmd,env,path):
    print("+ "+" ".join(map(str,cmd)),flush=True)
    with path.open("w") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in proc.stdout:
            print(line,end="",flush=True)
            log.write(line)
        result=proc.wait()
    require(result==0,f"command failed exit={result}, log={path}")
    return path.read_text(errors="replace")

def build(args):
    head=source_head()
    seal=seal_gate(args.seal)
    images=images_gate(args.cubin_dir,seal)
    neutral=neutral_gate(args.mmq_dir,seal)
    _,s53=historical_gate(args.s52_result,args.s53_result,seal)
    out=args.out.resolve()
    require(not out.exists(),"output directory exists; do not overwrite")
    out.mkdir(parents=True)
    env=clean_env()
    env["DKR_EMBED_NATIVE_DIR"]=seal["catalogue"]
    env["DKR_EMBED_CUBIN_DIR"]=str(args.cubin_dir.resolve(strict=True))
    env["DKR_EMBED_S55_MMQ_DIR"]=str(args.mmq_dir.resolve(strict=True))
    archive=out/"s55a-tests.tar.zst"
    execute(["cargo","nextest","archive","-p","candle-core","-p","candle-nn",
             "--no-default-features","--features","cuda-dkr-cubin-embedded,cuda-dkr-neutral-mmq",
             "--lib",
             "--test","dkr_s52_numeric","--test","moe_simt_sm61",
             "--test","cuda_legacy_bf16_tests","--test","dkr_catalog_mode",
             "--test","dkr_s53_perf","--archive-file",str(archive)],
            env,out/"archive-build.log")
    require(archive.is_file(),"archive not produced")
    receipt={"schema":"CANDLE-DKR-R124-S55A-ARCHIVE-V1",
             "source_head":head,
             "s51_seal":str(args.seal.resolve(strict=True)),
             "s51_seal_sha256":digest(args.seal.resolve(strict=True)),
             "s52_result":str(args.s52_result.resolve(strict=True)),
             "s52_result_sha256":digest(args.s52_result.resolve(strict=True)),
             "s53_result":str(args.s53_result.resolve(strict=True)),
             "s53_result_sha256":digest(args.s53_result.resolve(strict=True)),
             "cubin_dir":str(args.cubin_dir.resolve(strict=True)),
             "cubin_manifest_sha256":digest(args.cubin_dir.resolve(strict=True)/"manifest.json"),
             "mmq_dir":str(args.mmq_dir.resolve(strict=True)),
             "mmq_manifest_sha256":digest(args.mmq_dir.resolve(strict=True)/"manifest.json"),
             "archive":str(archive),"archive_sha256":digest(archive),
             "families":len(images["families"])}
    (out/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S55A_BUILD=PASS images=25 codecs=10")
    print("DKR_R124_S55A_RECEIPT="+str(out/"receipt.json"))

def verify_provenance(log,manifest,sealed,all_families=True):
    require("CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25" in log
            and "CANDLE_DKR_CUBIN_CATALOG source=embedded images=25 target=sm61" in log,
            "embedded CUBIN catalogue not admitted")
    triples=RE_LOAD.findall(log)
    require(bool(triples),"no loaded CUBIN provenance")
    seen=set()
    for name,cubin_sha,ptx_sha in triples:
        require(name in manifest["families"],"unexpected loaded module "+name)
        require((cubin_sha,ptx_sha)==(
            manifest["families"][name]["cubin_sha256"],
            sealed["catalogue_inventory"]["packages"][name]["ptx_sha256"]),
            "wrong loaded image digest "+name)
        seen.add(name)
    if all_families:
        require(seen==FAMILIES,"missing CUBIN loads: "+",".join(sorted(FAMILIES-seen)))
    return seen

def check_numeric(log):
    cases=re.findall(r"DKR_R124_S52_QUANT_CASE codec=(\w+) dtype=(\w+) batch=(\d+) route=(\w+)",log)
    required={(c,d,str(batch),"MMVQ" if batch<=8 else "MMQ")
              for c in CODECS for d in ("F16","BF16","F32") for batch in (1,8,9,17)}
    require(len(cases)==120 and set(cases)==required,"S5.2 quantized 120-case matrix incomplete")
    moes=re.findall(r"DKR_R124_S52_GGUF_MOE codec=(\w+) rows=4 n=5 k=256",log)
    require(len(moes)==6 and set(moes)==set(MOE_CODECS),"S5.2 MoE matrix incomplete")
    for marker in ("DKR_R124_S52_CORE=PASS",
                   "DKR_R124_S52_QUANT=PASS cases=120",
                   "DKR_R124_S52_GGUF_MOE=PASS codecs=6"):
        require(marker in log,"numeric marker absent: "+marker)
    passed=set(re.findall(r"(?m)^test ([a-zA-Z0-9_]+) \.\.\. ok$",log))
    for name in ("dkr_s52_ten_codecs_mmvq_mmq",
                 "dkr_s52_core_families_cpu_parity",
                 "dkr_s52_gguf_moe_six_codecs",
                 "sm61_simt_prefill_topk2_fp16",
                 "sm61_simt_decode_weighted_fp16"):
        require(name in passed,"missing numeric PASS: "+name)
    return {"quant_cases":len(cases),"gguf_moe":len(moes),"passed_tests":sorted(passed)}

def check_perf(log,round_id):
    triples=RE_CASE.findall(log)
    require(len(triples)==7,"S5.3 benchmark 7-case coverage missing")
    rows={}
    for rid,name,first,p50,p95,minimum,maximum,tokens in triples:
        require(int(rid)==round_id and name in CASES and name not in rows,
                "unexpected benchmark row "+name)
        row={"first_us":int(first),"p50_us":int(p50),"p95_us":int(p95),
             "min_us":int(minimum),"max_us":int(maximum),"tokens":int(tokens)}
        require(0<row["min_us"]<=row["p50_us"]<=row["p95_us"]<=row["max_us"]
                and row["first_us"]>0,"invalid timing "+name)
        rows[name]=row
    require(set(rows)==set(CASES),"missing benchmark case")
    require(f"DKR_R124_S53_TEST=PASS phase=cold round={round_id} cases=7 samples=80" in log,
            "benchmark execution failure")
    return rows

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "requires headless-relogin")
    rp=args.receipt.resolve(strict=True)
    rec=json.loads(rp.read_text())
    require(rec["schema"]=="CANDLE-DKR-R124-S55A-ARCHIVE-V1"
            and rec["source_head"]==source_head(),"source/receipt mismatch")
    for p,key in (("s51_seal","s51_seal_sha256"),
                  ("s52_result","s52_result_sha256"),
                  ("s53_result","s53_result_sha256")):
        require(digest(Path(rec[p]))==rec[key],p+" changed")
    sealed=seal_gate(Path(rec["s51_seal"]))
    manifest=images_gate(Path(rec["cubin_dir"]),sealed)
    neutral=neutral_gate(Path(rec["mmq_dir"]),sealed)
    require(digest(Path(rec["mmq_dir"])/"manifest.json")==rec["mmq_manifest_sha256"],
            "S5.5a MMQ launch table changed")
    require(digest(Path(rec["cubin_dir"])/"manifest.json")==rec["cubin_manifest_sha256"],
            "CUBIN manifest changed")
    _,old=historical_gate(Path(rec["s52_result"]),Path(rec["s53_result"]),sealed)
    archive=Path(rec["archive"])
    require(digest(archive)==rec["archive_sha256"],"Nextest archive changed")
    out=rp.parent/("headless-"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    env=clean_env()
    info=execute(["nvidia-smi","--query-gpu=index,name,compute_cap,driver_version,temperature.gpu,pstate,clocks.gr,clocks.mem,memory.used",
                  "--format=csv"],env,out/"gpu-before.log")
    require(", 6.1," in info,"GPU is not SM61")
    cmd=["cargo","nextest","run","--archive-file",str(archive),
         "--workspace-remap",str(ROOT),"--run-ignored","all",
         "--no-capture","--no-fail-fast","-E"]
    table_test=execute(cmd+["test(=dkr_s55a_neutral_mmq_matches_frozen_s31_qualified_shapes)"],
                       env,out/"s55a-table-parity.log")
    require("DKR_R124_S55A_TABLE_PARITY=PASS codecs=10" in table_test,
            "new MMQ geometry differs from frozen S3.1 reference")
    numeric=execute(cmd+["test(dkr_s52_) | test(sm61_simt_) | test(cuda_legacy_bf16_)"],
                    env,out/"s52-cubin-numeric.log")
    coverage=verify_provenance(numeric,manifest,sealed)
    cases=check_numeric(numeric)
    require("DKR_R124_S55A_MMQ_CATALOG source=embedded codecs=10 target=sm61" in numeric,
            "S5.5a MMQ table not admitted in runtime")
    plan_codecs=set(re.findall(r"DKR_R124_S55A_MMQ_PLAN source=embedded codec=(q[a-z0-9_]+) tile=",numeric))
    require(plan_codecs==set(neutral["codecs"]),
            "new MMQ table not exercised for all ten codecs")
    # Strict CUBIN mode must preserve S4.4 anti-override behavior.
    denied_source=dict(env,MISTRALRS_DKR_ARTIFACT_SOURCE="filesystem")
    reject_source=execute(cmd+["test(=dkr_embedded_rejects_filesystem)"],
                          denied_source,out/"reject-runtime-filesystem.log")
    require("test dkr_embedded_rejects_filesystem ... ok" in reject_source,
            "filesystem override not rejected")
    denied_dir=dict(env,CANDLE_DKR_S31_DIR="/tmp/never-valid-native-package")
    reject_dir=execute(cmd+["test(=dkr_embedded_rejects_directory_override)"],
                       denied_dir,out/"reject-runtime-dir.log")
    require("test dkr_embedded_rejects_directory_override ... ok" in reject_dir,
            "native directory override not rejected")
    results=[]
    for round_id in (1,2,3):
        cache=out/f"cubin-cache-r{round_id}"
        cache.mkdir()
        require(not list(cache.iterdir()),"expected private empty CUDA cache")
        benv=dict(env,CUDA_CACHE_PATH=str(cache),
                  DKR_R124_S53_PHASE="cold",DKR_R124_S53_ROUND=str(round_id))
        execute(["nvidia-smi","--query-gpu=pstate,temperature.gpu,clocks.gr,clocks.mem,memory.used",
                 "--format=csv"],benv,out/f"gpu-before-r{round_id}.log")
        output=execute(cmd+["test(=dkr_s53_bench)"],benv,out/f"benchmark-r{round_id}.log")
        seen=verify_provenance(output,manifest,sealed,all_families=False)
        require({"mmvq_gguf","mmq_quantize","mmq_instance_q4_k",
                 "mmq_instance_q8_0","moe_gguf","moe_simt_f16"}<=seen,
                 "benchmark missed a native CUBIN family")
        results.append(check_perf(output,round_id))
        execute(["nvidia-smi","--query-gpu=pstate,temperature.gpu,clocks.gr,clocks.mem,memory.used",
                 "--format=csv"],benv,out/f"gpu-after-r{round_id}.log")
    perf={}
    for name in CASES:
        new=statistics.median(r[name]["first_us"] for r in results)
        old_us=old["summary"][name]["cold"]["first_us_median_us"]
        new_p50=statistics.median(r[name]["p50_us"] for r in results)
        old_p50=old["summary"][name]["cold"]["p50_us_median_us"]
        perf[name]={"ptx_cold_first_us":old_us,"cubin_first_us_median":new,
                    "first_ratio_ptx_over_cubin":round(old_us/max(1,new),3),
                    "ptx_cold_p50_us":old_p50,"cubin_p50_us_median":new_p50,
                    "steady_state_p50_ratio_ptx_over_cubin":round(old_p50/max(1,new_p50),3)}
    report={"schema":"CANDLE-DKR-R124-S55A-RESULT-V1",
            "status":"PASS","source_head":rec["source_head"],
            "archive_sha256":rec["archive_sha256"],
            "numeric":cases,"cubin_modules_loaded":sorted(coverage),
            "neutral_mmq_codecs":sorted(plan_codecs),
            "neutral_mmq_manifest_sha256":rec["mmq_manifest_sha256"],
            "benchmark":perf,"benchmark_raw":results,
            "scope":"25-family S5.2 numerical parity and 7-case S5.3 first-use CUBIN comparison",
            "notes":["All 25 CUBIN modules are actually loaded through Candle",
                     "S5.2 tolerances are unchanged, not bit-exact PTX/CUBIN parity",
                     "The original PTX results are separate historical runs",
                     "First-use comparisons include host overhead and device synchronizations"]}
    (out/"result.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(f"DKR_R124_S55A_NUMERIC=PASS cases={cases['quant_cases']} gguf_moe={cases['gguf_moe']}")
    print("DKR_R124_S55A_CUBIN_MODULES=25/25")
    print("DKR_R124_S55A_NEUTRAL_MMQ=PASS codecs=10")
    print("DKR_R124_S55A_PERF=PASS rounds=3 cases=7")
    print("DKR_R124_S55A_REPORT="+str(out/"result.json"))
    print("DKR_R124_S55A_PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("build")
    b.add_argument("--seal",type=Path,required=True)
    b.add_argument("--cubin-dir",type=Path,required=True)
    b.add_argument("--mmq-dir",type=Path,required=True)
    b.add_argument("--s52-result",type=Path,required=True)
    b.add_argument("--s53-result",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True)
    h=sub.add_parser("headless")
    h.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    (build if args.mode=="build" else headless)(args)

if __name__=="__main__":
    main()
