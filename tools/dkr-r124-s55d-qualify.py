#!/usr/bin/env python3
"""R1.24 S5.5d: direct numeric PTX/CUBIN + offline Qwen3 GGUF cold start.

Separate archives/binaries, identical source, weights and token inputs,
fresh CUDA driver-cache directories, new GPU processes, full FP32 output
comparison. Never rebuilds CUDA kernels or changes S5.1/S5.4b seals.
"""
import argparse
import array
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location(
    "s55c_qualification",Path(__file__).with_name("dkr-r124-s55c-qualify.py"))
s55c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s55c)
s55b=s55c.s55b
SCHEMA="CANDLE-DKR-R124-S55D-ARCHIVE-V1"
RESULT_SCHEMA="CANDLE-DKR-R124-S55D-RESULT-V1"
DEFAULT_TOKENS="1,2,3,4,5,6,7,8"
MODES=("ptx","cubin")
METRICS=("device_init_ms","gguf_header_ms","model_build_ms",
         "prefill_first_ms","total_process_work_ms")
EXPECTED_NUMERIC={"gguf":120,"bf16":4,"fp8":10,"fp4":15,"moe":8}

def require(ok,msg):
    if not ok: raise SystemExit("DKR_R124_S55D_FAIL: "+msg)

def env_base():
    e=s55b.clean_env()
    e.pop("DKR_R124_S53_PHASE",None)
    e.pop("DKR_R124_S53_ROUND",None)
    e.pop("DKR_S55D_OUTPUT",None)
    e.pop("DKR_S55D_MODE",None)
    return e

def rust_env(mode,verified,cubins):
    e=env_base()
    e["DKR_EMBED_NATIVE_DIR"]=verified["seal"]["catalogue"]
    if mode=="cubin":
        root=Path(verified["catalogue_dir"])
        e["DKR_EMBED_CUBIN_DIR"]=str(cubins)
        e["DKR_EMBED_S55_MMQ_DIR"]=str(root/"mmq")
        e["DKR_EMBED_S55B_RECIPES_DIR"]=str(root/"recipes")
        e["DKR_EMBED_S55C_CUDAFORGE_DIR"]=str(root)
    return e

def execute(cmd,env,log):
    print("+ "+" ".join(str(x) for x in cmd),flush=True)
    with log.open("w") as file:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,errors="replace")
        for line in p.stdout:
            print(line,end="",flush=True)
            file.write(line)
        code=p.wait()
    require(code==0,f"command failed exit={code}; log={log}")
    return log.read_text(errors="replace")

def ensure_model(path):
    p=path.resolve(strict=True)
    require(p.is_file() and p.suffix.lower()==".gguf",
            "required local Qwen3 .gguf model not found")
    require(p.stat().st_size>1024*1024,
            "GGUF model is implausibly small")
    return p

def reference_gate(path,verified):
    p=path.resolve(strict=True)
    x=json.loads(p.read_text())
    require(x["schema"]=="CANDLE-DKR-R124-S55C-RESULT-V1"
            and x["status"]=="PASS" and x["backend"]=="cudaforge"
            and x["families"]==25 and x["recipes"]==4
            and x["target_cc"]==61
            and x["cudaforge_catalogue_sha256"]==verified["catalogue_sha256"],
            "S5.5c qualified Cudaforge report inconsistent")
    prior=Path(x["s55b_numeric_and_perf_report"])
    require(prior.is_file()
            and s55b.digest(prior)==x["s55b_report_sha256"],
            "referenced S5.5c numerical report modified")
    return p

def build(args):
    head=s55b.source_head()
    model=ensure_model(args.model)
    seal_path=args.seal.resolve(strict=True)
    cubins=args.cubin_dir.resolve(strict=True)
    verified=s55c.verify(args.catalogue_dir,seal_path,cubins)
    s55b.historical_gate(args.s52_result,args.s53_result,verified["seal"])
    previous=reference_gate(args.s55c_result,verified)
    tokens=[int(x) for x in args.tokens.split(",")]
    require(1<=len(tokens)<=64 and all(0<=x<2000000 for x in tokens),
            "invalid fixed prompt tokens")
    require(2<=args.decode<=32,"model decode must be between 2 and 32")
    out=args.out.resolve()
    require(not out.exists(),"output directory already exists; no overwrite")
    out.mkdir(parents=True)
    (out/"bin").mkdir()
    receipts={}
    for mode in MODES:
        env=rust_env(mode,verified,cubins)
        feature="cuda-dkr-embedded" if mode=="ptx" else "cuda-dkr-cudaforge-catalog"
        archive=out/f"s55d-{mode}-tests.tar.zst"
        execute([
            "cargo","nextest","archive",
            "-p","candle-nn","--no-default-features","--features",feature,
            "--test","dkr_s55d_parity","--archive-file",str(archive)
        ],env,out/f"archive-{mode}.log")
        require(archive.is_file(),f"missing {mode} Nextest archive")
        # Keep the executable in the root workspace to reuse Cargo's
        # existing cache. No candle-examples (whose CUDA build.rs uses NVCC).
        model_feature=mode
        execute([
            "cargo","build","--release","-p","dkr-s55d-model-probe",
            "--no-default-features","--features",model_feature
        ],env,out/f"model-build-{mode}.log")
        src=ROOT/"target/release/dkr-s55d-model-probe"
        require(src.is_file(),f"model probe executable not found for {mode}")
        dst=out/"bin"/f"model-{mode}"
        shutil.copy2(src,dst)
        dst.chmod(dst.stat().st_mode|0o111)
        probe_log=execute([str(dst),"--self-test"],env,out/f"model-self-test-{mode}.log")
        require(f"DKR_R124_S55D_MODEL_PROBE=PASS mode={mode} no_cuda_build=1" in probe_log,
                f"{mode}: model probe did not pass CPU-only startup preflight")
        receipts[mode]={
            "archive":str(archive),"archive_sha256":s55b.digest(archive),
            "model_bin":str(dst),"model_bin_sha256":s55b.digest(dst)}
    payload={
        "schema":SCHEMA,"source_head":head,
        "model":str(model),"model_sha256":s55b.digest(model),
        "model_bytes":model.stat().st_size,"model_family":"qwen3-gguf",
        "tokens":tokens,"decode":args.decode,
        "seal":str(seal_path),"seal_sha256":s55b.digest(seal_path),
        "cubin_dir":str(cubins),
        "cubin_manifest_sha256":s55b.digest(cubins/"manifest.json"),
        "catalogue_dir":verified["catalogue_dir"],
        "catalogue_sha256":verified["catalogue_sha256"],
        "s55c_result":str(previous),"s55c_result_sha256":s55b.digest(previous),
        "artifacts":receipts,
    }
    (out/"receipt.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print("DKR_R124_S55D_BUILD=PASS modes=ptx,cubin numeric=157 model=qwen3-gguf")
    print("DKR_R124_S55D_RECEIPT="+str(out/"receipt.json"))

def numerical_tolerance(key):
    if key.startswith("gguf/"):
        return 0.001,0.001
    if key.startswith("bf16/"):
        return 0.005,0.002
    if key.startswith("fp8/"):
        return 0.01,0.005
    if key.startswith("fp4/"):
        return 0.005,0.002
    if key.startswith("moe/"):
        return 0.005,0.002
    raise ValueError("unknown numeric family: "+key)

def compare_values(left,right,atol,rtol,label):
    require(len(left)==len(right) and len(left)>0,label+": vector cardinality")
    max_abs=max_rel=max_scaled=0.0
    count=0
    examples=[]
    exact=True
    for i,(a,b) in enumerate(zip(left,right)):
        require(math.isfinite(a) and math.isfinite(b),
                label+f": nonfinite value index={i}")
        d=abs(a-b)
        scale=atol+rtol*max(abs(a),abs(b))
        rel=d/max(1e-12,max(abs(a),abs(b)))
        max_abs=max(max_abs,d)
        max_rel=max(max_rel,rel)
        max_scaled=max(max_scaled,d/max(scale,1e-12))
        if a!=b:
            exact=False
        if d>scale:
            count+=1
            if len(examples)<8:
                examples.append({"index":i,"ptx":a,"cubin":b,
                                 "abs_error":d,"tolerance":scale})
    return {"elements":len(left),"max_abs":max_abs,"max_relative":max_rel,
            "max_tolerance_ratio":max_scaled,"mismatches":count,
            "examples":examples,"bit_equal_float_values":exact,
            "atol":atol,"rtol":rtol}

def numeric_compare(a,b):
    x=json.loads(a.read_text())
    y=json.loads(b.read_text())
    require(x["schema"]==y["schema"]=="CANDLE-DKR-R124-S55D-NUMERIC-V1"
            and x["mode"]=="ptx" and y["mode"]=="cubin"
            and x["fixture"]==y["fixture"]=="S55D_FIXED_INPUTS_V1",
            "numeric reports are not a common source fixture")
    require(x["quantized_weight_sha256"]==y["quantized_weight_sha256"]
            and len(x["quantized_weight_sha256"])==16,
            "bit-identical GGUF quantized source weights not established")
    ac=x["cases"];bc=y["cases"]
    require(set(ac)==set(bc) and len(ac)==157,
            "PTX/CUBIN numeric test matrices differ")
    counts={k:sum(name.startswith(k+"/") for name in ac)
            for k in EXPECTED_NUMERIC}
    require(counts==EXPECTED_NUMERIC,
            "157-case matrix coverage is incomplete: "+str(counts))
    rows={}
    mismatch=0
    n=0
    for name in sorted(ac):
        pa=ac[name];pb=bc[name]
        require(pa["shape"]==pb["shape"] and pa["dtype"]==pb["dtype"],
                "different output shapes or dtypes for "+name)
        atol,rtol=numerical_tolerance(name)
        row=compare_values(pa["values"],pb["values"],atol,rtol,name)
        rows[name]=row
        n+=row["elements"]
        mismatch+=row["mismatches"]
    require(n>70000,"too few full tensor elements compared")
    require(mismatch==0,f"{mismatch} PTX/CUBIN tensor elements exceed tolerances")
    return {"status":"PASS","cases":len(rows),"elements":n,
            "family_cases":counts,
            "bitwise_float_equal_cases":sum(v["bit_equal_float_values"] for v in rows.values()),
            "max_abs":max(v["max_abs"] for v in rows.values()),
            "max_tolerance_ratio":max(v["max_tolerance_ratio"] for v in rows.values()),
            "per_case":rows}

def scalar_f32(stream):
    # F32 Little Endian streaming decoder, no numpy/torch requirement.
    buf=stream.read(4*8192)
    require(len(buf)%4==0,"model logits file is not float32-aligned")
    if not buf:return None
    values=array.array("f")
    values.frombytes(buf)
    if sys.byteorder!="little":
        values.byteswap()
    return values

def compare_binary_logits(ptx,cubin,shape,atol=0.01,rtol=0.005):
    require(ptx.stat().st_size==cubin.stat().st_size
            and ptx.stat().st_size==shape[0]*shape[1]*4,
            "PTX/CUBIN full logits binary shape differs")
    with ptx.open("rb") as pa,cubin.open("rb") as pb:
        n=0;bad=0;max_abs=max_rel=0.0;example=[]
        while True:
            a=scalar_f32(pa);b=scalar_f32(pb)
            if a is None or b is None:
                require(a is None and b is None,"truncated logits")
                break
            row=compare_values(a,b,atol,rtol,"full-model-logits")
            n+=row["elements"];bad+=row["mismatches"]
            max_abs=max(max_abs,row["max_abs"])
            max_rel=max(max_rel,row["max_relative"])
            if row["examples"] and not example:
                example=row["examples"]
    return {"elements":n,"max_abs":max_abs,"max_relative":max_rel,
            "mismatches":bad,"atol":atol,"rtol":rtol,"examples":example}

def gpu_info(env,path):
    result=subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,driver_version,temperature.gpu,pstate,clocks.gr,clocks.mem,memory.used",
        "--format=csv"],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    path.write_text(result.stdout)
    print(result.stdout.strip(),flush=True)
    require("6.1" in result.stdout,"target GPU not SM61")
    return result.stdout.strip()

def verify_mode_log(log,mode):
    require("CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25" in log,
            f"{mode}: embedded native catalog not loaded")
    if mode=="cubin":
        require("DKR_R124_S55C_CATALOGUE_ADMITTED producer=cudaforge" in log
                and "CANDLE_DKR_IMAGE_LOADED" in log
                and "format=cubin" in log,
                "CUBIN runtime did not load attested Cudaforge image")
    else:
        require("DKR_R124_S55C_CATALOGUE_ADMITTED" not in log
                and "CANDLE_DKR_IMAGE_LOADED" not in log,
                "PTX baseline unexpectedly used CUBIN")

def run_one(cmd,env,log):
    start=time.perf_counter()
    out=execute(cmd,env,log)
    return out,(time.perf_counter()-start)*1000.

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "S5.5d requires an independent headless GPU session")
    receipt_path=args.receipt.resolve(strict=True)
    receipt=json.loads(receipt_path.read_text())
    require(receipt["schema"]==SCHEMA,"wrong S5.5d qualification receipt")
    # Reuse S5.5b's ancestral-diff guard for compiled native Rust inputs,
    # then explicitly add our model probe and numerical capture sources.
    s55b.check_archived_source(receipt["source_head"])
    # S5.5b's compiler-source guard predates this standalone model crate.
    # Guard the exact additional Rust inputs against script-only updates.
    require(subprocess.run([
        "git","diff","--quiet",receipt["source_head"],"HEAD","--",
        "tools/dkr-s55d-model-probe", "candle-nn/tests/dkr_s55d_parity.rs",
        "candle-nn/Cargo.toml"
    ],cwd=ROOT).returncode==0,
        "S5.5d model or numeric Rust sources changed since the archive build")
    require(s55b.digest(Path(receipt["model"]))==receipt["model_sha256"]
            and Path(receipt["model"]).stat().st_size==receipt["model_bytes"],
            "model GGUF changed after binary compilation")
    verified=s55c.verify(Path(receipt["catalogue_dir"]),
                          Path(receipt["seal"]),Path(receipt["cubin_dir"]))
    require(verified["catalogue_sha256"]==receipt["catalogue_sha256"]
            and s55b.digest(Path(receipt["seal"]))==receipt["seal_sha256"]
            and s55b.digest(Path(receipt["cubin_dir"])/"manifest.json")==receipt["cubin_manifest_sha256"],
            "sealed catalogue/CUBIN provenance changed")
    prior=reference_gate(Path(receipt["s55c_result"]),verified)
    require(s55b.digest(prior)==receipt["s55c_result_sha256"],
            "S5.5c qualification result modified")
    for mode in MODES:
        artifacts=receipt["artifacts"][mode]
        for item,sha in [("archive","archive_sha256"),
                         ("model_bin","model_bin_sha256")]:
            require(s55b.digest(Path(artifacts[item]))==artifacts[sha],
                    f"{mode}: compiled artifact changed")
    out=receipt_path.parent/("headless-"+datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True)
    env=env_base()
    gpu_info(env,out/"gpu-before.log")
    outputs={}
    for mode in MODES:
        sample=out/f"numeric-{mode}.json"
        archive=receipt["artifacts"][mode]["archive"]
        cache=out/f"numeric-cache-{mode}"
        cache.mkdir()
        require(not any(cache.iterdir()),"numeric CUDA cache must be empty")
        e=dict(env,CUDA_CACHE_PATH=str(cache),
               DKR_S55D_OUTPUT=str(sample),DKR_S55D_MODE=mode)
        cmd=["cargo","nextest","run","--archive-file",archive,
             "--workspace-remap",str(ROOT),"--run-ignored","all",
             "--no-capture","-E","test(dkr_s55d_capture_outputs)"]
        log,wall=run_one(cmd,e,out/f"numeric-{mode}.log")
        require(f"DKR_R124_S55D_CAPTURE=PASS mode={mode} cases=157" in log,
                f"{mode}: numeric matrix not completed")
        verify_mode_log(log,mode)
        require(sample.is_file(),f"{mode}: numeric tensor file missing")
        outputs[mode]=sample
        gpu_info(env,out/f"gpu-after-numeric-{mode}.log")
    numeric=numeric_compare(outputs["ptx"],outputs["cubin"])
    print(f"DKR_R124_S55D_NUMERIC=PASS cases={numeric['cases']} elements={numeric['elements']}")
    (out/"numeric-delta.json").write_text(json.dumps(numeric,sort_keys=True,indent=2)+"\n")

    model_reports={"ptx":[],"cubin":[]}
    comparisons=[]
    for round_id in range(1,4):
        # Alternate order to reduce confounding from clocks/temperature.
        order=MODES if round_id%2 else tuple(reversed(MODES))
        both={}
        for mode in order:
            prefix=out/f"model-{mode}-r{round_id}"
            mjson=prefix.with_suffix(".json")
            binary=prefix.with_suffix(".f32le")
            cache=out/f"model-cache-{mode}-r{round_id}"
            cache.mkdir()
            require(not any(cache.iterdir()),"model CUDA cache must be fresh")
            e=dict(env,CUDA_CACHE_PATH=str(cache))
            gpu_info(e,out/f"gpu-before-{mode}-r{round_id}.log")
            cmd=[receipt["artifacts"][mode]["model_bin"],
                 "--model",receipt["model"],
                 "--tokens",",".join(map(str,receipt["tokens"])),
                 "--decode",str(receipt["decode"]),
                 "--result",str(mjson),"--logits",str(binary)]
            log,wall=run_one(cmd,e,out/f"model-{mode}-r{round_id}.log")
            require(f"DKR_R124_S55D_MODEL=PASS mode={mode}" in log,
                    f"{mode} model did not complete")
            verify_mode_log(log,mode)
            gpu_info(e,out/f"gpu-after-{mode}-r{round_id}.log")
            q=json.loads(mjson.read_text())
            require(q["schema"]=="CANDLE-DKR-R124-S55D-MODEL-V1"
                    and q["backend"]==mode and q["target_cc"]==61
                    and q["model"]==receipt["model"]
                    and q["prompt_token_ids"]==receipt["tokens"]
                    and q["decode_steps"]==receipt["decode"]
                    and q["logits_bytes"]==binary.stat().st_size,
                    f"{mode}: model trial provenance or tensor shape mismatch")
            require(all(isinstance(q[key],(int,float)) and
                        math.isfinite(q[key]) and q[key]>0
                        for key in METRICS),
                    "invalid model timing")
            q["wall_ms"]=wall
            both[mode]=(q,binary)
            model_reports[mode].append(q)
        a,pa=both["ptx"];bc,pb=both["cubin"]
        require(a["generated_token_ids"]==bc["generated_token_ids"],
                f"model greedy token divergence in round {round_id}")
        require(a["vocabulary"]==bc["vocabulary"]
                and a["tensor_count"]==bc["tensor_count"],
                "different Qwen3 vocabulary/tensor count")
        diff=compare_binary_logits(pa,pb,
            (receipt["decode"]+1,a["vocabulary"]))
        require(diff["mismatches"]==0,
                f"full Qwen3 logits mismatch at round {round_id}: {diff['examples']}")
        comparisons.append(diff)
        print(f"DKR_R124_S55D_MODEL_PAIR=PASS round={round_id} logits={diff['elements']} tokens={len(a['generated_token_ids'])}",flush=True)
    require(len(comparisons)==3,"model round count changed")
    require(s55b.digest(Path(receipt["model"]))==receipt["model_sha256"],
            "GGUF changed during comparison")
    summary={}
    for key in METRICS+("wall_ms",):
        ptx=statistics.median(q[key] for q in model_reports["ptx"])
        cubin=statistics.median(q[key] for q in model_reports["cubin"])
        summary[key]={"ptx_median_ms":ptx,"cubin_median_ms":cubin,
                      "ptx_over_cubin":ptx/max(1e-9,cubin)}
    report={
        "schema":RESULT_SCHEMA,"status":"PASS","source_head":receipt["source_head"],
        "s55c_catalogue_sha256":receipt["catalogue_sha256"],
        "model_gguf_sha256":receipt["model_sha256"],
        "model_path":receipt["model"],"target_cc":61,
        "archive_sha256":{mode:receipt["artifacts"][mode]["archive_sha256"] for mode in MODES},
        "binary_sha256":{mode:receipt["artifacts"][mode]["model_bin_sha256"] for mode in MODES},
        "numerical":numeric,"model_rounds":3,
        "model_comparisons":comparisons,
        "model_timing_summary":summary,
        "model_reports":model_reports,
        "limitations":[
            "Cold CUDA process/isolated JIT cache, NOT cold OS filesystem page cache",
            "PTX vs CUBIN tested in separate new processes with alternating order",
            "No claim of equal bit patterns unless exact float equality is reported",
            "Qwen3 greedy tokens and full logits under explicit tolerances, not all possible prompts/models",
            "GPU clocks recorded before/after; kernel-execution clock histories not forced",
        ],
    }
    report_path=out/"result.json"
    report_path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S55D_MODEL=PASS rounds=3 full_logits_comparison=1")
    print("DKR_R124_S55D_PERF=PASS rounds=3 modes=ptx,cubin")
    print("DKR_R124_S55D_REPORT="+str(report_path))
    print("DKR_R124_S55D_PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("build")
    b.add_argument("--seal",type=Path,required=True)
    b.add_argument("--cubin-dir",type=Path,required=True)
    b.add_argument("--catalogue-dir",type=Path,required=True)
    b.add_argument("--s55c-result",type=Path,required=True)
    b.add_argument("--s52-result",type=Path,required=True)
    b.add_argument("--s53-result",type=Path,required=True)
    b.add_argument("--model",type=Path,required=True)
    b.add_argument("--tokens",default=DEFAULT_TOKENS)
    b.add_argument("--decode",type=int,default=8)
    b.add_argument("--out",type=Path,required=True)
    h=sub.add_parser("headless")
    h.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    (build if args.mode=="build" else headless)(args)

if __name__=="__main__":
    main()
