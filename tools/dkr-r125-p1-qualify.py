#!/usr/bin/env python3
"""R1.25 P1 SM61 canary: verified CUBIN default, explicit PTX rollback.

Builds one feature-unified Rust archive and one Qwen3 probe. Runs the same
binaries with auto, cubin and ptx policies; rejects silent fallback and
unverified family/image selection. Reuses, never rebuilds, S5.1/S5.5c CUBINs.
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location("s55d",Path(__file__).with_name("dkr-r124-s55d-qualify.py"))
s55d=importlib.util.module_from_spec(sp)
sp.loader.exec_module(s55d)
s55b=s55d.s55b
MODES=("auto","ptx","cubin")
SCHEMA="CANDLE-DKR-R125-P1-RECEIPT-V1"

def require(ok,why):
    if not ok: raise SystemExit("DKR_R125_P1_FAIL: "+why)

def source_environment(bundle, cubins):
    e=s55d.rust_env("cubin",bundle,cubins)
    e.pop("CANDLE_DKR_IMAGE_MODE",None)
    return e

def load_baseline(path,verified,model):
    p=path.resolve(strict=True)
    report=json.loads(p.read_text())
    require(report["schema"]=="CANDLE-DKR-R124-S55D-RESULT-V1"
            and report["status"]=="PASS" and report["target_cc"]==61
            and report["s55c_catalogue_sha256"]==verified["catalogue_sha256"]
            and report["model_gguf_sha256"]==s55b.digest(model)
            and report["numerical"]["cases"]==157
            and report["numerical"]["mismatches"]==0
            and report["model_rounds"]==3,
            "S5.5d reference is missing or not qualified for this model")
    return p

def build(args):
    source_head=s55b.source_head()
    model=s55d.ensure_model(args.model)
    cubins=args.cubin_dir.resolve(strict=True)
    seal=args.seal.resolve(strict=True)
    verified=s55d.s55c.verify(args.catalogue_dir,seal,cubins)
    reference=load_baseline(args.s55d_result,verified,model)
    out=args.out.resolve()
    require(not out.exists(),"output exists; keep past qualification immutable")
    out.mkdir(parents=True)
    (out/"bin").mkdir()
    env=source_environment(verified,cubins)
    archive=out/"promotion-tests.tar.zst"
    s55d.execute(["cargo","nextest","archive",
        "-p","candle-core","-p","candle-nn",
        "--no-default-features","--features","cuda-dkr-qualified-default",
        "--lib","--test","dkr_s55d_parity","--archive-file",str(archive)],
        env,out/"archive-build.log")
    s55d.execute(["cargo","build","--release","-p","dkr-s55d-model-probe",
        "--no-default-features","--features","promoted"],
        env,out/"model-build.log")
    target=ROOT/"target/release/dkr-s55d-model-probe"
    require(target.is_file(),"promoted Qwen3 executable missing")
    binary=out/"bin/promoted-qwen3"
    shutil.copy2(target,binary)
    binary.chmod(binary.stat().st_mode|0o111)
    probe=s55d.execute([str(binary),"--self-test"],env,out/"model-preflight.log")
    require("DKR_R124_S55D_MODEL_PROBE=PASS mode=promoted no_cuda_build=1" in probe,
            "promoted model probe self-test failed")
    receipt={
        "schema":SCHEMA,"source_head":source_head,"target_cc":61,
        "s55d_result":str(reference),"s55d_result_sha256":s55b.digest(reference),
        "model":str(model),"model_sha256":s55b.digest(model),
        "catalogue_dir":verified["catalogue_dir"],
        "catalogue_sha256":verified["catalogue_sha256"],
        "cubin_dir":str(cubins),
        "cubin_manifest_sha256":s55b.digest(cubins/"manifest.json"),
        "seal":str(seal),"seal_sha256":s55b.digest(seal),
        "archive":str(archive),"archive_sha256":s55b.digest(archive),
        "model_binary":str(binary),"model_binary_sha256":s55b.digest(binary),
        "tokens":[1,2,3,4,5,6,7,8],"decode":8,
    }
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("DKR_R125_P1_BUILD=PASS binaries=1 policies=auto,ptx,cubin")
    print("DKR_R125_P1_RECEIPT="+str(out/"receipt.json"))

def env_for(base,policy,cache):
    env=dict(base)
    env["CUDA_CACHE_PATH"]=str(cache)
    if policy=="auto":
        env.pop("CANDLE_DKR_IMAGE_MODE",None)
    else:
        env["CANDLE_DKR_IMAGE_MODE"]=policy
    return env

def check_log(log,mode):
    require("DKR_R125_P1_POLICY default=cubin requested="+mode in log,
            "runtime policy trace missing for "+mode)
    require("DKR_R124_S55C_CATALOGUE_ADMITTED producer=cudaforge" in log
            and "CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25" in log,
            "producer provenance missing for "+mode)
    lines=[l for l in log.splitlines() if "CANDLE_DKR_IMAGE_LOADED " in l]
    require(bool(lines),"no kernel images loaded in "+mode)
    expected="format=ptx" if mode=="ptx" else "format=cubin"
    require(all(expected in line for line in lines),
            "unexpected mixed image format under mode "+mode)
    require("DKR_R125_P1_FALLBACK" not in log,
            "CUBIN admission failed; not safe to promote default")
    if mode!="ptx":
        require(all("cubin_sha256=" in line and "ptx_sha256=" in line for line in lines),
                "CUBIN artifact identities absent")

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "headless qualification requires no graphical session")
    receipt_path=args.receipt.resolve(strict=True)
    r=json.loads(receipt_path.read_text())
    require(r["schema"]==SCHEMA and r["target_cc"]==61,"wrong promotion receipt")
    s55b.check_archived_source(r["source_head"])
    diff=subprocess.run(["git","diff","--quiet",r["source_head"],"HEAD","--",
        "tools/dkr-s55d-model-probe","tools/dkr-r124-feature-audit.py"],
        cwd=ROOT)
    require(diff.returncode==0,"compiled or feature-audit source changed; rebuild")
    source=s55d.s55c.verify(Path(r["catalogue_dir"]),Path(r["seal"]),Path(r["cubin_dir"]))
    require(source["catalogue_sha256"]==r["catalogue_sha256"]
            and s55b.digest(Path(r["seal"]))==r["seal_sha256"]
            and s55b.digest(Path(r["cubin_dir"])/"manifest.json")==r["cubin_manifest_sha256"],
            "sealed catalogue or CUBIN inventory changed")
    load_baseline(Path(r["s55d_result"]),source,Path(r["model"]))
    for key,path in [
        ("s55d_result_sha256",r["s55d_result"]),
        ("model_sha256",r["model"]),
        ("archive_sha256",r["archive"]),
        ("model_binary_sha256",r["model_binary"])]:
        require(s55b.digest(Path(path))==r[key],f"SHA-256 changed: {key}")
    out=receipt_path.parent/("headless-"+datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True)
    env=source_environment(source,Path(r["cubin_dir"]))
    s55d.gpu_info(env,out/"gpu-before.log")
    # Unit tests do not run the GPU: validate option parser and compiled
    # Cudaforge/neutral recipe admission before the model campaign.
    filters=(
        "test(qualified_default_policy_has_explicit_rollback_only) | "
        "test(dkr_s55c_cudaforge_catalogue_build_attestation) | "
        "test(dkr_s55b_typed_recipes_match_qualified_s31)"
    )
    preflight=s55d.execute([
        "cargo","nextest","run","--archive-file",r["archive"],
        "--workspace-remap",str(ROOT),"--run-ignored","all","--no-capture",
        "-E",filters
    ],env,out/"policy-preflight.log")
    require("DKR_R125_P1_POLICY=PASS auto=cubin rollback=ptx invalid=reject" in preflight,
            "policy parser unit test missing")
    require("DKR_R124_S55B_TYPED_PARITY=PASS" in preflight,
            "typed producer recipes not qualified")

    # Reject unsupported policy before loading a CUDA module/function.
    negative=out/"bad-policy.log"
    bad_env=dict(env,CANDLE_DKR_IMAGE_MODE="unsupported",
                 DKR_S55D_MODE="cubin",
                 DKR_S55D_OUTPUT=str(out/"bad-policy.json"))
    bad=subprocess.run([
        "cargo","nextest","run","--archive-file",r["archive"],
        "--workspace-remap",str(ROOT),"--run-ignored","all",
        "--no-capture","-E","test(dkr_s55d_capture_outputs)"
    ],cwd=ROOT,env=bad_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
      text=True,errors="replace")
    negative.write_text(bad.stdout)
    require(bad.returncode!=0
            and "CANDLE_DKR_IMAGE_MODE must be auto, cubin or ptx" in bad.stdout
            and not (out/"bad-policy.json").exists(),
            "unknown policy did not fail before inference")

    numeric={}
    for mode in MODES:
        path=out/f"numeric-{mode}.json"
        cache=out/f"numeric-cache-{mode}"
        cache.mkdir()
        e=env_for(env,mode,cache)
        e["DKR_S55D_MODE"]="ptx" if mode=="ptx" else "cubin"
        e["DKR_S55D_OUTPUT"]=str(path)
        log=s55d.execute([
            "cargo","nextest","run","--archive-file",r["archive"],
            "--workspace-remap",str(ROOT),"--run-ignored","all",
            "--no-capture","-E","test(dkr_s55d_capture_outputs)"
        ],e,out/f"numeric-{mode}.log")
        check_log(log,mode)
        require(path.is_file(),f"{mode}: numerical outputs absent")
        numeric[mode]=path
    comparisons={
        "ptx_vs_auto":s55d.numeric_compare(numeric["ptx"],numeric["auto"]),
        "ptx_vs_cubin":s55d.numeric_compare(numeric["ptx"],numeric["cubin"]),
    }
    require(all(x["status"]=="PASS" and x["cases"]==157
                and x["mismatches"]==0 for x in comparisons.values()),
            "numeric PTX/CUBIN/auto outputs differ")
    (out/"numeric-comparison.json").write_text(json.dumps(
        comparisons,indent=2,sort_keys=True)+"\n")
    print("DKR_R125_P1_NUMERIC=PASS policies=auto,ptx,cubin cases=157")

    model={}
    for mode in MODES:
        cache=out/f"model-cache-{mode}"
        cache.mkdir()
        e=env_for(env,mode,cache)
        report=out/f"model-{mode}.json"
        logits=out/f"model-{mode}.f32le"
        log=s55d.execute([
            r["model_binary"],"--model",r["model"],
            "--tokens",",".join(map(str,r["tokens"])),"--decode",str(r["decode"]),
            "--result",str(report),"--logits",str(logits)
        ],e,out/f"model-{mode}.log")
        check_log(log,mode)
        data=json.loads(report.read_text())
        require(data["schema"]=="CANDLE-DKR-R124-S55D-MODEL-V1"
                and data["backend"]=="promoted"
                and data["model"]==r["model"]
                and data["prompt_token_ids"]==r["tokens"]
                and data["decode_steps"]==r["decode"],
                f"model provenance invalid: {mode}")
        model[mode]=(data,logits)
    expected=model["ptx"][0]
    model_diff={}
    for mode in ("auto","cubin"):
        data,logits=model[mode]
        require(expected["generated_token_ids"]==data["generated_token_ids"]
                and expected["vocabulary"]==data["vocabulary"]
                and expected["tensor_count"]==data["tensor_count"],
                f"greedy token or model shape diverged: {mode}")
        delta=s55d.compare_binary_logits(model["ptx"][1],logits,
                   (r["decode"]+1,expected["vocabulary"]))
        model_diff[mode]=delta
        require(delta["mismatches"]==0,f"Qwen3 logits mismatch under {mode}")
    s55d.gpu_info(env,out/"gpu-after.log")
    report={
        "schema":"CANDLE-DKR-R125-P1-RESULT-V1","status":"PASS",
        "source_head":r["source_head"],"target_cc":61,
        "catalogue_sha256":r["catalogue_sha256"],
        "model_sha256":r["model_sha256"],
        "images":{"auto":"cubin","cubin":"cubin","ptx":"ptx"},
        "numeric":{k:{"cases":x["cases"],"elements":x["elements"],
            "mismatches":x["mismatches"]} for k,x in comparisons.items()},
        "model":{"tokens":expected["generated_token_ids"],
            "compared_logits":{k:x["elements"] for k,x in model_diff.items()},
            "mismatches":{k:x["mismatches"] for k,x in model_diff.items()},
            "timings":{k:v[0]["total_process_work_ms"] for k,v in model.items()}},
        "restrictions":[
            "Opt-in Cargo profile, not the universal cuda default",
            "The producer catalogue currently targets only SM61",
            "No silent fallback; auto permits PTX only on driver module or symbol load failure",
            "Does not qualify changes to ASD V3 or GPUs other than the sealed device target",
        ]
    }
    report_path=out/"result.json"
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("DKR_R125_P1_DEFAULT=PASS policy=auto selected=cubin")
    print("DKR_R125_P1_ROLLBACK=PASS policy=ptx selected=ptx")
    print("DKR_R125_P1_INVALID=PASS policy=invalid rejected=1")
    print("DKR_R125_P1_MODEL=PASS policies=3 logits_full=1")
    print("DKR_R125_P1_REPORT="+str(report_path))
    print("DKR_R125_P1_PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("build")
    for name in ("seal","cubin-dir","catalogue-dir","s55d-result","model","out"):
        b.add_argument("--"+name,type=Path,required=True)
    h=sub.add_parser("headless")
    h.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    (build if args.mode=="build" else headless)(args)

if __name__=="__main__":
    main()
