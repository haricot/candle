#!/usr/bin/env python3
"""R1.25/P2: SM61 app rollout canary + isolated real CUDA driver admission fault.

Two separate Rust release binaries are built:
 * sm61-app — shipping profile, default CUBIN, no fault-injection feature
 * p2-fault-canary — qualification-only, driver-invalid-image hook enabled
Both use the same sealed Cudaforge catalogue and model. No CUDA builds.
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
mod=importlib.util.spec_from_file_location("dkr_r125_p1",Path(__file__).with_name("dkr-r125-p1-qualify.py"))
p1=importlib.util.module_from_spec(mod)
mod.loader.exec_module(p1)
s55d=p1.s55d
s55b=p1.s55b
SCHEMA="CANDLE-DKR-R125-P2-RECEIPT-V1"
RESULT_SCHEMA="CANDLE-DKR-R125-P2-RESULT-V1"
# Explicit prompt IDs are all below the Qwen3-0.6B vocabulary size.
SCENARIOS=(
    ("short",tuple(range(1,9)),8),
    ("medium",tuple(range(1,25)),16),
    ("long",tuple(range(1,49)),24),
)
POLICIES=("auto","ptx","cubin")
FAULT_ENV="CANDLE_DKR_P2_TEST_FAULT_CAST"
FAULT_REJECT="DKR_R125_P2_DRIVER_REJECTED family=cast"
FAULT_INVOKE="DKR_R125_P2_DRIVER_FAULT family=cast"

def require(value,message):
    if not value: raise SystemExit("DKR_R125_P2_FAIL: "+message)

def baseline(p, source,model):
    report_path=p.resolve(strict=True)
    report=json.loads(report_path.read_text())
    require(report.get("schema")=="CANDLE-DKR-R125-P1-RESULT-V1"
            and report.get("status")=="PASS"
            and report.get("source_head")=="8c7e6610599f12caaa810e580258bf7e18b863e0"
            and report.get("target_cc")==61
            and report.get("catalogue_sha256")==source["catalogue_sha256"]
            and report.get("model_sha256")==s55b.digest(model)
            and report["numeric"]["ptx_vs_auto"]["mismatches"]==0
            and report["numeric"]["ptx_vs_cubin"]["mismatches"]==0
            and report["numeric"]["ptx_vs_auto"]["cases"]==157
            and report["model"]["mismatches"]["auto"]==0
            and report["model"]["mismatches"]["cubin"]==0,
            "R1.25/P1 PASS reference changed or not applicable")
    return report_path

def environment(source,cubins):
    e=p1.source_environment(source,cubins)
    e.pop(FAULT_ENV,None)
    e.pop("CANDLE_DKR_IMAGE_MODE",None)
    return e

def build(args):
    model=s55d.ensure_model(args.model)
    seal=args.seal.resolve(strict=True)
    cubins=args.cubin_dir.resolve(strict=True)
    source=s55d.s55c.verify(args.catalogue_dir,seal,cubins)
    p1_report=baseline(args.p1_result,source,model)
    head=s55b.source_head()
    out=args.out.resolve()
    require(not out.exists(),"qualification output must be a fresh directory")
    out.mkdir(parents=True)
    (out/"bin").mkdir()
    env=environment(source,cubins)
    binary_hashes={}
    for feature,name in [("sm61-app","sm61-app-qwen3"),("p2-fault-canary","p2-fault-canary-qwen3")]:
        p1.s55d.execute([
            "cargo","build","--release","-p","dkr-s55d-model-probe",
            "--no-default-features","--features",feature],
            env,out/f"build-{feature}.log")
        executable=ROOT/"target/release/dkr-s55d-model-probe"
        require(executable.is_file(),"model executable missing: "+feature)
        destination=out/"bin"/name
        shutil.copy2(executable,destination)
        destination.chmod(destination.stat().st_mode | 0o111)
        result=s55d.execute([str(destination),"--self-test"],env,
                             out/f"self-test-{feature}.log")
        require("DKR_R124_S55D_MODEL_PROBE=PASS mode=promoted no_cuda_build=1" in result,
                "model executable self-test failed: "+feature)
        binary_hashes[feature]={"path":str(destination),"sha256":s55b.digest(destination)}
    # The test archive contains the actual DKR image policy and complete
    # 157-case fixed-input numeric suite, linked with test-only fault feature.
    archive=out/"p2-fault-tests.tar.zst"
    s55d.execute([
        "cargo","nextest","archive","-p","candle-core","-p","candle-nn",
        "--no-default-features","--features","cuda-dkr-p2-test-fault",
        "--lib","--test","dkr_s55d_parity","--archive-file",str(archive)],
        env,out/"archive-build.log")
    require(archive.is_file(),"P2 Nextest archive missing")
    receipt={
        "schema":SCHEMA,"source_head":head,"target_cc":61,
        "seal":str(seal),"seal_sha256":s55b.digest(seal),
        "catalogue_dir":source["catalogue_dir"],
        "catalogue_sha256":source["catalogue_sha256"],
        "cubin_dir":str(cubins),
        "cubin_manifest_sha256":s55b.digest(cubins/"manifest.json"),
        "p1_result":str(p1_report),"p1_result_sha256":s55b.digest(p1_report),
        "model":str(model),"model_sha256":s55b.digest(model),
        "binary_profiles":binary_hashes,
        "archive":str(archive),"archive_sha256":s55b.digest(archive),
        "scenarios":[{"id":name,"tokens":list(tokens),"decode":decode}
                     for name,tokens,decode in SCENARIOS],
    }
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("DKR_R125_P2_BUILD=PASS shipping_profile=sm61-app fault_profile=isolated scenarios=3")
    print("DKR_R125_P2_RECEIPT="+str(out/"receipt.json"))

def capture_failure(argv,env,log):
    proc=subprocess.run(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,errors="replace")
    log.write_text(proc.stdout)
    print(f"EXPECTED_FAILURE exit={proc.returncode} logfile={log}",flush=True)
    return proc.returncode,proc.stdout

def model_once(binary,model,scenario,policy,directory,env,fault=False):
    name=scenario["id"]
    tag=f"{name}-{policy}"+("-fault" if fault else "")
    cache=directory/f"cuda-cache-{tag}"
    cache.mkdir()
    require(not list(cache.iterdir()),"expected fresh GPU cache for "+tag)
    e=p1.env_for(env,policy,cache)
    if fault: e[FAULT_ENV]="1"
    else: e.pop(FAULT_ENV,None)
    result=directory/f"{tag}.json"
    logits=directory/f"{tag}.f32le"
    log=directory/f"{tag}.log"
    cmd=[str(binary),"--model",str(model),
         "--tokens",",".join(map(str,scenario["tokens"])),
         "--decode",str(scenario["decode"]),
         "--result",str(result),"--logits",str(logits)]
    output=s55d.execute(cmd,e,log)
    require(result.is_file() and logits.is_file(),
            "model results missing for "+tag)
    obj=json.loads(result.read_text())
    require(obj["schema"]=="CANDLE-DKR-R124-S55D-MODEL-V1"
            and obj["backend"]=="promoted"
            and obj["model"]==str(model)
            and obj["prompt_token_ids"]==scenario["tokens"]
            and obj["decode_steps"]==scenario["decode"]
            and obj["logits_bytes"]==logits.stat().st_size,
            "model metadata invalid for "+tag)
    return obj,logits,output

def verify_loaded(log,mode,allow_fallback=False):
    require("DKR_R125_P1_POLICY default=cubin requested="+mode in log
            and "DKR_R124_S55C_CATALOGUE_ADMITTED producer=cudaforge" in log,
            "catalogue/policy trace missing for "+mode)
    loaded=[line for line in log.splitlines() if "CANDLE_DKR_IMAGE_LOADED " in line]
    require(loaded,"no native CUDA images loaded for "+mode)
    if allow_fallback:
        require(FAULT_INVOKE in log and FAULT_REJECT in log
                and log.count("DKR_R125_P1_FALLBACK family=cast ")==1,
                "a genuine CUDA module admission fault did not trigger exactly one fallback")
        require(any("family=cast " in line and "format=ptx " in line for line in loaded)
                and any("format=cubin " in line for line in loaded)
                and all("format=cubin " in line or
                        ("family=cast " in line and "format=ptx " in line) for line in loaded),
                "unexpected mixed PTX/CUBIN execution after fallback")
    else:
        require(FAULT_INVOKE not in log and "DKR_R125_P1_FALLBACK" not in log,
                "unexpected fallback in normal app run")
        expected="format=ptx " if mode=="ptx" else "format=cubin "
        require(all(expected in line for line in loaded),
                "wrong image format under "+mode)
    return [line.split("family=",1)[1].split()[0] for line in loaded]

def headless(args):
    require(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
            "P2 headless must run without graphical session")
    receipt_path=args.receipt.resolve(strict=True)
    receipt=json.loads(receipt_path.read_text())
    require(receipt["schema"]==SCHEMA and receipt["target_cc"]==61,
            "P2 receipt schema or target invalid")
    s55b.check_archived_source(receipt["source_head"])
    # Only documentation commits permitted since the tested source HEAD.
    names=["candle-core/src/cuda_backend/device.rs",
           "candle-core/src/cuda_backend/dkr_image_policy.rs",
           "candle-core/Cargo.toml","candle-nn/Cargo.toml",
           "tools/dkr-s55d-model-probe","tools/dkr-r125-p2-qualify.py"]
    for path in names:
        result=subprocess.run(["git","diff","--quiet",receipt["source_head"],
                "HEAD","--",path],cwd=ROOT)
        require(result.returncode==0,"compiled/test source changed since build: "+path)
    source=s55d.s55c.verify(
        Path(receipt["catalogue_dir"]),Path(receipt["seal"]),
        Path(receipt["cubin_dir"]))
    require(source["catalogue_sha256"]==receipt["catalogue_sha256"]
            and s55b.digest(Path(receipt["seal"]))==receipt["seal_sha256"]
            and s55b.digest(Path(receipt["cubin_dir"])/"manifest.json")==receipt["cubin_manifest_sha256"],
            "catalogue/seal/CUBIN inputs changed")
    baseline(Path(receipt["p1_result"]),source,Path(receipt["model"]))
    for path,key in [(receipt["p1_result"],"p1_result_sha256"),
                     (receipt["model"],"model_sha256"),
                     (receipt["archive"],"archive_sha256")]:
        require(s55b.digest(Path(path))==receipt[key],"immutable input changed: "+key)
    for data in receipt["binary_profiles"].values():
        require(s55b.digest(Path(data["path"]))==data["sha256"],
                "model binary changed after feature audit/build")
    require(receipt["scenarios"]==[
        {"id":name,"tokens":list(tokens),"decode":count}
        for name,tokens,count in SCENARIOS],
        "P2 scenario fixture has changed")

    out=receipt_path.parent/("headless-"+datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True)
    env=environment(source,Path(receipt["cubin_dir"]))
    s55d.gpu_info(env,out/"gpu-before.log")
    ship=Path(receipt["binary_profiles"]["sm61-app"]["path"])
    test=Path(receipt["binary_profiles"]["p2-fault-canary"]["path"])

    # CPU-only policy/attestation tests on a fault-capable archive.
    unit=s55d.execute([
        "cargo","nextest","run","--archive-file",receipt["archive"],
        "--workspace-remap",str(ROOT),"--run-ignored","all","--no-capture",
        "-E","test(qualified_default_policy_has_explicit_rollback_only) | test(dkr_s55c_cudaforge_catalogue_build_attestation)"],
        env,out/"unit-preflight.log")
    require("DKR_R125_P1_POLICY=PASS" in unit
            and "DKR_R124_S55C_ATTESTATION_TEST=PASS" in unit,
            "policy and sealed producer gates not passed")

    # Full 157-case PTX reference under the exact same fault-capable test
    # binary: when forced PTX, the corrupted-image path must not be reached.
    numeric={}
    for policy in ("ptx","auto"):
        path=out/f"numeric-{policy}.json"
        cache=out/f"numeric-cache-{policy}"
        cache.mkdir()
        e=p1.env_for(env,policy,cache)
        e[FAULT_ENV]="1"
        e["DKR_S55D_MODE"]= "ptx" if policy=="ptx" else "cubin"
        e["DKR_S55D_OUTPUT"]=str(path)
        log=s55d.execute([
            "cargo","nextest","run","--archive-file",receipt["archive"],
            "--workspace-remap",str(ROOT),"--run-ignored","all","--no-capture",
            "-E","test(dkr_s55d_capture_outputs)"
        ],e,out/f"numeric-{policy}.log")
        verify_loaded(log,policy,allow_fallback=policy=="auto")
        numeric[policy]=path
    delta=s55d.numeric_compare(numeric["ptx"],numeric["auto"])
    (out/"fault-numeric-comparison.json").write_text(
        json.dumps(delta,sort_keys=True,indent=2)+"\n")
    require(delta["status"]=="PASS" and delta["cases"]==157
            and delta["mismatches"]==0,
            "fault-fallback numerical parity failed (see comparison)")
    print(f"DKR_R125_P2_FAULT_NUMERIC=PASS cases={delta['cases']} elements={delta['elements']}")

    qualified_families=set(source["seal"]["catalogue_inventory"]["packages"])
    require(len(qualified_families)==25,"qualified native inventory incomplete")
    results=[]
    for scenario in receipt["scenarios"]:
        mode_results={}
        mode_families={}
        for mode in POLICIES:
            # The shipping binary does not compile in the test fault hook.
            # Intentionally set the test variable for one trial to prove it
            # cannot alter the shipping app's default.
            e=dict(env)
            if scenario["id"]=="short" and mode=="auto":
                e[FAULT_ENV]="1"
            obj,logits,log=model_once(ship,receipt["model"],scenario,mode,out,e)
            families=set(verify_loaded(log,mode))
            require(families <= qualified_families,
                    f"unqualified native family: {scenario['id']}/{mode}: {families-qualified_families}")
            mode_families[mode]=families
            mode_results[mode]=(obj,logits)
        require(mode_families["ptx"]==mode_families["auto"]==mode_families["cubin"],
                f"PTX/CUBIN dispatch chose different native family inventory for {scenario['id']}")
        ref=mode_results["ptx"][0]
        differences={}
        for mode in ("auto","cubin"):
            obj,logits=mode_results[mode]
            require(obj["generated_token_ids"]==ref["generated_token_ids"]
                    and obj["vocabulary"]==ref["vocabulary"]
                    and obj["tensor_count"]==ref["tensor_count"],
                    f"model token/shape mismatch: {scenario['id']}/{mode}")
            diff=s55d.compare_binary_logits(mode_results["ptx"][1],logits,
                    (scenario["decode"]+1,ref["vocabulary"]))
            differences[mode]={"elements":diff["elements"],
                               "mismatches":diff["mismatches"],
                               "max_abs":diff["max_abs"]}
            require(diff["mismatches"]==0,
                    f"model logits mismatch: {scenario['id']}/{mode}")
        timings={mode:mode_results[mode][0]["total_process_work_ms"] for mode in POLICIES}
        results.append({"scenario":scenario,"timings_ms":timings,
                        "loaded_families":sorted(mode_families["ptx"]),
                        "logits_comparison":differences,
                        "generated_token_ids":ref["generated_token_ids"]})
        print(f"DKR_R125_P2_MODEL=PASS scenario={scenario['id']} modes=3 logits={differences['auto']['elements']}",flush=True)

    # Separate fault-capable release executable: on 'auto', actual CUDA
    # driver rejection of the deliberately malformed cast image must
    # cause *one* verified PTX fallback, while unrelated modules load CUBIN.
    short=receipt["scenarios"][0]
    fault_obj,fault_logits,fault_log=model_once(
        test,receipt["model"],short,"auto",out,env,fault=True)
    loaded=verify_loaded(fault_log,"auto",allow_fallback=True)
    require(set(loaded)==set(results[0]["loaded_families"]),
            "fault-mode Qwen3 loaded different native families than shipping short case")
    reference=out/"short-auto.f32le"
    diff=s55d.compare_binary_logits(reference,fault_logits,
            (short["decode"]+1,fault_obj["vocabulary"]))
    require(diff["mismatches"]==0 and
            fault_obj["generated_token_ids"]==results[0]["generated_token_ids"],
            "real CUDA driver-fault rollback changed model outputs")

    # A strict-CUBIN process must reject the same driver error, not
    # silently retry PTX or write a completed inference report.
    cache=out/"cuda-cache-short-cubin-fault"
    cache.mkdir()
    e=p1.env_for(env,"cubin",cache)
    e[FAULT_ENV]="1"
    failed_report=out/"strict-cubin-fault.json"
    failed_logits=out/"strict-cubin-fault.f32le"
    code,log=capture_failure([
        str(test),"--model",receipt["model"],"--tokens",
        ",".join(map(str,short["tokens"])),"--decode",str(short["decode"]),
        "--result",str(failed_report),"--logits",str(failed_logits)
    ],e,out/"strict-cubin-fault.log")
    require(code!=0 and FAULT_INVOKE in log and FAULT_REJECT in log
            and "DKR_R125_P1_FALLBACK" not in log
            and not failed_report.exists(),
            "strict-CUBIN fault did not fail closed")
    print("DKR_R125_P2_FAULT=PASS actual_driver_rejection=1 auto_fallback=ptx strict_cubin=rejected")

    s55d.gpu_info(env,out/"gpu-after.log")
    data={
        "schema":RESULT_SCHEMA,"status":"PASS","source_head":receipt["source_head"],
        "target_cc":61,"catalogue_sha256":receipt["catalogue_sha256"],
        "model_sha256":receipt["model_sha256"],
        "shipping_app_profile":"sm61-app","test_fault_profile":"p2-fault-canary",
        "test_fault_compiled_into_shipping_app":False,
        "fault_family":"cast","fault_driver_rejected":True,
        "auto_fallback_family":"cast","strict_cubin_rejected":True,
        "fault_numeric":{"cases":delta["cases"],"elements":delta["elements"],
                         "mismatches":delta["mismatches"]},
        "fault_model_logits":{"elements":diff["elements"],
                              "mismatches":diff["mismatches"]},
        "scenarios":results,
        "restrictions":[
            "Candle generic cuda unchanged; only explicitly built SM61 app opts in",
            "No assumption of matching uncontrolled GPU clocks or cold OS page cache",
            "Injects an invalid in-memory driver image only in qualification-only binary",
            "Model matrix: one Qwen3-0.6B GGUF, 3 prompt lengths, no ASD V3",
        ],
    }
    dest=out/"result.json"
    dest.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    print("DKR_R125_P2_ROLLOUT=PASS sm61_app_cubin_default=1")
    print("DKR_R125_P2_REPORT="+str(dest))
    print("DKR_R125_P2_PASS")

def self_test():
    require(len(SCENARIOS)==3 and len({x[0] for x in SCENARIOS})==3,
            "scenario ids not unique")
    require(all(1<=len(tokens)<=64 and 2<=decode<=32 for _,tokens,decode in SCENARIOS),
            "P2 scenario bounds changed")
    require({len(x[1]) for x in SCENARIOS}=={8,24,48},
            "P2 input variation changed")
    print("DKR_R125_P2_SELF_TEST=PASS scenarios=3 prompt_lengths=8,24,48")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    builder=sub.add_parser("build")
    for name in ("seal","cubin-dir","catalogue-dir","p1-result","model","out"):
        builder.add_argument("--"+name,type=Path,required=True)
    replay=sub.add_parser("headless")
    replay.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="self-test":self_test()
    elif args.command=="build":build(args)
    else:headless(args)

if __name__=="__main__":main()
