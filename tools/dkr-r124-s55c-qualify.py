#!/usr/bin/env python3
"""R1.24 S5.5c: verify Cudaforge catalogue provenance, reuse S5.5b numerical gates.

No NVCC, ptxas, cargo clean or runtime filesystem recipe overrides.
The S5.1 PTX, S5.4b CUBIN and prior S5.5a/b reports stay immutable.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
helper_path=Path(__file__).with_name("dkr-r124-s55b-qualify.py")
module=importlib.util.spec_from_file_location("dkr_s55b_qualification",helper_path)
s55b=importlib.util.module_from_spec(module)
module.loader.exec_module(s55b)

def require(ok,msg):
    if not ok: raise SystemExit("DKR_R124_S55C_FAIL: "+msg)

def verify(catalogue_dir,seal_path,cubins_dir):
    folder=catalogue_dir.resolve(strict=True)
    seal_path=seal_path.resolve(strict=True)
    cubins_dir=cubins_dir.resolve(strict=True)
    seal=s55b.seal_gate(seal_path)
    images=s55b.images_gate(cubins_dir,seal)
    mmq=folder/"mmq"
    recipes=folder/"recipes"
    neutral=s55b.neutral_gate(mmq,seal,seal_path)
    typed=s55b.recipes_gate(recipes,seal,seal_path,mmq)
    file=folder/"catalogue.json"
    bundle=json.loads(file.read_text())
    require(bundle["schema"]=="CANDLE-DKR-R124-S55C-CUDAFORGE-CATALOG-V1"
            and bundle["producer"]=="cudaforge"
            and bundle["producer_command"]=="dkr-cudaforge-catalog"
            and bundle["target_cc"]==61
            and bundle["family_count"]==25
            and bundle["recipe_count"]==4
            and bundle["mmq_codec_count"]==10,
            "unknown Cudaforge catalogue producer/schema/target")
    require(bundle["source_seal_sha256"]==s55b.digest(seal_path)
            and bundle["source_seal_sha256"]==neutral["source_seal_sha256"]
            and bundle["source_seal_sha256"]==typed["source_seal_sha256"],
            "S5.1 sealed source changed")
    require(bundle["source_index_sha256"]==neutral["source_index_sha256"]
            and bundle["source_index_sha256"]==typed["source_index_sha256"]
            and bundle["source_index_mode"]==neutral["source_index_mode"]
            and bundle["source_index_mode"]==typed["source_index_mode"],
            "optional native source index mismatch")
    hashes={
        "recipe_spec_sha256":ROOT/"tools/dkr-s31-builder/recipes/sm61.json",
        "mmq_manifest_sha256":mmq/"manifest.json",
        "recipes_manifest_sha256":recipes/"manifest.json",
        "cubin_manifest_sha256":cubins_dir/"manifest.json"}
    for key,path in hashes.items():
        require(bundle[key]==s55b.digest(path),
                f"producer attestation does not bind {key}")
    require(set(images["families"])==s55b.FAMILIES
            and set(neutral["codecs"])=={
                c.replace("K","_k").lower() for c in s55b.CODECS}
            and set(typed["recipes"])=={"fp8","fp4","bf16","routed"},
            "producer inventory not complete")
    return {"catalogue_dir":str(folder),
            "catalogue_sha256":s55b.digest(file),
            "mmq_dir":mmq,"recipes_dir":recipes,
            "bundle":bundle,"seal":seal}

def build(args):
    head=s55b.source_head()
    inputs=verify(args.catalogue_dir,args.seal,args.cubin_dir)
    seal=inputs["seal"]
    s55b.historical_gate(args.s52_result,args.s53_result,seal)
    out=args.out.resolve()
    require(not out.exists(),"qualification directory already exists")
    out.mkdir(parents=True)
    env=s55b.clean_env()
    env["DKR_EMBED_NATIVE_DIR"]=seal["catalogue"]
    env["DKR_EMBED_CUBIN_DIR"]=str(args.cubin_dir.resolve(strict=True))
    env["DKR_EMBED_S55_MMQ_DIR"]=str(inputs["mmq_dir"])
    env["DKR_EMBED_S55B_RECIPES_DIR"]=str(inputs["recipes_dir"])
    env["DKR_EMBED_S55C_CUDAFORGE_DIR"]=inputs["catalogue_dir"]
    archive=out/"s55c-tests.tar.zst"
    command=[
        "cargo","nextest","archive","-p","candle-core","-p","candle-nn",
        "--no-default-features","--features","cuda-dkr-cudaforge-catalog",
        "--lib","--test","dkr_s52_numeric","--test","moe_simt_sm61",
        "--test","cuda_legacy_bf16_tests","--test","dkr_catalog_mode",
        "--test","dkr_s53_perf","--test","fp8_sm61_f8_tests",
        "--test","fp4_e7c_contract","--test","mxfp4_tests",
        "--archive-file",str(archive)]
    s55b.execute(command,env,out/"archive-build.log")
    require(archive.is_file(),"Nextest archive not produced")
    receipt={
        # S5.5b replayer deliberately owns the strict shared numerical gates.
        "schema":"CANDLE-DKR-R124-S55B-ARCHIVE-V1",
        "qualification_profile":"S55C_CUDAFORGE",
        "source_head":head,
        "s51_seal":str(args.seal.resolve(strict=True)),
        "s51_seal_sha256":s55b.digest(args.seal.resolve(strict=True)),
        "s52_result":str(args.s52_result.resolve(strict=True)),
        "s52_result_sha256":s55b.digest(args.s52_result.resolve(strict=True)),
        "s53_result":str(args.s53_result.resolve(strict=True)),
        "s53_result_sha256":s55b.digest(args.s53_result.resolve(strict=True)),
        "cubin_dir":str(args.cubin_dir.resolve(strict=True)),
        "cubin_manifest_sha256":s55b.digest(args.cubin_dir.resolve(strict=True)/"manifest.json"),
        "mmq_dir":str(inputs["mmq_dir"]),
        "mmq_manifest_sha256":s55b.digest(inputs["mmq_dir"]/"manifest.json"),
        "recipes_dir":str(inputs["recipes_dir"]),
        "recipes_manifest_sha256":s55b.digest(inputs["recipes_dir"]/"manifest.json"),
        "cudaforge_catalogue_dir":inputs["catalogue_dir"],
        "cudaforge_catalogue_sha256":inputs["catalogue_sha256"],
        "archive":str(archive),
        "archive_sha256":s55b.digest(archive),"families":25,
    }
    (out/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S55C_BUILD=PASS backend=cudaforge features=neutral,cubin")
    print("DKR_R124_S55C_RECEIPT="+str(out/"receipt.json"))

def headless(args):
    rp=args.receipt.resolve(strict=True)
    receipt=json.loads(rp.read_text())
    require(receipt["schema"]=="CANDLE-DKR-R124-S55B-ARCHIVE-V1"
            and receipt.get("qualification_profile")=="S55C_CUDAFORGE",
            "S5.5c receipt/profile mismatch")
    inputs=verify(Path(receipt["cudaforge_catalogue_dir"]),
                  Path(receipt["s51_seal"]),Path(receipt["cubin_dir"]))
    require(inputs["catalogue_sha256"]==receipt["cudaforge_catalogue_sha256"],
            "producer catalogue was modified after Rust archive was compiled")
    require(str(inputs["mmq_dir"])==receipt["mmq_dir"]
            and str(inputs["recipes_dir"])==receipt["recipes_dir"],
            "replay receipts do not use the producer's derived recipes")
    s55b.check_archived_source(receipt["source_head"])
    archive=Path(receipt["archive"])
    require(s55b.digest(archive)==receipt["archive_sha256"],
            "qualified Cudaforge build archive changed")
    work=rp.parent
    env=s55b.clean_env()
    cmd=["cargo","nextest","run","--archive-file",str(archive),
         "--workspace-remap",str(ROOT),"--run-ignored","all","--no-capture","-E"]
    att=s55b.execute(
        cmd+["test(dkr_s55c_cudaforge_catalogue_build_attestation)"],
        env,work/"s55c-attestation-preflight.log")
    require("DKR_R124_S55C_ATTESTATION_TEST=PASS families=25 recipes=4 codecs=10" in att,
            "build-time Cudaforge attestation unit test missing")
    before={p for p in work.glob("headless-*/result.json")}
    # Reuse the entire previously qualified S5.5b GPU matrix and performance
    # runner; feature-specific Cudaforge provenance checked both before/after.
    s55b.headless(argparse.Namespace(receipt=rp))
    reports={p for p in work.glob("headless-*/result.json")}-before
    require(len(reports)==1,"S5.5b GPU result for this headless run not unique")
    report_path=reports.pop()
    result=json.loads(report_path.read_text())
    require(result["status"]=="PASS"
            and result["schema"]=="CANDLE-DKR-R124-S55B-RESULT-V1"
            and len(result["cubin_modules_loaded"])==25
            and len(result["producer_recipes"])==4
            and len(result["fp8_fp4_extra_gpu_tests"])==6
            and result["numeric"]["quant_cases"]==120
            and result["numeric"]["gguf_moe"]==6,
            "S5.5b numerical CUBIN gates incomplete")
    log=(report_path.parent/"s52-cubin-numeric.log").read_text(errors="replace")
    marker="DKR_R124_S55C_CATALOGUE_ADMITTED producer=cudaforge target=sm61 families=25 recipes=4 sha256="
    require(marker in log,"Candle runtime did not admit the new Cudaforge catalogue")
    sha=inputs["catalogue_sha256"]
    record={
        "schema":"CANDLE-DKR-R124-S55C-RESULT-V1",
        "status":"PASS","backend":"cudaforge","target_cc":61,
        "families":25,"recipes":4,"mmq_codecs":10,
        "qualified_archive_sha256":receipt["archive_sha256"],
        "cudaforge_catalogue_sha256":sha,
        "s55b_numeric_and_perf_report":str(report_path),
        "s55b_report_sha256":s55b.digest(report_path),
        "attestation_log":str(work/"s55c-attestation-preflight.log"),
        "note":"S5.5b full numerical and 3-round CUBIN benchmarks reused on the S5.5c-feature binary; provenance separately verified"
    }
    result_file=report_path.parent/"s55c-result.json"
    result_file.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("DKR_R124_S55C_CATALOGUE=PASS producer=cudaforge families=25")
    print("DKR_R124_S55C_NUMERIC=PASS cases=120 gguf_moe=6 fp8_fp4_tests=6")
    print("DKR_R124_S55C_CUBIN=PASS families=25")
    print("DKR_R124_S55C_PERF=PASS rounds=3 cases=7")
    print("DKR_R124_S55C_REPORT="+str(result_file))
    print("DKR_R124_S55C_PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("build")
    b.add_argument("--seal",type=Path,required=True)
    b.add_argument("--catalogue-dir",type=Path,required=True)
    b.add_argument("--cubin-dir",type=Path,required=True)
    b.add_argument("--s52-result",type=Path,required=True)
    b.add_argument("--s53-result",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True)
    h=sub.add_parser("headless")
    h.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    if args.mode=="build":build(args)
    else:headless(args)

if __name__=="__main__":
    main()
