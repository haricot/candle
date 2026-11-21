#!/usr/bin/env python3
"""S5.5b: produce four typed host-launch descriptors outside Candle.

This is the frozen S31 producer implementation, NOT a runtime recipe:
exact symbol/parameter tables are recorded as data with S5.1 provenance.
The generic consumer must bound, verify and embed the data at build time.
Never change the sealed 25 native families or the existing CUBINs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

SCHEMA="CANDLE-DKR-R124-S55B-LAUNCH-V1"
BASE="d1929bea2a57beb82d466c9654c1d5f5d6ff616c"
DEPS={"fp8":("quantized",),"fp4":("quantized",),"bf16":("cast",),
      "routed":("mmvq_gguf","moe_simt_f16","moe_gguf")}
FP8={
    "simt_gemv":"fp8_sm61_w8a16_gemv",
    "simt_prefill":"fp8_sm61_w8a16_prefill4",
    "dp4a_gemv":"fp8_sm61_dp4a_gemv",
    "dp4a_prefill":"fp8_sm61_dp4a_prefill4",
    "quantize":"fp8_sm61_quant_x",
    "prefill_tokens":4,"rows_per_block":4,"threads":128,
    "k_alignment":128,"groups":[32,64,128],
}
FP4={
    "scalar":"nvfp4_arch_e6_w4a16_scalar",
    "cast":"nvfp4_arch_e6_f16_to_f32",
    "quantize":"quantize_q8_1",
    "reference":"nvfp4_mat_mul_q8_1",
    "tiled":"nvfp4_mat_mul_q8_1_tiled4x8",
    "mx_vec":[f"mul_mat_vec_mxfp4_q8_1_cuda{i}" for i in range(1,9)],
    "mx_mat":"mul_mat_mxfp4",
    "mx_moe":"indexed_moe_forward_mxfp4_q8_1",
    "auto_cc":61,"min_dp4a_cc":61,
    "auto_min_n":512,"auto_min_k":1024,"auto_min_batch":9,
    "scalar_threads":128,"cast_threads":256,"quant_threads":256,
    "warp":32,"matrix_warps":4,"tile_n":4,"tile_m":8,
}
BF16={"matmul":"cublas-sgemm-f32-round-bf16",
      "cast_to_f32":"cast_bf16_f32","cast_from_f32":"cast_f32_bf16"}
ROUTED={
    "max_batch":8,"quant_threads":256,"warp":32,"small_warps":4,
    "large_warps":2,"small_batch":4,"row_padding":512,
    "q8_block":32,"q8_bytes":36,"max_experts":1024,
    "simt_threads":128,"simt_tile_n":32,"simt_shared":2304,
    "gguf_tile_n":4,"gguf_shared_base":1024,"gguf_shared_limit":49152,
    "mmvq_family":"mmvq_gguf","simt_family":"moe_simt_f16",
    "gguf_family":"moe_gguf","simt_input":"f16","gguf_input":"f32",
    "gguf_abi":"RUNTIME_REPLACE_MOE_GGUF_ABI",
}
RECIPES={"fp8":FP8,"fp4":FP4,"bf16":BF16,"routed":ROUTED}

def require(ok,msg):
    if not ok:raise SystemExit("DKR_R124_S55B_PREPARE_FAIL: "+msg)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def source_symbols(p):
    text=p.read_text()
    return set(re.findall(r"\.entry\s+([^\s(]+)",text))

def self_test():
    assert FP8["groups"]==[32,64,128] and FP8["threads"]==128
    assert len(FP4["mx_vec"])==8 and FP4["matrix_warps"]==4
    assert ROUTED["simt_shared"]==2304 and ROUTED["max_batch"]==8
    assert BF16["cast_to_f32"]=="cast_bf16_f32"
    print("DKR_R124_S55B_PRODUCER_SELF_TEST=PASS")

def produce(args):
    sf=args.seal.resolve(strict=True)
    seal=json.loads(sf.read_text())
    require(seal["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1"
            and seal["source_baseline"]==BASE,"S5.1 seal invalid")
    root=Path(seal["catalogue"]).resolve(strict=True)
    inv=seal["catalogue_inventory"]
    require(len(inv["packages"])==25,"incomplete sealed catalogue")
    idx=root/"native-catalog.json"
    idx_sha=sha(idx) if idx.is_file() else None
    require(idx_sha==inv.get("index_sha256"),"sealed optional index changed")
    families={}
    for names in DEPS.values():
        for name in names:
            meta=root/name/"manifest.json"
            ptx=root/name/(name+".ptx")
            require(sha(meta)==inv["packages"][name]["manifest_sha256"]
                    and sha(ptx)==inv["packages"][name]["ptx_sha256"],
                    "sealed package changed: "+name)
            families[name]={
                "manifest_sha256":sha(meta),"ptx_sha256":sha(ptx)}
    entries={family:source_symbols(root/family/(family+".ptx")) for family in families}
    for name in ("simt_gemv","simt_prefill","dp4a_gemv","dp4a_prefill","quantize"):
        require(FP8[name] in entries["quantized"],"FP8 symbol missing "+FP8[name])
    for name in ("scalar","cast","quantize","reference","tiled","mx_mat","mx_moe"):
        require(FP4[name] in entries["quantized"],"FP4 symbol missing "+FP4[name])
    for sym in FP4["mx_vec"]:
        require(sym in entries["quantized"],"FP4 vector symbol missing "+sym)
    for name in ("cast_to_f32","cast_from_f32"):
        require(BF16[name] in entries["cast"],"BF16 cast symbol missing "+BF16[name])
    # MoE input dtype/ABI semantics are bound by the qualified native
    # package manifests. No arbitrary entry names or launch signatures.
    for family in ("mmvq_gguf","moe_simt_f16","moe_gguf"):
        require(bool(entries[family]),"routed kernel family has no entries: "+family)
    recipes={name:dict(recipe) for name,recipe in RECIPES.items()}
    # The ABI is a qualified contract constant but its string value is read
    # from the catalogue index if present; otherwise, use the published
    # contract given by S31 producer. See --gguf-abi.
    require(args.gguf_abi,"--gguf-abi is required (exact S31 native ABI)")
    recipes["routed"]["gguf_abi"]=args.gguf_abi
    out=args.out.resolve()
    require(not out.exists(),"output directory already exists")
    out.mkdir(parents=True)
    result={"schema":SCHEMA,"target_cc":61,"source_seal_sha256":sha(sf),
            "source_index_sha256":idx_sha,
            "source_index_mode":"present" if idx_sha else "legacy-absent",
            "family_provenance":families,
            "recipes":recipes}
    p=out/"manifest.json"
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("DKR_R124_S55B_PREPARE=PASS recipes=4 families=5")
    print("DKR_R124_S55B_MANIFEST="+str(p))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--seal",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--gguf-abi",default="") 
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        require(a.seal is not None and a.out is not None,
                "--seal and --out required")
        produce(a)

if __name__=="__main__":main()
