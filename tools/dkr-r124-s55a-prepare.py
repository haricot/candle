#!/usr/bin/env python3
"""S5.5a: materialize sealed MMQ launch geometry as producer-owned data.

The SM61-specific constants below are confined to the OFFLINE producer.
The generic runtime interpreter contains no Pascal-specific tiles/formulas.
This does not mutate the historical 25-family native catalog or CUBINs.
"""
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA="CANDLE-DKR-R124-S55A-MMQ-LAUNCH-V1"
BASE="d1929bea2a57beb82d466c9654c1d5f5d6ff616c"
CODECS=("q4_0","q4_1","q5_0","q5_1","q8_0",
        "q2_k","q3_k","q4_k","q5_k","q6_k")

def require(test, message):
    if not test:
        raise SystemExit("DKR_R124_S55A_FAIL: "+message)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def shared_bytes(codec,tile):
    # Frozen producer recipe from mmq_get_nbytes_shared, tile_y=64.
    y=64
    if codec in ("q4_0","q4_1"):
        qs,dm,sc=y*33,y*32//4+y//4,0
    elif codec in ("q5_0","q5_1","q8_0"):
        qs,dm,sc=y*65,y*64//8+y//4,0
    elif codec=="q2_k":
        qs,dm,sc=y*65,y*33,0
    elif codec=="q3_k":
        qs,dm,sc=y*65,y,y*32//8+y//8
    elif codec=="q4_k":
        qs,dm,sc=y*33,y*32//32,y*32//8+y//8
    elif codec in ("q5_k","q6_k"):
        qs,dm,sc=y*65,y*32//32+y//32,y*32//8+y//8
    else:
        raise ValueError("unsupported producer codec")
    return tile*4+(qs+dm+sc)*4+((tile*144+1023)//1024)*1024

def self_test():
    assert shared_bytes("q4_0",16)==13696
    assert shared_bytes("q4_k",40)==16064
    assert shared_bytes("q2_k",64)==34560
    assert len(CODECS)==10
    print("DKR_R124_S55A_PRODUCER_SELF_TEST=PASS")

def produce(args):
    sf=args.seal.resolve(strict=True)
    seal=json.loads(sf.read_text())
    require(seal["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1"
            and seal["source_baseline"]==BASE,"S5.1 seal not recognized")
    root=Path(seal["catalogue"]).resolve(strict=True)
    inventory=seal["catalogue_inventory"]
    require(len(inventory["packages"])==25,"S5.1 family inventory changed")
    index=root/"native-catalog.json"
    require(index.is_file(),"sealed catalogue index not found")
    index_sha=sha(index)
    require(inventory["index_sha256"]==index_sha,
            "native catalogue index differs from S5.1")
    rows={}
    for codec in CODECS:
        family="mmq_instance_"+codec
        pt=root/family/(family+".ptx")
        mf=root/family/"manifest.json"
        item=inventory["packages"][family]
        require(sha(pt)==item["ptx_sha256"]
                and sha(mf)==item["manifest_sha256"],
                "qualified family changed: "+family)
        meta=json.loads(mf.read_text())
        symbols=meta["mmq_entries"]
        expected={f"{t}:{checked}" for t in range(8,65,8) for checked in (0,1)}
        require(isinstance(symbols,dict) and set(symbols)==expected,
                "MMQ template inventory differs from qualified ABI: "+family)
        tiles=[{"width":t,"shared_bytes":shared_bytes(codec,t)}
               for t in range(8,65,8)]
        rows[codec]={"family":family,"ptx_sha256":item["ptx_sha256"],
                     "tiles":tiles}
    out=args.out.resolve()
    require(not out.exists(),"output directory already exists: "+str(out))
    out.mkdir(parents=True)
    result={"schema":SCHEMA,"target_cc":61,"source_index_sha256":index_sha,
            "source_seal_sha256":sha(sf),
            "row_tile":64,"quant_padding":512,"warp":32,
            "max_shared_bytes":49152,
            "quant_block":[128,1,1],"matrix_block":[32,8,1],
            "codecs":rows}
    path=out/"manifest.json"
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("DKR_R124_S55A_PREPARE=PASS codecs=10 tiles=80 native_packages_unchanged=25")
    print("DKR_R124_S55A_MANIFEST="+str(path))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--seal",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        require(args.seal is not None and args.out is not None,
                "--seal and --out are required")
        produce(args)

if __name__=="__main__":
    main()
