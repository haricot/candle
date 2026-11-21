#!/usr/bin/env python3
"""S5.4b: produce 25 exact-target embedded SM61 CUBIN images from sealed PTX.

Reuses pilot images when identical; compiles only the missing images with
CUDA 12.x ptxas. Never mutates S5.1 source packages or deletes CUDA caches.
All production code continues to be independent of the CUBIN producer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent.parent
SCHEMA="CANDLE-DKR-R124-S54B-CUBIN-V1"
MAX_BYTES=32*1024*1024

def require(ok,msg):
    if not ok: raise SystemExit("DKR_R124_S54B_PREPARE_FAIL: "+msg)

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

def seal_gate(path):
    seal=json.loads(path.resolve(strict=True).read_text())
    require(seal["schema"]=="CANDLE-DKR-R124-S51-SEAL-V1"
            and seal["source_baseline"]=="d1929bea2a57beb82d466c9654c1d5f5d6ff616c",
            "S5.1 seal invalid")
    families=seal["catalogue_inventory"]["packages"]
    require(len(families)==25,"S5.1 family count changed")
    root=Path(seal["catalogue"])
    for name,item in families.items():
        require(digest(root/name/"manifest.json")==item["manifest_sha256"]
                and digest(root/name/(name+".ptx"))==item["ptx_sha256"],
                "sealed source changed: "+name)
    require(digest(Path(seal["evidence"]["archive"]))==
            seal["evidence"]["archive_sha256"],"S4.4 archive changed")
    return seal

def toolchain(name):
    path=shutil.which(name) if "/" not in name else name
    require(path is not None and Path(path).is_file(),"CUDA 12.x ptxas not found; pass --ptxas")
    path=str(Path(path).resolve())
    p=subprocess.run([path,"--version"],text=True,capture_output=True)
    version=(p.stdout+p.stderr).strip()
    require(p.returncode==0 and re.search(r"release 12\.\d+",version) is not None,
            "SM61 requires CUDA 12.x ptxas, not CUDA 13: "+version)
    return path,version

def validate_cubin(path,expected_sha=None):
    require(path.is_file() and path.stat().st_size>=64 and path.stat().st_size<=MAX_BYTES,
            "invalid CUBIN size: "+str(path))
    with path.open("rb") as f:
        require(f.read(4)==b"\x7fELF","not ELF CUDA CUBIN: "+str(path))
    actual=digest(path)
    if expected_sha is not None:
        require(actual==expected_sha,"CUBIN digest modified: "+str(path))
    return actual,path.stat().st_size

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal",type=Path,required=True)
    p.add_argument("--pilot-receipt",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--ptxas",default="ptxas")
    args=p.parse_args()
    seal=seal_gate(args.seal)
    ptxas,version=toolchain(args.ptxas)
    pilot=json.loads(args.pilot_receipt.resolve(strict=True).read_text())
    require(pilot["schema"]=="CANDLE-DKR-R124-S54-PILOT-V1"
            and pilot["target_cc"]==61 and pilot["ptxas_version"]==version,
            "S5.4 pilot toolchain or target mismatch")
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    folder=out/"cubins"
    folder.mkdir(exist_ok=True)
    mf=out/"manifest.json"
    require(not mf.exists(),"manifest already finalized; no overwrite permitted")
    rows={}
    reuse=0
    compiled=0
    for name,item in sorted(seal["catalogue_inventory"]["packages"].items()):
        ptx=Path(seal["catalogue"])/name/(name+".ptx")
        dest=folder/(name+".cubin")
        note=folder/(name+".json")
        if dest.exists():
            require(note.is_file(),"existing CUBIN lacks receipt: "+name)
            row=json.loads(note.read_text())
            require(row["ptx_sha256"]==item["ptx_sha256"]
                    and row["target_cc"]==61
                    and row["ptxas_version"]==version,
                    "stale or incompatible CUBIN receipt: "+name)
            check,size=validate_cubin(dest,row["cubin_sha256"])
            require(size==row["cubin_bytes"],"CUBIN size drift: "+name)
            print("DKR_R124_S54B_REUSED="+name,flush=True)
            reuse+=1
        elif name in pilot["families"]:
            previous=pilot["families"][name]
            source=Path(previous["cubin"])
            require(previous["ptx_sha256"]==item["ptx_sha256"],
                    "pilot PTX provenance mismatch: "+name)
            check,size=validate_cubin(source,previous["cubin_sha256"])
            shutil.copyfile(source,dest)
            require(digest(dest)==check,"copied CUBIN changed: "+name)
            row={"target_cc":61,"ptx_sha256":item["ptx_sha256"],
                 "cubin_sha256":check,"cubin_bytes":size,
                 "ptxas_version":version,"origin":"S5.4 pilot"}
            note.write_text(json.dumps(row,sort_keys=True,indent=2)+"\n")
            print("DKR_R124_S54B_PILOT_REUSED="+name,flush=True)
            reuse+=1
        else:
            temporary=folder/(name+".cubin.partial")
            log=folder/(name+".ptxas.log")
            require(not temporary.exists(),"unfinished ptxas output: "+name)
            print("DKR_R124_S54B_COMPILING="+name,flush=True)
            with log.open("w") as f:
                result=subprocess.run([ptxas,"-arch=sm_61","-O3",str(ptx),
                                       "-o",str(temporary)],stdout=f,
                                      stderr=subprocess.STDOUT)
            if result.returncode!=0:
                if temporary.exists():temporary.unlink()
                require(False,"ptxas failed for "+name+"; see "+str(log))
            check,size=validate_cubin(temporary)
            temporary.replace(dest)
            row={"target_cc":61,"ptx_sha256":item["ptx_sha256"],
                 "cubin_sha256":check,"cubin_bytes":size,
                 "ptxas_version":version,"origin":"ptxas offline"}
            note.write_text(json.dumps(row,sort_keys=True,indent=2)+"\n")
            print("DKR_R124_S54B_BUILT="+name+" bytes="+str(size),flush=True)
            compiled+=1
        rows[name]=row
    require(len(rows)==25,"not all 25 CUBIN families produced")
    manifest={"schema":SCHEMA,"target_cc":61,"ptxas_version":version,
              "families":rows}
    temporary=out/"manifest.json.partial"
    temporary.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    temporary.replace(mf)
    print(f"DKR_R124_S54B_CUBIN_INVENTORY=PASS packages=25 reused={reuse} compiled={compiled}")
    print("DKR_R124_S54B_MANIFEST="+str(mf))
    print("DKR_R124_S54B_PREPARE=PASS")

if __name__=="__main__":
    main()
