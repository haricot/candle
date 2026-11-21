#!/usr/bin/env python3
"""R1.24 S5.4 isolated CUDA Driver API PTX/CUBIN load and fill parity probe.

Python ctypes + system libcuda.so.1; no CUDA toolkit or PyCUDA dependency.
Use via dkr-r124-s54-qualify.py. Never modifies the sealed PTX catalogue.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import time

def demand(ok, message):
    if not ok:
        raise RuntimeError("DKR_R124_S54_PROBE_FAIL: " + message)

def api(lib, name, args):
    fn=getattr(lib,name)
    fn.argtypes=args
    fn.restype=C.c_int
    return fn

def perform(path, symbol, check_fill):
    lib=C.CDLL("libcuda.so.1")
    ptr=C.c_void_p
    cint=C.c_int
    u32=C.c_uint
    u64=C.c_uint64
    size=C.c_size_t
    check_name=api(lib,"cuGetErrorName",[cint,C.POINTER(C.c_char_p)])
    def ck(status, step):
        if status:
            name=C.c_char_p()
            try: check_name(status,C.byref(name))
            except Exception: pass
            label=name.value.decode(errors="replace") if name.value else str(status)
            raise RuntimeError(f"CUDA {step}: {label} ({status})")
    init=api(lib,"cuInit",[u32])
    dev_get=api(lib,"cuDeviceGet",[C.POINTER(cint),cint])
    capability=api(lib,"cuDeviceComputeCapability",[C.POINTER(cint),C.POINTER(cint),cint])
    make_ctx=api(lib,"cuCtxCreate_v2",[C.POINTER(ptr),u32,cint])
    destroy_ctx=api(lib,"cuCtxDestroy_v2",[ptr])
    load=api(lib,"cuModuleLoadData",[C.POINTER(ptr),ptr])
    unload=api(lib,"cuModuleUnload",[ptr])
    get_func=api(lib,"cuModuleGetFunction",[C.POINTER(ptr),ptr,C.c_char_p])
    mem_info=api(lib,"cuMemGetInfo_v2",[C.POINTER(size),C.POINTER(size)])
    alloc=api(lib,"cuMemAlloc_v2",[C.POINTER(u64),size])
    free=api(lib,"cuMemFree_v2",[u64])
    copy=api(lib,"cuMemcpyDtoH_v2",[ptr,u64,size])
    launch=api(lib,"cuLaunchKernel",[
        ptr,u32,u32,u32,u32,u32,u32,u32,ptr,
        C.POINTER(ptr),C.POINTER(ptr)])
    sync=api(lib,"cuCtxSynchronize",[])
    ck(init(0),"init")
    dev=cint()
    ck(dev_get(C.byref(dev),0),"device")
    major,minor=cint(),cint()
    ck(capability(C.byref(major),C.byref(minor),dev),"compute capability")
    demand((major.value,minor.value)==(6,1),"requires exactly SM61")
    ctx=ptr()
    start=time.perf_counter_ns()
    ck(make_ctx(C.byref(ctx),0,dev),"context")
    context_us=(time.perf_counter_ns()-start)//1000
    def mem():
        f,t=size(),size()
        ck(mem_info(C.byref(f),C.byref(t)),"memory info")
        return {"free":f.value,"total":t.value,"used":t.value-f.value}
    module=ptr()
    device_mem=u64()
    try:
        before=mem()
        blob=Path(path).read_bytes()
        demand(len(blob)>0,"module is empty")
        raw=C.create_string_buffer(blob,len(blob)+1)
        start=time.perf_counter_ns()
        ck(load(C.byref(module),C.cast(raw,ptr)),"load module")
        load_us=(time.perf_counter_ns()-start)//1000
        after_load=mem()
        func=ptr()
        start=time.perf_counter_ns()
        ck(get_func(C.byref(func),module,symbol.encode("utf-8")),"get function "+symbol)
        lookup_us=(time.perf_counter_ns()-start)//1000
        after_lookup=mem()
        fill=None
        if check_fill:
            demand(symbol=="fill_f32","fill parity expects fill_f32")
            n=4096
            ck(alloc(C.byref(device_mem),4*n),"allocate output")
            after_alloc=mem()
            device_arg=u64(device_mem.value)
            value_arg=C.c_float(1.25)
            count_arg=size(n)
            argv=(ptr*3)(
                C.cast(C.byref(device_arg),ptr),
                C.cast(C.byref(value_arg),ptr),
                C.cast(C.byref(count_arg),ptr))
            start=time.perf_counter_ns()
            ck(launch(func,(n+255)//256,1,1,256,1,1,0,None,argv,None),"launch fill_f32")
            ck(sync(),"synchronize fill_f32")
            first_us=(time.perf_counter_ns()-start)//1000
            data=(C.c_float*n)()
            ck(copy(C.cast(data,ptr),device_mem.value,4*n),"copy fill result")
            demand(all(v==1.25 for v in data),"fill_f32 output does not equal 1.25")
            ck(free(device_mem),"release device output")
            device_mem=u64()
            fill={"first_us":first_us,"elements":n,"value":1.25,
                  "passed":True,"vram_after_alloc_bytes":after_alloc["used"]}
        after=mem()
        return {
            "format":"cubin" if path.suffix==".cubin" else "ptx",
            "family":path.stem,
            "symbol":symbol,"sm":f"{major.value}.{minor.value}",
            "image_bytes":len(blob),"context_init_us":context_us,
            "module_load_us":load_us,"function_lookup_us":lookup_us,
            "vram_context_used_bytes":before["used"],
            "vram_after_module_bytes":after_load["used"],
            "vram_after_lookup_bytes":after_lookup["used"],
            "vram_module_delta_bytes":after_lookup["used"]-before["used"],
            "vram_after_probe_bytes":after["used"],
            "fill_parity":fill
        }
    finally:
        if device_mem.value: ck(free(device_mem),"cleanup memory")
        if module.value: ck(unload(module),"unload module")
        ck(destroy_ctx(ctx),"destroy CUDA context")

def self_test():
    assert Path("fill.ptx").suffix==".ptx"
    assert Path("fill.cubin").suffix==".cubin"
    print("DKR_R124_S54_DRIVER_SELF_TEST=PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file",type=Path)
    p.add_argument("--symbol")
    p.add_argument("--fill-parity",action="store_true")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    demand(args.file is not None and args.symbol is not None,
           "--file and --symbol are required")
    result=perform(args.file.resolve(strict=True),args.symbol,args.fill_parity)
    print("DKR_R124_S54_PROBE="+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
