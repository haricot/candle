//! Native external SM61 MoE dispatch. Existing routing contract requires valid
//! sorted expert IDs and a permutation of token-pair IDs. Values stay on GPU.
use candle::{CudaDevice, CudaStorage, DType, Result, Tensor};
use candle::cuda_backend::{cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg}, WrapErr};
use candle::quantized::{GgmlDType,QTensor};

fn layout(t: &Tensor, dev: &CudaDevice, dtype: DType, len: usize) -> Result<()> {
    if t.dtype()!=dtype || t.elem_count()!=len || t.device().as_cuda_device()?.id()!=dev.id()
        || !t.is_contiguous() {
        candle::bail!("DKR MoE dtype, device, shape or layout mismatch");
    }
    let (_, l)=t.storage_and_layout();
    if l.start_offset()!=0 { candle::bail!("DKR MoE requires zero storage offset"); }
    Ok(())
}

fn dimensions(input: &Tensor, route: &Option<Tensor>, topk: usize, e: usize,n: usize,k: usize) -> Result<(usize,usize)> {
    let (r,ki)=input.dims2()?;
    let m=if route.is_some() { Some(r) } else { r.checked_mul(topk) }
        .ok_or_else(|| candle::Error::Msg("MoE route count overflow".into()))?;
    if ki!=k || [r,m,e,n,k,topk].iter().any(|&v| v==0 || v>i32::MAX as usize)
        || topk>e || (route.is_some() && m%topk!=0)
        || m.checked_mul(n).is_none_or(|v|v>i32::MAX as usize)
        || n.checked_mul(k).is_none_or(|v|v>i32::MAX as usize)
        || r.checked_mul(k).is_none_or(|v|v>i32::MAX as usize) {
        candle::bail!("DKR MoE geometry exceeds native ABI");
    }
    Ok((r,m))
}

fn result<T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr>(out: candle::cuda_backend::cudarc::driver::CudaSlice<T>,dev:&CudaDevice,m:usize,n:usize)->Result<Tensor>{
    Ok(Tensor::from_storage(candle::Storage::Cuda(CudaStorage::wrap_cuda_slice(out,dev.clone())),
        (m,n),candle::op::BackpropOp::none(),false))
}

// Keep each storage lock and device-pointer guard alive through the submissions.
// Pointer values are passed as 64-bit CUDA driver arguments.
macro_rules! ptr {
    ($tensor:expr,$ty:ty,$stream:expr,$storage:ident,$slice:ident,$address:ident,$guard:ident) => {
        let ($storage, _) = $tensor.storage_and_layout();
        let candle::Storage::Cuda($slice) = &*$storage else { candle::bail!("CUDA storage required"); };
        let $slice = $slice.as_cuda_slice::<$ty>()?;
        let ($address,$guard)=$slice.device_ptr($stream);
    }
}

fn routing(dev:&CudaDevice, route:&Option<Tensor>, sorted:&Tensor, experts:&Tensor,m:usize)->Result<()> {
    layout(sorted,dev,DType::U32,m)?;
    layout(experts,dev,DType::U32,m)?;
    if let Some(w)=route { layout(w,dev,DType::F32,m)?; }
    Ok(())
}

pub(super) fn simt(input:&Tensor, weights:&Tensor, route:&Option<Tensor>, sorted:&Tensor, experts:&Tensor,topk:usize)->Result<Tensor>{
    let dev=input.device().as_cuda_device()?;
    let (e,n,k)=weights.dims3()?;
    let (r,m)=dimensions(input,route,topk,e,n,k)?;
    if e>1024 || k%2!=0 || n.div_ceil(32)>65535 { candle::bail!("DKR SIMT MoE unsupported geometry"); }
    layout(input,dev,DType::F16,r*k)?;
    layout(weights,dev,DType::F16,e.checked_mul(n*k).ok_or_else(||candle::Error::Msg("MoE weight size overflow".into()))?)?;
    routing(dev,route,sorted,experts,m)?;
    let stream=dev.cuda_stream();
    let count=dev.dkr_moe_function("moe_simt_f16","count")?;
    let scan=dev.dkr_moe_function("moe_simt_f16","scan")?;
    let gemm=dev.dkr_moe_function("moe_simt_f16","gemm")?;
    let mut counts=dev.alloc_zeros::<u32>(e)?;
    let mut offsets=unsafe{dev.alloc::<u32>(e+1)?};
    let mut out=unsafe{dev.alloc::<half::f16>(m*n)?};
    ptr!(input,half::f16,&stream,is,ix,ip,ig);
    ptr!(weights,half::f16,&stream,ws,wx,wp,wg);
    ptr!(sorted,u32,&stream,ss,sx,sp,sg);
    ptr!(experts,u32,&stream,es,ex,ep,eg);
    let rw=route.as_ref().map(|w|w.storage_and_layout().0);
    let (rp,rg)=if let Some(w)=&rw {
        let candle::Storage::Cuda(w)=&**w else { candle::bail!("CUDA route weights required") };
        let (p,g)=w.as_cuda_slice::<f32>()?.device_ptr(&stream);
        (p,Some(g))
    } else {(0,None)};
    let (cp,cg)=counts.device_ptr_mut(&stream);
    let (op,og)=offsets.device_ptr_mut(&stream);
    let (yp,yg)=out.device_ptr_mut(&stream);
    let (ei,ti,mi,ni,ki)=(e as i32,topk as i32,m as i32,n as i32,k as i32);
    let mut b=stream.launch_builder(&count);
    b.arg(&ep).arg(&cp).arg(&mi);
    unsafe{b.launch(LaunchConfig{grid_dim:(m.div_ceil(256) as u32,1,1),block_dim:(256,1,1),shared_mem_bytes:0}).w()?;}
    // Same integer prefix sum for prefill and decode, bounded to 1024 experts.
    let scan_threads=e.max(32) as u32;
    let mut b=stream.launch_builder(&scan);
    b.arg(&cp).arg(&op).arg(&ei);
    unsafe{b.launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(scan_threads,1,1),shared_mem_bytes:scan_threads*4}).w()?;}
    let mut b=stream.launch_builder(&gemm);
    b.arg(&ip).arg(&wp).arg(&sp).arg(&op).arg(&rp).arg(&yp)
        .arg(&ei).arg(&ti).arg(&mi).arg(&ni).arg(&ki);
    unsafe{b.launch(LaunchConfig{grid_dim:(e as u32,n.div_ceil(32) as u32,1),block_dim:(128,1,1),shared_mem_bytes:2304}).w()?;}
    drop((ig,wg,sg,eg,rg,cg,og,yg));
    result(out,dev,m,n)
}

pub(super) fn gguf(input:&Tensor,weights:&QTensor,route:&Option<Tensor>,sorted:&Tensor,experts:&Tensor,topk:usize)->Result<Tensor>{
    let dev=input.device().as_cuda_device()?;
    let (e,n,k)=weights.shape().dims3()?;
    let (r,m)=dimensions(input,route,topk,e,n,k)?;
    if weights.device().as_cuda_device()?.id()!=dev.id() || m>65535 { candle::bail!("DKR GGUF MoE device/grid mismatch"); }
    let (codec,qk,bytes)=match weights.dtype(){
        GgmlDType::Q8_0=>("q8_0",32,34),
        GgmlDType::Q4K=>("q4_k",256,144),
        GgmlDType::Q2K=>("q2_k",256,84),
        GgmlDType::Q3K=>("q3_k",256,110),
        GgmlDType::Q5K=>("q5_k",256,176),
        GgmlDType::Q6K=>("q6_k",256,210),
        _=>candle::bail!("unsupported DKR GGUF MoE codec"),
    };
    if k%qk!=0 { candle::bail!("invalid GGUF MoE block shape"); }
    let padded=k.div_ceil(512)*512;
    if padded.checked_mul(r).is_none_or(|v|v>i32::MAX as usize) { candle::bail!("GGUF MoE padded index overflow"); }
    let shared=(k/qk)*bytes*4+1024;
    if shared>49152 { candle::bail!("GGUF MoE exceeds SM61 shared memory"); }
    layout(input,dev,DType::F32,r*k)?;
    routing(dev,route,sorted,experts,m)?;
    let stream=dev.cuda_stream();
    let quant=dev.dkr_moe_function("moe_gguf","quantize")?;
    let gemm=dev.dkr_moe_function("moe_gguf",codec)?;
    // Candle's stream-ordered allocations, no direct cudaMallocAsync FFI.
    let mut scratch=unsafe{dev.alloc::<u8>(r*(padded/32)*36)?};
    let mut out=unsafe{dev.alloc::<f32>(m*n)?};
    ptr!(input,f32,&stream,is,ix,ip,ig);
    ptr!(sorted,u32,&stream,ss,sx,sp,sg);
    ptr!(experts,u32,&stream,es,ex,ep,eg);
    let (wp,wg)=weights.device_ptr_with_guard(&stream)?;
    let wp=wp as u64;
    let rw=route.as_ref().map(|w|w.storage_and_layout().0);
    let (rp,rg)=if let Some(w)=&rw {
        let candle::Storage::Cuda(w)=&**w else { candle::bail!("CUDA route weights required") };
        let (p,g)=w.as_cuda_slice::<f32>()?.device_ptr(&stream);
        (p,Some(g))
    } else {(0,None)};
    let (qp,qg)=scratch.device_ptr_mut(&stream);
    let (yp,yg)=out.device_ptr_mut(&stream);
    let (ei,ti,mi,ni,ki,pi)=(e as i32,topk as i32,m as i32,n as i32,k as i32,padded as i32);
    let mut b=stream.launch_builder(&quant);
    b.arg(&ip).arg(&qp).arg(&ki).arg(&pi);
    unsafe{b.launch(LaunchConfig{grid_dim:(padded.div_ceil(256) as u32,r as u32,1),block_dim:(256,1,1),shared_mem_bytes:0}).w()?;}
    let mut b=stream.launch_builder(&gemm);
    b.arg(&wp).arg(&qp).arg(&sp).arg(&ep).arg(&rp).arg(&yp)
        .arg(&ei).arg(&ti).arg(&mi).arg(&ni).arg(&ki).arg(&pi);
    unsafe{b.launch(LaunchConfig{grid_dim:(n.div_ceil(4) as u32,m as u32,1),block_dim:(32,4,1),shared_mem_bytes:shared as u32}).w()?;}
    drop((ig,sg,eg,wg,rg,qg,yg));
    result(out,dev,m,n)
}
