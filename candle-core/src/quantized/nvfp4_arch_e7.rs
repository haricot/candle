//! ARCH-E7-A: opt-in Candle Tensor CustomOp1 for ModelOpt NVFP4, backed by E6.
//! No invented logits. Actual GLM MoE model routing/attention is NOT implemented here.
//! Native DP4A is used only when `W4A8Dp4a` is explicitly selected.
use crate::{CpuStorage, CudaDevice, CudaStorage, DType, Layout, Result, Shape, Tensor};
use crate::CustomOp1;
use crate::quantized::nvfp4_arch_e6::{Nvfp4Sm61Native, W4A8Variant};
use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum E7Mode { W4A16Reference, W4A8Reference, W4A8Dp4a }

pub struct Nvfp4E7Linear { backend: Nvfp4Sm61Native, device: CudaDevice, n:usize, k:usize }

impl Nvfp4E7Linear {
    pub fn from_modelopt_bytes(device: &CudaDevice, n:usize, k:usize,
        packed:&[u8], scales_e4m3:&[u8], weight_scale_2:f32)->Result<Self> {
        let backend=Nvfp4Sm61Native::from_packed(device,n,k,packed,scales_e4m3,weight_scale_2)?;
        Ok(Self{backend,device:device.clone(),n,k})
    }
    pub fn forward(&self,input:&Tensor,mode:E7Mode)->Result<Tensor>{
        // Explicitly require FP16: silently casting BF16 changes precision, and
        // re-quantizing FP16 weights is not the NVFP4 ModelOpt format.
        if input.dtype()!=DType::F16 || input.dims().len()!=2 || input.dims()[1]!=self.k {
            crate::bail!("ARCH-E7 input requires rank-2 FP16 Tensor [tokens,{}]",self.k)
        }
        if input.dims()[0]==0 || input.dims()[0]>65535 {crate::bail!("ARCH-E7 invalid token count")}
        let op=ForwardOp{linear:self,mode};
        input.apply_op1_no_bwd(&op)
    }
    pub fn shape(&self)->(usize,usize){(self.n,self.k)}
}
struct ForwardOp<'a>{linear:&'a Nvfp4E7Linear,mode:E7Mode}
impl CustomOp1 for ForwardOp<'_>{
    fn name(&self)->&'static str{"haricot.arch-e7.modelopt-nvfp4-sm61"}
    fn cpu_fwd(&self,_s:&CpuStorage,_l:&Layout)->Result<(CpuStorage,Shape)>{
        crate::bail!("ARCH-E7 native mode is CUDA-only; explicit fallback required")
    }
    fn cuda_fwd(&self,storage:&CudaStorage,layout:&Layout)->Result<(CudaStorage,Shape)>{
        if !layout.is_contiguous() || layout.start_offset()!=0 || layout.dims().len()!=2 {
            crate::bail!("ARCH-E7 requires contiguous, zero-offset [tokens,k] Tensor")
        }
        let batch=layout.dims()[0];
        if layout.dims()[1]!=self.linear.k {crate::bail!("ARCH-E7 K shape mismatch")}
        let input=storage.as_cuda_slice::<f16>()?;
        if input.len()!=batch*self.linear.k {
            crate::bail!("ARCH-E7 refuses storage with unmatched length or aliased slice")
        }
        let out=match self.mode {
            E7Mode::W4A16Reference=>self.linear.backend.forward_w4a16_scalar(input,batch)?,
            E7Mode::W4A8Reference=>self.linear.backend.forward_w4a8(input,batch,W4A8Variant::Reference)?,
            E7Mode::W4A8Dp4a=>self.linear.backend.forward_w4a8(input,batch,W4A8Variant::Tiled4x8)?,
        };
        Ok((CudaStorage::wrap_cuda_slice(out,self.linear.device.clone()),(batch,self.linear.n).into()))
    }
}
