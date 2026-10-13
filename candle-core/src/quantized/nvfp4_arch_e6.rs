//! ARCH-E6: local-only native Candle CUDA backend bridge, explicit opt-in.
//! This is NOT integrated into QMatMul or model forward. The borrowed A16 input
//! remains FP16; W4A8 quantization is performed by Candle's CUDA Q8_1 kernel.
use crate::{CudaDevice, Result};
use crate::cuda_backend::{cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg}, WrapErr};
use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4A8Variant { Reference, Tiled4x8, Auto }

pub struct Nvfp4Sm61Native {
    device: CudaDevice,
    weights: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    global_scale: f32,
    n: usize,
    k: usize,
}

impl Nvfp4Sm61Native {
    pub fn from_packed(device: &CudaDevice, n: usize, k: usize,
        packed: &[u8], scales: &[u8], global_scale: f32) -> Result<Self> {
        let elems=match n.checked_mul(k) {
            Some(elems)=>elems,
            None=>crate::bail!("ARCH-E6 weight shape overflow"),
        };
        if n==0 || k==0 || k%32!=0 || elems>i32::MAX as usize ||
            packed.len()!=elems/2 || scales.len()!=elems/16 ||
            !global_scale.is_finite() || global_scale<=0.0 ||
            scales.iter().any(|&x| x&0x80!=0 || x==0x7f) {
            crate::bail!("ARCH-E6 invalid NVFP4 payload or scale convention")
        }
        Ok(Self {device:device.clone(),weights:device.clone_htod(packed)?,
                 scales:device.clone_htod(scales)?,global_scale,n,k})
    }
    pub fn weight_shape(&self)->(usize,usize) {(self.n,self.k)}
    fn check_batch(&self, input:&CudaSlice<f16>, batch:usize) -> Result<()> {
        if batch==0 || batch>65535 || input.len()!=batch*self.k {
            crate::bail!("ARCH-E6 A16 buffer must be contiguous [batch,k], batch 1..65535")
        }
        Ok(())
    }
    pub fn forward_w4a16_scalar(&self,input:&CudaSlice<f16>,batch:usize)->Result<CudaSlice<f32>> {
        self.check_batch(input,batch)?;
        if self.n.checked_mul(batch).is_none_or(|x| x>i32::MAX as usize) {
            crate::bail!("ARCH-E6 output grid exceeds i32")
        }
        let out=self.device.alloc_zeros::<f32>(self.n*batch)?;
        let func=self.device.get_or_load_func("nvfp4_arch_e6_w4a16_scalar",&candle_kernels::QUANTIZED)?;
        let cfg=LaunchConfig {grid_dim:((self.n*batch).div_ceil(128) as u32,1,1),block_dim:(128,1,1),shared_mem_bytes:0};
        let ni=self.n as i32;let ki=self.k as i32;let bi=batch as i32;
        let mut b=func.builder();b.arg(&self.weights);b.arg(&self.scales);b.arg(&self.global_scale);
        b.arg(input);b.arg(&out);b.arg(&ni);b.arg(&ki);b.arg(&bi);
        unsafe{b.launch(cfg)}.w()?;Ok(out)
    }
    pub fn quantize_a16_to_q8(&self,input:&CudaSlice<f16>,batch:usize)->Result<CudaSlice<u8>> {
        self.check_batch(input,batch)?;
        let f32buf=self.device.alloc_zeros::<f32>(batch*self.k)?;
        let func=self.device.get_or_load_func("nvfp4_arch_e6_f16_to_f32",&candle_kernels::QUANTIZED)?;
        let total=(batch*self.k) as i32;
        let cfg=LaunchConfig{grid_dim:((batch*self.k).div_ceil(256) as u32,1,1),block_dim:(256,1,1),shared_mem_bytes:0};
        let mut b=func.builder();b.arg(input);b.arg(&f32buf);b.arg(&total);
        unsafe{b.launch(cfg)}.w()?;
        let padded=self.k.div_ceil(512)*512;
        let q8=self.device.alloc_zeros::<u8>(batch*(padded/32)*36)?;
        let qfunc=self.device.get_or_load_func("quantize_q8_1",&candle_kernels::QUANTIZED)?;
        let qc=LaunchConfig{grid_dim:((padded/256) as u32,batch as u32,1),block_dim:(256,1,1),shared_mem_bytes:0};
        let ki=self.k as i32;let pi=padded as i32;
        let mut qb=qfunc.builder();qb.arg(&f32buf);qb.arg(&q8);qb.arg(&ki);qb.arg(&pi);
        unsafe{qb.launch(qc)}.w()?;
        Ok(q8)
    }
    pub fn forward_q8(&self,q8:&CudaSlice<u8>,batch:usize,variant:W4A8Variant)->Result<CudaSlice<f32>> {
        if batch==0 || batch>65535 {crate::bail!("ARCH-E6 invalid batch")}
        let padded=self.k.div_ceil(512)*512;
        if q8.len()!=batch*(padded/32)*36 {crate::bail!("ARCH-E6 Q8_1 buffer byte count mismatch")}
        let actual=match variant {
            W4A8Variant::Auto=>{
                let cc=self.device.cuda_stream().context().compute_capability().w()?;
                if cc==(6,1) && self.n>=512 && self.k>=1024 && batch>=9 {W4A8Variant::Tiled4x8}
                else {W4A8Variant::Reference}
            }, x=>x
        };
        if actual==W4A8Variant::Tiled4x8 {
            let cc=self.device.cuda_stream().context().compute_capability().w()?;
            if cc.0*10+cc.1 < 61 {crate::bail!("ARCH-E6 DP4A requires SM61+")}
        }
        let out=self.device.alloc_zeros::<f32>(batch*self.n)?;
        let name=if actual==W4A8Variant::Tiled4x8 {"nvfp4_mat_mul_q8_1_tiled4x8"}else{"nvfp4_mat_mul_q8_1"};
        let func=self.device.get_or_load_func(name,&candle_kernels::QUANTIZED)?;
        let cfg=LaunchConfig{grid_dim:if actual==W4A8Variant::Tiled4x8
            {(self.n.div_ceil(4) as u32,batch.div_ceil(8) as u32,1)}else{(self.n as u32,batch as u32,1)},
            block_dim:(32,4,1),shared_mem_bytes:0};
        let ki=self.k as i32;let ni=self.n as i32;let bi=batch as i32;
        let pi=padded as i32;let di=ni;
        let mut b=func.builder();b.arg(&self.weights);b.arg(&self.scales);b.arg(&self.global_scale);
        b.arg(q8);b.arg(&out);b.arg(&ki);b.arg(&ni);b.arg(&bi);b.arg(&pi);b.arg(&di);
        unsafe{b.launch(cfg)}.w()?;Ok(out)
    }
    pub fn forward_w4a8(&self,input:&CudaSlice<f16>,batch:usize,variant:W4A8Variant)->Result<CudaSlice<f32>> {
        let q8=self.quantize_a16_to_q8(input,batch)?;
        self.forward_q8(&q8,batch,variant)
    }
}
