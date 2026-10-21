use candle::{DType, Device, Error, Result, Tensor};
use std::ffi::{c_float,c_int};

unsafe extern "C" {
    fn cblas_sgemm(layout:c_int,trans_a:c_int,trans_b:c_int,m:c_int,n:c_int,k:c_int,
        alpha:c_float,a:*const c_float,lda:c_int,b:*const c_float,ldb:c_int,
        beta:c_float,c:*mut c_float,ldc:c_int);
}
const ROW_MAJOR:i32=101;
const NO_TRANS:i32=111;
const TRANS:i32=112;
fn fail(s:&str)->Error {Error::Msg(s.into())}
fn dim(n:usize)->Result<i32>{i32::try_from(n).map_err(|_|fail("R13 MKL dimension exceeds LP64"))}
fn cpu(x:&Tensor)->Result<()> {
    if !matches!(x.device(),Device::Cpu) || x.dtype()!=DType::F32 {
        return Err(fail("R13 MKL requires CPU FP32"));
    }
    Ok(())
}

pub fn linear(layer:&candle_nn::Linear,x:&Tensor)->Result<Tensor> {
    cpu(x)?;cpu(layer.weight())?;
    crate::parakeet::hdb2c_r10_mkl::profile();
    let inner=*x.dims().last().ok_or_else(||fail("R13 scalar linear input"))?;
    let (channels,k)=layer.weight().dims2()?;
    if inner==0 || channels==0 || k!=inner {return Err(fail("R13 linear shape differs"));}
    let rows=x.elem_count()/inner;
    let (m,n,k)=(dim(rows)?,dim(channels)?,dim(inner)?);
    let input=x.flatten_all()?.to_vec1::<f32>()?;
    let weight=layer.weight().flatten_all()?.to_vec1::<f32>()?;
    let mut out=vec![0f32;rows*channels];
    let beta=if let Some(bias)=layer.bias() {
        cpu(bias)?;let bias=bias.to_vec1::<f32>()?;
        if bias.len()!=channels {return Err(fail("R13 linear bias shape differs"));}
        for row in out.chunks_exact_mut(channels) {row.copy_from_slice(&bias);}
        1.0
    } else {0.0};
    unsafe {cblas_sgemm(ROW_MAJOR,NO_TRANS,TRANS,m,n,k,1.0,input.as_ptr(),k,
        weight.as_ptr(),k,beta,out.as_mut_ptr(),n);}
    let mut shape=x.dims().to_vec();*shape.last_mut().unwrap()=channels;
    Tensor::from_vec(out,shape.as_slice(),x.device())
}

// Transposition is explicit so MKL sees the same dot-product orientation as
// NeMo's q @ k.T and q @ p.T rather than a pretransposed packed operand.
pub fn bmm(a:&Tensor,b:&Tensor,transpose_b:bool)->Result<Tensor> {
    cpu(a)?;cpu(b)?;crate::parakeet::hdb2c_r10_mkl::profile();
    let (batch,heads,m,k)=a.dims4()?;
    let (bb,bh,brows,bcols)=b.dims4()?;
    let (bk,n)=if transpose_b {(bcols,brows)} else {(brows,bcols)};
    if (batch,heads)!=(bb,bh) || k!=bk || m==0 || k==0 || n==0 {
        return Err(fail("R13 batched matmul shape differs"));
    }
    let (mi,ni,ki,ldb)=(dim(m)?,dim(n)?,dim(k)?,dim(bcols)?);
    let aa=a.flatten_all()?.to_vec1::<f32>()?;let bb=b.flatten_all()?.to_vec1::<f32>()?;
    let mut out=vec![0f32;batch*heads*m*n];
    for index in 0..batch*heads {
        unsafe {cblas_sgemm(ROW_MAJOR,NO_TRANS,if transpose_b {TRANS} else {NO_TRANS},
            mi,ni,ki,1.0,aa.as_ptr().add(index*m*k),ki,
            bb.as_ptr().add(index*brows*bcols),ldb,0.0,out.as_mut_ptr().add(index*m*n),ni);}
    }
    Tensor::from_vec(out,(batch,heads,m,n),a.device())
}

pub fn attention_scores(a:&Tensor,b:&Tensor,head_dim:usize)->Result<Tensor> {
    cpu(a)?;cpu(b)?;
    if head_dim==0 || a.shape()!=b.shape() {return Err(fail("R13 attention score shape differs"));}
    let sum=a.add(b)?;
    // NeMo adds the two FP32 matrices, then divides by a FP32 scalar.
    // Tensor/scalar affine multiplication would change rounding here.
    let divisor=Tensor::new((head_dim as f64).sqrt() as f32,a.device())?;
    sum.broadcast_div(&divisor)
}

pub fn conv1d(layer:&candle_nn::Conv1d,x:&Tensor)->Result<Tensor> {
    cpu(x)?;let config=layer.config();
    // The checkpoint performs temporal padding explicitly before depthwise
    // convolution; its three Conv1d operators all use stride 1 and padding 0.
    if config.padding!=0 || config.stride!=1 {
        return Err(fail("R13 requires the checkpoint's explicit Conv1d padding"));
    }
    let conv=candle_nn::Conv2d::new(layer.weight().unsqueeze(2)?,layer.bias().cloned(),
        candle_nn::Conv2dConfig {padding:0,stride:1,dilation:config.dilation,
            groups:config.groups,cudnn_fwd_algo:None});
    crate::parakeet::hdb2c_r11_onednn::conv2d(&conv,&x.unsqueeze(2)?)?.squeeze(2)
}

#[cfg(test)]
mod hdb2c13_math_tests {
    use super::*;
    #[test] fn linear_optional_bias_and_noncontiguous_rows()->Result<()> {
        let weight=Tensor::from_vec(vec![1f32,2.,3.,4.],(2,2),&Device::Cpu)?;
        let x=Tensor::from_vec(vec![1f32,3.,2.,4.],(1,2,2),&Device::Cpu)?.transpose(1,2)?;
        let layer=candle_nn::Linear::new(weight.clone(),None);
        assert_eq!(linear(&layer,&x)?.flatten_all()?.to_vec1::<f32>()?,vec![5.,11.,11.,25.]);
        let layer=candle_nn::Linear::new(weight,Some(Tensor::from_vec(vec![10f32,20.],2,&Device::Cpu)?));
        assert_eq!(linear(&layer,&x)?.flatten_all()?.to_vec1::<f32>()?,vec![15.,31.,21.,45.]);
        Ok(())
    }
    #[test] fn bmm_head_layout_transpose_and_guards()->Result<()> {
        let a=Tensor::from_vec(vec![1f32,2.,3.,4.,2.,1.,4.,3.],(1,2,2,2),&Device::Cpu)?;
        let b=Tensor::from_vec(vec![1f32,0.,0.,2.,3.,0.,0.,4.],(1,2,2,2),&Device::Cpu)?;
        let expected=vec![1f32,4.,3.,8.,6.,4.,12.,12.];
        assert_eq!(bmm(&a,&b,false)?.flatten_all()?.to_vec1::<f32>()?,expected);
        assert_eq!(bmm(&a,&b.transpose(2,3)?,true)?.flatten_all()?.to_vec1::<f32>()?,expected);
        assert!(bmm(&a,&b.narrow(1,0,1)?,false).is_err());
        assert!(bmm(&a.to_dtype(DType::F64)?,&b,false).is_err());
        Ok(())
    }
    #[test] fn attention_adds_before_actual_fp32_division()->Result<()> {
        let a=Tensor::from_vec(vec![100001f32,-233.,91.],3,&Device::Cpu)?;
        let b=Tensor::from_vec(vec![-99991f32,173.,22.],3,&Device::Cpu)?;
        let y=attention_scores(&a,&b,80)?.to_vec1::<f32>()?;
        assert_eq!(y,vec![10f32/(80f32).sqrt(),-60f32/(80f32).sqrt(),113f32/(80f32).sqrt()]);
        Ok(())
    }
    #[test] fn temporal_grouped_convolution_and_explicit_padding_guard()->Result<()> {
        let w=Tensor::from_vec(vec![1f32,0.,-1.,1.,1.,1.],(2,1,3),&Device::Cpu)?;
        let bias=Tensor::from_vec(vec![1f32,2.],2,&Device::Cpu)?;
        let config=candle_nn::Conv1dConfig {groups:2,..Default::default()};
        let layer=candle_nn::Conv1d::new(w.clone(),Some(bias.clone()),config);
        let x=Tensor::from_vec(vec![1f32,2.,3.,4.,5.,6.,7.,8.],(1,2,4),&Device::Cpu)?;
        assert_eq!(conv1d(&layer,&x)?.flatten_all()?.to_vec1::<f32>()?,vec![-1.,-1.,20.,23.]);
        let layer=candle_nn::Conv1d::new(w,Some(bias),
            candle_nn::Conv1dConfig {groups:2,padding:1,..Default::default()});
        assert!(conv1d(&layer,&x).is_err());
        Ok(())
    }
}
