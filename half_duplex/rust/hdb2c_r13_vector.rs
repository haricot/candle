use candle::{DType,Device,Error,Result,Tensor};
use std::ffi::{c_float,c_int};
unsafe extern "C" {
    fn hdb2c13_unary_8(op:c_int,n:c_int,x:*const c_float,y:*mut c_float)->c_int;
    fn hdb2c13_unary_16(op:c_int,n:c_int,x:*const c_float,y:*mut c_float)->c_int;
    fn hdb2c13_glu_8(batch:c_int,channels:c_int,time:c_int,x:*const c_float,y:*mut c_float)->c_int;
    fn hdb2c13_glu_16(batch:c_int,channels:c_int,time:c_int,x:*const c_float,y:*mut c_float)->c_int;
    fn hdb2c13_softmax_8(rows:c_int,n:c_int,x:*const c_float,y:*mut c_float)->c_int;
    fn hdb2c13_softmax_16(rows:c_int,n:c_int,x:*const c_float,y:*mut c_float)->c_int;
}
fn fail(s:&str)->Error {Error::Msg(s.into())}
fn dim(n:usize)->Result<i32>{i32::try_from(n).map_err(|_|fail("R13 vector dimension exceeds ABI"))}
fn cpu(x:&Tensor)->Result<()> {
    if !matches!(x.device(),Device::Cpu) || x.dtype()!=DType::F32 {return Err(fail("R13 vector CPU FP32 required"));}
    Ok(())
}
pub fn width()->usize {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vl") {16} else {8}
}
pub fn unary(op:i32,x:&[f32])->Result<Vec<f32>> {
    crate::parakeet::hdb2c_r10_mkl::profile();let n=dim(x.len())?;
    if x.is_empty() || !(0..=3).contains(&op) {return Err(fail("R13 invalid vector operation"));}
    let mut y=vec![0f32;x.len()];
    let status=unsafe {if width()==16 {hdb2c13_unary_16(op,n,x.as_ptr(),y.as_mut_ptr())}
        else {hdb2c13_unary_8(op,n,x.as_ptr(),y.as_mut_ptr())}};
    if status!=0 {return Err(fail("R13 native vector operation failed"));}Ok(y)
}
pub fn silu(x:&Tensor)->Result<Tensor> {
    cpu(x)?;let input=x.flatten_all()?.to_vec1::<f32>()?;
    Tensor::from_vec(unary(3,&input)?,x.shape(),x.device())
}
pub fn glu(x:&Tensor)->Result<Tensor> {
    cpu(x)?;let (b,c2,t)=x.dims3()?;
    if c2==0 || c2%2!=0 || b==0 || t==0 {return Err(fail("R13 GLU shape differs"));}
    let (bi,ci,ti)=(dim(b)?,dim(c2/2)?,dim(t)?);
    let input=x.flatten_all()?.to_vec1::<f32>()?;let mut out=vec![0f32;b*(c2/2)*t];
    let status=unsafe {if width()==16 {hdb2c13_glu_16(bi,ci,ti,input.as_ptr(),out.as_mut_ptr())}
        else {hdb2c13_glu_8(bi,ci,ti,input.as_ptr(),out.as_mut_ptr())}};
    if status!=0 {return Err(fail("R13 native GLU failed"));}
    Tensor::from_vec(out,(b,c2/2,t),x.device())
}
pub fn softmax(x:&Tensor)->Result<Tensor> {
    cpu(x)?;let n=*x.dims().last().ok_or_else(||fail("R13 scalar softmax"))?;
    if n==0 {return Err(fail("R13 empty softmax"));}
    let rows=dim(x.elem_count()/n)?;let ni=dim(n)?;
    let input=x.flatten_all()?.to_vec1::<f32>()?;let mut out=vec![0f32;input.len()];
    let status=unsafe {if width()==16 {hdb2c13_softmax_16(rows,ni,input.as_ptr(),out.as_mut_ptr())}
        else {hdb2c13_softmax_8(rows,ni,input.as_ptr(),out.as_mut_ptr())}};
    if status!=0 {return Err(fail("R13 native softmax failed"));}
    Tensor::from_vec(out,x.shape(),x.device())
}
#[cfg(test)]
mod hdb2c13_vector_tests {
    use super::*;
    #[test] fn softmax_stability_and_shape_guards()->Result<()> {
        let x=Tensor::from_vec(vec![10000f32,10000.,-10000.,-10000.],(2,2),&Device::Cpu)?;
        assert_eq!(softmax(&x)?.flatten_all()?.to_vec1::<f32>()?,vec![0.5;4]);
        assert!(softmax(&x.to_dtype(DType::F64)?).is_err());
        assert!(glu(&Tensor::zeros((1,3,2),DType::F32,&Device::Cpu)?).is_err());
        assert!(unary(4,&[1.]).is_err());assert!(unary(3,&[]).is_err());
        Ok(())
    }
}
