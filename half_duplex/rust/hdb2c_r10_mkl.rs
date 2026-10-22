use candle::{Device,Error,Result,Tensor};
use std::{ffi::{c_char,c_float,c_int,c_long},sync::OnceLock};

unsafe extern "C" {
    fn hdb2c10_cpu_init(threads:c_int,version:*mut c_char,length:c_int);
    fn hdb2c10_cpu_rfft(batch:c_int,n:c_int,input:*const c_float,output:*mut c_float)->c_long;
    fn hdb2c10_cpu_addmm(rows:c_int,channels:c_int,inner:c_int,x:*const c_float,w:*const c_float,bias:*const c_float,output:*mut c_float);
    fn hdb2c10_cpu_mel(channels:c_int,frames:c_int,bins:c_int,w:*const c_float,x:*const c_float,out:*mut c_float);
    fn hdb2c10_cpu_ln(n:c_int,x:*const c_float,out:*mut c_float);
    fn hdb2c10_cpu_sqrt(n:c_int,x:*const c_float,out:*mut c_float);
}
fn fail(s:&str)->Error {Error::Msg(s.into())}
fn dim(n:usize)->Result<i32>{i32::try_from(n).map_err(|_|fail("R10 MKL dimension exceeds LP64 interface"))}
pub fn profile()->&'static str {
    static VERSION:OnceLock<String>=OnceLock::new();
    VERSION.get_or_init(||{
        let mut text=vec![0u8;256];
        unsafe {hdb2c10_cpu_init(candle::utils::get_num_threads() as i32,text.as_mut_ptr().cast(),text.len() as i32);}
        let n=text.iter().position(|v|*v==0).unwrap_or(text.len());
        String::from_utf8_lossy(&text[..n]).into_owned()
    })
}
pub fn rfft(input:&[f32],frames:usize,n:usize)->Result<Vec<f32>>{
    profile();
    if n==0 || frames==0 || frames.checked_mul(n)!=Some(input.len()) || input.iter().any(|v|!v.is_finite()) {return Err(fail("R10 invalid FFT shape/value"));}
    dim(n)?;dim(frames)?;
    let mut output=vec![0f32;frames*(n/2+1)*2];
    let status=unsafe{hdb2c10_cpu_rfft(dim(frames)?,dim(n)?,input.as_ptr(),output.as_mut_ptr())};
    if status!=0{return Err(Error::Msg(format!("R10 MKL DFTI status {status}")));}
    Ok(output)
}
pub fn sqrt(input:&[f32])->Result<Vec<f32>>{
    if input.iter().any(|v|!v.is_finite() || *v<0.0){return Err(fail("R10 invalid square-root argument"));}
    profile();let mut out=vec![0f32;input.len()];
    unsafe{hdb2c10_cpu_sqrt(dim(input.len())?,input.as_ptr(),out.as_mut_ptr());} Ok(out)
}
pub fn ln(input:&[f32])->Result<Vec<f32>>{
    if input.iter().any(|v|!v.is_finite() || *v<=0.0){return Err(fail("R10 invalid logarithm argument"));}
    profile();let mut out=vec![0f32;input.len()];
    unsafe{hdb2c10_cpu_ln(dim(input.len())?,input.as_ptr(),out.as_mut_ptr());} Ok(out)
}
pub fn mel(weights:&[f32],power:&[f32],channels:usize,frames:usize,bins:usize)->Result<Vec<f32>>{
    profile();
    if channels==0 || frames==0 || bins==0 || channels.checked_mul(bins)!=Some(weights.len()) || frames.checked_mul(bins)!=Some(power.len()) {return Err(fail("R10 invalid Mel shape"));}
    dim(channels)?;dim(frames)?;dim(bins)?;
    let mut x=vec![0f32;power.len()];
    for t in 0..frames {for k in 0..bins {x[k*frames+t]=power[t*bins+k];}}
    let mut out=vec![0f32;channels*frames];
    unsafe{hdb2c10_cpu_mel(dim(channels)?,dim(frames)?,dim(bins)?,weights.as_ptr(),x.as_ptr(),out.as_mut_ptr());}
    Ok(out)
}
pub fn linear(layer:&candle_nn::Linear,x:&Tensor)->Result<Tensor>{
    profile();
    if !matches!(x.device(),Device::Cpu) || x.dtype()!=candle::DType::F32 {return Err(fail("R10 requires native CPU FP32"));}
    let shape=x.dims();let inner=*shape.last().ok_or_else(||fail("R10 scalar linear input"))?;
    let (channels,wi)=layer.weight().dims2()?;
    if wi!=inner || inner==0{return Err(fail("R10 linear shape mismatch"));}
    let rows=x.elem_count()/inner;
    let values=x.flatten_all()?.to_vec1::<f32>()?;
    let weight=layer.weight().flatten_all()?.to_vec1::<f32>()?;
    let bias=layer.bias().ok_or_else(||fail("R10 qualified projection requires its checkpoint bias"))?.to_vec1::<f32>()?;
    if bias.len()!=channels{return Err(fail("R10 linear bias shape mismatch"));}
    let mut out=vec![0f32;rows*channels];
    unsafe{hdb2c10_cpu_addmm(dim(rows)?,dim(channels)?,dim(inner)?,values.as_ptr(),weight.as_ptr(),bias.as_ptr(),out.as_mut_ptr());}
    let mut shape=shape.to_vec();*shape.last_mut().unwrap()=channels;
    Tensor::from_vec(out,shape.as_slice(),x.device())
}

#[cfg(test)]
mod hdb2c10_tests {
    use super::*;
    #[test] fn fft_impulse_and_bounds(){
        let mut x=vec![0f32;512];x[0]=1.0;
        let y=rfft(&x,1,512).unwrap();
        assert!(y.chunks_exact(2).all(|v|v[0]==1.0 && v[1]==0.0));
        assert!(rfft(&x,2,512).is_err());
        assert!(rfft(&[f32::NAN],1,1).is_err());
    }
    #[test] fn vector_math_domains(){
        assert_eq!(sqrt(&[0.0,4.0]).unwrap(),vec![0.0,2.0]);
        assert_eq!(ln(&[1.0]).unwrap(),vec![0.0]);
        assert!(ln(&[0.0]).is_err());assert!(sqrt(&[-1.0]).is_err());
    }
    #[test] fn projection_layout_and_bias(){
        let w=Tensor::from_vec(vec![1f32,0.0,0.0,1.0],(2,2),&Device::Cpu).unwrap();
        let b=Tensor::from_vec(vec![2f32,3.0],2,&Device::Cpu).unwrap();
        let layer=candle_nn::Linear::new(w,Some(b));
        let x=Tensor::from_vec(vec![4f32,5.0,6.0,7.0],(1,2,2),&Device::Cpu).unwrap();
        assert_eq!(linear(&layer,&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(),vec![6.0,8.0,8.0,10.0]);
    }
}
