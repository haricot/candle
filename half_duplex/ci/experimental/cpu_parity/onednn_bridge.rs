use candle::{Device,DType,Error,Result,Tensor};
use std::ffi::{c_char,c_float,c_int};
unsafe extern "C" {
    fn hdb2c11_conv2d(batch:c_int,ic:c_int,ih:c_int,iw:c_int,oc:c_int,kh:c_int,kw:c_int,
        stride:c_int,pad:c_int,dilation:c_int,groups:c_int,threads:c_int,
        x:*const c_float,w:*const c_float,b:*const c_float,y:*mut c_float,
        message:*mut c_char,size:c_int)->c_int;
}
fn fail(s:&str)->Error {Error::Msg(s.into())}
fn dim(n:usize)->Result<i32>{i32::try_from(n).map_err(|_|fail("oneDNN dimension exceeds prototype limit"))}
pub fn conv2d(layer:&candle_nn::Conv2d,x:&Tensor)->Result<Tensor>{
    if !matches!(x.device(),Device::Cpu) || x.dtype()!=DType::F32{return Err(fail("oneDNN CPU FP32 only"));}
    let (batch,ic,ih,iw)=x.dims4()?;let(oc,wic,kh,kw)=layer.weight().dims4()?;let c=layer.config();
    if c.groups==0 || c.stride==0 || c.dilation==0 || ic/c.groups!=wic || ic%c.groups!=0 || oc%c.groups!=0{return Err(fail("invalid oneDNN grouping/config"));}
    let effective_h=c.dilation*(kh-1)+1;let effective_w=c.dilation*(kw-1)+1;
    if ih+2*c.padding<effective_h || iw+2*c.padding<effective_w{return Err(fail("invalid oneDNN convolution size"));}
    let oh=(ih+2*c.padding-effective_h)/c.stride+1;let ow=(iw+2*c.padding-effective_w)/c.stride+1;
    let input=x.flatten_all()?.to_vec1::<f32>()?;let weight=layer.weight().flatten_all()?.to_vec1::<f32>()?;
    let bias=layer.bias().ok_or_else(||fail("oneDNN prototype requires original bias"))?.to_vec1::<f32>()?;
    if bias.len()!=oc{return Err(fail("oneDNN bias shape mismatch"));}
    let mut output=vec![0f32;batch*oc*oh*ow];let mut message=vec![0u8;512];
    let status=unsafe{hdb2c11_conv2d(dim(batch)?,dim(ic)?,dim(ih)?,dim(iw)?,dim(oc)?,dim(kh)?,dim(kw)?,
        dim(c.stride)?,dim(c.padding)?,dim(c.dilation)?,dim(c.groups)?,dim(candle::utils::get_num_threads())?,
        input.as_ptr(),weight.as_ptr(),bias.as_ptr(),output.as_mut_ptr(),message.as_mut_ptr().cast(),message.len() as i32)};
    if status!=0 {return Err(Error::Msg(format!("oneDNN status {status}: {}",String::from_utf8_lossy(&message))));}
    Tensor::from_vec(output,(batch,oc,oh,ow),&Device::Cpu)
}
