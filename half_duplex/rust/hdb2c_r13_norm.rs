use candle::{DType, Device, Error, Result, Tensor};
use candle_nn::{LayerNormConfig, Module, VarBuilder};
use std::ffi::{c_float, c_int};

unsafe extern "C" {
    fn hdb2c12_layernorm(rows:c_int,n:c_int,lanes:c_int,epsilon:c_float,
        x:*const c_float,gamma:*const c_float,beta:*const c_float,out:*mut c_float)->c_int;
}
fn error(message:&str)->Error { Error::Msg(message.into()) }

#[derive(Debug, Clone)]
pub struct OrderedLayerNorm { weight:Tensor, bias:Tensor, epsilon:f32 }

impl OrderedLayerNorm {
    pub fn load(size:usize,config:LayerNormConfig,vb:VarBuilder)->Result<Self> {
        if !config.affine || !config.remove_mean || size==0 || size>16384
            || !config.eps.is_finite() || config.eps<=0.0 {
            return Err(error("R13 requires affine mean-removing FP32 LayerNorm"));
        }
        Ok(Self {weight:vb.get(size,"weight")?,bias:vb.get(size,"bias")?,epsilon:config.eps as f32})
    }
}

impl Module for OrderedLayerNorm {
    fn forward(&self,x:&Tensor)->Result<Tensor> {
        if !matches!(x.device(),Device::Cpu) || x.dtype()!=DType::F32 || x.rank()<2 {
            return Err(error("R13 ordered LayerNorm requires CPU FP32 rows"));
        }
        let n=*x.dims().last().unwrap();
        if n==0 || n>16384 || self.weight.dims()!=[n] || self.bias.dims()!=[n] {
            return Err(error("R13 ordered LayerNorm affine/input shape differs"));
        }
        let rows=x.elem_count()/n;
        let rows=i32::try_from(rows).map_err(|_|error("R13 LayerNorm rows exceed ABI"))?;
        let input=x.flatten_all()?.to_vec1::<f32>()?;
        let gamma=self.weight.to_vec1::<f32>()?;
        let beta=self.bias.to_vec1::<f32>()?;
        let mut out=vec![0f32;input.len()];
        let status=unsafe {hdb2c12_layernorm(rows,n as i32,8,self.epsilon,
            input.as_ptr(),gamma.as_ptr(),beta.as_ptr(),out.as_mut_ptr())};
        if status!=0 {return Err(error("R13 native ordered LayerNorm failed"));}
        Tensor::from_vec(out,x.shape(),x.device())
    }
}

#[cfg(test)]
mod hdb2c13_tests {
    use super::*;
    #[test] fn norm_affine_layout_and_guard() {
        let weight=Tensor::from_vec(vec![2f32,3.],2,&Device::Cpu).unwrap();
        let bias=Tensor::from_vec(vec![4f32,5.],2,&Device::Cpu).unwrap();
        let norm=OrderedLayerNorm {weight,bias,epsilon:1e-5};
        let x=Tensor::from_vec(vec![-1f32,1.,2.,4.],(1,2,2),&Device::Cpu).unwrap();
        let output=norm.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((output[0]-2.).abs()<1e-4 && (output[1]-8.).abs()<1e-4);
        assert!((output[2]-2.).abs()<1e-4 && (output[3]-8.).abs()<1e-4);
        assert!(norm.forward(&x.to_dtype(DType::F64).unwrap()).is_err());
        assert!(norm.forward(&Tensor::zeros((1,3),DType::F32,&Device::Cpu).unwrap()).is_err());
    }
}
