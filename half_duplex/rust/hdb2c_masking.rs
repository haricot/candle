// HDB2C3_R6_ENCODER_PADDING_MASKS_V1
// Appended to the pinned native attention module by the isolated R6 builder.
// NeMo 2.7.3 regular global attention: true means blocked (keys AND queries).
pub(crate) fn hdb2c6_masks(x: &Tensor, lengths: &Tensor) -> Result<(Tensor, Tensor)> {
    let (batch,time,_) = x.dims3()?;
    let lengths = lengths.to_device(&Device::Cpu)?.to_vec1::<i64>()?;
    if lengths.len()!=batch || time==0 || lengths.iter().any(|&n| n<=0 || n as usize>time) {
        return Err(candle::Error::Msg("R6 invalid encoder valid lengths".into()));
    }
    let mut pad = vec![0u8;batch*time];
    let mut att = vec![0u8;batch*time*time];
    for (b,&length) in lengths.iter().enumerate() {
        for q in 0..time {
            pad[b*time+q] = u8::from(q>=length as usize);
            for k in 0..time {
                att[(b*time+q)*time+k] = u8::from(q>=length as usize || k>=length as usize);
            }
        }
    }
    Ok((Tensor::from_vec(pad,(batch,time),x.device())?,
        Tensor::from_vec(att,(batch,1,time,time),x.device())?))
}

pub(crate) fn hdb2c6_zero_padding(x: &Tensor, pad: Option<&Tensor>) -> Result<Tensor> {
    match pad {
        None => Ok(x.clone()),
        Some(pad) => {
            let blocked = pad.unsqueeze(2)?.broadcast_as(x.shape())?;
            blocked.where_cond(&x.zeros_like()?,x)
        }
    }
}

pub(crate) fn hdb2c6_masked_softmax(scores: &Tensor, pad: Option<&Tensor>) -> Result<Tensor> {
    match pad {
        None => softmax_last_dim(scores),
        Some(pad) => {
            if scores.dtype()!=DType::F32 {
                return Err(candle::Error::Msg("R6 requires f32 scores".into()));
            }
            let blocked = pad.broadcast_as(scores.shape())?;
            // masked_fill, not additive -10000: blocked scores are replaced.
            let filled = blocked.where_cond(&Tensor::full(-10000f32,scores.shape(),scores.device())?,scores)?;
            let prob = softmax_last_dim(&filled)?;
            // NeMo zeros masked probabilities, including entirely masked rows.
            blocked.where_cond(&prob.zeros_like()?,&prob)
        }
    }
}

#[cfg(test)]
mod hdb2c6_tests {
    use super::*;
    #[test]
    fn masks_include_padded_queries_and_keys() -> Result<()> {
        let x=Tensor::zeros((2,4,2),DType::F32,&Device::Cpu)?;
        let lengths=Tensor::from_vec(vec![4i64,3],(2,),&Device::Cpu)?;
        let (pad,att)=hdb2c6_masks(&x,&lengths)?;
        assert_eq!(pad.flatten_all()?.to_vec1::<u8>()?,vec![0,0,0,0,0,0,0,1]);
        let bits=att.flatten_all()?.to_vec1::<u8>()?;
        assert_eq!(bits[..16], [0u8;16]);
        assert_eq!(bits[16..], [0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1]);
        let bad=Tensor::from_vec(vec![4i64,5],(2,),&Device::Cpu)?;
        assert!(hdb2c6_masks(&x,&bad).is_err());
        Ok(())
    }
    #[test]
    fn masked_softmax_replaces_huge_scores_and_zeros_entire_rows() -> Result<()> {
        let scores=Tensor::from_vec(vec![0f32,1_000_000.,3.,4.,0.,1_000_000.,3.,4.],(1,2,2,2),&Device::Cpu)?;
        let pad=Tensor::from_vec(vec![0u8,1,1,1],(1,1,2,2),&Device::Cpu)?;
        let prob=hdb2c6_masked_softmax(&scores,Some(&pad))?;
        assert_eq!(prob.flatten_all()?.to_vec1::<f32>()?,vec![1.,0.,0.,0.,1.,0.,0.,0.]);
        Ok(())
    }
    #[test]
    fn convolution_zeroing_is_after_glu_and_preserves_valid_values() -> Result<()> {
        let x=Tensor::from_vec(vec![1f32,-2.,3.,4.,99.,-99.],(1,3,2),&Device::Cpu)?;
        let pad=Tensor::from_vec(vec![0u8,0,1],(1,3),&Device::Cpu)?;
        assert_eq!(hdb2c6_zero_padding(&x,Some(&pad))?.flatten_all()?.to_vec1::<f32>()?,
                   vec![1.,-2.,3.,4.,0.,0.]);
        Ok(())
    }
}
