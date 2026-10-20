// HDB2C3_R7_TDT_ADVANCE_AND_TRACE_V1. Included in parakeet::model.
// NeMo 2.7.3 batched TDT: blank with duration zero advances one frame.
fn hdb2c7_advance(frame: usize, token: usize, blank: usize, duration: usize,
    symbols: usize, maximum: Option<usize>) -> (usize, usize, bool) {
    let count = symbols + usize::from(token != blank);
    let effective = if token == blank {duration.max(1)} else {duration};
    let forced = effective == 0 && maximum.is_some_and(|m|count >= m);
    (frame + effective + usize::from(forced),
     if effective > 0 || forced {0} else {count}, forced)
}

fn hdb2c7_tensor(index: usize, name: &str, tensor: &Tensor) -> Result<()> {
    use std::io::Write;
    let Some(root) = std::env::var_os("HDB2C7_TDT_TRACE_DIR") else {return Ok(());};
    let dir = std::path::PathBuf::from(root).join(format!("{index:06}"));
    std::fs::create_dir_all(&dir).map_err(|e|candle::Error::Msg(e.to_string()))?;
    let t = tensor.to_device(&Device::Cpu)?.to_dtype(candle::DType::F32)?;
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    let mut file = std::fs::File::create(dir.join(format!("{name}.f32le")))
        .map_err(|e|candle::Error::Msg(e.to_string()))?;
    for v in data {file.write_all(&v.to_le_bytes()).map_err(|e|candle::Error::Msg(e.to_string()))?;}
    std::fs::write(dir.join(format!("{name}.json")),
        serde_json::json!({"shape":t.dims(),"dtype":"f32le"}).to_string())
        .map_err(|e|candle::Error::Msg(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod hdb2c7_tests {
    use super::*;
    #[test] fn blank_zero_advances_without_committing_a_symbol() {
        assert_eq!(hdb2c7_advance(4,10,10,0,2,Some(10)),(5,0,false));
        assert_eq!(hdb2c7_advance(4,10,10,0,0,None),(5,0,false));
    }
    #[test] fn zero_duration_labels_obey_the_same_frame_limit() {
        assert_eq!(hdb2c7_advance(4,2,10,0,8,Some(10)),(4,9,false));
        assert_eq!(hdb2c7_advance(4,2,10,0,9,Some(10)),(5,0,true));
    }
    #[test] fn positive_duration_does_not_add_an_extra_limit_frame() {
        assert_eq!(hdb2c7_advance(4,2,10,2,9,Some(10)),(6,0,false));
        assert_eq!(hdb2c7_advance(4,10,10,2,9,Some(10)),(6,0,false));
    }
}
