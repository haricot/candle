//! R8 diagnostic I/O. No tensor supplied here replaces a production activation.
use candle::{DType, Device, Error, Result, Tensor};
use std::{fs, path::Path};

fn err(e: impl std::fmt::Display) -> Error { Error::Msg(e.to_string()) }
pub fn save(dir: &Path, name: &str, t: &Tensor) -> Result<()> {
    fs::create_dir_all(dir).map_err(err)?;
    let cpu = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let values = cpu.flatten_all()?.to_vec1::<f32>()?;
    save_values(dir, name, &values, cpu.dims())
}
pub fn save_values(dir: &Path, name: &str, values: &[f32], shape: &[usize]) -> Result<()> {
    if values.len() != shape.iter().product::<usize>() || !values.iter().all(|v|v.is_finite()) {
        return Err(err("R8 invalid tensor shape or nonfinite value"));
    }
    fs::create_dir_all(dir).map_err(err)?;
    let raw: Vec<u8> = values.iter().flat_map(|v|v.to_le_bytes()).collect();
    fs::write(dir.join(format!("{name}.f32le")), raw).map_err(err)?;
    fs::write(dir.join(format!("{name}.json")), serde_json::to_vec_pretty(
        &serde_json::json!({"shape":shape,"dtype":"f32le"})).map_err(err)?).map_err(err)?;
    Ok(())
}
pub fn read(dir: &Path, name: &str, device: &Device) -> Result<Tensor> {
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(dir.join(format!("{name}.json"))).map_err(err)?).map_err(err)?;
    let shape: Vec<usize> = meta["shape"].as_array().ok_or_else(||err("R8 missing shape"))?
        .iter().map(|v|v.as_u64().filter(|v|*v>0).map(|v|v as usize).ok_or_else(||err("R8 invalid dimension")))
        .collect::<Result<_>>()?;
    let raw = fs::read(dir.join(format!("{name}.f32le"))).map_err(err)?;
    if raw.len()!=4*shape.iter().product::<usize>() {return Err(err("R8 truncated fixture"));}
    let values:Vec<f32>=raw.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
    if !values.iter().all(|v|v.is_finite()) {return Err(err("R8 nonfinite fixture"));}
    Tensor::from_vec(values,shape.as_slice(),device)
}
