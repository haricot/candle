//! Explicit, bounded, fail-closed Conv1d intermediate-value capture for MotionBricks.
//! No dispatch or numerical result is changed. This is POST-BIAS module output,
//! NOT an isolated CUDA-kernel oracle. The tensor input is never made contiguous
//! in the model: flatten_all is applied solely to a temporary readback view.
use candle_core::{DType, Error, Result, Tensor};
use serde_json::json;
use std::{fs, io::Write as _, path::PathBuf, sync::{Mutex, OnceLock}};

const PREFIX: &str = "HARICOT_CONV_INTERMEDIATE_V1";
const MAX_ELEMENTS: usize = 100_000;
static SEQUENCE: OnceLock<Mutex<u64>> = OnceLock::new();

fn fail(message: impl Into<String>) -> Error { Error::Msg(message.into()) }

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit(
    input: &Tensor, weight: &Tensor, output: &Tensor,
    padding: usize, stride: usize, dilation: usize, groups: usize,
) -> Result<()> {
    let Some(path) = std::env::var_os("HARICOT_CONV_PARITY_TRACE") else { return Ok(()) };
    let nonce = std::env::var("HARICOT_CONV_PARITY_NONCE")
        .map_err(|_| fail("parity nonce missing"))?;
    if nonce.len()!=32 || !nonce.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(fail("invalid parity nonce"));
    }
    let path=PathBuf::from(path);
    if !path.is_absolute() { return Err(fail("parity trace path must be absolute")); }
    let meta=fs::symlink_metadata(&path).map_err(|e|fail(format!("parity trace: {e}")))?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(fail("parity trace must be a regular, non-symlink file"));
    }
    if input.dtype()!=DType::F32 || weight.dtype()!=DType::F32 || output.dtype()!=DType::F32 {
        return Err(fail("parity trace is F32-only"));
    }
    let input_elems=input.elem_count();
    let output_elems=output.elem_count();
    if input_elems==0 || output_elems==0 || input_elems>MAX_ELEMENTS || output_elems>MAX_ELEMENTS {
        return Err(fail("parity trace tensor element limit exceeded"));
    }
    // Serialize the SEQUENCE and readbacks as one operation. This is essential:
    // a different ordering of CPU/CUDA events cannot be paired safely.
    let mut sequence=SEQUENCE.get_or_init(||Mutex::new(0)).lock()
        .map_err(|_|fail("parity sequence lock poisoned"))?;
    if *sequence>=128 { return Err(fail("parity trace event limit exceeded")); }
    let device=match input.device() {
        candle_core::Device::Cpu=>"cpu", candle_core::Device::Cuda(_)=>"cuda",
        candle_core::Device::Metal(_)=>return Err(fail("Metal is outside parity campaign")),
    };
    // These copies only read tensor values. They do not replace model inputs,
    // change the model's strides, mutate the kernel, or alter dispatch.
    let input_values=input.flatten_all()?.to_vec1::<f32>()?;
    let output_values=output.flatten_all()?.to_vec1::<f32>()?;
    if input_values.len()!=input_elems || output_values.len()!=output_elems {
        return Err(fail("parity readback element count drift"));
    }
    let signature=json!({
        "operation":"conv1d", "device":device, "dtype":"F32", "weight_dtype":"F32",
        "output_dtype":"F32", "groups":groups, "padding":padding, "stride":stride,
        "dilation":dilation, "output_padding":0,
        "input":input.dims(), "weight":weight.dims(), "output":output.dims(),
        "input_contiguous":input.is_contiguous(),
        "weight_contiguous":weight.is_contiguous(),
        "output_contiguous":output.is_contiguous(),
    });
    let event=json!({
        "schema":PREFIX, "kind":"CONV1D", "nonce":nonce, "seq":*sequence,
        "signature":signature,
        "input_strides":input.layout().stride(), "weight_strides":weight.layout().stride(),
        "output_strides":output.layout().stride(),
        "input_start_offset":input.layout().start_offset(),
        "weight_start_offset":weight.layout().start_offset(),
        "output_start_offset":output.layout().start_offset(),
        "input_bits":input_values.iter().map(|v|v.to_bits()).collect::<Vec<u32>>(),
        "output_bits":output_values.iter().map(|v|v.to_bits()).collect::<Vec<u32>>(),
    });
    let mut line=serde_json::to_vec(&event).map_err(|e|fail(format!("parity JSON: {e}")))?;
    line.push(b'\n');
    let mut file=fs::OpenOptions::new().append(true).open(&path)
        .map_err(|e|fail(format!("parity trace open: {e}")))?;
    file.write_all(&line).map_err(|e|fail(format!("parity trace write: {e}")))?;
    *sequence+=1;
    Ok(())
}
