use candle_core::{
    diagnostics::{Capture, Payload},
    stateful::causal_depthwise_conv1d_step,
    DType, Device, Tensor,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let device = match args.next().as_deref() {
        Some("cpu") => Device::Cpu,
        Some("cuda") => Device::new_cuda(0)?,
        _ => return Err("usage: d7_shortconv_validate cpu|cuda OUTPUT_DIR".into()),
    };
    let out = PathBuf::from(args.next().ok_or("missing output directory")?);
    if out.exists() {
        return Err("output directory exists".into());
    }
    let mut checked = 0;
    for trace in [false, true] {
        let capture = if trace {
            Some(Capture::start(
                "shortconv-validation".into(),
                "standalone-validation".into(),
                100000,
                Payload::new().text("source", "synthetic_validation"),
                false,
            )?)
        } else {
            None
        };
        {
            let _root = candle_core::diagnostics::scope_in_window(
                "engine.forward",
                Some("validation"),
                Payload::new,
            );
            for dtype in [DType::F32, DType::F16] {
                for b in [1, 2] {
                    for c in [7, 2048] {
                        for k in [1, 3, 4] {
                            let state = Tensor::arange(0f32, (b * c * k) as f32, &device)?
                                .reshape((b, c, k))?
                                .affine(0.001, 0.)?
                                .to_dtype(dtype)?;
                            let original = state
                                .flatten_all()?
                                .to_dtype(DType::F32)?
                                .to_vec1::<f32>()?;
                            let x = Tensor::full(0.125f32, (b, c, 1), &device)?.to_dtype(dtype)?;
                            let weight =
                                Tensor::full(0.25f32, (c, 1, k), &device)?.to_dtype(dtype)?;
                            let (y, next) = causal_depthwise_conv1d_step(&x, &state, &weight)?;
                            let y = y.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            let next_values =
                                next.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            for row in 0..b * c {
                                let expected =
                                    original[row * k + 1..row * k + k].iter().sum::<f32>() * 0.25
                                        + 0.03125;
                                if (y[row] - expected).abs()
                                    > 1e-5_f32.max(
                                        expected.abs()
                                            * if dtype == DType::F16 { 0.001 } else { 0.000001 },
                                    )
                                {
                                    return Err(format!("output parity failure: {dtype:?} B={b} C={c} K={k} row={row}").into());
                                }
                                for j in 0..k {
                                    let expected = if j + 1 == k {
                                        0.125
                                    } else {
                                        original[row * k + j + 1]
                                    };
                                    assert_eq!(next_values[row * k + j], expected);
                                }
                            }
                            let _ = causal_depthwise_conv1d_step(&x, &next, &weight)?;
                            assert_eq!(
                                original,
                                state
                                    .flatten_all()?
                                    .to_dtype(DType::F32)?
                                    .to_vec1::<f32>()?
                            );
                            assert_eq!(
                                next_values,
                                next.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        if let Some(capture) = capture {
            capture.finish(&out.join("trace"))?;
        }
    }
    std::fs::create_dir_all(&out)?;
    let report = serde_json::json!({"status":"PASS_REFERENCE_OUTPUT_AND_IMMUTABLE_STATE","cases":checked,
        "device":format!("{:?}",device.location()),"candidate":"not_implemented","scope":"synthetic_operator_validation_not_engine_gain"});
    std::fs::write(
        out.join("validation.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
