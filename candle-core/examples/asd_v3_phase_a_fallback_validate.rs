use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

#[derive(Clone, Copy)]
struct Case {
    decision_id: &'static str,
    groups: usize,
}

const CASES: &[Case] = &[
    Case {
        decision_id: "ct1d-sm61-s32-g2-raw-exact",
        groups: 2,
    },
    Case {
        decision_id: "ct1d-sm61-s32-g4-raw-exact",
        groups: 4,
    },
];

fn parse_case() -> Result<Case> {
    let args = std::env::args().collect::<Vec<_>>();
    let requested = args
        .windows(2)
        .find_map(|pair| (pair[0] == "--decision").then(|| pair[1].as_str()))
        .unwrap_or(CASES[0].decision_id);
    CASES
        .iter()
        .copied()
        .find(|case| case.decision_id == requested)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "invalid --decision {requested:?}; expected {}",
                CASES
                    .iter()
                    .map(|case| case.decision_id)
                    .collect::<Vec<_>>()
                    .join(" or ")
            ))
        })
}

fn tensors(device: &Device, case: Case) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(deterministic(128 * 32, 37, -50), (1, 128, 32), device)?;
    let c_out_per_group = 128 / case.groups;
    let w = Tensor::from_vec(
        deterministic(128 * c_out_per_group * 3, 53, -50),
        (128, c_out_per_group, 3),
        device,
    )?;
    Ok((x, w))
}

fn call(x: &Tensor, w: &Tensor, case: Case) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, case.groups)
}

fn max_abs_rel(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
    let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (&a, &b) in lhs.iter().zip(&rhs) {
        let abs = (a - b).abs();
        let rel = abs / b.abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    Ok((max_abs, max_rel))
}

fn empty_module_dir() -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "candle-asd-phase-a-fallback-{}",
        std::process::id()
    ));
    if path.exists() {
        std::fs::remove_dir_all(&path).map_err(|err| {
            candle_core::Error::Msg(format!("failed to clear {}: {err}", path.display()))
        })?;
    }
    std::fs::create_dir_all(&path).map_err(|err| {
        candle_core::Error::Msg(format!("failed to create {}: {err}", path.display()))
    })?;
    Ok(path)
}

fn main() -> Result<()> {
    let case = parse_case()?;
    let module_dir = empty_module_dir()?;

    let profile_id = candle_kernels::asd_exact::PROFILE_ID.ok_or_else(|| {
        candle_core::Error::Msg(
            "Phase A qualified fallback validation requires an embedded ASD Exact Profile; set CANDLE_ASD_EXACT_POLICY before building/running".into(),
        )
    })?;
    let target_gpu_uuid = candle_kernels::asd_exact::TARGET_GPU_UUID.ok_or_else(|| {
        candle_core::Error::Msg(
            "Phase A qualified fallback validation requires a device-scoped ASD Exact Profile with target GPU UUID".into(),
        )
    })?;

    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "1");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");

    let device = Device::new_cuda(0)?;
    let (x, w) = tensors(&device, case)?;

    println!("=== ASD V3 PHASE A QUALIFIED FALLBACK VALIDATION ===");
    println!("profile_id={profile_id}");
    println!("target_gpu_uuid={target_gpu_uuid}");
    println!("decision_id={}", case.decision_id);
    println!("ct1d_groups={}", case.groups);
    println!("primary_provider=raw_cuda");
    println!("fallback_provider=cudnn");
    println!("fallback_implementation=candle.cudnn.grouped-transpose.v1");
    println!(
        "external_module_dir={} expected_empty=true",
        module_dir.display()
    );

    // Reference: explicitly selected cuDNN.
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn");
    std::env::remove_var("CANDLE_ASD_BUILTIN_RAW_DISABLE");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    let cudnn_reference = call(&x, &w, case)?;
    device.synchronize()?;

    // Phase A fallback: exact raw remains promoted, but both external and builtin
    // raw are unavailable. Strict validation forbids falling through to an
    // unqualified generic raw/fallback path.
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    std::env::set_var("CANDLE_ASD_BUILTIN_RAW_DISABLE", "1");
    std::env::set_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED", "1");
    let qualified_fallback = call(&x, &w, case)?;
    device.synchronize()?;

    let (max_abs, max_rel) = max_abs_rel(&cudnn_reference, &qualified_fallback)?;
    let parity = max_abs <= 1e-5 || max_rel <= 1e-5;

    println!(
        "PARITY cudnn_reference_vs_qualified_fallback max_abs={max_abs:.8} max_rel={max_rel:.8} pass={parity}"
    );
    println!("primary_artifact_available=false");
    println!("qualified_fallback_required=true");
    println!("STATUS={}", if parity { "PASS" } else { "HOLD" });

    std::env::remove_var("CANDLE_ASD_BUILTIN_RAW_DISABLE");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    std::env::remove_var("CANDLE_ASD_MODULE_DIR");
    let _ = std::fs::remove_dir_all(&module_dir);

    if !parity {
        candle_core::bail!("ASD Phase A qualified fallback parity failed")
    }
    Ok(())
}
