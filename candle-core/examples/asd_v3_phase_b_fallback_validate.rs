use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Ct1d,
    Ct2d,
    Conv1d,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    decision_id: &'static str,
    kind: Kind,
    groups: usize,
}

impl Case {
    const fn weight_channel_per_group(self) -> usize {
        match self.kind {
            Kind::Ct1d | Kind::Ct2d => 128 / self.groups,
            Kind::Conv1d => 64 / self.groups,
        }
    }

    const fn fallback_implementation(self) -> &'static str {
        match self.kind {
            Kind::Ct1d | Kind::Ct2d => "candle.cudnn.grouped-transpose.v1",
            Kind::Conv1d => "candle.cudnn.grouped-conv1d.v1",
        }
    }
}

const CASES: &[Case] = &[
    Case {
        decision_id: "ct1d-sm61-s32-g2-raw-exact",
        kind: Kind::Ct1d,
        groups: 2,
    },
    Case {
        decision_id: "ct1d-sm61-s32-g4-raw-exact",
        kind: Kind::Ct1d,
        groups: 4,
    },
    Case {
        decision_id: "ct1d-sm61-s32-g8-raw-exact",
        kind: Kind::Ct1d,
        groups: 8,
    },
    Case {
        decision_id: "ct1d-sm61-s32-g16-raw-exact",
        kind: Kind::Ct1d,
        groups: 16,
    },
    Case {
        decision_id: "ct2d-sm61-s32-g16-raw-exact",
        kind: Kind::Ct2d,
        groups: 16,
    },
    Case {
        decision_id: "ct2d-sm61-s32-g32-raw-exact",
        kind: Kind::Ct2d,
        groups: 32,
    },
    Case {
        decision_id: "conv1d-sm61-l128-g8-raw-exact",
        kind: Kind::Conv1d,
        groups: 8,
    },
];

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn selected_cases() -> Result<Vec<Case>> {
    let args = std::env::args().collect::<Vec<_>>();
    let requested = args
        .windows(2)
        .find_map(|pair| (pair[0] == "--decision").then(|| pair[1].as_str()));
    match requested {
        None | Some("all") => Ok(CASES.to_vec()),
        Some(requested) => CASES
            .iter()
            .copied()
            .find(|case| case.decision_id == requested)
            .map(|case| vec![case])
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "invalid --decision {requested:?}; expected all or one of {}",
                    CASES
                        .iter()
                        .map(|case| case.decision_id)
                        .collect::<Vec<_>>()
                        .join(",")
                ))
            }),
    }
}

fn tensors(device: &Device, case: Case) -> Result<(Tensor, Tensor)> {
    match case.kind {
        Kind::Ct1d => {
            let x = Tensor::from_vec(deterministic(128 * 32, 37, -50), (1, 128, 32), device)?;
            let w = Tensor::from_vec(
                deterministic(128 * case.weight_channel_per_group() * 3, 53, -50),
                (128, case.weight_channel_per_group(), 3),
                device,
            )?;
            Ok((x, w))
        }
        Kind::Ct2d => {
            let x = Tensor::from_vec(
                deterministic(128 * 32 * 32, 37, -50),
                (1, 128, 32, 32),
                device,
            )?;
            let w = Tensor::from_vec(
                deterministic(128 * case.weight_channel_per_group() * 3 * 3, 53, -50),
                (128, case.weight_channel_per_group(), 3, 3),
                device,
            )?;
            Ok((x, w))
        }
        Kind::Conv1d => {
            let x = Tensor::from_vec(deterministic(64 * 128, 37, -50), (1, 64, 128), device)?;
            let w = Tensor::from_vec(
                deterministic(64 * case.weight_channel_per_group() * 3, 53, -50),
                (64, case.weight_channel_per_group(), 3),
                device,
            )?;
            Ok((x, w))
        }
    }
}

fn call(x: &Tensor, w: &Tensor, case: Case) -> Result<Tensor> {
    match case.kind {
        Kind::Ct1d => x.conv_transpose1d(w, 1, 1, 2, 1, case.groups),
        Kind::Ct2d => x.conv_transpose2d_with_groups(w, 1, 1, 2, 1, case.groups),
        Kind::Conv1d => x.conv1d(w, 1, 1, 1, case.groups),
    }
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
        "candle-asd-phase-b-fallback-matrix-{}",
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

fn clear_provider_controls() {
    std::env::remove_var("CANDLE_GROUPED_CONV1D_REQUIRE_CUDNN");
    std::env::remove_var("CANDLE_GROUPED_CONV1D_DISABLE_CUDNN");
    std::env::remove_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");
}

fn configure_reference(device: &Device, case: Case) -> Result<()> {
    clear_provider_controls();
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    match case.kind {
        Kind::Ct1d | Kind::Ct2d => {
            std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn");
        }
        Kind::Conv1d => {
            std::env::set_var("CANDLE_GROUPED_CONV1D_REQUIRE_CUDNN", "1");
        }
    }
    device.as_cuda_device()?.refresh_asd_runtime_policy();
    Ok(())
}

fn configure_qualified_fallback(device: &Device) -> Result<()> {
    clear_provider_controls();
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    std::env::set_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED", "1");
    device.as_cuda_device()?.refresh_asd();
    Ok(())
}

fn main() -> Result<()> {
    let cases = selected_cases()?;
    let module_dir = empty_module_dir()?;

    let profile_id = candle_kernels::asd_exact::PROFILE_ID.ok_or_else(|| {
        candle_core::Error::Msg(
            "Phase B qualified fallback validation requires an embedded ASD Exact Profile; set CANDLE_ASD_EXACT_POLICY before building/running".into(),
        )
    })?;
    let target_gpu_uuid = candle_kernels::asd_exact::TARGET_GPU_UUID.ok_or_else(|| {
        candle_core::Error::Msg(
            "Phase B qualified fallback validation requires a device-scoped ASD Exact Profile with target GPU UUID".into(),
        )
    })?;

    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_TRACE", "1");
    std::env::set_var("CANDLE_GROUPED_CONV1D_TRACE", "1");

    let device = Device::new_cuda(0)?;

    println!("=== ASD V3 PHASE B EXTERNAL-MISSING QUALIFIED FALLBACK MATRIX ===");
    println!("profile_id={profile_id}");
    println!("target_gpu_uuid={target_gpu_uuid}");
    println!("cases={}", cases.len());
    println!(
        "external_module_dir={} expected_empty=true",
        module_dir.display()
    );
    println!("builtin_raw_present=false");
    println!("qualified_fallback_required=true");

    let mut passed = 0usize;
    for case in cases {
        let (x, w) = tensors(&device, case)?;

        configure_reference(&device, case)?;
        let reference = call(&x, &w, case)?;
        device.synchronize()?;

        configure_qualified_fallback(&device)?;
        let fallback = call(&x, &w, case)?;
        device.synchronize()?;

        let (max_abs, max_rel) = max_abs_rel(&reference, &fallback)?;
        let parity = max_abs <= 1e-5 || max_rel <= 1e-5;
        println!(
            "CASE decision={} kind={:?} groups={} fallback_implementation={} max_abs={max_abs:.8} max_rel={max_rel:.8} status={}",
            case.decision_id,
            case.kind,
            case.groups,
            case.fallback_implementation(),
            if parity { "PASS" } else { "HOLD" },
        );

        if !parity {
            clear_provider_controls();
            std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
            std::env::remove_var("CANDLE_ASD_MODULE_DIR");
            let _ = std::fs::remove_dir_all(&module_dir);
            candle_core::bail!(
                "ASD Phase B qualified fallback parity failed for {}",
                case.decision_id
            )
        }
        passed += 1;
    }

    clear_provider_controls();
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    std::env::remove_var("CANDLE_ASD_MODULE_DIR");
    device.as_cuda_device()?.refresh_asd();
    let _ = std::fs::remove_dir_all(&module_dir);

    println!("MATRIX passed={passed} total={passed}");
    println!("STATUS=PASS");
    Ok(())
}
