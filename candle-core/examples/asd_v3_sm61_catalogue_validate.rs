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
    implementation_id: &'static str,
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
}

const CASES: &[Case] = &[
    Case {
        decision_id: "ct1d-sm61-s32-g2-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256",
        kind: Kind::Ct1d,
        groups: 2,
    },
    Case {
        decision_id: "ct1d-sm61-s32-g4-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g4-u1-b256",
        kind: Kind::Ct1d,
        groups: 4,
    },
    Case {
        decision_id: "ct1d-sm61-s32-g8-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256",
        kind: Kind::Ct1d,
        groups: 8,
    },
    Case {
        decision_id: "ct1d-sm61-s32-g16-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g16-u1-b256",
        kind: Kind::Ct1d,
        groups: 16,
    },
    Case {
        decision_id: "ct2d-sm61-s32-g16-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct2d-s32-g16-u4-b128",
        kind: Kind::Ct2d,
        groups: 16,
    },
    Case {
        decision_id: "ct2d-sm61-s32-g32-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct2d-s32-g32-u4-b64",
        kind: Kind::Ct2d,
        groups: 32,
    },
    Case {
        decision_id: "conv1d-sm61-l128-g8-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.gc1d-l128-g8-u1-b256",
        kind: Kind::Conv1d,
        groups: 8,
    },
];

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn artifact_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("CANDLE_ASD_MODULE_DIR").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    candle_kernels::asd_paths::artifacts_dir("sm61").ok_or_else(|| {
        candle_core::Error::Msg("unable to resolve ASD sm61 artifact directory".into())
    })
}

fn tensors(device: &Device, case: Case) -> Result<(Tensor, Tensor)> {
    match case.kind {
        Kind::Ct1d => {
            let x = Tensor::from_vec(
                deterministic(128 * 32, 37, -50),
                (1, 128, 32),
                device,
            )?;
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
            let x = Tensor::from_vec(
                deterministic(64 * 128, 37, -50),
                (1, 64, 128),
                device,
            )?;
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

fn clear_provider_controls() {
    std::env::remove_var("CANDLE_GROUPED_CONV1D_REQUIRE_CUDNN");
    std::env::remove_var("CANDLE_GROUPED_CONV1D_DISABLE_CUDNN");
    std::env::remove_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
}

fn configure_reference(device: &Device, case: Case) -> Result<()> {
    clear_provider_controls();
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

fn configure_external_raw(device: &Device) {
    clear_provider_controls();
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    device.as_cuda_device().unwrap().refresh_asd();
}

fn require_source(device: &Device, implementation_id: &str, expected: &str) -> Result<()> {
    let source = device
        .as_cuda_device()?
        .asd_resolved_module_source(implementation_id)?;
    if source != Some(expected) {
        candle_core::bail!(
            "unexpected ASD source for {}: expected={} actual={:?}",
            implementation_id,
            expected,
            source
        )
    }
    Ok(())
}

fn main() -> Result<()> {
    let external_dir = artifact_dir()?;
    for case in CASES {
        let cubin = external_dir.join(format!("{}.cubin", case.implementation_id));
        let manifest = external_dir.join(format!("{}.manifest", case.implementation_id));
        if !cubin.is_file() || !manifest.is_file() {
            candle_core::bail!(
                "missing external Phase B artifact for {} under {}",
                case.implementation_id,
                external_dir.display()
            )
        }
    }

    std::env::set_var("CANDLE_ASD_MODULE_DIR", &external_dir);
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "1");

    let device = Device::new_cuda(0)?;

    println!("=== ASD V3 PHASE B EXTERNAL CATALOGUE VALIDATION ===");
    println!("external_dir={}", external_dir.display());
    println!("expected_source=external_cubin");
    println!("builtin_raw_present=false");
    println!("cases={}", CASES.len());

    let mut passed = 0usize;
    for case in CASES {
        let (x, w) = tensors(&device, *case)?;

        configure_reference(&device, *case)?;
        let reference = call(&x, &w, *case)?;
        device.synchronize()?;

        configure_external_raw(&device);
        let external = call(&x, &w, *case)?;
        device.synchronize()?;
        require_source(&device, case.implementation_id, "external_cubin")?;

        let (max_abs, max_rel) = max_abs_rel(&reference, &external)?;
        let parity = max_abs <= 1e-5 || max_rel <= 1e-5;
        println!(
            "CASE decision={} implementation={} kind={:?} groups={} source=external_cubin max_abs={max_abs:.8} max_rel={max_rel:.8} status={}",
            case.decision_id,
            case.implementation_id,
            case.kind,
            case.groups,
            if parity { "PASS" } else { "HOLD" },
        );
        if !parity {
            clear_provider_controls();
            std::env::remove_var("CANDLE_ASD_MODULE_DIR");
            candle_core::bail!(
                "ASD Phase B external CUBIN parity failed for {}",
                case.decision_id
            )
        }
        passed += 1;
    }

    clear_provider_controls();
    std::env::remove_var("CANDLE_ASD_MODULE_DIR");
    device.as_cuda_device()?.refresh_asd();

    println!("MATRIX passed={passed} total={passed}");
    println!("STATUS=PASS");
    Ok(())
}
