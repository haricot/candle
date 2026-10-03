use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;

#[derive(Clone, Copy)]
enum Kind {
    Ct1d { groups: usize },
    Ct2d { groups: usize },
    Conv1d,
}

#[derive(Clone, Copy)]
struct Case {
    implementation_id: &'static str,
    kind: Kind,
}

const CASES: &[Case] = &[
    Case {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256",
        kind: Kind::Ct1d { groups: 2 },
    },
    Case {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g4-u1-b256",
        kind: Kind::Ct1d { groups: 4 },
    },
    Case {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256",
        kind: Kind::Ct1d { groups: 8 },
    },
    Case {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g16-u1-b256",
        kind: Kind::Ct1d { groups: 16 },
    },
    Case {
        implementation_id: "candle.sm61-exact-grouped.ct2d-s32-g16-u4-b128",
        kind: Kind::Ct2d { groups: 16 },
    },
    Case {
        implementation_id: "candle.sm61-exact-grouped.ct2d-s32-g32-u4-b64",
        kind: Kind::Ct2d { groups: 32 },
    },
    Case {
        implementation_id: "candle.sm61-exact-grouped.gc1d-l128-g8-u1-b256",
        kind: Kind::Conv1d,
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
    let root = if let Some(home) = std::env::var_os("CANDLE_ASD_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(home)
    } else if let Some(xdg) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg).join("asd")
    } else {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| candle_core::Error::Msg("HOME is not set".into()))?;
        PathBuf::from(home).join(".local/share/asd")
    };
    Ok(root.join("artifacts/sm61"))
}

fn execute(case: Case, device: &Device) -> Result<Tensor> {
    match case.kind {
        Kind::Ct1d { groups } => {
            let x = Tensor::from_vec(
                deterministic(128 * 32, 37, -50),
                (1, 128, 32),
                device,
            )?;
            let c_out_per_group = 128 / groups;
            let w = Tensor::from_vec(
                deterministic(128 * c_out_per_group * 3, 53, -50),
                (128, c_out_per_group, 3),
                device,
            )?;
            x.conv_transpose1d(&w, 1, 1, 2, 1, groups)
        }
        Kind::Ct2d { groups } => {
            let x = Tensor::from_vec(
                deterministic(128 * 32 * 32, 37, -50),
                (1, 128, 32, 32),
                device,
            )?;
            let c_out_per_group = 128 / groups;
            let w = Tensor::from_vec(
                deterministic(128 * c_out_per_group * 3 * 3, 53, -50),
                (128, c_out_per_group, 3, 3),
                device,
            )?;
            x.conv_transpose2d_with_groups(&w, 1, 1, 2, 1, groups)
        }
        Kind::Conv1d => {
            let x = Tensor::from_vec(
                deterministic(64 * 128, 37, -50),
                (1, 64, 128),
                device,
            )?;
            let w = Tensor::from_vec(
                deterministic(64 * 8 * 3, 53, -50),
                (64, 8, 3),
                device,
            )?;
            x.conv1d(&w, 1, 1, 1, 8)
        }
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
                "missing external artifact for {} under {}",
                case.implementation_id,
                external_dir.display()
            )
        }
    }

    let builtin_only_dir = std::env::temp_dir().join(format!(
        "candle-asd-sm61-catalogue-builtin-{}",
        std::process::id()
    ));
    if builtin_only_dir.exists() {
        std::fs::remove_dir_all(&builtin_only_dir).map_err(|err| {
            candle_core::Error::Msg(format!(
                "failed to clear {}: {err}",
                builtin_only_dir.display()
            ))
        })?;
    }
    std::fs::create_dir_all(&builtin_only_dir).map_err(|err| {
        candle_core::Error::Msg(format!(
            "failed to create {}: {err}",
            builtin_only_dir.display()
        ))
    })?;

    std::env::remove_var("CANDLE_ASD_BUILTIN_RAW_DISABLE");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "0");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "0");

    let device = Device::new_cuda(0)?;

    println!("=== ASD V3 SM61 EXTERNAL CATALOGUE VALIDATION ===");
    println!("external_dir={}", external_dir.display());
    println!("cases={}", CASES.len());

    let mut failures = 0usize;
    for case in CASES {
        std::env::set_var("CANDLE_ASD_MODULE_DIR", &builtin_only_dir);
        device
            .as_cuda_device()?
            .refresh_asd_module(case.implementation_id)?;
        let builtin = execute(*case, &device)?;
        device.synchronize()?;
        require_source(&device, case.implementation_id, "builtin_ptx")?;

        std::env::set_var("CANDLE_ASD_MODULE_DIR", &external_dir);
        device
            .as_cuda_device()?
            .refresh_asd_module(case.implementation_id)?;
        let external = execute(*case, &device)?;
        device.synchronize()?;
        require_source(&device, case.implementation_id, "external_cubin")?;

        let (max_abs, max_rel) = max_abs_rel(&builtin, &external)?;
        let pass = max_abs <= 1e-5 || max_rel <= 1e-5;
        println!(
            "CASE implementation={} builtin_source=builtin_ptx external_source=external_cubin max_abs={:.8} max_rel={:.8} pass={}",
            case.implementation_id,
            max_abs,
            max_rel,
            pass
        );
        if !pass {
            failures += 1;
        }
    }

    let _ = std::fs::remove_dir_all(&builtin_only_dir);

    if failures != 0 {
        println!("STATUS=HOLD failures={failures} cases={}", CASES.len());
        candle_core::bail!("ASD sm61 external catalogue parity failed")
    }

    println!("BUILTIN_RAW=preserved");
    println!("STATUS=PASS failures=0 cases={}", CASES.len());
    Ok(())
}
