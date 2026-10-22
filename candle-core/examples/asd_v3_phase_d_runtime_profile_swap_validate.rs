use candle_core::{Device, Result, Tensor};
use std::path::{Path, PathBuf};

const DECISION_ID: &str = "ct1d-sm61-s32-g2-raw-exact";
const RAW_IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256";
const CUDNN_IMPLEMENTATION_ID: &str = "candle.cudnn.grouped-transpose.v1";

struct TempTree(PathBuf);

impl Drop for TempTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
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

fn current_profile_path() -> Result<PathBuf> {
    candle_kernels::asd_paths::current_profile_path()
        .filter(|path| path.is_file())
        .ok_or_else(|| candle_core::Error::Msg("missing ASD current profile".into()))
}

fn required_module_dir() -> Result<PathBuf> {
    std::env::var_os("CANDLE_ASD_MODULE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            candle_core::Error::Msg(
                "D2 runtime profile swap validation requires CANDLE_ASD_MODULE_DIR".into(),
            )
        })
}

fn mutate_provider(source: &str) -> Result<String> {
    let mut found = false;
    let mut output = String::new();

    for line in source.lines() {
        if line.starts_with(&format!("decision|{DECISION_ID}|")) {
            let mut fields = line.split('|').map(str::to_owned).collect::<Vec<_>>();
            if fields.len() != 8 {
                candle_core::bail!("unexpected decision row shape for {DECISION_ID}")
            }
            if fields[2] != "promoted"
                || fields[3] != "raw_cuda"
                || fields[5] != format!("impl={RAW_IMPLEMENTATION_ID}")
            {
                candle_core::bail!(
                    "unexpected current profile row for {DECISION_ID}: {line}"
                )
            }
            fields[3] = "cudnn".into();
            fields[5] = format!("impl={CUDNN_IMPLEMENTATION_ID}");
            output.push_str(&fields.join("|"));
            output.push('\n');
            found = true;
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }

    if !found {
        candle_core::bail!("current profile is missing {DECISION_ID}")
    }
    Ok(output)
}

fn call(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, 2)
}

fn set_auto(device: &Device) -> Result<()> {
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    device.as_cuda_device()?.refresh_asd();
    Ok(())
}

fn main() -> Result<()> {
    let canonical_profile = current_profile_path()?;
    let raw_source = std::fs::read_to_string(&canonical_profile)?;
    let cudnn_source = mutate_provider(&raw_source)?;
    let raw_module_dir = required_module_dir()?;

    let root = std::env::temp_dir().join(format!(
        "candle-asd-d2-profile-swap-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root)?;
    let _cleanup = TempTree(root.clone());

    let runtime_profile = root.join("current.asd");
    let empty_modules = root.join("empty-modules");
    let isolated_home = root.join("isolated-home");
    std::fs::create_dir_all(&empty_modules)?;
    std::fs::create_dir_all(&isolated_home)?;
    std::fs::write(&runtime_profile, &raw_source)?;

    // Explicit runtime profile means the isolated ASD home can contain no
    // fallback store. This makes the B phase fail closed if refresh keeps a
    // stale raw decision after the external module is removed.
    std::env::set_var("CANDLE_ASD_RUNTIME_PROFILE", &runtime_profile);
    std::env::set_var("CANDLE_ASD_HOME", &isolated_home);
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_TRACE", "1");

    let device = Device::new_cuda(0)?;
    let x = Tensor::from_vec(
        deterministic(128 * 32, 37, -50),
        (1, 128, 32),
        &device,
    )?;
    let w = Tensor::from_vec(
        deterministic(128 * 64 * 3, 53, -50),
        (128, 64, 3),
        &device,
    )?;

    // Independent cuDNN reference.
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    device.as_cuda_device()?.refresh_asd();
    let reference = call(&x, &w)?;
    device.synchronize()?;

    // A: authoritative runtime profile selects raw external CUBIN.
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &raw_module_dir);
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    set_auto(&device)?;
    let raw_a = call(&x, &w)?;
    device.synchronize()?;
    let source_a = device
        .as_cuda_device()?
        .asd_resolved_module_source(RAW_IMPLEMENTATION_ID)?
        .unwrap_or("none");
    if source_a != "external_cubin" {
        candle_core::bail!("phase A source is {source_a}, expected external_cubin")
    }

    // B: same profile path, same process, provider becomes cuDNN. Remove all
    // executable raw artifacts and require an authenticated exact decision.
    // With the isolated ASD home there is no qualified fallback store, so a
    // stale raw decision cannot silently rescue this phase.
    std::fs::write(&runtime_profile, &cudnn_source)?;
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &empty_modules);
    std::env::set_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED", "1");
    set_auto(&device)?;
    let cudnn_b = call(&x, &w)?;
    device.synchronize()?;
    let source_b = device
        .as_cuda_device()?
        .asd_resolved_module_source(RAW_IMPLEMENTATION_ID)?;
    if source_b.is_some() {
        candle_core::bail!(
            "phase B unexpectedly resolved raw module source {:?}",
            source_b
        )
    }

    // A2: restore the original profile bytes and external raw artifact.
    std::fs::write(&runtime_profile, &raw_source)?;
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &raw_module_dir);
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    set_auto(&device)?;
    let raw_a2 = call(&x, &w)?;
    device.synchronize()?;
    let source_a2 = device
        .as_cuda_device()?
        .asd_resolved_module_source(RAW_IMPLEMENTATION_ID)?
        .unwrap_or("none");
    if source_a2 != "external_cubin" {
        candle_core::bail!("phase A2 source is {source_a2}, expected external_cubin")
    }

    let (ref_a_abs, ref_a_rel) = max_abs_rel(&reference, &raw_a)?;
    let (ref_b_abs, ref_b_rel) = max_abs_rel(&reference, &cudnn_b)?;
    let (a_a2_abs, a_a2_rel) = max_abs_rel(&raw_a, &raw_a2)?;
    let parity_a = ref_a_abs <= 1e-5 || ref_a_rel <= 1e-5;
    let parity_b = ref_b_abs <= 1e-5 || ref_b_rel <= 1e-5;
    let parity_a2 = a_a2_abs <= 1e-5 || a_a2_rel <= 1e-5;

    println!("=== ASD V3 PHASE D2 RUNTIME PROFILE AUTHORITY SWAP ===");
    println!("canonical_profile={}", canonical_profile.display());
    println!("runtime_profile={}", runtime_profile.display());
    println!("decision_id={DECISION_ID}");
    println!("phase_a_provider=raw_cuda source={source_a}");
    println!("phase_b_provider=cudnn raw_module_source=none");
    println!("phase_a2_provider=raw_cuda source={source_a2}");
    println!(
        "PARITY reference_vs_a max_abs={ref_a_abs:.8} max_rel={ref_a_rel:.8} pass={parity_a}"
    );
    println!(
        "PARITY reference_vs_b max_abs={ref_b_abs:.8} max_rel={ref_b_rel:.8} pass={parity_b}"
    );
    println!(
        "PARITY a_vs_a2 max_abs={a_a2_abs:.8} max_rel={a_a2_rel:.8} pass={parity_a2}"
    );
    println!("same_process=true");
    println!("same_runtime_profile_path=true");
    println!("candle_rebuild=false");
    println!("fallback_store_isolated=true");

    if !(parity_a && parity_b && parity_a2) {
        println!("STATUS=HOLD");
        candle_core::bail!("D2 runtime profile authority swap parity failed")
    }

    println!("D2_RUNTIME_PROFILE_SWAP=PASS");
    println!("STATUS=PASS");
    Ok(())
}
