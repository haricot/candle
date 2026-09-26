mod asd_exact_build_adapter_v2;

use cudaforge::{detect_compute_cap, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

const CUTILE_FEATURE: &str = "CARGO_FEATURE_CUTILE";
const LEGACY_BF16_FEATURE: &str = "CARGO_FEATURE_CUDA_LEGACY_BF16";
const LEGACY_FP8_FEATURE: &str = "CARGO_FEATURE_CUDA_LEGACY_FP8";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=asd_exact_build_adapter_v2.rs");
    println!("cargo::rerun-if-changed=asd_exact_v2_dispatch_template.rs");
    println!("cargo::rerun-if-changed=src/sm61_exact_grouped");
    println!("cargo::rerun-if-env-changed=CANDLE_SM61_EXACT_GROUPED_EVIDENCE_GPU_UUID");
    for var in [
        "CUDA_COMPUTE_CAP",
        "CANDLE_ASD_EXACT_POLICY",
        "CANDLE_ASD_VALIDATION",
        "CANDLE_ASD_TARGET_GPU_UUID",
    ] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed={LEGACY_BF16_FEATURE}");
    println!("cargo::rerun-if-env-changed={LEGACY_FP8_FEATURE}");

    let compute_cap = detect_compute_cap().map(|arch| arch.base()).unwrap_or(80);
    println!("cargo:rustc-env=CANDLE_CUDA_COMPUTE_CAP={compute_cap}");
    let legacy_bf16 = compute_cap < 80 && env::var_os(LEGACY_BF16_FEATURE).is_some();
    let legacy_fp8 = compute_cap < 89 && env::var_os(LEGACY_FP8_FEATURE).is_some();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut ptx_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if legacy_bf16 {
        ptx_builder = ptx_builder.arg("-DCANDLE_CUDA_BF16_FALLBACK=1");
    }
    if legacy_fp8 {
        ptx_builder = ptx_builder.arg("-DCANDLE_CUDA_LEGACY_FP8=1");
    }

    let bindings = ptx_builder.build_ptx()?;
    bindings.write(&ptx_path)?;
    // Exact SM61 kernels are kept in a separately generated PTX module.
    // This avoids pulling legacy BF16/FP8 compilation flags into asd_core.
    let sm61_ptx = KernelBuilder::new()
        .source_files(vec![
            "src/sm61_exact_grouped/sm61_exact_grouped_k00.cu",
            "src/sm61_exact_grouped/sm61_exact_grouped_k01.cu",
            "src/sm61_exact_grouped/sm61_exact_grouped_k02.cu",
            "src/sm61_exact_grouped/sm61_exact_grouped_k03.cu",
            "src/sm61_exact_grouped/sm61_exact_grouped_k04.cu",
            "src/sm61_exact_grouped/sm61_exact_grouped_k05.cu",
            "src/sm61_exact_grouped/sm61_exact_grouped_k06.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;
    sm61_ptx.write(&out_dir.join("sm61_ptx.rs"))?;


    let mut moe_sources = vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_simt_f16.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
        "src/mmvq_gguf.cu",
        "src/mmq_gguf/mmq_quantize.cu",
        "src/mmq_gguf/mmq_instance_q4_0.cu",
        "src/mmq_gguf/mmq_instance_q4_1.cu",
        "src/mmq_gguf/mmq_instance_q5_0.cu",
        "src/mmq_gguf/mmq_instance_q5_1.cu",
        "src/mmq_gguf/mmq_instance_q8_0.cu",
        "src/mmq_gguf/mmq_instance_q2_k.cu",
        "src/mmq_gguf/mmq_instance_q3_k.cu",
        "src/mmq_gguf/mmq_instance_q4_k.cu",
        "src/mmq_gguf/mmq_instance_q5_k.cu",
        "src/mmq_gguf/mmq_instance_q6_k.cu",
    ];
    if env::var_os(CUTILE_FEATURE).is_some() {
        moe_sources.push("src/moe/moe_align.cu");
    }

    // Preserve the corrected shared baseline: Pascal cannot execute WMMA.
    // SIMT grouped MoE is introduced by the separately pinned MoE source.
    if compute_cap < 70 {
        moe_sources.retain(|source| {
            !matches!(*source, "src/moe/moe_wmma.cu" | "src/moe/moe_wmma_gguf.cu")
        });
    }

    let mut moe_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_files(moe_sources)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if legacy_bf16 {
        moe_builder = moe_builder.arg("-DCANDLE_CUDA_BF16_FALLBACK=1");
    }

    // On pre-Volta SM61 there are no WMMA translation units; provide the
    // fallback FFI stub from the SIMT MoE unit without touching BF16 flags.
    if compute_cap < 70 {
        moe_builder = moe_builder.arg("-DNO_WMMA_KERNEL");
    }

    // BF16 WMMA fragments require Ampere.
    let evidence_uuid = env::var("CANDLE_SM61_EXACT_GROUPED_EVIDENCE_GPU_UUID").ok();
    std::fs::write(
        out_dir.join("sm61_exact_grouped_scope.rs"),
        format!(
            "pub const TARGET_SCOPE: &str = \"device\";\npub const EVIDENCE_GPU_UUID: Option<&str> = {};\n",
            evidence_uuid
                .as_deref()
                .map(|v| format!("Some({v:?})"))
                .unwrap_or_else(|| "None".to_owned())
        ),
    )?;
    std::fs::write(
        out_dir.join("cuda_build_info.rs"),
        format!("pub const CUDA_BUILD_COMPUTE_CAP: u32 = {compute_cap};\n"),
    )?;
    let validation_requested = matches!(
        env::var("CANDLE_ASD_VALIDATION").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    );
    if validation_requested && env::var_os("CANDLE_ASD_EXACT_POLICY").is_none() {
        panic!("CANDLE_ASD_VALIDATION requires CANDLE_ASD_EXACT_POLICY");
    }
    asd_exact_build_adapter_v2::materialize_for_candle_build(compute_cap)
        .unwrap_or_else(|err| panic!("failed to materialize ASD V2 policy: {err}"));
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
