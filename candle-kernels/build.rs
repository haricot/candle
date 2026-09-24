mod asd_exact_build_adapter_v2;

use cudaforge::{detect_compute_cap, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=asd_exact_v2_dispatch_template.rs");
    println!("cargo::rerun-if-changed=asd_exact_build_adapter_v2.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-changed=src/grouped_transpose.cu");
    println!("cargo::rerun-if-changed=src/asd_dw5x5.cu");
    println!("cargo::rerun-if-env-changed=CARGO_FEATURE_CUDA_LEGACY_FP8");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed=CARGO_FEATURE_CUDA_LEGACY_BF16");
    println!("cargo::rerun-if-env-changed=CANDLE_SM61_EXACT_GROUPED_EVIDENCE_GPU_UUID");
    println!("cargo::rerun-if-env-changed=CANDLE_ASD_EXACT_POLICY");
    println!("cargo::rerun-if-env-changed=CANDLE_ASD_VALIDATION");
    println!("cargo::rerun-if-env-changed=CANDLE_ASD_TARGET_GPU_UUID");
    println!("cargo::rerun-if-env-changed=CANDLE_ASD_EXACT_DISABLE");
    println!("cargo::rerun-if-env-changed=CANDLE_ASD_EXACT_TRACE");

    let compute_cap = detect_compute_cap().map(|arch| arch.base()).unwrap_or(80);
    let legacy_bf16 = compute_cap < 80 && env::var_os("CARGO_FEATURE_CUDA_LEGACY_BF16").is_some();
    let legacy_fp8 = compute_cap < 89 && env::var_os("CARGO_FEATURE_CUDA_LEGACY_FP8").is_some();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sm61_exact_grouped_evidence_uuid =
        env::var("CANDLE_SM61_EXACT_GROUPED_EVIDENCE_GPU_UUID").ok();
    std::fs::write(
        out_dir.join("sm61_exact_grouped_scope.rs"),
        format!(
            "pub const TARGET_SCOPE: &str = \"device\";\npub const EVIDENCE_GPU_UUID: Option<&str> = {};\n",
            sm61_exact_grouped_evidence_uuid.as_deref().map(|v| format!("Some({v:?})")).unwrap_or_else(|| "None".to_owned())
        ),
    )
    .expect("failed to write SM61 exact-grouped scope");
    std::fs::write(
        out_dir.join("cuda_build_info.rs"),
        format!("pub const CUDA_BUILD_COMPUTE_CAP: u32 = {compute_cap};\n"),
    )
    .expect("failed to write CUDA build compute capability");
    let asd_build_sm =
        u32::try_from(compute_cap).expect("CUDA compute capability does not fit in u32");

    let validation_requested = matches!(
        env::var("CANDLE_ASD_VALIDATION").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    );
    if validation_requested && env::var_os("CANDLE_ASD_EXACT_POLICY").is_none() {
        panic!(
            "CANDLE_ASD_VALIDATION=1 requires CANDLE_ASD_EXACT_POLICY to be present at build time"
        );
    }

    // Stage 2D: the V2 reader is unconditional; its fallback represents no policy.
    asd_exact_build_adapter_v2::materialize_for_candle_build(asd_build_sm)
        .unwrap_or_else(|err| panic!("failed to materialize V2-only ASD policy: {err}"));

    let ptx_path = out_dir.join("ptx.rs");
    let mut ptx_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_files(vec![
            "src/affine.cu",
            "src/asd_dw5x5.cu",
            "src/binary.cu",
            "src/cast.cu",
            "src/conv.cu",
            "src/fill.cu",
            "src/grouped_transpose.cu",
            "src/indexing.cu",
            "src/quantized.cu",
            "src/reduce.cu",
            "src/sort.cu",
            "src/ternary.cu",
            "src/unary.cu",
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
        .arg("-O3");

    if legacy_bf16 {
        ptx_builder = ptx_builder.arg("-DCANDLE_CUDA_BF16_FALLBACK=1");
    }
    if legacy_fp8 {
        ptx_builder = ptx_builder.arg("-DCANDLE_CUDA_LEGACY_FP8=1");
    }

    let bindings = ptx_builder.build_ptx()?;
    bindings.write(&ptx_path)?;

    let mut moe_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_files(vec![
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
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if legacy_bf16 {
        moe_builder = moe_builder.arg("-DCANDLE_CUDA_BF16_FALLBACK=1");
    }

    if compute_cap < 70 {
        moe_builder = moe_builder
            .with_compute_override("moe_wmma.cu", 70)
            .with_compute_override("moe_wmma_gguf.cu", 70);
    }
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
