use cudaforge::{detect_compute_cap, KernelBuilder, Result};
use ir_caps::register_capabilities;
use ir_caps::KernelBuilderExt;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    // println!("cargo::rerun-if-changed=src/compatibility.cuh");
    // println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    // println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = detect_compute_cap().unwrap_or_else(|_| cudaforge::GpuArch::new(0));
    let compute_cap = env::var("CUDA_COMPUTE_CAP")
        .map(|s| s.parse::<usize>().unwrap_or(80))
        .unwrap_or_else(|_| arch.base());

    // Initialize builders early
    let mut builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .watch([
            "src/compatibility.cuh",
            "src/cuda_utils.cuh",
            "src/binary_op_macros.cuh",
        ])
        .exclude(&["moe_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    builder = register_capabilities(builder, arch.base).emit_defines();

    let mut moe_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/moe/moe_hfma2.cu",
        ]);

    let target = env::var("TARGET").unwrap_or_default();
    let is_target_msvc = target.contains("msvc");
    if is_target_msvc {
        moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
    } else {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder = register_capabilities(moe_builder, arch.base);

    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    // Build for PTX
    let ptx_output = builder.build_ptx()?;
    ptx_output.write(out_dir.join("ptx.rs"))?;

    // Build for MOE Static Library
    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
