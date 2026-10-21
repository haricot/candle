use std::{env,path::PathBuf,process::Command};
fn main() {
    assert!(env::var_os("CARGO_FEATURE_CUDA").is_none(),"R13 is CPU-only");
    let root=PathBuf::from(env::var_os("HDB2C_MKL_ROOT").expect("standalone MKL root required"));
    let dnnl=PathBuf::from(env::var_os("HDB2C_ONEDNN_BUILD").expect("CPU oneDNN build required"));
    let include=PathBuf::from(env::var_os("HDB2C_ONEDNN_INCLUDE").expect("oneDNN source include required"));
    let sleef=PathBuf::from(env::var_os("HDB2C_SLEEF_BUILD").expect("standalone SLEEF build required"));
    let out=PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mkl=out.join("mkl.o");let conv=out.join("conv.o");let norm=out.join("norm.o");
    assert!(Command::new("gcc").args(["-O2","-fPIC","-c","src/parakeet/hdb2c_r10_mkl.c","-o"])
        .arg(&mkl).arg("-I").arg(root.join("include")).status().unwrap().success());
    assert!(Command::new("g++").args(["-O2","-fPIC","-fopenmp","-std=c++17","-c",
        "src/parakeet/hdb2c_r11_onednn.cpp","-o"]).arg(&conv)
        .arg("-I").arg(include).arg("-I").arg(dnnl.join("include")).status().unwrap().success());
    assert!(Command::new("g++").args(["-O2","-fPIC","-mavx2","-mfma","-fno-math-errno",
        "-ffp-contract=fast","-c","src/parakeet/hdb2c_r13_norm.cpp","-o"]).arg(&norm).status().unwrap().success());
    let mut objects=vec![mkl,conv,norm];
    for lanes in [8,16] {
        let object=out.join(format!("vector{lanes}.o"));
        let mut command=Command::new("g++");
        command.args(["-O2","-fPIC","-ffp-contract=off","-mfma","-c","src/parakeet/hdb2c_r13_vector.cpp"])
            .arg(format!("-DHDB2C13_LANES={lanes}"));
        if lanes==16 {command.args(["-mavx512f","-mavx512dq","-mavx512bw","-mavx512vl"]);}
        else {command.arg("-mavx2");}
        assert!(command.arg("-I").arg(root.join("include")).arg("-I").arg(sleef.join("include"))
            .arg("-o").arg(&object).status().unwrap().success());
        objects.push(object);
    }
    assert!(Command::new("ar").arg("rcs").arg(out.join("libhdb2c_cpu.a"))
        .args(objects).status().unwrap().success());
    println!("cargo:rustc-link-search=native={}",out.display());
    println!("cargo:rustc-link-lib=static=hdb2c_cpu");
    println!("cargo:rustc-link-search=native={}/src",dnnl.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/src",dnnl.display());
    println!("cargo:rustc-link-lib=dylib=dnnl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-search=native={}/lib",sleef.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib",sleef.display());
    println!("cargo:rustc-link-lib=dylib=sleef");
    println!("cargo:rustc-link-arg={}/lib/libmkl_rt.so.2",root.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib",root.display());
    for name in ["HDB2C_MKL_ROOT","HDB2C_ONEDNN_BUILD","HDB2C_ONEDNN_INCLUDE","HDB2C_SLEEF_BUILD"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for file in ["hdb2c_r10_mkl.c","hdb2c_r11_onednn.cpp","hdb2c_r13_norm.cpp","hdb2c_r13_vector.cpp"] {
        println!("cargo:rerun-if-changed=src/parakeet/{file}");
    }
}
