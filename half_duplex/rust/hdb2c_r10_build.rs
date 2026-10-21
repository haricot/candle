use std::{env,path::PathBuf,process::Command};
fn main(){
 let root=PathBuf::from(env::var_os("HDB2C_MKL_ROOT").expect("HDB2C_MKL_ROOT required for isolated R10 CPU profile"));
 let out=PathBuf::from(env::var_os("OUT_DIR").unwrap());
 let obj=out.join("hdb2c_cpu.o");let lib=out.join("libhdb2c_cpu.a");
 assert!(Command::new("gcc").args(["-O2","-fPIC","-c","src/parakeet/hdb2c_r10_mkl.c","-o"]).arg(&obj).arg("-I").arg(root.join("include")).status().unwrap().success());
 assert!(Command::new("ar").arg("rcs").arg(&lib).arg(&obj).status().unwrap().success());
 println!("cargo:rustc-link-search=native={}",out.display());
 println!("cargo:rustc-link-lib=static=hdb2c_cpu");
 println!("cargo:rustc-link-arg={}/lib/libmkl_rt.so.2",root.display());
 println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib",root.display());
 println!("cargo:rerun-if-env-changed=HDB2C_MKL_ROOT");
 println!("cargo:rerun-if-changed=src/parakeet/hdb2c_r10_mkl.c");
}
