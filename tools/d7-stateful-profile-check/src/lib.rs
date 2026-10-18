#![allow(dead_code)]
extern crate self as candle_kernels;

#[derive(Debug)]
pub enum Error {
    Msg(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Msg(message) => formatter.write_str(message),
        }
    }
}
pub type Result<T> = std::result::Result<T, Error>;
pub const CUDA_BUILD_COMPUTE_CAP: u32 = 61;

#[path = "../../../candle-kernels/src/asd_exact.rs"]
pub mod asd_exact;
#[path = "../../../candle-kernels/src/asd_paths.rs"]
pub mod asd_paths;
#[path = "../../../candle-core/src/asd_runtime_extensions.rs"]
mod profile;
