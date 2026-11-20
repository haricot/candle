//! Compatibility API for existing F8E clients. New code uses fp8_linear.
pub use super::fp8_linear::*;
pub use super::fp8_linear::{Fp8Linear as Fp8Sm61Linear, Fp8Mode as Fp8Sm61Mode, select_mode as fp8_sm61_select_mode};
