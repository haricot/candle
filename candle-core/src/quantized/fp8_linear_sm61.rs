//! Compatibility API for existing F8E clients. New code uses fp8_linear.
pub use super::fp8_linear::*;
pub use super::fp8_linear::{Fp8Linear as Fp8Sm61Linear, Fp8Mode as Fp8Sm61Mode, select_mode as fp8_sm61_select_mode};

/// Historical helper retained for source compatibility. The generic executor
/// does not call it; its recipe is chosen by the active catalogue.
pub fn dp4a_launch_for_tokens(tokens: usize) -> (&'static str, usize) {
    super::fp8_recipe::S31_FP8.matrix_symbol(true, tokens)
}
