//static MOE_KERNEL_WMMA: &str = "wmma";
static MOE_KERNEL_HFMA2: &str = "hfma2";

#[cfg(has_wmma)]
fn select_wmma_for_dtype(dtype: i32) -> bool {
    if dtype == DTYPE_BF16 {
        cfg!(has_wmma_bf16)
    } else {
        cfg!(has_wmma_f16) || cfg!(has_wmma)
    }
}

#[cfg(not(has_wmma))]
pub fn select_moe_kernel(_m: usize, _n: usize, _k: usize, _dtype: i32) -> &'static str {
    MOE_KERNEL_HFMA2
}

#[cfg(has_wmma)]
pub fn select_moe_kernel(_m: usize, _n: usize, _k: usize, dtype: i32) -> &'static str {
    if select_wmma_for_dtype(dtype) {
        return MOE_KERNEL_WMMA;
    }

    MOE_KERNEL_HFMA2
}

#[cfg(any(has_bf16, allow_legacy_bf16))]
pub const HAS_BF16_SUPPORT: bool = true;
#[cfg(not(any(has_bf16, allow_legacy_bf16)))]
pub const HAS_BF16_SUPPORT: bool = false;

ir_caps::export_capabilities!();
