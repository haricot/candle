#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeBackend {
    SimtF16,
    Wmma,
}

pub fn select_moe_backend(compute_cap: usize, dtype: i32) -> Option<MoeBackend> {
    match dtype {
        0 if compute_cap >= 70 => Some(MoeBackend::Wmma),
        0 if compute_cap >= 53 => Some(MoeBackend::SimtF16),
        1 if compute_cap >= 80 => Some(MoeBackend::Wmma),
        _ => None,
    }
}

pub fn compiled_moe_backend(dtype: i32) -> Option<MoeBackend> {
    let compute_cap = option_env!("CANDLE_CUDA_COMPUTE_CAP")
        .and_then(|value| value.parse().ok())
        .unwrap_or(80);
    select_moe_backend(compute_cap, dtype)
}

#[cfg(test)]
mod tests {
    use super::{select_moe_backend, MoeBackend};

    #[test]
    fn fp16_dispatch_respects_architecture_floor() {
        assert_eq!(select_moe_backend(52, 0), None);
        assert_eq!(select_moe_backend(53, 0), Some(MoeBackend::SimtF16));
        assert_eq!(select_moe_backend(61, 0), Some(MoeBackend::SimtF16));
        assert_eq!(select_moe_backend(69, 0), Some(MoeBackend::SimtF16));
        assert_eq!(select_moe_backend(70, 0), Some(MoeBackend::Wmma));
        assert_eq!(select_moe_backend(80, 0), Some(MoeBackend::Wmma));
    }

    #[test]
    fn bf16_needs_ampere_and_unknown_dtype_is_rejected() {
        assert_eq!(select_moe_backend(61, 1), None);
        assert_eq!(select_moe_backend(70, 1), None);
        assert_eq!(select_moe_backend(79, 1), None);
        assert_eq!(select_moe_backend(80, 1), Some(MoeBackend::Wmma));
        assert_eq!(select_moe_backend(90, 2), None);
    }
}
