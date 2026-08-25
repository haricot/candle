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
