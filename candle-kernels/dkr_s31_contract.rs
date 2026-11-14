//! S3.1 native Candle module package contract, shared with the standalone builder.
pub const SCHEMA: &str = "CANDLE-DKR-S31-NATIVE-MODULE-V1";
pub const BASE: &str = "d99f6d77e507688bd9e4b654fbe20ac9e18238fc";
pub const CC: u32 = 61;
pub const FLAGS: &[&str] = &[
    "--expt-relaxed-constexpr", "-std=c++17", "-O3",
    "-DCANDLE_CUDA_BF16_FALLBACK=1",
    "-DCANDLE_CUDA_LEGACY_FP8=1",
    "-DCANDLE_CUDA_LEGACY_FP4=1",
];
pub const FAMILIES: &[&str] = &["affine", "binary", "cast", "fill", "ternary", "unary"];

pub fn is_sm61_ptx(ptx: &str) -> bool {
    let mut targets = ptx.lines().filter_map(|line| {
        let mut words = line.split_whitespace();
        if words.next() == Some(".target") {
            words.next().map(|word| word.trim_end_matches(','))
        } else {
            None
        }
    });
    targets.next() == Some("sm_61") && targets.next().is_none()
        && ptx.lines().any(|line| line.split_whitespace().any(|word| word == ".entry"))
}
