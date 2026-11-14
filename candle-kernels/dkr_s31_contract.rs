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
pub const FAMILIES: &[&str] = &["affine", "binary", "cast", "fill", "indexing", "reduce", "ternary", "unary", "quantized", "mmvq_gguf"];

pub const MMVQ_CODECS: &[&str] = &["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"];
pub const MMVQ_DTYPES: &[&str] = &["bf16", "f16", "f32"];

pub fn mmvq_entries() -> Vec<String> {
    let mut entries = Vec::with_capacity(243);
    for dtype in MMVQ_DTYPES {
        entries.push(format!("mmvq_gguf_quantize_q8_1_{dtype}"));
        for codec in MMVQ_CODECS {
            for batch in 1..=8 {
                entries.push(format!("mmvq_gguf_{codec}_{dtype}_plain_cuda{batch}"));
            }
        }
    }
    entries
}

pub fn has_mmvq_entries(ptx: &str) -> bool {
    let entries: std::collections::HashSet<_> = ptx.lines().filter_map(|line| {
        let mut words = line.split_whitespace();
        if !words.any(|word| word == ".entry") { return None; }
        Some(words.next()?.split('(').next()?)
    }).collect();
    mmvq_entries().iter().all(|entry| entries.contains(entry.as_str()))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mmvq_entry_inventory() {
        let entries = mmvq_entries();
        assert_eq!(entries.len(), 243);
        assert_eq!(entries.iter().collect::<std::collections::HashSet<_>>().len(), 243);
        let ptx = entries.iter().map(|entry| format!(".visible .entry\t{entry}() {{ ret; }}\n")).collect::<String>();
        assert!(has_mmvq_entries(&ptx));
        assert!(!has_mmvq_entries(&ptx.replace(&entries[0], "wrong_symbol")));
    }
}
