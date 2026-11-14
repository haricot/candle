//! S3.1 native Candle module package contract, shared with the standalone builder.
pub const SCHEMA: &str = "CANDLE-DKR-S31-NATIVE-MODULE-V1";
pub const BASE: &str = "d99f6d77e507688bd9e4b654fbe20ac9e18238fc";
pub const CC: u32 = 61;
// The GGUF MoE source differs from BASE only by reviewed warp-tail synchronization.
pub const MOE_GGUF_ABI: &str = "moe-gguf-warp-tail-v1";
pub const FLAGS: &[&str] = &[
    "--expt-relaxed-constexpr", "-std=c++17", "-O3",
    "-DCANDLE_CUDA_BF16_FALLBACK=1",
    "-DCANDLE_CUDA_LEGACY_FP8=1",
    "-DCANDLE_CUDA_LEGACY_FP4=1",
];
pub const FAMILIES: &[&str] = &["affine", "binary", "cast", "conv", "fill", "indexing", "reduce", "sort", "ternary", "unary", "quantized", "mmvq_gguf", "mmq_quantize",
    "mmq_instance_q4_0", "mmq_instance_q4_1", "mmq_instance_q5_0", "mmq_instance_q5_1", "mmq_instance_q8_0",
    "mmq_instance_q2_k", "mmq_instance_q3_k", "mmq_instance_q4_k", "mmq_instance_q5_k", "mmq_instance_q6_k", "moe_gguf", "moe_simt_f16"];

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

// The original kernels are C++ template instantiations, not extern-C entries.
// Discover their exact NVCC symbol names; reject unknown/missing/ambiguous ABI.
pub fn mmq_codec_id(codec: &str) -> Option<u32> {
    Some(match codec {
        "q4_0" => 2, "q4_1" => 3, "q5_0" => 6, "q5_1" => 7, "q8_0" => 8,
        "q2_k" => 10, "q3_k" => 11, "q4_k" => 12, "q5_k" => 13, "q6_k" => 14,
        _ => return None,
    })
}

pub fn mmq_entries(ptx: &str, family: &str) -> Option<std::collections::BTreeMap<String, String>> {
    let entries: Vec<_> = ptx.lines().filter_map(|line| {
        let mut words = line.split_whitespace();
        if !words.any(|word| word == ".entry") { return None; }
        Some(words.next()?.split('(').next()?)
    }).collect();
    let mut map = std::collections::BTreeMap::new();
    let mut add = |key: String, marker: String, widths: &[u32]| -> Option<()> {
        let mut hits = entries.iter().filter(|&&name| {
            name.strip_prefix("_ZL").or_else(|| name.strip_prefix("_Z"))
                .is_some_and(|s| s.starts_with(&marker))
        });
        let name = *hits.next()?;
        if hits.next().is_some() { return None; }
        // Validate the actual PTX declaration, not just the C++ mangling.
        // NVCC emits scalar parameters as .u64/.b64 or .u32/.b32.
        let mut declarations = ptx.match_indices(name).filter_map(|(offset, _)| {
            let start = ptx[..offset].rfind('\n').map_or(0, |p| p+1);
            let prefix = &ptx[start..offset];
            if !prefix.split_whitespace().any(|w| w == ".entry") {
                return None;
            }
            let tail = &ptx[offset+name.len()..];
            let tail = tail.trim_start().strip_prefix('(')?;
            let (params, _) = tail.split_once(')')?;
            let parsed: Option<Vec<u32>> = params.split(',').map(|p| {
                let mut words = p.split_whitespace();
                if !words.any(|w| w == ".param") { return None; }
                match words.next()? {
                    ".u64" | ".b64" | ".s64" => Some(64),
                    ".u32" | ".b32" | ".s32" => Some(32),
                    _ => None,
                }
            }).collect();
            parsed
        });
        if declarations.next()?.as_slice() != widths || declarations.next().is_some() {
            return None;
        }
        map.insert(key, name.to_string());
        Some(())
    };
    if family == "mmq_quantize" {
        for (id, name) in ["D4", "DS4", "D2S6"].into_iter().enumerate() {
            add(name.to_string(), format!("17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout{id}EE"),
                &[64,64,64,64,64,64,64,64,32,32])?;
        }
    } else {
        let id = mmq_codec_id(family.strip_prefix("mmq_instance_")?)?;
        for tile in (8..=64).step_by(8) {
            for check in 0..=1 {
                add(format!("{tile}:{check}"), format!("9mul_mat_qIL9ggml_type{id}ELi{tile}ELb{check}EE"),
                    &[64,64,64,64,64,64,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32])?;
            }
        }
    }
    Some(map)
}

/// Exact DP4A shared-memory formula from mmq_get_nbytes_shared, SM61 only.
/// QI4_0/1=4, QI8_0/1=8, QI4/5/6_K=32; tile_y=64.
pub fn mmq_smem_bytes(codec: &str, tile: u32) -> Option<u32> {
    if !(8..=64).contains(&tile) || tile % 8 != 0 { return None; }
    let y = 64;
    let (qs, dm, sc) = match codec {
        "q4_0" | "q4_1" => (y*33, y*32/4+y/4, 0),
        "q5_0" | "q5_1" | "q8_0" => (y*65, y*64/8+y/4, 0),
        "q2_k" => (y*65, y*33, 0),
        "q3_k" => (y*65, y, y*32/8+y/8),
        "q4_k" => (y*33, y*32/32, y*32/8+y/8),
        "q5_k" | "q6_k" => (y*65, y*32/32+y/32, y*32/8+y/8),
        _ => return None,
    };
    Some(tile*4 + (qs+dm+sc)*4 + (tile*144).div_ceil(1024)*1024)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmqTile {
    pub x: u32,
    pub shared: u32,
    pub grid: (u32,u32,u32),
    pub checked: bool,
}

pub fn mmq_tile(codec: &str, rows: u32, batch: u32, limit: u32) -> Option<MmqTile> {
    if rows == 0 || batch == 0 { return None; }
    let mut best = None;
    let mut best_tiles = u32::MAX;
    for x in (8..=64).step_by(8) {
        let shared = mmq_smem_bytes(codec, x)?;
        if shared > limit { continue; }
        let tiles = batch.div_ceil(x);
        if tiles < best_tiles {
            best = Some(MmqTile { x, shared, grid: (rows.div_ceil(64), tiles, 1), checked: rows % 64 != 0 });
            best_tiles = tiles;
        }
        if best_tiles == 1 { break; }
    }
    best.filter(|p| p.grid.1 <= 65535)
}

#[cfg(test)]
mod mmq_tests {
    use super::*;
    #[test]
    fn sm61_tiles_and_limits() {
        assert_eq!(mmq_tile("q4_0", 64, 9, 49152).unwrap(),
            MmqTile { x:16, shared:mmq_smem_bytes("q4_0",16).unwrap(), grid:(1,1,1), checked:false });
        assert_eq!(mmq_tile("q4_k", 65, 65, 49152).unwrap().x, 40);
        assert!(mmq_tile("q4_k", 65, 65, 49152).unwrap().checked);
        assert!(mmq_tile("q2_k", 64, 64, 1024).is_none());
        for codec in MMVQ_CODECS {
            for batch in 1..=1024 {
                let p = mmq_tile(codec, 129, batch, 49152).unwrap();
                assert!(p.shared <= 49152 && p.x <= 64 && p.checked);
                assert_eq!(p.grid, (3, batch.div_ceil(p.x), 1));
            }
        }
    }
    #[test]
    fn exact_mangled_template_inventory() {
        fn params(pointers_and_i64: usize, i32_count: usize) -> String {
            (0..pointers_and_i64+i32_count).map(|i| format!(".param .u{} p{i}", if i < pointers_and_i64 {64} else {32})).collect::<Vec<_>>().join(",\n")
        }
        let mut ptx = String::new();
        for tile in (8..=64).step_by(8) {
            for check in 0..=1 {
                ptx += &format!(".entry _ZL9mul_mat_qIL9ggml_type12ELi{tile}ELb{check}EEvPKc({}) {{ ret; }}\n", params(6,17));
            }
        }
        assert_eq!(mmq_entries(&ptx,"mmq_instance_q4_k").unwrap().len(),16);
        assert!(mmq_entries(&ptx,"mmq_instance_q4_0").is_none());
        assert!(mmq_entries(&ptx.replace("ELi8ELb0EE","ELi9ELb0EE"),"mmq_instance_q4_k").is_none());
        assert!(mmq_entries(&ptx.replacen(".param .u64", ".param .u32", 1),"mmq_instance_q4_k").is_none());
        let q = (0..3).map(|i| format!(".entry _ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout{i}EEvPKf({}) {{ ret; }}\n", params(8,2))).collect::<String>();
        assert_eq!(mmq_entries(&q,"mmq_quantize").unwrap().len(),3);
    }
}

/// These five source units are not part of the SM61 external module set.
/// Keep the reason explicit: "static" is not external dispatch coverage.
pub const SM61_REMAINING: &[(&str, &str)] = &[
    ("moe/moe_wmma.cu", "EXCLUDED_PRE_VOLTA"),
    ("moe/moe_wmma_gguf.cu", "EXCLUDED_PRE_VOLTA"),
    ("moe/moe_align.cu", "CUTILE_ONLY"),
];

pub fn conv_sort_entries(family: &str) -> Option<Vec<String>> {
    let (ops, types): (&[&str], &[&str]) = match family {
        "conv" => (&["conv1d", "conv2d", "conv_transpose1d", "conv_transpose2d",
            "avg_pool2d", "max_pool2d", "upsample_nearest2d", "upsample_bilinear2d",
            "im2col", "im2col1d", "col2im1d"],
            &["bf16", "f16", "f32", "f64", "u8", "u32"]),
        "sort" => (&["asort_asc", "asort_desc"],
            &["bf16", "f16", "f32", "f64", "u8", "u32", "i64"]),
        _ => return None,
    };
    Some(ops.iter().flat_map(|op| types.iter().map(move |dtype| format!("{op}_{dtype}"))).collect())
}

pub fn has_conv_sort_entries(ptx: &str, family: &str) -> bool {
    let Some(expected) = conv_sort_entries(family) else { return false; };
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for line in ptx.lines() {
        let mut words = line.split_whitespace();
        if words.any(|word| word == ".entry") {
            if let Some(name) = words.next() {
                *counts.entry(name.split('(').next().unwrap_or(name)).or_default() += 1;
            }
        }
    }
    expected.iter().all(|name| counts.get(name.as_str()) == Some(&1))
}

#[cfg(test)]
mod conv_sort_tests {
    use super::*;
    #[test]
    fn complete_source_and_entry_inventory() {
        assert_eq!(FAMILIES.len(), 25);
        assert_eq!(SM61_REMAINING.len(), 3);
        assert_eq!(FAMILIES.iter().collect::<std::collections::HashSet<_>>().len(), 25);
        for (family, count) in [("conv", 66), ("sort", 14)] {
            let entries = conv_sort_entries(family).unwrap();
            assert_eq!(entries.len(), count);
            let ptx = entries.iter().map(|name| format!(".visible .entry\t{name}() {{ ret; }}\n")).collect::<String>();
            assert!(has_conv_sort_entries(&ptx, family));
            assert!(!has_conv_sort_entries(&ptx.replace(&entries[0], "missing"), family));
            assert!(!has_conv_sort_entries(&format!("{ptx}{ptx}"), family));
        }
    }
}

/// Resolve exact native MoE symbols and check scalar argument widths.
/// Do not require exact mangling of the function-pointer template argument.
pub fn moe_entries(ptx: &str, family: &str) -> Option<std::collections::BTreeMap<String,String>> {
    let entries: Vec<_> = ptx.lines().filter_map(|line| {
        let mut words = line.split_whitespace();
        if !words.any(|w| w == ".entry") { return None; }
        Some(words.next()?.split('(').next()?)
    }).collect();
    let mut map = std::collections::BTreeMap::new();
    let mut add = |key: &str, marker: &str, block: Option<&str>, widths: &[u32]| -> Option<()> {
        let mut hits = entries.iter().filter(|&&n| n.contains(marker) && block.is_none_or(|b| n.contains(b)));
        let name = *hits.next()?;
        if hits.next().is_some() { return None; }
        let offset = ptx.match_indices(name).find_map(|(i,_)| {
            let start = ptx[..i].rfind('\n').map_or(0, |p| p+1);
            ptx[start..i].split_whitespace().any(|w| w == ".entry").then_some(i)
        })?;
        let (params,_) = ptx[offset+name.len()..].trim_start().strip_prefix('(')?.split_once(')')?;
        let actual: Option<Vec<u32>> = params.split(',').map(|p| {
            let mut w = p.split_whitespace();
            if !w.any(|s| s == ".param") { return None; }
            match w.next()? {
                ".u64" | ".b64" | ".s64" => Some(64),
                ".u32" | ".b32" | ".s32" => Some(32),
                _ => None,
            }
        }).collect();
        if actual?.as_slice() != widths { return None; }
        map.insert(key.to_string(),name.to_string());
        Some(())
    };
    match family {
        "moe_simt_f16" => {
            add("count","count_tokens_per_expert_kernel",None,&[64,64,32])?;
            add("scan","expert_prefix_sum_kernel",None,&[64,64,32])?;
            add("gemm","moe_gemm_simt_f16_kernel",None,&[64,64,64,64,64,64,32,32,32,32,32])?;
        }
        "moe_gguf" => {
            if !entries.contains(&"candle_dkr_moe_gguf_warp_tail_v1") { return None; }
            add("quantize","quantize_q8_1",None,&[64,64,32,32])?;
            for (tag, block) in [("q8_0","block_q8_0"),("q4_k","block_q4_K"),
                ("q2_k","block_q2_K"),("q3_k","block_q3_K"),("q5_k","block_q5_K"),("q6_k","block_q6_K")] {
                add(tag,"moe_gemm_gguf_kernel",Some(block),&[64,64,64,64,64,64,32,32,32,32,32,32])?;
            }
        }
        _ => return None,
    }
    Some(map)
}

#[cfg(test)]
mod moe_tests {
    use super::*;
    fn entry(name:&str, ptr:usize, scalars:usize)->String {
        let params=(0..ptr+scalars).map(|i|format!(".param .u{} p{i}",if i<ptr{64}else{32})).collect::<Vec<_>>().join(",\n");
        format!(".entry {name}({params}) {{ ret; }}\n")
    }
    #[test]
    fn moe_symbol_width_and_uniqueness() {
        let simt=entry("_Zcount_tokens_per_expert_kernel",2,1)+
            &entry("_Zexpert_prefix_sum_kernel",2,1)+&entry("_Zmoe_gemm_simt_f16_kernel",6,5);
        assert_eq!(moe_entries(&simt,"moe_simt_f16").unwrap().len(),3);
        assert!(moe_entries(&simt.replacen(".u64",".u32",1),"moe_simt_f16").is_none());
        let mut gguf=entry("_Zquantize_q8_1",2,2)+".entry candle_dkr_moe_gguf_warp_tail_v1() { ret; }\n";
        for tag in ["q8_0","q4_K","q2_K","q3_K","q5_K","q6_K"] {
            gguf+=&entry(&format!("_Zmoe_gemm_gguf_kernel_block_{tag}"),6,6);
        }
        assert_eq!(moe_entries(&gguf,"moe_gguf").unwrap().len(),7);
        assert!(moe_entries(&(gguf.clone()+&entry("_Zmoe_gemm_gguf_kernel_block_q4_K_duplicate",6,6)),"moe_gguf").is_none());
    }
}
