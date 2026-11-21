//! S5.5b build-only verifier/compiler for producer-owned typed launch recipes.
//! This file is compiled by candle-core/build.rs, not linked into the runtime.
use sha2::{Digest, Sha256};
use serde_json::Value;
use std::{collections::BTreeSet, error::Error, fs, path::Path};

const SCHEMA: &str = "CANDLE-DKR-R124-S55B-LAUNCH-V1";

fn sha(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
fn obj<'a>(value: &'a Value, where_: &str) -> Result<&'a serde_json::Map<String,Value>, Box<dyn Error>> {
    value.as_object().ok_or_else(|| format!("{where_}: not an object").into())
}
fn keys(v: &Value, allowed: &[&str], name: &str) -> Result<(), Box<dyn Error>> {
    let actual = obj(v,name)?.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = allowed.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!("{name}: unexpected or missing field keys").into());
    }
    Ok(())
}
fn string(v: &Value, key: &str) -> Result<String, Box<dyn Error>> {
    let s=v[key].as_str().ok_or_else(|| format!("missing {key}"))?;
    if s.is_empty() || s.len()>192 || !s.is_ascii() ||
       s.bytes().any(|b| b<=b' ' || b>=127) {
        return Err(format!("invalid recipe field: {key}").into());
    }
    Ok(s.to_owned())
}
fn string_literal(v: &Value,key:&str)->Result<String,Box<dyn Error>> {
    Ok(format!("{:?}", string(v,key)?))
}
fn positive(v: &Value,key: &str,max: u64)->Result<u64,Box<dyn Error>> {
    let n=v[key].as_u64().ok_or_else(||format!("missing uint field {key}"))?;
    if n==0 || n>max {return Err(format!("{key}: out of range").into())}
    Ok(n)
}
fn checked_symbol(
    v: &Value,key:&str,artifact: &dkr_kernel_registry::VerifiedArtifact
)->Result<String, Box<dyn Error>> {
    let symbol=string(v,key)?;
    if !artifact.has_entry(&symbol) {
        return Err(format!("qualified PTX missing symbol {key}: {symbol}").into());
    }
    Ok(format!("{symbol:?}"))
}
fn vec_values(v: &Value, name:&str, count:usize, max:u64)
    ->Result<Vec<u64>,Box<dyn Error>>
{
    let arr=v[name].as_array().ok_or_else(|| format!("missing array {name}"))?;
    if arr.len()!=count {return Err(format!("incorrect array length: {name}").into());}
    arr.iter().map(|x|{
        let n=x.as_u64().ok_or_else(||format!("invalid value for {name}"))?;
        if n==0 || n>max {return Err(format!("out of range {name}").into())}
        Ok(n)
    }).collect()
}

/// Admission intentionally ties derived recipes to the S5.1/S5.5a
/// fingerprints and the current build's verified PTX packages.
/// Runtime never receives or accesses the producer's file path.
pub(super) fn generate(
    out: &Path, native_root: &Path, recipes_root: &Path, mmq_root:&Path,
    target_cc: u32,
)->Result<(),Box<dyn Error>> {
    if !recipes_root.is_absolute() || !mmq_root.is_absolute() {
        return Err("S5.5b and S5.5a recipe directories must be absolute".into());
    }
    let recipes_root=recipes_root.canonicalize()?;
    let mmq_root=mmq_root.canonicalize()?;
    let path=recipes_root.join("manifest.json");
    println!("cargo:rerun-if-changed={}",path.display());
    let contents=fs::read(&path)?;
    if contents.len()>256*1024 {return Err("S5.5b manifest too large".into());}
    let v:Value=serde_json::from_slice(&contents)?;
    keys(&v,&["schema","target_cc","source_seal_sha256",
               "source_index_sha256","source_index_mode",
               "family_provenance","recipes"],"S5.5b root")?;
    if v["schema"].as_str()!=Some(SCHEMA) ||
       v["target_cc"].as_u64()!=Some(target_cc.into()) {
        return Err("S5.5b schema or device target mismatch".into());
    }
    let s55a:Value=serde_json::from_slice(&fs::read(mmq_root.join("manifest.json"))?)?;
    if v["source_seal_sha256"]!=s55a["source_seal_sha256"] ||
       v["source_index_sha256"]!=s55a["source_index_sha256"] ||
       v["source_index_mode"]!=s55a["source_index_mode"] ||
       v["target_cc"]!=s55a["target_cc"] {
        return Err("S5.5b source seal/index/target differs from qualified S5.5a".into());
    }
    let index=native_root.join(super::native_catalog::INDEX);
    let real_index=if index.is_file(){Some(sha(&index)?)}else{None};
    let real_mode=if real_index.is_some(){"present"}else{"legacy-absent"};
    let actual_index=match real_index {
        Some(x)=>Value::String(x),None=>Value::Null
    };
    if v["source_index_sha256"]!=actual_index ||
       v["source_index_mode"].as_str()!=Some(real_mode) {
        return Err("S5.5b native index was replaced or added".into());
    }

    let expected_families=[
        "quantized","cast","mmvq_gguf","moe_simt_f16","moe_gguf"];
    let declared=obj(&v["family_provenance"],"S5.5b family_provenance")?;
    if declared.len()!=expected_families.len() {
        return Err("S5.5b family provenance inventory incomplete".into());
    }
    let catalog=super::native_catalog::NativeCatalog::load(native_root)?;
    let mut artifacts=std::collections::HashMap::new();
    for &name in &expected_families {
        let row=declared.get(name).ok_or_else(||format!("missing {name} family"))?;
        keys(row,&["manifest_sha256","ptx_sha256"],name)?;
        let (manifest,ptx)=super::native_catalog::NativeCatalog::paths(native_root,name);
        if row["manifest_sha256"].as_str()!=Some(sha(&manifest)?.as_str())
           || row["ptx_sha256"].as_str()!=Some(sha(&ptx)?.as_str()) {
            return Err(format!("S5.5b qualified package hashes changed: {name}").into());
        }
        let (artifact,_)=catalog.read_package(native_root,name)?;
        if artifact.key().target_cc()!=target_cc {
            return Err(format!("S5.5b incompatible target in {name}").into());
        }
        artifacts.insert(name,artifact);
    }

    let recipes=&v["recipes"];
    keys(recipes,&["fp8","fp4","bf16","routed"],"S5.5b recipe inventory")?;
    let q=artifacts.get("quantized").ok_or("missing quantized artifact")?;
    let cast=artifacts.get("cast").ok_or("missing cast artifact")?;
    let fp8=&recipes["fp8"];
    const FP8_STR:&[&str]=&["simt_gemv","simt_prefill","dp4a_gemv","dp4a_prefill","quantize"];
    const FP8_NUM:&[&str]=&["prefill_tokens","rows_per_block","threads","k_alignment"];
    keys(fp8,&["simt_gemv","simt_prefill","dp4a_gemv","dp4a_prefill",
            "quantize","prefill_tokens","rows_per_block","threads",
            "k_alignment","groups"],"FP8")?;
    let mut fp8_fields=Vec::new();
    for &name in FP8_STR {
        let symbol=checked_symbol(fp8,name,q)?;
        fp8_fields.push(format!("{name}:{symbol}"));
    }
    for &name in FP8_NUM {
        let n=positive(fp8,name,if name=="k_alignment"{4096}else{1024})?;
        fp8_fields.push(format!("{name}:{n}"));
    }
    let groups=vec_values(fp8,"groups",3,4096)?;
    if !(groups[0]<groups[1] && groups[1]<groups[2]) ||
       groups.iter().any(|g|positive(fp8,"k_alignment",4096).unwrap()%g!=0) {
        return Err("FP8 activation groups misaligned or unordered".into());
    }
    fp8_fields.push(format!("groups:[{},{},{}]",groups[0],groups[1],groups[2]));
    let fp8_generated=format!("pub(super) const FP8:Fp8Recipe=Fp8Recipe{{{}}};\n",fp8_fields.join(","));

    let fp4=&recipes["fp4"];
    const FP4_STR:&[&str]=&["scalar","cast","quantize","reference","tiled","mx_mat","mx_moe"];
    const FP4_NUM:&[&str]=&["auto_cc","min_dp4a_cc","auto_min_n","auto_min_k",
                           "auto_min_batch","scalar_threads","cast_threads",
                           "quant_threads","warp","matrix_warps","tile_n","tile_m"];
    keys(fp4,&["scalar","cast","quantize","reference","tiled","mx_vec",
               "mx_mat","mx_moe","auto_cc","min_dp4a_cc","auto_min_n",
               "auto_min_k","auto_min_batch","scalar_threads","cast_threads",
               "quant_threads","warp","matrix_warps","tile_n","tile_m"],"FP4")?;
    let mut fp4_fields=Vec::new();
    for &name in FP4_STR {
        fp4_fields.push(format!("{name}:{}",checked_symbol(fp4,name,q)?));
    }
    let vectors=fp4["mx_vec"].as_array().ok_or("FP4 mx_vec array missing")?;
    if vectors.len()!=8 {return Err("FP4 requires eight vector entrypoints".into());}
    let mut quoted=Vec::new();
    for v in vectors {
        let name=v.as_str().ok_or("invalid FP4 vector entry")?;
        if !q.has_entry(name) {return Err(format!("FP4 vector missing: {name}").into());}
        quoted.push(format!("{name:?}"));
    }
    fp4_fields.push(format!("mx_vec:[{}]",quoted.join(",")));
    for &name in FP4_NUM {
        let n=positive(fp4,name,if name.contains("threads") || name=="warp" || name=="matrix_warps"{1024}else{65535})?;
        fp4_fields.push(format!("{name}:{n}"));
    }
    if positive(fp4,"auto_cc",65535)?!=u64::from(target_cc)
       || positive(fp4,"min_dp4a_cc",65535)?>u64::from(target_cc)
       || positive(fp4,"warp",1024)?*positive(fp4,"matrix_warps",1024)?>1024 {
        return Err("FP4 launch block or target exceeds device capability".into());
    }
    let fp4_generated=format!("pub(super) const FP4:Fp4Recipe=Fp4Recipe{{{}}};\n",fp4_fields.join(","));

    let bf16=&recipes["bf16"];
    keys(bf16,&["matmul","cast_to_f32","cast_from_f32"],"BF16")?;
    let mode=string(bf16,"matmul")?;
    let enum_name=match mode.as_str() {
        "cublas-sgemm-f32-round-bf16"=>"WidenF32",
        "cublas-native-bf16" if target_cc>=80=>"Native",
        _=>return Err("BF16 GEMM route incompatible with target".into())
    };
    let bf16_generated=format!(
        "pub(super) const BF16:Bf16Recipe=Bf16Recipe{{\
         matmul:bf16_recipe::Bf16MatmulRoute::{enum_name},\
         cast_to_f32:{},cast_from_f32:{} }};\n",
        checked_symbol(bf16,"cast_to_f32",cast)?,
        checked_symbol(bf16,"cast_from_f32",cast)?
    );

    let routed=&recipes["routed"];
    const ROUTED_NUM:&[&str]=&[
        "max_batch","quant_threads","warp","small_warps","large_warps",
        "small_batch","row_padding","q8_block","q8_bytes","max_experts",
        "simt_threads","simt_tile_n","simt_shared","gguf_tile_n",
        "gguf_shared_base","gguf_shared_limit",
    ];
    keys(routed,&["max_batch","quant_threads","warp","small_warps","large_warps",
                 "small_batch","row_padding","q8_block","q8_bytes","max_experts",
                 "simt_threads","simt_tile_n","simt_shared","gguf_tile_n",
                 "gguf_shared_base","gguf_shared_limit",
                 "mmvq_family","simt_family","gguf_family","simt_input","gguf_input","gguf_abi"],"Routed")?;
    if string(routed,"mmvq_family")?!="mmvq_gguf" ||
       string(routed,"simt_family")?!="moe_simt_f16" ||
       string(routed,"gguf_family")?!="moe_gguf" ||
       string(routed,"simt_input")?!="f16" ||
       string(routed,"gguf_input")?!="f32" ||
       string(routed,"gguf_abi")?!=super::contract::MOE_GGUF_ABI {
        return Err("routed family/type/ABI differs from qualified native contract".into());
    }
    let mut routed_fields=Vec::new();
    for &name in ROUTED_NUM {
        let bound=if name=="gguf_shared_limit"{65536}else if name=="row_padding"{65536}
                  else if name.contains("threads"){1024}else{65535};
        let n=positive(routed,name,bound)?;
        routed_fields.push(format!("{name}:{n}"));
    }
    if positive(routed,"max_batch",65535)?>8 ||
       positive(routed,"small_batch",65535)?>positive(routed,"max_batch",65535)? ||
       positive(routed,"q8_block",65535)?!=32 ||
       positive(routed,"q8_bytes",65535)?!=36 ||
       positive(routed,"row_padding",65535)?%32!=0 ||
       positive(routed,"simt_threads",1024)?>1024 ||
       positive(routed,"quant_threads",1024)?>1024 ||
       positive(routed,"warp",65535)?*positive(routed,"small_warps",65535)?>1024 ||
       positive(routed,"gguf_shared_base",65535)?>positive(routed,"gguf_shared_limit",65536)? {
        return Err("Routed parameters incompatible with qualified GGUF/SM61 kernel ABI".into());
    }
    let mmvq=artifacts.get("mmvq_gguf").ok_or("missing MMVQ artifact")?;
    for codec in super::contract::MMVQ_CODECS {
        for tag in ["bf16","f16","f32"] {
            for batch in 1..=positive(routed,"max_batch",8)? {
                let symbol=format!("mmvq_gguf_{codec}_{tag}_plain_cuda{batch}");
                if !mmvq.has_entry(&symbol) {
                    return Err(format!("Routed MMVQ missing qualified symbol {symbol}").into());
                }
            }
        }
    }
    let routed_generated=format!(
        "pub(super) const ROUTED:RoutedRecipe=RoutedRecipe{{{}}};\n",
        routed_fields.join(",")
    );
    fs::write(out.join("dkr_neutral_recipes.rs"),format!(
        "{fp8_generated}{fp4_generated}{bf16_generated}{routed_generated}"
    ))?;
    println!("cargo:warning=DKR_R124_S55B_RECIPES_EMBEDDED=4");
    Ok(())
}
