//! S5.5c: offline, Cudaforge-backed native recipe catalogue producer.
//!
//! No CUDA compiler is run. Source of truth is the versioned producer-local
//! recipes/sm61.json; the Rust runtime consumes only sealed derived manifests.
//! PTX, CUBIN, S5.1 receipts and historical S5.5a/b outputs stay unchanged.
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

type R<T> = Result<T, Box<dyn Error>>;

const SPEC_SCHEMA: &str = "DKR-CUDAFORGE-RECIPE-SPEC-V1";
const SEAL_SCHEMA: &str = "CANDLE-DKR-R124-S51-SEAL-V1";
const MMQ_SCHEMA: &str = "CANDLE-DKR-R124-S55A-MMQ-LAUNCH-V1";
const RECIPES_SCHEMA: &str = "CANDLE-DKR-R124-S55B-LAUNCH-V1";
const CUBIN_SCHEMA: &str = "CANDLE-DKR-R124-S54B-CUBIN-V1";
const BUNDLE_SCHEMA: &str = "CANDLE-DKR-R124-S55C-CUDAFORGE-CATALOG-V1";
const MAX_JSON_BYTES: u64 = 1024 * 1024;
const MAX_PTX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CUBIN_BYTES: u64 = 32 * 1024 * 1024;
const CODECS: &[&str] = &[
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
];
const DEPENDENCIES: &[&str] = &[
    "quantized", "cast", "mmvq_gguf", "moe_simt_f16", "moe_gguf",
];

fn read_limited(path: &Path, max: u64) -> R<Vec<u8>> {
    if !path.is_file() { return Err(format!("missing required file: {}", path.display()).into()); }
    let mut input = fs::File::open(path)?.take(max + 1);
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(format!("input exceeds byte limit: {}", path.display()).into());
    }
    Ok(bytes)
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_json(path: &Path) -> R<(Value, Vec<u8>)> {
    let bytes = read_limited(path, MAX_JSON_BYTES)?;
    let value = serde_json::from_slice(&bytes)?;
    Ok((value, bytes))
}

fn field<'a>(v: &'a Value, key: &str) -> R<&'a str> {
    v[key].as_str().ok_or_else(|| format!("missing string field {key}").into())
}

fn object<'a>(v: &'a Value, name: &str) -> R<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| format!("missing object: {name}").into())
}

fn equal(actual: &Value, expected: &Value, description: &str) -> R<()> {
    if actual != expected {
        return Err(format!("S5.5c: mismatch in {description}").into());
    }
    Ok(())
}

fn source_symbols(ptx: &str) -> BTreeSet<&str> {
    ptx.lines()
        .filter_map(|line| line.split_once(".entry"))
        .filter_map(|(_, after)| {
            let symbol = after.trim_start()
                .split(|ch: char| ch == '(' || ch.is_whitespace()).next()?;
            (!symbol.is_empty()).then_some(symbol)
        })
        .collect()
}

fn check_entry(symbols: &BTreeSet<&str>, recipe: &Value, key: &str) -> R<()> {
    let symbol = field(recipe, key)?;
    if !symbols.contains(symbol) {
        return Err(format!("missing qualified PTX symbol: {key}={symbol}").into());
    }
    Ok(())
}

fn validate_recipe_spec(spec: &Value) -> R<u32> {
    if spec["schema"] != SPEC_SCHEMA { return Err("unsupported Cudaforge recipe spec".into()); }
    let cc = spec["target_cc"].as_u64()
        .and_then(|cc| u32::try_from(cc).ok()).ok_or("invalid target compute capability")?;
    if cc != 61 { return Err("this frozen producer profile is only qualified on sm61".into()); }
    if field(spec, "source_baseline")?.len() != 40 {
        return Err("invalid qualified source baseline".into());
    }
    // Check Cudaforge's target selection without invoking NVCC/ptxas.
    let selected = cudaforge::KernelBuilder::new()
        .compute_cap(cc as usize).get_compute_cap();
    if selected != Some(cc as usize) {
        return Err("Cudaforge compute capability selection changed".into());
    }
    let mmq = &spec["mmq"];
    if mmq["row_tile"] != 64 || mmq["quant_padding"] != 512
        || mmq["warp"] != 32 || mmq["max_shared_bytes"] != 49152
        || mmq["quant_block"] != json!([128,1,1])
        || mmq["matrix_block"] != json!([32,8,1])
    {
        return Err("qualified SM61 MMQ ABI and geometry changed".into());
    }
    let table = object(&mmq["codecs"], "mmq codecs")?;
    if table.len() != CODECS.len() { return Err("incomplete MMQ codec inventory".into()); }
    for &codec in CODECS {
        let tiles = table.get(codec).and_then(Value::as_array)
            .ok_or_else(|| format!("missing MMQ tiles for {codec}"))?;
        if tiles.len() != 8 { return Err(format!("MMQ tile count for {codec}").into()); }
        for (i, entry) in tiles.iter().enumerate() {
            let width = 8 * (i as u64 + 1);
            if entry["width"] != width
                || entry["shared_bytes"].as_u64().is_none_or(|b| b == 0 || b > 49152) {
                return Err(format!("invalid MMQ tile for {codec}/{width}").into());
            }
        }
    }
    let recipes = object(&spec["recipes"], "typed recipes")?;
    if recipes.len() != 4
        || !["fp8", "fp4", "bf16", "routed"].iter().all(|key| recipes.contains_key(*key)) {
        return Err("incomplete four-recipe inventory".into());
    }
    Ok(cc)
}

fn validate_source(
    seal_path: &Path, spec: &Value,
) -> R<(PathBuf, Value, String, Value, String)> {
    let (seal, seal_bytes) = read_json(seal_path)?;
    if seal["schema"] != SEAL_SCHEMA ||
       seal["source_baseline"] != spec["source_baseline"] {
        return Err("S5.1 seal/schema/source baseline mismatch".into());
    }
    let catalogue = PathBuf::from(field(&seal, "catalogue")?).canonicalize()?;
    let inventory = &seal["catalogue_inventory"];
    let packages = object(&inventory["packages"], "S5.1 family inventory")?;
    if packages.len() != 25 { return Err("S5.1 native family cardinality changed".into()); }
    let index_path = catalogue.join("native-catalog.json");
    let index_sha = if index_path.exists() {
        Some(digest(&read_limited(&index_path, MAX_JSON_BYTES)?))
    } else { None };
    let index_value = match index_sha.as_ref() {
        Some(sha) => Value::String(sha.clone()), None => Value::Null,
    };
    equal(&inventory["index_sha256"], &index_value, "S5.1 optional index state")?;
    for (name, hashes) in packages {
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            return Err(format!("invalid native family name: {name}").into());
        }
        let manifest = read_limited(&catalogue.join(name).join("manifest.json"), MAX_JSON_BYTES)?;
        let ptx = read_limited(&catalogue.join(name).join(format!("{name}.ptx")), MAX_PTX_BYTES)?;
        if hashes["manifest_sha256"] != digest(&manifest)
            || hashes["ptx_sha256"] != digest(&ptx) {
            return Err(format!("sealed native package modified: {name}").into());
        }
    }
    Ok((catalogue, seal, digest(&seal_bytes), index_value,
        if index_sha.is_some() { "present" } else { "legacy-absent" }.to_owned()))
}

fn validate_cubins(
    cubin_dir: &Path, seal: &Value, cc: u32,
) -> R<String> {
    let (manifest, bytes) = read_json(&cubin_dir.join("manifest.json"))?;
    if manifest["schema"] != CUBIN_SCHEMA || manifest["target_cc"] != cc {
        return Err("incompatible CUBIN inventory".into());
    }
    let packages = object(&seal["catalogue_inventory"]["packages"], "sealed packages")?;
    let families = object(&manifest["families"], "CUBIN families")?;
    if families.len() != packages.len() { return Err("missing CUBIN family".into()); }
    for (family, hashes) in packages {
        let row = families.get(family).ok_or("CUBIN family missing from inventory")?;
        if row["target_cc"] != cc || row["ptx_sha256"] != hashes["ptx_sha256"] {
            return Err(format!("CUBIN/PTX target or provenance mismatch: {family}").into());
        }
        let image = read_limited(
            &cubin_dir.join("cubins").join(format!("{family}.cubin")), MAX_CUBIN_BYTES
        )?;
        if !image.starts_with(b"\x7fELF") || row["cubin_sha256"] != digest(&image)
            || row["cubin_bytes"] != image.len() as u64 {
            return Err(format!("CUBIN ELF/digest/size mismatch: {family}").into());
        }
    }
    Ok(digest(&bytes))
}

fn make_manifests(
    spec: &Value, catalogue: &Path, seal: &Value, seal_sha: &str,
    index_sha: &Value, index_mode: &str, cc: u32,
) -> R<(Value, Value)> {
    let packages = object(&seal["catalogue_inventory"]["packages"], "sealed packages")?;
    let mut mmq_rows = Map::new();
    for &codec in CODECS {
        let family = format!("mmq_instance_{codec}");
        let package = packages.get(&family).ok_or("S5.1 MMQ family missing")?;
        let (meta, _) = read_json(&catalogue.join(&family).join("manifest.json"))?;
        let entries = object(&meta["mmq_entries"], "qualified MMQ entries")?;
        if entries.len() != 16 { return Err(format!("MMQ ABI symbol count: {family}").into()); }
        let ptx = read_limited(&catalogue.join(&family).join(format!("{family}.ptx")),MAX_PTX_BYTES)?;
        let text = std::str::from_utf8(&ptx)?;
        let symbols = source_symbols(text);
        for width in (8..=64).step_by(8) {
            for checked in 0..=1 {
                let key = format!("{width}:{checked}");
                let symbol = entries.get(&key).and_then(Value::as_str)
                    .ok_or_else(||format!("MMQ symbol key missing: {family}/{key}"))?;
                if !symbols.contains(symbol) {
                    return Err(format!("MMQ symbol absent from PTX: {family}/{key}").into());
                }
            }
        }
        mmq_rows.insert(codec.to_owned(),json!({
            "family":family,
            "ptx_sha256":package["ptx_sha256"],
            "tiles":spec["mmq"]["codecs"][codec]
        }));
    }
    let mmq = json!({
        "schema":MMQ_SCHEMA, "target_cc":cc,
        "source_seal_sha256":seal_sha,
        "source_index_sha256":index_sha,"source_index_mode":index_mode,
        "row_tile":spec["mmq"]["row_tile"],
        "quant_padding":spec["mmq"]["quant_padding"],
        "warp":spec["mmq"]["warp"],
        "max_shared_bytes":spec["mmq"]["max_shared_bytes"],
        "quant_block":spec["mmq"]["quant_block"],
        "matrix_block":spec["mmq"]["matrix_block"],
        "codecs":mmq_rows
    });

    let mut families = Map::new();
    for &family in DEPENDENCIES {
        let info = packages.get(family).ok_or("S5.1 recipe dependency missing")?;
        families.insert(family.to_owned(),json!({
            "manifest_sha256":info["manifest_sha256"],
            "ptx_sha256":info["ptx_sha256"]
        }));
    }
    let quant_text = std::str::from_utf8(&read_limited(
        &catalogue.join("quantized").join("quantized.ptx"), MAX_PTX_BYTES)?)?.to_owned();
    let cast_text = std::str::from_utf8(&read_limited(
        &catalogue.join("cast").join("cast.ptx"), MAX_PTX_BYTES)?)?.to_owned();
    let quant_symbols = source_symbols(&quant_text);
    let cast_symbols = source_symbols(&cast_text);
    for key in ["simt_gemv","simt_prefill","dp4a_gemv","dp4a_prefill","quantize"] {
        check_entry(&quant_symbols,&spec["recipes"]["fp8"],key)?;
    }
    for key in ["scalar","cast","quantize","reference","tiled","mx_mat","mx_moe"] {
        check_entry(&quant_symbols,&spec["recipes"]["fp4"],key)?;
    }
    let mx = spec["recipes"]["fp4"]["mx_vec"].as_array().ok_or("missing eight MXFP4 entries")?;
    if mx.len()!=8 { return Err("MXFP4 entry count changed".into()); }
    for symbol in mx {
        let symbol=symbol.as_str().ok_or("invalid MXFP4 symbol")?;
        if !quant_symbols.contains(symbol) { return Err(format!("missing MXFP4 symbol {symbol}").into()); }
    }
    for key in ["cast_to_f32","cast_from_f32"] {
        check_entry(&cast_symbols,&spec["recipes"]["bf16"],key)?;
    }
    for family in ["mmvq_gguf","moe_simt_f16","moe_gguf"] {
        let filename = catalogue.join(family).join(format!("{family}.ptx"));
        let ptx = read_limited(&filename, MAX_PTX_BYTES)?;
        if source_symbols(std::str::from_utf8(&ptx)?).is_empty() {
            return Err(format!("routed CUDA symbols missing: {family}").into());
        }
    }
    let recipes = json!({
        "schema":RECIPES_SCHEMA, "target_cc":cc,
        "source_seal_sha256":seal_sha,
        "source_index_sha256":index_sha,"source_index_mode":index_mode,
        "family_provenance":families,
        "recipes":spec["recipes"]
    });
    Ok((mmq,recipes))
}

fn encoded(value: &Value) -> R<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn compare_optional(result: &Value, path: &Option<PathBuf>, label: &str) -> R<()> {
    if let Some(path) = path {
        let (prior, _) = read_json(path)?;
        equal(result, &prior, label)?;
        println!("DKR_R124_S55C_SEMANTIC_PARITY=PASS manifest={label}");
    }
    Ok(())
}

fn main_inner() -> R<()> {
    let mut args = env::args_os().skip(1);
    if args.len()==1 {
        let first=args.next().ok_or("missing argument")?;
        if first=="--self-test" {
            let spec_path=Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes/sm61.json");
            let (spec,_) = read_json(&spec_path)?;
            validate_recipe_spec(&spec)?;
            assert_eq!(spec["mmq"]["codecs"]["q4_0"][1]["shared_bytes"],13696);
            assert_eq!(spec["mmq"]["codecs"]["q4_k"][4]["shared_bytes"],16064);
            assert_eq!(spec["mmq"]["codecs"]["q2_k"][7]["shared_bytes"],34560);
            println!("DKR_R124_S55C_PRODUCER_SELF_TEST=PASS");
            return Ok(());
        }
    }
    let mut seal=None;
    let mut cubins=None;
    let mut out=None;
    let mut compare_mmq=None;
    let mut compare_recipes=None;
    let mut args=env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let value=PathBuf::from(args.next().ok_or("expected option value")?);
        match flag.to_string_lossy().as_ref() {
            "--seal" => seal=Some(value),
            "--cubins" => cubins=Some(value),
            "--out" => out=Some(value),
            "--compare-s55a" => compare_mmq=Some(value),
            "--compare-s55b" => compare_recipes=Some(value),
            _ => return Err(format!("unknown producer argument: {}", flag.to_string_lossy()).into()),
        }
    }
    let seal=seal.ok_or("missing --seal")?;
    let cubins=cubins.ok_or("missing --cubins")?;
    let out:PathBuf=out.ok_or("missing --out")?;
    if !out.is_absolute() || out.exists() {
        return Err("output must be a fresh absolute directory; historical data cannot be overwritten".into());
    }
    let seal=seal.canonicalize()?;
    let cubins=cubins.canonicalize()?;
    let spec_path=Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes/sm61.json");
    let (spec,spec_bytes) = read_json(&spec_path)?;
    let cc=validate_recipe_spec(&spec)?;
    let (catalogue,source,seal_sha,index_sha,index_mode)=validate_source(&seal,&spec)?;
    let cubin_sha=validate_cubins(&cubins,&source,cc)?;
    let (mmq,recipes)=make_manifests(
        &spec,&catalogue,&source,&seal_sha,&index_sha,&index_mode,cc)?;
    compare_optional(&mmq,&compare_mmq,"mmq")?;
    compare_optional(&recipes,&compare_recipes,"recipes")?;
    let mmq_bytes=encoded(&mmq)?;
    let recipes_bytes=encoded(&recipes)?;
    let bundle=json!({
        "schema":BUNDLE_SCHEMA, "producer":"cudaforge", "producer_command":"dkr-cudaforge-catalog",
        "producer_version":env!("CARGO_PKG_VERSION"),
        "target_cc":cc, "family_count":25, "recipe_count":4, "mmq_codec_count":10,
        "source_seal_sha256":seal_sha,
        "source_index_sha256":index_sha, "source_index_mode":index_mode,
        "recipe_spec_sha256":digest(&spec_bytes),
        "mmq_manifest_sha256":digest(&mmq_bytes),
        "recipes_manifest_sha256":digest(&recipes_bytes),
        "cubin_manifest_sha256":cubin_sha
    });
    // Publish only after all input/provenance/parity checks have completed.
    fs::create_dir(&out)?;
    fs::create_dir(out.join("mmq"))?;
    fs::create_dir(out.join("recipes"))?;
    fs::write(out.join("mmq/manifest.json"),&mmq_bytes)?;
    fs::write(out.join("recipes/manifest.json"),&recipes_bytes)?;
    fs::write(out.join("catalogue.json"),encoded(&bundle)?)?;
    println!("DKR_R124_S55C_PRODUCER=PASS backend=cudaforge target=sm{cc}");
    println!("DKR_R124_S55C_CATALOGUE=PASS families=25 codecs=10 recipes=4");
    println!("DKR_R124_S55C_CATALOGUE_PATH={}",out.join("catalogue.json").display());
    Ok(())
}

fn main() {
    if let Err(e)=main_inner() {
        eprintln!("DKR_R124_S55C_PRODUCER_FAIL: {e}");
        std::process::exit(1);
    }
}
