//! Opt-in, strict external module loading. Only trusted local packages.
//! Embedded modules remain available when the provider is not selected.
//! Reuse Candle's original ABI, context, stream, module cache and launch path.

use std::path::{Path, PathBuf};
#[cfg(test)]
use sha2::{Digest, Sha256};
use candle_kernels as kernels;
use cudarc::driver::CudaContext;
use crate::Result;
use super::WrapErr;

use candle_kernels::dkr_s31_contract as contract;
#[path = "dkr_s31_validation.rs"]
mod validation;
#[path = "dkr_native_catalog.rs"]
mod native_catalog;
use native_catalog::NativeRecipe;
#[cfg(not(feature = "cuda-dkr-embedded"))]
use native_catalog::NativeCatalog;
use crate::quantized::fp8_recipe::{self, Fp8Recipe};
use crate::quantized::fp4_recipe::{self, Fp4Recipe};
use crate::cuda_backend::bf16_recipe::{self, Bf16Recipe};
use crate::cuda_backend::routed_recipe::{self,RoutedRecipe};
#[cfg(test)]
use validation::{validate, verify_package};
mod embedded {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/dkr_native_embedded.rs"));
}

#[cfg(feature = "cuda-dkr-neutral-mmq")]
mod neutral_mmq {
    include!(concat!(env!("OUT_DIR"), "/dkr_neutral_mmq.rs"));
}
#[cfg(feature = "cuda-dkr-neutral-recipes")]
mod neutral_recipes {
    use super::{Fp8Recipe,Fp4Recipe,Bf16Recipe,RoutedRecipe,bf16_recipe};
    include!(concat!(env!("OUT_DIR"), "/dkr_neutral_recipes.rs"));
}
#[cfg(feature = "cuda-dkr-cudaforge-catalog")]
mod cudaforge_catalogue {
    include!(concat!(env!("OUT_DIR"), "/dkr_cudaforge_catalogue.rs"));
}
use dkr_kernel_registry::{Registry, RegistryKey, VerifiedArtifact, NATIVE_ABI};

pub(super) struct ExternalModules {
    modules: Registry<VerifiedArtifact>,
    native_keys: std::collections::HashMap<String, RegistryKey>,
    trace: bool,
    device_cc: u32,
    recipe: NativeRecipe,
    fp8: Fp8Recipe,
    fp4: Fp4Recipe,
    bf16: Bf16Recipe,
    routed:RoutedRecipe,
    mmq_symbols: std::collections::HashMap<String, std::collections::BTreeMap<String, String>>,
    #[cfg(feature = "cuda-dkr-neutral-mmq")]
    neutral_mmq: dkr_kernel_registry::mmq_recipe::MmqRecipe,
    #[cfg(feature = "cuda-dkr-cubin-embedded")]
    cubin_images: std::collections::HashMap<dkr_kernel_registry::ArtifactKey, EmbeddedCubin>,
}

#[cfg(feature = "cuda-dkr-cubin-embedded")]
pub(super) struct EmbeddedCubin {
    pub(super) family: &'static str,
    pub(super) image: &'static [u8],
    pub(super) sha256: &'static str,
}

pub(super) fn family(index: usize) -> Option<&'static str> {
    [
        (kernels::AFFINE.index(), "affine"),
        (kernels::BINARY.index(), "binary"),
        (kernels::CAST.index(), "cast"),
        (kernels::CONV.index(), "conv"),
        (kernels::SORT.index(), "sort"),
        (kernels::FILL.index(), "fill"),
        (kernels::INDEXING.index(), "indexing"),
        (kernels::REDUCE.index(), "reduce"),
        (kernels::QUANTIZED.index(), "quantized"),
        (kernels::TERNARY.index(), "ternary"),
        (kernels::UNARY.index(), "unary"),
    ].into_iter().find_map(|(id, name)| (id == index).then_some(name))
}

impl ExternalModules {
    pub(super) fn from_env(context: &CudaContext) -> Result<Option<Self>> {
        let started = std::time::Instant::now();
        #[cfg(feature = "cuda-dkr-embedded")]
        {
            // Do not inspect or canonicalize the original directory at runtime.
            // The deployment binary cannot be redirected to filesystem packages.
            if let Some(source) = std::env::var_os("MISTRALRS_DKR_ARTIFACT_SOURCE") {
                if source != "embedded" {
                    crate::bail!("cuda-dkr-embedded requires embedded artifact source");
                }
            }
            if std::env::var_os("CANDLE_DKR_S31_DIR").is_some() {
                crate::bail!("cuda-dkr-embedded forbids CANDLE_DKR_S31_DIR runtime override");
            }
            if embedded::ROOT.is_none() {
                crate::bail!("cuda-dkr-embedded missing build-verified catalogue");
            }
        }
        let source = std::env::var("MISTRALRS_DKR_ARTIFACT_SOURCE")
            .unwrap_or_else(|_| if embedded::ROOT.is_some() { "embedded".into() } else { "filesystem".into() });
        let Some(root) = std::env::var_os("CANDLE_DKR_S31_DIR").or_else(|| {
            (source == "embedded").then_some(embedded::ROOT).flatten().map(std::ffi::OsString::from)
        }) else {
            return Ok(None);
        };
        let (major, minor) = context.compute_capability().w()?;
        let device_cc = (major * 10 + minor) as u32;
        let root = PathBuf::from(root);
        let trace = std::env::var_os("CANDLE_DKR_TRACE").is_some();
        let (recipe, fp8, fp4, bf16, routed, rows) = match source.as_str() {
            "embedded" => {
                let compiled_root = embedded::ROOT.ok_or_else(|| crate::Error::Msg("native artifacts were not embedded".into()))?;
                if root != Path::new(compiled_root) {
                    crate::bail!("native directory differs from the embedded build input");
                }
                let id = embedded::RECIPE.ok_or_else(|| crate::Error::Msg("missing embedded native launch recipe".into()))?;
                let recipe = NativeRecipe::from_id(id).map_err(crate::Error::Msg)?;
                let rows = embedded::modules().into_iter()
                    .map(|(name, artifact, symbols)| (name.to_owned(), artifact, symbols))
                    .collect::<Vec<_>>();
                #[cfg(feature = "cuda-dkr-neutral-recipes")]
                let (fp8, fp4, bf16, routed) = {
                    // These are build-verified, producer-owned descriptors.
                    // Do not require equality with the frozen S31 policy in
                    // the experimental catalogue-driven runtime.
                    if trace {
                        eprintln!("DKR_R124_S55B_RECIPES_CATALOG source=embedded families=fp8,fp4,bf16,routed target=sm{device_cc}");
                        #[cfg(feature = "cuda-dkr-cudaforge-catalog")]
                        eprintln!("DKR_R124_S55C_CATALOGUE_ADMITTED producer={} target=sm{} families=25 recipes=4 sha256={}",
                            cudaforge_catalogue::PRODUCER,
                            cudaforge_catalogue::TARGET_CC,
                            cudaforge_catalogue::ATTESTATION_SHA256);
                    }
                    (neutral_recipes::FP8,neutral_recipes::FP4,
                     neutral_recipes::BF16,neutral_recipes::ROUTED)
                };
                #[cfg(not(feature = "cuda-dkr-neutral-recipes"))]
                let (fp8, fp4, bf16, routed) = {
                    let fp8=embedded::FP8_RECIPE.ok_or_else(|| crate::Error::Msg("missing embedded FP8 recipe".into()))?;
                    if fp8!=recipe.fp8(){crate::bail!("embedded FP8 recipe differs from native ABI");}
                    let fp4=embedded::FP4_RECIPE.ok_or_else(||crate::Error::Msg("missing embedded FP4 recipe".into()))?;
                    if fp4!=recipe.fp4(){crate::bail!("embedded FP4 recipe differs from native ABI");}
                    let bf16=embedded::BF16_RECIPE.ok_or_else(||crate::Error::Msg("missing embedded BF16 recipe".into()))?;
                    if bf16!=recipe.bf16(){crate::bail!("embedded BF16 recipe differs from native ABI");}
                    let routed=embedded::ROUTED_RECIPE.ok_or_else(||crate::Error::Msg("missing embedded MMVQ/MoE recipe".into()))?;
                    if routed!=recipe.routed(){crate::bail!("embedded MMVQ/MoE recipe differs from native ABI");}
                    (fp8,fp4,bf16,routed)
                };
                (recipe, fp8, fp4, bf16,routed, rows)
            }
            #[cfg(not(feature = "cuda-dkr-embedded"))]
            "filesystem" => {
                if !root.is_absolute() || !root.is_dir() {
                    crate::bail!("CANDLE_DKR_S31_DIR must be an existing absolute directory");
                }
                let root = root.canonicalize().map_err(|e| crate::Error::Msg(e.to_string()))?;
                let catalog = NativeCatalog::load(&root).map_err(crate::Error::Msg)?;
                let mut rows = Vec::with_capacity(catalog.families.len());
                for name in &catalog.families {
                    let (artifact, symbols) = catalog.read_package(&root, name).map_err(crate::Error::Msg)?;
                    rows.push((name.clone(), artifact, symbols));
                }
                (catalog.recipe, catalog.recipe.fp8(), catalog.recipe.fp4(), catalog.recipe.bf16(),catalog.recipe.routed(), rows)
            }
            _ => crate::bail!("MISTRALRS_DKR_ARTIFACT_SOURCE must be embedded or filesystem"),
        };
        let mut modules = Registry::default();
        let mut native_keys = std::collections::HashMap::new();
        let mut mmq_symbols = std::collections::HashMap::new();
        for (name, artifact, symbols) in rows {
            artifact.admit_exact_target(device_cc).map_err(crate::Error::Msg)?;
            let key = RegistryKey::new(NATIVE_ABI, &name, "native", artifact.key().target_cc());
            modules.insert(key.clone(), artifact).map_err(crate::Error::Msg)?;
            native_keys.insert(name.clone(), key);
            if let Some(symbols) = symbols { mmq_symbols.insert(name, symbols); }
        }
        #[cfg(feature = "cuda-dkr-cubin-embedded")]
        let cubin_images = {
            if source != "embedded" {
                crate::bail!("CUBIN feature requires a sealed embedded provider");
            }
            let mut images = std::collections::HashMap::new();
            for (family, image, sha, target_cc, ptx_sha) in embedded::cubins() {
                let native_key = native_keys.get(family).ok_or_else(||
                    crate::Error::Msg(format!("CUBIN family not in native catalogue: {family}")))?;
                let original = modules.get(native_key).ok_or_else(||
                    crate::Error::Msg(format!("CUBIN missing PTX identity: {family}")))?;
                if original.key().target_cc() != target_cc
                    || original.key().sha256() != ptx_sha
                    || target_cc != device_cc
                {
                    crate::bail!("CUBIN target/PTX provenance mismatch: {family}");
                }
                if images.insert(original.key().clone(), EmbeddedCubin {
                    family, image, sha256: sha,
                }).is_some() {
                    crate::bail!("duplicate physical CUBIN module: {family}");
                }
            }
            if images.len() != modules.len() {
                crate::bail!("missing CUBIN images; refusing PTX fallback in strict CUBIN mode");
            }
            if trace {
                eprintln!("CANDLE_DKR_CUBIN_CATALOG source=embedded images={} target=sm{}", images.len(),device_cc);
            }
            images
        };
        #[cfg(feature = "cuda-dkr-neutral-mmq")]
        let neutral_mmq = {
            if source != "embedded" {
                crate::bail!("neutral MMQ requires a verified embedded catalogue");
            }
            let table = neutral_mmq::RECIPE;
            if trace {
                eprintln!("DKR_R124_S55A_MMQ_CATALOG source=embedded codecs={} target=sm{}",
                    table.codecs.len(),device_cc);
            }
            table
        };
        if trace {
            eprintln!("CANDLE_DKR_NATIVE_CATALOG source={source} manifests={}", modules.len());
            eprintln!("CANDLE_DKR_CATALOG_ADMISSION source={source} elapsed_us={}", started.elapsed().as_micros());
        }
        Ok(Some(Self { modules, native_keys, mmq_symbols, trace, device_cc, recipe, fp8, fp4, bf16,routed,
            #[cfg(feature = "cuda-dkr-neutral-mmq")]
            neutral_mmq,
            #[cfg(feature = "cuda-dkr-cubin-embedded")]
            cubin_images,
        }))
    }

    #[cfg(feature = "cuda-dkr-cubin-embedded")]
    pub(super) fn cubin_for(&self, artifact: &VerifiedArtifact) -> Option<&EmbeddedCubin> {
        self.cubin_images.get(artifact.key())
    }

    pub(super) fn routed_recipe(&self)->RoutedRecipe {self.routed}

    pub(super) fn bf16_recipe(&self) -> Bf16Recipe { self.bf16 }

    pub(super) fn fp4_recipe(&self) -> Fp4Recipe { self.fp4 }

    pub(super) fn fp8_recipe(&self) -> Fp8Recipe { self.fp8 }

    pub(super) fn mmq_configs(&self, codec: &str, rows: u32, batch: u32,
        padded: u32, warp: i32, limit: i32)
        -> Result<(String, cudarc::driver::LaunchConfig, cudarc::driver::LaunchConfig)>
    {
        use cudarc::driver::LaunchConfig;
        #[cfg(feature = "cuda-dkr-neutral-mmq")]
        let geometry = self.neutral_mmq.select(codec,rows,batch,padded,warp,limit)
            .map_err(|e|crate::Error::Msg(e.into()))?;
        #[cfg(not(feature = "cuda-dkr-neutral-mmq"))]
        let geometry = self.recipe.mmq_geometry(codec, rows, batch, padded, warp, limit)
            .map_err(crate::Error::Msg)?;
        #[cfg(feature = "cuda-dkr-neutral-mmq")]
        if self.trace {
            eprintln!("DKR_R124_S55A_MMQ_PLAN source=embedded codec={codec} tile={} shared_bytes={}",
                geometry.matrix_key,geometry.matrix_shared);
        }
        Ok((geometry.matrix_key,
            LaunchConfig { grid_dim: geometry.quant_grid, block_dim: geometry.quant_block, shared_mem_bytes: 0 },
            LaunchConfig { grid_dim: geometry.matrix_grid, block_dim: geometry.matrix_block,
                shared_mem_bytes: geometry.matrix_shared }))
    }

    fn artifact(&self, name: &str) -> Option<&VerifiedArtifact> {
        self.modules.get(self.native_keys.get(name)?)
    }

    pub(super) fn ptx(&self, index: usize) -> Option<&VerifiedArtifact> {
        family(index).and_then(|name| self.artifact(name))
    }

    pub(super) fn mmq_entry(&self, family: &str, key: &str) -> Option<(&VerifiedArtifact, &str)> {
        Some((self.artifact(family)?, self.mmq_symbols.get(family)?.get(key)?.as_str()))
    }

    pub(super) fn record_mmq_loaded(&self, family: &str) {
        if self.trace {
            eprintln!("CANDLE_DKR_MODULE_LOADED family={family} target=sm{} source=external", self.device_cc);
        }
    }

    pub(super) fn mmvq_ptx(&self) -> Option<&VerifiedArtifact> {
        self.artifact("mmvq_gguf")
    }

    pub(super) fn record_mmvq_loaded(&self) {
        if self.trace {
            eprintln!("CANDLE_DKR_MODULE_LOADED family=mmvq_gguf target=sm{} source=external", self.device_cc);
        }
    }

    pub(super) fn record_loaded(&self, index: usize) {
        if self.trace {
            if let Some(name) = family(index) {
                eprintln!("CANDLE_DKR_MODULE_LOADED family={name} target=sm{} source=external", self.device_cc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda-dkr-cudaforge-catalog")]
    #[test]
    fn dkr_s55c_cudaforge_catalogue_build_attestation() {
        assert_eq!(cudaforge_catalogue::PRODUCER, "cudaforge");
        assert_eq!(cudaforge_catalogue::TARGET_CC, contract::CC);
        assert_eq!(cudaforge_catalogue::ATTESTATION_SHA256.len(), 64);
        assert!(cudaforge_catalogue::ATTESTATION_SHA256.bytes()
            .all(|b| b.is_ascii_hexdigit()));
        assert_eq!(neutral_mmq::RECIPE.codecs.len(), 10);
        assert_eq!(neutral_recipes::FP8, fp8_recipe::S31_FP8);
        assert_eq!(neutral_recipes::FP4, fp4_recipe::S31_FP4);
        assert_eq!(neutral_recipes::BF16, bf16_recipe::S31_BF16);
        assert_eq!(neutral_recipes::ROUTED, routed_recipe::S31_ROUTED);
        println!("DKR_R124_S55C_ATTESTATION_TEST=PASS families=25 recipes=4 codecs=10");
    }

    #[cfg(feature = "cuda-dkr-neutral-recipes")]
    #[test]
    fn dkr_s55b_typed_recipes_match_qualified_s31() {
        // Frozen S31 values are an independent test oracle only.
        // The S5.5b runtime never uses these equalities to admit a recipe.
        let fp8=neutral_recipes::FP8;
        let fp4=neutral_recipes::FP4;
        let bf16=neutral_recipes::BF16;
        let routed=neutral_recipes::ROUTED;
        assert_eq!(fp8,fp8_recipe::S31_FP8);
        assert_eq!(fp4,fp4_recipe::S31_FP4);
        assert_eq!(bf16,bf16_recipe::S31_BF16);
        assert_eq!(routed,routed_recipe::S31_ROUTED);
        for tokens in [1,3,4,8,32] {
            for rows in [1,65,512] {
                for dp4a in [true,false] {
                    assert_eq!(fp8.matrix(dp4a,tokens,rows),
                        fp8_recipe::S31_FP8.matrix(dp4a,tokens,rows));
                }
            }
        }
        for n in [1,511,512,1024] {
            for batch in [1,8,9,17] {
                assert_eq!(fp4.native_matrix(61,false,n,batch),
                    fp4_recipe::S31_FP4.native_matrix(61,false,n,batch));
                assert_eq!(fp4.native_matrix(61,true,n,batch),
                    fp4_recipe::S31_FP4.native_matrix(61,true,n,batch));
            }
        }
        for batch in 1..=8 {
            let plan=routed.mmvq("q4_k","f16",1024,1024,129,batch).unwrap();
            let prior=routed_recipe::S31_ROUTED.mmvq("q4_k","f16",1024,1024,129,batch).unwrap();
            assert_eq!(plan.quant,prior.quant);
            assert_eq!(plan.matrix,prior.matrix);
            assert_eq!(plan.qcfg,prior.qcfg);
            assert_eq!(plan.mcfg,prior.mcfg);
        }
        for e in [1,32,33,128] {
            assert_eq!(routed.simt(e,129,1024,17).unwrap(),
                routed_recipe::S31_ROUTED.simt(e,129,1024,17).unwrap());
        }
        for codec in ["q8_0","q4_k","q2_k","q3_k","q5_k","q6_k"] {
            let x=routed.gguf(codec,129,1024,2,4).unwrap();
            let y=routed_recipe::S31_ROUTED.gguf(codec,129,1024,2,4).unwrap();
            assert_eq!(x.padded,y.padded);
            assert_eq!(x.scratch,y.scratch);
            assert_eq!(x.qcfg,y.qcfg);
            assert_eq!(x.mcfg,y.mcfg);
        }
        println!("DKR_R124_S55B_TYPED_PARITY=PASS fp8=1 fp4=1 bf16=1 routed=1");
    }


    #[cfg(feature = "cuda-dkr-embedded")]
    #[test]
    fn dkr_embedded_catalog_digest_inventory() {
        let rows = embedded::modules();
        assert!(embedded::ROOT.is_some());
        assert_eq!(rows.len(), contract::FAMILIES.len());
        let families: std::collections::BTreeSet<_> = rows.iter().map(|(name, _, _)| *name).collect();
        assert_eq!(families, contract::FAMILIES.iter().copied().collect());
        for (name, artifact, _) in rows {
            let actual = format!("{:x}", Sha256::digest(artifact.ptx().as_bytes()));
            assert_eq!(actual, artifact.key().sha256());
            println!("DKR_R124_S44_DIGEST family={name} sha256={actual}");
        }
    }

    #[test]
    fn dkr_routed_recipe_embedded_table(){
        match embedded::ROOT {
            Some(_) => {
                let recipe=NativeRecipe::from_id(embedded::RECIPE.unwrap()).unwrap();
                assert_eq!(embedded::ROUTED_RECIPE.unwrap(),recipe.routed());
                let rows=embedded::modules();
                let (_,artifact,_)=rows.iter().find(|(name,_,_)|*name=="mmvq_gguf").unwrap();
                for symbol in contract::mmvq_entries(){assert!(artifact.has_entry(&symbol));}
                for (family,count) in [("moe_simt_f16",3),("moe_gguf",7)] {
                    let (_,artifact,map)=rows.iter().find(|(name,_,_)|*name==family).unwrap();
                    let map=map.as_ref().unwrap(); assert_eq!(map.len(),count);
                    for symbol in map.values(){assert!(artifact.has_entry(symbol));}
                }
            }
            None=>assert!(embedded::ROUTED_RECIPE.is_none()),
        }
    }

    #[test]
    fn dkr_bf16_recipe_embedded_table() {
        match embedded::ROOT {
            Some(_) => {
                let recipe = NativeRecipe::from_id(embedded::RECIPE.unwrap()).unwrap();
                let bf16 = embedded::BF16_RECIPE.unwrap();
                assert_eq!(bf16, recipe.bf16());
                let rows = embedded::modules();
                let (_, artifact, _) = rows.iter().find(|(name, _, _)| *name == "cast").unwrap();
                for symbol in bf16.cast_symbols() {
                    assert!(artifact.has_entry(symbol), "missing embedded BF16 entry {symbol}");
                }
            }
            None => assert!(embedded::BF16_RECIPE.is_none()),
        }
    }

    #[test]
    fn dkr_fp4_recipe_embedded_table() {
        match embedded::ROOT {
            Some(_) => {
                let recipe = NativeRecipe::from_id(embedded::RECIPE.unwrap()).unwrap();
                let fp4 = embedded::FP4_RECIPE.unwrap();
                assert_eq!(fp4, recipe.fp4());
                let rows = embedded::modules();
                let (_, artifact, _) = rows.iter().find(|(name, _, _)| *name == "quantized").unwrap();
                for symbol in fp4.symbols() { assert!(artifact.has_entry(symbol), "missing FP4 entry {symbol}"); }
            }
            None => assert!(embedded::FP4_RECIPE.is_none()),
        }
    }

    #[test]
    fn dkr_fp8_recipe_embedded_table() {
        match embedded::ROOT {
            Some(_) => {
                let recipe = NativeRecipe::from_id(embedded::RECIPE.unwrap()).unwrap();
                let fp8 = embedded::FP8_RECIPE.unwrap();
                assert_eq!(fp8, recipe.fp8());
                let rows = embedded::modules();
                let (_, artifact, _) = rows.iter().find(|(name, _, _)| *name == "quantized").unwrap();
                for symbol in fp8.symbols() {
                    assert!(artifact.has_entry(symbol), "missing embedded FP8 entry {symbol}");
                }
            }
            None => {
                assert!(embedded::RECIPE.is_none());
                assert!(embedded::FP8_RECIPE.is_none());
                assert!(embedded::modules().is_empty());
            }
        }
    }

    #[cfg(feature = "cuda-dkr-neutral-mmq")]
    #[test]
    fn dkr_s55a_neutral_mmq_matches_frozen_s31_qualified_shapes() {
        let neutral = neutral_mmq::RECIPE;
        assert_eq!(neutral.codecs.len(), contract::MMVQ_CODECS.len());
        for codec in contract::MMVQ_CODECS {
            for rows in [1, 64, 65, 129, 1024] {
                for batch in [1, 8, 9, 16, 17, 64, 65, 1024] {
                    let frozen = NativeRecipe::S31NativeV1
                        .mmq_geometry(codec,rows,batch,1024,32,49152).unwrap();
                    let plan = neutral.select(codec,rows,batch,1024,32,49152).unwrap();
                    assert_eq!(plan.matrix_key,frozen.matrix_key, "{codec}: {rows}/{batch}");
                    assert_eq!(plan.matrix_grid,frozen.matrix_grid);
                    assert_eq!(plan.quant_grid,frozen.quant_grid);
                    assert_eq!(plan.quant_block,frozen.quant_block);
                    assert_eq!(plan.matrix_block,frozen.matrix_block);
                    assert_eq!(plan.matrix_shared,frozen.matrix_shared);
                }
            }
        }
        assert!(neutral.select("q4_k",64,9,0,32,49152).is_err());
        assert!(neutral.select("unknown",64,9,1024,32,49152).is_err());
        assert!(neutral.select("q4_k",64,9,1024,16,49152).is_err());
        println!("DKR_R124_S55A_TABLE_PARITY=PASS codecs=10");
    }

    #[test]
    fn dkr_s31_manifest_contract() {
        let text = ".version 8.0\n.target sm_61\n.address_size 64\n.visible .entry sample() { ret; }\n";
        let mut m = serde_json::json!({
            "schema": contract::SCHEMA, "base_revision": contract::BASE,
            "target_cc": 61, "family": "unary", "flags": contract::FLAGS,
            "sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
        });
        assert!(validate(&m, "unary", text).is_ok());
        let (artifact, symbols) = verify_package(&m, "unary", text.to_owned()).unwrap();
        assert_eq!(artifact.key().target_cc(), 61);
        assert!(artifact.admit_exact_target(61).is_ok());
        assert!(artifact.admit_exact_target(80).is_err());
        assert!(symbols.is_none());
        let mut wrong_target = m.clone();
        wrong_target["target_cc"] = serde_json::json!(80);
        assert!(verify_package(&wrong_target, "unary", text.to_owned()).is_err());
        assert!(validate(&m, "cast", text).is_err());
        assert!(validate(&m, "unary", &text.replace("sm_61", "sm_80")).is_err());
        m["sha256"] = serde_json::json!("bad");
        assert!(validate(&m, "unary", text).is_err());
        assert_eq!(family(kernels::UNARY.index()), Some("unary"));
        assert_eq!(family(kernels::REDUCE.index()), Some("reduce"));
        assert_eq!(family(kernels::INDEXING.index()), Some("indexing"));
        assert_eq!(family(kernels::QUANTIZED.index()), Some("quantized"));
        assert_eq!(family(kernels::CONV.index()), Some("conv"));
        assert_eq!(family(kernels::SORT.index()), Some("sort"));
        assert_eq!(contract::mmvq_entries().len(), 243);
    }
}
