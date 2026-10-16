//! Architecture-independent E4M3 SafeTensors inventory and CUDA construction.
//!
//! Unlike the historical `from_safetensors` convenience constructor, this loader
//! deserializes EACH shard ONCE for the entire build, never once per projection.
//! Every FP8 weight with an explicit `weight_scale_inv` is inventoried before
//! the global dispatch is evaluated or any weights are uploaded to the GPU.
//!
//! Memory figures are weight-residency estimates, not process-level CUDA peaks.
//! The caller must supply a device-wide memory snapshot and KV/workspace/headroom
//! reservations; this module cannot guarantee CUDA allocation success.
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use super::fp8_linear_sm61::Fp8Sm61Linear;
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use super::fp8_sm61_dispatch::{dispatch_residence, dispatch_uniform_residence};
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use super::fp8_sm61_dispatch::{Fp8Backend, Fp8DispatchPlan};
use super::fp8_sm61_dispatch::{
    Fp8DispatchBudget, Fp8LinearSpec, Fp8MeasuredProfile, Fp8WorkloadProfile,
};
use super::fp8_sm61_format::decode_f8e_inverse_scales;
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use crate::CudaDevice;
use crate::Result;
use safetensors::SafeTensors;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fp8DispatchStrategy {
    /// No workload-wide profile => stay entirely on SIMT.
    Uniform(Option<Fp8WorkloadProfile>),
    /// Apply shape profiles, independent of projection identifiers.
    ByGeometry,
}

#[derive(Clone, Debug)]
pub struct Fp8LoadRequest {
    pub budget: Fp8DispatchBudget,
    pub strategy: Fp8DispatchStrategy,
    pub expected_calls_per_projection: u64,
    /// Optional measurements indexed ONLY by (N, K, block-N, block-K).
    pub geometry_profiles: BTreeMap<(usize, usize, usize, usize), Fp8MeasuredProfile>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fp8TensorDescriptor {
    /// Names are used for locating tensors and diagnostic output, NEVER dispatch.
    pub identifier: String,
    pub out_features: usize,
    pub in_features: usize,
    pub block_n: usize,
    pub block_k: usize,
    pub scale_dtype: String,
}

impl Fp8TensorDescriptor {
    #[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
    fn spec(&self, request: &Fp8LoadRequest) -> Fp8LinearSpec {
        let profile = request
            .geometry_profiles
            .get(&(
                self.out_features,
                self.in_features,
                self.block_n,
                self.block_k,
            ))
            .copied();
        Fp8LinearSpec {
            out_features: self.out_features,
            in_features: self.in_features,
            block_n: self.block_n,
            block_k: self.block_k,
            resident_calls: request.expected_calls_per_projection,
            profile,
        }
    }
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
pub struct Fp8ResidentProjection {
    pub identifier: String,
    pub linear: Fp8Sm61Linear,
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
pub struct Fp8ResidentCollection {
    pub projections: Vec<Fp8ResidentProjection>,
    pub dispatch: Fp8DispatchPlan,
    pub input_shards: usize,
    pub realized_weight_bytes: usize,
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
impl Fp8ResidentCollection {
    pub fn get(&self, identifier: &str) -> Option<&Fp8Sm61Linear> {
        self.projections
            .iter()
            .find(|entry| entry.identifier == identifier)
            .map(|entry| &entry.linear)
    }
}

fn load_shards(paths: &[PathBuf]) -> Result<Vec<Vec<u8>>> {
    if paths.is_empty() {
        crate::bail!("F8E requires at least one SafeTensors shard");
    }
    let mut uniq = BTreeSet::new();
    let mut bytes = Vec::with_capacity(paths.len());
    for path in paths {
        // Identical names/paths can otherwise hide duplicated weight storage.
        let canonical = path.canonicalize()?;
        if !uniq.insert(canonical) {
            crate::bail!("F8E duplicate shard path");
        }
        bytes.push(std::fs::read(path)?);
    }
    Ok(bytes)
}

/// Parse each shard exactly once per call; the returned views borrow `buffers`.
fn parse_shards<'a>(buffers: &'a [Vec<u8>]) -> Result<Vec<SafeTensors<'a>>> {
    buffers
        .iter()
        .map(|bytes| {
            SafeTensors::deserialize(bytes).map_err(|e| {
                crate::Error::Msg(format!("F8E invalid SafeTensors shard: {e}").into())
            })
        })
        .collect()
}

#[derive(Clone)]
#[cfg_attr(
    not(all(feature = "cuda", feature = "cuda-legacy-fp8")),
    allow(dead_code)
)]
struct LocatedDescriptor {
    meta: Fp8TensorDescriptor,
    weight_shard: usize,
    scale_shard: usize,
}

/// Inventory tensors and validate all supported scale layouts without
/// architecture-specific allowlists or identifier-based dispatch.
fn index_tensors(shards: &[SafeTensors<'_>]) -> Result<Vec<LocatedDescriptor>> {
    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    for (shard_idx, shard) in shards.iter().enumerate() {
        for name in shard.names() {
            if index.insert(name.to_string(), shard_idx).is_some() {
                crate::bail!("F8E duplicate tensor name across shards: {}", name);
            }
        }
    }
    let mut descriptors = Vec::new();
    for (name, &weight_shard) in &index {
        if !name.ends_with(".weight") {
            continue;
        }
        let wt = shards[weight_shard]
            .tensor(name)
            .map_err(|e| crate::Error::Msg(format!("F8E weight view: {e}").into()))?;
        let dt = format!("{:?}", wt.dtype());
        if dt != "F8_E4M3" && dt != "F8_E4M3FN" {
            continue;
        }
        if wt.shape().len() != 2 {
            crate::bail!("F8E FP8 weight must be rank 2: {}", name);
        }
        let n = wt.shape()[0];
        let k = wt.shape()[1];
        if n == 0 || k == 0 {
            crate::bail!("F8E empty FP8 projection: {name}");
        }
        let suffix = name.strip_suffix(".weight").expect("checked above");
        let scale_name = format!("{suffix}.weight_scale_inv");
        let &scale_shard = index.get(&scale_name).ok_or_else(|| {
            crate::Error::Msg(format!("F8E FP8 projection without inverse scales: {name}").into())
        })?;
        let scale = shards[scale_shard]
            .tensor(&scale_name)
            .map_err(|e| crate::Error::Msg(format!("F8E scale view: {e}").into()))?;
        let (_, bn, bk) = decode_f8e_inverse_scales(n, k, &scale)?;
        // Reject malformed geometry before CUDA allocation.
        super::fp8_sm61_dispatch::required_bytes(&Fp8LinearSpec {
            out_features: n,
            in_features: k,
            block_n: bn,
            block_k: bk,
            resident_calls: 0,
            profile: None,
        })
        .map_err(|e| crate::Error::Msg(format!("F8E dispatch geometry: {e}").into()))?;
        descriptors.push(LocatedDescriptor {
            meta: Fp8TensorDescriptor {
                identifier: suffix.to_owned(),
                out_features: n,
                in_features: k,
                block_n: bn,
                block_k: bk,
                scale_dtype: format!("{:?}", scale.dtype()),
            },
            weight_shard,
            scale_shard,
        });
    }
    if descriptors.is_empty() {
        crate::bail!("F8E shards contain no supported E4M3 inverse-scale projections");
    }
    Ok(descriptors)
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
fn choose_dispatch(index: &[LocatedDescriptor], req: &Fp8LoadRequest) -> Result<Fp8DispatchPlan> {
    let specs: Vec<_> = index.iter().map(|x| x.meta.spec(req)).collect();
    let result = match req.strategy {
        Fp8DispatchStrategy::Uniform(profile) => {
            dispatch_uniform_residence(&specs, req.budget, profile)
        }
        Fp8DispatchStrategy::ByGeometry => dispatch_residence(&specs, req.budget),
    };
    result.map_err(|e| crate::Error::Msg(format!("F8E dispatch failed: {e}").into()))
}

/// CPU-side inventory, usable for arbitrary producer names and shard ordering.
/// This is diagnostic-only; load_cuda revalidates when it reads its own shards.
pub fn inspect_safetensors(paths: &[PathBuf]) -> Result<Vec<Fp8TensorDescriptor>> {
    let buffers = load_shards(paths)?;
    let shards = parse_shards(&buffers)?;
    Ok(index_tensors(&shards)?
        .into_iter()
        .map(|x| x.meta)
        .collect())
}

/// Build a whole collection. No live switch, no CUDA fallback on allocation
/// failure, and no duplicate device representation per projection.
/// `request.budget.free_device_bytes` must be a conservative caller-measured
/// snapshot from BEFORE this call, not an invented value.
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
pub fn load_cuda(
    paths: &[PathBuf],
    device: &CudaDevice,
    request: &Fp8LoadRequest,
) -> Result<Fp8ResidentCollection> {
    let buffers = load_shards(paths)?;
    let shards = parse_shards(&buffers)?;
    let index = index_tensors(&shards)?;
    let dispatch = choose_dispatch(&index, request)?;
    if dispatch.entries.len() != index.len() || dispatch.projection_count != index.len() {
        crate::bail!("F8E dispatch count mismatch");
    }
    let mut projections = Vec::with_capacity(index.len());
    let mut actual_total = 0usize;
    for (item, decision) in index.iter().zip(&dispatch.entries) {
        let wt = shards[item.weight_shard]
            .tensor(&format!("{}.weight", item.meta.identifier))
            .map_err(|e| crate::Error::Msg(format!("F8E weight view: {e}").into()))?;
        let scale = shards[item.scale_shard]
            .tensor(&format!("{}.weight_scale_inv", item.meta.identifier))
            .map_err(|e| crate::Error::Msg(format!("F8E scale view: {e}").into()))?;
        let (scales, bn, bk) =
            decode_f8e_inverse_scales(item.meta.out_features, item.meta.in_features, &scale)?;
        if (bn, bk) != (item.meta.block_n, item.meta.block_k) {
            crate::bail!("F8E scale shape changed after inventory");
        }
        let linear = Fp8Sm61Linear::from_bytes_with_dispatch(
            device,
            item.meta.out_features,
            item.meta.in_features,
            wt.data(),
            &scales,
            bn,
            bk,
            decision,
        )?;
        let expected = match decision.backend {
            Fp8Backend::SimtW8A16 => decision.simt_bytes,
            Fp8Backend::ExpandedF16 => decision.expanded_f16_bytes,
        };
        if linear.resident_weight_bytes() != expected || expected != decision.resident_bytes {
            crate::bail!("F8E realized backend residency mismatch");
        }
        actual_total = actual_total
            .checked_add(expected)
            .ok_or_else(|| crate::Error::Msg("F8E resident byte counter overflow".into()))?;
        if actual_total > dispatch.budget_for_fp8_bytes {
            crate::bail!("F8E resident bytes exceed global budget");
        }
        projections.push(Fp8ResidentProjection {
            identifier: item.meta.identifier.clone(),
            linear,
        });
    }
    if actual_total != dispatch.selected_total_bytes {
        crate::bail!("F8E global resident-byte mismatch");
    }
    Ok(Fp8ResidentCollection {
        projections,
        dispatch,
        input_shards: shards.len(),
        realized_weight_bytes: actual_total,
    })
}

/// Synchronized device-wide CUDA memory counter (NOT a process allocator peak).
/// Driver-reported free memory can change due to other GPU processes; all
/// numbers are observations, not exact exclusive allocations by this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8CudaMemorySample {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

/// Uses the same CUDA context and stream as our Candle allocations.
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
pub fn sample_cuda_memory(device: &CudaDevice) -> Result<Fp8CudaMemorySample> {
    use crate::cuda_backend::WrapErr;
    let stream = device.cuda_stream();
    stream.synchronize().w()?;
    let (free_bytes, total_bytes) = stream.context().mem_get_info().w()?;
    if total_bytes == 0 || free_bytes > total_bytes {
        crate::bail!("F8E driver reported invalid CUDA memory counters");
    }
    Ok(Fp8CudaMemorySample {
        free_bytes,
        total_bytes,
    })
}

/// Operational configuration for the native loader. The device-wide free
/// memory is intentionally NOT passed by the application: it is sampled from
/// the actual CUDA context immediately before inventory and allocation.
#[derive(Clone, Debug)]
pub struct Fp8NativeDispatchConfig {
    pub workload_reserve_bytes: usize,
    pub safety_headroom_bytes: usize,
    pub min_steady_speedup: f64,
    pub allow_auto_f16: bool,
    pub strategy: Fp8DispatchStrategy,
    pub expected_calls_per_projection: u64,
    pub geometry_profiles: BTreeMap<(usize, usize, usize, usize), Fp8MeasuredProfile>,
}

impl Default for Fp8NativeDispatchConfig {
    fn default() -> Self {
        Self {
            workload_reserve_bytes: 0,
            // Default headroom is deliberately conservative for SM61.
            safety_headroom_bytes: 256 * 1024 * 1024,
            min_steady_speedup: 1.03,
            allow_auto_f16: false,
            strategy: Fp8DispatchStrategy::Uniform(None),
            expected_calls_per_projection: 0,
            geometry_profiles: BTreeMap::new(),
        }
    }
}

/// Auditable physical-memory evidence. `accounted_weight_bytes` is the exact
/// sum of underlying resident weight/scales buffer lengths, whereas
/// `observed_driver_used_delta_bytes` is a signed *device-wide* observation.
/// Do not equate the latter to process VRAM, an instantaneous peak, or a
/// measurement of temporary loading allocations.
#[derive(Clone, Debug)]
pub struct Fp8NativeResidenceEvidence {
    pub before: Fp8CudaMemorySample,
    pub after: Fp8CudaMemorySample,
    pub observed_driver_used_delta_bytes: i128,
    pub accounted_weight_bytes: usize,
    pub planned_weight_bytes: usize,
    pub model_wide_f16_projections: usize,
    pub model_wide_simt_projections: usize,
    pub model_wide_weight_budget_bytes: usize,
    pub requested_reserve_bytes: usize,
    pub requested_headroom_bytes: usize,
    /// The enum representation ensures there cannot be concurrent FP8+F16
    /// buffers belonging to the same projection in this resident collection.
    pub single_representation_per_projection: bool,
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
pub struct Fp8NativeResidentModel {
    pub weights: Fp8ResidentCollection,
    pub evidence: Fp8NativeResidenceEvidence,
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
impl Fp8NativeResidentModel {
    /// Native Candle entry point: no model-architecture-dependent adapter.
    pub fn linear(&self, identifier: &str) -> Option<&Fp8Sm61Linear> {
        self.weights.get(identifier)
    }
    pub fn forward(&self, identifier: &str, input: &crate::Tensor) -> Result<crate::Tensor> {
        let linear = self.linear(identifier).ok_or_else(|| {
            crate::Error::Msg(format!("F8E unknown FP8 projection {identifier}").into())
        })?;
        linear.forward(input)
    }
}

/// Genuine native CUDA residency admission. Takes its own pre-load memory
/// sample, computes dispatch globally before weight allocation, and refuses to
/// return a loaded model if the user reserves are not available after loading.
/// No OOM-triggered backend switching, no silent CPU/DP4A fallback.
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
pub fn load_cuda_native(
    paths: &[PathBuf],
    device: &CudaDevice,
    config: &Fp8NativeDispatchConfig,
) -> Result<Fp8NativeResidentModel> {
    let before = sample_cuda_memory(device)?;
    let budget = Fp8DispatchBudget {
        free_device_bytes: before.free_bytes,
        workload_reserve_bytes: config.workload_reserve_bytes,
        safety_headroom_bytes: config.safety_headroom_bytes,
        min_steady_speedup: config.min_steady_speedup,
        allow_auto_f16: config.allow_auto_f16,
    };
    let request = Fp8LoadRequest {
        budget,
        strategy: config.strategy,
        expected_calls_per_projection: config.expected_calls_per_projection,
        geometry_profiles: config.geometry_profiles.clone(),
    };
    let weights = load_cuda(paths, device, &request)?;
    let after = sample_cuda_memory(device)?;
    let reserved = config
        .workload_reserve_bytes
        .checked_add(config.safety_headroom_bytes)
        .ok_or_else(|| crate::Error::Msg("F8E native reserves overflow".into()))?;
    if after.free_bytes < reserved {
        crate::bail!("F8E native CUDA post-load free memory below reserved budget");
    }
    let f16 = weights.dispatch.expanded_f16_count;
    let simt = weights
        .projections
        .len()
        .checked_sub(f16)
        .ok_or_else(|| crate::Error::Msg("F8E invalid backend counts".into()))?;
    if weights.dispatch.projection_count != weights.projections.len()
        || f16 + simt != weights.projections.len()
        || weights.realized_weight_bytes != weights.dispatch.selected_total_bytes
    {
        crate::bail!("F8E native resident allocation accounting mismatch");
    }
    let evidence = Fp8NativeResidenceEvidence {
        before,
        after,
        observed_driver_used_delta_bytes: before.free_bytes as i128 - after.free_bytes as i128,
        accounted_weight_bytes: weights.realized_weight_bytes,
        planned_weight_bytes: weights.dispatch.selected_total_bytes,
        model_wide_f16_projections: f16,
        model_wide_simt_projections: simt,
        model_wide_weight_budget_bytes: weights.dispatch.budget_for_fp8_bytes,
        requested_reserve_bytes: config.workload_reserve_bytes,
        requested_headroom_bytes: config.safety_headroom_bytes,
        single_representation_per_projection: true,
    };
    Ok(Fp8NativeResidentModel { weights, evidence })
}
