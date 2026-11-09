//! ARCH-E7-B: real tiny GLM-MoE-DSA *complete causal prefill* (short sequence) with
//! original ModelOpt tensors. All model layers, MLA, MoE router/experts, and lm_head.
//! Sparse indexer shortcut is mathematically exact only if seqlen <= index_topk:
//! topk then selects all possible past keys, and the causal mask alone suffices.
//! No KV-cache, no autoregressive decode. Routed expert execution is dense-eval,
//! validation-only, with routing decision readback for audit, NOT optimized MoE.
#![cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
use candle_core::quantized::nvfp4_arch_e7::{E7Mode, Nvfp4E7Linear};
use candle_core::{CudaDevice, DType, Device, Result, Tensor};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

fn fail<T>(why: impl std::fmt::Display) -> Result<T> {
    candle_core::bail!("ARCH-E7B: {why}")
}
fn read_json(p: &Path) -> Result<Value> {
    let content = std::fs::read_to_string(p)?;
    Ok(serde_json::from_str(&content)
        .map_err(|e| candle_core::Error::Msg(format!("json {}: {e}", p.display())))?)
}
fn config_usize(c: &Value, key: &str) -> Result<usize> {
    c.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing/invalid config {key}")))
}
fn cfg_float(c: &Value, key: &str, default: f64) -> f64 {
    c.get(key).and_then(Value::as_f64).unwrap_or(default)
}
fn get_vec_f32(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len()
        != count
            .checked_mul(4)
            .ok_or_else(|| candle_core::Error::Msg("overflow".into()))?
    {
        return fail(format!("wrong weight bytes for {}", path.display()));
    }
    let mut out = Vec::with_capacity(count);
    for c in bytes.chunks_exact(4) {
        let x = f32::from_le_bytes(c.try_into().unwrap());
        if !x.is_finite() {
            return fail("nonfinite weight");
        };
        out.push(x)
    }
    Ok(out)
}
fn stats(a: &[f32], b: &[f32]) -> (f64, f64, f64, usize) {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut se = 0f64;
    let mut max = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        let d = (x - y).abs();
        max = max.max(d);
        se += d * d;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let cos = if na == 0.0 && nb == 0.0 {
        1.0
    } else {
        dot / (na.sqrt() * nb.sqrt()).max(1e-50)
    };
    (max, (se / a.len() as f64).sqrt(), cos, 0)
}
fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut f = File::create(path)?;
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

struct TinyGlm {
    w: HashMap<String, Tensor>,
    q: HashMap<String, Nvfp4E7Linear>,
    config: Value,
    dev: Device,
    dp4a_calls: usize,
    quant_calls: usize,
    routes: Vec<String>,
}
impl TinyGlm {
    fn load(root: &Path, cuda: &CudaDevice) -> Result<Self> {
        let man = read_json(&root.join("weights-manifest.json"))?;
        if man["schema"] != "haricot.arch-e7b.model-weights.v1" {
            return fail("wrong model assets schema");
        }
        let cfg = read_json(&root.join("config.json"))?;
        if cfg["model_type"] != "glm_moe_dsa" {
            return fail("requires glm_moe_dsa config");
        }
        let rp = cfg
            .get("rope_parameters")
            .and_then(|x| x.get("rope_type"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        if rp != "default" {
            return fail("nondefault RoPE unsupported, refuse logits claim");
        }
        if cfg
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return fail("attention_bias unsupported");
        }
        let ng = config_usize(&cfg, "n_group")?;
        if ng != 1 {
            return fail("router n_group > 1 unsupported");
        }
        let dev = Device::Cuda(cuda.clone());
        let mut w = HashMap::new();
        let mut q = HashMap::new();
        for item in man["float_tensors"]
            .as_array()
            .ok_or_else(|| candle_core::Error::Msg("float tensors absent".into()))?
        {
            let key = item["key"]
                .as_str()
                .ok_or_else(|| candle_core::Error::Msg("missing key".into()))?;
            let file = item["file"]
                .as_str()
                .ok_or_else(|| candle_core::Error::Msg("missing file".into()))?;
            let dims: Vec<usize> = item["shape"]
                .as_array()
                .ok_or_else(|| candle_core::Error::Msg("missing shape".into()))?
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| candle_core::Error::Msg("bad shape".into()))?;
            let count = dims
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .ok_or_else(|| candle_core::Error::Msg("shape overflow".into()))?;
            let vec = get_vec_f32(&root.join("float").join(file), count)?;
            let tensor = Tensor::from_vec(vec, candle_core::Shape::from(dims), &dev)?;
            w.insert(key.into(), tensor);
        }
        for item in man["quant_tensors"]
            .as_array()
            .ok_or_else(|| candle_core::Error::Msg("quant tensors absent".into()))?
        {
            let key = item["prefix"]
                .as_str()
                .ok_or_else(|| candle_core::Error::Msg("missing quant key".into()))?;
            let subdir = item["directory"]
                .as_str()
                .ok_or_else(|| candle_core::Error::Msg("missing quant directory".into()))?;
            let (n, k) = (
                item["n"].as_u64().unwrap() as usize,
                item["k"].as_u64().unwrap() as usize,
            );
            let d = root.join("quant").join(subdir);
            let weights = std::fs::read(d.join("packed.bin"))?;
            let scales = std::fs::read(d.join("scales.bin"))?;
            let gs = std::fs::read_to_string(d.join("global_scale.txt"))?
                .trim()
                .parse::<f32>()
                .map_err(|e| candle_core::Error::Msg(format!("global scale: {e}")))?;
            q.insert(
                key.into(),
                Nvfp4E7Linear::from_modelopt_bytes(cuda, n, k, &weights, &scales, gs)?,
            );
        }
        if q.is_empty() {
            return fail("no actual NVFP4 projections loaded");
        }
        Ok(Self {
            w,
            q,
            config: cfg,
            dev,
            dp4a_calls: 0,
            quant_calls: 0,
            routes: vec![],
        })
    }
    fn weight(&self, key: &str) -> Result<&Tensor> {
        self.w
            .get(key)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing model weight: {key}")))
    }
    fn linear(&mut self, x: &Tensor, prefix: &str, mode: E7Mode) -> Result<Tensor> {
        if let Some(q) = self.q.get(prefix) {
            self.quant_calls += 1;
            if mode == E7Mode::W4A8Dp4a {
                self.dp4a_calls += 1
            }
            let input = x.to_dtype(DType::F16)?;
            return q.forward(&input, mode);
        }
        let name = format!("{prefix}.weight");
        let weight = self.weight(&name)?;
        if weight.rank() != 2 {
            return fail(format!("linear weight rank incorrect {name}"));
        }
        x.matmul(&weight.t()?)
    }
    fn rms(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        // HF GLM uses eps=1e-6 for MLA q_a/kv_a RMSNorm, config.rms_norm_eps for model layers.
        let eps = if prefix.ends_with(".q_a_layernorm") || prefix.ends_with(".kv_a_layernorm") {
            1e-6
        } else {
            cfg_float(&self.config, "rms_norm_eps", 1e-5)
        };
        let a = x.sqr()?.mean_keepdim(1)?;
        let inv = (a + eps)?.powf(-0.5)?;
        x.broadcast_mul(&inv)?
            .broadcast_mul(self.weight(&format!("{prefix}.weight"))?)
    }
    fn swish(&self, g: &Tensor) -> Result<Tensor> {
        let exp = g.neg()?.exp()?;
        let d = (exp + 1.0)?;
        g.broadcast_div(&d)
    }
    fn mlp(&mut self, x: &Tensor, root: &str, mode: E7Mode) -> Result<Tensor> {
        let gate = self.linear(x, &format!("{root}.gate_proj"), mode)?;
        let up = self.linear(x, &format!("{root}.up_proj"), mode)?;
        let prod = self.swish(&gate)?.broadcast_mul(&up)?;
        self.linear(&prod, &format!("{root}.down_proj"), mode)
    }
    fn rotate(&self, x: &Tensor, positions: usize, heads: usize, dim: usize) -> Result<Tensor> {
        let theta = self.config["rope_parameters"]["rope_theta"]
            .as_f64()
            .unwrap_or(10_000.0);
        if dim % 2 != 0 {
            return fail("odd rotary dimension");
        }
        let mut cv = Vec::with_capacity(positions * dim / 2);
        let mut sv = Vec::with_capacity(positions * dim / 2);
        for p in 0..positions {
            for j in 0..dim / 2 {
                let a = (p as f64) / theta.powf((2 * j) as f64 / dim as f64);
                cv.push(a.cos() as f32);
                sv.push(a.sin() as f32);
            }
        }
        let cos = Tensor::from_vec(cv, (positions, 1, dim / 2, 1), &self.dev)?;
        let sin = Tensor::from_vec(sv, (positions, 1, dim / 2, 1), &self.dev)?;
        let evenodd = x.reshape((positions, heads, dim / 2, 2))?;
        let even = evenodd.narrow(3, 0, 1)?;
        let odd = evenodd.narrow(3, 1, 1)?;
        let e = (&even.broadcast_mul(&cos)? - &odd.broadcast_mul(&sin)?)?.reshape((
            positions,
            heads,
            dim / 2,
        ))?;
        let o = (&odd.broadcast_mul(&cos)? + &even.broadcast_mul(&sin)?)?.reshape((
            positions,
            heads,
            dim / 2,
        ))?;
        // Hugging Face apply_rotary_pos_emb_interleave concatenates both halves,
        // not stack-interleaves them. This ordering is essential for logits parity.
        Tensor::cat(&[e, o], 2)
    }
    fn attention(&mut self, x: &Tensor, layer: usize, mode: E7Mode) -> Result<Tensor> {
        let seq = x.dims()[0];
        let heads = config_usize(&self.config, "num_attention_heads")?;
        let qrank = config_usize(&self.config, "q_lora_rank")?;
        let kv_rank = config_usize(&self.config, "kv_lora_rank")?;
        let rdim = config_usize(&self.config, "qk_rope_head_dim")?;
        let nd = config_usize(&self.config, "qk_nope_head_dim")?;
        let vd = config_usize(&self.config, "v_head_dim")?;
        let rt = format!("model.layers.{layer}.self_attn");
        let qa = self.linear(x, &format!("{rt}.q_a_proj"), mode)?;
        if qa.dims() != [seq, qrank] {
            return fail("q_a shape mismatch");
        }
        let qa = self.rms(&qa, &format!("{rt}.q_a_layernorm"))?;
        let q = self
            .linear(&qa, &format!("{rt}.q_b_proj"), mode)?
            .reshape((seq, heads, nd + rdim))?;
        let qpass = q.narrow(2, 0, nd)?;
        let qrot = self.rotate(&q.narrow(2, nd, rdim)?, seq, heads, rdim)?;
        let q = Tensor::cat(&[qpass, qrot], 2)?;
        let kv = self.linear(x, &format!("{rt}.kv_a_proj_with_mqa"), mode)?;
        let kvpass = self.rms(&kv.narrow(1, 0, kv_rank)?, &format!("{rt}.kv_a_layernorm"))?;
        let krot = self.rotate(
            &kv.narrow(1, kv_rank, rdim)?.reshape((seq, 1, rdim))?,
            seq,
            1,
            rdim,
        )?;
        let expand = self
            .linear(&kvpass, &format!("{rt}.kv_b_proj"), mode)?
            .reshape((seq, heads, nd + vd))?;
        let knope = expand.narrow(2, 0, nd)?;
        let v = expand.narrow(2, nd, vd)?.transpose(0, 1)?.contiguous()?;
        let k = Tensor::cat(&[knope, krot.broadcast_as((seq, heads, rdim))?], 2)?;
        // Candle CUDA batched matmul requires fully contiguous row-major operands.
        // Transpose only creates strided views: [H,S,D] and [H,D,S] must be
        // materialized before calling cuBLAS. No change to MLA's arithmetic.
        let q_batched = q.transpose(0, 1)?.contiguous()?;
        let kt_batched = k.transpose(0, 1)?.transpose(1, 2)?.contiguous()?;
        let scores = q_batched.matmul(&kt_batched)?;
        let scale = 1.0 / ((nd + rdim) as f64).sqrt();
        let scores = (scores * scale)?;
        let mut cm = Vec::with_capacity(seq * seq);
        for i in 0..seq {
            for j in 0..seq {
                cm.push(if j > i { -1e9f32 } else { 0.0f32 });
            }
        }
        let mask = Tensor::from_vec(cm, (1, seq, seq), &self.dev)?;
        let scores = scores.broadcast_add(&mask)?;
        let ex = scores.broadcast_sub(&scores.max_keepdim(2)?)?.exp()?;
        let p = ex.broadcast_div(&ex.sum_keepdim(2)?)?;
        // Softmax produces a dense tensor in most paths, but make the layout
        // contract explicit for both operands of this second batched matmul.
        let out = p
            .contiguous()?
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq, heads * vd))?;
        self.linear(&out, &format!("{rt}.o_proj"), mode)
    }
    fn moe(&mut self, x: &Tensor, layer: usize, mode: E7Mode) -> Result<Tensor> {
        let seq = x.dims()[0];
        let hid = config_usize(&self.config, "hidden_size")?;
        let ne = config_usize(&self.config, "n_routed_experts")?;
        let topk = config_usize(&self.config, "num_experts_per_tok")?;
        if topk == 0 || topk > ne {
            return fail("invalid topk");
        }
        let prefix = format!("model.layers.{layer}.mlp");
        let router = self.weight(&format!("{prefix}.gate.weight"))?;
        let logits = x
            .matmul(&router.t()?)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let bias_name = format!("{prefix}.gate.e_score_correction_bias");
        let bias = match self.w.get(&bias_name) {
            Some(t) => t.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
            None => vec![0.0; ne],
        };
        let norm = self.config["norm_topk_prob"].as_bool().unwrap_or(true);
        let factor = cfg_float(&self.config, "routed_scaling_factor", 1.0) as f32;
        let mut routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(seq);
        let mut used = HashSet::new();
        for (t, lr) in logits.iter().enumerate() {
            let probabilities: Vec<f32> = lr.iter().map(|&a| sigmoid(a)).collect();
            let mut scored: Vec<(usize, f32)> =
                (0..ne).map(|e| (e, probabilities[e] + bias[e])).collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let selected: Vec<usize> = scored.iter().take(topk).map(|(i, _)| *i).collect();
            let z: f32 = if norm {
                selected.iter().map(|i| probabilities[*i]).sum::<f32>() + 1e-20
            } else {
                1.0
            };
            let row: Vec<(usize, f32)> = selected
                .iter()
                .map(|&i| {
                    used.insert(i);
                    (i, probabilities[i] / z * factor)
                })
                .collect();
            self.routes
                .push(format!("layer={layer},token={t},experts={:?}", selected));
            routes.push(row);
        }
        // Validation-only dense expert evaluation for ALL seq tokens per selected expert.
        // This guarantees a full-batch (>8 tokens) DP4A invocation and leaves routing
        // / gather-scatter optimization for a separate milestone.
        let mut expert_cache: HashMap<usize, Tensor> = HashMap::new();
        let im = config_usize(&self.config, "moe_intermediate_size")?;
        for e in used {
            let p = format!("{prefix}.experts.{e}");
            let (gate, up) = if self.q.contains_key(&format!("{p}.gate_up_proj"))
                || self.w.contains_key(&format!("{p}.gate_up_proj.weight"))
            {
                let both = self.linear(x, &format!("{p}.gate_up_proj"), mode)?;
                if both.dims() != [seq, 2 * im] {
                    return fail("gate_up shape mismatch");
                }
                (both.narrow(1, 0, im)?, both.narrow(1, im, im)?)
            } else {
                (
                    self.linear(x, &format!("{p}.gate_proj"), mode)?,
                    self.linear(x, &format!("{p}.up_proj"), mode)?,
                )
            };
            let act = self.swish(&gate)?.broadcast_mul(&up)?;
            let out = self.linear(&act, &format!("{p}.down_proj"), mode)?;
            expert_cache.insert(e, out);
        }
        let mut out_rows = Vec::with_capacity(seq);
        for (t, row) in routes.iter().enumerate() {
            let mut subtotal = Tensor::zeros((1, hid), DType::F32, &self.dev)?;
            for &(e, weight) in row {
                let p = expert_cache.get(&e).unwrap().narrow(0, t, 1)?;
                subtotal = (subtotal + (&p * (weight as f64))?)?;
            }
            out_rows.push(subtotal)
        }
        let routed = Tensor::cat(&out_rows, 0)?;
        let shared = self.mlp(x, &format!("{prefix}.shared_experts"), mode)?;
        routed + shared
    }
    fn forward(&mut self, ids: &[u32], mode: E7Mode) -> Result<Vec<f32>> {
        let seq = ids.len();
        let topk = config_usize(&self.config, "index_topk")?;
        if seq < 9 || seq > topk || seq > 256 {
            return fail("requires 9<=sequence<=min(index_topk,256) for exact DSA full-selection");
        }
        let vocab = config_usize(&self.config, "vocab_size")?;
        if ids.iter().any(|&i| i as usize >= vocab) {
            return fail("token id outside vocabulary");
        }
        let indices = Tensor::from_vec(ids.to_vec(), (seq,), &self.dev)?;
        let mut x = self
            .weight("model.embed_tokens.weight")?
            .index_select(&indices, 0)?;
        let layers = config_usize(&self.config, "num_hidden_layers")?;
        for layer in 0..layers {
            let pre = format!("model.layers.{layer}");
            let a = self.rms(&x, &format!("{pre}.input_layernorm"))?;
            let att = self.attention(&a, layer, mode)?;
            x = (x + att)?;
            let a = self.rms(&x, &format!("{pre}.post_attention_layernorm"))?;
            let is_sparse = match self.config.get("mlp_layer_types").and_then(Value::as_array) {
                Some(ty) => ty.get(layer).and_then(Value::as_str) == Some("sparse"),
                None => layer >= config_usize(&self.config, "first_k_dense_replace")?,
            };
            let mlp = if is_sparse {
                self.moe(&a, layer, mode)?
            } else {
                self.mlp(&a, &format!("{pre}.mlp"), mode)?
            };
            x = (x + mlp)?;
        }
        let norm = self.rms(&x, "model.norm")?;
        let logits = self.linear(&norm, "lm_head", mode)?;
        if logits.dims() != [seq, vocab] {
            return fail("bad full logits shape");
        }
        let vals = logits
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if vals.iter().any(|x| !x.is_finite()) {
            return fail("nonfinite logits");
        }
        Ok(vals)
    }
}
#[test]
#[ignore = "requires original ModelOpt checkpoint assets and physical sm_61 GPU"]
fn arch_e7b_glm_fullforward_logits() -> Result<()> {
    let root = PathBuf::from(std::env::var("ARCH_E7B_ASSETS").expect("ARCH_E7B_ASSETS"));
    let dumps = PathBuf::from(std::env::var("ARCH_E7B_DUMP").expect("ARCH_E7B_DUMP"));
    std::fs::create_dir_all(&dumps)?;
    let ids: Vec<u32> = std::env::var("ARCH_E7B_IDS")
        .expect("ARCH_E7B_IDS")
        .split(',')
        .map(|x| x.parse::<u32>().expect("id"))
        .collect();
    let cuda = CudaDevice::new_with_stream(0)?;
    let mut outputs = Vec::new();
    let mut traces = Vec::new();
    let mut counts = Vec::new();
    for (tag, mode) in [
        ("w4a16", E7Mode::W4A16Reference),
        ("w4a8ref", E7Mode::W4A8Reference),
        ("dp4a", E7Mode::W4A8Dp4a),
    ] {
        let mut m = TinyGlm::load(&root, &cuda)?;
        let logits = m.forward(&ids, mode)?;
        let calls = m.dp4a_calls;
        let quant = m.quant_calls;
        if quant == 0 {
            return fail("No NVFP4 expert projections executed; refuse logits parity");
        }
        if tag == "dp4a" && calls == 0 {
            return fail("Zero actual DP4A dispatches");
        }
        write_f32(&dumps.join(format!("{tag}.f32le")), &logits)?;
        std::fs::write(
            dumps.join(format!("{tag}.routes.txt")),
            m.routes.join("\n") + "\n",
        )?;
        println!(
            "ARCHE7B_BACKEND tag={tag} nvfp4_calls={quant} dp4a_calls={calls} routes={}",
            m.routes.len()
        );
        outputs.push(logits);
        traces.push(m.routes);
        counts.push(calls);
    }
    // Compare like-for-like W4A8 only for backend qualification. W4A16 has
    // different activation precision and can legitimately choose another expert.
    // Print both mismatch counts even if W4A8 fails, for a real failure diagnosis.
    let a16_ref_mismatches = traces[0]
        .iter()
        .zip(&traces[1])
        .filter(|(a, b)| a != b)
        .count();
    let a8_dp_mismatches = traces[1]
        .iter()
        .zip(&traces[2])
        .filter(|(a, b)| a != b)
        .count();
    let counts_match = traces[0].len() == traces[1].len() && traces[1].len() == traces[2].len();
    println!("ARCHE7B_ROUTES w4a16_vs_w4a8_mismatches={a16_ref_mismatches} w4a8_vs_dp4a_mismatches={a8_dp_mismatches} route_counts_match={counts_match}");
    for (i, (a, b)) in traces[1].iter().zip(&traces[2]).enumerate() {
        if a != b {
            println!("ARCHE7B_FIRST_A8_DP_ROUTE_DIFF index={i} ref={a} candidate={b}");
            break;
        }
    }
    if !counts_match || a8_dp_mismatches != 0 {
        return fail(
            "W4A8 reference vs DP4A MoE route divergence; logits parity cannot be isolated",
        );
    }
    let dr = stats(&outputs[1], &outputs[2]);
    let da = stats(&outputs[0], &outputs[2]);
    let vocab = config_usize(&read_json(&root.join("config.json"))?, "vocab_size")?;
    let mut flips_ref = 0usize;
    let mut flips_a16 = 0usize;
    for i in 0..ids.len() {
        let argmax = |v: &Vec<f32>| -> usize {
            (0..vocab)
                .max_by(|&a, &b| {
                    v[i * vocab + a]
                        .total_cmp(&v[i * vocab + b])
                        .then(a.cmp(&b))
                })
                .unwrap()
        };
        let ref_id = argmax(&outputs[1]);
        let dp = argmax(&outputs[2]);
        let a16 = argmax(&outputs[0]);
        flips_ref += (ref_id != dp) as usize;
        flips_a16 += (a16 != dp) as usize;
    }
    println!("ARCHE7B_LOGITS seq={} vocab={} w4a8_parity_max_abs={:.12} w4a8_parity_rmse={:.12} w4a8_parity_cosine={:.12} w4a16_rmse={:.12} w4a16_cosine={:.12} a8_argmax_flips={} a16_argmax_flips={} dp4a_calls={}",ids.len(),vocab,dr.0,dr.1,dr.2,da.1,da.2,flips_ref,flips_a16,counts[2]);
    // The Python driver separately checks per-position cosine and argmax flips.
    // Honor the CLI gate here too; do not silently impose a different threshold.
    let min_cosine: f64 = std::env::var("ARCH_E7B_MIN_LOGIT_COSINE")
        .map_err(|e| candle_core::Error::Msg(format!("ARCH_E7B_MIN_LOGIT_COSINE required: {e}")))?
        .parse()
        .map_err(|e| candle_core::Error::Msg(format!("invalid cosine threshold: {e}")))?;
    if !(0.9..=1.0).contains(&min_cosine) || !dr.2.is_finite() {
        return fail("invalid cosine threshold/result");
    }
    if dr.2 < min_cosine || flips_ref > 0 {
        return fail("W4A8 DP4A / W4A8 reference logit parity FAIL");
    }
    Ok(())
}
