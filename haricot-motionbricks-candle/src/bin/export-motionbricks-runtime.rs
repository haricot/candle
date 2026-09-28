use anyhow::{bail, Context, Result};
use candle_core::{pickle::PthTensors, safetensors, DType, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const PINNED_REV: &str = "087f9ac01d46f6d8e4d0b73c01ae64799f292a38";

#[derive(Debug)]
struct Args {
    motionbricks_root: PathBuf,
    out: PathBuf,
    audit_keys: bool,
}

fn usage() {
    println!(
        "export-motionbricks-runtime \\\n  --motionbricks-root <GR00T-WholeBodyControl checkout> \\\n  --out <motionbricks-runtime-v1.safetensors> [--audit-keys]"
    );
}

fn parse_args() -> Result<Args> {
    let mut motionbricks_root = None;
    let mut out = None;
    let mut audit_keys = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--motionbricks-root" => {
                motionbricks_root = Some(PathBuf::from(
                    it.next().context("missing value for --motionbricks-root")?,
                ));
            }
            "--out" => {
                out = Some(PathBuf::from(it.next().context("missing value for --out")?));
            }
            "--audit-keys" => audit_keys = true,
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(Args {
        motionbricks_root: motionbricks_root.context("missing --motionbricks-root <path>")?,
        out: out.context("missing --out <path>")?,
        audit_keys,
    })
}

#[derive(Debug)]
struct Layout {
    repo: PathBuf,
    mb: PathBuf,
}

fn resolve_layout(root: &Path) -> Result<Layout> {
    let root = root
        .canonicalize()
        .with_context(|| format!("cannot canonicalize {}", root.display()))?;
    if root.join("motionbricks/motionbricks").is_dir() {
        return Ok(Layout {
            repo: root.clone(),
            mb: root.join("motionbricks"),
        });
    }
    if root.join("motionbricks").is_dir() && root.join("out").is_dir() {
        let repo = root
            .parent()
            .context("MotionBricks project root has no parent repository")?
            .to_path_buf();
        return Ok(Layout { repo, mb: root });
    }
    bail!(
        "cannot resolve GR00T/MotionBricks layout from {}",
        root.display()
    )
}

fn git_revision(repo: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .context("failed to execute git rev-parse HEAD")?;
    if !output.status.success() {
        bail!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn require_checkpoint(path: &Path) -> Result<()> {
    let md = std::fs::metadata(path)
        .with_context(|| format!("missing official checkpoint: {}", path.display()))?;
    if md.len() < 1024 * 1024 {
        bail!(
            "checkpoint looks like a Git LFS pointer ({} bytes): {}",
            md.len(),
            path.display()
        )
    }
    Ok(())
}

fn is_float(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
}

fn load_state_dict(path: &Path) -> Result<PthTensors> {
    PthTensors::new(path, Some("state_dict"))
        .with_context(|| format!("failed to read Lightning state_dict from {}", path.display()))
}

fn copy_prefix(
    src: &PthTensors,
    src_prefix: &str,
    dst_prefix: &str,
    dst: &mut HashMap<String, Tensor>,
    audit_keys: bool,
) -> Result<usize> {
    let mut names = src
        .tensor_infos()
        .keys()
        .filter(|name| name.starts_with(src_prefix))
        .cloned()
        .collect::<Vec<_>>();
    names.sort();

    let mut copied = 0usize;
    for name in names {
        let info = src
            .tensor_infos()
            .get(&name)
            .with_context(|| format!("missing tensor info for {name}"))?;
        if !is_float(info.dtype) {
            continue;
        }
        let rest = name
            .strip_prefix(src_prefix)
            .context("prefix disappeared while copying state dict")?;
        let output_name = format!("{dst_prefix}{rest}");
        let tensor = src
            .get(&name)?
            .with_context(|| format!("tensor listed but unavailable: {name}"))?
            .to_dtype(DType::F32)?;
        if audit_keys {
            println!(
                "map={} -> {} shape={:?} dtype={:?}",
                name,
                output_name,
                tensor.dims(),
                tensor.dtype()
            );
        }
        dst.insert(output_name, tensor);
        copied += 1;
    }
    Ok(copied)
}

fn get_f32(src: &PthTensors, name: &str) -> Result<Tensor> {
    Ok(src
        .get(name)?
        .with_context(|| format!("missing checkpoint tensor {name}"))?
        .to_dtype(DType::F32)?)
}

fn find_pose_codebook(src: &PthTensors, audit_keys: bool) -> Result<(String, Tensor)> {
    // vector-quantize-pytorch stores the live EMA codebook here for the pinned release.
    const EXACT: &[&str] = &[
        "pose_net.quantizer.vq._codebook.embed",
        "pose_net.quantizer.vq._codebook.embed_avg",
    ];
    for name in EXACT {
        if let Some(info) = src.tensor_infos().get(*name) {
            if info.layout.dims() == [8, 10, 32] && is_float(info.dtype) {
                let tensor = get_f32(src, name)?;
                if audit_keys {
                    println!("codebook_source={name} shape={:?}", tensor.dims());
                }
                return Ok(((*name).to_owned(), tensor));
            }
        }
    }

    // Fallback is deliberately narrow. It permits minor vector-quantize-pytorch
    // naming changes but refuses an ambiguous tensor instead of guessing.
    let mut candidates = src
        .tensor_infos()
        .iter()
        .filter_map(|(name, info)| {
            let shape_ok = info.layout.dims() == [8, 10, 32];
            let name_ok = name.starts_with("pose_net.quantizer.") && name.ends_with(".embed");
            (shape_ok && name_ok && is_float(info.dtype)).then(|| name.clone())
        })
        .collect::<Vec<_>>();
    candidates.sort();
    if candidates.len() != 1 {
        bail!(
            "cannot identify unique pose codebook [8,10,32]; candidates={candidates:?}"
        )
    }
    let name = candidates.remove(0);
    let tensor = get_f32(src, &name)?;
    if audit_keys {
        println!("codebook_source={name} shape={:?}", tensor.dims());
    }
    Ok((name, tensor))
}

fn require_output_tensor(tensors: &HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<()> {
    let t = tensors
        .get(name)
        .with_context(|| format!("required exported tensor missing: {name}"))?;
    if t.dims() != shape {
        bail!(
            "exported tensor {name} has shape {:?}, expected {shape:?}",
            t.dims()
        )
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let layout = resolve_layout(&args.motionbricks_root)?;
    let rev = git_revision(&layout.repo)?;
    if rev != PINNED_REV {
        bail!("MotionBricks revision mismatch: got {rev}, expected {PINNED_REV}")
    }

    let vqvae_ckpt = layout
        .mb
        .join("out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt");
    let pose_ckpt = layout
        .mb
        .join("out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt");
    let root_ckpt = layout
        .mb
        .join("out/motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt");
    for path in [&vqvae_ckpt, &pose_ckpt, &root_ckpt] {
        require_checkpoint(path)?;
    }

    println!("=== HARICOT MOTIONBRICKS RUNTIME RUST EXPORT V0.3-R1 ===");
    println!("source_rev={rev}");
    println!("checkpoint_reader=candle_core::pickle::PthTensors(state_dict)");
    println!("python=false");
    println!("hydra=false");
    println!("lightning_runtime=false");

    let pose = load_state_dict(&pose_ckpt)?;
    let root = load_state_dict(&root_ckpt)?;
    let vqvae = load_state_dict(&vqvae_ckpt)?;
    println!("pose_state_tensor_infos={}", pose.tensor_infos().len());
    println!("root_state_tensor_infos={}", root.tensor_infos().len());
    println!("vqvae_state_tensor_infos={}", vqvae.tensor_infos().len());

    let mut tensors = HashMap::<String, Tensor>::new();
    let pose_count = copy_prefix(
        &pose,
        "backbone_net.",
        "pose.",
        &mut tensors,
        args.audit_keys,
    )?;
    let root_count = copy_prefix(
        &root,
        "backbone_net.",
        "root.",
        &mut tensors,
        args.audit_keys,
    )?;
    let decoder_count = copy_prefix(
        &vqvae,
        "pose_net.decoder.",
        "vqvae.decoder.",
        &mut tensors,
        args.audit_keys,
    )?;
    let (codebook_source, codebook) = find_pose_codebook(&vqvae, args.audit_keys)?;
    tensors.insert("vqvae.codebook".to_owned(), codebook);

    if pose_count == 0 || root_count == 0 || decoder_count == 0 {
        bail!(
            "empty exported module: pose={pose_count} root={root_count} decoder={decoder_count}"
        )
    }

    // Structural anchors used by the Candle runtime. These fail early if checkpoint
    // naming or architecture ever diverges from the pinned release.
    require_output_tensor(
        &tensors,
        "pose._transformer_model.layers.0.self_attn.in_proj_weight",
        &[3072, 1024],
    )?;
    require_output_tensor(
        &tensors,
        "root._shared_transformer_model.layers.0.self_attn.in_proj_weight",
        &[1536, 512],
    )?;
    require_output_tensor(&tensors, "vqvae.decoder.model.0.weight", &[512, 256, 3])?;
    // Effective feature dimensions are derived by upstream from motion_rep at runtime.
    // The YAML's historical 241/329 constructor values are overwritten in VQVAE.__init__.
    require_output_tensor(&tensors, "root._proj_local_pose.weight", &[256, 304])?;
    require_output_tensor(&tensors, "pose._proj_local_pose.weight", &[160, 304])?;
    require_output_tensor(&tensors, "vqvae.decoder.target_cond_blocks.0.weight", &[128, 304])?;
    require_output_tensor(&tensors, "vqvae.decoder.model.6.weight", &[413, 512, 3])?;
    require_output_tensor(&tensors, "vqvae.codebook", &[8, 10, 32])?;

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    safetensors::save(&tensors, &args.out)
        .with_context(|| format!("failed to save {}", args.out.display()))?;

    let parameter_count = tensors.values().map(|t| t.elem_count()).sum::<usize>();
    println!("pose_tensor_count={pose_count}");
    println!("root_tensor_count={root_count}");
    println!("vqvae_decoder_tensor_count={decoder_count}");
    println!("codebook_source={codebook_source}");
    println!("codebook_shape=[8,10,32]");
    println!("tensor_count={}", tensors.len());
    println!("parameter_count={parameter_count}");
    println!("f32_mib={:.3}", parameter_count as f64 * 4.0 / 1024.0 / 1024.0);
    println!("conv_policy=official-standard-conv1d-groups1");
    println!("conv_transpose_count=0");
    println!("out={}", args.out.display());
    println!("MOTIONBRICKS_RUNTIME_RUST_EXPORT=PASS");
    Ok(())
}
