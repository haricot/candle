use candle::{DType, Device, Result, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder, VarMap};
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    (lhs - rhs)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
}

fn copy_varmap_to_device(src: &VarMap, dst: &VarMap, device: &Device, dtype: DType) -> Result<()> {
    let mut data_dst = dst.data().lock().unwrap();
    let data_src = src.data().lock().unwrap();
    for (name, var) in data_src.iter() {
        let tensor = var.as_tensor().to_device(device)?.to_dtype(dtype)?;
        data_dst.insert(name.clone(), candle::Var::from_tensor(&tensor)?);
    }
    Ok(())
}

fn test_qwen3_config() -> Config {
    Config {
        vocab_size: 256,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 32,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 128,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: false,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    }
}

fn assert_qwen3_cuda_matches_cpu(dtype_cuda: DType, max_diff_expected: f32) -> Result<()> {
    if !candle::utils::cuda_is_available() {
        println!("CUDA not available, skipping");
        return Ok(());
    }

    let device_cpu = Device::Cpu;
    let device_cuda = Device::new_cuda(0)?;
    let config = test_qwen3_config();

    let input_ids = Tensor::new(&[[1u32, 5, 10, 20]], &device_cpu)?;
    let logits_cpu = {
        let varmap_cpu = VarMap::new();
        let vb_cpu = VarBuilder::from_varmap(&varmap_cpu, DType::F32, &device_cpu);
        let mut model_cpu = ModelForCausalLM::new(&config, vb_cpu)?;
        let logits = model_cpu.forward(&input_ids, 0)?;

        let varmap_cuda = VarMap::new();
        copy_varmap_to_device(&varmap_cpu, &varmap_cuda, &device_cuda, dtype_cuda)?;
        let vb_cuda = VarBuilder::from_varmap(&varmap_cuda, dtype_cuda, &device_cuda);
        let mut model_cuda = ModelForCausalLM::new(&config, vb_cuda)?;
        let logits_cuda = model_cuda.forward(&input_ids.to_device(&device_cuda)?, 0)?;
        let logits_cuda = logits_cuda.to_device(&device_cpu)?.to_dtype(DType::F32)?;

        let logits = logits.to_dtype(DType::F32)?;
        let diff = max_abs_diff(&logits, &logits_cuda)?;
        println!("Qwen3 CPU vs CUDA {:?} max diff: {}", dtype_cuda, diff);
        assert!(
            diff < max_diff_expected,
            "Qwen3 CPU vs CUDA {:?} diverged, max diff: {}",
            dtype_cuda,
            diff
        );
        logits
    };

    assert_eq!(logits_cpu.rank(), 3);
    Ok(())
}

fn assert_rms_norm_forward_diff_matches_cpu(
    dtype_cuda: DType,
    max_diff_expected: f32,
) -> Result<()> {
    if !candle::utils::cuda_is_available() {
        println!("CUDA not available, skipping");
        return Ok(());
    }

    let device_cpu = Device::Cpu;
    let device_cuda = Device::new_cuda(0)?;

    let hidden = 128;
    let head_dim = 32;

    let weight_hidden_cpu = Tensor::randn(0f32, 1f32, hidden, &device_cpu)?;
    let weight_head_cpu = Tensor::randn(0f32, 1f32, head_dim, &device_cpu)?;
    let weight_hidden_cuda = weight_hidden_cpu
        .to_device(&device_cuda)?
        .to_dtype(dtype_cuda)?;
    let weight_head_cuda = weight_head_cpu
        .to_device(&device_cuda)?
        .to_dtype(dtype_cuda)?;

    let norm_hidden_cpu = RmsNorm::new(weight_hidden_cpu, 1e-6);
    let norm_hidden_cuda = RmsNorm::new(weight_hidden_cuda, 1e-6);
    let norm_head_cpu = RmsNorm::new(weight_head_cpu, 1e-6);
    let norm_head_cuda = RmsNorm::new(weight_head_cuda, 1e-6);

    let xs_hidden_cpu = Tensor::randn(0f32, 1f32, (1, 10, hidden), &device_cpu)?;
    let xs_head_cpu = Tensor::randn(0f32, 1f32, (1, 4, 10, head_dim), &device_cpu)?;
    let xs_hidden_cuda = xs_hidden_cpu
        .to_device(&device_cuda)?
        .to_dtype(dtype_cuda)?;
    let xs_head_cuda = xs_head_cpu.to_device(&device_cuda)?.to_dtype(dtype_cuda)?;

    let hidden_cpu = norm_hidden_cpu.forward(&xs_hidden_cpu)?;
    let hidden_cuda = norm_hidden_cuda
        .forward_diff(&xs_hidden_cuda)?
        .to_device(&device_cpu)?
        .to_dtype(DType::F32)?;
    let hidden_cpu = hidden_cpu.to_dtype(DType::F32)?;
    let hidden_diff = max_abs_diff(&hidden_cpu, &hidden_cuda)?;

    let head_cpu = norm_head_cpu.forward(&xs_head_cpu)?;
    let head_cuda = norm_head_cuda
        .forward_diff(&xs_head_cuda)?
        .to_device(&device_cpu)?
        .to_dtype(DType::F32)?;
    let head_cpu = head_cpu.to_dtype(DType::F32)?;
    let head_diff = max_abs_diff(&head_cpu, &head_cuda)?;

    println!(
        "RmsNorm forward_diff CPU vs CUDA {:?} max diff: hidden={}, head={}",
        dtype_cuda, hidden_diff, head_diff
    );

    assert!(
        hidden_diff < max_diff_expected,
        "RmsNorm hidden CPU vs CUDA {:?} diverged, max diff: {}",
        dtype_cuda,
        hidden_diff
    );
    assert!(
        head_diff < max_diff_expected,
        "RmsNorm head CPU vs CUDA {:?} diverged, max diff: {}",
        dtype_cuda,
        head_diff
    );

    Ok(())
}

#[test]
fn test_qwen3_cuda_f32() -> Result<()> {
    assert_qwen3_cuda_matches_cpu(DType::F32, 1e-4)
}

#[test]
fn test_qwen3_cuda_f16() -> Result<()> {
    assert_qwen3_cuda_matches_cpu(DType::F16, 0.1)
}

#[test]
fn test_qwen3_cuda_bf16() -> Result<()> {
    assert_qwen3_cuda_matches_cpu(DType::BF16, 0.2)
}

#[test]
fn test_qwen3_rms_norm_forward_diff_f16() -> Result<()> {
    assert_rms_norm_forward_diff_matches_cpu(DType::F16, 0.1)
}

#[test]
fn test_qwen3_rms_norm_forward_diff_bf16() -> Result<()> {
    assert_rms_norm_forward_diff_matches_cpu(DType::BF16, 0.2)
}
