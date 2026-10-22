"""Integrated CPU profile: checkpoint frontend, oneDNN subsampling, ordered norms."""
from .c3_r8_source import once
from .c3_r10_source import outputs as r10_outputs


def outputs(parent, root, metadata):
    result = r10_outputs(parent, root, metadata)
    conformer = result['src/parakeet/conformer.rs']
    conformer = once(conformer, 'Conv2dConfig, LayerNorm, Linear,', 'Conv2dConfig, Linear,')
    conformer = once(conformer, 'use crate::parakeet::attention::{',
        'use crate::parakeet::hdb2c_r13_norm::OrderedLayerNorm as LayerNorm;\n\nuse crate::parakeet::attention::{')
    conformer = once(conformer, 'candle_nn::layer_norm(', 'LayerNorm::load(')
    conformer = once(conformer, 'let x = self.linear1.forward(x)?;',
        'let x = crate::parakeet::hdb2c_r13_math::linear(&self.linear1,x)?;')
    conformer = once(conformer, 'self.linear2.forward(&x)',
        'crate::parakeet::hdb2c_r13_math::linear(&self.linear2,&x)')
    for name in ('pointwise_conv1', 'depthwise_conv', 'pointwise_conv2'):
        conformer = once(conformer, f'self.{name}.forward(&x)',
            f'crate::parakeet::hdb2c_r13_math::conv1d(&self.{name},&x)')
    conformer = once(conformer, 'let x = x.silu()?;',
        'let x = crate::parakeet::hdb2c_r13_vector::silu(&x)?;')
    conformer = once(conformer, 'x = x.silu()?;',
        'x = crate::parakeet::hdb2c_r13_vector::silu(&x)?;')
    conformer = once(conformer, '''        x = x.transpose(1, 2)?; // (B, T, 2C)
        let (_, _, c2) = x.dims3()?;
        let c = c2 / 2;
        let a = x.narrow(2, 0, c)?;
        let b_part = x.narrow(2, c, c)?;
        let gate = candle_nn::ops::sigmoid(&b_part)?;
        let mut x = (&a * &gate)?;''',
        '        let mut x = crate::parakeet::hdb2c_r13_vector::glu(&x)?.transpose(1,2)?;')
    conformer = once(conformer, 'x = conv.forward(&x)?;',
        'x = crate::parakeet::hdb2c_r11_onednn::conv2d(conv,&x)?;')
    conformer = once(conformer, 'let y=pre.conv[index].forward(&x)?;',
        'let y=crate::parakeet::hdb2c_r11_onednn::conv2d(&pre.conv[index],&x)?;')
    result['src/parakeet/conformer.rs'] = conformer
    result['src/parakeet/mod.rs'] += '\npub mod hdb2c_r11_onednn;\npub mod hdb2c_r13_norm;\npub mod hdb2c_r13_math;\npub mod hdb2c_r13_vector;\n'
    attention = (parent/'src/parakeet/attention.rs').read_text()
    start = attention.index('impl RelPositionMultiHeadAttention {')
    end = attention.index('#[derive(Debug, Clone)]\npub struct RelPositionalEncoding', start)
    rel = attention[start:end]
    for layer, operand in (('q_proj', 'q'), ('k_proj', 'k'), ('v_proj', 'v'), ('out_proj', '&out')):
        rel = once(rel, f'self.inner.{layer}.forward({operand})',
            f'crate::parakeet::hdb2c_r13_math::linear(&self.inner.{layer},{operand})')
    rel = once(rel, 'self.linear_pos.forward(pos_emb)',
        'crate::parakeet::hdb2c_r13_math::linear(&self.linear_pos,pos_emb)')
    rel = once(rel, 'let scale = (self.inner.head_dim as f64).powf(-0.5);\n        let k_t = k.transpose(2, 3)?.contiguous()?;\n        let mut matrix_ac = q_u.matmul(&k_t)?;',
        'let matrix_ac = crate::parakeet::hdb2c_r13_math::bmm(&q_u,&k,true)?;')
    rel = once(rel, 'let p_t = p.transpose(2, 3)?.contiguous()?;\n            let bd = q_v.matmul(&p_t)?;',
        'let bd = crate::parakeet::hdb2c_r13_math::bmm(&q_v,&p,true)?;')
    rel = once(rel, 'matrix_ac = (&matrix_ac * scale)?;\n        matrix_bd = (&matrix_bd * scale)?;\n        let mut scores = matrix_ac.broadcast_add(&matrix_bd)?;',
        'let mut scores = crate::parakeet::hdb2c_r13_math::attention_scores(&matrix_ac,&matrix_bd,self.inner.head_dim)?;')
    rel = once(rel, 'let out = attn.matmul(&v)?;',
        'let out = crate::parakeet::hdb2c_r13_math::bmm(&attn,&v,false)?;')
    attention = attention[:start]+rel+attention[end:]
    attention = once(attention, 'None => softmax_last_dim(scores),',
        'None => crate::parakeet::hdb2c_r13_vector::softmax(scores),')
    attention = once(attention, 'let prob = softmax_last_dim(&filled)?;',
        'let prob = crate::parakeet::hdb2c_r13_vector::softmax(&filled)?;')
    start = attention.index('    fn build_pe(d_model: usize')
    end = attention.index('    pub fn forward(&mut self, x: &Tensor, offset: usize)', start)
    attention = attention[:start]+'''    fn build_pe(d_model: usize, max_len: usize, device: &Device) -> Result<Tensor> {
        let len=2*max_len-1;let half=d_model/2;
        let factor=(-((10000.0f64).ln()/d_model as f64)) as f32;
        let exponents:Vec<f32>=(0..half).map(|i|(2*i) as f32*factor).collect();
        let terms=crate::parakeet::hdb2c_r13_vector::unary(0,&exponents)?;
        let mut angles=Vec::with_capacity(len*half);
        for idx in 0..len {
            let pos=((max_len-1) as i32-idx as i32) as f32;
            for &term in &terms {angles.push(pos*term);}
        }
        let sin=crate::parakeet::hdb2c_r13_vector::unary(1,&angles)?;
        let cos=crate::parakeet::hdb2c_r13_vector::unary(2,&angles)?;
        let mut data=Vec::with_capacity(len*d_model);
        for i in 0..angles.len() {data.push(sin[i]);data.push(cos[i]);}
        Tensor::from_vec(data,(1,len,d_model),device)?.to_dtype(DType::F32)
    }

'''+attention[end:]
    result['src/parakeet/attention.rs'] = attention
    probe = result['src/bin/hdb2c_r10_probe.rs']
    probe = once(probe, 'R10_NATIVE_CPU_MKL_CHECKPOINT_FRONTEND_FUSED_PROJECTION',
        'R13_FULL_CPU_MKL_ONEDNN_ORDERED_LAYERNORM')
    probe = once(probe, '"threads":candle::utils::get_num_threads(),',
        '"vector_lanes":parakeet_candle::parakeet::hdb2c_r13_vector::width(),"layernorm_lanes":8,"threads":candle::utils::get_num_threads(),')
    result['src/bin/hdb2c_r13_probe.rs'] = probe
    result['build.rs'] = (root/'rust/hdb2c_r13_build.rs').read_text()
    result['src/parakeet/hdb2c_r13_norm.rs'] = (root/'rust/hdb2c_r13_norm.rs').read_text()
    result['src/parakeet/hdb2c_r13_math.rs'] = (root/'rust/hdb2c_r13_math.rs').read_text()
    for name in ('hdb2c_r13_vector.rs', 'hdb2c_r13_vector.cpp'):
        result['src/parakeet/'+name] = (root/'rust'/name).read_text()
    experimental = root/'ci/experimental/cpu_parity'
    result['src/parakeet/hdb2c_r13_norm.cpp'] = (experimental/'ordered_layernorm.cpp').read_text()
    result['src/parakeet/hdb2c_r11_onednn.cpp'] = (experimental/'onednn_bridge.cpp').read_text()
    result['src/parakeet/hdb2c_r11_onednn.rs'] = (experimental/'onednn_bridge.rs').read_text()
    return result
