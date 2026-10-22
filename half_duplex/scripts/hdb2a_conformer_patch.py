"""Pin-specific FastConformer patch: NeMo layer_norm + convolution padding.
Only the local vendored parakeet-candle checkout is modified.
"""

def patch_conformer(source: str) -> str:
    if 'HDB2A_CONVNORM_LAYER_SUPPORT' in source:
        return source

    def exactly(old: str, new: str):
        nonlocal source
        count = source.count(old)
        if count != 1:
            raise ValueError(f'Pin drift: expected 1 match, found {count}: {old[:90]!r}')
        source = source.replace(old, new)

    exactly('    pub conv_kernel_size: usize,\n', '''    pub conv_kernel_size: usize,
    #[serde(default = "hdb2a_default_batch_norm")]
    pub conv_norm_type: String,
    #[serde(default)]
    pub conv_context_size: Option<serde_json::Value>,
''')
    exactly('fn default_true() -> bool {\n', '''// HDB2A_CONVNORM_LAYER_SUPPORT: match NeMo convolution normalization.
fn hdb2a_default_batch_norm() -> String {
    "batch_norm".to_owned()
}

// In NeMo, conv_context_size=None => centered, "causal" => left-only.
fn hdb2a_conv_padding(args: &ConformerArgs) -> Result<(usize, usize)> {
    let k = args.conv_kernel_size;
    if k == 0 { return Err(candle::Error::Msg("conv_kernel_size cannot be zero".into())); }
    match args.conv_context_size.as_ref() {
        None | Some(serde_json::Value::Null) => Ok(((k - 1) / 2, k / 2)),
        Some(serde_json::Value::String(s)) if s == "default" => Ok(((k - 1) / 2, k / 2)),
        Some(serde_json::Value::String(s)) if s == "causal" => Ok((k - 1, 0)),
        Some(serde_json::Value::Array(v)) if v.len() == 2 => {
            let l = v[0].as_u64().and_then(|x| usize::try_from(x).ok());
            let r = v[1].as_u64().and_then(|x| usize::try_from(x).ok());
            match (l, r) {
                (Some(l), Some(r)) if l.checked_add(r).and_then(|n| n.checked_add(1)) == Some(k) => Ok((l, r)),
                _ => Err(candle::Error::Msg("Invalid conv_context_size [left,right]".into())),
            }
        }
        other => Err(candle::Error::Msg(format!("Unsupported conv_context_size: {other:?}"))),
    }
}

fn default_true() -> bool {
''')
    # The original #[derive(Debug, Clone)] belongs to the new enum, and
    # a second derive is added to preserve Convolution's existing derives.
    exactly('struct Convolution {\n', '''enum Hdb2aConvNorm {
    Batch(BatchNorm),
    Layer(LayerNorm),
}

#[derive(Debug, Clone)]
struct Convolution {
''')
    exactly('    batch_norm: BatchNorm,\n', '    norm: Hdb2aConvNorm,\n')
    exactly('    padding: usize,\n    causal_padding: usize,\n',
            '    padding_left: usize,\n    padding_right: usize,\n    causal_padding: usize,\n')
    exactly('        let padding = (args.conv_kernel_size - 1) / 2;\n',
            '        let (padding_left, padding_right) = hdb2a_conv_padding(args)?;\n')
    exactly('        let batch_norm = candle_nn::batch_norm(args.d_model, 1e-5, vb.pp("batch_norm"))?;\n', '''        let norm = match args.conv_norm_type.as_str() {
            "batch_norm" => Hdb2aConvNorm::Batch(
                candle_nn::batch_norm(args.d_model, 1e-5, vb.pp("batch_norm"))?
            ),
            "layer_norm" => Hdb2aConvNorm::Layer(
                layer_norm(args.d_model, vb.pp("batch_norm"))?
            ),
            other => return Err(candle::Error::Msg(format!(
                "Unsupported Conformer conv_norm_type: {other}"
            ))),
        };
''')
    exactly('            batch_norm,\n', '            norm,\n')
    exactly('            pointwise_conv2,\n            padding,\n            causal_padding,\n',
            '            pointwise_conv2,\n            padding_left,\n            padding_right,\n            causal_padding,\n')
    exactly('        } else if self.padding > 0 {\n            x = x.pad_with_zeros(D::Minus2, self.padding, self.padding)?;\n',
'''        } else if self.padding_left != 0 || self.padding_right != 0 {
            x = x.pad_with_zeros(D::Minus2, self.padding_left, self.padding_right)?;
''')
    exactly('        x = self.batch_norm.forward_t(&x, false)?;\n', '''        x = match &self.norm {
            Hdb2aConvNorm::Batch(bn) => bn.forward_t(&x, false)?,
            // NeMo's LayerNorm parameters retain the conv.batch_norm prefix;
            // normalize across channels, not across time.
            Hdb2aConvNorm::Layer(ln) =>
                ln.forward(&x.transpose(1, 2)?.contiguous()?)?.transpose(1, 2)?,
        };
''')
    return source
