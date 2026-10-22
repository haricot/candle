"""Strict source transforms for the isolated R6 encoder padding-mask experiment."""

MARKER = 'HDB2C3_R6_ENCODER_PADDING_MASKS_V1'


def once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError('R6 pinned source changed: '+label)
    return source.replace(old, new, 1)


def transform_attention(source, helper):
    if MARKER in source or 'HDB2C3_R5_POSITIONAL_F32_ORDER_V1' not in source:
        raise ValueError('R6 requires the unchanged R5 attention source')
    start = source.index('impl RelPositionMultiHeadAttention {')
    end = source.index('\n#[derive(Debug, Clone)]\npub struct RelPositionalEncoding', start)
    block = source[start:end]
    block = once(block, '        cache: Option<&mut dyn CacheLike>,\n    ) -> Result<Tensor> {',
                 '        cache: Option<&mut dyn CacheLike>,\n        hdb2c6_padding: Option<&Tensor>,\n    ) -> Result<Tensor> {', 'relative attention signature')
    block = once(block, '        let attn = softmax_last_dim(&scores)?;',
                 '        let attn = hdb2c6_masked_softmax(&scores,hdb2c6_padding)?;', 'relative attention softmax')
    return source[:start]+block+source[end:]+'\n'+helper


def transform_conformer(source):
    if MARKER in source or 'HDB2C3_R4_BLOCK0_INTERNAL_TRACE_V1' not in source:
        raise ValueError('R6 requires the unchanged R5/R4 instrumented conformer')
    source = once(source,
        '    fn forward(&self, x: &Tensor, cache: Option<&mut dyn CacheLike>) -> Result<Tensor> {',
        '    fn forward(&self, x: &Tensor, cache: Option<&mut dyn CacheLike>, hdb2c6_pad: Option<&Tensor>, r6_trace: bool) -> Result<Tensor> {', 'convolution signature')
    source = once(source, '        let mut x = (&a * &gate)?;\n', '''        let mut x = (&a * &gate)?;
        if r6_trace {hdb2c3_trace_stage("block0_conv_before_mask",&x)?;}
        x = crate::parakeet::attention::hdb2c6_zero_padding(&x,hdb2c6_pad)?;
        if r6_trace {hdb2c3_trace_stage("block0_conv_after_mask",&x)?;}
''', 'mask exactly after GLU before depthwise convolution')
    source = once(source, '        r4_trace: bool,\n    ) -> Result<Tensor> {',
        '        r4_trace: bool,\n        hdb2c6_att: Option<&Tensor>,\n        hdb2c6_pad: Option<&Tensor>,\n    ) -> Result<Tensor> {', 'block signature')
    source = once(source, '''                        mask.as_ref(),
                        Some(cache),
                    )?''', '''                        mask.as_ref(),
                        Some(cache),
                        None,
                    )?''', 'cached relative attention call')
    source = once(source, 'attn.forward(&x_norm, &x_norm, &x_norm, pos_emb, mask.as_ref(), None)?',
        'attn.forward(&x_norm, &x_norm, &x_norm, pos_emb, mask.as_ref(), None, hdb2c6_att)?', 'offline relative attention call')
    source = once(source, 'self.conv.forward(&norm_x, cache)?',
        'self.conv.forward(&norm_x, cache, hdb2c6_pad, r4_trace)?', 'convolution mask propagation')
    source = once(source, '        hdb2c3_trace_stage("post_pos_enc", &x)?;\n', '''        hdb2c3_trace_stage("post_pos_enc", &x)?;
        // HDB2C3_R6_ENCODER_PADDING_MASKS_V1: offline, regular global rel_pos only.
        if cache.is_some() || self.args.self_attention_model!="rel_pos" ||
            self.args.att_context_size.as_ref().is_some_and(|v|v.as_slice()!=[-1i64,-1].as_slice()) {
            return Err(candle::Error::Msg("R6 only qualifies offline global rel_pos".into()));
        }
        let r6_mode=std::env::var("HDB2C6_MASK_MODE")
            .map_err(|_|candle::Error::Msg("R6 explicit mask mode required".into()))?;
        let (r6_att_on,r6_conv_on)=match r6_mode.as_str(){
            "none"=>(false,false),"attention"=>(true,false),
            "convolution"=>(false,true),"both"=>(true,true),
            _=>return Err(candle::Error::Msg("R6 unknown mask mode".into()))
        };
        let (r6_pad,r6_att)=crate::parakeet::attention::hdb2c6_masks(&x,&out_lengths)?;
        hdb2c3_trace_stage("encoder_pad_mask",&r6_pad.unsqueeze(2)?)?;
        hdb2c3_trace_stage("encoder_att_mask",&r6_att.squeeze(1)?)?;
''', 'native mask construction from actual encoder lengths')
    source = once(source, 'x = layer.forward(&x, pos_emb.as_ref(), Some(cache), false)?;',
        'x = layer.forward(&x, pos_emb.as_ref(), Some(cache), false, None, None)?;', 'cached block call')
    source = once(source, 'x = layer.forward(&x, pos_emb.as_ref(), None, index==0)?;',
        'x = layer.forward(&x, pos_emb.as_ref(), None, index==0, r6_att_on.then_some(&r6_att), r6_conv_on.then_some(&r6_pad))?;', 'offline block call')
    return source
