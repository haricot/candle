//! Reusable PCM-to-token model and bounded, independent streaming sessions.
use super::{
    encoder::{Encoder, EncoderState},
    frontend::{Features, FrontendState, LogMel},
    rnnt::{DecodeTrace, DecoderState, Rnnt},
    Config, LanguageConditioner, ProcessorConfig,
};
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_MODEL_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Debug, Clone)]
pub struct Model {
    config: Config,
    processor: ProcessorConfig,
    frontend: LogMel,
    encoder: Encoder,
    language: LanguageConditioner,
    decoder: Rnnt,
    device: Device,
    dtype: DType,
    identity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> Result<Model> {
        let (mut config, mut processor) = super::super::config::official_config();
        let e = &mut config.encoder_config;
        e.hidden_size = 8;
        e.intermediate_size = 16;
        e.num_hidden_layers = 2;
        e.num_attention_heads = 2;
        e.num_key_value_heads = 2;
        e.num_mel_bins = 8;
        e.subsampling_conv_channels = 2;
        e.sliding_window = 9;
        e.default_num_lookahead_tokens = 0;
        e.supported_num_lookahead_tokens = vec![0, 3];
        config.decoder_hidden_size = 4;
        config.vocab_size = 5;
        config.blank_token_id = 4;
        config.num_prompts = 3;
        config.default_prompt_id = 1;
        config.prompt_intermediate_size = 8;
        processor.feature_extractor.feature_size = 8;
        processor.num_prompts = 3;
        processor.default_num_lookahead_tokens = 0;
        processor.supported_num_lookahead_tokens = vec![0, 3];
        processor.prompt_dictionary = [("fr-FR".into(), 2), ("auto".into(), 1)].into();
        let vars = candle_nn::VarMap::new();
        let model = Model::load(
            config,
            processor,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        // Deterministic quiet model. All parameters are zero; only the blank
        // bias is positive, so every encoder frame must advance exactly once.
        for (name, var) in vars.data().lock().unwrap().iter() {
            let value = if name == "joint.head.bias" {
                Tensor::new(&[0f32, 0., 0., 0., 1.], &Device::Cpu)?
            } else {
                Tensor::zeros(var.shape(), var.dtype(), var.device())?
            };
            var.set(&value)?;
        }
        Ok(model)
    }

    #[test]
    fn streams_are_independent_bounded_and_flush_once() -> Result<()> {
        let model = tiny_model()?;
        let pcm = vec![0f32; 25_600];
        for right in [0, 3] {
            let mut a = model.new_stream("fr-FR", right)?;
            let mut b = model.new_stream("fr-FR", right)?;
            let mut expected = model.push_pcm(&mut a, &pcm, 16_000)?;
            expected.append(model.finish(&mut a)?);
            let mut actual = DecodeTrace::default();
            for packet in pcm.chunks(79) {
                actual.append(model.push_pcm(&mut b, packet, 16_000)?);
                assert!(b.attention_cache_frames() <= 8);
                assert!(b.buffered_pcm_samples() <= 512);
                assert!(b.buffered_mel_frames(8) < 8 * (right + 1));
            }
            actual.append(model.finish(&mut b)?);
            assert_eq!(actual.decisions, expected.decisions);
            assert!(actual.tokens.is_empty());
            assert_eq!(actual.decisions.len(), b.encoder_frames());
            assert!(b.encoder_frames() > 8); // Exercise cache eviction, not just startup.
            assert!(model.finish(&mut b)?.decisions.is_empty());
            assert!(model.push_pcm(&mut b, &[0.], 16_000).is_err());
        }
        let mut state = model.new_stream("auto", 3)?;
        assert!(model.push_pcm(&mut state, &[f32::NAN], 16_000).is_err());
        assert!(model.push_pcm(&mut state, &[0.], 8_000).is_err());
        assert_eq!(state.samples(), 0);
        assert!(tiny_model()?.push_pcm(&mut state, &[0.], 16_000).is_err());
        assert_eq!(state.samples(), 0);
        assert!(model.new_stream("invented", 3).is_err());
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StreamState {
    frontend: FrontendState,
    encoder: EncoderState,
    decoder: DecoderState,
    pending_mel: VecDeque<f32>,
    prompt: usize,
    lookahead: usize,
    identity: usize,
    finished: bool,
}
impl StreamState {
    pub fn encoder_frames(&self) -> usize {
        self.encoder.frames_seen()
    }
    pub fn mel_frames(&self) -> usize {
        self.frontend.frame_count()
    }
    pub fn samples(&self) -> usize {
        self.frontend.sample_count()
    }
    pub fn attention_cache_frames(&self) -> usize {
        self.encoder.attention_cache_frames()
    }
    pub fn buffered_pcm_samples(&self) -> usize {
        self.frontend.buffered_samples()
    }
    pub fn buffered_mel_frames(&self, mel_bins: usize) -> usize {
        self.pending_mel.len() / mel_bins
    }
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

impl Model {
    /// Root VarBuilder for the unchanged official HF model.safetensors.
    pub fn load(config: Config, processor: ProcessorConfig, vb: VarBuilder<'_>) -> Result<Self> {
        processor.validate(&config)?;
        if !matches!(vb.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            candle::bail!("Nemotron inference requires floating-point weights")
        }
        Ok(Self {
            frontend: LogMel::new(processor.feature_extractor.clone())?,
            encoder: Encoder::load(&config, vb.pp("encoder"))?,
            language: LanguageConditioner::load(&config, vb.clone())?,
            decoder: Rnnt::load(&config, vb.clone())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            identity: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            processor,
        })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn processor(&self) -> &ProcessorConfig {
        &self.processor
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn mel(&self, pcm: &[f32], rate: usize) -> Result<Features> {
        self.frontend
            .process_mono(pcm, rate, self.dtype, &self.device)
    }
    pub fn encode_features(
        &self,
        features: &Tensor,
        valid: usize,
        lookahead: usize,
    ) -> Result<(Tensor, usize)> {
        self.encoder.forward(features, valid, lookahead)
    }
    pub fn decode_encoded(
        &self,
        encoded: &Tensor,
        valid: usize,
        language: &str,
    ) -> Result<DecodeTrace> {
        let prompt = self.processor.prompt_id(language)?;
        let projected = self.language.forward(encoded, &[prompt])?;
        self.decoder
            .decode(&projected, valid, &mut DecoderState::default())
    }
    pub fn transcribe_mono(
        &self,
        pcm: &[f32],
        rate: usize,
        language: &str,
        lookahead: usize,
    ) -> Result<DecodeTrace> {
        let mel = self.mel(pcm, rate)?;
        let (encoded, valid) = self.encode_features(&mel.values, mel.valid_frames, lookahead)?;
        self.decode_encoded(&encoded, valid, language)
    }
    pub fn new_stream(&self, language: &str, lookahead: usize) -> Result<StreamState> {
        self.config.chunk_plan(&self.processor, lookahead)?;
        Ok(StreamState {
            frontend: FrontendState::default(),
            encoder: self.encoder.new_state(),
            decoder: DecoderState::default(),
            pending_mel: VecDeque::new(),
            prompt: self.processor.prompt_id(language)?,
            lookahead,
            identity: self.identity,
            finished: false,
        })
    }
    fn drain_blocks<F: FnMut(&Tensor) -> Result<()>>(
        &self,
        state: &mut StreamState,
        final_block: bool,
        observe: &mut F,
    ) -> Result<DecodeTrace> {
        let bins = self.config.encoder_config.num_mel_bins;
        let factor = self.config.encoder_config.subsampling_factor;
        let mut result = DecodeTrace::default();
        loop {
            let frames = if state.encoder.frames_seen() == 0 {
                1 + factor * state.lookahead
            } else {
                factor * (state.lookahead + 1)
            };
            let required = frames * bins;
            if state.pending_mel.len() < required {
                if !final_block || state.pending_mel.is_empty() {
                    break;
                }
                // Match the official fixed-size chunk protocol: masked/absent
                // final Mel frames are zero, and the entire final block is decoded.
                state.pending_mel.resize(required, 0.0);
            }
            let values: Vec<_> = state.pending_mel.drain(..required).collect();
            let mel = Tensor::from_vec(values, (1, frames, bins), &Device::Cpu)?
                .to_dtype(self.dtype)?
                .to_device(&self.device)?;
            let encoded = self
                .encoder
                .forward_stream(&mel, state.lookahead, &mut state.encoder)?;
            observe(&encoded)?;
            let projected = self.language.forward(&encoded, &[state.prompt])?;
            result.append(self.decoder.decode(
                &projected,
                projected.dim(1)?,
                &mut state.decoder,
            )?);
        }
        Ok(result)
    }
    pub fn push_pcm(
        &self,
        state: &mut StreamState,
        pcm: &[f32],
        rate: usize,
    ) -> Result<DecodeTrace> {
        self.push_pcm_with_encoder(state, pcm, rate, &mut |_| Ok(()))
    }

    /// Optional read-only encoder observer for callers doing numerical qualification.
    pub fn push_pcm_with_encoder<F: FnMut(&Tensor) -> Result<()>>(
        &self,
        state: &mut StreamState,
        pcm: &[f32],
        rate: usize,
        observe: &mut F,
    ) -> Result<DecodeTrace> {
        if state.identity != self.identity
            || state.finished
            || rate != self.processor.feature_extractor.sampling_rate
            || pcm.iter().any(|v| !v.is_finite())
        {
            candle::bail!("invalid Nemotron stream, PCM or sample rate")
        }
        let mut next = state.clone();
        let mut trace = DecodeTrace::default();
        // No full-utterance PCM or activation history is retained, even when
        // the caller hands us a long packet. Errors leave the old state intact.
        for chunk in pcm.chunks(self.processor.feature_extractor.hop_length * 8) {
            next.pending_mel
                .extend(self.frontend.push(&mut next.frontend, chunk, rate)?);
            trace.append(self.drain_blocks(&mut next, false, observe)?);
        }
        *state = next;
        Ok(trace)
    }
    /// Flush STFT context and zero-pad at most one final encoder block.
    /// Repeating finish is harmless; pushing more PCM after it is rejected.
    pub fn finish(&self, state: &mut StreamState) -> Result<DecodeTrace> {
        self.finish_with_encoder(state, &mut |_| Ok(()))
    }

    /// Flush with the same optional read-only encoder observer as push_pcm_with_encoder.
    pub fn finish_with_encoder<F: FnMut(&Tensor) -> Result<()>>(
        &self,
        state: &mut StreamState,
        observe: &mut F,
    ) -> Result<DecodeTrace> {
        if state.identity != self.identity {
            candle::bail!("stream belongs to another model")
        }
        if state.finished {
            return Ok(DecodeTrace::default());
        }
        let mut next = state.clone();
        next.pending_mel
            .extend(self.frontend.finish(&mut next.frontend)?);
        let result = self.drain_blocks(&mut next, true, observe)?;
        next.finished = true;
        *state = next;
        Ok(result)
    }
}
