//! Optional multilingual punctuation restoration after an ASR utterance is finalized.
//! The unmodified ASR token stream remains the source of exact-parity comparisons.
//! External XLM-RoBERTa weights are loaded locally; no Python, ONNX or download.

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{
    Config as XlmRobertaConfig, XLMRobertaForTokenClassification,
};
use std::path::Path;
use std::time::{Duration, Instant};

pub const MODEL_ID: &str = "oliverguhr/fullstop-punctuation-multilingual-sonar-base";

// Bound the context and reserve a little space for subword tokenization.
// A word near a boundary is classified with both left and right context.
const WORDS_PER_STEP: usize = 80;
const CONTEXT_WORDS: usize = 12;

pub struct Punctuator {
    model: XLMRobertaForTokenClassification,
    tokenizer: tokenizers::Tokenizer,
    labels: Vec<Option<char>>,
    max_sequence: usize,
    device: Device,
}

impl Punctuator {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let config_bytes =
            std::fs::read(dir.join("config.json")).context("reading punctuation config.json")?;
        let config: XlmRobertaConfig = serde_json::from_slice(&config_bytes)
            .context("parsing XLM-RoBERTa punctuation config")?;
        let value: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let label_map = value["id2label"]
            .as_object()
            .context("punctuation config must have id2label")?;
        if label_map.len() < 2 {
            bail!("punctuation classifier must have at least two labels");
        }
        let mut labels = vec![None; label_map.len()];
        for (index, label) in label_map {
            let index: usize = index.parse().context("invalid punctuation label index")?;
            if index >= labels.len() {
                bail!("non-contiguous punctuation label indices");
            }
            labels[index] = match label.as_str() {
                Some("0") => None,
                Some(".") => Some('.'),
                Some(",") => Some(','),
                Some("?") => Some('?'),
                Some("-") => Some('-'),
                Some(":") => Some(':'),
                _ => bail!("unexpected punctuation label: {label}"),
            };
        }
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)
            .context("reading punctuation tokenizer.json")?;
        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            bail!(
                "missing {}; download the safetensors and tokenizer locally first",
                weights.display()
            )
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)
                .context("mapping punctuation model.safetensors")?
        };
        let model = XLMRobertaForTokenClassification::new(labels.len(), &config, vb)
            .context("loading punctuation XLM-RoBERTa head")?;
        Ok(Self {
            model,
            tokenizer,
            labels,
            max_sequence: config.max_position_embeddings.saturating_sub(2),
            device,
        })
    }

    /// Restore punctuation on finalized text only; do not rewrite ASR tokens.
    pub fn restore(&self, raw_text: &str, min_confidence: f32) -> Result<(String, Duration)> {
        if !(0.0..=1.0).contains(&min_confidence) || min_confidence.is_nan() {
            bail!("punctuation minimum confidence must be between 0 and 1");
        }
        let start = Instant::now();
        let words: Vec<&str> = raw_text.split_whitespace().collect();
        if words.is_empty() {
            return Ok((String::new(), start.elapsed()));
        }
        let mut punctuation = vec![None; words.len()];
        let mut cursor = 0;
        while cursor < words.len() {
            let from = cursor.saturating_sub(CONTEXT_WORDS);
            let to = (cursor + WORDS_PER_STEP + CONTEXT_WORDS).min(words.len());
            let predicted = self.predict_chunk(&words[from..to], min_confidence)?;
            let accepted_end = (cursor + WORDS_PER_STEP).min(words.len());
            for word_index in cursor..accepted_end {
                punctuation[word_index] = predicted[word_index - from];
            }
            cursor = accepted_end;
        }
        Ok((render_punctuation(&words, &punctuation), start.elapsed()))
    }

    fn predict_chunk(&self, words: &[&str], min_confidence: f32) -> Result<Vec<Option<char>>> {
        let encoded = self
            .tokenizer
            .encode(words, true)
            .map_err(anyhow::Error::msg)
            .context("tokenizing words for punctuation model")?;
        let ids = encoded.get_ids();
        if ids.is_empty() || ids.len() > self.max_sequence {
            bail!(
                "punctuation window has {} tokens (maximum {}); shorten the utterance/window",
                ids.len(),
                self.max_sequence
            );
        }
        let input_ids = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)?;
        let attention_mask = Tensor::from_vec(
            encoded.get_attention_mask().to_vec(),
            (1, ids.len()),
            &self.device,
        )?;
        let token_type_ids = Tensor::from_vec(
            encoded.get_type_ids().to_vec(),
            (1, ids.len()),
            &self.device,
        )?;
        let logits = self
            .model
            .forward(&input_ids, &attention_mask, &token_type_ids)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .squeeze(0)?
            .to_vec2::<f32>()?;

        let word_ids = encoded.get_word_ids();
        let mut predicted = vec![None; words.len()];
        for (token_idx, word_id) in word_ids.iter().enumerate() {
            let Some(word_id) = word_id else {
                continue;
            };
            // Classify the final subtoken of each word, never BOS/EOS.
            if word_ids.get(token_idx + 1) == Some(&Some(*word_id)) {
                continue;
            }
            let word_index = *word_id as usize;
            if word_index >= words.len() {
                bail!("punctuation tokenizer returned an invalid word id");
            }
            predicted[word_index] =
                best_punctuation(&logits[token_idx], &self.labels, min_confidence)?;
        }
        Ok(predicted)
    }
}

fn best_punctuation(
    logits: &[f32],
    labels: &[Option<char>],
    min_confidence: f32,
) -> Result<Option<char>> {
    if logits.len() != labels.len() || logits.iter().any(|x| !x.is_finite()) {
        bail!("invalid punctuation classifier logits");
    }
    let (winning_index, maximum) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .context("empty punctuation classifier logits")?;
    let total: f32 = logits.iter().map(|x| (*x - *maximum).exp()).sum();
    if total <= 0.0 || !total.is_finite() {
        bail!("invalid punctuation probabilities");
    }
    let confidence = 1.0 / total;
    Ok(if confidence >= min_confidence {
        labels[winning_index]
    } else {
        None
    })
}

fn render_punctuation(words: &[&str], marks: &[Option<char>]) -> String {
    let mut result = String::new();
    for (index, (word, mark)) in words.iter().zip(marks).enumerate() {
        if index != 0 {
            result.push(' ');
        }
        result.push_str(word);
        // Never duplicate explicit marks from the original ASR text.
        if !matches!(
            word.chars().last(),
            Some('.' | ',' | '?' | '!' | ':' | ';' | '-')
        ) {
            if let Some(mark) = mark {
                result.push(*mark);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_mark_after_last_word() {
        let words = ["Est-ce", "que", "tu", "viens", "demain"];
        let marks = [None, None, None, None, Some('?')];
        assert_eq!(
            render_punctuation(&words, &marks),
            "Est-ce que tu viens demain?"
        );
    }

    #[test]
    fn existing_punctuation_is_not_duplicated() {
        let words = ["Est-ce", "que", "tu", "viens", "demain?"];
        let marks = [None, None, None, None, Some('?')];
        assert_eq!(
            render_punctuation(&words, &marks),
            "Est-ce que tu viens demain?"
        );
    }

    #[test]
    fn reject_non_finite_logits() {
        assert!(best_punctuation(&[f32::NAN, 0.0], &[None, Some('?')], 0.5).is_err());
    }

    #[test]
    fn no_forced_punctuation_below_threshold() {
        assert_eq!(
            best_punctuation(&[0.0, 1.0], &[None, Some('?')], 0.95).unwrap(),
            None
        );
    }
}
