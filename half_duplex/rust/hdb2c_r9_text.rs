//! Original SentencePiece unknown surface, for the verified checkpoint subset.
//! Metadata is extracted from the original .nemo; no guessed special-token IDs.
use candle::{Error, Result};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SentencePieceText {
    pub model_sha256: String,
    pub vocabulary_size: usize,
    pub unknown_id: usize,
    pub unknown_piece: String,
    pub unknown_surface: String,
}

impl SentencePieceText {
    pub fn validate(&self, vocabulary: &[String]) -> Result<()> {
        if self.model_sha256.len()!=64 || self.vocabulary_size!=vocabulary.len()
            || vocabulary.get(self.unknown_id)!=Some(&self.unknown_piece) {
            return Err(Error::Msg("R9 tokenizer metadata/vocabulary mismatch".into()));
        }
        Ok(())
    }

    pub fn piece(&self, id: usize, vocabulary: &[String]) -> Result<String> {
        let piece=vocabulary.get(id).ok_or_else(||Error::Msg("R9 token ID out of range".into()))?;
        Ok(if id==self.unknown_id {self.unknown_surface.clone()} else {piece.replace('▁'," ")})
    }

    pub fn text(&self, ids: &[usize], vocabulary: &[String]) -> Result<String> {
        let mut output=String::new();
        for &id in ids {output.push_str(&self.piece(id,vocabulary)?);}
        Ok(output.trim().to_string())
    }
}

#[cfg(test)]
mod hdb2c9_tests {
    use super::*;
    fn fixture()->(SentencePieceText,Vec<String>){
        (SentencePieceText{model_sha256:"a".repeat(64),vocabulary_size:3,
          unknown_id:0,unknown_piece:"<unk>".into(),unknown_surface:" ⁇ ".into()},
         vec!["<unk>".into(),"▁bon".into(),"jour".into()])
    }
    #[test]
    fn original_unknown_surface_preserves_its_internal_spaces(){
        let(s,v)=fixture();s.validate(&v).unwrap();
        assert_eq!(s.text(&[1,2,0,1,2],&v).unwrap(),"bonjour ⁇  bonjour");
        assert_eq!(s.piece(0,&v).unwrap()," ⁇ ");
    }
    #[test]
    fn ordinary_subwords_and_empty_hypothesis(){
        let(s,v)=fixture();assert_eq!(s.text(&[1,2],&v).unwrap(),"bonjour");
        assert_eq!(s.text(&[],&v).unwrap(),"");
    }
    #[test]
    fn invalid_id_or_mismatched_metadata_is_rejected(){
        let(mut s,v)=fixture();assert!(s.text(&[3],&v).is_err());
        s.unknown_id=1;assert!(s.validate(&v).is_err());
    }
}
