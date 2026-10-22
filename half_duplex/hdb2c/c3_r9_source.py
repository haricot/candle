"""Isolated text-only correction after R8B, preserving neural arithmetic."""
import hashlib
import json
import tarfile

from .c3_r8_source import once


def metadata_from_proto(data, vocabulary):
    from sentencepiece import sentencepiece_model_pb2
    proto=sentencepiece_model_pb2.ModelProto.FromString(data)
    if vocabulary!=[p.piece for p in proto.pieces]:
        raise ValueError('R9 tokenizer vocabulary differs from the original model')
    # This is deliberately a qualified subset, not a generic SentencePiece port.
    if (any(p.type not in (1,2,4) for p in proto.pieces)
            or not proto.normalizer_spec.add_dummy_prefix
            or proto.normalizer_spec.remove_extra_whitespaces
            or proto.denormalizer_spec.precompiled_charsmap):
        raise ValueError('R9 unsupported SentencePiece decoding semantics')
    unknown=[i for i,p in enumerate(proto.pieces) if p.type==2]
    if len(unknown)!=1:raise ValueError('R9 requires one original unknown token')
    return {'model_sha256':hashlib.sha256(data).hexdigest(),'vocabulary_size':len(vocabulary),
        'unknown_id':unknown[0],'unknown_piece':vocabulary[unknown[0]],
        'unknown_surface':proto.trainer_spec.unk_surface}


def original_metadata(nemo, model_dir):
    cfg=json.loads((model_dir/'config.json').read_text())
    name=cfg['tokenizer']['model_path'].split('/')[-1].removeprefix('nemo:')
    with tarfile.open(nemo,'r:*') as archive:
        found=[m for m in archive.getmembers() if m.isfile() and m.name.removeprefix('./').split('/')[-1]==name]
        if len(found)!=1:raise ValueError('R9 original tokenizer asset is not unique')
        data=archive.extractfile(found[0]).read()
    return metadata_from_proto(data,cfg['joint']['vocabulary'])


def model_source(source):
    source=once(source,'    pub decoding: TdtDecodingArgs,',
        '    pub decoding: TdtDecodingArgs,\n'
        '    #[serde(default)] pub hdb2c9_tokenizer: Option<crate::parakeet::hdb2c_r9_text::SentencePieceText>,')
    source=once(source,'pub struct ParakeetTdt {',
        'pub struct ParakeetTdt {\n'
        '    pub hdb2c9_tokenizer: Option<crate::parakeet::hdb2c_r9_text::SentencePieceText>,')
    start=source.index('impl ParakeetTdt {');end=source.index('pub struct ParakeetRnnt',start)
    body=source[start:end]
    body=once(body,'        let vocabulary = args.joint.vocabulary.clone();',
        '        let vocabulary = args.joint.vocabulary.clone();\n'
        '        let hdb2c9_tokenizer=args.hdb2c9_tokenizer;\n'
        '        if let Some(ref spec)=hdb2c9_tokenizer {spec.validate(&vocabulary)?;}')
    body=once(body,'            vocabulary,\n            durations,',
        '            vocabulary,\n            hdb2c9_tokenizer,\n            durations,')
    for token in ('pred_token','token'):
        body=once(body,f'tokenizer::decode(&[{token}], &self.vocabulary)',f'self.hdb2c9_piece({token})?')
    body=once(body,'impl ParakeetTdt {','''impl ParakeetTdt {
    fn hdb2c9_piece(&self,id:usize)->Result<String>{
        match &self.hdb2c9_tokenizer {
            Some(spec)=>spec.piece(id,&self.vocabulary),
            None=>Ok(tokenizer::decode(&[id],&self.vocabulary)),
        }
    }
    pub fn hdb2c9_text(&self,ids:&[usize])->Result<String>{
        match &self.hdb2c9_tokenizer {
            Some(spec)=>spec.text(ids,&self.vocabulary),
            None=>Ok(tokenizer::decode(ids,&self.vocabulary).trim().to_string()),
        }
    }
''')
    return source[:start]+body+source[end:]


def outputs(parent,root,metadata):
    probe=(parent/'src/bin/hdb2c_r8_probe.rs').read_text()
    encoded=json.dumps(metadata,ensure_ascii=False,separators=(',',':'))
    if '"###' in encoded:raise ValueError('R9 metadata source delimiter collision')
    probe=once(probe,'    let mut model=from_config_value(cfg,vb)?;',
        '    let mut cfg=cfg;\n'
        f'    cfg["hdb2c9_tokenizer"]=serde_json::from_str(r###"{encoded}"###)?;\n'
        '    let mut model=from_config_value(cfg,vb)?;')
    probe=once(probe,'let text=parakeet_candle::parakeet::tokenizer::decode(&ids,&m.vocabulary).trim().to_string();',
        'let text=m.hdb2c9_text(&ids)?;')
    return {'src/bin/hdb2c_r9_probe.rs':probe,
        'src/parakeet/model.rs':model_source((parent/'src/parakeet/model.rs').read_text()),
        'src/parakeet/mod.rs':(parent/'src/parakeet/mod.rs').read_text()+'\npub mod hdb2c_r9_text;\n',
        'src/parakeet/hdb2c_r9_text.rs':(root/'rust/hdb2c_r9_text.rs').read_text()}
