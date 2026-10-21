import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from sentencepiece import sentencepiece_model_pb2

from hdb2c.c3_r9_source import metadata_from_proto, model_source, original_metadata
from hdb2c.c3_r9_text import strict_numeric


class TextCandidateTests(unittest.TestCase):
    def proto(self):
        p=sentencepiece_model_pb2.ModelProto()
        p.normalizer_spec.add_dummy_prefix=True
        p.normalizer_spec.remove_extra_whitespaces=False
        p.trainer_spec.unk_surface=' ⁇ '
        for text,kind in [('▁bonjour',1),('[noise]',4),('<unknown>',2)]:
            piece=p.pieces.add();piece.piece=text;piece.type=kind;piece.score=0
        return p

    def test_original_metadata_defines_unknown_id_and_surface(self):
        p=self.proto();v=[x.piece for x in p.pieces]
        spec=metadata_from_proto(p.SerializeToString(),v)
        self.assertEqual(spec['unknown_id'],2)
        self.assertEqual(spec['unknown_piece'],'<unknown>')
        self.assertEqual(spec['unknown_surface'],' ⁇ ')
        self.assertEqual(len(spec['model_sha256']),64)

    def test_unsupported_semantics_and_wrong_vocabulary_fail_closed(self):
        for change in ('control','byte','denormalizer','extra_whitespace','vocabulary'):
            p=self.proto();v=[x.piece for x in p.pieces]
            if change=='control':p.pieces[0].type=3
            if change=='byte':p.pieces[0].type=6
            if change=='denormalizer':p.denormalizer_spec.precompiled_charsmap=b'x'
            if change=='extra_whitespace':p.normalizer_spec.remove_extra_whitespaces=True
            if change=='vocabulary':v[0]='wrong'
            with self.subTest(change=change),self.assertRaises(ValueError):
                metadata_from_proto(p.SerializeToString(),v)

    def test_original_nemo_uri_resolves_the_embedded_asset(self):
        p=self.proto();data=p.SerializeToString()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir()
            (model/'config.json').write_text(json.dumps({'tokenizer':{'model_path':'nemo:hash_tokenizer.model'},
                'joint':{'vocabulary':[x.piece for x in p.pieces]}}))
            archive=root/'original.nemo'
            with tarfile.open(archive,'w') as tar:
                entry=tarfile.TarInfo('./hash_tokenizer.model');entry.size=len(data)
                tar.addfile(entry,io.BytesIO(data))
            self.assertEqual(original_metadata(archive,model)['unknown_id'],2)

    def test_text_success_cannot_promote_failed_neural_gate(self):
        baseline={'frontend':[{'numeric_pass':False}], 'common_input_frontend':[],
            'common_input_subsampling':[],'common_input_tdt':{'numeric_pass':True,'argmax_exact':True}}
        tdt={'native':{'numeric_pass':True,'tdt_discrete_exact':True}}
        self.assertFalse(strict_numeric(baseline,{},tdt))
        with self.assertRaises(ValueError):model_source('unverified source')
