import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from hdb2c.c2 import (transform_audio,transform_model,transform_probe,prepare,SIGNATURE,complete_hdb2a_unit_test_constructors)
from hdb2c.compare import compare_probes

class HDB2C2SourceTests(unittest.TestCase):
    def audio(self):
        return ('#[derive(Clone)]\npub struct PreprocessArgs {\n    pub mag_power: f64,\n}\n'
          'pub fn get_logmel(samples: &[f32], args: &PreprocessArgs, device: &Device) -> Result<Tensor> {\n'
          '    let old=1;\n}\n'
          '#[cfg(test)]\nmod tests {\nfn x() {\n'
          '            mag_power: 2.0,\n        }\n'
          '            mag_power: 2.0,\n        }\n}\n')
    def test_audio_complete_frontend_integration(self):
        out=transform_audio(self.audio())
        self.assertIn('mod hdb2c_nemo;',out)
        self.assertIn('hdb2c_nemo::get_logmel',out)
        self.assertNotIn('let old=1;',out)
        self.assertEqual(out.count('            exact_pad: false,'),2)
    def test_audio_reject_unknown(self):
        with self.assertRaises(ValueError):transform_audio('unexpected code')
    def test_audio_refuse_twice(self):
        with self.assertRaises(ValueError):transform_audio(transform_audio(self.audio()))
    @staticmethod
    def model_test_fixture():
        # Reproduces pinned n0madic/parakeet-candle test; HDB2A extends ConformerArgs.
        return ("#[cfg(test)]\nmod tests {\n"
                "    fn time_ratio_positive() {\n"
                "        let pre = PreprocessArgs {\n"
                "            mag_power: 2.0,\n"
                "        };\n"
                "        let enc = ConformerArgs {\n"
                "            att_context_size: None,\n"
                "        };\n"
                "        let tr = time_ratio(&pre, &enc);\n"
                "    }\n"
                "    #[test]\n    fn another_test() {}\n}\n")

    def test_model_legacy_unit_initializers_completed(self):
        target=('        if chunk_duration.is_none() {\n'
                '            let mel = get_logmel(&audio, &pre, &device)?;\n'
                '            return Ok(self.generate(&mel, config)?[0].clone());\n'
                '        }')
        out=transform_model(target+'\n'+self.model_test_fixture())
        self.assertIn('m.encoder.forward(&mel, Some(&valid_t))?',out)
        self.assertEqual(out.count('            exact_pad: false,'),1)
        self.assertEqual(out.count('            frame_splicing: None,'),1)
        self.assertEqual(out.count('            conv_norm_type: "batch_norm".to_owned(),'),1)
        self.assertEqual(out.count('            conv_context_size: None,'),1)
        self.assertIn('fn another_test()',out)
        self.assertIn('Some(&encoded_lengths)',out)
        self.assertEqual(out.count('hdb2c_valid_frames'),1)

    def test_unit_constructor_refuses_unrecognized_upstream(self):
        text=self.model_test_fixture().replace('mag_power: 2.0','mag_power: 2.5')
        with self.assertRaises(ValueError):complete_hdb2a_unit_test_constructors(text)

    def test_unit_constructor_refuses_reapplication(self):
        first=complete_hdb2a_unit_test_constructors(self.model_test_fixture())
        with self.assertRaises(ValueError):complete_hdb2a_unit_test_constructors(first)

    def test_model_target_exact(self):
        target=('        if chunk_duration.is_none() {\n'
                '            let mel = get_logmel(&audio, &pre, &device)?;\n'
                '            return Ok(self.generate(&mel, config)?[0].clone());\n'
                '        }')
        res=transform_model(target+'\n'+self.model_test_fixture())
        self.assertIn('m.encoder.forward(&mel, Some(&valid_t))?',res)
        self.assertIn('m.decode(&encoded, Some(&encoded_lengths)',res)
    def test_model_reject_unknown(self):
        with self.assertRaises(ValueError):transform_model('no match')
    def test_probe_valid_length(self):
        from hdb2c.cli import ROOT
        raw=(ROOT/'rust/hdb2c_probe.rs').read_text()
        got=transform_probe(raw)
        self.assertIn('nemo_valid_mel_frames',got)
        self.assertIn('forward(&mel,Some(&valid_t))',got)
    def test_complete_isolated_candidate_and_idempotence(self):
        from hdb2c.cli import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);runtime=base/'runtime';candidate=base/'runtime-c2';model=base/'model'
            (runtime/'src/parakeet').mkdir(parents=True);(runtime/'src').mkdir(exist_ok=True)
            model.mkdir()
            (runtime/'src/parakeet/audio.rs').write_text(self.audio())
            template=('        if chunk_duration.is_none() {\n'
                '            let mel = get_logmel(&audio, &pre, &device)?;\n'
                '            return Ok(self.generate(&mel, config)?[0].clone());\n'
                '        }')
            (runtime/'src/parakeet/model.rs').write_text(template+'\n'+self.model_test_fixture())
            (runtime/'src/main.rs').write_text('// HDB2A_CANDLE_LOCAL_MODEL: test fixture')
            (runtime/'Cargo.toml').write_text('[package]\nname="fixture"\nversion="0.0.1"')
            (runtime/'target').mkdir();(runtime/'target/SHOULD_NOT_COPY').write_text('test')
            (model/'config.json').write_text(json.dumps({'preprocessor':{'normalize':'NA','n_fft':512,'features':128}}))
            m1=prepare(runtime,candidate,ROOT,model)
            self.assertEqual(m1['signature'],SIGNATURE)
            self.assertTrue((candidate/'src/parakeet/hdb2c_nemo.rs').exists())
            self.assertTrue((candidate/'src/bin/hdb2c_probe.rs').exists())
            self.assertFalse((candidate/'target').exists())
            self.assertEqual(m1,prepare(runtime,candidate,ROOT,model))
            (candidate/'src/parakeet/audio.rs').write_text('user modified')
            with self.assertRaises(RuntimeError):prepare(runtime,candidate,ROOT,model)
    def test_default_candidate_uses_current_version(self):
        from hdb2c.cli import ROOT
        source=(ROOT/'hdb2c/c2.py').read_text()
        self.assertIn("'-hdb2c-c2-v0.6.6'",source)
        self.assertNotIn("'-hdb2c-c2-v0.6.5'",source)
    def test_signature_fixed(self):self.assertEqual(SIGNATURE,'hdb2c-c2-nemo-frontend-v0.6.6')

class HDB2C2LengthGate(unittest.TestCase):
    def test_explicit_mel_valid_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);out=base/'candle';out.mkdir()
            arr=np.ones((1,128,16),dtype=np.float32)
            enc=np.ones((1,640,2),dtype=np.float32)
            np.savez(base/'nemo_mel.npz',values=arr,valid_frames=np.array([13]))
            np.savez(base/'nemo_encoder.npz',values=enc,valid_frames=np.array([2]))
            (base/'nemo_report.json').write_text(json.dumps({'status':'REAL_NEMO_ASR','probe_status':'REAL_NEMO_MEL_AND_ENCODER'}))
            np.ones((1,16,128),dtype='<f4').tofile(out/'candle_mel.f32le')
            np.ones((1,2,640),dtype='<f4').tofile(out/'candle_encoder.f32le')
            meta={'status':'REAL_CANDLE_MEL_AND_ENCODER','nemo_valid_mel_frames':13,'encoder_valid_frames':2,
                  'mel':{'shape':[1,16,128],'filename':'candle_mel.f32le'},
                  'encoder':{'shape':[1,2,640],'filename':'candle_encoder.f32le'}}
            (out/'candle_probe.json').write_text(json.dumps(meta))
            got=compare_probes(base,out,base/'result.json')
            self.assertEqual(got['mel']['status'],'PASS_NUMERIC_PARITY')
            self.assertEqual(got['encoder']['status'],'PASS_NUMERIC_PARITY')
            del meta['nemo_valid_mel_frames']
            (out/'candle_probe.json').write_text(json.dumps(meta))
            got=compare_probes(base,out,base/'result.json')
            self.assertEqual(got['mel']['status'],'VALID_SHAPE_MISMATCH')

if __name__=='__main__':unittest.main()
