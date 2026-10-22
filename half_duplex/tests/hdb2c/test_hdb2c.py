import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hdb2c.audio import describe,normalize
from hdb2c.metrics import wer,normalize_text,compare_arrays
from hdb2c.compare import compare_texts,compare_probes
from hdb2c.probe import install
from hdb2c.cli import parse_candle_json
from hdb2c.static_audit import audit

class HDB2CContractTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.path=Path(self.t.name)
    def json(self,name,obj):
        p=self.path/name;p.parent.mkdir(exist_ok=True,parents=True)
        p.write_text(json.dumps(obj));return p
    def test_text_whitespace_accent(self):
        self.assertEqual(normalize_text('Bonjour,  À toi !'), 'bonjour à toi')
    def test_wer(self):
        self.assertEqual(wer('bonjour le monde','bonjour la monde')['wer'],1/3)
        self.assertEqual(wer('un deux trois','un deux trois')['wer'],0.)
    def test_wer_empty_reference_rejected(self):
        with self.assertRaises(ValueError):wer('','bonjour')
    def test_tensor_identical(self):
        x=np.array([[1.,2.],[3.,4.]])
        self.assertEqual(compare_arrays(x,x,'mel')['status'],'PASS_NUMERIC_PARITY')
    def test_tensor_unequal(self):
        x=np.ones((2,3));y=x*2
        self.assertEqual(compare_arrays(x,y,'mel')['status'],'NUMERIC_MISMATCH')
    def test_tensor_shape_mismatch(self):
        self.assertEqual(compare_arrays(np.ones((2,3)),np.ones((2,4)),'mel')['status'],'SHAPE_MISMATCH')
    def test_nan_rejected(self):
        self.assertEqual(compare_arrays([float('nan')],[1],'mel')['status'],'NONFINITE')
    def test_missing_real_nemo_rejected(self):
        n=self.json('n.json',{'status':'NOT_QUALIFIED','pcm_sha256':'x'})
        c=self.json('c.json',{'status':'REAL_CANDLE_ASR','pcm_sha256':'x'})
        with self.assertRaises(ValueError):compare_texts(n,c,self.path/'o.json')
    def test_pcm_mismatch_rejected(self):
        n=self.json('n.json',{'status':'REAL_NEMO_ASR','pcm_sha256':'a'})
        c=self.json('c.json',{'status':'REAL_CANDLE_ASR','pcm_sha256':'b'})
        with self.assertRaises(ValueError):compare_texts(n,c,self.path/'o.json')
    def test_c1_without_gold_no_fake_wer(self):
        n=self.json('n.json',{'status':'REAL_NEMO_ASR','pcm_sha256':'a','text':'bonjour','checkpoint_sha256':'h'})
        c=self.json('c.json',{'status':'REAL_CANDLE_ASR','pcm_sha256':'a','text':'bonsoir','checkpoint_source_nemo_sha256':'h'})
        o=compare_texts(n,c,self.path/'out.json')
        self.assertNotIn('candle_wer',o)
        self.assertEqual(o['ground_truth_status'],'NOT_PROVIDED')
    def test_c1_gold_provided(self):
        n=self.json('n.json',{'status':'REAL_NEMO_ASR','pcm_sha256':'a','text':'bonjour','checkpoint_sha256':'h'})
        c=self.json('c.json',{'status':'REAL_CANDLE_ASR','pcm_sha256':'a','text':'bonsoir','checkpoint_source_nemo_sha256':'h'})
        (self.path/'ref.txt').write_text('bonjour')
        o=compare_texts(n,c,self.path/'out.json',self.path/'ref.txt')
        self.assertEqual(o['nemo_wer']['wer'],0)
        self.assertEqual(o['candle_wer']['wer'],1)
    def test_c1_source_checkpoint_mismatch(self):
        n=self.json('n.json',{'status':'REAL_NEMO_ASR','pcm_sha256':'a','text':'bonjour','checkpoint_sha256':'h'})
        c=self.json('c.json',{'status':'REAL_CANDLE_ASR','pcm_sha256':'a','text':'bonjour','checkpoint_source_nemo_sha256':'other'})
        o=compare_texts(n,c,self.path/'out.json')
        self.assertEqual(o['status'],'CHECKPOINT_SOURCE_MISMATCH')
    def test_candle_json_strict(self):
        self.assertEqual(parse_candle_json('{"text":"Bonjour"}')['text'],'Bonjour')
        with self.assertRaises(RuntimeError):parse_candle_json('loading...\n{"text":"Bonjour"}')
    def test_real_tensors_compare(self):
        nd=self.path/'n';cd=self.path/'c';nd.mkdir();cd.mkdir()
        self.json('n/nemo_report.json',{'status':'REAL_NEMO_ASR','probe_status':'REAL_NEMO_MEL_AND_ENCODER'})
        self.json('c/candle_probe.json',{'status':'REAL_CANDLE_MEL_AND_ENCODER',
           'mel':{'filename':'mel.f32le','shape':[1,2,3]},'encoder':{'filename':'enc.f32le','shape':[1,1,4]},'encoder_valid_frames':1})
        nemo_m=np.arange(6,dtype=np.float32).reshape(1,3,2)
        nemo_e=np.arange(4,dtype=np.float32).reshape(1,4,1)
        np.savez_compressed(nd/'nemo_mel.npz',values=nemo_m,valid_frames=np.array([2]))
        np.savez_compressed(nd/'nemo_encoder.npz',values=nemo_e,valid_frames=np.array([1]))
        nemo_m.transpose(0,2,1).astype('<f4').tofile(cd/'mel.f32le')
        nemo_e.transpose(0,2,1).astype('<f4').tofile(cd/'enc.f32le')
        o=compare_probes(nd,cd,self.path/'numeric.json')
        self.assertEqual(o['status'],'PASS_NUMERIC_MEL_AND_ENCODER_ONLY')
        self.assertEqual(o['mel']['status'],'PASS_NUMERIC_PARITY')
    def test_valid_frames_mismatch(self):
        nd=self.path/'n';cd=self.path/'c';nd.mkdir();cd.mkdir()
        self.json('n/nemo_report.json',{'status':'REAL_NEMO_ASR','probe_status':'REAL_NEMO_MEL_AND_ENCODER'})
        self.json('c/candle_probe.json',{'status':'REAL_CANDLE_MEL_AND_ENCODER',
            'mel':{'filename':'mel.f32le','shape':[1,3,2]},'encoder':{'filename':'enc.f32le','shape':[1,1,4]},'encoder_valid_frames':1})
        np.savez_compressed(nd/'nemo_mel.npz',values=np.ones((1,2,4)),valid_frames=np.array([2]))
        np.savez_compressed(nd/'nemo_encoder.npz',values=np.ones((1,4,1)),valid_frames=np.array([1]))
        np.ones((1,3,2),dtype='<f4').tofile(cd/'mel.f32le')
        np.ones((1,1,4),dtype='<f4').tofile(cd/'enc.f32le')
        o=compare_probes(nd,cd,self.path/'out.json')
        self.assertEqual(o['mel']['status'],'VALID_SHAPE_MISMATCH')
    def test_pcm_frame_sample(self):
        wav=self.path/'test.wav'
        with wave.open(str(wav),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes(bytes(2*16000))
        self.assertEqual(describe(wav)['frames'],16000)
        self.assertEqual(len(describe(wav)['pcm_sha256']),64)
    def test_static_audit_unverified(self):
        self.json('config.json',{'preprocessor':{'window':'hamming'}})
        a=audit(self.path)
        self.assertEqual(a['status'],'UNVERIFIED_SOURCE_DIFFERENCES')
    def test_probe_no_checkout(self):
        with self.assertRaises(FileNotFoundError):install(ROOT/'rust/hdb2c_probe.rs',self.path/'not_here')

if __name__=='__main__':unittest.main()
