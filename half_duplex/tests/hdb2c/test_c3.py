import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]))
from hdb2c.c3 import transform_conformer, transform_probe, compare_stages, STAGES

class C3Tests(unittest.TestCase):
    def test_transform_conformer_strict(self):
        source='''        let mut pos_emb = None;
        if let Some(cache) = cache.as_mut() {
            for (layer, cache)
            for layer in &self.layers {
                x = layer.forward(&x, pos_emb.as_ref(), None)?;
            }
'''
        t=transform_conformer(source)
        self.assertIn('HDB2C3_TRACE_INSTRUMENTATION_OFFLINE_CPU_V1',t)
        self.assertIn('hdb2c3_trace_stage("pre_encode", &x)?;',t)
        self.assertIn('format!("layer_{index:02}")',t)
        with self.assertRaises(ValueError):transform_conformer(t)
    def test_probe_requires_exact_anchor(self):
        pinned='''    let mel=get_logmel(&audio,&pre,&dev)?;
    let (features,lengths)=match &mut model {
        ParakeetModel::Tdt(m) => m.encoder.forward(&mel,None)?,
        _ => bail!("not tdt"),
    };
    let meta=json!({"stage":"test",
        "mel":mel_meta,"encoder":feat_meta,"encoder_valid_frames":lengths[0],
        "note":"probe"});
'''
        t=transform_probe(pinned)
        self.assertIn('HDB2C3_NEMO_MEL_PATH',t)
        self.assertIn('HDB2C3_NEMO_MEL_VALID_FRAMES',t)
        self.assertIn('m.encoder.forward(&mel,encoder_input_lengths.as_ref())?',t)
        self.assertIn('"injected_mel_valid_frames":injected_mel_valid_frames',t)
        self.assertNotIn('m.encoder.forward(&mel,None)?',t)
        with self.assertRaises(ValueError):transform_probe('let mel=bad;')
        with self.assertRaises(ValueError):transform_probe(t)
    def test_first_layer_divergence_detected(self):
        from tempfile import TemporaryDirectory
        import json
        with TemporaryDirectory() as t:
            root=Path(t);n=root/'n';c=root/'c';n.mkdir();c.mkdir()
            for index,name in enumerate(STAGES):
                for folder in (n,c):
                    data=np.ones((1,4,3),dtype='<f4')
                    if folder==c and index>=3:data[:,:2,:]+=0.1
                    data.tofile(folder/f'{name}.f32le')
                    (folder/f'{name}.json').write_text(json.dumps({'shape':[1,4,3]}))
            result=compare_stages(n,c,valid_frames=2)
            self.assertEqual(result['first_divergence'],'layer_01')
            self.assertEqual(result['status'],'FIRST_DIVERGENCE_LOCALIZED')
    def test_padding_ignored(self):
        from tempfile import TemporaryDirectory
        import json
        with TemporaryDirectory() as t:
            root=Path(t);n=root/'n';c=root/'c';n.mkdir();c.mkdir()
            for name in STAGES:
                for folder in (n,c):
                    data=np.ones((1,4,3),dtype='<f4')
                    if folder==c:data[:,2:,:]+=3.0
                    data.tofile(folder/f'{name}.f32le')
                    (folder/f'{name}.json').write_text(json.dumps({'shape':[1,4,3]}))
            result=compare_stages(n,c,valid_frames=2)
            self.assertEqual(result['status'],'ALL_STAGES_NUMERIC_PARITY')
    def test_nemo_valid_reference_ignores_padding_but_rejects_drift(self):
        from hdb2c.c3_nemo import compare_nemo_reference
        a=np.ones((1,3,5),dtype=np.float32)
        b=a.copy();b[...,3:]=100.0
        ok=compare_nemo_reference(a,b,3)
        self.assertEqual(ok['status'],'PASS_VALID_FRAMES_NUMERIC_REPRODUCIBILITY')
        self.assertGreater(ok['max_abs_all_frames'],90)
        b[...,1]+=0.01
        bad=compare_nemo_reference(a,b,3)
        self.assertEqual(bad['status'],'NEMO_SELF_REFERENCE_DIVERGENCE')
    def test_nemo_valid_reference_tolerance_and_shape(self):
        from hdb2c.c3_nemo import compare_nemo_reference
        a=np.ones((1,3,5),dtype=np.float32)
        b=a.copy();b[...,0]+=1e-5
        self.assertEqual(compare_nemo_reference(a,b,4)['status'],
                         'PASS_VALID_FRAMES_NUMERIC_REPRODUCIBILITY')
        self.assertEqual(compare_nemo_reference(a,b,6)['status'],
                         'SHAPE_OR_VALID_LENGTH_MISMATCH')
    def test_c3_r2_signature_isolated_from_c2_and_r1(self):
        from hdb2c.c3 import SIGNATURE
        self.assertIn('r2',SIGNATURE)
        self.assertIn('explicit-valid-frames',SIGNATURE)
    def test_mel_valid_frame_gate(self):
        from hdb2c.c3 import validate_mel_valid_frames
        import numpy as np
        self.assertEqual(validate_mel_valid_frames(np.array([10]),12),10)
        for lengths,total in ((np.array([0]),12),(np.array([13]),12),
                              (np.array([1,2]),12),(np.array([3]),0)):
            with self.assertRaises(ValueError):
                validate_mel_valid_frames(lengths,total)
if __name__=='__main__':unittest.main()