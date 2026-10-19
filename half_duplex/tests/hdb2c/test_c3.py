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
        t=transform_probe('    let mel=get_logmel(&audio,&pre,&dev)?;')
        self.assertIn('HDB2C3_NEMO_MEL_PATH',t)
        with self.assertRaises(ValueError):transform_probe('let mel=bad;')
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
if __name__=='__main__':unittest.main()