import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]))
from hdb2c.c3 import transform_conformer, transform_probe, compare_stages, STAGES
from hdb2c.c3_r3 import transform_dw_source, compare_dw, stage_valid_frames, metrics

class C3Tests(unittest.TestCase):
    def test_transform_conformer_strict(self):
        source='''        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for (idx, conv) in self.conv.iter().enumerate() {
            x = conv.forward(&x)?;
            if idx == 0 || idx % 2 == 0 {
                x = x.relu()?;
            }
        }
        let x = x.transpose(1, 2)?; // (B, T, C, F)
        let (b, t, c, f) = x.dims4()?;
        let x = x.reshape((b, t, c * f))?;
        let x = self.out.forward(&x)?;
        let mut pos_emb = None;
        if let Some(cache) = cache.as_mut() {
            for (layer, cache)
            for layer in &self.layers {
                x = layer.forward(&x, pos_emb.as_ref(), None)?;
            }
'''
        t=transform_conformer(source)
        self.assertIn('HDB2C3_TRACE_INSTRUMENTATION_OFFLINE_CPU_V1',t)
        self.assertIn('HDB2C3_R3_DW_TRACE_CPU',t)
        self.assertIn('hdb2c3_trace_stage("pre_encode", &x)?;',t)
        self.assertIn('format!("layer_{index:02}")',t)
        with self.assertRaises(ValueError):transform_conformer(t)
    def test_probe_requires_exact_anchor(self):
        pinned='''    let mel=get_logmel(&audio,&pre,&dev)?;
    let valid=parakeet_candle::parakeet::audio::hdb2c_valid_frames(audio.len(),&pre)?;
    let valid_t=Tensor::from_vec(vec![valid as i64],(1,),&dev)?;
    let (features,lengths)=match &mut model {
        ParakeetModel::Tdt(m) => m.encoder.forward(&mel,Some(&valid_t))?,
        _ => bail!("not tdt"),
    };
    let meta=json!({"stage":"test",
        "mel":mel_meta,"encoder":feat_meta,"encoder_valid_frames":lengths[0],
        "note":"probe"});
'''
        t=transform_probe(pinned)
        self.assertIn('HDB2C3_NEMO_MEL_PATH',t)
        self.assertIn('HDB2C3_NEMO_MEL_VALID_FRAMES',t)
        self.assertIn('m.encoder.forward(&mel,encoder_input_lengths.as_ref().or(Some(&valid_t)))?',t)
        self.assertIn('"injected_mel_valid_frames":injected_mel_valid_frames',t)
        self.assertNotIn('m.encoder.forward(&mel,Some(&valid_t))?',t)
        with self.assertRaises(ValueError):
            transform_probe(pinned.replace('Some(&valid_t)','None'))
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
        self.assertIn('r3',SIGNATURE)
        self.assertIn('dw-weight-operation-traces',SIGNATURE)
    def test_mel_valid_frame_gate(self):
        from hdb2c.c3 import validate_mel_valid_frames
        import numpy as np
        self.assertEqual(validate_mel_valid_frames(np.array([10]),12),10)
        for lengths,total in ((np.array([0]),12),(np.array([13]),12),
                              (np.array([1,2]),12),(np.array([3]),0)):
            with self.assertRaises(ValueError):
                validate_mel_valid_frames(lengths,total)
    def test_r3_dw_pinned_forward_instrumentation(self):
        fragment='''        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for (idx, conv) in self.conv.iter().enumerate() {
            x = conv.forward(&x)?;
            if idx == 0 || idx % 2 == 0 {
                x = x.relu()?;
            }
        }
        let x = x.transpose(1, 2)?; // (B, T, C, F)
        let (b, t, c, f) = x.dims4()?;
        let x = x.reshape((b, t, c * f))?;
        let x = self.out.forward(&x)?;
'''
        src=transform_dw_source(fragment)
        self.assertIn('HDB2C3_R3_DW_TRACE_CPU',src)
        self.assertIn('conv_{nemo_idx:02}',src)
        self.assertIn('weight_out',src)
        with self.assertRaises(ValueError):transform_dw_source(src)
    def test_r3_valid_length_sequence(self):
        self.assertEqual(stage_valid_frames(394)['out'],50)
        self.assertEqual(stage_valid_frames(403)['out'],51)
    def test_r3_comparison_first_operation_and_weights(self):
        import json
        from hdb2c.c3_r3 import WEIGHTS,STAGES as DW_STAGES
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as t:
            root=Path(t); n=root/'n';c=root/'c';n.mkdir();c.mkdir()
            for name in (*WEIGHTS,*DW_STAGES):
                shape=(1,2,4,3) if name!='out' and name in DW_STAGES else (1,5,3)
                if name in WEIGHTS:shape=(2,3)
                for p in (n,c):
                    a=np.ones(shape,dtype='<f4')
                    if name=='conv_02' and p==c:a[...] += .1
                    a.tofile(p/f'{name}.f32le')
                    (p/f'{name}.json').write_text(json.dumps({'shape':list(shape)}))
            # synthetic masks would require valid axis 2 >=3, and output length 1.
            # For conv00 valid=2, conv02=1; the objective is first mismatch.
            got=compare_dw(n,c,mel_valid=3,encoder_valid=1)
            self.assertEqual(got['first_operation_divergence'],'conv_02')
            self.assertIsNone(got['first_weight_mismatch'])
            self.assertEqual(got['weights_status'],'PASS_NUMERIC_PARITY')
    def test_r3_shape_mismatch_fails_closed(self):
        x=np.ones((1,2,3,4),dtype=np.float32)
        y=np.ones((1,2,4,4),dtype=np.float32)
        self.assertEqual(metrics(x,y,axis=2,valid=3)['status'],'SHAPE_MISMATCH')
    def test_r3c_mask_exact_pinned_source_only(self):
        from hdb2c.c3_r3c import transform_masked_dw_source, MARKER
        source='''    fn forward(&self, x: &Tensor, lengths: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut lengths = lengths.to_dtype(DType::F32)?;
        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for (idx, conv) in self.conv.iter().enumerate() {
            x = conv.forward(&x)?;
            if idx == 0 || idx % 2 == 0 {
                x = x.relu()?;
            }
        }
        let x = x.transpose(1, 2)?; // (B, T, C, F)
        let (b, t, c, f) = x.dims4()?;
        let x = x.reshape((b, t, c * f))?;
        let x = self.out.forward(&x)?;
'''
        traced=transform_dw_source(source)
        masked=transform_masked_dw_source(traced)
        self.assertIn(MARKER,masked)
        self.assertIn('let hdb2c3_original_mel_lengths=lengths.to_device(&candle::Device::Cpu)?.to_vec1::<i64>()?;',masked)
        self.assertIn('let initial=hdb2c3_original_mel_lengths;',masked)
        self.assertEqual(masked.count('hdb2c3_r3c_mask_time(&x,&current)?'),3)
        self.assertIn('for v in current.iter_mut(){*v=(*v+1)/2;}',masked)
        with self.assertRaises(ValueError):transform_masked_dw_source(masked)
        with self.assertRaises(ValueError):transform_masked_dw_source(source)
    def test_r3c_metadata_requires_original_r3_probe(self):
        from hdb2c.c3_r3c import transform_mask_probe
        original='"encoder_length_argument":"EXPLICIT_NEMO_VALID_FRAMES_IF_INJECTED",'
        self.assertIn('"dw_mask_propagation":"NEMO_MASKED_CONV_SEQUENTIAL"',
                      transform_mask_probe(original))
        with self.assertRaises(ValueError):transform_mask_probe('bad')
    def test_r3c_summary_explicitly_separates_encoder_and_tdt(self):
        from hdb2c.c3_r3c import summarize_ab
        names=('conv_00','relu_00','conv_02','conv_03','relu_03',
               'conv_05','conv_06','relu_06','out')
        stages=[{'name':name,'status':'NUMERIC_MISMATCH','rmse':1.0,
                 'cosine':.9,'max_abs':2.0} for name in names]
        layers=[{'name':('pre_encode' if i==0 else f'layer_{i:02}'),
                  'status':'NUMERIC_MISMATCH'} for i in range(26)]
        before={'weights_status':'PASS_NUMERIC_PARITY',
            'operations':stages,'first_operation_divergence':'conv_02'}
        after={'weights_status':'PASS_NUMERIC_PARITY',
            'operations':stages,'first_operation_divergence':'conv_02',
            'status':'FIRST_DW_DIVERGENCE_LOCALIZED'}
        enc={'stages':layers,'first_divergence':'pre_encode',
             'status':'FIRST_DIVERGENCE_LOCALIZED'}
        r=summarize_ab(before,after,enc,enc,'a','b')
        self.assertEqual(r['tdt_numeric_parity'],'NOT_TESTED')
        self.assertEqual(r['pre_encode_status'],'NOT_QUALIFIED')
        self.assertEqual(r['status'],'DW_NUMERIC_PARITY_NOT_ESTABLISHED')
if __name__=='__main__':unittest.main()