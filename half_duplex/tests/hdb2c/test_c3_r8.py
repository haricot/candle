"""R8 evidence/gate contracts; fixtures here do not represent ASR inference."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hdb2c.c3_r8 import metric, selected_corpus, input_for, EMOTIONS
from hdb2c.c3_r8_source import frontend, probe


class R8Tests(unittest.TestCase):
    def test_relative_diagnostics_do_not_relax_absolute_gate(self):
        a=np.array([10000.,20000.],dtype=np.float32)
        b=a+np.float32(.01)
        result=metric(a,b)
        self.assertLess(result['relative_l2'],1e-5)
        self.assertFalse(result['numeric_pass'])
        self.assertEqual(result['numeric_limits']['max_abs'],1e-3)

    def test_invalid_and_incomplete_corpus_rejected(self):
        rows=[{'id':f'{s}-{e}','split':s,'emotion':e,'pcm_sha256':f'{s}-{e}','wav':f'{s}-{e}.wav'}
              for s in ('cafe','oreau') for e in EMOTIONS]
        with patch('hdb2c.audio.describe',side_effect=lambda p:{'pcm_sha256':p.stem}):
            self.assertEqual(len(selected_corpus({'rows':rows})),14)
            with self.assertRaises(ValueError):selected_corpus({'rows':rows[:-1]})
            with self.assertRaises(ValueError):selected_corpus({'rows':rows[:-1]+[rows[0]]})
            duplicated=[dict(r) for r in rows];duplicated[1]['pcm_sha256']=duplicated[0]['pcm_sha256']
            with self.assertRaises(ValueError):selected_corpus({'rows':duplicated})

    def test_changed_pcm_rejected_before_any_measurement(self):
        rows=[{'id':f'{s}-{e}','split':s,'emotion':e,'pcm_sha256':f'{s}-{e}','wav':'x.wav'}
              for s in ('cafe','oreau') for e in EMOTIONS]
        with patch('hdb2c.audio.describe',return_value={'pcm_sha256':'changed'}):
            with self.assertRaisesRegex(ValueError,'PCM changed'):selected_corpus({'rows':rows})

    def test_projection_input_uses_actual_channel_frequency_order(self):
        source=np.arange(24,dtype=np.float32).reshape(1,2,3,4)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);trace=root/'encoder_trace';trace.mkdir()
            (trace/'relu_06.f32le').write_bytes(source.tobytes())
            (trace/'relu_06.json').write_text(json.dumps({'shape':list(source.shape)}))
            actual=input_for('out',root)
            np.testing.assert_array_equal(actual[0,0],[0,1,2,3,12,13,14,15])
            np.testing.assert_array_equal(actual[0,2],[8,9,10,11,20,21,22,23])

    def test_unknown_native_source_fails_closed(self):
        with self.assertRaises(ValueError):frontend('unverified source','')
        with self.assertRaises(ValueError):probe('unverified source',operators=True)


class R8TorchPrimitiveTests(unittest.TestCase):
    def test_derived_frames_reproduce_actual_stft_bytes(self):
        import torch
        # A primitive framing control, not a fabricated transcription/model run.
        for size in (640,1001,4097):
            x=torch.sin(torch.arange(size,dtype=torch.float32)*.071).reshape(1,-1)
            window=torch.hann_window(400,periodic=False)
            actual=torch.stft(x,n_fft=512,hop_length=160,win_length=400,
                              window=window,center=True,pad_mode='constant',return_complex=True)
            frames=torch.nn.functional.pad(x,(256,256))[0].unfold(0,512,160)*torch.nn.functional.pad(window,(56,56))
            replay=torch.fft.rfft(frames)
            self.assertEqual(torch.view_as_real(actual)[0].permute(1,0,2).contiguous().numpy().tobytes(),
                             torch.view_as_real(replay).contiguous().numpy().tobytes())


if __name__=='__main__':unittest.main()
