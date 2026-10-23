"""Small, stdlib-only contracts for native QVAC SafeTensors; no model download."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from hdb2a_rnnt_patch import patch_rnnt
from prepare_qvac_official import inspect_weights, prepare_official
from run_candle_hdb2a import checkpoint_identity, QVAC_SAFETENSORS_SHA256


def config():
    return {
        'target': 'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel',
        'preprocessor': {'sample_rate': 16000, 'normalize': 'per_feature',
                         'window_size': 0.025, 'window_stride': 0.01,
                         'window': 'hann', 'features': 128, 'n_fft': 512,
                         'dither': 0.0},
        'encoder': {'feat_in': 128, 'n_layers': 1, 'd_model': 16, 'n_heads': 4,
                    'ff_expansion_factor': 4, 'subsampling_factor': 8,
                    'self_attention_model': 'rel_pos', 'subsampling': 'dw_striding',
                    'conv_kernel_size': 9, 'subsampling_conv_channels': 8,
                    'pos_emb_max_len': 5000, 'conv_norm_type': 'layer_norm'},
        'decoder': {'blank_as_pad': True, 'vocab_size': 2,
                    'prednet': {'pred_hidden': 8, 'pred_rnn_layers': 1}},
        'joint': {'num_classes': 2, 'vocabulary': ['▁oui', '▁non'],
                  'jointnet': {'joint_hidden': 8, 'activation': 'relu',
                               'encoder_hidden': 16, 'pred_hidden': 8}},
        'decoding': {'model_type': 'tdt', 'durations': [0, 1, 2, 4]},
    }


def small_weights(path: Path, *, missing=None, wrong_shape=None):
    tensors = {
        'decoder.prediction.dec_rnn.lstm.weight_ih_l0': [32, 8],
        'decoder.prediction.dec_rnn.lstm.weight_hh_l0': [32, 8],
        'decoder.prediction.dec_rnn.lstm.bias_ih_l0': [32],
        'decoder.prediction.dec_rnn.lstm.bias_hh_l0': [32],
        'encoder.pre_encode.conv.0.weight': [8, 1, 3, 3],
        'encoder.layers.0.norm_feed_forward1.weight': [16],
        'decoder.prediction.embed.weight': [3, 8],
        'joint.joint_net.2.weight': [7, 8],
    }
    if missing:
        del tensors[missing]
    if wrong_shape:
        tensors[wrong_shape] = [4, 8]
    meta = {}
    pos = 0
    for name, shape in tensors.items():
        count = 1
        for n in shape:
            count *= n
        size = count * 4
        meta[name] = {'dtype': 'F32', 'shape': shape,
                      'data_offsets': [pos, pos + size]}
        pos += size
    head = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    head += b' ' * (-len(head) % 8)
    with path.open('wb') as f:
        f.write(struct.pack('<Q', len(head)))
        f.write(head)
        f.write(bytes(pos))
    return hashlib.sha256(path.read_bytes()).hexdigest(), len(tensors), path.stat().st_size


def rnnt_fragment():
    # Exact anchors retained from the pinned n0madic RNNT loader.
    weight = '''        fn load_weight(
            vb_layer: &VarBuilder,
            vb_root: &VarBuilder,
            vb_nemo: &VarBuilder,
            base: &str,
        ) -> Result<Tensor> {
            let candidates = [
                (vb_layer, base),
                (vb_root, base),
            ];
        }
'''
    bias = '''        fn load_bias(
            vb_layer: &VarBuilder,
            vb_root: &VarBuilder,
            vb_nemo: &VarBuilder,
            base: &str,
        ) -> Result<Tensor> {
            let candidates = [
                (vb_layer, base),
                (vb_root, base),
            ];
            Tensor::zeros((size,), vb_root.dtype(), vb_root.device())
        }
'''
    return (weight + bias +
            '        let vb_nemo = vb.pp("lstm").pp(format!("{layer_idx}"));\n' +
            ('            &vb_nemo,\n' * 4))


class QvacOfficialTests(unittest.TestCase):
    def test_rnnt_native_layout_patch_idempotent_and_drift_guard(self):
        original = rnnt_fragment()
        patched = patch_rnnt(original)
        self.assertEqual(patched, patch_rnnt(patched))
        self.assertEqual(patched.count('HDB2A_QVAC_NATIVE_LSTM'), 1)
        self.assertEqual(patched.count('(vb_native, layer_name.as_str())'), 2)
        self.assertEqual(patched.count('&vb_native,'), 4)
        self.assertIn('missing native QVAC LSTM bias', patched)
        with self.assertRaisesRegex(ValueError, 'Pin drift'):
            patch_rnnt('incompatible upstream RNNT implementation')

    def test_inspect_native_checkpoint_header(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.safetensors'
            sha, count, size = small_weights(path)
            details = inspect_weights(path, config(), expected_bytes=size,
                                      expected_tensors=count, expected_sha256=sha)
            self.assertEqual(details['tensor_count'], 8)
            self.assertEqual(details['sha256'], sha)

            with self.assertRaisesRegex(ValueError, 'SHA256'):
                inspect_weights(path, config(), expected_bytes=size,
                                expected_tensors=count, expected_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'tensor count'):
                inspect_weights(path, config(), expected_bytes=size,
                                expected_tensors=count + 1, expected_sha256=sha)
            sha2, count2, size2 = small_weights(path, wrong_shape=
                'decoder.prediction.dec_rnn.lstm.weight_ih_l0')
            with self.assertRaisesRegex(ValueError, 'native tensor'):
                inspect_weights(path, config(), expected_bytes=size2,
                                expected_tensors=count2, expected_sha256=sha2)
            sha3, count3, size3 = small_weights(path, missing=
                'decoder.prediction.dec_rnn.lstm.bias_hh_l0')
            with self.assertRaisesRegex(ValueError, 'native tensor'):
                inspect_weights(path, config(), expected_bytes=size3,
                                expected_tensors=count3, expected_sha256=sha3)

    def test_prepare_only_symlink_config_and_clean_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'converted'
            source.mkdir()
            cfg = config()
            (source / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
            (source / 'conversion.json').write_text(json.dumps({
                'stage': 'HDB2A_CONVERSION_ONLY',
                'model': 'qvac/TranscribePsy-250M-EU-sub',
                'source_nemo_sha256': 'a' * 64,
            }), encoding='utf-8')
            weights = root / 'qvac.safetensors'
            sha, count, size = small_weights(weights)
            out = root / 'official'
            manifest = prepare_official(weights, source, out, expected_bytes=size,
                                        expected_tensors=count, expected_sha256=sha)
            self.assertTrue((out / 'model.safetensors').is_symlink())
            self.assertEqual((out / 'model.safetensors').resolve(), weights.resolve())
            self.assertEqual((out / 'config.json').read_bytes(),
                             (source / 'config.json').read_bytes())
            self.assertEqual(manifest['stage'], 'HDB2A_OFFICIAL_SAFETENSORS')
            self.assertFalse(manifest['asr_qualified'])
            self.assertNotIn('model_dir', (out / 'conversion.json').read_text())
            self.assertNotIn(str(root), (out / 'conversion.json').read_text())
            with self.assertRaises(FileExistsError):
                prepare_official(weights, source, out, expected_bytes=size,
                                 expected_tensors=count, expected_sha256=sha)

    def test_runner_provenance_accepts_both_layouts(self):
        self.assertEqual(checkpoint_identity({
            'stage': 'HDB2A_CONVERSION_ONLY',
            'model': 'qvac/TranscribePsy-250M-EU-sub',
            'source_nemo_sha256': '1' * 64,
        }), ('converted-nemo', '1' * 64))
        self.assertEqual(checkpoint_identity({
            'stage': 'HDB2A_OFFICIAL_SAFETENSORS',
            'model': 'qvac/TranscribePsy-250M-EU-sub',
            'source_safetensors_sha256': QVAC_SAFETENSORS_SHA256,
        }), ('official-qvac-safetensors', QVAC_SAFETENSORS_SHA256))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            checkpoint_identity({
                'stage': 'HDB2A_OFFICIAL_SAFETENSORS',
                'model': 'qvac/TranscribePsy-250M-EU-sub',
                'source_safetensors_sha256': '0' * 64,
            })


if __name__ == '__main__':
    unittest.main()
