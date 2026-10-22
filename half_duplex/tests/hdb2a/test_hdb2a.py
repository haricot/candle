import json
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_candle_runtime import modify_cargo,patch_main,PIN_CANDLE,PIN_PARAKEET
from hdb2a_conformer_patch import patch_conformer
from run_candle_hdb2a import wav_info
from convert_nemo_hdb2a import extract_model
from hdb2a_cargo_lock import libc_locked_version, MIN_LIBC

class AdapterTests(unittest.TestCase):
    def test_cargo_git_pinned(self):
        orig='''candle = { package = "candle-core", version = "0.10", default-features = false }\ncandle-nn = { version = "0.10", default-features = false }'''
        # 0.10 upstream actually does not use package= for candle-nn; change
        # must detect and replace the exact pinned text rather than guessing.
        src='''candle = { package = "candle-core", version = "0.10", default-features = false }\ncandle-nn = { version = "0.10", default-features = false }'''
        adapted=modify_cargo(src,'git')
        self.assertIn(PIN_CANDLE,adapted)
        self.assertIn('version = "0.11"',adapted)
        self.assertNotIn('version = "0.10"',adapted)

    def test_patches_local_loader(self):
        fragment='''use std::path::{Path, PathBuf};
struct Args {
    model_id: String,
}
fn main() -> Result<()> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(args.model_id.clone(), RepoType::Model));
    if args.debug { debug_log!(t0, "fetching config"); }
    let config_path = repo.get("config.json").context("missing config.json")?;
    if args.debug { debug_log!(t0, "config loaded from {}", config_path.display()); }
    let config: serde_json::Value = serde_json::from_reader(File::open(&config_path)?)?;
    let safetensors_files = match repo.get("model.safetensors.index.json") {
        Ok(_) => hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        Err(_) => vec![repo.get("model.safetensors")?],
    };
    if args.debug {
        let total_bytes: u64 = 0;
    }
}
'''
        adapted=patch_main(fragment)
        self.assertIn('args.model_dir.as_ref()',adapted)
        self.assertIn('HDB2A_CANDLE_LOCAL_MODEL',adapted)
        self.assertEqual(adapted.count('let config: serde_json::Value'),1)

    def test_converts_native_nemo_minimal(self):
        import torch
        import yaml
        from safetensors import safe_open
        cfg={
            'target':'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel',
            'model_defaults':{'tdt_durations':[0,1,2,4]},
            'preprocessor':{'sample_rate':16000,'features':128,'window':'hann','n_fft':512,
                            'window_size':0.025,'window_stride':0.01,'normalize':'per_feature','dither':0.0},
            'encoder':{'n_layers':1,'d_model':16,'n_heads':4,'feat_in':128,
                       'subsampling_factor':8,'subsampling':'dw_striding',
                       'subsampling_conv_channels':8,'self_attention_model':'rel_pos',
                       'ff_expansion_factor':4,'conv_kernel_size':9,'pos_emb_max_len':5000},
            'decoder':{'vocab_size':2,'blank_as_pad':True,'prednet':{'pred_hidden':8,'pred_rnn_layers':1}},
            'joint':{'vocabulary':['▁oui','▁non'], 'num_classes':2,'num_extra_outputs':4,
                     'jointnet':{'encoder_hidden':16,'pred_hidden':8,'joint_hidden':8,'activation':'relu'}}}
        state={'encoder.pre_encode.conv.0.weight':torch.zeros((8,1,3,3)),
               'encoder.layers.0.norm_feed_forward1.weight':torch.ones((16,)),
               'decoder.prediction.embed.weight':torch.ones((3,8)),
               'joint.joint_net.2.weight':torch.zeros((7,8)),
               'decoder.prediction.dec_rnn.lstm.weight_ih_l0':torch.zeros((32,8)),
               'decoder.prediction.dec_rnn.lstm.weight_hh_l0':torch.zeros((32,8)),
               'decoder.prediction.dec_rnn.lstm.bias_ih_l0':torch.zeros((32,)),
               'decoder.prediction.dec_rnn.lstm.bias_hh_l0':torch.zeros((32,)),
               'encoder.layers.0.conv.batch_norm.num_batches_tracked':torch.tensor(0)}
        with tempfile.TemporaryDirectory() as d:
            tar_path=Path(d)/'checkpoint.nemo'
            with tarfile.open(tar_path,'w') as tar:
                for name,data in [('model_config.yaml',yaml.safe_dump(cfg,sort_keys=False).encode()),
                                  ('model_weights.ckpt',self.to_bytes(state))]:
                    meta=tarfile.TarInfo('./'+name);meta.size=len(data)
                    tar.addfile(meta,io.BytesIO(data))
            out=Path(d)/'converted'
            extract_model(tar_path,out)
            with safe_open(str(out/'model.safetensors'),framework='pt',device='cpu') as st:
                self.assertEqual(len(st.keys()),12)
            loaded=json.loads((out/'config.json').read_text())
            self.assertEqual(loaded['decoding']['durations'],[0,1,2,4])
            self.assertEqual(loaded['joint']['vocabulary'],['▁oui','▁non'])
            self.assertFalse(json.loads((out/'conversion.json').read_text())['asr_qualified'])

    def test_conformer_patch_and_idempotence(self):
        # The exact anchors from the pinned implementation are required.
        fragment = """pub struct ConformerArgs {
    pub conv_kernel_size: usize,
}
fn default_true() -> bool {
    true
}
#[derive(Debug, Clone)]
struct Convolution {
    batch_norm: BatchNorm,
    padding: usize,
    causal_padding: usize,
}
impl Convolution {
    fn load(args: &ConformerArgs, vb: VarBuilder) -> Result<Self> {
        let padding = (args.conv_kernel_size - 1) / 2;
        let batch_norm = candle_nn::batch_norm(args.d_model, 1e-5, vb.pp("batch_norm"))?;
        Ok(Self {
            batch_norm,
            pointwise_conv2,
            padding,
            causal_padding,
        })
    }
    fn forward(&self, mut x: Tensor) -> Result<Tensor> {
        if let Some(cache) = cache {
        } else if self.padding > 0 {
            x = x.pad_with_zeros(D::Minus2, self.padding, self.padding)?;
        }
        x = self.batch_norm.forward_t(&x, false)?;
        Ok(x)
    }
}
"""
        out = patch_conformer(fragment)
        self.assertEqual(patch_conformer(out), out)
        self.assertIn('Hdb2aConvNorm::Layer', out)
        self.assertIn('ln.forward(&x.transpose(1, 2)?.contiguous()?)?', out)
        self.assertIn('conv_context_size', out)
        self.assertIn('batch_norm(args.d_model', out)
        self.assertNotIn('self.batch_norm.forward_t', out)
        with self.assertRaisesRegex(ValueError, 'Pin drift'):
            patch_conformer('not the pinned conformer')
        # Regression: real pinned Conformer also contains a *second* padding
        # field, owned by DwStridingSubsampling. It must remain untouched.
        original = fragment + """
#[derive(Debug, Clone)]
struct DwStridingSubsampling {
    padding: usize,
}
impl DwStridingSubsampling {
    fn new(padding: usize) -> Self {
        Self { padding, }
    }
}
"""
        patched = patch_conformer(original)
        self.assertIn('struct DwStridingSubsampling {\n    padding: usize,',patched)
        self.assertEqual(patch_conformer(patched),patched)

    def test_cargo_lock_compatibility_detection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'Cargo.lock'
            self.assertIsNone(libc_locked_version(path))
            path.write_text('''
[[package]]
name = "libc"
version = "0.2.184"
'''
)
            self.assertEqual(libc_locked_version(path), '0.2.184')
            self.assertLess((0,2,184),MIN_LIBC)


    def test_layer_norm_nemo_conversion(self):
        import torch
        import yaml
        cfg = {
            'target': 'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel',
            'model_defaults': {'tdt_durations': [0, 1, 2, 4]},
            'preprocessor': {'sample_rate': 16000, 'features': 128, 'window': 'hann',
                             'n_fft': 512, 'window_size': 0.025, 'window_stride': 0.01,
                             'normalize': 'per_feature', 'dither': 0.0},
            'encoder': {'n_layers': 1, 'd_model': 16, 'n_heads': 4, 'feat_in': 128,
                        'conv_kernel_size': 9, 'conv_norm_type': 'layer_norm',
                        'conv_context_size': 'causal',
                        'subsampling_factor': 8, 'subsampling': 'dw_striding',
                        'subsampling_conv_channels': 8, 'self_attention_model': 'rel_pos',
                        'ff_expansion_factor':4,'pos_emb_max_len':5000},
            'decoder': {'vocab_size': 2, 'blank_as_pad': True,
                        'prednet': {'pred_hidden': 8, 'pred_rnn_layers': 1}},
            'joint': {'vocabulary': ['▁oui', '▁non'], 'num_classes': 2,
                      'num_extra_outputs': 4,
                      'jointnet': {'encoder_hidden': 16, 'pred_hidden': 8,
                                   'joint_hidden': 8, 'activation': 'relu'}}
        }
        w = {'encoder.pre_encode.conv.0.weight': torch.zeros((8, 1, 3, 3)),
             'encoder.layers.0.norm_feed_forward1.weight': torch.ones(16),
             'encoder.layers.0.conv.batch_norm.weight': torch.ones(16),
             'encoder.layers.0.conv.batch_norm.bias': torch.zeros(16),
             'decoder.prediction.embed.weight': torch.ones((3, 8)),
             'joint.joint_net.2.weight': torch.zeros((7, 8))}
        for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            shape = (32, 8) if name == 'weight_ih' else (32, 8) if name == 'weight_hh' else (32,)
            w[f'decoder.prediction.dec_rnn.lstm.{name}_l0'] = torch.zeros(shape)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            cp = base/'small.nemo'
            def write_checkpoint():
                with tarfile.open(cp, 'w') as tar:
                    for fname, data in [('model_config.yaml', yaml.safe_dump(cfg).encode()),
                                        ('model_weights.ckpt', self.to_bytes(w))]:
                        info = tarfile.TarInfo('./'+fname)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
            write_checkpoint()
            extract_model(cp, base/'converted')
            output = json.loads((base/'converted/config.json').read_text())
            self.assertEqual(output['encoder']['conv_norm_type'], 'layer_norm')
            self.assertEqual(output['encoder']['conv_context_size'], 'causal')
            meta = json.loads((base/'converted/conversion.json').read_text())
            self.assertEqual(meta['conv_norm_type'], 'layer_norm')
            self.assertEqual(meta['effective_window'], 'hann')
            self.assertFalse(meta['asr_qualified'])
            cfg['preprocessor']['window'] = 'hamming'
            write_checkpoint()
            extract_model(cp, base/'hamming')
            self.assertEqual(json.loads((base/'hamming/config.json').read_text())['preprocessor']['window'], 'hamming')
            self.assertEqual(json.loads((base/'hamming/conversion.json').read_text())['original_window'], 'hamming')
            # Real checkpoint omitted dither entirely: Candle needs explicit 0.0.
            cfg['preprocessor'].pop('dither')
            write_checkpoint()
            extract_model(cp, base/'without-dither')
            exported=json.loads((base/'without-dither/config.json').read_text())
            self.assertEqual(exported['preprocessor']['dither'], 0.0)
            self.assertIsNone(json.loads((base/'without-dither/conversion.json').read_text())['original_dither'])
            cfg['encoder']['conv_norm_type'] = 'groupnorm8'
            write_checkpoint()
            with self.assertRaisesRegex(ValueError, 'groupnorm8'):
                extract_model(cp, base/'rejected')

    def test_missing_dither_export_and_local_repair(self):
        # Reproduce the reported runtime failure with a real-looking config.
        from hdb2a_model_config import normalize_dither, validate_config, repair_local_config
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            cfg = {
                'target':'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel',
                'preprocessor':{'sample_rate':16000,'normalize':'per_feature',
                                'window_size':0.025,'window_stride':0.01,'window':'hamming',
                                'features':128,'n_fft':512},
                'encoder':{'feat_in':128,'n_layers':24,'d_model':640,'n_heads':8,
                           'ff_expansion_factor':4,'subsampling_factor':8,
                           'self_attention_model':'rel_pos','subsampling':'dw_striding',
                           'conv_kernel_size':9,'subsampling_conv_channels':256,
                           'pos_emb_max_len':5000,'conv_norm_type':'layer_norm'},
                'decoder':{'blank_as_pad':True,'vocab_size':2,
                           'prednet':{'pred_hidden':640,'pred_rnn_layers':2}},
                'joint':{'num_classes':2,'vocabulary':['▁oui','▁non'],
                         'jointnet':{'joint_hidden':640,'activation':'relu',
                                     'encoder_hidden':640,'pred_hidden':640}},
                'decoding':{'model_type':'tdt','durations':[0,1,2,4]}}
            with self.assertRaisesRegex(ValueError, 'preprocessor.dither'):
                validate_config(cfg)
            (folder/'config.json').write_text(json.dumps(cfg),encoding='utf-8')
            (folder/'conversion.json').write_text(json.dumps({'model':'qvac/TranscribePsy-250M-EU-sub',
                                                               'stage':'HDB2A_CONVERSION_ONLY'}), encoding='utf-8')
            fixed=repair_local_config(folder)
            self.assertEqual(fixed['preprocessor']['dither'],0.0)
            self.assertTrue(validate_config(fixed))
            first=(folder/'config.json').read_bytes()
            repair_local_config(folder)
            self.assertEqual(first,(folder/'config.json').read_bytes())
            self.assertEqual(normalize_dither(fixed),0.0)

    def test_schema_refuses_other_missing_fields_and_bad_dither(self):
        from hdb2a_model_config import normalize_dither, validate_config
        with self.assertRaisesRegex(ValueError, 'Invalid NeMo preprocessor.dither'):
            normalize_dither({'preprocessor':{'dither':'not-a-number'}})
        with self.assertRaisesRegex(ValueError, 'Missing preprocessor'):
            normalize_dither({})
        cfg = {'preprocessor':{'dither':0.0}}
        with self.assertRaisesRegex(ValueError, 'encoder.n_layers'):
            validate_config(cfg)

    def to_bytes(self,weight):
        import torch
        buf=io.BytesIO();torch.save(weight,buf);return buf.getvalue()

if __name__=='__main__':unittest.main()
