#!/usr/bin/env python3
"""Export QVAC TranscribePsy EU-sub NeMo weights to native Candle-compatible
SafeTensors + config.json for the pinned parakeet-candle FastConformer/TDT runtime.

Never invokes NeMo inference or substitutes a different ASR model.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha256_file(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def normalize_nemo_padding(pre):
    """Materialize AudioToMelSpectrogramPreprocessor's NeMo 2.7.3 default.

    The original YAML may omit pad_to. NeMo defaults to 16, whereas the
    Candle usize serde default is zero. Explicit checkpoint values prevail.
    """
    original = pre.get('pad_to')
    if 'pad_to' not in pre:
        pre['pad_to'] = 16
    value = pre['pad_to']
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Unsupported NeMo preprocessor.pad_to; expected a nonnegative integer')
    return original


def extract_model(nemo_path: Path, dest: Path):
    import torch
    import yaml
    from safetensors.torch import save_file
    import sentencepiece as spm

    def member(t, filename):
        xs = [m for m in t.getmembers() if m.isfile() and m.name.removeprefix('./') == filename]
        if len(xs) != 1:
            raise ValueError(f'Expected exactly one {filename} in NeMo archive, found {len(xs)}')
        return xs[0]

    with tarfile.open(nemo_path, 'r:*') as tar:
        cfg_file = tar.extractfile(member(tar, 'model_config.yaml'))
        assert cfg_file is not None
        cfg = yaml.safe_load(cfg_file.read())
        ck_file = tar.extractfile(member(tar, 'model_weights.ckpt'))
        assert ck_file is not None
        # weights_only=True is intentional; no pickle execution.
        state = torch.load(io.BytesIO(ck_file.read()), map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError('model_weights.ckpt is not a non-empty state_dict')

        if isinstance(cfg, dict) and 'model' in cfg and 'encoder' not in cfg:
            cfg = cfg['model']
        if not isinstance(cfg, dict):
            raise ValueError('NeMo model_config.yaml must be a mapping')
        if not {'preprocessor','encoder','decoder','joint'} <= cfg.keys():
            raise ValueError('NeMo config lacks required FastConformer/RNNT sections')

        # Limit interpolation resolution to simple ${section.key} references.
        # Missing or unsupported expressions fail instead of falling back to guesses.
        def resolve(obj):
            if isinstance(obj, dict): return {k: resolve(v) for k,v in obj.items()}
            if isinstance(obj, list): return [resolve(x) for x in obj]
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                key = obj[2:-1]
                if not key.replace('_','').replace('.','').isalnum():
                    raise ValueError(f'Unsafe/unresolved interpolation: {obj}')
                if key.startswith('model.'):
                    key = key[6:]
                value = cfg
                for part in key.split('.'):
                    value = value[part]
                if value == obj:
                    raise ValueError(f'Self-referential interpolation: {obj}')
                return resolve(value)
            return obj
        cfg = resolve(cfg)
        original_pad_to = normalize_nemo_padding(cfg['preprocessor'])
        target = cfg.get('target', cfg.get('_target_', ''))
        if target not in [
            'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel',
            'nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel',
        ]:
            raise ValueError(f'Unsupported NeMo model target: {target}')
        durations = (cfg.get('model_defaults') or {}).get('tdt_durations')
        if durations is None:
            durations = (cfg.get('loss') or {}).get('tdt_kwargs',{}).get('durations')
        if durations is None:
            durations = (cfg.get('decoding') or {}).get('durations')
        if not isinstance(durations,list) or not durations or any(not isinstance(v,int) or v<0 for v in durations):
            raise ValueError('TDT duration list missing or invalid in original checkpoint')
        cfg['model_defaults'] = dict(cfg.get('model_defaults') or {})
        cfg['model_defaults']['tdt_durations'] = durations
        cfg['target'] = target
        cfg['decoding'] = dict(cfg.get('decoding') or {})
        cfg['decoding']['model_type'] = 'tdt'
        cfg['decoding']['durations'] = durations
        # The model may omit the list from the YAML, but it can be reconstructed
        # from the original NeMo SentencePiece tokenizer, never guessed.
        vocab = cfg['joint'].get('vocabulary')
        if not vocab:
            tok_cfg = cfg.get('tokenizer') or {}
            model_name = (tok_cfg.get('model_path') or '').split('/')[-1]
            found = [m for m in tar.getmembers() if m.isfile() and model_name and m.name.endswith('/'+model_name)]
            if len(found) != 1:
                raise ValueError(f'Missing joint vocabulary and tokenizer file {model_name!r}')
            bytes_ = tar.extractfile(found[0]).read()
            sp = spm.SentencePieceProcessor(model_proto=bytes_)
            vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        if not isinstance(vocab,list) or not all(isinstance(x,str) for x in vocab) or not vocab:
            raise ValueError('Invalid SentencePiece vocabulary')
        cfg['joint']['vocabulary'] = vocab
        joint_size = int(cfg['joint']['num_classes'])
        if len(vocab) != joint_size:
            raise ValueError(f'Vocabulary count {len(vocab)} differs from joint.num_classes {joint_size}')
        cfg['decoder']['vocab_size'] = int(cfg['decoder']['vocab_size'])
        if cfg['decoder']['vocab_size'] != len(vocab):
            raise ValueError(f'Decoder vocab_size {cfg["decoder"]["vocab_size"]} != {len(vocab)}')
        if not cfg['decoder'].get('blank_as_pad'):
            raise ValueError('NeMo model blank_as_pad=false is unsupported by HDB2-A')
        if int(cfg['encoder']['subsampling_factor']) != 8:
            raise ValueError('HDB2-A requires tested 8x dw_striding encoder topology')
        if cfg['encoder']['subsampling'] != 'dw_striding':
            raise ValueError('Only dw_striding is compatible with current Candle loader')
        conv_norm = cfg['encoder'].get('conv_norm_type', 'batch_norm')
        if conv_norm not in ('batch_norm', 'layer_norm'):
            raise ValueError(f'Unsupported encoder.conv_norm_type={conv_norm!r}; supported: batch_norm, layer_norm')
        cfg['encoder']['conv_norm_type'] = conv_norm
        print(f'HDB2A_CHECKPOINT_CONV_NORM={conv_norm}', flush=True)
        conv_ctx = cfg['encoder'].get('conv_context_size')
        kernel = int(cfg['encoder'].get('conv_kernel_size', 0))
        if conv_ctx is None or conv_ctx == 'default':
            pass  # NeMo centered convolution
        elif conv_ctx == 'causal':
            pass  # NeMo left-only convolution, handled by patched Candle runtime
        elif (isinstance(conv_ctx, list) and len(conv_ctx) == 2 and
              all(isinstance(v, int) and v >= 0 for v in conv_ctx) and
              sum(conv_ctx) + 1 == kernel):
            pass
        else:
            raise ValueError(f'Unsupported encoder.conv_context_size={conv_ctx!r} for kernel={kernel}')
        if cfg['encoder']['d_model'] % cfg['encoder']['n_heads'] != 0:
            raise ValueError('d_model not divisible by n_heads')
        if cfg['encoder'].get('self_attention_model') not in ('rel_pos','rel_pos_local_attn'):
            raise ValueError('Unverified attention type')
        if cfg['encoder'].get('att_context_size') and isinstance(cfg['encoder']['att_context_size'][0], list):
            raise ValueError('Nested context sizes unsupported')
        if cfg['preprocessor'].get('sample_rate') != 16000:
            raise ValueError('Expected 16kHz audio')
        # Match NeMo's supported STFT window using the upstream Candle runtime's
        # `window_values()` implementation. `hanning` aliases `hann` in that
        # runtime; `ones`/null mean a rectangular window (not Hann).
        original_window = cfg['preprocessor'].get('window', 'hann')
        effective_window = ('ones' if original_window is None else original_window)
        if effective_window == 'hanning':
            effective_window = 'hann'
        if effective_window == 'none':
            effective_window = 'ones'
        supported_windows = ('hann', 'hamming', 'blackman', 'bartlett', 'ones')
        if effective_window not in supported_windows:
            raise ValueError(f'Unsupported NeMo preprocessor.window={original_window!r}; '
                             f'supported: {supported_windows}')
        cfg['preprocessor']['window'] = effective_window
        print(f'HDB2A_CHECKPOINT_WINDOW={original_window!r} EFFECTIVE_WINDOW={effective_window}', flush=True)
        # NeMo feature dither is training-time noise; the pinned Candle
        # get_logmel() routine is deterministic and never applies dither.
        # Store explicit inference value 0.0 so its strict Serde reader
        # does not fail with `missing field dither` on QVAC checkpoints.
        from hdb2a_model_config import normalize_dither, validate_config
        original_dither = cfg['preprocessor'].get('dither')
        normalize_dither(cfg)
        print(f'HDB2A_DITHER_CHECKPOINT={original_dither!r} INFERENCE=0.0', flush=True)
        # Report *all* missing mandatory runtime fields before exporting
        # another GB-sized safetensors file; other fields are never guessed.
        validate_config(cfg)
        # Loaded checkpoints contain auxiliary preprocessor buffers; preserve
        # original inference weights and names, dropping non-FP parameters only.
        weights = {}
        for name, tensor in state.items():
            if not isinstance(name,str) or not isinstance(tensor,torch.Tensor):
                raise ValueError(f'Unexpected non-tensor in state_dict: {name}')
            if tensor.is_floating_point():
                weights[name] = tensor.detach().cpu().contiguous().float()
            elif name.endswith('num_batches_tracked'):
                continue
            else:
                raise ValueError(f'Unexpected integer state tensor {name}')
        # Compatibility aliases for the upstream Candle loader: NeMo nests
        # LSTM matrices under dec_rnn.lstm whereas that loader checks the
        # flattened dec_rnn.weight_ih_lN / weight_hh_lN / biases first.
        for i in range(int(cfg['decoder']['prednet']['pred_rnn_layers'])):
            for kind in ('weight_ih','weight_hh','bias_ih','bias_hh'):
                src = f'decoder.prediction.dec_rnn.lstm.{kind}_l{i}'
                dst = f'decoder.prediction.dec_rnn.{kind}_l{i}'
                if src not in weights:
                    raise ValueError(f'Missing mandatory LSTM tensor {src}')
                if dst in weights and not torch.equal(weights[dst],weights[src]):
                    raise ValueError(f'Existing Candle LSTM alias differs from NeMo: {dst}')
                weights[dst] = weights[src].clone()
        if conv_norm == 'layer_norm':
            # NeMo stores LayerNorm affine parameters under the historical
            # conv.batch_norm prefix; it has NO running statistics.
            channels = int(cfg['encoder']['d_model'])
            for layer in range(int(cfg['encoder']['n_layers'])):
                prefix = f'encoder.layers.{layer}.conv.batch_norm'
                for suffix in ('weight', 'bias'):
                    key = f'{prefix}.{suffix}'
                    if key not in weights or tuple(weights[key].shape) != (channels,):
                        raise ValueError(f'LayerNorm requires {key} of shape [{channels}]')
                if any(k in weights for k in (f'{prefix}.running_mean', f'{prefix}.running_var')):
                    raise ValueError(f'Layer {layer}: layer_norm must not contain BatchNorm running statistics')
        required = ['encoder.pre_encode.conv.0.weight',
                    'encoder.layers.0.norm_feed_forward1.weight',
                    'decoder.prediction.embed.weight',
                    'joint.joint_net.2.weight']
        absent = [k for k in required if k not in weights]
        if absent:
            raise ValueError(f'NeMo weight inventory missing essential tensors: {absent}')
        if int(weights['encoder.pre_encode.conv.0.weight'].shape[0]) != int(cfg['encoder']['subsampling_conv_channels']):
            raise ValueError('Subsampling weight/config channel mismatch')
        out_proj = weights['joint.joint_net.2.weight']
        expected_out = len(vocab)+1+int(cfg['joint'].get('num_extra_outputs',0))
        if int(out_proj.shape[0]) != expected_out:
            raise ValueError(f'Joint output dimensions mismatch: {out_proj.shape[0]} != {expected_out}')

        dest.mkdir(parents=True, exist_ok=True)
        save_file(weights, str(dest / 'model.safetensors'))
        (dest / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        meta = {
            'stage': 'HDB2A_CONVERSION_ONLY',
            'model': 'qvac/TranscribePsy-250M-EU-sub',
            'architecture': 'fastconformer-tdt',
            'source_nemo_sha256': sha256_file(nemo_path),
            'weight_tensors': len(weights), 'vocab_size': len(vocab),
            'n_layers': cfg['encoder']['n_layers'],
            'conv_norm_type': conv_norm, 'conv_context_size': conv_ctx,
            'original_window': original_window, 'effective_window': effective_window,
            'original_dither': original_dither, 'inference_dither': 0.0,
            'original_pad_to': original_pad_to, 'effective_pad_to': cfg['preprocessor']['pad_to'],
            'd_model': cfg['encoder']['d_model'],
            'asr_qualified': False,
        }
        (dest / 'conversion.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
        print(f'HDB2A_CONVERSION=PASS_SHAPES_ONLY tensors={len(weights)} encoder_dim={cfg["encoder"]["d_model"]} layers={cfg["encoder"]["n_layers"]} conv_norm={conv_norm}')
        print(f'HDB2A_MODEL_DIR={dest.resolve()}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nemo', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a=p.parse_args()
    if not a.nemo.is_file():
        p.error(f'NeMo checkpoint missing: {a.nemo}')
    extract_model(a.nemo,a.output_dir)

if __name__=='__main__': main()
