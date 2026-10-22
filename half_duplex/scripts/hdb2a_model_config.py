#!/usr/bin/env python3
"""Validate/repair the strictly required Candle ASR config, without weights.

Only one inferred/defaulted field is allowed: preprocessor.dither=0.0.
The pinned upstream get_logmel() does not apply dither for inference.
All other unknown or missing model fields remain explicit hard failures.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

MODEL_ID = 'qvac/TranscribePsy-250M-EU-sub'
REQUIRED = (
    'target',
    'preprocessor.sample_rate', 'preprocessor.normalize',
    'preprocessor.window_size', 'preprocessor.window_stride',
    'preprocessor.window', 'preprocessor.features',
    'preprocessor.n_fft', 'preprocessor.dither',
    'encoder.feat_in', 'encoder.n_layers', 'encoder.d_model',
    'encoder.n_heads', 'encoder.ff_expansion_factor',
    'encoder.subsampling_factor', 'encoder.self_attention_model',
    'encoder.subsampling', 'encoder.conv_kernel_size',
    'encoder.subsampling_conv_channels', 'encoder.pos_emb_max_len',
    'decoder.blank_as_pad', 'decoder.vocab_size',
    'decoder.prednet.pred_hidden', 'decoder.prednet.pred_rnn_layers',
    'joint.num_classes', 'joint.vocabulary',
    'joint.jointnet.joint_hidden', 'joint.jointnet.activation',
    'joint.jointnet.encoder_hidden', 'joint.jointnet.pred_hidden',
    'decoding.model_type', 'decoding.durations',
)


def lookup(mapping, path):
    value = mapping
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def normalize_dither(cfg):
    """Record deterministic inference semantics, rejecting invalid originals."""
    pre = cfg.get('preprocessor')
    if not isinstance(pre, dict):
        raise ValueError('Missing preprocessor mapping')
    original = pre.get('dither')
    if original is not None:
        if isinstance(original, bool) or not isinstance(original, (int, float)) \
           or not math.isfinite(original) or original < 0:
            raise ValueError(f'Invalid NeMo preprocessor.dither: {original!r}')
    pre['dither'] = 0.0
    return original


def validate_config(cfg):
    absent = [p for p in REQUIRED if not lookup(cfg, p)[0] or lookup(cfg,p)[1] is None]
    if absent:
        raise ValueError('CANDLE_CONFIG_MISSING_FIELDS=' + ','.join(absent))
    pre = cfg['preprocessor']
    dither = pre['dither']
    if isinstance(dither, bool) or not isinstance(dither,(int,float)) or not math.isfinite(dither) or dither != 0.0:
        raise ValueError(f'CANDLE_CONFIG_DITHER_NOT_ZERO={dither!r}')
    if cfg['decoding']['model_type'] != 'tdt':
        raise ValueError('HDB2A requires a TDT checkpoint')
    if cfg['encoder']['conv_norm_type'] not in ('layer_norm', 'batch_norm'):
        raise ValueError('HDB2A unsupported conv_norm_type')
    return True


def repair_local_config(model_dir: Path):
    cfg_path = model_dir / 'config.json'
    provenance_path = model_dir / 'conversion.json'
    if not cfg_path.is_file() or not provenance_path.is_file():
        raise ValueError(f'Missing HDB2A config or conversion.json: {model_dir}')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance.get('model') != MODEL_ID or provenance.get('stage') != 'HDB2A_CONVERSION_ONLY':
        raise ValueError('Not an HDB2A QVAC EU-sub converted model; repair refused')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    previous = cfg.get('preprocessor',{}).get('dither')
    normalize_dither(cfg)
    validate_config(cfg)
    target = json.dumps(cfg, indent=2, ensure_ascii=False) + '\n'
    if cfg_path.read_text(encoding='utf-8') != target:
        # Atomic replace, preserving original on any validation failure.
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=model_dir, prefix='.config-hdb2a-',suffix='.tmp',delete=False) as f:
            temp = Path(f.name)
            f.write(target)
        try:
            os.replace(temp,cfg_path)
        finally:
            temp.unlink(missing_ok=True)
        print(f'HDB2A_CONFIG_REPAIR=UPDATED original_dither={previous!r} inference_dither=0.0')
    else:
        print('HDB2A_CONFIG_REPAIR=ALREADY_VALID')
    return cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation', choices=['check','repair'])
    p.add_argument('--model-dir',type=Path,required=True)
    a=p.parse_args()
    if a.operation == 'repair':
        repair_local_config(a.model_dir)
    else:
        path=a.model_dir/'config.json'
        if not path.is_file():
            p.error(f'No local model config: {path}')
        validate_config(json.loads(path.read_text(encoding='utf-8')))
        print('HDB2A_CONFIG_SCHEMA=PASS_REQUIRED_FIELDS_NOT_NUMERIC_PARITY')

if __name__=='__main__':main()
