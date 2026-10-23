#!/usr/bin/env python3
"""Use the unchanged, official QVAC SafeTensors with HDB2-A.

Only a small locally generated config/manifest and a symlink are created;
no 1 GB weight conversion/copy and no published Haricot checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

from hdb2a_model_config import MODEL_ID, validate_config

OFFICIAL_REVISION = 'ccf70de9ff9848f5cd5975adda1370ce4addf6f1'
OFFICIAL_SHA256 = '481fec004f2074909a5d253cb674fe676a49ce856bca2ce0156136ea920c3168'
OFFICIAL_BYTES = 1015921896
OFFICIAL_TENSORS = 913


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_weights(path: Path, cfg: dict, *, expected_bytes=OFFICIAL_BYTES,
                    expected_tensors=OFFICIAL_TENSORS,
                    expected_sha256=OFFICIAL_SHA256) -> dict:
    """Bounded SafeTensors-header check plus whole-file digest; no tensor loading."""
    file_size = path.stat().st_size
    if file_size != expected_bytes:
        raise ValueError(f'Wrong official SafeTensors size: {file_size} != {expected_bytes}')
    with path.open('rb') as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError('Truncated SafeTensors header length')
        header_length = int.from_bytes(raw_length, 'little')
        if not (2 <= header_length <= 16 * 1024 * 1024) or header_length + 8 > file_size:
            raise ValueError('Unsafe or truncated SafeTensors header')
        header = json.loads(stream.read(header_length))
    if not isinstance(header, dict):
        raise ValueError('SafeTensors header is not a mapping')
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    if len(tensors) != expected_tensors:
        raise ValueError(f'Unexpected QVAC tensor count: {len(tensors)} != {expected_tensors}')
    data_bytes = file_size - 8 - header_length
    for name, entry in tensors.items():
        if not isinstance(entry, dict) or entry.get('dtype') != 'F32':
            raise ValueError(f'Unexpected QVAC tensor dtype: {name}')
        bounds = entry.get('data_offsets')
        if not (isinstance(bounds, list) and len(bounds) == 2
                and all(type(n) is int for n in bounds)
                and 0 <= bounds[0] <= bounds[1] <= data_bytes):
            raise ValueError(f'Invalid SafeTensors data offsets: {name}')
    hidden = int(cfg['decoder']['prednet'].get('rnn_hidden_size')
                 or cfg['decoder']['prednet']['pred_hidden'])
    input_size = int(cfg['decoder']['prednet']['pred_hidden'])
    layers = int(cfg['decoder']['prednet']['pred_rnn_layers'])
    for layer in range(layers):
        for kind, shape in (
            ('weight_ih', [4 * hidden, input_size if layer == 0 else hidden]),
            ('weight_hh', [4 * hidden, hidden]),
            ('bias_ih', [4 * hidden]),
            ('bias_hh', [4 * hidden]),
        ):
            name = f'decoder.prediction.dec_rnn.lstm.{kind}_l{layer}'
            if name not in tensors or tensors[name].get('shape') != shape:
                raise ValueError(f'Missing/mismatched QVAC native tensor: {name} expected={shape}')
    for name in ('encoder.pre_encode.conv.0.weight',
                 'encoder.layers.0.norm_feed_forward1.weight',
                 'decoder.prediction.embed.weight',
                 'joint.joint_net.2.weight'):
        if name not in tensors:
            raise ValueError(f'Missing essential QVAC tensor: {name}')
    observed_sha256 = digest(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f'Wrong official SafeTensors SHA256: {observed_sha256}')
    return {'tensor_count': len(tensors), 'sha256': observed_sha256}


def prepare_official(weights: Path, converted_dir: Path, output_dir: Path,
                     *, expected_bytes=OFFICIAL_BYTES, expected_tensors=OFFICIAL_TENSORS,
                     expected_sha256=OFFICIAL_SHA256) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'Refusing to overwrite HDB2-A model directory: {output_dir}')
    if not weights.is_file():
        raise FileNotFoundError(f'Official model.safetensors not found: {weights}')
    config_path = converted_dir / 'config.json'
    provenance_path = converted_dir / 'conversion.json'
    if not config_path.is_file() or not provenance_path.is_file():
        raise ValueError('Expected validated HDB2-A converted config.json and conversion.json')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance.get('model') != MODEL_ID or provenance.get('stage') != 'HDB2A_CONVERSION_ONLY':
        raise ValueError('Config source is not a recognized QVAC HDB2-A conversion')
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes)
    validate_config(cfg)
    details = inspect_weights(weights, cfg, expected_bytes=expected_bytes,
                              expected_tensors=expected_tensors, expected_sha256=expected_sha256)
    manifest = {
        'stage': 'HDB2A_OFFICIAL_SAFETENSORS',
        'model': MODEL_ID,
        'source_revision': OFFICIAL_REVISION,
        'source_safetensors_sha256': details['sha256'],
        'weight_tensors': details['tensor_count'],
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'asr_qualified': False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    # Resolve once, so the model works when invoked from a different cwd.
    (output_dir / 'model.safetensors').symlink_to(weights.resolve())
    (output_dir / 'config.json').write_bytes(config_bytes)
    (output_dir / 'conversion.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('HDB2A_QVAC_OFFICIAL=PASS_INVENTORY_AND_SHA256_NOT_ASR_PARITY')
    print(f'HDB2A_QVAC_TENSORS={details["tensor_count"]}')
    print(f'HDB2A_MODEL_DIR={output_dir.resolve()}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True,
                        help='HF Hub official qvac model.safetensors, already downloaded')
    parser.add_argument('--converted-model-dir', type=Path, required=True,
                        help='Existing HDB2-A conversion, used ONLY for verified config.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare_official(args.weights, args.converted_model_dir, args.output_dir)


if __name__ == '__main__':
    main()
