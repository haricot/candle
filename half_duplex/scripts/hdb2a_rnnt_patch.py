#!/usr/bin/env python3
"""Minimal, idempotent adaptation of the pinned RNNT loader for QVAC tensor names.

The official QVAC SafeTensors stores the predictor LSTM under
decoder.prediction.dec_rnn.lstm.{weight,bias}_{ih,hh}_lN.
Keep every previous layout/fallback for older converted checkpoints.
"""

MARKER = 'HDB2A_QVAC_NATIVE_LSTM'


def patch_rnnt(source: str) -> str:
    if MARKER in source:
        if source.count(MARKER) != 1:
            raise ValueError('Unexpected RNNT patch marker count')
        return source

    def exactly(old: str, new: str, n: int) -> None:
        nonlocal source
        found = source.count(old)
        if found != n:
            raise ValueError(f'Pin drift in RNNT source: expected {n} x {old!r}, found {found}')
        source = source.replace(old, new)

    exactly(
        '            vb_nemo: &VarBuilder,\n            base: &str,',
        '            vb_nemo: &VarBuilder,\n            vb_native: &VarBuilder,\n            base: &str,',
        2,
    )
    exactly(
        '                (vb_root, base),\n            ];',
        '                (vb_root, base),\n                (vb_native, layer_name.as_str()),\n            ];',
        2,
    )
    exactly(
        '        let vb_nemo = vb.pp("lstm").pp(format!("{layer_idx}"));\n',
        '        let vb_nemo = vb.pp("lstm").pp(format!("{layer_idx}"));\n'
        '        let vb_native = vb.pp("lstm"); // HDB2A_QVAC_NATIVE_LSTM\n',
        1,
    )
    exactly(
        '            &vb_nemo,\n',
        '            &vb_nemo,\n            &vb_native,\n',
        4,
    )
    # The historical loader allows absent biases for other RNNT layouts.
    # In QVAC's native layout, silent zero biases would corrupt transcription.
    exactly(
        '            Tensor::zeros((size,), vb_root.dtype(), vb_root.device())\n',
        '''            if vb_native
                .get((size, size / 4), &format!("weight_hh_l{layer_idx}"))
                .is_ok()
            {
                return Err(candle::Error::Msg(format!(
                    "missing native QVAC LSTM bias {base}_l{layer_idx}"
                )));
            }
            Tensor::zeros((size,), vb_root.dtype(), vb_root.device())
''',
        1,
    )
    return source
