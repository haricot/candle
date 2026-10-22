"""Source-level *risk inventory*, not a substitute for numerical tensor evidence."""
import json
from pathlib import Path

CHECKS=[
 ('window','NeMo torch spectral windows may be symmetric (periodic=False); Candle uses periodic=True for named windows.'),
 ('log_zero_guard_value','Candle currently uses ln(mel + 1e-5) regardless of the original log-zero guard.'),
 ('mel_norm','Candle triangular Slaney filterbank implementation must be compared with NeMo/librosa.'),
 ('pad_to','NeMo pads time features to a multiple; Candle logmel path may not.'),
 ('exact_pad','STFT padding choice may alter frame alignment.'),
 ('preemph','Preemphasis must be identical, including first-sample behavior.'),
 ('normalize','Per-feature mean/std behavior and valid frame counts must agree.'),
 ('n_fft','FFT binning and normalization must agree.'),
 ('mag_power','Magnitude exponent must agree.'),
]

def audit(model_dir):
    cfg=json.loads((Path(model_dir)/'config.json').read_text())
    pre=cfg.get('preprocessor') or {}
    return {'stage':'HDB2C_SOURCE_LEVEL_PREPROCESSOR_RISKS','status':'UNVERIFIED_SOURCE_DIFFERENCES',
            'model_dir':str(Path(model_dir)),'checks':[{'field':field,
                'checkpoint_value':pre.get(field),'risk_to_measure':explanation} for field,explanation in CHECKS],
            'note':'Do not infer actual tensor errors without the real numeric probes.'}
