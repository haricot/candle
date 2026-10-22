"""Actual Smart Turn v3.2 ONNX inference after VAD candidate boundary."""
from pathlib import Path


class SmartTurnOnnx:
    def __init__(self, path: Path):
        import onnxruntime as ort
        from .pipeline import hash_file
        self.model_sha256 = hash_file(path)
        from transformers import WhisperFeatureExtractor
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        ins = self.session.get_inputs()
        if len(ins) != 1 or ins[0].name != 'input_features':
            raise ValueError(f'Unexpected Smart Turn input contract: {[(x.name,x.shape) for x in ins]}')
        self.extractor = WhisperFeatureExtractor(chunk_length=8)

    def predict(self, samples) -> float:
        import numpy as np
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples) > 128000: samples = samples[-128000:]
        elif len(samples) < 128000: samples = np.pad(samples, (128000-len(samples), 0))
        inp = self.extractor(samples, sampling_rate=16000, return_tensors='np',
                             padding='max_length', max_length=128000,
                             truncation=True, do_normalize=True)
        feat = np.asarray(inp.input_features, dtype=np.float32)
        if feat.shape != (1, 80, 800):
            raise ValueError(f'Unexpected Smart Turn feature shape: {feat.shape}')
        result = self.session.run(None, {'input_features': feat})
        p = float(np.asarray(result[0]).reshape(-1)[0])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise RuntimeError(f'Smart Turn output outside [0,1]: {p}')
        return p
