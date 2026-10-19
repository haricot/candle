"""FireRedVAD ONNX with streaming model caches and Kaldi FBank.

Derived independently from the documented ONNX contract of
https://github.com/eschmidbauer/fireredvad-onnx (MIT); postprocessor
behavior follows FireRedTeam/FireRedVAD (Apache-2.0). Does not use
loudness/RMS as a surrogate for real VAD inference.
"""
from collections import deque
from pathlib import Path


class FireRedOnnx:
    def __init__(self, model_file: Path, cmvn_file: Path, threshold: float = .4):
        import numpy as np
        import onnxruntime as ort
        from .pipeline import hash_file
        self.model_sha256 = hash_file(model_file)
        import kaldi_native_fbank as knf
        import kaldiio
        self.np = np
        self.session = ort.InferenceSession(str(model_file), providers=['CPUExecutionProvider'])
        ins = [(x.name, list(x.shape)) for x in self.session.get_inputs()]
        outs = [x.name for x in self.session.get_outputs()]
        if len(ins) != 2 or ins[0][0] != 'feat' or ins[1][0] != 'caches_in':
            raise ValueError(f'Unexpected FireRedVAD ONNX inputs: {ins!r}')
        if len(outs) != 2:
            raise ValueError(f'Unexpected FireRedVAD ONNX outputs: {outs!r}')
        # Freeze cached-stream model topology, fail if not compatible.
        self.caches = np.zeros((8, 1, 128, 19), dtype=np.float32)
        stats = np.asarray(kaldiio.load_mat(str(cmvn_file)))
        if stats.shape != (2, 81) or stats[0, 80] < 1:
            raise ValueError(f'CMVN shape/count invalid: {stats.shape!r}')
        count = stats[0, 80]
        self.mean = np.asarray(stats[0, :80] / count, dtype=np.float32)
        variance = np.maximum(stats[1, :80] / count - self.mean**2, 1e-20)
        self.invstd = np.asarray(1 / np.sqrt(variance), dtype=np.float32)
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        self.opts = opts
        self.threshold = threshold
        self.remainder = np.empty(0, dtype=np.int16)
        self.events = deque()
        self.reset()

    def reset(self):
        np = self.np
        self.caches.fill(0)
        self.remainder = np.empty(0, dtype=np.int16)
        self.window = deque(maxlen=5)
        self.state = 'SILENCE'
        self.speech_count = 0
        self.silence_count = 0
        self.frame_count = 0
        self.start_frame = None
        self.events.clear()

    def feed(self, chunk):
        np = self.np
        chunk = np.asarray(chunk, dtype=np.int16)
        samples = np.concatenate((self.remainder, chunk))
        if len(samples) < 400:
            self.remainder = samples
            return []
        import kaldi_native_fbank as knf
        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(16000, samples.tolist())
        frames = fbank.num_frames_ready
        if frames <= 0:
            self.remainder = samples
            return []
        feat = np.stack([fbank.get_frame(i) for i in range(frames)]).astype(np.float32)
        self.remainder = samples[frames * 160:]
        feat = ((feat - self.mean) * self.invstd)[np.newaxis, :, :]
        out = self.session.run(None, {'feat': feat, 'caches_in': self.caches})
        probs = np.asarray(out[0], dtype=np.float32).reshape(-1)
        caches = np.asarray(out[1], dtype=np.float32)
        if caches.shape != self.caches.shape or len(probs) != frames:
            raise RuntimeError(f'VAD ONNX shape mismatch: probs={probs.shape}, caches={caches.shape}, frames={frames}')
        self.caches = caches
        events = []
        for p in probs:
            value = float(p)
            if not np.isfinite(value) or value < 0 or value > 1:
                raise RuntimeError('VAD probability invalid')
            self.frame_count += 1
            self.window.append(value)
            speech = sum(self.window)/len(self.window) >= self.threshold
            if self.state == 'SILENCE':
                if speech:
                    self.speech_count = 1
                    self.state = 'POSSIBLE_SPEECH'
            elif self.state == 'POSSIBLE_SPEECH':
                if speech:
                    self.speech_count += 1
                    if self.speech_count >= 8:
                        self.state = 'SPEECH'
                        self.silence_count = 0
                        self.start_frame = max(0, self.frame_count-self.speech_count-5)
                        events.append(('speech_start', self.start_frame * 160))
                else:
                    self.state='SILENCE'
                    self.speech_count=0
            elif self.state == 'SPEECH':
                self.speech_count += 1
                if speech:
                    self.silence_count = 0
                else:
                    self.silence_count = 1
                    self.state = 'POSSIBLE_SILENCE'
            elif self.state == 'POSSIBLE_SILENCE':
                self.speech_count += 1
                if speech:
                    self.silence_count = 0
                    self.state = 'SPEECH'
                else:
                    self.silence_count += 1
                    if self.silence_count >= 20:
                        events.append(('speech_end', self.frame_count * 160))
                        self.state = 'SILENCE'
                        self.start_frame = None
                        self.speech_count = 0
            if self.state in ('SPEECH','POSSIBLE_SILENCE') and self.speech_count >= 2000:
                events.append(('speech_end', self.frame_count * 160))
                self.state = 'SILENCE'
                self.start_frame = None
                self.speech_count = 0
        return events

    def flush(self):
        if self.state in ('SPEECH','POSSIBLE_SILENCE'):
            self.state = 'SILENCE'
            return [('speech_end', self.frame_count*160)]
        return []
