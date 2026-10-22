"""Half-duplex: true streaming FireRedVAD -> SmartTurn -> QVAC Candle ASR.

ONNX sessions are held resident. Candle ASR is a separate, already-built Rust
executable launched per accepted turn (COLD per turn). VAD endpoints are held
for uninterrupted silence before Smart Turn is consulted.
No fallback to energy VAD, no fake ASR, no silent model substitution.
"""
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import wave

from .decision import Config, EndpointPolicy

MODEL_ID = 'qvac/TranscribePsy-250M-EU-sub'


def hash_file(path, block_size=2**20):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while x := f.read(block_size): h.update(x)
    return h.hexdigest()


def dump_wav(path, pcm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(pcm.astype('<i2', copy=False).tobytes())


class CandleAsr:
    def __init__(self, model_dir, runtime_dir, variant='cuda-cudnn'):
        self.model_dir = Path(model_dir)
        self.runtime_dir = Path(runtime_dir)
        self.variant = variant
        self.binary = self.runtime_dir / f'target/hdb2b-{variant}/release/parakeet-candle'
        if not self.binary.is_file():
            raise FileNotFoundError(f'ASR binary missing: {self.binary}. Build HDB2-B with ./run-hdb2b.sh build {variant}')
        if not os.access(self.binary, os.X_OK):
            raise PermissionError(f'ASR binary not executable: {self.binary}')
        for item in ('model.safetensors','config.json','conversion.json'):
            if not (self.model_dir / item).is_file():
                raise FileNotFoundError(f'ASR model missing {item} in {self.model_dir}')
        provenance = json.loads((self.model_dir/'conversion.json').read_text())
        if provenance.get('model') != MODEL_ID: raise ValueError('ASR model provenance mismatch')
        self.checkpoint_sha256 = provenance['source_nemo_sha256']

    def transcribe(self, wav_file, output_dir):
        args = [str(self.binary), '--model-id', MODEL_ID, '--model-dir', str(self.model_dir),
                '--input', str(wav_file), '--format', 'json']
        if self.variant == 'cpu': args.append('--cpu')
        t0 = time.perf_counter()
        run = subprocess.run(args, capture_output=True, text=True, timeout=120)
        elapsed = (time.perf_counter()-t0)*1000
        (output_dir/'asr.stderr').write_text(run.stderr, encoding='utf-8')
        (output_dir/'asr.stdout').write_text(run.stdout, encoding='utf-8')
        if run.returncode: raise RuntimeError(f'Candle ASR exited {run.returncode}: {run.stderr[-1000:]}')
        obj = json.loads(run.stdout)
        if obj.get('model_id') != MODEL_ID: raise RuntimeError('ASR model ID mismatch')
        device = str(obj.get('device', ''))
        if self.variant == 'cpu' and 'cpu' not in device.lower():
            raise RuntimeError('ASR CPU fallback/identity mismatch')
        if self.variant != 'cpu' and 'cuda' not in device.lower():
            raise RuntimeError('ASR GPU fell back to CPU')
        text = obj.get('text', '')
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('Candle did not return a nonempty transcription')
        (output_dir/'transcription.txt').write_text(text.strip()+'\n', encoding='utf-8')
        return {'text': text.strip(), 'runtime_device': device, 'asr_cold_wall_ms': round(elapsed,3)}


class TurnPipeline:
    def __init__(self, vad, smart, asr, cfg: Config, report_root: Path, do_asr=True):
        import numpy as np
        self.np = np
        self.vad = vad
        self.smart = smart
        self.asr = asr
        self.do_asr = do_asr
        self.cfg = cfg
        self.cfg.validate()
        self.policy = EndpointPolicy(cfg)
        self.root = Path(report_root)
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.run_dir = self.root / f'hdb1-{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}'
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.event_file = (self.run_dir/'events.jsonl').open('w', encoding='utf-8', buffering=1)
        self.pre_roll = deque()
        self.pre_roll_samples = 0
        self.turn_chunks = []
        self.turn_n = 0
        self.cursor = 0
        self.vad_base = 0
        self.turn_index = 0
        self.turns = []
        self.stats = dict(vad_speech_starts=0, vad_speech_ends=0,
                          endpoint_candidates_deferred=0, endpoint_candidates_cancelled=0,
                          smart_rejected=0, smart_accepted=0, forced=0,
                          asr_completed=0, asr_failed=0)

    def log(self, event, **fields):
        row = dict(event=event, sample=self.cursor, seconds=round(self.cursor/16000, 4), **fields)
        self.event_file.write(json.dumps(row, ensure_ascii=False)+'\n')
        if event in ('speech_start','speech_end','smart_turn','transcription','forced_endpoint','error',
                     'endpoint_candidate_deferred','endpoint_candidate_cancelled',
                     'capture_restart_required_after_asr','capture_restarted_after_asr'):
            print('HDB1_EVENT='+json.dumps(row, ensure_ascii=False), flush=True)

    def _append(self, pcm):
        self.turn_chunks.append(pcm.copy())
        self.turn_n += len(pcm)
        if self.turn_n > self.cfg.max_pcm_ms * 16:
            raise RuntimeError('Turn exceeded PCM allocation bound; refusing silent truncation')

    def _pre_roll_add(self, pcm):
        self.pre_roll.append(pcm.copy())
        self.pre_roll_samples += len(pcm)
        bound = self.cfg.pre_roll_ms * 16
        while self.pre_roll and self.pre_roll_samples - len(self.pre_roll[0]) >= bound:
            self.pre_roll_samples -= len(self.pre_roll.popleft())

    def _clip(self):
        if not self.turn_chunks: return self.np.empty(0, dtype=self.np.int16)
        return self.np.concatenate(self.turn_chunks)

    def feed(self, pcm):
        np = self.np
        pcm = np.asarray(pcm, dtype=np.int16)
        if pcm.ndim != 1: raise ValueError('PCM must be mono')
        if len(pcm) == 0: return False
        start_cursor = self.cursor
        self.cursor += len(pcm)
        # Process the chunk ONCE, keeping the streaming VAD cache across chunks.
        events = self.vad.feed(pcm)
        was_active = self.policy.started is not None and not self.policy.finished
        if was_active: self._append(pcm)
        for name, frame_sample in events:
            # Stream VAD includes a backdated speech onset pad. Keep monotonic
            # control time, but preserve the acoustic pre-roll in turn buffer.
            abs_frame_sample = self.vad_base + int(frame_sample)
            point = max(self.policy.last_sample, min(self.cursor, abs_frame_sample))
            if name == 'speech_start':
                self.stats['vad_speech_starts'] += 1
                if self.policy.started is None:
                    for x in self.pre_roll: self._append(x)
                    if not was_active: self._append(pcm)
                cancelled = self.policy.speech_start(point)
                if cancelled:
                    self.stats['endpoint_candidates_cancelled'] += 1
                    self.log('endpoint_candidate_cancelled',
                             reason='SPEECH_RESUMED_DURING_HOLD',
                             vad_frame_sample=abs_frame_sample)
                self.log('speech_start', vad_frame_sample=abs_frame_sample)
            elif name == 'speech_end':
                self.stats['vad_speech_ends'] += 1
                candidate = self.policy.speech_end(point)
                self.log('speech_end', candidate=candidate, vad_frame_sample=abs_frame_sample)
                if candidate:
                    self.stats['endpoint_candidates_deferred'] += 1
                    self.log('endpoint_candidate_deferred',
                             hold_ms=self.cfg.endpoint_hold_ms,
                             reason='AWAIT_UNINTERRUPTED_SILENCE')
            else:
                raise RuntimeError(f'Unrecognized VAD event {name}')
        timeout = self.policy.advance(self.cursor)
        if timeout == 'SMART_TURN_CHECK':
            t0 = time.perf_counter()
            probability = self.smart.predict(self._clip().astype(np.float32)/32768.0)
            ms = (time.perf_counter()-t0)*1000
            accepted = self.policy.smart_result(probability)
            self.stats['smart_accepted' if accepted else 'smart_rejected'] += 1
            self.log('smart_turn', probability=round(probability,6),
                     latency_ms=round(ms,3), complete=accepted,
                     post_vad_silence_ms=round((self.cursor-self.policy.last_vad_end)/16,1))
            if accepted:
                self._finalize('SMART_TURN_CONFIRMED')
                return True
        elif timeout:
            self.stats['forced'] += 1
            self.log('forced_endpoint', reason=timeout)
            self._finalize(timeout)
            return True
        self._pre_roll_add(pcm)
        return False

    def _finalize(self, reason):
        pcm = self._clip()
        if len(pcm) == 0: raise RuntimeError('Empty speech buffer')
        turn_dir = self.run_dir/f'turn-{self.turn_index:03d}'
        self.turn_index += 1
        turn_dir.mkdir()
        path = turn_dir/'turn.wav'
        dump_wav(path, pcm)
        outcome = dict(reason=reason, samples=len(pcm), duration_sec=len(pcm)/16000,
                       wav=str(path), asr_status='NOT_REQUESTED')
        try:
            if self.do_asr:
                outcome.update(self.asr.transcribe(path, turn_dir))
                outcome['asr_status']='REAL_CANDLE_ASR'
                self.stats['asr_completed'] += 1
                self.log('transcription', text=outcome['text'], turn_index=self.turn_index-1,
                         asr_cold_wall_ms=outcome['asr_cold_wall_ms'])
            else:
                outcome['asr_status']='DELIBERATELY_DISABLED'
        except Exception as e:
            outcome['asr_status']='FAILED'
            outcome['error']=f'{type(e).__name__}: {e}'
            self.stats['asr_failed'] += 1
            self.log('error', stage='CANDLE_ASR', message=str(e))
            raise
        finally:
            (turn_dir/'turn-report.json').write_text(json.dumps(outcome,indent=2,ensure_ascii=False)+'\n')
            self.turns.append(outcome)
            self.turn_chunks.clear()
            self.turn_n = 0
            # Never splice audio from a finalized turn into the next turn.
            self.pre_roll.clear()
            self.pre_roll_samples = 0
            self.policy.reset()
            self.vad.reset()
            # VAD time-frame origin restarts but sample cursor does not.
            self.vad_base = self.cursor

    def end_of_stream(self):
        # Deliberately reject any false claim that an EOF was a Smart Turn success.
        if self.policy.started is not None and not self.policy.finished:
            reason = self.policy.eof()
            self.stats['forced'] += 1
            self.log('forced_endpoint', reason=reason)
            self._finalize(reason)

    def close(self, status='COMPLETE'):
        report=dict(stage='HDB1_FIREREDVAD_SMART_TURN_CANDLE',status=status,
                    model_id=MODEL_ID,turns=self.turns,stats=self.stats,
                    streaming_vad='FireRedVAD_ONNX',endpoint='Smart_Turn_v3.2_ONNX',
                    audio_rate=16000,total_samples=self.cursor,
                    no_synthetic_inference=True,
                    asr_execution='SEPARATE_COLD_PROCESS_PER_TURN',
                    endpoint_hold_ms=self.cfg.endpoint_hold_ms,
                    report_dir=str(self.run_dir),
                    model_fingerprints=dict(
                        firered_onnx_sha256=getattr(self.vad,'model_sha256',None),
                        smartturn_onnx_sha256=getattr(self.smart,'model_sha256',None),
                        qvac_checkpoint_sha256=getattr(self.asr,'checkpoint_sha256',None)))
        self.event_file.close()
        (self.run_dir/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        print(f'HDB1_STATUS={status} HDB1_REPORT={self.run_dir/"report.json"}', flush=True)
        return report
