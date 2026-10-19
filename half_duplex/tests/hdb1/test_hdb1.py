"""Pure policy & adapter contract tests; NEVER treated as real ONNX/ASR validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hdb1.decision import Config, EndpointPolicy
from hdb1.pipeline import TurnPipeline, dump_wav


class FakeVad:
    """Scripted fake only in tests, NEVER used by production CLI."""
    def __init__(self,events_per_call): self.events_per_call=events_per_call;self.n=0
    def feed(self,data):
        self.n+=1
        return self.events_per_call.get(self.n,[])
    def reset(self): self.n=0


class FakeSmart:
    def __init__(self, probs): self.probs=list(probs);self.seen=[]
    def predict(self,pcm):
        self.seen.append(len(pcm))
        return self.probs.pop(0)


class PolicyTest(unittest.TestCase):
    def test_nonfinal_keeps_turn(self):
        p=EndpointPolicy(Config(min_turn_ms=100,endpoint_hold_ms=200,max_turn_ms=1200,max_hesitation_ms=400))
        p.speech_start(100)
        self.assertTrue(p.speech_end(4000))
        self.assertEqual(p.advance(4000+200*16),'SMART_TURN_CHECK')
        self.assertFalse(p.smart_result(.1))
        self.assertFalse(p.finished)
        p.speech_start(7500)
        self.assertTrue(p.speech_end(12000))
        self.assertEqual(p.advance(12000+200*16),'SMART_TURN_CHECK')
        self.assertTrue(p.smart_result(.8))
        self.assertTrue(p.finished)

    def test_forced_timeout_and_eof_distinct(self):
        p=EndpointPolicy(Config(endpoint_hold_ms=200,max_hesitation_ms=600,max_turn_ms=1200))
        p.speech_start(0)
        self.assertTrue(p.speech_end(5000))
        self.assertEqual(p.advance(5000+200*16),'SMART_TURN_CHECK')
        self.assertFalse(p.smart_result(.2))
        self.assertIsNone(p.advance(5000+599*16))
        self.assertEqual(p.advance(5000+600*16),'HESITATION_TIMEOUT')
        p.reset();p.speech_start(30)
        self.assertEqual(p.eof(),'EOF_WHILE_SPEAKING')

    def test_invalid_and_regression(self):
        with self.assertRaises(ValueError): Config(threshold=1.5).validate()
        p=EndpointPolicy()
        p.advance(100)
        with self.assertRaises(ValueError):p.advance(99)
        with self.assertRaises(ValueError):p.smart_result(.9)


class PipelineTest(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory()
    def tearDown(self): self.tmp.cleanup()

    def build(self, probs):
        events={2:[('speech_start',800)], 5:[('speech_end',4600)],
                7:[('speech_start',11000)],10:[('speech_end',15500)]}
        vad=FakeVad(events)
        smart=FakeSmart(probs)
        pipeline=TurnPipeline(vad,smart,asr=None,cfg=Config(min_turn_ms=100,endpoint_hold_ms=100),
                             report_root=Path(self.tmp.name),do_asr=False)
        return pipeline, smart

    def test_real_policy_holds_on_smart_incomplete(self):
        pipe, smart=self.build([.15,.88])
        block=np.full(1600,1500,dtype=np.int16)
        for i in range(11): pipe.feed(block)
        self.assertEqual(pipe.turn_index,1)
        self.assertEqual(pipe.stats['smart_rejected'],1)
        self.assertEqual(pipe.stats['smart_accepted'],1)
        self.assertGreater(smart.seen[1],smart.seen[0])
        self.assertGreater(pipe.turns[0]['samples'],8000)
        self.assertEqual(pipe.turns[0]['reason'],'SMART_TURN_CONFIRMED')
        report=pipe.close()
        self.assertEqual(report['turns'][0]['asr_status'],'DELIBERATELY_DISABLED')
        self.assertTrue((pipe.run_dir/'turn-000/turn.wav').is_file())
        self.assertTrue((pipe.run_dir/'events.jsonl').is_file())

    def test_eof_wont_masquerade_as_smart_success(self):
        pipe, smart=self.build([.1])
        block=np.full(1600,2000,dtype=np.int16)
        for _ in range(5):pipe.feed(block)
        pipe.end_of_stream()
        self.assertEqual(pipe.turns[0]['reason'],'EOF_FORCED_UNCONFIRMED')
        self.assertEqual(pipe.stats['smart_accepted'],0)
        pipe.close()

    def test_fred_stream_reset_with_global_sample_cursor(self):
        events={1:[('speech_start',160)],3:[('speech_end',4640)]}
        pipe=TurnPipeline(FakeVad(events), FakeSmart([.9,.95]),None,
                          Config(min_turn_ms=100,endpoint_hold_ms=100),Path(self.tmp.name),False)
        for _ in range(8):pipe.feed(np.full(1600,350,dtype=np.int16))
        pipe.end_of_stream()
        self.assertEqual(len(pipe.turns),2)
        self.assertEqual(pipe.turns[0]['reason'],'SMART_TURN_CONFIRMED')
        self.assertEqual(pipe.turns[1]['reason'],'SMART_TURN_CONFIRMED')
        pipe.close()


class ModelContractTest(unittest.TestCase):
    def test_smartturn_hash_pin_full_length(self):
        from hdb1.bootstrap import SMART_SHA256, FIRERED_PIN
        self.assertEqual(len(SMART_SHA256),64)
        self.assertEqual(len(FIRERED_PIN),40)

    def test_wav_header(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'clip.wav'
            dump_wav(out,np.array([1,2,-3],dtype=np.int16))
            import wave
            with wave.open(str(out),'rb') as f:
                self.assertEqual((f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()),(16000,1,2,3))


if __name__=='__main__':unittest.main()

class CandleContractTest(unittest.TestCase):
    def test_refuses_gpu_cpu_fallback_and_keeps_artifacts(self):
        from hdb1.pipeline import CandleAsr
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            runtime=root/'runtime'
            model=root/'model'
            model.mkdir()
            binary=runtime/'target/hdb2b-cuda-cudnn/release/parakeet-candle'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\n',encoding='utf-8')
            binary.chmod(0o755)
            for name in ('model.safetensors','config.json'):(model/name).write_bytes(b'x')
            (model/'conversion.json').write_text(json.dumps({'model':'qvac/TranscribePsy-250M-EU-sub','source_nemo_sha256':'123'}))
            asr=CandleAsr(model,runtime,'cuda-cudnn')
            output=root/'report'
            output.mkdir()
            wav=root/'voice.wav'
            dump_wav(wav,np.zeros(16000,dtype=np.int16))
            fake=MagicMock(returncode=0,stdout=json.dumps({'model_id':'qvac/TranscribePsy-250M-EU-sub',
                 'device':'Cpu','text':'Bonjour'}),stderr='')
            with patch('hdb1.pipeline.subprocess.run',return_value=fake):
                with self.assertRaisesRegex(RuntimeError,'fell back to CPU'):
                    asr.transcribe(wav,output)
            self.assertTrue((output/'asr.stdout').is_file())
            self.assertFalse((output/'transcription.txt').exists())


class RegressionR1Test(unittest.TestCase):
    """Synthetic events reproduce the *timings* in the 2026-10-18 mic log.

    NOT a simulation of FireRed, Smart Turn model confidence, or ASR quality.
    """
    def setUp(self): self.tmp=tempfile.TemporaryDirectory()
    def tearDown(self): self.tmp.cleanup()

    def test_logged_first_intervention_becomes_one_turn(self):
        # The user's first utterance was split at 3.4, 8.1 and 11.4 s.
        # New onsets at 3.8 and 9.0 s cancel the previous candidate.
        def event(kind, n): return (kind, n*1600)
        vad=FakeVad({29:[event('speech_start',29)],
                     34:[event('speech_end',34)],
                     38:[event('speech_start',38)],
                     81:[event('speech_end',81)],
                     90:[event('speech_start',90)],
                     114:[event('speech_end',114)]})
        smart=FakeSmart([.988007])
        pipe=TurnPipeline(vad,smart,None,Config(endpoint_hold_ms=1200),
                          Path(self.tmp.name),do_asr=False)
        for n in range(140):
            pipe.feed(np.full(1600,2200,dtype=np.int16))
        self.assertEqual(pipe.turn_index,1)
        self.assertEqual(pipe.stats['endpoint_candidates_cancelled'],2)
        self.assertEqual(pipe.stats['endpoint_candidates_deferred'],3)
        self.assertEqual(pipe.stats['smart_accepted'],1)
        self.assertEqual(len(smart.seen),1, 'Smart Turn only after sustained silence')
        self.assertEqual(pipe.turns[0]['reason'],'SMART_TURN_CONFIRMED')
        self.assertGreaterEqual(pipe.turns[0]['duration_sec'],9)
        pipe.close()

    def test_short_pause_does_not_trigger_smart_turn(self):
        cfg=Config(endpoint_hold_ms=1200)
        policy=EndpointPolicy(cfg)
        policy.speech_start(0)
        self.assertTrue(policy.speech_end(16000))
        self.assertIsNone(policy.advance(16000+800*16))
        self.assertTrue(policy.speech_start(16000+900*16))
        self.assertFalse(policy.pending)
        self.assertIsNone(policy.advance(16000+1300*16))

    def test_pending_stale_smart_result_is_rejected(self):
        policy=EndpointPolicy(Config())
        policy.speech_start(0)
        policy.speech_end(10000)
        with self.assertRaisesRegex(ValueError,'before endpoint hold'):
            policy.smart_result(.99)
        self.assertTrue(policy.speech_start(12000))
        with self.assertRaisesRegex(ValueError,'no unchecked pending'):
            policy.smart_result(.99)

    def test_hesitation_timeout_is_not_semantic_approval(self):
        cfg=Config(endpoint_hold_ms=1200,max_hesitation_ms=3200)
        policy=EndpointPolicy(cfg)
        policy.speech_start(0)
        policy.speech_end(16000)
        self.assertEqual(policy.advance(16000+1200*16),'SMART_TURN_CHECK')
        self.assertFalse(policy.smart_result(.4))
        self.assertIsNone(policy.advance(16000+3100*16))
        self.assertEqual(policy.advance(16000+3200*16),'HESITATION_TIMEOUT')
        self.assertTrue(policy.finished)

    def test_config_guard_rejects_unsafe_bounds(self):
        for c in (Config(endpoint_hold_ms=0),
                  Config(endpoint_hold_ms=3200),
                  Config(endpoint_hold_ms=3300)):
            with self.assertRaises(ValueError): c.validate()

    def test_mic_restart_path_is_active(self):
        import inspect
        from hdb1.cli import main
        source=inspect.getsource(main)
        self.assertIn("if pipeline.feed(pcm) and a.mode=='mic':",source)
        self.assertIn("capture_restart_required_after_asr",source)

