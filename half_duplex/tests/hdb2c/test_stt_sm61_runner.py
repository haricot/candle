"""Bind the physical runner to one real STT job; no fabricated GPU evidence."""
import importlib.util
import errno
import os
from pathlib import Path
import pty
import select
import signal
import sys
import time
import unittest

spec=importlib.util.spec_from_file_location('stt_runner',Path(__file__).resolve().parents[2]/'ci/stt_sm61_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerBindingTests(unittest.TestCase):
    def setUp(self):
        self.run={'id':12345678,'run_attempt':2,'path':runner.WORKFLOW,'head_branch':'half_duplex',
            'event':'push','status':'queued','conclusion':None,'head_sha':'a'*40}
        self.job={'id':87654321,'name':'cuda-cudnn-sm61','run_id':12345678,'run_attempt':2,
            'status':'queued','conclusion':None,'labels':['self-hosted','linux','x64','cuda','sm61',
                'half-duplex-stt','half-duplex-stt-12345678']}

    def test_exact_campaign_and_candidate_required(self):
        self.assertEqual(runner.validate(self.run,[self.job],'a'*40)['gpu_job_id'],87654321)
        with self.assertRaisesRegex(ValueError,'candidate SHA changed'):
            runner.validate(self.run,[self.job],'b'*40)
        self.run['head_branch']='extended'
        with self.assertRaises(ValueError):runner.validate(self.run,[self.job])

    def test_generic_or_other_campaign_labels_rejected(self):
        self.job['labels']=self.job['labels'][:-1]
        with self.assertRaisesRegex(ValueError,'exact campaign labels'):
            runner.validate(self.run,[self.job])
        self.job['labels'].append('half-duplex-stt-99999999')
        with self.assertRaises(ValueError):runner.validate(self.run,[self.job])

    def test_finished_busy_duplicated_and_stale_attempt_jobs_rejected(self):
        with self.assertRaises(ValueError):runner.validate(self.run,[self.job,self.job])
        for update in ({'status':'in_progress'},{'run_attempt':1},{'run_id':99999999}):
            with self.subTest(update=update),self.assertRaises(ValueError):
                runner.validate(self.run,[{**self.job,**update}])
        self.run.update(status='completed',conclusion='failure')
        with self.assertRaisesRegex(ValueError,'finished'):runner.validate(self.run,[self.job])

    def test_registration_accepts_hidden_input_on_non_seekable_tty(self):
        # Exercise a real controlling terminal. Mocking open/getpass would miss
        # the r+ TextIO seek failure observed on the physical laptop.
        source=str(Path(runner.__file__).parent)
        script=(f'import sys; sys.path.insert(0,{source!r}); '
            'from stt_sm61_runner import registration_token; '
            'value=registration_token(); '
            'print("HIDDEN_TOKEN_RECEIVED="+str(value=="synthetic-pty-token"),flush=True)')
        pid,fd=pty.fork()
        if pid==0:
            os.execv(sys.executable,[sys.executable,'-c',script])
        output=bytearray();sent=False;status=None
        try:
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                if select.select([fd],[],[],.1)[0]:
                    try:part=os.read(fd,4096)
                    except OSError as exc:
                        if exc.errno==errno.EIO:break
                        raise
                    if not part:break
                    output.extend(part)
                    if b'(hidden): ' in output and not sent:
                        os.write(fd,b'synthetic-pty-token\n');sent=True
                child,status_value=os.waitpid(pid,os.WNOHANG)
                if child:status=status_value;break
            if status is None:
                child,status_value=os.waitpid(pid,os.WNOHANG)
                if child:status=status_value
            self.assertTrue(sent,bytes(output))
            self.assertIn(b'HIDDEN_TOKEN_RECEIVED=True',output)
            self.assertNotIn(b'synthetic-pty-token',output)
            if status is not None:self.assertEqual(os.waitstatus_to_exitcode(status),0)
        finally:
            if status is None:
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
                os.waitpid(pid,0)
            os.close(fd)
