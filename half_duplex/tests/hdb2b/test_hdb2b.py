"""HDB2-B structural tests. No CUDA perf fabricated from these tests."""
import contextlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE/'scripts'))
from hdb2b_runtime_patch import patch_main, MARKER
from hdb2b_driver import (compare, nearest_p, ensure_gpu, TARGET_CAP, check_build,
                         VARIANTS, pcm_digest, GlobalVramSampler)

MAIN_SOURCE='''use std::path::{Path, PathBuf};
struct Args {
    model_dir: Option<PathBuf>,
}
fn main() -> Result<()> {
    let t0 = Instant::now();
    eprintln!("HDB2A_CANDLE_LOCAL_MODEL={}", dir.display());
    let mut model = from_config_value(config, vb)?;
    let decoding_config = DecodingConfig::default();
    if args.debug {
        print_debug(&mut model, &args.input)?;
    }
}
'''

class Hdb2bTests(unittest.TestCase):
    def test_patches_main_idempotently_without_changing_regular_path(self):
        out=patch_main(MAIN_SOURCE)
        self.assertIn(MARKER,out)
        self.assertIn('hdb2b_bench_runs: usize',out)
        self.assertIn('hdb2b_bench_warmup: usize',out)
        self.assertIn('samples_ms.push(elapsed);',out)
        self.assertIn('model.transcribe(&args.input, &decoding_config, None, 0.0, None)',out)
        self.assertIn('print_debug(&mut model, &args.input)?;',out)
        self.assertEqual(patch_main(out),out)
    def test_rejects_unknown_runtime_revision_and_no_baseline(self):
        with self.assertRaisesRegex(ValueError,'HDB2A'):
            patch_main(MAIN_SOURCE.replace('HDB2A_CANDLE_LOCAL_MODEL','some_other_model'))
        with self.assertRaisesRegex(ValueError,'pin drift'):
            patch_main(MAIN_SOURCE.replace('    model_dir: Option<PathBuf>,\n',''))
    def test_nearest_rank_p95_and_arch(self):
        self.assertEqual(nearest_p([20,10,40,30],0.95),40)
        self.assertEqual(nearest_p([20,10,40,30],0.5),20)
        self.assertEqual(TARGET_CAP,'61')
        self.assertEqual(VARIANTS['cuda-cudnn'],('cuda','cudnn'))
    def test_sm61_guard(self):
        with patch('hdb2b_driver.query_gpu',return_value={'compute_cap':'8.9'}):
            with self.assertRaisesRegex(RuntimeError,'sm_61'):ensure_gpu()
        with patch('hdb2b_driver.query_gpu',return_value={'compute_cap':'6.1','name':'Pascal'}):
            self.assertEqual(ensure_gpu()['name'],'Pascal')
    def reports(self,where, gpu_text=None, gpu_sha='abc'):
        cpu={'status':'REAL_CANDLE_HOT_BENCH_OBSERVED_UNQUALIFIED_WER',
             'variant':'cpu','checkpoint_sha256':'abc','normalized_pcm_sha256':'wav',
             'model_id':'qvac/TranscribePsy-250M-EU-sub','runtime_pin':'runtime','candle_pin':'candle',
             'transcript':'Bonjour CUDA', 'measured_device':'Cpu',
             'metrics':{'dtype':'f32','text_consistent':True,'measured_device':'Cpu'},'hot_ms_median':400,
             'hot_rtf_median':0.4}
        gpu={**cpu,'variant':'cuda','measured_device':'Cuda(CudaDevice(DeviceId(1)))',
             'metrics':{'dtype':'f32','text_consistent':True,'measured_device':'Cuda(CudaDevice(DeviceId(1)))'},
             'checkpoint_sha256':gpu_sha,
             'transcript':gpu_text if gpu_text is not None else cpu['transcript'],
             'hot_ms_median':200,'hot_rtf_median':0.2}
        cpu_file=where/'cpu.json';gpu_file=where/'gpu.json'
        cpu_file.write_text(json.dumps(cpu));gpu_file.write_text(json.dumps(gpu))
        return SimpleNamespace(first=cpu_file,second=gpu_file,report_root=where)
    def test_comparison_honors_exact_transcription(self):
        with tempfile.TemporaryDirectory() as d:
            where=Path(d)
            a=self.reports(where)
            with contextlib.redirect_stdout(StringIO()): compare(a)
            paths=list(where.glob('hdb2b-compare-*/comparison.json'))
            self.assertEqual(len(paths),1)
            report=json.loads(paths[0].read_text())
            self.assertEqual(report['status'],'PASS_IDENTICAL_TEXT')
            self.assertEqual(report['speedup_cpu_over_gpu'],2.0)
    def test_comparison_rejects_text_divergence(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.reports(Path(d),gpu_text='Autre texte')
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as e:compare(a)
            self.assertEqual(e.exception.code,1)
    def test_comparison_rejects_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.reports(Path(d),gpu_sha='different')
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):compare(a)
    def test_comparison_cuda_vs_cuda_cudnn(self):
        with tempfile.TemporaryDirectory() as d:
            where=Path(d)
            args=self.reports(where)
            gpu=json.loads(args.second.read_text())
            gpu['variant']='cuda-cudnn'
            gpu['hot_ms_median']=100
            gpu['hot_rtf_median']=0.1
            second=where/'cuda-cudnn.json'
            second.write_text(json.dumps(gpu))
            with contextlib.redirect_stdout(StringIO()):
                compare(SimpleNamespace(first=args.second,second=second,report_root=where))
            files=list(where.glob('hdb2b-compare-*/comparison.json'))
            report=json.loads(files[0].read_text())
            self.assertEqual(report['status'],'PASS_IDENTICAL_TEXT')
            self.assertEqual(report['first_variant'],'cuda')
            self.assertEqual(report['second_variant'],'cuda-cudnn')
            self.assertEqual(report['speedup_first_over_second'],2.0)
            self.assertEqual(report['second_latency_reduction_pct'],50.0)
            self.assertNotIn('speedup_cpu_over_gpu',report)
    def test_comparison_reverse_order(self):
        with tempfile.TemporaryDirectory() as d:
            where=Path(d)
            a=self.reports(where)
            with contextlib.redirect_stdout(StringIO()):
                compare(SimpleNamespace(first=a.second,second=a.first,report_root=where))
            report=json.loads(next(where.glob('hdb2b-compare-*/comparison.json')).read_text())
            self.assertEqual(report['first_variant'],'cuda')
            self.assertEqual(report['second_variant'],'cpu')
            self.assertEqual(report['speedup_first_over_second'],0.5)
    def test_comparison_rejects_same_variant(self):
        with tempfile.TemporaryDirectory() as d:
            where=Path(d)
            a=self.reports(where)
            other=where/'another-cpu.json'
            other.write_text(a.first.read_text())
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    compare(SimpleNamespace(first=a.first,second=other,report_root=where))
    def test_comparison_rejects_wrong_measured_device(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.reports(Path(d))
            gpu=json.loads(a.second.read_text())
            gpu['measured_device']='Cpu'
            a.second.write_text(json.dumps(gpu))
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):compare(a)
    def test_comparison_rejects_mismatched_audio(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.reports(Path(d))
            gpu=json.loads(a.second.read_text())
            gpu['normalized_pcm_sha256']='different'
            a.second.write_text(json.dumps(gpu))
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):compare(a)
    def test_comparison_rejects_uncorroborated_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.reports(Path(d))
            gpu=json.loads(a.second.read_text())
            gpu['metrics']['text_consistent']=False
            a.second.write_text(json.dumps(gpu))
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):compare(a)
    def test_vram_scope_is_not_misattributed(self):
        s=GlobalVramSampler(False)
        r=s.result()
        self.assertIsNone(r['peak_mib'])
        self.assertIn('not process-attributed',r['scope'])

if __name__=='__main__':unittest.main()
