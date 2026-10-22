"""Software-only, never pretends to run the GPU or real QVAC model."""
import contextlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from hdb2b2_runtime_patch import patch_main,MARKER
from hdb2b_runtime_patch import patch_main as b_patch
from hdb2b2_driver import (describe,nearest_rank,finite_positive,validate_report,compare3,
                            PASS,VramObserver,mib,PHASES)

BASE='''struct Args {
    model_dir: Option<PathBuf>,
}
fn main() -> Result<()> {
    eprintln!("HDB2A_CANDLE_LOCAL_MODEL={}", dir.display());
    let mut model = from_config_value(config, vb)?;
    if args.debug {
        print_debug(&mut model, &args.input)?;
    }
}
'''

def fake(v,ms,text='Un enregistrement de test'):
    from hdb2b2_driver import VARIANTS
    return dict(stage='HDB2B2_SM61_FINAL_SINGLE_AUDIO',status=PASS,variant=v,
        measured_device='Cpu' if v=='cpu' else 'Cuda(CudaDevice(DeviceId(1)))',
        model_id='qvac/TranscribePsy-250M-EU-sub',checkpoint_sha256='samesha',
        normalized_pcm_sha256='samepcm',runtime_pin='same_runtime',candle_pin='same_candle',
        transcript=text,binary_sha256=v+'sha',
        build_manifest={'variant':v,'binary_sha256':v+'sha'},
        metrics=dict(dtype='f32',text_consistent=True,
          measured_device='Cpu' if v=='cpu' else 'Cuda(CudaDevice(DeviceId(1)))',
          warmup_ms=[ms]*3,hot_samples_ms=[ms]*10,
          phase_profile_samples=[{key:ms/6 for key in PHASES}]*3),
        hot_ms_median=ms,hot_ms_p95=ms,hot_rtf_median=ms/10000,
        phase_profile={key:{'median_ms':ms/6} for key in PHASES})

class Tests(unittest.TestCase):
    def test_patch_real_hdb2b_synthetic_and_idempotence(self):
        baseline=b_patch(BASE)
        self.assertIn('HDB2B_IN_PROCESS_BENCH_V1',baseline)
        patched=patch_main(baseline)
        self.assertEqual(patch_main(patched),patched)
        self.assertEqual(patched.count(MARKER),1)
        self.assertIn('device_ref.synchronize()?;',patched)
        self.assertIn('m.encoder.forward(&mel, None)?',patched)
        self.assertIn('m.decode(',patched)
        self.assertLess(patched.index(MARKER),patched.index('// HDB2B_IN_PROCESS_BENCH_V1'))
    def test_patch_requires_existing_b(self):
        with self.assertRaisesRegex(ValueError,'baseline'):patch_main(BASE)
    def test_percentile_nearest_rank(self):
        x=list(range(1,11))
        self.assertEqual(nearest_rank(x,.95),10)
        self.assertEqual(describe(x)['median_ms'],5.5)
        self.assertEqual(describe(x)['p95_nearest_rank_ms'],10)
    def test_invalid_samples(self):
        for bad in (0,-1,True,float('nan'),float('inf'),None):
            self.assertFalse(finite_positive(bad))
        with self.assertRaises(ValueError):describe([1,0])
    def test_memory_parse(self):
        self.assertEqual(mib('350 MiB'),350.0)
        self.assertIsNone(mib('N/A'))
        self.assertIsNone(VramObserver(False).result()['process_peak_mib'])
    def test_all_three_comparison(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            paths=[]
            for v,t in [('cpu',6400),('cuda',730),('cuda-cudnn',325)]:
                p=root/f'{v}.json';p.write_text(json.dumps(fake(v,t)));paths.append(p)
            a=SimpleNamespace(cpu=paths[0],cuda=paths[1],cuda_cudnn=paths[2],report_root=root)
            with contextlib.redirect_stdout(StringIO()):compare3(a)
            doc=json.loads(next(root.glob('hdb2b2-compare3-*/comparison.json')).read_text())
            self.assertEqual(doc['status'],'PASS_IDENTICAL_TEXT_ALL_VARIANTS')
            self.assertEqual(doc['fastest_variant'],'cuda-cudnn')
            self.assertAlmostEqual(doc['speedup_cuda_over_cuda_cudnn'],730/325)
    def test_strict_divergence_fails(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);paths=[]
            for v,t in [('cpu',6400),('cuda',730),('cuda-cudnn',325)]:
                p=root/f'{v}.json';p.write_text(json.dumps(fake(v,t,'Different' if v=='cuda-cudnn' else 'Same')));paths.append(p)
            a=SimpleNamespace(cpu=paths[0],cuda=paths[1],cuda_cudnn=paths[2],report_root=root)
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):compare3(a)
    def test_mismatched_audio_fails(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);paths=[]
            for v,t in [('cpu',6400),('cuda',730),('cuda-cudnn',325)]:
                r=fake(v,t)
                if v=='cuda':r['normalized_pcm_sha256']='other'
                p=root/f'{v}.json';p.write_text(json.dumps(r));paths.append(p)
            a=SimpleNamespace(cpu=paths[0],cuda=paths[1],cuda_cudnn=paths[2],report_root=root)
            with contextlib.redirect_stdout(StringIO()),contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):compare3(a)
    def test_requires_ten_runs_three_warmups(self):
        r=fake('cuda',325)
        self.assertEqual(validate_report(r,'cuda'),'cuda')
        r['metrics']['hot_samples_ms']=[325]*3
        with self.assertRaisesRegex(ValueError,'insufficient'):validate_report(r,'cuda')
    def test_rejects_forged_cpu_gpu_device(self):
        r=fake('cuda',200)
        r['measured_device']='Cpu'
        with self.assertRaisesRegex(ValueError,'CUDA'):validate_report(r,'cuda')
    def test_does_not_perturb_b2_legacy_marker(self):
        code=patch_main(b_patch(BASE))
        self.assertIn('hdb2b_bench_runs',code)
        self.assertIn('hdb2b2_hot_runs',code)
        self.assertIn('HDB2B_BENCH_JSON=',code)
        self.assertIn('HDB2B2_BENCH_JSON=',code)

if __name__=='__main__':unittest.main()
