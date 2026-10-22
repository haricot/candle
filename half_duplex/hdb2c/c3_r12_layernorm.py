"""R12: actual first-LayerNorm common operands on 14 public French clips.

This is an isolated operator experiment. It does not integrate the candidate
into FastConformer or qualify the full STT, TDT, microphone or streaming path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .audio import read_pcm, sha256
from .c3_r7_metrics import tensor_metrics
from .c3_r8 import selected_corpus, write_json
from .layernorm_operator import CpuNorm, CudaNorm, sha
from .nemo_runner import load_nemo, ensure_probe_eval

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'BLOCK0_NORM_FF1_COMMON_INPUT_ONLY_NOT_FULL_STT'


def capture(model, torch, wav):
    """Observe the real unmodified NeMo operator and verify the hook's effect."""
    norm = model.encoder.layers[0].norm_feed_forward1
    if (not isinstance(norm, torch.nn.LayerNorm) or not norm.elementwise_affine
            or norm.bias is None or norm.eps != 1e-5):
        raise ValueError('Expected the original affine FP32 first LayerNorm with epsilon=1e-5')
    pcm = read_pcm(wav)
    signal = torch.from_numpy(pcm.copy())[None, :]
    lengths = torch.tensor([len(pcm)], dtype=torch.int64)
    captured = []
    def observe(module, args, output):
        if args[0].dtype != torch.float32 or output.dtype != torch.float32:
            raise ValueError('Expected real FP32 LayerNorm operands')
        captured.append(tuple(t.detach().cpu().contiguous().numpy().copy()
                              for t in (args[0], module.weight, module.bias, output)))
    with torch.inference_mode():
        mel, mel_length = model.preprocessor(input_signal=signal, length=lengths)
        baseline, baseline_length = model.encoder(audio_signal=mel, length=mel_length)
        hook = norm.register_forward_hook(observe)
        try:
            observed, observed_length = model.encoder(audio_signal=mel, length=mel_length)
        finally:
            hook.remove()
    control = tensor_metrics(baseline.detach().cpu().numpy(), observed.detach().cpu().numpy())
    if len(captured) != 1 or not control['bitwise_equal'] or not torch.equal(baseline_length, observed_length):
        raise RuntimeError('LayerNorm observation changed NeMo encoder output or coverage')
    return captured[0], control


def summary(results, backend, complete):
    complete = complete and len(results) == 14
    cpu_bits = complete and all(r['cpu_vs_nemo']['bitwise_equal'] for r in results)
    cuda_bits = complete and backend == 'cuda' and all(r['cuda_vs_nemo']['bitwise_equal'] for r in results)
    passed = cpu_bits and (backend == 'cpu' or cuda_bits)
    return {'scope': SCOPE, 'expected_clips': 14, 'completed_clips': len(results),
            'backend_requested': backend, 'execution_status': 'COMPLETE' if complete else 'INCOMPLETE',
            'cpu_bitwise_passed': sum(r['cpu_vs_nemo']['bitwise_equal'] for r in results),
            'cuda_bitwise_passed': sum(r.get('cuda_vs_nemo', {}).get('bitwise_equal', False) for r in results),
            'qualification': 'PASS_BITWISE_14_COMMON_INPUTS' if passed else 'BITWISE_PARITY_NOT_ESTABLISHED',
            'gpu_execution': 'CUDA_KERNEL' if backend == 'cuda' and results else 'NOT_RUN',
            'full_stt_qualification': 'NOT_EVALUATED', 'model_operator_replaced': False,
            'microphone': 'NOT_RUN', 'results': results}


def run(corpus, checkpoint, backend, output, keep_raw=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results, errors, gpu = [], [], None
    metadata = {}
    try:
        rows = selected_corpus(json.loads(corpus.read_text()))
        pinned = json.loads((ROOT/'ci/r7_verified_native_sources.json').read_text())['checkpoint_sha256']
        if sha256(checkpoint) != pinned:
            raise ValueError('Original TranscribePsy checkpoint identity differs from the qualified R7 source')
        cpu = CpuNorm(output/'build-cpu')
        metadata['cpu'] = cpu.metadata
        metadata['source_sha256'] = {name: sha((ROOT/name).read_bytes()) for name in
            ('hdb2c/c3_r12_layernorm.py', 'hdb2c/layernorm_operator.py',
             'ci/r12_layernorm.sh', 'ci/local_layernorm.py')}
        # This must fail if CUDA is unavailable, before restoring the reference.
        if backend == 'cuda':
            gpu = CudaNorm()
            metadata['cuda'] = gpu.metadata
        import torch
        import importlib.metadata
        if torch.__version__ != '2.12.1+cpu' or torch.version.cuda is not None:
            raise ValueError('Reference requires torch==2.12.1+cpu; no GPU PyTorch is used')
        if importlib.metadata.version('nemo_toolkit') != '2.7.3':
            raise ValueError('Reference requires the qualified nemo_toolkit==2.7.3')
        threads = int(os.environ.get('HDB2C_REFERENCE_THREADS', '4'))
        if not 1 <= threads <= 64:
            raise ValueError('HDB2C_REFERENCE_THREADS must be in 1..64')
        torch.set_num_threads(threads)
        model, _ = load_nemo(checkpoint)
        evaluation = ensure_probe_eval(model)
        if len(model.encoder.layers) != 24:
            raise ValueError('Expected the actual 24-block TranscribePsy encoder')
        metadata['reference'] = {'torch': torch.__version__, 'nemo_toolkit': '2.7.3',
                                 'checkpoint_sha256': pinned, 'threads': torch.get_num_threads(),
                                 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                                 'eval_state': evaluation}
        for index, row in enumerate(rows):
            try:
                (x, gamma, beta, expected), control = capture(model, torch, Path(row['wav']))
                candidate = cpu(x, gamma, beta, 1e-5)
                result = {'clip': {key: row[key] for key in ('id', 'split', 'emotion', 'pcm_sha256')},
                          'shape': list(x.shape), 'epsilon': 1e-5, 'lanes': 8,
                          'input_sha256': sha(x.tobytes()), 'gamma_sha256': sha(gamma.tobytes()),
                          'beta_sha256': sha(beta.tobytes()), 'reference_output_sha256': sha(expected.tobytes()),
                          'instrumentation_control': control,
                          'cpu_vs_nemo': tensor_metrics(expected, candidate)}
                if gpu is not None:
                    cuda = gpu(x, gamma, beta, 1e-5)
                    result.update({'cuda_vs_nemo': tensor_metrics(expected, cuda),
                                   'cuda_vs_cpu': tensor_metrics(candidate, cuda),
                                   'cuda_common_input_echo_bitwise': True})
                if keep_raw:
                    values = {'input': x, 'gamma': gamma, 'beta': beta, 'expected': expected, 'cpu': candidate}
                    if gpu is not None:
                        values['cuda'] = cuda
                    np.savez(output/f'clip-{index:02}-LOCAL-ONLY.npz', **values)
                results.append(result)
                print(f'R12_CLIP={index+1}/14 CPU_BITWISE={result["cpu_vs_nemo"]["bitwise_equal"]}'
                      +(f' CUDA_BITWISE={result["cuda_vs_nemo"]["bitwise_equal"]}' if gpu is not None else ''), flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                errors.append({'index': index, 'type': type(exc).__name__})
            finally:
                write_json(output/'report.json', {**summary(results, backend, False),
                                                  'metadata': metadata, 'errors': errors})
        complete = len(results) == 14 and not errors
        report = {**summary(results, backend, complete), 'metadata': metadata, 'errors': errors}
        write_json(output/'report.json', report)
        print('R12_LAYERNORM='+report['execution_status']+' '+report['qualification'], flush=True)
        print('R12_REPORT='+str(output/'report.json'), flush=True)
        return 1 if errors else 0 if report['qualification'] == 'PASS_BITWISE_14_COMMON_INPUTS' else 2
    except Exception as exc:
        write_json(output/'report.json', {**summary(results, backend, False), 'metadata': metadata,
                                          'errors': [*errors, {'type': type(exc).__name__}]})
        raise
    finally:
        if gpu is not None:
            gpu.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--nemo', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keep-raw', action='store_true')
    args = parser.parse_args()
    sys.exit(run(args.corpus, args.nemo, args.backend, args.output, args.keep_raw))
