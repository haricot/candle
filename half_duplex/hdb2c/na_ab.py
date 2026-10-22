"""HDB2C-NA isolated parity A/B runner. Changes ONE branch in a COPIED Rust runtime.

Never touches the live parakeet-candle source, binary, or weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .audio import report
from .compare import compare_probes
from .cli import settings, MODEL_ID

IDENTITY = 'hdb2c-na-v0.6.4'
OLD = '    } else {\n        let mean = mel.iter().sum::<f32>() / mel.len().max(1) as f32;'
NEW = '    } else if args.normalize == "all_features" {\n        let mean = mel.iter().sum::<f32>() / mel.len().max(1) as f32;'
PROBE_REL = Path('src/bin/hdb2c_probe.rs')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def correct_na_code(source: str) -> str:
    """Only repair the else branch; never silently match unexpected source code."""
    if source.count(OLD) != 1:
        raise ValueError('Refusing to modify unknown get_logmel source (exact normalization branch not found)')
    if 'if args.normalize == "per_feature"' not in source:
        raise ValueError('No NeMo per-feature branch')
    result = source.replace(OLD, NEW, 1)
    assert result != source
    return result


def prepare_copy(runtime: Path, candidate: Path, probe_source: Path, model_dir: Path):
    source=runtime/'src/parakeet/audio.rs'
    main=runtime/'src/main.rs'
    cfg=model_dir/'config.json'
    if not (source.is_file() and main.is_file() and cfg.is_file()):
        raise FileNotFoundError('Missing runtime source, local-model loader or model config')
    config=json.loads(cfg.read_text(encoding='utf-8'))
    norm=config.get('preprocessor',{}).get('normalize')
    if norm != 'NA':
        raise ValueError(f'Checkpoint normalize must be NA, got {norm!r}; this experimental fix is specific')
    if 'HDB2A_CANDLE_LOCAL_MODEL' not in main.read_text():
        raise ValueError('Not the HDB2A local-model runtime; refusing unknown main.rs')
    before=source.read_text(encoding='utf-8')
    after=correct_na_code(before)
    original_sha=sha_bytes(before.encode())
    after_sha=sha_bytes(after.encode())
    manifest={'fix':IDENTITY,'source':str(runtime.resolve()),'original_audio_sha256':original_sha,
              'candidate_audio_sha256':after_sha, 'allowed_change':
              'else => else if args.normalize == "all_features"; leave NA unnormalized',
              'model_normalize':'NA'}
    if candidate.exists():
        m=candidate/'HDB2C-NA-MANIFEST.json'
        if not m.is_file() or json.loads(m.read_text()) != manifest:
            raise RuntimeError(f'Candidate dir exists but does not match manifest: {candidate}')
        if sha_bytes((candidate/'src/parakeet/audio.rs').read_bytes()) != after_sha:
            raise RuntimeError('Candidate source modified since preparation; refusing to proceed')
        if (candidate/PROBE_REL).read_bytes()!=probe_source.read_bytes():
            raise RuntimeError('Candidate probe unexpectedly modified; refusing to proceed')
        print('HDB2C_NA_CANDIDATE=VERIFIED_EXISTING',flush=True)
        return manifest
    candidate.parent.mkdir(parents=True,exist_ok=True)
    tmp=candidate.with_name(candidate.name+'.preparing')
    if tmp.exists():
        raise RuntimeError(f'Stale preparation directory {tmp} (refusing to overwrite)')
    try:
        shutil.copytree(runtime,tmp,symlinks=True,ignore=shutil.ignore_patterns(
            '.git','target','*.pyc','__pycache__','.venv*','.cache'))
        (tmp/'src/parakeet/audio.rs').write_text(after,encoding='utf-8')
        (tmp/PROBE_REL).parent.mkdir(parents=True,exist_ok=True)
        (tmp/PROBE_REL).write_bytes(probe_source.read_bytes())
        (tmp/'HDB2C-NA-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        tmp.rename(candidate)
    except BaseException:
        if tmp.exists():shutil.rmtree(tmp)
        raise
    print('HDB2C_NA_CANDIDATE=CREATED_ISOLATED_COPY',flush=True)
    return manifest


def run_cmd(cmd,stdout_file,stderr_file,env=None):
    with stdout_file.open('w') as so,stderr_file.open('w') as se:
        p=subprocess.run([str(x) for x in cmd],stdout=so,stderr=se,env=env)
    if p.returncode:raise RuntimeError(f'Command failed rc={p.returncode}; see {stderr_file}: {cmd}')
    return stdout_file.read_text()


def execute(a):
    s=settings()
    base=a.report_dir.resolve()
    if not (base/'nemo_report.json').is_file() or not (base/'nemo_mel.npz').is_file() or not (base/'input-16k.wav').is_file():
        raise ValueError('Pass the REAL C1 --probes directory with nemo_report.json, nemo_mel.npz and input-16k.wav')
    nr=json.loads((base/'nemo_report.json').read_text())
    if nr.get('probe_status')!='REAL_NEMO_MEL_AND_ENCODER':
        raise ValueError('NeMo probes were not qualified in this C1 report')
    candidate=a.candidate or s['runtime'].with_name(s['runtime'].name+'-hdb2c-na-v0.6.4')
    manifest=prepare_copy(s['runtime'],candidate,Path(__file__).parents[1]/'rust/hdb2c_probe.rs',s['model'])
    output=base/('candidate_na_'+a.variant.replace('-','_'))
    output.mkdir(exist_ok=True)
    probe_exe=candidate/'target'/f'hdb2c-na-{a.variant}'/'release'/'hdb2c_probe'
    asr_exe=probe_exe.with_name('parakeet-candle')
    if not a.skip_build:
        # Only copied runtime is modified; its own Cargo.lock can be updated.
        # Existing system CUDA artefacts and production binary remain untouched.
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(candidate/'target'/f'hdb2c-na-{a.variant}')
        env['CUDA_COMPUTE_CAP']='61'
        manifest_path=candidate/'Cargo.toml'
        cargo=['cargo','build','--release','--manifest-path',str(manifest_path),
               '--bin','parakeet-candle','--bin','hdb2c_probe']
        if a.variant=='cuda-cudnn':cargo+=['--features','cuda cudnn']
        run_cmd(cargo,output/'build.stdout',output/'build.stderr',env)
    if not (asr_exe.is_file() and probe_exe.is_file()):
        raise FileNotFoundError('Candidate release binaries missing; build first')
    args=[asr_exe,'--model-id',MODEL_ID,'--model-dir',s['model'],'--input',base/'input-16k.wav','--format','json']
    if a.variant=='cpu':args.append('--cpu')
    txt=run_cmd(args,output/'asr.stdout',output/'asr.stderr')
    response=json.loads(txt)
    if response.get('model_id')!=MODEL_ID:raise ValueError('Candidate model_id mismatch')
    if not response.get('text'):raise ValueError('Candidate ASR text empty')
    with (output/'probe.stdout').open('w') as so,(output/'probe.stderr').open('w') as se:
        args=[probe_exe,'--model-dir',s['model'],'--input',base/'input-16k.wav','--output',output]
        if a.variant=='cpu':args.append('--cpu')
        p=subprocess.run([str(x) for x in args],stdout=so,stderr=se)
    if p.returncode: raise RuntimeError(f'Candidate numeric probe failed; see {output}/probe.stderr')
    measures=compare_probes(base,output,base/f'c2c3_na_{a.variant}.json')
    baseline=json.loads((base/'candle_report.json').read_text())
    c1={'stage':'HDB2C_NA_SINGLE_CHANGE_AB','fix':IDENTITY,'candidate_model_id':response['model_id'],
        'variant':a.variant,'nemo_text':nr.get('text'),'baseline_candle_text':baseline.get('text'),
        'candidate_candle_text':response.get('text'),'candidate_device':response.get('device'),
        'source_manifest':manifest,'numeric_mel_status':measures.get('mel',{}).get('status'),
        'numeric_encoder_status':measures.get('encoder',{}).get('status'),
        'decision':'CANDIDATE_UNQUALIFIED_PENDING_NUMERIC_AND_TEXT_REVIEW'}
    report(base/f'na_ab_{a.variant}.json',c1)
    print('HDB2C_NA_BASELINE='+str(baseline['text']))
    print('HDB2C_NA_CANDIDATE='+str(response['text']))
    print('HDB2C_NA_NEMO='+str(nr.get('text')))
    print('HDB2C_NA_NUMERIC='+str(measures['status']))
    print('HDB2C_NA_REPORT='+str(base/f'na_ab_{a.variant}.json'))
    return 0


def main():
    p=argparse.ArgumentParser(description='Isolated Rust get_logmel NA correction A/B')
    p.add_argument('report_dir',type=Path)
    p.add_argument('--variant',choices=['cpu','cuda-cudnn'],default='cpu')
    p.add_argument('--candidate',type=Path)
    p.add_argument('--skip-build',action='store_true')
    a=p.parse_args()
    return execute(a)

if __name__=='__main__':sys.exit(main())
