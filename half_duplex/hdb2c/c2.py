"""HDB2-C2 build + A/B with an isolated copy of the existing Rust runtime.

No dynamic patch files distributed: the complete Rust frontend source is in the ZIP.
Installation generates a complete isolated candidate checkout from the pinned local
runtime, with an auditable manifest. It never writes into the live runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .cli import settings, ROOT, MODEL_ID
from .audio import report, describe
from .compare import compare_probes

VERSION='v0.6.6'
SIGNATURE='hdb2c-c2-nemo-frontend-v0.6.6'
SOURCE_AUDIO='src/parakeet/audio.rs'
SOURCE_MODEL='src/parakeet/model.rs'
RUST_MODULE='src/parakeet/hdb2c_nemo.rs'
RUST_PROBE='src/bin/hdb2c_probe.rs'
OPTIONAL_CONFIG_FIELDS='''    #[serde(default)] pub exact_pad: bool,
    #[serde(default)] pub mel_norm: Option<String>,
    #[serde(default)] pub lowfreq: Option<f64>,
    #[serde(default)] pub highfreq: Option<f64>,
    #[serde(default)] pub log_zero_guard_value: Option<serde_json::Value>,
    #[serde(default)] pub log_zero_guard_type: Option<String>,
    #[serde(default)] pub frame_splicing: Option<usize>,
    #[serde(default)] pub log: Option<bool>,
'''


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def transform_audio(source:str)->str:
    anchor='    pub mag_power: f64,\n}'
    if source.count(anchor)!=1:raise ValueError('Unexpected PreprocessArgs struct; refused')
    source=source.replace(anchor,'    pub mag_power: f64,\n'+OPTIONAL_CONFIG_FIELDS+'}',1)
    first='pub fn get_logmel(samples: &[f32], args: &PreprocessArgs, device: &Device) -> Result<Tensor> {'
    suffix='\n#[cfg(test)]\nmod tests {'
    if source.count(first)!=1 or source.count(suffix)!=1:raise ValueError('Unknown get_logmel boundary; refused')
    begin=source.index(first)
    end=source.index(suffix,begin)
    sub='''#[path = "hdb2c_nemo.rs"]
mod hdb2c_nemo;

pub fn hdb2c_valid_frames(samples: usize, args: &PreprocessArgs) -> Result<usize> {
    hdb2c_nemo::valid_frames(samples,args)
}

pub fn get_logmel(samples: &[f32], args: &PreprocessArgs, device: &Device) -> Result<Tensor> {
    hdb2c_nemo::get_logmel(samples,args,device)
}
'''
    source=source[:begin]+sub+source[end:]
    # Upstream audio.rs unit tests construct PreprocessArgs via struct literals.
    # Keep those tests compiling after extending the serde config contract.
    lit='            mag_power: 2.0,\n        }'
    if source.count(lit)!=2:
        raise ValueError('Upstream PreprocessArgs unit constructors changed; fail closed')
    default_fields=('            exact_pad: false,\n'
        '            mel_norm: None,\n'
        '            lowfreq: None,\n'
        '            highfreq: None,\n'
        '            log_zero_guard_value: None,\n'
        '            log_zero_guard_type: None,\n'
        '            frame_splicing: None,\n'
        '            log: None,\n')
    source=source.replace(lit,'            mag_power: 2.0,\n'+default_fields+'        }')
    return source

def complete_hdb2a_unit_test_constructors(source: str) -> str:
    """Complete only the upstream model.rs `time_ratio_positive` test.

    The HDB2A converter adds two fields to ConformerArgs and C2 adds eight
    fields to PreprocessArgs. Rust cfg(test) compiles EVERY library unit test
    when cargo test --lib is invoked, even with a specific test name filter.
    Refuse unfamiliar sources instead of silently changing a real model config.
    """
    section_start='    fn time_ratio_positive() {\n'
    next_test='\n    #[test]\n'
    if source.count(section_start)!=1:
        raise ValueError('Expected exactly one pinned upstream time_ratio_positive test')
    pos=source.index(section_start)
    end=source.find(next_test,pos+len(section_start))
    if end < 0:
        raise ValueError('Unable to bound upstream time_ratio_positive test')
    body=source[pos:end]
    pre_end='            mag_power: 2.0,\n        };'
    enc_end='            att_context_size: None,\n        };'
    if body.count(pre_end)!=1 or body.count(enc_end)!=1:
        raise ValueError('Pinned HDB2A unit constructors differ; refusing unsafe edit')
    # Explicit values are important: NeMo C2 fields aren't generic defaults.
    pre=('            mag_power: 2.0,\n'
         '            exact_pad: false,\n'
         '            mel_norm: None,\n'
         '            lowfreq: None,\n'
         '            highfreq: None,\n'
         '            log_zero_guard_value: None,\n'
         '            log_zero_guard_type: None,\n'
         '            frame_splicing: None,\n'
         '            log: None,\n'
         '        };')
    enc=('            att_context_size: None,\n'
         '            conv_norm_type: "batch_norm".to_owned(),\n'
         '            conv_context_size: None,\n'
         '        };')
    body=body.replace(pre_end,pre,1).replace(enc_end,enc,1)
    return source[:pos]+body+source[end:]


def transform_model(source:str)->str:
    original='''        if chunk_duration.is_none() {
            let mel = get_logmel(&audio, &pre, &device)?;
            return Ok(self.generate(&mel, config)?[0].clone());
        }'''
    if source.count(original)!=1:raise ValueError('Unexpected ParakeetModel::transcribe block; refused')
    new='''        if chunk_duration.is_none() {
            let mel = get_logmel(&audio, &pre, &device)?;
            // HDB2-C2: the padded Mel time dimension is NOT the valid length.
            // NeMo explicitly propagates the number of valid frames to the encoder.
            if let Self::Tdt(m) = self {
                let valid = crate::parakeet::audio::hdb2c_valid_frames(audio.len(), &pre)?;
                let valid_t = Tensor::from_vec(vec![valid as i64], (1,), &device)?;
                let (encoded, encoded_lengths) = m.encoder.forward(&mel, Some(&valid_t))?;
                let (tokens, _) = m.decode(&encoded, Some(&encoded_lengths), None, None, config)?;
                let tokens = tokens.into_iter().next().ok_or_else(|| {
                    candle::Error::Msg("HDB2C2 encoder returned no hypothesis".into())
                })?;
                return Ok(sentences_to_result(&tokens_to_sentences(&tokens, &config.sentence)));
            }
            return Ok(self.generate(&mel, config)?[0].clone());
        }'''
    # Complete the test-only struct literals so cargo test --lib compiles all tests.
    source=complete_hdb2a_unit_test_constructors(source)
    return source.replace(original,new,1)

def transform_probe(source:str)->str:
    pat='let (features,lengths)=match &mut model {'
    if source.count(pat)!=1:raise ValueError('Expected v0.6.4 mutable probe source')
    target='''let valid=parakeet_candle::parakeet::audio::hdb2c_valid_frames(audio.len(),&pre)?;
    let valid_t=Tensor::from_vec(vec![valid as i64],(1,),&dev)?;
    let (features,lengths)=match &mut model {'''
    source=source.replace(pat,target,1)
    original='m.encoder.forward(&mel,None)?'
    if source.count(original)!=1:raise ValueError('Expected TDT encoder call')
    source=source.replace(original,'m.encoder.forward(&mel,Some(&valid_t))?',1)
    line='"source_pcm_frames":audio.len(),'
    if source.count(line)!=1:raise ValueError('Probe metadata unknown')
    source=source.replace(line,line+'"nemo_valid_mel_frames":valid,',1)
    return source


def prepare(runtime:Path,candidate:Path,root:Path,model_dir:Path):
    runtime=runtime.resolve();candidate=candidate.resolve();root=root.resolve()
    if candidate==runtime or runtime in candidate.parents or candidate in runtime.parents:
        raise ValueError('Candidate cannot be runtime nor be nested in it')
    cfg=json.loads((model_dir/'config.json').read_text())['preprocessor']
    if cfg.get('normalize')!='NA' or cfg.get('n_fft')!=512 or cfg.get('features')!=128:
        raise ValueError('C2 validated ONLY for exact 128-Mel TranscribePsy checkpoint (NA/512)')
    if not (runtime/SOURCE_AUDIO).is_file() or not (runtime/SOURCE_MODEL).is_file():
        raise FileNotFoundError('Existing runtime must contain complete Rust sources')
    if 'HDB2A_CANDLE_LOCAL_MODEL' not in (runtime/'src/main.rs').read_text():
        raise ValueError('Existing runtime is not bootstrapped HDB2A local loader')
    audio_src=(runtime/SOURCE_AUDIO).read_text();model_src=(runtime/SOURCE_MODEL).read_text()
    new_audio=transform_audio(audio_src)
    new_model=transform_model(model_src)
    new_probe=transform_probe((root/'rust/hdb2c_probe.rs').read_text())
    new_frontend=(root/'rust/hdb2c_nemo.rs').read_text()
    source_hashes={p:sha(runtime/p) for p in (SOURCE_AUDIO,SOURCE_MODEL)}
    output_hashes={SOURCE_AUDIO:hashlib.sha256(new_audio.encode()).hexdigest(),
                   SOURCE_MODEL:hashlib.sha256(new_model.encode()).hexdigest(),
                   RUST_PROBE:hashlib.sha256(new_probe.encode()).hexdigest(),
                   RUST_MODULE:hashlib.sha256(new_frontend.encode()).hexdigest()}
    manifest={'signature':SIGNATURE,'source_root':str(runtime),'source_hashes':source_hashes,
              'generated_files':output_hashes, 'model_config':cfg,
              'scope':'isolated CPU A/B only; no production source or weight mutation'}
    if candidate.exists():
        existing=candidate/'HDB2C2-MANIFEST.json'
        if not existing.is_file() or json.loads(existing.read_text())!=manifest:
            raise RuntimeError('Candidate exists with different manifest. Refusing overwrite.')
        for name,expected in output_hashes.items():
            if sha(candidate/name)!=expected:raise RuntimeError(f'Candidate modified: {name}')
        print('HDB2C2_CANDIDATE=REUSED_VERIFIED_COPY',flush=True)
        return manifest
    temp=candidate.with_name(candidate.name+'.creating')
    if temp.exists():raise RuntimeError('Stale candidate preparation directory: '+str(temp))
    def ignore(_,names):return [x for x in names if x in {'.git','target','__pycache__','.venv','.venv-hdb2c-py313'}]
    try:
        shutil.copytree(runtime,temp,ignore=ignore)
        for name,data in ((SOURCE_AUDIO,new_audio),(SOURCE_MODEL,new_model),(RUST_PROBE,new_probe),(RUST_MODULE,new_frontend)):
            loc=temp/name;loc.parent.mkdir(parents=True,exist_ok=True);loc.write_text(data)
        (temp/'HDB2C2-MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        temp.rename(candidate)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print('HDB2C2_CANDIDATE=CREATED_ISOLATED_COPY',flush=True)
    return manifest


def execute(args):
    s=settings()
    root=ROOT
    candidate=Path(args.candidate).resolve() if args.candidate else s['runtime'].with_name(s['runtime'].name+'-hdb2c-c2-v0.6.6')
    manifest=prepare(s['runtime'],candidate,root,s['model'])
    if args.mode=='prepare':return 0
    variant=args.variant
    build_target=candidate/'target'/f'hdb2c2-{variant}'
    asr=build_target/'release/parakeet-candle'
    probe=build_target/'release/hdb2c_probe'
    if not args.skip_build:
        cmd=[sys.executable,str(root/'scripts/hdb2a_cargo_lock.py'),'--manifest-path',str(candidate/'Cargo.toml')]
        subprocess.run(cmd,check=True)
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(build_target);env['CUDA_COMPUTE_CAP']='61'
        cargo=['cargo','build','--release','--manifest-path',str(candidate/'Cargo.toml'),'--bin','parakeet-candle','--bin','hdb2c_probe']
        if variant=='cuda-cudnn':cargo+=['--features','cuda cudnn']
        print('HDB2C2_BUILD='+ ' '.join(cargo),flush=True)
        subprocess.run(cargo,check=True,env=env)
        # Compile AND execute the Rust frontend's CPU/feature-specific unit tests.
        # This catches struct/API drift before attempting real NeMo/Candle A/B.
        cargo_tests=['cargo','test','--release','--manifest-path',str(candidate/'Cargo.toml'),
                     '--lib','hdb2c_nemo::tests']
        if variant=='cuda-cudnn':cargo_tests+=['--features','cuda cudnn']
        print('HDB2C2_RUST_TEST='+ ' '.join(cargo_tests),flush=True)
        subprocess.run(cargo_tests,check=True,env=env)
    if not asr.is_file() or not probe.is_file():raise FileNotFoundError('Compilation not finished or binaries missing')
    print('HDB2C2_BUILD=COMPLETED_BINARIES_PRESENT',flush=True)
    if args.mode=='build':return 0
    if not args.report_dir:raise ValueError('--report-dir required for compare')
    reportdir=args.report_dir.resolve()
    if not (reportdir/'nemo_report.json').is_file() or not (reportdir/'nemo_mel.npz').is_file() or not (reportdir/'input-16k.wav').is_file():
        raise FileNotFoundError('Use the genuine C1 --probes report directory')
    nemo=json.loads((reportdir/'nemo_report.json').read_text())
    original=json.loads((reportdir/'candle_report.json').read_text())
    input_pcm=describe(reportdir/'input-16k.wav')['pcm_sha256']
    if not (nemo.get('pcm_sha256')==original.get('pcm_sha256')==input_pcm):
        raise ValueError('C1 source WAV PCM differs from Nemo/Candle reports; refuse A/B')
    source_meta=json.loads((s['model']/'conversion.json').read_text())
    if source_meta.get('source_nemo_sha256')!=nemo.get('checkpoint_sha256'):
        raise ValueError('Candle checkpoint source no longer agrees with NeMo; refuse A/B')
    output=reportdir/f'candidate_c2_{variant.replace("-","_")}'
    output.mkdir(exist_ok=True)
    def run(command,name):
        p=subprocess.run([str(x) for x in command],capture_output=True,text=True)
        (output/f'{name}.stdout').write_text(p.stdout)
        (output/f'{name}.stderr').write_text(p.stderr)
        if p.returncode:raise RuntimeError(f'{name} failed with {p.returncode}; see {output}/{name}.stderr')
        return p.stdout
    wav=reportdir/'input-16k.wav'
    asr_args=[asr,'--model-id',MODEL_ID,'--model-dir',s['model'],'--input',wav,'--format','json']
    if variant=='cpu':asr_args.append('--cpu')
    output_obj=json.loads(run(asr_args,'asr'))
    if output_obj.get('model_id')!=MODEL_ID or not output_obj.get('text'):
        raise RuntimeError('Incorrect/empty C2 ASR result')
    probe_args=[probe,'--model-dir',s['model'],'--input',wav,'--output',output]
    if variant=='cpu':probe_args.append('--cpu')
    run(probe_args,'probe')
    comparison=compare_probes(reportdir,output,reportdir/f'c2c3_exact_{variant.replace("-","_")}.json')
    result={'stage':'HDB2C2_NEMO_FRONTEND_AB','status':'REAL_AB_OBSERVED_NUMERIC_GATE_SEPARATE',
         'nemo_text':nemo['text'],'baseline_text':original['text'],'candidate_text':output_obj['text'],
         'nemo_model_checkpoint_sha256':nemo['checkpoint_sha256'],
         'candidate_source_manifest':manifest,
         'mel_metrics':comparison['mel'],'encoder_metrics':comparison['encoder'],
         'numeric_status':comparison['status'],
         'note':'No claim of transcription parity from numerical frontend alone; TDT still unqualified'}
    outfile=reportdir/f'c2_ab_{variant.replace("-","_")}.json'
    report(outfile,result)
    print('HDB2C2_NEMO_TEXT='+nemo['text'],flush=True)
    print('HDB2C2_BASELINE_TEXT='+original['text'],flush=True)
    print('HDB2C2_CANDIDATE_TEXT='+output_obj['text'],flush=True)
    print('HDB2C2_NUMERIC='+comparison['status'],flush=True)
    print('HDB2C2_REPORT='+str(outfile),flush=True)
    return 0


def main():
    ap=argparse.ArgumentParser(description='HDB2-C2 exact NeMo frontend isolated source candidate')
    ap.add_argument('mode',choices=['prepare','build','compare'])
    ap.add_argument('--variant',choices=['cpu','cuda-cudnn'],default='cpu')
    ap.add_argument('--candidate',type=Path)
    ap.add_argument('--report-dir',type=Path)
    ap.add_argument('--skip-build',action='store_true')
    args=ap.parse_args()
    return execute(args)
if __name__=='__main__':sys.exit(main())
