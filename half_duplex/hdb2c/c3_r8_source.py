"""Observation-only source transforms, pinned to the already tested R7 bytes."""
from pathlib import Path

MARKER = 'HDB2C3_R8_OBSERVATION_ONLY_V1'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('R8 pinned source anchor changed: '+old[:80])
    return source.replace(old, new, 1)


def frontend(source, replay):
    if MARKER in source:
        raise ValueError('R8 frontend already instrumented')
    source = once(source, '    let nfft=p.n_fft;let hop=p.hop_length();let wlen=p.win_length();',
        '    // '+MARKER+'\n'
        '    let r8_dir=std::env::var_os("HDB2C8_FRONTEND_TRACE_DIR").map(std::path::PathBuf::from);\n'
        '    let nfft=p.n_fft;let hop=p.hop_length();let wlen=p.win_length();')
    source = once(source, '    if !p.exact_pad {\n',
        '    if let Some(ref dir)=r8_dir {crate::parakeet::hdb2c_r8::save_values(dir,"preemphasis",&wave,&[1,wave.len()])?;}\n'
        '    if !p.exact_pad {\n')
    source = once(source, '    let left=(nfft-wlen)/2;',
        '    if let Some(ref dir)=r8_dir {\n'
        '        crate::parakeet::hdb2c_r8::save_values(dir,"window",&window,&[wlen])?;\n'
        '        crate::parakeet::hdb2c_r8::save_values(dir,"filterbank",&weights,&[p.features,nfft/2+1])?;\n'
        '    }\n    let left=(nfft-wlen)/2;')
    source = once(source, '    for t in 0..raw_frames {',
        '    let (mut r8_frames,mut r8_fft,mut r8_mag,mut r8_power)=(Vec::new(),Vec::new(),Vec::new(),Vec::new());\n'
        '    let mut r8_projected=if r8_dir.is_some(){vec![0f32;channels*raw_frames]}else{Vec::new()};\n'
        '    for t in 0..raw_frames {')
    source = once(source, '        let fft=fft(&frame);',
        '        if r8_dir.is_some(){r8_frames.extend_from_slice(&frame);}\n'
        '        let fft=fft(&frame);\n'
        '        if r8_dir.is_some(){r8_fft.extend_from_slice(&fft[..nbins*2]);}')
    source = once(source, '            let mag=(re*re+im*im).sqrt();',
        '            let mag=(re*re+im*im).sqrt();\n'
        '            if r8_dir.is_some(){r8_mag.push(mag);}')
    source = once(source, '\n        for c in 0..channels {',
        '\n        if r8_dir.is_some(){r8_power.extend_from_slice(&powers);}\n        for c in 0..channels {')
    source = once(source, '            mel[c*raw_frames+t]=if p.log.unwrap_or(true) {',
        '            if r8_dir.is_some(){r8_projected[c*raw_frames+t]=value;}\n'
        '            mel[c*raw_frames+t]=if p.log.unwrap_or(true) {')
    source = once(source, '    normalise(&mut mel,channels,raw_frames,valid,&p.normalize)?;',
        '    if let Some(ref dir)=r8_dir {\n'
        '        use crate::parakeet::hdb2c_r8::save_values;\n'
        '        save_values(dir,"frames",&r8_frames,&[raw_frames,nfft])?;\n'
        '        save_values(dir,"fft_complex",&r8_fft,&[raw_frames,nbins,2])?;\n'
        '        save_values(dir,"magnitude",&r8_mag,&[raw_frames,nbins])?;\n'
        '        save_values(dir,"power",&r8_power,&[raw_frames,nbins])?;\n'
        '        save_values(dir,"mel_projection",&r8_projected,&[channels,raw_frames])?;\n'
        '        save_values(dir,"log_mel",&mel,&[channels,raw_frames])?;\n'
        '    }\n'
        '    normalise(&mut mel,channels,raw_frames,valid,&p.normalize)?;\n'
        '    if let Some(ref dir)=r8_dir {crate::parakeet::hdb2c_r8::save_values(dir,"normalized",&mel,&[channels,raw_frames])?;}')
    source = once(source, '    Tensor::from_vec(output,(1,target,channels),device)?.to_dtype(DType::F32)',
        '    if let Some(ref dir)=r8_dir {crate::parakeet::hdb2c_r8::save_values(dir,"mel_output",&output,&[1,target,channels])?;}\n'
        '    Tensor::from_vec(output,(1,target,channels),device)?.to_dtype(DType::F32)')
    return source+'\n'+replay


CONFORMER_REPLAY = r'''
// HDB2C3_R8_OBSERVATION_ONLY_V1: independent common-input operator execution.
impl Conformer {
    pub fn hdb2c8_operators(&self, reference:&std::path::Path, output:&std::path::Path, device:&candle::Device)->Result<()> {
        use crate::parakeet::hdb2c_r8::{read,save};
        let PreEncode::Subsampling(pre)=&self.pre_encode else {return Err(candle::Error::Msg("R8 requires DW subsampling".into()));};
        for (name,index) in [("conv_00",0usize),("conv_02",1),("conv_03",2),("conv_05",3),("conv_06",4)] {
            let folder=reference.join(name);let out=output.join(name);
            let x=read(&folder,"input",device)?;
            let y=pre.conv[index].forward(&x)?;
            save(&out,"input",&x)?;save(&out,"output",&y)?;
            save(&out,"weight",pre.conv[index].weight())?;
            if let Some(b)=pre.conv[index].bias(){save(&out,"bias",b)?;}
        }
        for name in ["relu_00","relu_03","relu_06"] {
            let x=read(&reference.join(name),"input",device)?;
            save(&output.join(name),"input",&x)?;
            save(&output.join(name),"output",&x.relu()?)?;
        }
        let x=read(&reference.join("out"),"input",device)?;
        save(&output.join("out"),"input",&x)?;
        save(&output.join("out"),"output",&pre.out.forward(&x)?)?;
        save(&output.join("out"),"weight",pre.out.weight())?;
        if let Some(b)=pre.out.bias(){save(&output.join("out"),"bias",b)?;}
        Ok(())
    }
}
'''


def probe(source, *, operators):
    source = once(source, '    #[arg(long)] tdt_cases: Option<PathBuf>,',
        '    #[arg(long)] tdt_cases: Option<PathBuf>,\n'
        '    #[arg(long,default_value="cpu")] device: String,\n'+
        ('    #[arg(long)] operator_cases: Option<PathBuf>,\n' if operators else ''))
    source = once(source, 'fn read(dir:&Path,name:&str)->Result<Tensor> {',
        'fn read(dir:&Path,name:&str,dev:&Device)->Result<Tensor> {')
    source = once(source, 'shape.as_slice(),&Device::Cpu)', 'shape.as_slice(),dev)')
    source = once(source, '    let dev=Device::Cpu;',
        '    let dev=match a.device.as_str(){"cpu"=>Device::Cpu,"cuda-cudnn"=>Device::new_cuda(0)?,_=>bail!("unsupported R8 device")};')
    for old in ['read(&folder,"input_h")','read(&folder,"input_c")','read(&folder,"enc")','read(&folder,"g")','read(&folder,"mel")']:
        source = once(source,old,old[:-1]+',&dev)')
    source = once(source, '"device":"CPU",', '"device":format!("{:?}",dev),')
    source = once(source, 'println!("HDB2C3_R7_NATIVE={scope} text={text}");',
        'println!("HDB2C3_R8_NATIVE={scope}");')
    if operators:
        source = once(source, '    if let Some(cases)=a.tdt_cases {', '''    if let Some(cases)=a.operator_cases {
        ensure!(a.tdt_cases.is_none() && a.shared_mel.is_none() && a.mel_valid.is_none(),"R8 mixed operator/pipeline scopes");
        m.encoder.hdb2c8_operators(&cases.join("subsampling"),&a.output.join("subsampling"),&dev)?;
        parakeet_candle::parakeet::audio::hdb2c8_frontend_operators(&cases.join("frontend"),&a.output.join("frontend"),&m.preprocessor_config)?;
        fs::write(a.output.join("scope.json"),br#"{"scope":"COMMON_INPUT_FRONTEND_AND_SUBSAMPLING_ONLY"}"#)?;
        return Ok(());
    }
    if let Some(cases)=a.tdt_cases {''')
    return source


def outputs(reference: Path, root: Path):
    audio=(reference/'src/parakeet/audio.rs').read_text()
    audio += '''\npub fn hdb2c8_frontend_operators(reference:&std::path::Path,output:&std::path::Path,p:&PreprocessArgs)->Result<()> {
    hdb2c_nemo::r8_replay(reference,output,p)
}\n'''
    return {
        'src/parakeet/audio.rs':audio,
        'src/parakeet/mod.rs':(reference/'src/parakeet/mod.rs').read_text()+'\npub mod hdb2c_r8;\n',
        'src/parakeet/hdb2c_r8.rs':(root/'rust/hdb2c_r8_io.rs').read_text(),
        'src/parakeet/hdb2c_nemo.rs':frontend((reference/'src/parakeet/hdb2c_nemo.rs').read_text(),(root/'rust/hdb2c_r8_frontend.rs').read_text()),
        'src/parakeet/conformer.rs':(reference/'src/parakeet/conformer.rs').read_text()+CONFORMER_REPLAY,
        'src/bin/hdb2c_r8_probe.rs':probe((root/'rust/hdb2c_r7_probe.rs').read_text(),operators=True),
    }
