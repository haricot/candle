use super::audio::PreprocessArgs;
use super::hdb2c_r10_mkl as cpu;
use candle::{Device,Error,Result,Tensor};
use candle_nn::VarBuilder;

fn fail(s:&str)->Error {Error::Msg(s.into())}
pub fn get_logmel(samples:&[f32],p:&PreprocessArgs,device:&Device,vb:&VarBuilder)->Result<Tensor>{
    p.validate()?;
    if !matches!(device,Device::Cpu) || p.normalize!="NA" || p.exact_pad || p.mag_power!=2.0
        || p.frame_splicing.unwrap_or(1)!=1 || !p.log.unwrap_or(true) || samples.is_empty()
        || samples.iter().any(|v|!v.is_finite()) {
        return Err(fail("R10 CPU profile requires checkpoint NA, CPU FP32, power 2, center padding"));
    }
    let n=p.n_fft;let hop=p.hop_length();let wlen=p.win_length();let channels=p.features;
    if n<wlen || hop==0{return Err(fail("R10 invalid STFT configuration"));}
    let vb=vb.pp("preprocessor.featurizer");
    let window=vb.get(wlen,"window")?.flatten_all()?.to_vec1::<f32>()?;
    let weights=vb.get((1,channels,n/2+1),"fb")?.flatten_all()?.to_vec1::<f32>()?;
    if window.iter().chain(weights.iter()).any(|v|!v.is_finite()){return Err(fail("R10 nonfinite checkpoint coefficient"));}
    let trace=std::env::var_os("HDB2C8_FRONTEND_TRACE_DIR").map(std::path::PathBuf::from);
    let save=|name:&str,values:&[f32],shape:&[usize]|->Result<()> {
        if let Some(ref dir)=trace {super::hdb2c_r8::save_values(dir,name,values,shape)?;} Ok(())
    };
    let mut wave=samples.to_vec();
    if let Some(v)=p.preemph {
        for i in (1..wave.len()).rev(){wave[i]=wave[i]-v as f32*wave[i-1];}
    }
    save("preemphasis",&wave,&[1,wave.len()])?;
    let pad=n/2;let mut padded=vec![0f32;pad];padded.extend_from_slice(&wave);padded.resize(padded.len()+pad,0.0);
    let frames=(padded.len()-n)/hop+1;
    let valid=super::audio::hdb2c_valid_frames(samples.len(),p)?;
    if valid<2 || frames<valid{return Err(fail("R10 invalid frame count"));}
    save("window",&window,&[wlen])?;save("filterbank",&weights,&[channels,n/2+1])?;
    let left=(n-wlen)/2;let mut padded_window=vec![0f32;n];padded_window[left..left+wlen].copy_from_slice(&window);
    let mut framed=vec![0f32;frames*n];
    for t in 0..frames {for i in 0..n {framed[t*n+i]=padded[t*hop+i]*padded_window[i];}}
    save("frames",&framed,&[frames,n])?;
    let complex=cpu::rfft(&framed,frames,n)?;let nb=n/2+1;
    save("fft_complex",&complex,&[frames,nb,2])?;
    let square:Vec<f32>=complex.chunks_exact(2).map(|v|v[0]*v[0]+v[1]*v[1]).collect();
    let magnitude=cpu::sqrt(&square)?;
    let power:Vec<f32>=magnitude.iter().map(|v|v*v).collect();
    save("magnitude",&magnitude,&[frames,nb])?;save("power",&power,&[frames,nb])?;
    let projected=cpu::mel(&weights,&power,channels,frames,nb)?;
    save("mel_projection",&projected,&[channels,frames])?;
    let guard=match p.log_zero_guard_value.as_ref(){
        None|Some(serde_json::Value::Null)=>2f32.powi(-24),
        Some(serde_json::Value::Number(n))=>n.as_f64().ok_or_else(||fail("R10 invalid log guard"))? as f32,
        _=>return Err(fail("R10 CPU profile requires numeric checkpoint log guard")),
    };
    let adjusted:Vec<f32>=match p.log_zero_guard_type.as_deref().unwrap_or("add"){
        "add"=>projected.iter().map(|v|v+guard).collect(),
        "clamp"=>projected.iter().map(|v|v.max(guard)).collect(),
        _=>return Err(fail("R10 invalid log guard type")),
    };
    let logged=cpu::ln(&adjusted)?;
    save("log_mel",&logged,&[channels,frames])?;save("normalized",&logged,&[channels,frames])?;
    let target=if p.pad_to>0{frames.max(valid).div_ceil(p.pad_to)*p.pad_to}else{frames};
    let mut output=vec![p.pad_value as f32;target*channels];
    for t in 0..valid {for c in 0..channels {output[t*channels+c]=logged[c*frames+t];}}
    save("mel_output",&output,&[1,target,channels])?;
    Tensor::from_vec(output,(1,target,channels),device)
}

// Separate common-input experiments. Never called by the PCM pipeline.
pub fn replay(reference:&std::path::Path,output:&std::path::Path,p:&PreprocessArgs,vb:&VarBuilder)->Result<()> {
    use super::hdb2c_r8::{read,save,save_values};
    let device=Device::Cpu;
    let frames=read(reference,"frames",&device)?;save(output,"input_frames",&frames)?;
    let (nt,nfft)=frames.dims2()?;let nb=nfft/2+1;
    if nfft!=p.n_fft || p.mag_power!=2.0{return Err(fail("R10 common-input configuration mismatch"));}
    let flat=frames.flatten_all()?.to_vec1::<f32>()?;
    save_values(output,"fft_complex",&cpu::rfft(&flat,nt,nfft)?,&[nt,nb,2])?;
    let params=vb.pp("preprocessor.featurizer");
    let window=params.get(p.win_length(),"window")?;
    let bank=params.get((1,p.features,nb),"fb")?.reshape((p.features,nb))?;
    save(output,"window",&window)?;save(output,"filterbank",&bank)?;
    let complex=read(reference,"fft_complex",&device)?;
    save(output,"input_fft_complex",&complex)?;
    if complex.dims()!=[nt,nb,2]{return Err(fail("R10 invalid common FFT shape"));}
    let complex=complex.flatten_all()?.to_vec1::<f32>()?;
    let square:Vec<f32>=complex.chunks_exact(2).map(|v|v[0]*v[0]+v[1]*v[1]).collect();
    save_values(output,"magnitude",&cpu::sqrt(&square)?,&[nt,nb])?;
    let magnitude=read(reference,"magnitude",&device)?;save(output,"input_magnitude",&magnitude)?;
    let power:Vec<f32>=magnitude.flatten_all()?.to_vec1::<f32>()?.iter().map(|v|v*v).collect();
    save_values(output,"power",&power,&[nt,nb])?;
    let power=read(reference,"power",&device)?;let bank=read(reference,"filterbank",&device)?;
    save(output,"input_power",&power)?;save(output,"input_filterbank",&bank)?;
    let weights=bank.flatten_all()?.to_vec1::<f32>()?;let power=power.flatten_all()?.to_vec1::<f32>()?;
    save_values(output,"mel_projection",&cpu::mel(&weights,&power,p.features,nt,nb)?,&[p.features,nt])?;
    let projected=read(reference,"mel_projection",&device)?;save(output,"input_mel_projection",&projected)?;
    let guard=match p.log_zero_guard_value.as_ref(){
        None|Some(serde_json::Value::Null)=>2f32.powi(-24),
        Some(serde_json::Value::Number(n))=>n.as_f64().ok_or_else(||fail("R10 invalid log guard"))? as f32,
        _=>return Err(fail("R10 requires numeric checkpoint log guard")),
    };
    let values=projected.flatten_all()?.to_vec1::<f32>()?;
    let adjusted:Vec<f32>=match p.log_zero_guard_type.as_deref().unwrap_or("add"){
        "add"=>values.iter().map(|v|v+guard).collect(),
        "clamp"=>values.iter().map(|v|v.max(guard)).collect(),
        _=>return Err(fail("R10 invalid log guard type")),
    };
    save_values(output,"log_mel",&cpu::ln(&adjusted)?,&[p.features,nt])
}
