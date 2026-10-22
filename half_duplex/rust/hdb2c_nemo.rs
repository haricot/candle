//! HDB2-C2 — Rust implementation of NVIDIA NeMo FilterbankFeatures inference path.
//! Source-level reference: NVIDIA-NeMo/Speech, parts/preprocessing/features.py,
//! librosa.filters.mel (Slaney), torch.stft(center=true, pad_mode="constant").
//! This is independently authored Rust, not a copy of upstream Python.
//! No dither is applied at inference (NeMo: only if self.training).
use super::{fft, PreprocessArgs};
use candle::{DType, Device, Error, Result, Tensor};
use serde_json::Value;
use std::f32::consts::PI;

fn fail(message: impl Into<String>) -> Error { Error::Msg(message.into()) }

/// NeMo get_seq_len: floor((samples + pad*2 - n_fft)/hop), excludes
/// the last STFT frame when center=true. May be shorter than raw STFT width.
pub(super) fn valid_frames(samples: usize, p: &PreprocessArgs) -> Result<usize> {
    let hop=p.hop_length();
    if hop == 0 || p.n_fft < p.win_length() || p.n_fft < hop {return Err(fail("invalid NeMo STFT lengths"));}
    let padding=if p.exact_pad {(p.n_fft-hop)/2} else {p.n_fft/2};
    let length=samples.checked_add(2*padding).ok_or_else(||fail("audio length overflow"))?;
    if length < p.n_fft {return Err(fail("WAV too short for NeMo STFT"));}
    Ok((length-p.n_fft)/hop)
}

fn mel_freq_to_hz(m: f64) -> f64 {
    let f_sp=200.0/3.0;
    let min_log_mel=1000.0/f_sp;
    let logstep=(6.4f64).ln()/27.0;
    if m < min_log_mel {f_sp*m} else {1000.0*((m-min_log_mel)*logstep).exp()}
}
fn hz_to_mel(hz: f64) -> f64 {
    let f_sp=200.0/3.0;
    if hz<1000.0 {hz/f_sp} else {1000.0/f_sp + (hz/1000.0).ln()/((6.4f64).ln()/27.0)}
}

/// Mel triangles evaluated at real FFT bin frequencies, not bin-rounded vertices.
/// Matches librosa.filters.mel with htk=False, norm='slaney'.
pub(super) fn filterbank(p: &PreprocessArgs) -> Result<Vec<f32>> {
    if p.mel_norm.as_deref().unwrap_or("slaney")!="slaney" {return Err(fail("HDB2C2 currently requires mel_norm=slaney"));}
    let lo=p.lowfreq.unwrap_or(0.0);
    let hi=p.highfreq.unwrap_or(p.sample_rate as f64/2.0);
    if !(lo>=0.0 && hi>lo && hi<=p.sample_rate as f64 / 2.0) {return Err(fail("invalid mel frequency bounds"));}
    let points:Vec<f64>=(0..p.features+2).map(|i| {
        let t=i as f64/(p.features+1) as f64;
        mel_freq_to_hz(hz_to_mel(lo)+(hz_to_mel(hi)-hz_to_mel(lo))*t)
    }).collect();
    let count=p.n_fft/2+1;
    let mut weights=vec![0f32;p.features*count];
    for m in 0..p.features {
        let a=points[m];let b=points[m+1];let c=points[m+2];
        let area=2.0/(c-a);
        for k in 0..count {
            let hz=p.sample_rate as f64 * k as f64 / p.n_fft as f64;
            let rising=(hz-a)/(b-a);
            let falling=(c-hz)/(c-b);
            weights[m*count+k]=rising.min(falling).max(0.0) as f32 * area as f32;
        }
    }
    Ok(weights)
}
fn win(kind:&str,n:usize)->Result<Vec<f32>>{
    if n<2 {return Err(fail("window shorter than 2"));}
    let denom=(n-1) as f32; // torch.<window>(periodic=False)
    (0..n).map(|i| {
        let a=2.0*PI*i as f32/denom;
        match kind {
            "hann"|"hanning"=>Ok(0.5-0.5*a.cos()),
            "hamming"=>Ok(0.54-0.46*a.cos()),
            "blackman"=>Ok(0.42-0.5*a.cos()+0.08*(2.0*a).cos()),
            "bartlett"=>Ok(1.0-((2.0*i as f32/denom)-1.0).abs()),
            "none"=>Ok(1.0),
            _=>Err(fail(format!("unknown NeMo window: {kind}"))),
        }
    }).collect()
}
fn log_guard(p:&PreprocessArgs)->Result<f32>{
    match p.log_zero_guard_value.as_ref() {
        None|Some(Value::Null)=>Ok(2f32.powi(-24)),
        Some(Value::String(s)) if s=="tiny"=>Ok(f32::MIN_POSITIVE),
        Some(Value::String(s)) if s=="eps"=>Ok(f32::EPSILON),
        Some(Value::Number(n))=>{
            let f=n.as_f64().ok_or_else(||fail("log guard cannot be parsed"))? as f32;
            if !f.is_finite() || f<=0.0 {return Err(fail("log guard must be >0"));} Ok(f)
        },
        _=>Err(fail("unsupported NeMo log guard"))
    }
}
fn normalise(mel:&mut [f32],channels:usize,frame_count:usize,valid:usize,mode:&str)->Result<()> {
    if valid<2 {return Err(fail("NeMo normalization needs >=2 valid frames"));}
    match mode {
        "NA"|"none"|""=>{},
        "per_feature"=>{
            for ch in 0..channels {
                let reference=mel[ch*frame_count];
                let avg=reference+(0..valid).map(|t| (mel[ch*frame_count+t]-reference) as f64).sum::<f64>() as f32/valid as f32;
                let var=(0..valid).map(|t| {let x=(mel[ch*frame_count+t]-avg) as f64;x*x}).sum::<f64>()/(valid-1) as f64;
                let std=var.sqrt() as f32+1e-5;
                for t in 0..valid {mel[ch*frame_count+t]=(mel[ch*frame_count+t]-avg)/std;}
            }
        },
        "all_features"=>{
            let denom=valid*channels;
            let mut total=0.0f64;for c in 0..channels {for t in 0..valid {total+=mel[c*frame_count+t] as f64;}}
            let avg=total/denom as f64;
            let mut var=0.0f64;
            for c in 0..channels {for t in 0..valid {let d=mel[c*frame_count+t] as f64-avg;var+=d*d;}}
            let std=(var/(denom-1) as f64).sqrt() as f32+1e-5;
            for c in 0..channels {for t in 0..valid {mel[c*frame_count+t]=(mel[c*frame_count+t]-avg as f32)/std;}}
        },
        _=>return Err(fail(format!("unsupported NeMo normalization: {mode}")))
    }
    Ok(())
}

pub(super) fn get_logmel(samples:&[f32],p:&PreprocessArgs,device:&Device)->Result<Tensor>{
    p.validate()?;
    if p.frame_splicing.unwrap_or(1)!=1 {return Err(fail("frame_splicing!=1 unsupported by HDB2C2"));}
    if samples.is_empty() {return Err(fail("empty NeMo audio"));}
    let nfft=p.n_fft;let hop=p.hop_length();let wlen=p.win_length();
    if nfft < wlen || hop==0 || p.features==0 {return Err(fail("unsupported NeMo FFT configuration"));}
    let pad=if p.exact_pad {(nfft-hop)/2} else {nfft/2};
    let mut wave=if p.exact_pad {
        let mut padded=vec![0f32;pad];padded.extend_from_slice(samples);padded.resize(padded.len()+pad,0.0);padded
    } else {samples.to_vec()};
    if let Some(v)=p.preemph {
        if !v.is_finite(){return Err(fail("invalid preemphasis"));}
        for i in (1..wave.len()).rev(){let prev=wave[i-1];let now=wave[i];wave[i]=now-v as f32*prev;}
    }
    if !p.exact_pad {
        let mut padded=vec![0f32;pad];padded.extend_from_slice(&wave);padded.resize(padded.len()+pad,0.0);wave=padded;
    }
    let raw_frames= if wave.len()>=nfft {(wave.len()-nfft)/hop+1} else {0};
    let valid=valid_frames(samples.len(),p)?;
    if valid==0 || raw_frames<valid {return Err(fail("NeMo zero or invalid frame count"));}
    let weights=filterbank(p)?;
    let window=win(&p.window,wlen)?;
    let left=(nfft-wlen)/2;
    let guard=log_guard(p)?;
    let guard_type=p.log_zero_guard_type.as_deref().unwrap_or("add");
    let channels=p.features;let nbins=nfft/2+1;
    let mut mel=vec![0f32;channels*raw_frames]; // [features,raw_frames]
    // STFT: center=True constant waveform padding; symmetric window centered in nfft.
    // NeMo uses torch.stft with one-sided FFT, unnormalized magnitude.
    for t in 0..raw_frames {
        let off=t*hop;
        let mut frame=vec![0f32;nfft];
        for i in 0..wlen {frame[left+i]=wave[off+left+i]*window[i];}
        let fft=fft(&frame);
        let mut powers=vec![0f32;nbins];
        for k in 0..nbins {
            let re=fft[2*k];let im=fft[2*k+1];
            let mag=(re*re+im*im).sqrt();
            powers[k]=if (p.mag_power-1.0).abs() < f64::EPSILON {mag} else {mag.powf(p.mag_power as f32)};
        }
        for c in 0..channels {
            let mut value=0f32;
            for k in 0..nbins {value += weights[c*nbins+k]*powers[k];}
            mel[c*raw_frames+t]=if p.log.unwrap_or(true) {
                match guard_type {
                    "add"=>(value+guard).ln(),
                    "clamp"=>value.max(guard).ln(),
                    _=>return Err(fail("invalid log_zero_guard_type")),
                }
            } else {value};
        }
    }
    normalise(&mut mel,channels,raw_frames,valid,&p.normalize)?;
    // NeMo masks all frames beyond its valid length *before* padding to pad_to.
    let alignment=p.pad_to;let target=if alignment>0 {valid.max(raw_frames).div_ceil(alignment)*alignment} else {raw_frames};
    let mut output=vec![p.pad_value as f32;target*channels];
    for t in 0..valid {for c in 0..channels {output[t*channels+c]=mel[c*raw_frames+t];}}
    Tensor::from_vec(output,(1,target,channels),device)?.to_dtype(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    // These tests are compile-time/CPU guards, not a substitute for real NeMo tensor parity.
    fn cfg()->PreprocessArgs {serde_json::from_str(r#"{"sample_rate":16000,"normalize":"NA","window_size":0.025,"window_stride":0.01,"window":"hann","features":128,"n_fft":512,"dither":0,"pad_to":16,"pad_value":0,"preemph":0.97,"mag_power":2.0}"#).unwrap()}
    #[test] fn valid_frame_rule(){assert_eq!(valid_frames(216378,&cfg()).unwrap(),1352);}
    #[test] fn full_bank_nonzero(){let f=filterbank(&cfg()).unwrap();let n=257;assert!((0..128).all(|m|f[m*n..(m+1)*n].iter().any(|v|*v>0.0)));}
    #[test] fn padded_frames_zero(){let p=cfg();let wav=vec![0.0f32;216378];let v=get_logmel(&wav,&p,&Device::Cpu).unwrap();assert_eq!(v.dims(),&[1,1360,128]);let raw=v.flatten_all().unwrap().to_vec1::<f32>().unwrap();assert!(raw[1352*128..].iter().all(|x|*x==0.0));}
}
