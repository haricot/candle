// Included inside the instrumented hdb2c_nemo module. The normal frontend's
// expressions remain unchanged; this mode only replays separate common inputs.
pub(super) fn r8_replay(reference:&std::path::Path,output:&std::path::Path,p:&PreprocessArgs)->Result<()> {
    use crate::parakeet::hdb2c_r8::{read,save,save_values};
    let cpu=Device::Cpu;
    let frames=read(reference,"frames",&cpu)?;
    save(output,"input_frames",&frames)?;
    let (nt,nfft)=frames.dims2()?;
    if nfft!=p.n_fft {return Err(fail("R8 FFT fixture/config mismatch"));}
    let nb=nfft/2+1;
    let input=frames.flatten_all()?.to_vec1::<f32>()?;
    let mut complex=Vec::with_capacity(nt*nb*2);
    for frame in input.chunks_exact(nfft) {complex.extend_from_slice(&fft(frame)[..nb*2]);}
    save_values(output,"fft_complex",&complex,&[nt,nb,2])?;
    let window=win(&p.window,p.win_length())?;
    save_values(output,"window",&window,&[window.len()])?;
    let weights=filterbank(p)?;
    save_values(output,"filterbank",&weights,&[p.features,nb])?;
    // The SAME NeMo complex bins are inputs to magnitude/power, independently
    // of the native FFT above. This does not feed the autonomous PCM path.
    let common=read(reference,"fft_complex",&cpu)?;
    save(output,"input_fft_complex",&common)?;
    if common.dims()!=[nt,nb,2] {return Err(fail("R8 complex shape mismatch"));}
    let common=common.flatten_all()?.to_vec1::<f32>()?;
    let mut magnitude=Vec::with_capacity(nt*nb);
    for v in common.chunks_exact(2) {
        let re=v[0];let im=v[1];let mag=(re*re+im*im).sqrt();
        magnitude.push(mag);
    }
    let common_magnitude=read(reference,"magnitude",&cpu)?;
    if common_magnitude.dims()!=[nt,nb] {return Err(fail("R8 common magnitude shape mismatch"));}
    save(output,"input_magnitude",&common_magnitude)?;
    let power:Vec<f32>=common_magnitude.flatten_all()?.to_vec1::<f32>()?.into_iter()
        .map(|mag|if (p.mag_power-1.0).abs()<f64::EPSILON {mag}else{mag.powf(p.mag_power as f32)}).collect();
    save_values(output,"magnitude",&magnitude,&[nt,nb])?;
    save_values(output,"power",&power,&[nt,nb])?;
    let common_power=read(reference,"power",&cpu)?;
    let common_weights=read(reference,"filterbank",&cpu)?;
    save(output,"input_power",&common_power)?;
    save(output,"input_filterbank",&common_weights)?;
    if common_power.dims()!=[nt,nb] || common_weights.dims()!=[p.features,nb] {
        return Err(fail("R8 common Mel input shape mismatch"));
    }
    let pow=common_power.flatten_all()?.to_vec1::<f32>()?;
    let w=common_weights.flatten_all()?.to_vec1::<f32>()?;
    let mut projected=vec![0f32;p.features*nt];
    for t in 0..nt {for c in 0..p.features {
        let mut value=0f32;
        for k in 0..nb {value+=w[c*nb+k]*pow[t*nb+k];}
        projected[c*nt+t]=value;
    }}
    save_values(output,"mel_projection",&projected,&[p.features,nt])?;
    let common=read(reference,"mel_projection",&cpu)?;
    save(output,"input_mel_projection",&common)?;
    let guard=log_guard(p)?;
    let mut values=common.flatten_all()?.to_vec1::<f32>()?;
    for value in &mut values {
        *value=if p.log.unwrap_or(true) {match p.log_zero_guard_type.as_deref().unwrap_or("add") {
            "add"=>(*value+guard).ln(),"clamp"=>value.max(guard).ln(),
            _=>return Err(fail("R8 invalid logarithm guard")),
        }}else{*value};
    }
    save_values(output,"log_mel",&values,&[p.features,nt])?;
    Ok(())
}
