//! Native F32 Kimodo two-stage denoiser and deterministic DDIM (initial port).
//! Reference: nv-tlabs/kimodo at e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954.
//! Fixed-length batches, externally prepared embeddings/conditions and statistics.
//! Root conversion uses an explicit small host-side F32 calculation; no Python runtime.
//! Real checkpoint parity has not yet been established.
use candle::{DType, Device, Result, Tensor};
use candle_nn::{linear, layer_norm, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

pub const MOTION_DIM: usize = 369; // 5 + 30 * (3 + 6 + 3) + 4
const ROOT: usize = 5;
const LOCAL_ROOT: usize = 4;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub latent_dim: usize,
    pub ff_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub text_dim: usize,
    pub text_tokens: usize,
    pub use_text_mask: bool,
    pub input_first_heading_angle: bool,
    pub norm_first: bool,
    pub activation: String,
    pub num_base_steps: usize,
    pub fps: f32,
}
impl Config {
    fn validate(&self) -> Result<()> {
        if self.latent_dim == 0 || self.latent_dim % 2 != 0
            || self.num_heads == 0 || self.latent_dim % self.num_heads != 0
            || self.num_layers == 0 || self.ff_size == 0
            || self.text_dim == 0 || self.text_tokens == 0 || self.text_tokens > 4096
            || self.num_base_steps < 2 || self.num_base_steps > 5000
            || !self.fps.is_finite() || self.fps <= 0.0
            || !matches!(self.activation.as_str(), "gelu" | "relu") {
            candle::bail!("invalid/unsupported Kimodo configuration");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RootStats {
    pub global_mean: [f32; ROOT],
    pub global_std: [f32; ROOT],
    pub local_mean: [f32; LOCAL_ROOT],
    pub local_std: [f32; LOCAL_ROOT],
}
impl RootStats {
    fn validate(&self) -> Result<()> {
        if self.global_mean.iter().chain(self.local_mean.iter()).any(|v| !v.is_finite())
            || self.global_std.iter().chain(self.local_std.iter()).any(|v| !v.is_finite() || *v < 0.) {
            candle::bail!("invalid Kimodo root statistics");
        }
        Ok(())
    }
    /// Reference uses sqrt(std^2+1e-5), wrapped angular differences, world XZ
    /// finite differences and repeats the last valid velocity.
    pub fn global_to_local(&self, root: &Tensor, fps: f32) -> Result<Tensor> {
        self.validate()?;
        let (b,t,d) = root.dims3()?;
        if d != ROOT || t < 2 || root.dtype() != DType::F32 || !fps.is_finite() || fps <= 0. {
            candle::bail!("invalid Kimodo root input");
        }
        let rows = root.to_vec3::<f32>()?;
        let mut out = Vec::with_capacity(b*t*LOCAL_ROOT);
        for batch in rows {
            let mut physical = Vec::with_capacity(t);
            for r in batch {
                let mut p = [0f32; ROOT];
                for i in 0..ROOT {
                    p[i] = r[i] * (self.global_std[i]*self.global_std[i]+1e-5).sqrt() + self.global_mean[i];
                }
                physical.push(p);
            }
            for i in 0..t {
                let j = i.min(t-2);
                let p = physical[j];
                let q = physical[j+1];
                let a = p[4].atan2(p[3]);
                let z = q[4].atan2(q[3]);
                let delta = (z-a).sin().atan2((z-a).cos()) * fps;
                let row = [delta, (q[0]-p[0])*fps, (q[2]-p[2])*fps, physical[i][1]];
                for k in 0..LOCAL_ROOT {
                    out.push((row[k]-self.local_mean[k])/(self.local_std[k]*self.local_std[k]+1e-5).sqrt());
                }
            }
        }
        Tensor::from_vec(out, (b,t,LOCAL_ROOT), root.device())
    }
}

struct EncoderLayer {
    qkv: Linear, out: Linear, ff1: Linear, ff2: Linear,
    norm1: LayerNorm, norm2: LayerNorm, heads: usize,
    norm_first: bool, gelu: bool,
}
impl EncoderLayer {
    fn load(c: &Config, vb: VarBuilder) -> Result<Self> {
        let d = c.latent_dim;
        let a = vb.pp("self_attn");
        let qkv = Linear::new(a.get((3*d,d),"in_proj_weight")?, Some(a.get(3*d,"in_proj_bias")?));
        Ok(Self {
            qkv, out: linear(d,d,a.pp("out_proj"))?,
            ff1: linear(d,c.ff_size,vb.pp("linear1"))?,
            ff2: linear(c.ff_size,d,vb.pp("linear2"))?,
            norm1: layer_norm(d,1e-5,vb.pp("norm1"))?,
            norm2: layer_norm(d,1e-5,vb.pp("norm2"))?,
            heads:c.num_heads, norm_first:c.norm_first, gelu:c.activation=="gelu",
        })
    }
    fn attention(&self, x:&Tensor, mask:&Tensor) -> Result<Tensor> {
        let (b,t,d)=x.dims3()?;
        let h=self.heads;
        let hd=d/h;
        let qkv=self.qkv.forward(x)?;
        let split=|i| -> Result<Tensor> {
            qkv.narrow(2,i*d,d)?.contiguous()?.reshape((b,t,h,hd))?.transpose(1,2)?.contiguous()?.reshape((b*h,t,hd))
        };
        let q=split(0)?;
        let k=split(1)?;
        let v=split(2)?;
        let score=(q.matmul(&k.transpose(1,2)?.contiguous()?)? / (hd as f64).sqrt())?
            .reshape((b,h,t,t))?.broadcast_add(mask)?;
        let a=candle_nn::ops::softmax_last_dim(&score)?.reshape((b*h,t,t))?;
        let y=a.matmul(&v)?.reshape((b,h,t,hd))?.transpose(1,2)?.contiguous()?.reshape((b,t,d))?;
        self.out.forward(&y)
    }
    fn ff(&self,x:&Tensor)->Result<Tensor>{
        let y=self.ff1.forward(x)?;
        let y=if self.gelu { y.gelu_erf()? } else { y.relu()? };
        self.ff2.forward(&y)
    }
    fn forward(&self,x:&Tensor,mask:&Tensor)->Result<Tensor>{
        if self.norm_first {
            let y=(x+self.attention(&self.norm1.forward(x)?,mask)?)?;
            &y+self.ff(&self.norm2.forward(&y)?)?
        } else {
            let y=self.norm1.forward(&(x+self.attention(x,mask)?)?)?;
            self.norm2.forward(&(&y+self.ff(&y)?)?)
        }
    }
}

fn pe(start:usize,n:usize,d:usize,device:&Device)->Result<Tensor>{
    let mut data=Vec::with_capacity(n*d);
    for p in start..start+n {
        for j in 0..d {
            let a=p as f32 * 10000f32.powf(-((j/2)*2) as f32/d as f32);
            data.push(if j%2==0 {a.sin()} else {a.cos()});
        }
    }
    Tensor::from_vec(data,(1,n,d),device)
}

struct Backbone {
    input:Linear, output:Linear, text:Linear, time1:Linear, time2:Linear,
    heading:Linear, layers:Vec<EncoderLayer>, cfg:Config,
}
impl Backbone {
    fn load(c:&Config,input:usize,output:usize,vb:VarBuilder)->Result<Self>{
        let d=c.latent_dim;
        let mut layers=Vec::with_capacity(c.num_layers);
        for i in 0..c.num_layers {
            layers.push(EncoderLayer::load(c,vb.pp(format!("seqTransEncoder.layers.{i}")))?);
        }
        Ok(Self{
            input:linear(input,d,vb.pp("input_linear"))?,
            output:linear(d,output,vb.pp("output_linear"))?,
            text:linear(c.text_dim,d,vb.pp("embed_text"))?,
            time1:linear(d,d,vb.pp("embed_timestep.time_embed.0"))?,
            time2:linear(d,d,vb.pp("embed_timestep.time_embed.2"))?,
            heading:linear(2,d,vb.pp("linear_first_heading_angle"))?,
            layers,cfg:c.clone(),
        })
    }
    fn forward(&self,x:&Tensor,text:&Tensor,text_lengths:&[usize],heading:&[f32],step:usize)->Result<Tensor>{
        let (b,t,_)=x.dims3()?;
        let (tb,nt,td)=text.dims3()?;
        let c=&self.cfg;
        let n=c.text_tokens;
        if tb!=b || td!=c.text_dim || nt==0 || nt>n || text_lengths.len()!=b || heading.len()!=b
            || text_lengths.iter().any(|&l|l>nt) || step>=c.num_base_steps
            || heading.iter().any(|v|!v.is_finite()) || t+n+2>5000 {
            candle::bail!("Kimodo text/heading/timestep shape mismatch");
        }
        let d=c.latent_dim;
        let dev=x.device();
        let text=if nt<n {Tensor::cat(&[text,&Tensor::zeros((b,n-nt,td),DType::F32,dev)?],1)?}else{text.clone()};
        let e=self.text.forward(&text)?;
        let tim=self.time2.forward(&candle_nn::ops::silu(&self.time1.forward(&pe(step,1,d,dev)?)?)?)?.broadcast_as((b,1,d))?.contiguous()?;
        let x=self.input.forward(x)?;
        let prefix=n+1+usize::from(c.input_first_heading_angle);
        let seq=if c.input_first_heading_angle {
            let hs=heading.iter().flat_map(|a|[a.cos(),a.sin()]).collect::<Vec<_>>();
            let h=self.heading.forward(&Tensor::from_vec(hs,(b,1,2),dev)?)?;
            Tensor::cat(&[&e,&tim,&h,&x],1)?
        }else{Tensor::cat(&[&e,&tim,&x],1)?};
        let len=prefix+t;
        let mut m=vec![0f32;b*len];
        if c.use_text_mask {
            for (i,&valid) in text_lengths.iter().enumerate(){
                for j in valid..n {m[i*len+j]=f32::NEG_INFINITY;}
            }
        }
        let mask=Tensor::from_vec(m,(b,1,1,len),dev)?;
        let mut y=seq.broadcast_add(&pe(0,len,d,dev)?)?;
        for layer in &self.layers {y=layer.forward(&y,&mask)?;}
        self.output.forward(&y.narrow(1,prefix,t)?.contiguous()?)
    }
}

pub struct Kimodo {
    root:Backbone,body:Backbone,stats:RootStats,cfg:Config,
}
pub enum Guidance { None, Regular(f64), Separated{ text:f64, constraint:f64 } }
impl Kimodo {
    /// Pass a F32 VarBuilder rooted at the actual checkpoint's backbone prefix
    /// (official source removes "denoiser.backbone." before loading).
    pub fn load(cfg:Config,stats:RootStats,vb:VarBuilder)->Result<Self>{
        cfg.validate()?;stats.validate()?;
        if vb.dtype()!=DType::F32 {candle::bail!("Kimodo requires F32 weights");}
        Ok(Self{
            root:Backbone::load(&cfg,2*MOTION_DIM,ROOT,vb.pp("root_model"))?,
            body:Backbone::load(&cfg,2*MOTION_DIM-ROOT+LOCAL_ROOT,MOTION_DIM-ROOT,vb.pp("body_model"))?,
            stats,cfg,
        })
    }
    /// A fixed-length, normalized motion batch and F32 binary observation mask.
    /// The text lengths are key-padding masks; no causal attention is used.
    pub fn forward(&self,x:&Tensor,text:&Tensor,text_lengths:&[usize],heading:&[f32],
        observed:&Tensor,mask:&Tensor,timestep:usize)->Result<Tensor>{
        let (b,t,d)=x.dims3()?;
        if b==0 || !(2..=300).contains(&t) || d!=MOTION_DIM
            || observed.dims()!=x.dims() || mask.dims()!=x.dims()
            || [x,text,observed,mask].iter().any(|v|v.dtype()!=DType::F32) {
            candle::bail!("Kimodo expected normalized F32 [B,2..300,369] and matching conditions");
        }
        let keep=mask.affine(-1.,1.)?;
        let x=((&keep*x)?+(observed*mask)?)?;
        let root_input=Tensor::cat(&[&x,mask],2)?;
        let root=self.root.forward(&root_input,text,text_lengths,heading,timestep)?;
        let local=self.stats.global_to_local(&root,self.cfg.fps)?;
        let body_input=Tensor::cat(&[&local,&x.narrow(2,ROOT,MOTION_DIM-ROOT)?,mask],2)?;
        let body=self.body.forward(&body_input,text,text_lengths,heading,timestep)?;
        Tensor::cat(&[&root,&body],2)
    }
    pub fn guided(&self,x:&Tensor,text:&Tensor,lengths:&[usize],heading:&[f32],observed:&Tensor,
        mask:&Tensor,timestep:usize,guidance:&Guidance)->Result<Tensor>{
        match *guidance{
            Guidance::None=>self.forward(x,text,lengths,heading,observed,mask,timestep),
            Guidance::Regular(w)=>{
                if !w.is_finite(){candle::bail!("invalid CFG weight");}
                let cond=self.forward(x,text,lengths,heading,observed,mask,timestep)?;
                let un=self.forward(x,&text.zeros_like()?,&vec![0;lengths.len()],heading,observed,&mask.zeros_like()?,timestep)?;
                &un+(&cond-&un)?.affine(w,0.)?
            },
            Guidance::Separated{text:wt,constraint:wc}=>{
                if !wt.is_finite()||!wc.is_finite(){candle::bail!("invalid CFG weight");}
                let empty=text.zeros_like()?;
                let zero_mask=mask.zeros_like()?;
                let no_text=vec![0;lengths.len()];
                let un=self.forward(x,&empty,&no_text,heading,observed,&zero_mask,timestep)?;
                let tx=self.forward(x,text,lengths,heading,observed,&zero_mask,timestep)?;
                let co=self.forward(x,&empty,&no_text,heading,observed,mask,timestep)?;
                ((&un+(&tx-&un)?.affine(wt,0.)?)?+(&co-&un)?.affine(wc,0.)?)
            }
        }
    }
    /// Explicit noise input makes cross-runtime comparisons reproducible.
    pub fn sample(&self,noise:&Tensor,text:&Tensor,lengths:&[usize],heading:&[f32],observed:&Tensor,
        mask:&Tensor,steps:usize,guidance:Guidance)->Result<Tensor>{
        let schedule=Ddim::new(self.cfg.num_base_steps,steps)?;
        let mut x=noise.clone();
        for i in (0..steps).rev(){
            let clean=self.guided(&x,text,lengths,heading,observed,mask,schedule.timesteps[i],&guidance)?;
            x=schedule.step(&x,&clean,i)?;
        }
        Ok(x)
    }
}

/// Cosine beta schedule, x0 prediction, deterministic DDIM eta=0.
pub struct Ddim { pub timesteps:Vec<usize>, alpha:Vec<f32> }
impl Ddim {
    pub fn new(base:usize,steps:usize)->Result<Self>{
        if base<2 || base>5000 || steps==0 || steps>base {candle::bail!("invalid DDIM steps");}
        let ab=|t:f64| ((t+0.008)/1.008*std::f64::consts::FRAC_PI_2).cos().powi(2);
        let mut cum=1f32;
        let mut train=Vec::with_capacity(base);
        for i in 0..base{
            let beta=(1.-ab((i+1) as f64/base as f64)/ab(i as f64/base as f64)).min(0.999) as f32;
            cum*=1.-beta;train.push(cum);
        }
        let stride=(base-1) as f32/(steps.saturating_sub(1).max(1) as f32);
        let timesteps=(0..steps).map(|i|((i as f32*stride).round_ties_even() as usize).min(base-1)).collect::<Vec<_>>();
        let mut prev=1f32;
        let mut prod=1f32;
        let alpha=timesteps.iter().map(|&i|{
            let a=train[i];
            let beta=1.-a/prev;
            prev=a;
            prod*=1.-beta;
            prod.max(1e-9)
        }).collect();
        Ok(Self{timesteps,alpha})
    }
    pub fn step(&self,x:&Tensor,clean:&Tensor,i:usize)->Result<Tensor>{
        if i>=self.alpha.len() || x.dims()!=clean.dims() || x.dtype()!=DType::F32 || clean.dtype()!=DType::F32 {
            candle::bail!("invalid DDIM input");
        }
        if i==0{return Ok(clean.clone());}
        let a=self.alpha[i];
        let p=self.alpha[i-1];
        let eps=(x.affine((1./a).sqrt() as f64,0.)?-clean)?.affine((a/(1.-a)).sqrt() as f64,0.)?;
        clean.affine(p.sqrt() as f64,0.)?+eps.affine((1.-p).sqrt() as f64,0.)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn smoke(dev:&Device)->Result<()>{
        let cfg=Config{latent_dim:8,ff_size:16,num_layers:1,num_heads:2,text_dim:4,text_tokens:3,
            use_text_mask:true,input_first_heading_angle:true,norm_first:false,activation:"gelu".into(),num_base_steps:10,fps:30.};
        let stats=RootStats{global_mean:[0.;5],global_std:[1.;5],local_mean:[0.;4],local_std:[1.;4]};
        let model=Kimodo::load(cfg,stats,VarBuilder::zeros(DType::F32,dev))?;
        let noise=Tensor::ones((1,8,MOTION_DIM),DType::F32,dev)?;
        let text=Tensor::ones((1,2,4),DType::F32,dev)?;
        let zero=noise.zeros_like()?;
        let y=model.sample(&noise,&text,&[2],&[0.],&zero,&zero,3,Guidance::Separated{text:2.,constraint:2.})?;
        assert_eq!(y.dims(),noise.dims());
        assert!(y.flatten_all()?.to_vec1::<f32>()?.iter().all(|v|*v==0.));
        Ok(())
    }
    #[test]
    fn kimodo_cpu_smoke()->Result<()>{smoke(&Device::Cpu)}
    #[test]
    fn kimodo_ddim_schedule()->Result<()>{
        let d=Ddim::new(1000,50)?;
        assert_eq!(d.timesteps[0],0);assert_eq!(d.timesteps[49],999);
        assert!(d.alpha.windows(2).all(|p|p[0]>=p[1]));
        let x=Tensor::new(&[2f32],&Device::Cpu)?;
        let y=Tensor::new(&[3f32],&Device::Cpu)?;
        assert_eq!(d.step(&x,&y,0)?.to_vec1::<f32>()?,vec![3.]);
        Ok(())
    }
    #[cfg(feature="cuda")]
    #[test]
    #[ignore="requires actual CUDA hardware"]
    fn kimodo_cuda_smoke()->Result<()>{smoke(&Device::new_cuda(0)?)}
}
