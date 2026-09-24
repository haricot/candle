//! Stage 2F V2-only production reader. All twelve exact device-scoped rows are promoted.
//! Historical performance is not remeasured; current kernel identity and runtime are gated separately.
use std::{collections::{BTreeMap,BTreeSet}, env, fs, io::Write, path::PathBuf};
pub const HEADER:&str="ASD-EXACT-POLICY-V2";
const SM61_PROMOTION_EVIDENCE:&str="b6092036513164f71eb1e1d8b83bc2270b34c476a0aed808c608c86e8436b907";
type AResult<T>=Result<T,Box<dyn std::error::Error>>;
#[derive(Debug)] struct Decision { id:String,state:String,backend:String,signature:BTreeMap<String,String>,implementation_id:String,evidence_ref:String,min_speedup:Option<f64> }
#[derive(Debug)] struct Policy { fields:BTreeMap<String,String>,decisions:Vec<Decision> }
fn required<'a>(m:&'a BTreeMap<String,String>,k:&str)->AResult<&'a str>{Ok(m.get(k).ok_or_else(||format!("missing V2 field {k}"))?.as_str())}
fn n(m:&BTreeMap<String,String>,k:&str)->AResult<usize>{Ok(required(m,k)?.parse()?) }
fn pos(m:&BTreeMap<String,String>,k:&str)->AResult<usize>{let x=n(m,k)?;if x==0{return Err(format!("zero {k}").into())}Ok(x)}
fn dims(m:&BTreeMap<String,String>,k:&str,rank:usize)->AResult<Vec<usize>>{let x=required(m,k)?.split('x').map(str::parse).collect::<Result<Vec<usize>,_>>()?;if x.len()!=rank||x.contains(&0){return Err(format!("bad {k}").into())}Ok(x)}
fn kv(s:&str)->AResult<BTreeMap<String,String>>{let mut m=BTreeMap::new();for t in s.split(','){let(k,v)=t.split_once('=').ok_or("malformed signature")?;if k.is_empty()||v.is_empty()||m.insert(k.into(),v.into()).is_some(){return Err("duplicate/empty signature field".into())}}Ok(m)}
fn parse(text:&str)->AResult<Policy>{let mut lines=text.lines();if lines.next()!=Some(HEADER){return Err("V2 only".into())}let mut fields=BTreeMap::new();let mut decisions=Vec::new();for line in lines{if line.is_empty(){continue}if line.starts_with("decision|"){let p:Vec<_>=line.split('|').collect();if p.len()!=8{return Err("decision requires eight fields".into())}let implementation_id=p[5].strip_prefix("impl=").ok_or("missing impl")?;let evidence_ref=p[6].strip_prefix("evidence=").ok_or("missing evidence")?;let min=p[7].strip_prefix("min_integrated_speedup_x=").ok_or("missing speedup")?;let min_speedup=if min=="none"{None}else{Some(min.parse::<f64>()?)};decisions.push(Decision{id:p[1].into(),state:p[2].into(),backend:p[3].into(),signature:kv(p[4])?,implementation_id:implementation_id.into(),evidence_ref:evidence_ref.into(),min_speedup})}else{let(k,v)=line.split_once('=').ok_or("bad header")?;if fields.insert(k.into(),v.into()).is_some(){return Err("duplicate header".into())}}}Ok(Policy{fields,decisions})}
fn common(d:&Decision)->AResult<(&str,usize,usize,usize,usize,Vec<usize>,Vec<usize>,usize,usize,usize,usize,usize)>{let s=&d.signature;let keys=["op","dim","batch","c_in","c_out","spatial","weight_shape","groups","kernel","stride","padding","output_padding","dilation","dtype","input_layout","weight_layout"];if s.len()!=keys.len()||keys.iter().any(|k|!s.contains_key(*k)){return Err("signature fields differ".into())}if required(s,"dtype")?!="f32"||required(s,"input_layout")?!="contiguous_zero_offset"||required(s,"weight_layout")?!="contiguous_zero_offset"||d.backend!="raw_cuda"{return Err("unsupported dtype/layout/backend".into())}let op=required(s,"op")?;let dim=pos(s,"dim")?;let batch=pos(s,"batch")?;let ci=pos(s,"c_in")?;let co=pos(s,"c_out")?;let g=pos(s,"groups")?;let k=pos(s,"kernel")?;let st=pos(s,"stride")?;let pad=n(s,"padding")?;let dil=pos(s,"dilation")?;if ci%g!=0||co%g!=0{return Err("channels/groups mismatch".into())}let rank=if dim==1{1}else{2};let wrank=if dim==1{3}else{4};Ok((op,dim,batch,ci,co,dims(s,"spatial",rank)?,dims(s,"weight_shape",wrank)?,g,k,st,pad,dil))}
fn validate_evidence_hash(d:&Decision)->AResult<()> { if d.evidence_ref.len()!=64||!d.evidence_ref.bytes().all(|b|b.is_ascii_hexdigit()){return Err("invalid evidence hash".into())} Ok(()) }
fn validate_thresholded_proven(d:&Decision)->AResult<()> { validate_evidence_hash(d)?; if d.min_speedup.is_none_or(|x|!x.is_finite()||x<1.1){return Err("invalid thresholded proven metadata".into())} Ok(()) }
fn specialized_expected(d:&Decision)->AResult<&'static str>{let(op,dim,b,ci,co,sp,w,g,k,st,pad,dil)=common(d)?;if b!=1{return Err("specialized batch".into())}let opad=required(&d.signature,"output_padding")?;let id=match(op,dim,ci,co,sp.as_slice(),g,k,st,pad,opad,dil,w.as_slice()){
("conv_transpose1d",1,128,128,[32],2,3,2,1,"1",1,[128,64,3])=>"ct1d-s32-g2-u1-b256",
("conv_transpose1d",1,128,128,[32],4,3,2,1,"1",1,[128,32,3])=>"ct1d-s32-g4-u1-b256",
("conv_transpose1d",1,128,128,[32],8,3,2,1,"1",1,[128,16,3])=>"ct1d-s32-g8-u1-b256",
("conv_transpose1d",1,128,128,[32],16,3,2,1,"1",1,[128,8,3])=>"ct1d-s32-g16-u1-b256",
("conv_transpose2d",2,128,128,[32,32],16,3,2,1,"1",1,[128,8,3,3])=>"ct2d-s32-g16-u4-b128",
("conv_transpose2d",2,128,128,[32,32],32,3,2,1,"1",1,[128,4,3,3])=>"ct2d-s32-g32-u4-b64",
("conv1d",1,64,64,[128],8,3,1,1,"none",1,[64,8,3])=>"gc1d-l128-g8-u1-b256",
_=>return Err("specialized geometry differs from frozen table".into())};Ok(id)}
fn validate_decision(d:&Decision)->AResult<()> {
    let(op,dim,b,ci,co,sp,w,g,k,st,pad,dil)=common(d)?;
    if d.state!="promoted" { return Err("Stage2F requires promoted decisions only".into()) }
    match d.implementation_id.as_str() {
        "candle.grouped-transpose.raw.v1" => {
            validate_thresholded_proven(d)?;
            if d.id!="ct1d-f32-b1-c64-l128-g2-k3-s2-p1-op1-d1-raw"||op!="conv_transpose1d"||dim!=1||b!=1||ci!=64||co!=64||sp!=[128]||w!=[64,32,3]||g!=2||k!=3||st!=2||pad!=1||required(&d.signature,"output_padding")?!="1"||dil!=1||d.evidence_ref!="3844cacd61a400dc1979c554e5bce787a11693c265e12b2698dae6a1e855fc47" { return Err("historical CT1D changed".into()) }
        }
        "candle.depthwise-conv2d-5x5.raw.v1" => {
            validate_thresholded_proven(d)?;
            if op!="conv2d"||dim!=2||b!=1||ci!=co||g!=ci||k!=5||st!=1||pad!=2||dil!=1||w!=[co,1,5,5]||required(&d.signature,"output_padding")?!="none"||d.evidence_ref!="84f3dc50433e225b1f63c92a08355b7b04f0afeec49694e5e8d5e040cf092a9a" { return Err("DW5x5 promoted metadata mismatch".into()) }
            let ok=matches!((ci,sp.as_slice()),(48,[64,48])|(96,[32,24])|(192,[16,12])|(384,[8,6]));
            if !ok { return Err("DW5x5 exact point unknown".into()) }
        }
        implementation_id if implementation_id.starts_with("candle.sm61-exact-grouped.") => {
            let cand=specialized_expected(d)?;
            validate_evidence_hash(d)?;
            if d.evidence_ref!=SM61_PROMOTION_EVIDENCE||d.min_speedup.is_some()||d.implementation_id!=format!("candle.sm61-exact-grouped.{cand}") { return Err("Stage2F specialized metadata mismatch".into()) }
        }
        _ => return Err("unknown Stage2F implementation".into()),
    }
    Ok(())
}
fn validate(p:&Policy,build_sm:u32)->AResult<()> {let keys=["policy_id","target.vendor","target.architecture","target.scope","target.sm","target.gpu_uuid"];if p.fields.len()!=6||keys.iter().any(|k|!p.fields.contains_key(*k)){return Err("header differs".into())}let policy_uuid=required(&p.fields,"target.gpu_uuid")?;if required(&p.fields,"target.vendor")?!="nvidia"||required(&p.fields,"target.architecture")?!="sm61"||required(&p.fields,"target.scope")?!="device"||required(&p.fields,"target.sm")?!="61"||!policy_uuid.starts_with("GPU-")||policy_uuid.len()<8||build_sm!=61{return Err("target mismatch".into())}let build_uuid=env::var("CANDLE_ASD_TARGET_GPU_UUID").map_err(|_|"missing CANDLE_ASD_TARGET_GPU_UUID")?;if build_uuid!=policy_uuid{return Err("build target UUID mismatch".into())}if p.decisions.len()!=12{return Err("Stage2F requires 12 decisions".into())}let mut ids=BTreeSet::new();let mut sigs=BTreeSet::new();let mut states=BTreeMap::<String,usize>::new();for d in &p.decisions{if !ids.insert(d.id.clone())||!sigs.insert(format!("{:?}",d.signature)){return Err("duplicate id/signature".into())}validate_decision(d)?;*states.entry(d.state.clone()).or_default()+=1}if states.get("promoted")!=Some(&12)||states.len()!=1{return Err("Stage2F requires 12 promoted decisions".into())}Ok(())}
fn rs(s:&str)->String{format!("{s:?}")}
fn op_variant(op:&str)->AResult<&'static str>{Ok(match op{"conv1d"=>"ExactOperation::Conv1d","conv2d"=>"ExactOperation::Conv2d","conv_transpose1d"=>"ExactOperation::ConvTranspose1d","conv_transpose2d"=>"ExactOperation::ConvTranspose2d",_=>return Err("unknown op".into())})}
fn emit(d:&Decision)->AResult<String>{let s=&d.signature;let dim=pos(s,"dim")?;let sp=dims(s,"spatial",if dim==1{1}else{2})?;let w=dims(s,"weight_shape",if dim==1{3}else{4})?;let get=|v:&Vec<usize>,i|v.get(i).copied().unwrap_or(0);let min=match d.min_speedup{Some(x)=>format!("Some({x:.8})"),None=>"None".into()};let opad=if required(s,"output_padding")?=="none"{0}else{n(s,"output_padding")?};Ok(format!("    ExactDecision {{ id: {}, state: {}, backend: {}, implementation_id: {}, evidence_ref: {}, min_integrated_speedup_x: {}, op: {}, dim: {}, batch: {}, c_in: {}, c_out: {}, spatial0: {}, spatial1: {}, weight_rank: {}, weight0: {}, weight1: {}, weight2: {}, weight3: {}, groups: {}, kernel: {}, stride: {}, padding: {}, output_padding: {}, dilation: {}, dtype: {} }},\n",rs(&d.id),rs(&d.state),rs(&d.backend),rs(&d.implementation_id),rs(&d.evidence_ref),min,op_variant(required(s,"op")?)?,dim,pos(s,"batch")?,pos(s,"c_in")?,pos(s,"c_out")?,sp[0],get(&sp,1),w.len(),w[0],w[1],w[2],get(&w,3),pos(s,"groups")?,pos(s,"kernel")?,pos(s,"stride")?,n(s,"padding")?,opad,pos(s,"dilation")?,rs(required(s,"dtype")?)))}
fn render(p:&Policy,validation:bool)->AResult<String>{let mut rows=String::new();for d in &p.decisions{rows.push_str(&emit(d)?)}let t=include_str!("asd_exact_v2_dispatch_template.rs");let policy_uuid=required(&p.fields,"target.gpu_uuid")?;let out=t.replace("__POLICY_ID__",&format!("Some({})",rs(required(&p.fields,"policy_id")?))).replace("__GPU_UUID__",&format!("Some({})",rs(policy_uuid))).replace("__VALIDATION__",if validation{"true"}else{"false"}).replace("__DECISIONS__",&rows);if out.contains("__DECISIONS__"){return Err("unexpanded template".into())}Ok(out)}
pub fn materialize_for_candle_build(build_sm: u32) -> AResult<Option<PathBuf>> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    let output = out_dir.join("asd_exact_dispatch.rs");
    for name in [
        "CANDLE_ASD_EXACT_POLICY",
        "CANDLE_ASD_VALIDATION",
        "CANDLE_ASD_TARGET_GPU_UUID",
        "CANDLE_ASD_EXACT_DISABLE",
        "CANDLE_ASD_V2_CT1D_ENABLE",
        "CANDLE_ASD_REAL_DISPATCH_VALIDATION",
        "CANDLE_ASD_SM61_EXACT_GROUPED_ENABLE",
        "CANDLE_SM61_EXACT_GROUPED_DISABLE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let Some(path) = env::var_os("CANDLE_ASD_EXACT_POLICY") else {
        let t = include_str!("asd_exact_v2_dispatch_template.rs")
            .replace("__POLICY_ID__", "None")
            .replace("__GPU_UUID__", "None")
            .replace("__VALIDATION__", "false")
            .replace("__DECISIONS__", "");
        fs::write(&output, t)?;
        return Ok(None);
    };

    let path = PathBuf::from(path);
    println!("cargo:rerun-if-changed={}", path.display());
    let p = parse(&fs::read_to_string(path)?)?;
    validate(&p, build_sm)?;
    let validation = matches!(
        env::var("CANDLE_ASD_VALIDATION").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    );
    let generated = render(&p, validation)?;
    let mut f = fs::File::create(&output)?;
    f.write_all(generated.as_bytes())?;
    f.sync_all()?;
    println!(
        "cargo:warning=ASD Stage2F production profile {} decisions=12 promoted=12 production=YES device_scope=exact_uuid validation_build={}",
        required(&p.fields, "policy_id")?, validation
    );
    Ok(Some(output))
}
