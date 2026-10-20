"""C3-A-R3-C: isolated NeMo MaskedConvSequential temporal mask propagation A/B.

The original R3 instrumented candidate is immutable (control). Create a second
candidate with precise per-op temporal masking (treatment). No TDT or GPU edits.
"""
from pathlib import Path
import hashlib
import json
import shutil

SIGNATURE="hdb2c3-r3c-dw-mask-propagation-v0.7.3"
MARKER="HDB2C3_R3C_EXACT_MASK_PROPAGATION_CPU_V1"

RUST_MASK_FN=r"""
// HDB2C3_R3C_EXACT_MASK_PROPAGATION_CPU_V1
// NeMo MaskedConvSequential: current valid temporal length, broadcast over C,F.
fn hdb2c3_r3c_mask_time(x:&Tensor, valid:&[usize])->Result<Tensor>{
    let (batch,_,time,_)=x.dims4()?;
    if batch!=valid.len() || valid.iter().any(|&v|v==0 || v>time) {
        return Err(candle::Error::Msg(format!(
            "R3C mask lengths invalid: valid={:?}, B={}, T={}",valid,batch,time)));
    }
    let mut mask=vec![0f32;batch*time];
    for (bi,&count) in valid.iter().enumerate(){
        for t in 0..count {mask[bi*time+t]=1.0;}
    }
    let mask=Tensor::from_vec(mask,(batch,1,time,1),x.device())?;
    x.broadcast_mul(&mask)
}
"""

def transform_masked_dw_source(source:str)->str:
    """Apply after R3 tracing; preserve all tracing and weight operations."""
    if MARKER in source:raise ValueError("R3C already applied")
    def once(a,b,name):
        nonlocal source
        if source.count(a)!=1:raise ValueError(f"R3C unexpected R3 source: {name}")
        source=source.replace(a,b,1)
    once("        // HDB2C3_R3_DW_TRACE_CPU: observation only; do not change computation.",
         "        // HDB2C3_R3_DW_TRACE_CPU: reference trace retained in R3C.\n"
         "        // HDB2C3_R3C: NeMo masking before each Conv/ReLU, final output.",
         "R3 stage marker")
    once("        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)\n        for (idx, conv) in self.conv.iter().enumerate() {",
         """        let initial=lengths.to_device(&candle::Device::Cpu)?.to_vec1::<i64>()?;
        if initial.iter().any(|&v|v<=0){
            return Err(candle::Error::Msg("R3C invalid input Mel lengths".into()));
        }
        let mut current:Vec<usize>=initial.iter().map(|&v|v as usize).collect();
        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for (idx, conv) in self.conv.iter().enumerate() {""",
         "initial lengths")
    once('''            x = conv.forward(&x)?;
            hdb2c3_trace_stage(&format!("conv_{nemo_idx:02}"),&x)?;''',
         '''            // NeMo masks BEFORE each operation, not only after stride.
            x=hdb2c3_r3c_mask_time(&x,&current)?;
            x = conv.forward(&x)?;
            if idx==0 || idx==1 || idx==3 {
                for v in current.iter_mut(){*v=(*v+1)/2;}
            }
            hdb2c3_trace_stage(&format!("conv_{nemo_idx:02}"),&x)?;''',
         "conv mask and stride length")
    once('''            if idx == 0 || idx % 2 == 0 {
                x = x.relu()?;
                hdb2c3_trace_stage(&format!("relu_{nemo_idx:02}"),&x)?;
            }''',
         '''            if idx == 0 || idx % 2 == 0 {
                x=hdb2c3_r3c_mask_time(&x,&current)?;
                x = x.relu()?;
                hdb2c3_trace_stage(&format!("relu_{nemo_idx:02}"),&x)?;
            }''',
         "relu mask")
    once('''        let x = x.transpose(1, 2)?; // (B, T, C, F)''',
         '''        // NeMo final MaskedConvSequential output mask before flatten/out.
        x=hdb2c3_r3c_mask_time(&x,&current)?;
        let x = x.transpose(1, 2)?; // (B, T, C, F)''',
         "final mask")
    return source+"\n"+RUST_MASK_FN

def transform_mask_probe(src:str)->str:
    old='"encoder_length_argument":"EXPLICIT_NEMO_VALID_FRAMES_IF_INJECTED",'
    if src.count(old)!=1:raise ValueError("R3C needs pinned R3 C3 probe metadata")
    return src.replace(old,old+'\n        "dw_mask_propagation":"NEMO_MASKED_CONV_SEQUENTIAL",',1)

def build_masked_candidate(reference:Path,masked:Path):
    reference=reference.resolve();masked=masked.resolve()
    if reference==masked or reference in masked.parents or masked in reference.parents:
        raise ValueError("R3C candidate must be isolated from R3")
    manifest_path=reference/"HDB2C3-MANIFEST.json"
    if not manifest_path.is_file():raise ValueError("R3 source is not a verified C3 candidate")
    conf=reference/"src/parakeet/conformer.rs"
    probe=reference/"src/bin/hdb2c_probe.rs"
    if not conf.is_file() or not probe.is_file():raise FileNotFoundError("R3 source missing")
    conf_src=transform_masked_dw_source(conf.read_text())
    probe_src=transform_mask_probe(probe.read_text())
    outputs={"src/parakeet/conformer.rs":conf_src,"src/bin/hdb2c_probe.rs":probe_src}
    manifest={"signature":SIGNATURE,"parent":str(reference),
        "source_sha256":{"conformer":hashlib.sha256(conf.read_bytes()).hexdigest(),
                         "probe":hashlib.sha256(probe.read_bytes()).hexdigest()},
        "target_sha256":{p:hashlib.sha256(s.encode()).hexdigest() for p,s in outputs.items()},
        "scope":"isolated CPU temporal masks on DW subsampler; keep baseline R3/C2 immutable"}
    if masked.exists():
        path=masked/"HDB2C3-R3C-MANIFEST.json"
        if not path.is_file() or json.loads(path.read_text())!=manifest:
            raise RuntimeError("R3C candidate exists with different manifest")
        for p,digest in manifest["target_sha256"].items():
            if hashlib.sha256((masked/p).read_bytes()).hexdigest()!=digest:
                raise RuntimeError("R3C candidate source mutated: "+p)
        print("HDB2C3_R3C_CANDIDATE=REUSED_VERIFIED_COPY",flush=True)
        return
    tmp=masked.with_name(masked.name+".creating")
    if tmp.exists():raise RuntimeError("R3C stale creation dir")
    def ignore(_,names):return [v for v in names if v in {"target",".git","__pycache__",".venv"}]
    try:
        shutil.copytree(reference,tmp,ignore=ignore)
        for path,content in outputs.items():(tmp/path).write_text(content)
        (tmp/"HDB2C3-R3C-MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
        tmp.rename(masked)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    print("HDB2C3_R3C_CANDIDATE=CREATED_ISOLATED_COPY",flush=True)

def summarize_ab(before_dw,after_dw,before_layers,after_layers,checkpoint_sha,pcm_sha):
    if before_dw["weights_status"]!="PASS_NUMERIC_PARITY" or after_dw["weights_status"]!="PASS_NUMERIC_PARITY":
        raise ValueError("R3C must preserve 12/12 NeMo weights")
    if len(before_dw["operations"])!=9 or len(after_dw["operations"])!=9:
        raise ValueError("R3C missing DW operation trace")
    if len(before_layers["stages"])!=26 or len(after_layers["stages"])!=26:
        raise ValueError("R3C missing encoder trace")
    if [x["name"] for x in before_dw["operations"]] != [x["name"] for x in after_dw["operations"]]:
        raise ValueError("R3C operation ordering changed")
    if [x["name"] for x in before_layers["stages"]] != [x["name"] for x in after_layers["stages"]]:
        raise ValueError("R3C encoder stage ordering changed")
    improved=[]
    for a,b in zip(before_dw["operations"],after_dw["operations"]):
        improved.append({"name":a["name"],"baseline_status":a["status"],
          "masked_status":b["status"],"baseline_rmse":a.get("rmse"),"masked_rmse":b.get("rmse"),
          "baseline_cosine":a.get("cosine"),"masked_cosine":b.get("cosine"),
          "baseline_max_abs":a.get("max_abs"),"masked_max_abs":b.get("max_abs")})
    return {"stage":"HDB2C3_A_R3C_EXACT_MASK_PROPAGATION_AB",
      "status":"ALL_DW_NUMERIC_PARITY" if after_dw["status"]=="ALL_DW_STAGES_NUMERIC_PARITY" else "DW_NUMERIC_PARITY_NOT_ESTABLISHED",
      "pre_encode_status":"PASS_NUMERIC_PARITY" if after_layers["stages"][0]["status"]=="PASS_NUMERIC_PARITY" else "NOT_QUALIFIED",
      "full_encoder_status":"ALL_STAGES_NUMERIC_PARITY" if after_layers["status"]=="ALL_STAGES_NUMERIC_PARITY" else "NOT_QUALIFIED",
      "first_operation_baseline":before_dw["first_operation_divergence"],
      "first_operation_masked":after_dw["first_operation_divergence"],
      "first_encoder_stage_masked":after_layers["first_divergence"],
      "checkpoint_sha256":checkpoint_sha,"pcm_sha256":pcm_sha,
      "dw_mask_propagation":"NEMO_MASKED_CONV_SEQUENTIAL_ISOLATED_CPU",
      "tdt_numeric_parity":"NOT_TESTED","operations_ab":improved,
      "pre_encode_ab":{"baseline":before_layers["stages"][0],"masked":after_layers["stages"][0]},
      "encoder_last_ab":{"baseline":before_layers["stages"][-1],"masked":after_layers["stages"][-1]},
      "note":"Report A/B only, not TDT or GPU validation. C2 model and normal ASR unchanged."}
