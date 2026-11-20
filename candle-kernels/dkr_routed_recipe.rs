//! Package-side MMVQ and MoE launch ABI; no CUDA or tensor dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoutedRecipe {
    pub(crate) max_batch: usize,
    pub(crate) quant_threads: u32,
    pub(crate) warp: u32,
    pub(crate) small_warps: u32,
    pub(crate) large_warps: u32,
    pub(crate) small_batch: usize,
    pub(crate) row_padding: usize,
    pub(crate) q8_block: usize,
    pub(crate) q8_bytes: usize,
    pub(crate) max_experts: usize,
    pub(crate) simt_threads: u32,
    pub(crate) simt_tile_n: usize,
    pub(crate) simt_shared: u32,
    pub(crate) gguf_tile_n: usize,
    pub(crate) gguf_shared_base: usize,
    pub(crate) gguf_shared_limit: usize,
}
pub(crate) const S31_ROUTED: RoutedRecipe = RoutedRecipe {
    max_batch: 8, quant_threads: 256, warp: 32, small_warps: 4, large_warps: 2,
    small_batch: 4, row_padding: 512, q8_block: 32, q8_bytes: 36,
    max_experts: 1024, simt_threads: 128, simt_tile_n: 32, simt_shared: 2304,
    gguf_tile_n: 4, gguf_shared_base: 1024, gguf_shared_limit: 49152,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub(crate) grid: (u32,u32,u32),
    pub(crate) block: (u32,u32,u32),
    pub(crate) shared: u32,
}
fn geom(grid:(u32,u32,u32),block:(u32,u32,u32),shared:u32)->Geometry {
    Geometry { grid, block, shared }
}
pub(crate) struct MmvqPlan {
    pub(crate) quant: String,
    pub(crate) matrix: String,
    pub(crate) qcfg: Geometry,
    pub(crate) mcfg: Geometry,
}
pub(crate) struct GgufPlan {
    pub(crate) padded: usize,
    pub(crate) scratch: usize,
    pub(crate) qcfg: Geometry,
    pub(crate) mcfg: Geometry,
}
const LIMIT: usize = i32::MAX as usize;
impl RoutedRecipe {
    pub(crate) fn mmvq_workspace(self,k:usize,n:usize,batch:usize)->Result<(usize,usize,usize),&'static str> {
        if k==0 || n==0 || batch==0 || batch>self.max_batch || k>LIMIT || n>LIMIT {
            return Err("MMVQ workspace outside native ABI");
        }
        let padded=k.div_ceil(self.row_padding)*self.row_padding;
        let elements=padded.checked_mul(batch).filter(|&v|v<=LIMIT).ok_or("MMVQ padded index overflow")?;
        if n.checked_mul(batch).is_none_or(|v|v>LIMIT) { return Err("MMVQ output index overflow"); }
        let scratch=(elements/self.q8_block).checked_mul(self.q8_bytes).ok_or("MMVQ scratch overflow")?;
        Ok((padded,padded/self.q8_block,scratch))
    }
    pub(crate) fn mmvq(self, codec:&str, tag:&str,k:usize,padded:usize,n:usize,batch:usize) -> Result<MmvqPlan,&'static str> {
        if !matches!(codec,"q4_0"|"q4_1"|"q5_0"|"q5_1"|"q8_0"|"q2_k"|"q3_k"|"q4_k"|"q5_k"|"q6_k")
            || !matches!(tag,"bf16"|"f16"|"f32")
            || batch==0 || batch>self.max_batch || k==0 || n==0 || padded<k
            || padded%self.row_padding!=0 || [k,padded,n].iter().any(|&v|v>LIMIT)
            || padded.checked_mul(batch).is_none_or(|v|v>LIMIT)
            || n.checked_mul(batch).is_none_or(|v|v>LIMIT) {
            return Err("MMVQ geometry or codec outside native ABI");
        }
        Ok(MmvqPlan {
            quant:format!("mmvq_gguf_quantize_q8_1_{tag}"),
            matrix:format!("mmvq_gguf_{codec}_{tag}_plain_cuda{batch}"),
            qcfg:geom((padded.div_ceil(self.quant_threads as usize) as u32,batch as u32,1),(self.quant_threads,1,1),0),
            mcfg:geom((n.div_ceil(if batch==1{1}else{2}) as u32,1,1),
                (self.warp,if batch<=self.small_batch{self.small_warps}else{self.large_warps},1),0),
        })
    }
    pub(crate) fn simt(self,e:usize,n:usize,k:usize,m:usize)->Result<[Geometry;3],&'static str>{
        if e==0 || e>self.max_experts || n==0 || n>LIMIT || k==0 || k>LIMIT || k%2!=0
            || m==0 || m>LIMIT || n.div_ceil(self.simt_tile_n)>65535 {
            return Err("SIMT MoE geometry outside native ABI");
        }
        let scan=e.max(self.warp as usize) as u32;
        Ok([
            geom((m.div_ceil(self.quant_threads as usize) as u32,1,1),(self.quant_threads,1,1),0),
            geom((1,1,1),(scan,1,1),scan*4),
            geom((e as u32,n.div_ceil(self.simt_tile_n) as u32,1),(self.simt_threads,1,1),self.simt_shared)
        ])
    }
    pub(crate) fn gguf(self,codec:&str,n:usize,k:usize,r:usize,m:usize)->Result<GgufPlan,&'static str>{
        let (qk,bytes)=match codec {
            "q8_0"=>(32,34), "q4_k"=>(256,144), "q2_k"=>(256,84),
            "q3_k"=>(256,110), "q5_k"=>(256,176), "q6_k"=>(256,210),
            _=>return Err("unsupported GGUF MoE codec"),
        };
        if [n,k,r,m].iter().any(|&v|v==0||v>LIMIT) || m>65535 || r>65535 || k%qk!=0 {
            return Err("GGUF MoE geometry outside native ABI");
        }
        let padded=k.div_ceil(self.row_padding)*self.row_padding;
        let elements=padded.checked_mul(r).filter(|&v|v<=LIMIT).ok_or("GGUF MoE padded index overflow")?;
        let shared=(k/qk).checked_mul(bytes*self.small_warps as usize)
            .and_then(|v|v.checked_add(self.gguf_shared_base))
            .filter(|&v|v<=self.gguf_shared_limit).ok_or("GGUF MoE exceeds catalogue shared memory")?;
        let scratch=(elements/self.q8_block).checked_mul(self.q8_bytes).ok_or("GGUF MoE scratch overflow")?;
        Ok(GgufPlan { padded,scratch,
            qcfg:geom((padded.div_ceil(self.quant_threads as usize) as u32,r as u32,1),(self.quant_threads,1,1),0),
            mcfg:geom((n.div_ceil(self.gguf_tile_n) as u32,m as u32,1),(self.warp,self.small_warps,1),shared as u32),
        })
    }
    pub(crate) fn moe_accepts(self,gguf:bool,dtype:&str)->bool {
        if gguf { dtype=="f32" } else { dtype=="f16" }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_routed_recipe_legacy_geometry(){
        let r=S31_ROUTED;
        for codec in ["q4_0","q4_1","q5_0","q5_1","q8_0","q2_k","q3_k","q4_k","q5_k","q6_k"]{
            for tag in ["bf16","f16","f32"]{
                for batch in 1..=8{
                    let p=r.mmvq(codec,tag,1024,1024,65,batch).unwrap();
                    assert_eq!(p.matrix,format!("mmvq_gguf_{codec}_{tag}_plain_cuda{batch}"));
                    assert_eq!(p.quant,format!("mmvq_gguf_quantize_q8_1_{tag}"));
                    assert_eq!(p.qcfg,geom((4,batch as u32,1),(256,1,1),0));
                    assert_eq!(p.mcfg,geom((if batch==1{65}else{33},1,1),(32,if batch<=4{4}else{2},1),0));
                }
            }
        }
        assert_eq!(r.mmvq_workspace(768,65,3).unwrap(),(1024,32,3456));
        assert!(r.mmvq_workspace(i32::MAX as usize,65,1).is_err());
        assert!(r.mmvq_workspace(1024,65,9).is_err());
        assert!(r.mmvq("q4_k","f16",1024,1024,65,9).is_err());
        assert!(r.mmvq("q4_k","f16",1024,512,65,1).is_err());
        for e in [1,31,32,33,1024]{
            let p=r.simt(e,65,1024,257).unwrap();
            assert_eq!(p[0],geom((2,1,1),(256,1,1),0));
            assert_eq!(p[1],geom((1,1,1),(e.max(32) as u32,1,1),e.max(32) as u32*4));
            assert_eq!(p[2],geom((e as u32,3,1),(128,1,1),2304));
        }
        assert!(r.simt(1025,65,1024,257).is_err());
        for (codec,qk,bytes) in [("q8_0",32,34),("q4_k",256,144),("q2_k",256,84),("q3_k",256,110),("q5_k",256,176),("q6_k",256,210)]{
            let p=r.gguf(codec,65,1024,3,6).unwrap();
            assert_eq!(p.padded,1024);
            assert_eq!(p.scratch,3*1024/32*36);
            assert_eq!(p.qcfg,geom((4,3,1),(256,1,1),0));
            assert_eq!(p.mcfg,geom((17,6,1),(32,4,1),(1024/qk*bytes*4+1024) as u32));
        }
        assert!(r.gguf("q8_0",65,16384,1,1).is_err());
        assert!(r.gguf("q4_k",65,1023,1,1).is_err());
        assert!(r.gguf("q4_k",65,1024,65536,65536).is_err());
        assert!(r.moe_accepts(false,"f16"));
        assert!(!r.moe_accepts(false,"bf16"));
        assert!(r.moe_accepts(true,"f32"));
    }
}
