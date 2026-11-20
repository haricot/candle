//! Frozen S3.1 FP4 launch descriptor. Contains no CUDA driver/tensor dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp4Recipe {
    pub(crate) scalar: &'static str,
    pub(crate) cast: &'static str,
    pub(crate) quantize: &'static str,
    pub(crate) reference: &'static str,
    pub(crate) tiled: &'static str,
    pub(crate) mx_vec: [&'static str; 8],
    pub(crate) mx_mat: &'static str,
    pub(crate) mx_moe: &'static str,
    pub(crate) auto_cc: u32,
    pub(crate) min_dp4a_cc: u32,
    pub(crate) auto_min_n: usize,
    pub(crate) auto_min_k: usize,
    pub(crate) auto_min_batch: usize,
    pub(crate) scalar_threads: u32,
    pub(crate) cast_threads: u32,
    pub(crate) quant_threads: u32,
    pub(crate) warp: u32,
    pub(crate) matrix_warps: u32,
    pub(crate) tile_n: usize,
    pub(crate) tile_m: usize,
}

pub(crate) const S31_FP4: Fp4Recipe = Fp4Recipe {
    scalar: "nvfp4_arch_e6_w4a16_scalar",
    cast: "nvfp4_arch_e6_f16_to_f32",
    quantize: "quantize_q8_1",
    reference: "nvfp4_mat_mul_q8_1",
    tiled: "nvfp4_mat_mul_q8_1_tiled4x8",
    mx_vec: ["mul_mat_vec_mxfp4_q8_1_cuda1", "mul_mat_vec_mxfp4_q8_1_cuda2",
        "mul_mat_vec_mxfp4_q8_1_cuda3", "mul_mat_vec_mxfp4_q8_1_cuda4",
        "mul_mat_vec_mxfp4_q8_1_cuda5", "mul_mat_vec_mxfp4_q8_1_cuda6",
        "mul_mat_vec_mxfp4_q8_1_cuda7", "mul_mat_vec_mxfp4_q8_1_cuda8"],
    mx_mat: "mul_mat_mxfp4",
    mx_moe: "indexed_moe_forward_mxfp4_q8_1",
    auto_cc: 61, min_dp4a_cc: 61,
    auto_min_n: 512, auto_min_k: 1024, auto_min_batch: 9,
    scalar_threads: 128, cast_threads: 256, quant_threads: 256,
    warp: 32, matrix_warps: 4, tile_n: 4, tile_m: 8,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp4Launch {
    pub(crate) symbol: &'static str,
    pub(crate) grid: (u32, u32, u32),
    pub(crate) block: (u32, u32, u32),
}

impl Fp4Recipe {
    pub(crate) fn symbols(self) -> impl Iterator<Item = &'static str> {
        [self.scalar, self.cast, self.quantize, self.reference, self.tiled,
         self.mx_mat, self.mx_moe].into_iter().chain(self.mx_vec)
    }
    pub(crate) fn auto_tiled(self, cc: u32, n: usize, k: usize, batch: usize) -> bool {
        cc == self.auto_cc && n >= self.auto_min_n && k >= self.auto_min_k
            && batch >= self.auto_min_batch
    }
    fn flat(symbol: &'static str, elements: usize, threads: u32) -> Result<Fp4Launch, &'static str> {
        if elements == 0 || elements > i32::MAX as usize { return Err("FP4 flat launch outside ABI"); }
        Ok(Fp4Launch { symbol, grid: (elements.div_ceil(threads as usize) as u32, 1, 1),
            block: (threads, 1, 1) })
    }
    pub(crate) fn scalar_plan(self, elements: usize) -> Result<Fp4Launch, &'static str> {
        Self::flat(self.scalar, elements, self.scalar_threads)
    }
    pub(crate) fn cast_plan(self, elements: usize) -> Result<Fp4Launch, &'static str> {
        Self::flat(self.cast, elements, self.cast_threads)
    }
    pub(crate) fn quant_plan(self, padded: usize, batch: usize) -> Result<Fp4Launch, &'static str> {
        if padded == 0 || padded > i32::MAX as usize || padded % self.quant_threads as usize != 0
            || batch == 0 || batch > 65535 { return Err("FP4 Q8 launch outside ABI"); }
        Ok(Fp4Launch { symbol: self.quantize,
            grid: ((padded / self.quant_threads as usize) as u32, batch as u32, 1),
            block: (self.quant_threads, 1, 1) })
    }
    pub(crate) fn native_matrix(self, cc: u32, tiled: bool, n: usize, batch: usize) -> Result<Fp4Launch, &'static str> {
        if n == 0 || n > i32::MAX as usize || batch == 0 || batch > 65535
            || (tiled && cc < self.min_dp4a_cc) { return Err("FP4 native matrix outside device/shape ABI"); }
        let (symbol, x, y) = if tiled {
            (self.tiled, n.div_ceil(self.tile_n), batch.div_ceil(self.tile_m))
        } else { (self.reference, n, batch) };
        Ok(Fp4Launch { symbol, grid: (x as u32, y as u32, 1),
            block: (self.warp, self.matrix_warps, 1) })
    }
    pub(crate) fn mx_vector(self, rows: usize, batch: usize) -> Result<Fp4Launch, &'static str> {
        if rows == 0 || rows > i32::MAX as usize || !(1..=8).contains(&batch) {
            return Err("MXFP4 vector launch outside ABI");
        }
        let (x, warps) = if batch == 1 { (rows, 4) } else if batch <= 4 {
            (rows.div_ceil(2), 4)
        } else { (rows.div_ceil(2), 2) };
        Ok(Fp4Launch { symbol: self.mx_vec[batch-1], grid: (x as u32, 1, 1),
            block: (self.warp, warps, 1) })
    }
    pub(crate) fn mx_matrix(self, rows: usize, batch: usize) -> Result<Fp4Launch, &'static str> {
        if rows == 0 || rows > i32::MAX as usize || batch == 0 || batch > 65535 {
            return Err("MXFP4 matrix launch outside ABI");
        }
        Ok(Fp4Launch { symbol: self.mx_mat, grid: (rows as u32, batch as u32, 1),
            block: (self.warp, self.matrix_warps, 1) })
    }
    pub(crate) fn mx_indexed(self, rows: usize, batch: usize, topk: usize) -> Result<Fp4Launch, &'static str> {
        let mut p = self.mx_matrix(rows, batch)?;
        if topk == 0 || topk > 65535 { return Err("MXFP4 indexed launch outside ABI"); }
        p.symbol = self.mx_moe;
        p.grid.2 = topk as u32;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_fp4_recipe_legacy_geometry() {
        let r = S31_FP4;
        for cc in [60, 61, 75, 80] {
            for n in [1, 511, 512, 513, 4096] {
                for k in [32, 1023, 1024, 4096] {
                    for batch in [1, 8, 9, 17, 65535] {
                        assert_eq!(r.auto_tiled(cc,n,k,batch), cc == 61 && n >= 512 && k >= 1024 && batch >= 9);
                        let reference = r.native_matrix(cc,false,n,batch).unwrap();
                        assert_eq!(reference.grid,(n as u32,batch as u32,1));
                        assert_eq!(reference.block,(32,4,1));
                        if cc >= 61 {
                            let tiled = r.native_matrix(cc,true,n,batch).unwrap();
                            assert_eq!(tiled.grid,(n.div_ceil(4) as u32,batch.div_ceil(8) as u32,1));
                            assert_eq!(tiled.block,(32,4,1));
                        } else { assert!(r.native_matrix(cc,true,n,batch).is_err()); }
                    }
                }
            }
        }
        assert_eq!(r.scalar_plan(129).unwrap().grid,(2,1,1));
        assert_eq!(r.cast_plan(257).unwrap().grid,(2,1,1));
        assert_eq!(r.quant_plan(512,9).unwrap().grid,(2,9,1));
        assert!(r.quant_plan(513,9).is_err());
        for batch in 1..=8 {
            let p = r.mx_vector(65,batch).unwrap();
            assert_eq!(p.symbol,format!("mul_mat_vec_mxfp4_q8_1_cuda{batch}"));
            assert_eq!(p.grid.0,if batch == 1 {65} else {33});
            assert_eq!(p.block.1,if batch <= 4 {4} else {2});
        }
        assert_eq!(r.mx_matrix(65,17).unwrap().grid,(65,17,1));
        assert_eq!(r.mx_indexed(65,17,3).unwrap().grid,(65,17,3));
        assert!(r.mx_matrix(65,65536).is_err());
        assert!(r.mx_vector(65,9).is_err());
        assert!(r.mx_indexed(65,17,65536).is_err());
    }
}
