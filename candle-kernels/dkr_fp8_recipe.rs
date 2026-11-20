//! Pure host-launch ABI for the F8E native package.
//! Shared by the catalogue producer, build verifier and generic executor.
//! The S3.1 descriptor below is a frozen legacy package adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp8Recipe {
    pub(crate) simt_gemv: &'static str,
    pub(crate) simt_prefill: &'static str,
    pub(crate) dp4a_gemv: &'static str,
    pub(crate) dp4a_prefill: &'static str,
    pub(crate) quantize: &'static str,
    pub(crate) prefill_tokens: usize,
    pub(crate) rows_per_block: usize,
    pub(crate) threads: u32,
    pub(crate) k_alignment: usize,
    pub(crate) groups: [usize; 3],
}

/// F8E-v1 checkpoint/packing contract; independent of launch tile choice.
pub(crate) const F8E_V1_K_ALIGNMENT: usize = 128;

pub(crate) const S31_FP8: Fp8Recipe = Fp8Recipe {
    simt_gemv: "fp8_sm61_w8a16_gemv",
    simt_prefill: "fp8_sm61_w8a16_prefill4",
    dp4a_gemv: "fp8_sm61_dp4a_gemv",
    dp4a_prefill: "fp8_sm61_dp4a_prefill4",
    quantize: "fp8_sm61_quant_x",
    prefill_tokens: 4,
    rows_per_block: 4,
    threads: 128,
    k_alignment: F8E_V1_K_ALIGNMENT,
    groups: [32, 64, 128],
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp8Launch {
    pub(crate) symbol: &'static str,
    pub(crate) grid: (u32, u32, u32),
    pub(crate) block: (u32, u32, u32),
}

impl Fp8Recipe {
    pub(crate) fn symbols(self) -> [&'static str; 5] {
        [self.simt_gemv, self.simt_prefill, self.dp4a_gemv, self.dp4a_prefill, self.quantize]
    }

    pub(crate) fn matrix_symbol(self, dp4a: bool, tokens: usize) -> (&'static str, usize) {
        match (dp4a, tokens >= self.prefill_tokens) {
            (false, false) => (self.simt_gemv, 1),
            (false, true) => (self.simt_prefill, self.prefill_tokens),
            (true, false) => (self.dp4a_gemv, 1),
            (true, true) => (self.dp4a_prefill, self.prefill_tokens),
        }
    }

    pub(crate) fn matrix(self, dp4a: bool, tokens: usize, rows: usize) -> Result<Fp8Launch, &'static str> {
        if tokens == 0 || tokens > 65535 || rows == 0 || rows > i32::MAX as usize {
            return Err("F8E matrix launch dimensions outside ABI");
        }
        let (symbol, tile) = self.matrix_symbol(dp4a, tokens);
        Ok(Fp8Launch {
            symbol,
            grid: (rows.div_ceil(self.rows_per_block) as u32, tokens.div_ceil(tile) as u32, 1),
            block: (self.threads, 1, 1),
        })
    }

    pub(crate) fn activation(self, tokens: usize, k: usize, group: usize) -> Result<Fp8Launch, &'static str> {
        if tokens == 0 || tokens > 65535 || k == 0 || !self.groups.contains(&group)
            || k % group != 0 || k % self.k_alignment != 0 {
            return Err("F8E activation launch dimensions outside ABI");
        }
        let elements = tokens.checked_mul(k).filter(|&v| v <= i32::MAX as usize)
            .ok_or("F8E activation launch overflow")?;
        Ok(Fp8Launch {
            symbol: self.quantize,
            grid: ((elements / group) as u32, 1, 1),
            block: (self.threads, 1, 1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_fp8_recipe_legacy_geometry() {
        let r = S31_FP8;
        for tokens in [1, 2, 3, 4, 5, 8, 16, 127, 65535] {
            for rows in [1, 4, 5, 127, 4096] {
                for dp4a in [false, true] {
                    let plan = r.matrix(dp4a, tokens, rows).unwrap();
                    let expected = match (dp4a, tokens >= 4) {
                        (false, false) => "fp8_sm61_w8a16_gemv",
                        (false, true) => "fp8_sm61_w8a16_prefill4",
                        (true, false) => "fp8_sm61_dp4a_gemv",
                        (true, true) => "fp8_sm61_dp4a_prefill4",
                    };
                    assert_eq!(plan.symbol, expected);
                    assert_eq!(plan.grid, (rows.div_ceil(4) as u32,
                        if tokens >= 4 { tokens.div_ceil(4) as u32 } else { tokens as u32 }, 1));
                    assert_eq!(plan.block, (128, 1, 1));
                }
            }
        }
        for group in [32, 64, 128] {
            let plan = r.activation(17, 1024, group).unwrap();
            assert_eq!(plan.grid, ((17 * 1024 / group) as u32, 1, 1));
            assert_eq!(plan.block, (128, 1, 1));
            assert_eq!(plan.symbol, "fp8_sm61_quant_x");
        }
        assert!(r.activation(1, 128, 0).is_err());
        assert!(r.activation(1, 129, 32).is_err());
        assert!(r.activation(65535, 131072, 128).is_err());
        assert!(r.matrix(false, 65536, 1).is_err());
        assert!(r.matrix(true, 1, 0).is_err());
    }
}
