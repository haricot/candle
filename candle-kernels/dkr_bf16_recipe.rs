//! BF16 host dispatch contract, independent of tensor storage and CUDA APIs.
//! The S3.1 package implements BF16 operations in software and uses exact
//! BF16 -> F32 casts, cuBLAS SGEMM and a single BF16 rounding for GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bf16MatmulRoute {
    Native,
    WidenF32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bf16Recipe {
    pub(crate) matmul: Bf16MatmulRoute,
    pub(crate) cast_to_f32: &'static str,
    pub(crate) cast_from_f32: &'static str,
}

pub(crate) const S31_BF16: Bf16Recipe = Bf16Recipe {
    matmul: Bf16MatmulRoute::WidenF32,
    cast_to_f32: "cast_bf16_f32",
    cast_from_f32: "cast_f32_bf16",
};

impl Bf16Recipe {
    pub(crate) fn route_id(self) -> &'static str {
        match self.matmul {
            Bf16MatmulRoute::Native => "cublas-native-bf16",
            Bf16MatmulRoute::WidenF32 => "cublas-sgemm-f32-round-bf16",
        }
    }

    pub(crate) fn cast_symbols(self) -> [&'static str; 2] {
        [self.cast_to_f32, self.cast_from_f32]
    }
}

/// Historical static compatibility policy, ONLY used without a native
/// catalogue. Preserve the feature's existing pre-Ampere boundary.
pub(crate) fn legacy_static_matmul(cc: u32) -> Bf16MatmulRoute {
    if cc < 80 { Bf16MatmulRoute::WidenF32 } else { Bf16MatmulRoute::Native }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_bf16_recipe_legacy_routes() {
        for cc in [50, 60, 61, 70, 75, 79] {
            assert_eq!(legacy_static_matmul(cc), Bf16MatmulRoute::WidenF32);
        }
        for cc in [80, 86, 89, 90, 100, 120] {
            assert_eq!(legacy_static_matmul(cc), Bf16MatmulRoute::Native);
        }
        assert_eq!(S31_BF16.matmul, Bf16MatmulRoute::WidenF32);
        assert_eq!(S31_BF16.route_id(), "cublas-sgemm-f32-round-bf16");
        assert_eq!(S31_BF16.cast_symbols(), ["cast_bf16_f32", "cast_f32_bf16"]);
    }
}
