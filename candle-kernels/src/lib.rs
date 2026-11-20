#[cfg(candle_builtin_ptx)]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: Option<&'static str>,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    /// Static payload, if this feature configuration built one.
    pub fn static_ptx(&self) -> Option<&'static str> {
        self.ptx
    }

    /// Compatibility accessor for callers using the static CUDA build.
    /// Catalogue consumers should resolve the module identifier via DKR.
    pub fn ptx(&self) -> &'static str {
        self.static_ptx().expect("built-in PTX omitted in DKR catalogue mode")
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            #[cfg(candle_builtin_ptx)]
            ptx: Some(ptx::$cst),
            #[cfg(not(candle_builtin_ptx))]
            ptx: None,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);

/// Build-time evidence: no static PTX payload is linked in the decoupled mode.
pub const BUILTIN_PTX_ENABLED: bool = cfg!(candle_builtin_ptx);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_builtin_ptx_inventory() {
        let modules = [AFFINE, BINARY, CAST, CONV, FILL, INDEXING, QUANTIZED,
            REDUCE, SORT, TERNARY, UNARY];
        for (index, module) in modules.iter().enumerate() {
            assert_eq!(module.index(), index);
            assert_eq!(module.static_ptx().is_some(), BUILTIN_PTX_ENABLED);
            if let Some(ptx) = module.static_ptx() { assert!(!ptx.is_empty()); }
        }
        let expected = !cfg!(feature = "cuda-dkr-catalog") || cfg!(any(
            feature = "cuda-legacy-bf16", feature = "cuda-legacy-fp8", feature = "cuda-legacy-fp4"));
        assert_eq!(BUILTIN_PTX_ENABLED, expected);
        assert_eq!(STATIC_FFI_ENABLED, expected);
    }
}

#[path = "../dkr_s31_contract.rs"]
pub mod dkr_s31_contract;

/// Static CUDA host launchers are not linked in native catalogue mode.
pub const STATIC_FFI_ENABLED: bool = cfg!(candle_static_ffi);
#[cfg(candle_static_ffi)]
pub mod ffi;
#[cfg(candle_static_ffi)]
pub mod moe_selection;
