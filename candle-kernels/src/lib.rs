pub mod sm61_exact_grouped_scope {
    include!(concat!(env!("OUT_DIR"), "/sm61_exact_grouped_scope.rs"));
}

include!(concat!(env!("OUT_DIR"), "/cuda_build_info.rs"));

mod sm61_ptx {
    include!(concat!(env!("OUT_DIR"), "/sm61_ptx.rs"));
}

pub mod asd_exact {
    include!(concat!(env!("OUT_DIR"), "/asd_exact_dispatch.rs"));
}

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
    GroupedTranspose,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 12] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::GroupedTranspose,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
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
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(GROUPED_TRANSPOSE, GroupedTranspose);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);

pub mod ffi;

/// Exact, device-scoped sm61 kernels; the V2 ASD policy selects their domain.
pub fn sm61_exact_grouped_ptx(candidate_id: &str) -> Option<&'static str> {
    match candidate_id {
        "ct1d-s32-g2-u1-b256" => Some(sm61_ptx::SM61_EXACT_GROUPED_K00),
        "ct1d-s32-g4-u1-b256" => Some(sm61_ptx::SM61_EXACT_GROUPED_K01),
        "ct1d-s32-g8-u1-b256" => Some(sm61_ptx::SM61_EXACT_GROUPED_K02),
        "ct1d-s32-g16-u1-b256" => Some(sm61_ptx::SM61_EXACT_GROUPED_K03),
        "ct2d-s32-g16-u4-b128" => Some(sm61_ptx::SM61_EXACT_GROUPED_K04),
        "ct2d-s32-g32-u4-b64" => Some(sm61_ptx::SM61_EXACT_GROUPED_K05),
        "gc1d-l128-g8-u1-b256" => Some(sm61_ptx::SM61_EXACT_GROUPED_K06),
        _ => None,
    }
}
