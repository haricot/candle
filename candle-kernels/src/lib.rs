include!(concat!(env!("OUT_DIR"), "/cuda_build_info.rs"));

pub mod asd_exact {
    include!(concat!(env!("OUT_DIR"), "/asd_exact_dispatch.rs"));
}

pub mod asd_exact_conv2d {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ExactConv2dCall {
        pub batch: usize,
        pub c_in: usize,
        pub c_out: usize,
        pub spatial0: usize,
        pub spatial1: usize,
        pub weight0: usize,
        pub weight1: usize,
        pub weight2: usize,
        pub weight3: usize,
        pub groups: usize,
        pub kernel: usize,
        pub stride: usize,
        pub padding: usize,
        pub dilation: usize,
        pub dtype: &'static str,
        pub input_contiguous: bool,
        pub input_start_offset: usize,
        pub weight_contiguous: bool,
        pub weight_start_offset: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ExactAsdMatch {
        pub policy_id: &'static str,
        pub decision_id: &'static str,
        pub state: &'static str,
        pub selected_backend: &'static str,
        pub evidence_sha256: &'static str,
        pub min_integrated_speedup_x: f64,
    }

    #[cfg(candle_asd_exact_v2)]
    pub const POLICY_EMBEDDED: bool = true;

    #[cfg(not(candle_asd_exact_v2))]
    pub const POLICY_EMBEDDED: bool = false;

    #[cfg(candle_asd_exact_v2)]
    pub const VALIDATION_BUILD: bool = super::asd_exact::VALIDATION_BUILD;

    #[cfg(not(candle_asd_exact_v2))]
    pub const VALIDATION_BUILD: bool = false;

    #[cfg(candle_asd_exact_v2)]
    pub fn lookup(call: ExactConv2dCall) -> Option<ExactAsdMatch> {
        let generated = super::asd_exact::lookup_conv2d(super::asd_exact::ExactConv2dCall {
            batch: call.batch,
            c_in: call.c_in,
            c_out: call.c_out,
            spatial0: call.spatial0,
            spatial1: call.spatial1,
            weight0: call.weight0,
            weight1: call.weight1,
            weight2: call.weight2,
            weight3: call.weight3,
            groups: call.groups,
            kernel: call.kernel,
            stride: call.stride,
            padding: call.padding,
            dilation: call.dilation,
            dtype: call.dtype,
            input_contiguous: call.input_contiguous,
            input_start_offset: call.input_start_offset,
            weight_contiguous: call.weight_contiguous,
            weight_start_offset: call.weight_start_offset,
        })?;
        Some(ExactAsdMatch {
            policy_id: generated.policy_id,
            decision_id: generated.decision_id,
            state: generated.state,
            selected_backend: generated.selected_backend,
            evidence_sha256: generated.evidence_sha256,
            min_integrated_speedup_x: generated.min_integrated_speedup_x,
        })
    }

    #[cfg(not(candle_asd_exact_v2))]
    pub fn lookup(_call: ExactConv2dCall) -> Option<ExactAsdMatch> {
        None
    }
}

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    AsdDw5x5,
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

pub const ALL_IDS: [Id; 13] = [
    Id::Affine,
    Id::AsdDw5x5,
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
mdl!(ASD_DW5X5, AsdDw5x5);
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
pub mod moe_selection;
