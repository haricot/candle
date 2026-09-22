//! Operation-keyed ASD decision contract shared by candle-core and candle-nn.
//! No model identifier, CUDA ordinal, GPU UUID, or implicit promotion appears in a key.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Conv1d,
    ConvTranspose1d,
    ConvTranspose2d,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperationKey {
    pub op: Operation,
    pub input: [usize; 4],
    pub weight: [usize; 4],
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl OperationKey {
    /// Input [N,C,L], weight [Cout,Cin/groups,K].
    pub fn conv1d(input: [usize; 3], weight: [usize; 3], padding: usize,
                  stride: usize, dilation: usize, groups: usize) -> Self {
        Self {
            op: Operation::Conv1d,
            input: [input[0], input[1], input[2], 1],
            weight: [weight[0], weight[1], weight[2], 1],
            padding, output_padding: 0, stride, dilation, groups,
        }
    }

    /// Input [N,Cin,L], weight [Cin,Cout/groups,K].
    pub fn conv_transpose1d(input: [usize; 3], weight: [usize; 3],
                            padding: usize, output_padding: usize,
                            stride: usize, dilation: usize, groups: usize) -> Self {
        Self {
            op: Operation::ConvTranspose1d,
            input: [input[0], input[1], input[2], 1],
            weight: [weight[0], weight[1], weight[2], 1],
            padding, output_padding, stride, dilation, groups,
        }
    }

    /// Input [N,Cin,H,W], weight [Cin,Cout/groups,Kh,Kw].
    pub fn conv_transpose2d(input: [usize; 4], weight: [usize; 4],
                            padding: usize, output_padding: usize,
                            stride: usize, dilation: usize, groups: usize) -> Self {
        Self {
            op: Operation::ConvTranspose2d,
            input, weight, padding, output_padding, stride, dilation, groups,
        }
    }
}

/// A route's evidence status is not inferred from its geometry or device class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceStatus {
    CandidateOnly,
    DeviceScopedEvidence,
    IndependentlyPromoted,
}

/// Both embedded kernel tables and externally loaded candidate rules use this
/// exact-operation matching function. Hardware, dtype, layout and opt-in
/// checks remain independent mandatory guards at the respective call sites.
pub fn first_exact<'a, T>(
    observed: &OperationKey,
    decisions: &'a [T],
    key_of: impl Fn(&T) -> Option<OperationKey>,
) -> Option<&'a T> {
    decisions.iter().find(|decision| key_of(decision).as_ref() == Some(observed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_geometry_cannot_match_ungrouped() {
        let grouped = OperationKey::conv1d([1, 64, 128], [64, 8, 3], 1, 1, 1, 8);
        let ungrouped = OperationKey::conv1d([1, 64, 128], [64, 64, 3], 1, 1, 1, 1);
        assert_ne!(grouped, ungrouped);
        assert!(first_exact(&ungrouped, &[grouped], |key| Some(*key)).is_none());
        assert!(first_exact(&grouped, &[grouped], |key| Some(*key)).is_some());
    }

    #[test]
    fn transpose_and_forward_are_distinct_operations() {
        let fwd = OperationKey::conv1d([1, 64, 128], [64, 8, 3], 1, 1, 1, 8);
        let trans = OperationKey::conv_transpose1d([1, 64, 128], [64, 8, 3], 1, 0, 1, 1, 8);
        assert_ne!(fwd, trans);
    }

    #[test]
    fn padding_and_output_padding_cannot_alias() {
        let a = OperationKey::conv_transpose1d([1, 128, 32], [128, 16, 3], 1, 0, 2, 1, 8);
        let b = OperationKey::conv_transpose1d([1, 128, 32], [128, 16, 3], 1, 1, 2, 1, 8);
        assert_ne!(a, b);
    }
}
