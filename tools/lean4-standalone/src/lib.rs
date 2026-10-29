//! Standalone checked index contracts. Lean proves the corresponding arithmetic
//! model in lean/HaricotLeanContracts/Index.lean. The CUDA implementation and
//! its correspondence to this model are NOT formally verified in v0.1.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractError {
    EmptyExtent,
    IndexOutOfBounds,
    SizeOverflow,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ContractError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    pub rows: usize,
    pub cols: usize,
    pub elements: usize,
}

impl MatrixLayout {
    pub fn checked(rows: usize, cols: usize) -> Result<Self, ContractError> {
        if rows == 0 || cols == 0 {
            return Err(ContractError::EmptyExtent);
        }
        let elements = rows.checked_mul(cols).ok_or(ContractError::SizeOverflow)?;
        Ok(Self { rows, cols, elements })
    }

    /// Row-major linear index, using checked arithmetic and explicit bounds.
    pub fn index(&self, row: usize, col: usize) -> Result<usize, ContractError> {
        if row >= self.rows || col >= self.cols {
            return Err(ContractError::IndexOutOfBounds);
        }
        row.checked_mul(self.cols)
            .and_then(|base| base.checked_add(col))
            .filter(|&offset| offset < self.elements)
            .ok_or(ContractError::SizeOverflow)
    }

    /// Index in the transposed layout. This is an index calculation, not a
    /// claim of numerical equivalence to a particular CUDA kernel.
    pub fn transposed_index(&self, row: usize, col: usize) -> Result<usize, ContractError> {
        self.index(row, col)?;
        col.checked_mul(self.rows)
            .and_then(|base| base.checked_add(row))
            .filter(|&offset| offset < self.elements)
            .ok_or(ContractError::SizeOverflow)
    }
}

#[cfg(feature = "candle")]
pub mod candle_adapter {
    use super::MatrixLayout;
    use candle_core::{Error, Result, Tensor};

    /// A safe consumer of the standalone contract using Candle's existing
    /// CPU/CUDA backend dispatch. No ASD-Core or CUDA-specific kernel required.
    pub fn checked_transpose2d(tensor: &Tensor) -> Result<Tensor> {
        let (rows, cols) = tensor.dims2()?;
        MatrixLayout::checked(rows, cols).map_err(|error| Error::Msg(error.to_string()))?;
        tensor.transpose(0, 1)
    }
}

#[cfg(feature = "lean-ffi")]
pub mod native_candidate {
    // Stage-1 compile/link probe for argumentcomputer/lean-ffi. This does not
    // yet establish a Rust -> exported Lean function call or certify CUDA.
    pub use lean_ffi::object::{LeanBorrowed, LeanOwned, LeanShared};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_small_transposes_are_bijections() {
        for rows in 1..17 {
            for cols in 1..17 {
                let layout = MatrixLayout::checked(rows, cols).unwrap();
                let mut used = vec![false; layout.elements];
                for row in 0..rows {
                    for col in 0..cols {
                        assert!(layout.index(row, col).unwrap() < layout.elements);
                        let transposed = layout.transposed_index(row, col).unwrap();
                        assert!(!used[transposed]);
                        used[transposed] = true;
                        assert_eq!(transposed, col * rows + row);
                    }
                }
                assert!(used.into_iter().all(|v| v));
            }
        }
    }

    #[test]
    fn invalid_shapes_indices_and_overflow_fail_closed() {
        assert_eq!(MatrixLayout::checked(0, 3), Err(ContractError::EmptyExtent));
        assert_eq!(MatrixLayout::checked(2, 0), Err(ContractError::EmptyExtent));
        assert_eq!(MatrixLayout::checked(usize::MAX, 2), Err(ContractError::SizeOverflow));
        let layout = MatrixLayout::checked(3, 4).unwrap();
        assert_eq!(layout.index(3, 0), Err(ContractError::IndexOutOfBounds));
        assert_eq!(layout.transposed_index(0, 4), Err(ContractError::IndexOutOfBounds));
    }
}
