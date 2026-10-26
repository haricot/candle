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


/**
 * The 2D part of Candle's Layout (dimensions, strides and start offset).
 * This represents a view over existing storage, not a dense transposed copy.
 *
 * The Lean StridedLayout2D model has exactly these five nonnegative fields.
 * Rust/Lean semantic equivalence is not yet machine-proved; this module
 * supplies checked machine arithmetic and an executable Candle conformance gate.
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StridedMatrixLayout {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
    pub start_offset: usize,
}

impl StridedMatrixLayout {
    pub fn transpose(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            start_offset: self.start_offset,
        }
    }

    pub fn physical_offset(&self, row: usize, col: usize) -> Result<usize, ContractError> {
        if row >= self.rows || col >= self.cols {
            return Err(ContractError::IndexOutOfBounds);
        }
        let row_part = row
            .checked_mul(self.row_stride)
            .ok_or(ContractError::SizeOverflow)?;
        let col_part = col
            .checked_mul(self.col_stride)
            .ok_or(ContractError::SizeOverflow)?;
        self.start_offset
            .checked_add(row_part)
            .and_then(|offset| offset.checked_add(col_part))
            .ok_or(ContractError::SizeOverflow)
    }

    pub fn offset_in_storage(
        &self,
        row: usize,
        col: usize,
        storage_len: usize,
    ) -> Result<usize, ContractError> {
        let offset = self.physical_offset(row, col)?;
        if offset >= storage_len {
            return Err(ContractError::IndexOutOfBounds);
        }
        Ok(offset)
    }
}

#[cfg(feature = "candle")]
pub mod candle_adapter {
    use super::{MatrixLayout, StridedMatrixLayout};
    use candle_core::{Error, Result, Tensor};

    pub fn model_layout2d(tensor: &Tensor) -> Result<StridedMatrixLayout> {
        let (rows, cols) = tensor.dims2()?;
        let layout = tensor.layout();
        Ok(StridedMatrixLayout {
            rows,
            cols,
            row_stride: layout.stride()[0],
            col_stride: layout.stride()[1],
            start_offset: layout.start_offset(),
        })
    }

    /// An executable conformance gate against Candle's existing transpose.
    /// It checks the real output layout, not CUDA kernel correctness.
    pub fn checked_transpose2d(tensor: &Tensor) -> Result<Tensor> {
        let (rows, cols) = tensor.dims2()?;
        MatrixLayout::checked(rows, cols).map_err(|error| Error::Msg(error.to_string()))?;
        let expected = model_layout2d(tensor)?.transpose();
        let output = tensor.transpose(0, 1)?;
        if model_layout2d(&output)? != expected {
            return Err(Error::Msg("Candle transpose layout differs from verified model".into()));
        }
        Ok(output)
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

    #[test]
    fn strided_offsets_are_invariant_under_view_transpose() {
        for rows in 1..8 {
            for cols in 1..8 {
                for row_stride in 0..5 {
                    for col_stride in 0..5 {
                        let layout = StridedMatrixLayout {
                            rows, cols, row_stride, col_stride, start_offset: 3,
                        };
                        let transposed = layout.transpose();
                        assert_eq!(transposed.transpose(), layout);
                        for row in 0..rows {
                            for col in 0..cols {
                                assert_eq!(
                                    layout.physical_offset(row, col),
                                    transposed.physical_offset(col, row)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn strided_model_distinguishes_view_from_dense_transpose() {
        let original = StridedMatrixLayout {
            rows: 2, cols: 3, row_stride: 3, col_stride: 1, start_offset: 0,
        };
        let transposed = original.transpose();
        assert_eq!(original.physical_offset(0, 1), Ok(1));
        assert_eq!(transposed.physical_offset(1, 0), Ok(1));
        assert_eq!(MatrixLayout::checked(2, 3).unwrap().transposed_index(0, 1), Ok(2));
        assert_eq!(transposed.offset_in_storage(1, 0, 6), Ok(1));
        assert_eq!(
            transposed.offset_in_storage(1, 0, 1),
            Err(ContractError::IndexOutOfBounds)
        );
        assert_eq!(
            transposed.physical_offset(3, 0),
            Err(ContractError::IndexOutOfBounds)
        );
        let overflow = StridedMatrixLayout {
            rows: 3, cols: 2, row_stride: usize::MAX, col_stride: 1, start_offset: 1,
        };
        assert_eq!(overflow.physical_offset(2, 0), Err(ContractError::SizeOverflow));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn strided_model_matches_candle_cpu_for_contiguous_and_noncontiguous_views() {
        use candle_core::{Device, Tensor};
        let source = Tensor::new(&[[0u32, 1, 2], [3, 4, 5]], &Device::Cpu).unwrap();
        let model = candle_adapter::model_layout2d(&source).unwrap();
        assert_eq!(model.row_stride, 3);
        assert_eq!(model.col_stride, 1);
        let transposed = candle_adapter::checked_transpose2d(&source).unwrap();
        assert_eq!(candle_adapter::model_layout2d(&transposed).unwrap(), model.transpose());
        assert_eq!(transposed.to_vec2::<u32>().unwrap(), vec![vec![0, 3], vec![1, 4], vec![2, 5]]);
        let twice = candle_adapter::checked_transpose2d(&transposed).unwrap();
        assert_eq!(candle_adapter::model_layout2d(&twice).unwrap(), model);
        assert_eq!(twice.to_vec2::<u32>().unwrap(), source.to_vec2::<u32>().unwrap());
    }

}
