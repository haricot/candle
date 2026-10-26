import Init

/-!
A model of Candle's 2D strided *view* semantics, independent of data copying.
For a transposition, the underlying storage and start offset do not change:
only the two dimensions and the two strides are exchanged.

These theorems use unbounded natural numbers. They do not yet establish
the correspondence of Rust machine arithmetic or CUDA/PTX implementations.
-/

namespace HaricotLeanContracts

structure StridedLayout2D where
  rows : Nat
  cols : Nat
  rowStride : Nat
  colStride : Nat
  startOffset : Nat
deriving Repr

def StridedLayout2D.transpose (l : StridedLayout2D) : StridedLayout2D :=
  { rows := l.cols
    cols := l.rows
    rowStride := l.colStride
    colStride := l.rowStride
    startOffset := l.startOffset }

def StridedLayout2D.offset (l : StridedLayout2D) (row col : Nat) : Nat :=
  l.startOffset + row * l.rowStride + col * l.colStride

def StridedLayout2D.inBounds (l : StridedLayout2D) (row col : Nat) : Prop :=
  row < l.rows ∧ col < l.cols

def StridedLayout2D.validFor (l : StridedLayout2D) (storageLength : Nat) : Prop :=
  ∀ row col, l.inBounds row col → l.offset row col < storageLength

theorem StridedLayout2D.transpose_twice (l : StridedLayout2D) :
    l.transpose.transpose = l := by
  cases l
  rfl

theorem StridedLayout2D.transpose_inBounds (l : StridedLayout2D) (row col : Nat) :
    l.transpose.inBounds col row ↔ l.inBounds row col := by
  constructor
  · intro h
    exact ⟨h.2, h.1⟩
  · intro h
    exact ⟨h.2, h.1⟩

theorem StridedLayout2D.transpose_offset (l : StridedLayout2D) (row col : Nat) :
    l.transpose.offset col row = l.offset row col := by
  simp [StridedLayout2D.offset, StridedLayout2D.transpose,
    Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]

theorem StridedLayout2D.transpose_valid_iff (l : StridedLayout2D)
    (storageLength : Nat) :
    l.transpose.validFor storageLength ↔ l.validFor storageLength := by
  constructor
  · intro h row col hrc
    have htc : l.transpose.inBounds col row :=
      (l.transpose_inBounds row col).2 hrc
    have h' := h col row htc
    simpa only [StridedLayout2D.transpose_offset] using h'
  · intro h row col hrc
    have hoc : l.inBounds col row :=
      (l.transpose_inBounds col row).1 hrc
    have h' := h col row hoc
    simpa only [StridedLayout2D.transpose_offset] using h'

end HaricotLeanContracts
