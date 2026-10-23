import Init

/-!
A mathematical model of row-major tensor indexing and transposition.
This proves index bounds for natural-number arithmetic, NOT the correctness
of any particular CUDA/PTX program or its floating-point semantics.
-/

namespace HaricotLeanContracts

def rowMajorIndex (cols row col : Nat) : Nat := row * cols + col

theorem rowMajorIndex_lt
    (rows cols row col : Nat) (hrow : row < rows) (hcol : col < cols) :
    rowMajorIndex cols row col < rows * cols := by
  unfold rowMajorIndex
  calc
    row * cols + col < row * cols + cols := Nat.add_lt_add_left hcol _
    _ = (row + 1) * cols := by simp [Nat.add_mul]
    _ ≤ rows * cols := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hrow)

theorem transposeIndex_lt
    (rows cols row col : Nat) (hrow : row < rows) (hcol : col < cols) :
    rowMajorIndex rows col row < cols * rows := by
  exact rowMajorIndex_lt cols rows col row hcol hrow

end HaricotLeanContracts
