!> Matrix multiplication for dense matrices.
module matrix_multiply_dense

  use matrix_type

  implicit none

contains

  !> Multiply two dense matrices.
  !!
  !! \f$ C \leftarrow A+B \f$
  !!
  !! @param A Matrix \f$ A \f$.
  !! @param B Matrix \f$ B \f$.
  !! @param C Matrix \f$ C \f$.
  subroutine multiply_dense (A, B, C)

    type(matrix_t), intent(in) :: A, B
    type(matrix_t), intent(inout) :: C

  end subroutine multiply_dense

end module matrix_multiply_dense
