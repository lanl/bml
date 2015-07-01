!> @copyright Los Alamos National Laboratory 2015

!> Matrix multiplication for dense matrices.
module bml_multiply_dense

  use bml_type_dense

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

    type(bml_matrix_dense_t), intent(in) :: A, B
    type(bml_matrix_dense_t), intent(inout) :: C

    C%matrix = matmul(A%matrix, B%matrix)

  end subroutine multiply_dense

end module bml_multiply_dense
