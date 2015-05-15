!> @copyright Los Alamos National Laboratory 2015

!> Matrix addition for dense matrices.
module matrix_add_dense

  use matrix_type

  implicit none

contains

  !> Add two dense matrices.
  !!
  !! \f$ C \leftarrow A+B \f$
  !!
  !! @param A Matrix \f$ A \f$.
  !! @param B Matrix \f$ B \f$.
  !! @param C Matrix \f$ C \f$.
  subroutine add_dense (A, B, C)

    type(matrix_t), intent(in) :: A, B
    type(matrix_t), intent(inout) :: C

  end subroutine add_dense

end module matrix_add_dense
