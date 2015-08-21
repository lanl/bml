!> \copyright Los Alamos National Laboratory 2015

!> Copy operations on matrices.
module bml_copy_dense_m
  implicit none
contains

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ B \leftarrow A \f$.
  !!
  !! \param A Matrix to copy.
  !! \param B Matrix to copy to.
  subroutine copy_dense(A, B)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: A
    type(bml_matrix_dense_double_t), intent(inout) :: B

    ! This is an implicit re-allocation.
    B%matrix = A%matrix

  end subroutine copy_dense

end module bml_copy_dense_m
