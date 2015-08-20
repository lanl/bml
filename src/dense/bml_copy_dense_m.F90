!> \copyright Los Alamos National Laboratory 2015

!> Copy operations on matrices.
module bml_copy_dense_m
  implicit none
contains

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ A \leftarrow B \f$.
  !!
  !! \param A Matrix to copy to.
  !! \param B Matrix to copy.
  subroutine copy_dense(A, B)

    use bml_type_dense_m

    type(bml_matrix_dense_t), intent(inout) :: A
    type(bml_matrix_dense_t), intent(in) :: B

    ! This is an implicit re-allocation.
    A%matrix = B%matrix

  end subroutine copy_dense

end module bml_copy_dense_m
