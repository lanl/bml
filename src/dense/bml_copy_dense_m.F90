!> \copyright Los Alamos National Laboratory 2015

!> Copy operations on matrices.
module bml_copy_dense_m

  implicit none

  private

  !> Copy operations.
  interface bml_copy_dense
     module procedure copy_dense_single
     module procedure copy_dense_double
  end interface bml_copy_dense

  public :: bml_copy_dense

contains

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ B \leftarrow A \f$.
  !!
  !! \param a Matrix to copy.
  !! \param b Matrix to copy to.
  subroutine copy_dense_single(a, b)

    use bml_type_dense_m

    type(bml_matrix_dense_single_t), intent(in) :: a
    type(bml_matrix_dense_single_t), intent(inout) :: b

    b%matrix = a%matrix

  end subroutine copy_dense_single

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ B \leftarrow A \f$.
  !!
  !! \param a Matrix to copy.
  !! \param b Matrix to copy to.
  subroutine copy_dense_double(a, b)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: a
    type(bml_matrix_dense_double_t), intent(inout) :: b

    b%matrix = a%matrix

  end subroutine copy_dense_double

end module bml_copy_dense_m
