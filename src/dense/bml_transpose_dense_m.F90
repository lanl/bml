!> \copyright Los Alamos National Laboratory 2015

!> Matrix transpose functions.
module bml_transpose_dense_m

  implicit none

  private

  !> Matrix transpose functions.
  interface bml_transpose_dense
     module procedure bml_transpose_dense_single
     module procedure bml_transpose_dense_double
  end interface bml_transpose_dense

  public :: bml_transpose_dense

contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @return a_t The transpose.
  subroutine bml_transpose_dense_single(a, a_t)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_dense_m

    class(bml_matrix_dense_single_t), intent(in) :: a
    class(bml_matrix_dense_single_t), intent(inout) :: a_t

    a_t%matrix = transpose(a%matrix)

  end subroutine bml_transpose_dense_single

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @return a_t The transpose.
  subroutine bml_transpose_dense_double(a, a_t)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_dense_m

    class(bml_matrix_dense_double_t), intent(in) :: a
    class(bml_matrix_dense_double_t), intent(inout) :: a_t

    a_t%matrix = transpose(a%matrix)

  end subroutine bml_transpose_dense_double

end module bml_transpose_dense_m
