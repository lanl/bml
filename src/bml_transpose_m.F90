!> \copyright Los Alamos National Laboratory 2015

!> Tranpose functions.
module bml_transpose_m
  implicit none
contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @return a_t The transpose.
  subroutine bml_transpose(a, a_t)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_m
    use bml_transpose_dense_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: a
    class(bml_matrix_t), allocatable, intent(out) :: a_t

    select type(a)
    class is(bml_matrix_dense_single_t)
       call bml_allocate(BML_MATRIX_DENSE, a%n, a_t, BML_PRECISION_SINGLE)
       select type(a_t)
       type is (bml_matrix_dense_single_t)
          call bml_transpose_dense(a, a_t)
       end select
    class is(bml_matrix_dense_double_t)
       call bml_allocate(BML_MATRIX_DENSE, a%n, a_t, BML_PRECISION_DOUBLE)
       select type(a_t)
       type is (bml_matrix_dense_double_t)
          call bml_transpose_dense(a, a_t)
       end select
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine bml_transpose

end module bml_transpose_m
