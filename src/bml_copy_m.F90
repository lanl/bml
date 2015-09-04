!> \copyright Los Alamos National Laboratory 2015

!> Copy operations on matrices.
module bml_copy_m
  implicit none
contains

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ B \leftarrow A \f$.
  !!
  !! \param a Matrix to copy.
  !! \param b Matrix to copy to.
  subroutine bml_copy(a, b)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_m
    use bml_copy_dense_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: a
    class(bml_matrix_t), allocatable, intent(out) :: b

    select type(a)
    type is(bml_matrix_dense_single_t)
       call bml_allocate(BML_MATRIX_DENSE, a%n, b, BML_PRECISION_SINGLE)
       select type(b)
       type is(bml_matrix_dense_single_t)
          call bml_copy_dense(a, b)
       end select
    type is(bml_matrix_dense_double_t)
       call bml_allocate(BML_MATRIX_DENSE, a%n, b, BML_PRECISION_DOUBLE)
       select type(b)
       type is(bml_matrix_dense_double_t)
          call bml_copy_dense(a, b)
       end select
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine bml_copy

end module bml_copy_m
