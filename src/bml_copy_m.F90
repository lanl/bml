!> \copyright Los Alamos National Laboratory 2015

!> Copy operations on matrices.
module bml_copy_m
  implicit none
contains

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ A \leftarrow B \f$.
  !!
  !! \param A Matrix to copy to.
  !! \param B Matrix to copy.
  subroutine copy(A, B)

    use bml_type_dense_m
    use bml_allocate_m
    use bml_copy_dense_m
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(inout) :: A
    class(bml_matrix_t), intent(in) :: B

    select type(B)
    type is(bml_matrix_dense_t)
       call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, B%N, A)
       select type(A)
       type is(bml_matrix_dense_t)
          call copy_dense(A, B)
       class default
          call error(__FILE__, __LINE__, "unknown matrix type")
       end select
    class default
       call error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine copy

end module bml_copy_m
