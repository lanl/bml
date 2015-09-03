!> \copyright Los Alamos National Laboratory 2015

!> Copy operations on matrices.
module bml_copy_m
  implicit none
contains

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ B \leftarrow A \f$.
  !!
  !! \param A Matrix to copy.
  !! \param B Matrix to copy to.
  subroutine bml_copy(A, B)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_m
    use bml_copy_dense_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: A
    class(bml_matrix_t), pointer, intent(inout) :: B

    select type(A)
    type is(bml_matrix_dense_double_t)
       call allocate_matrix(BML_MATRIX_DENSE, A%N, B)
       select type(B)
       type is(bml_matrix_dense_double_t)
          call copy_dense(A, B)
       class default
          call error(__FILE__, __LINE__, "unknown matrix type")
       end select
    class default
       call error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine bml_copy

end module bml_copy_m
