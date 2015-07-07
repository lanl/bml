!> \copyright Los Alamos National Laboratory 2015

!> Matrix scaling for matrices.
module bml_scale_m
  implicit none
contains

  !> Scale a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param A The matrix
  !! \param C The matrix
  subroutine scale(alpha, A, C)

    use bml_type_dense

    use bml_allocate_m
    use bml_error_m
    use bml_scale_dense

    double precision, intent(in) :: alpha
    class(bml_matrix_t), allocatable, intent(in) :: A
    class(bml_matrix_t), allocatable, intent(out) :: C

    if(.not. allocated(A)) then
       call error(__FILE__, __LINE__, "A is not allocated")
    endif

    select type(A)
    type is(bml_matrix_dense_t)
       call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, A%N, C)
       select type(C)
       type is(bml_matrix_dense_t)
          call scale_dense(alpha, A, C)
       end select
    class default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine scale

end module bml_scale_m
