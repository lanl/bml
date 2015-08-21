!> \copyright Los Alamos National Laboratory 2015

!> Matrix scaling for matrices.
module bml_scale_m

  implicit none

  !> Scale a matrix.
  interface scale
     module procedure scale_one
     module procedure scale_two
  end interface scale

contains

  !> Scale a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param A The matrix
  subroutine scale_one(alpha, A)

    use bml_type_dense_m
    use bml_allocate_m
    use bml_error_m
    use bml_scale_dense_m

    double precision, intent(in) :: alpha
    class(bml_matrix_t), intent(inout) :: A

    select type(A)
    type is(bml_matrix_dense_t)
       call scale_one_dense(alpha, A)
    class default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine scale_one

  !> Scale a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param A The matrix
  !! \param C The matrix
  subroutine scale_two(alpha, A, C)

    use bml_type_dense_m
    use bml_allocate_m
    use bml_error_m
    use bml_scale_dense_m

    double precision, intent(in) :: alpha
    class(bml_matrix_t), intent(in) :: A
    class(bml_matrix_t), allocatable, intent(out) :: C

    select type(A)
    type is(bml_matrix_dense_t)
       call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, A%N, C)
       select type(C)
       type is(bml_matrix_dense_t)
          call scale_two_dense(alpha, A, C)
       end select
    class default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine scale_two

end module bml_scale_m
