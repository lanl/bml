!> \copyright Los Alamos National Laboratory 2015

!> Matrix scaling for matrices.
module bml_scale_m

  implicit none

  private

  !> Scale a matrix.
  interface bml_scale
     module procedure scale_one
     module procedure scale_two
  end interface bml_scale

  public :: bml_scale

contains

  !> Scale a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  subroutine scale_one(alpha, a)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_m
    use bml_error_m
    use bml_scale_dense_m

    double precision, intent(in) :: alpha
    class(bml_matrix_t), intent(inout) :: a

    select type(a)
    type is(bml_matrix_dense_double_t)
       call scale_one_dense(alpha, a)
    class default
       call bml_error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine scale_one

  !> Scale a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  !! \param c The matrix
  subroutine scale_two(alpha, a, c)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_m
    use bml_error_m
    use bml_scale_dense_m

    double precision, intent(in) :: alpha
    class(bml_matrix_t), intent(in) :: a
    class(bml_matrix_t), allocatable, intent(out) :: c

    select type(a)
    type is(bml_matrix_dense_double_t)
       call bml_allocate(BML_MATRIX_DENSE, a%n, c)
       select type(c)
       type is(bml_matrix_dense_double_t)
          call scale_two_dense(alpha, a, c)
       end select
    class default
       call bml_error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine scale_two

end module bml_scale_m
