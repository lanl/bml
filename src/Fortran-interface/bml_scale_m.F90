!> Matrix scaling for matrices.
module bml_scale_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  !> Scale a matrix.
  interface bml_scale
     module procedure scale_one
     module procedure scale_two
    !  module procedure bml_scale_cmplx
  end interface bml_scale

  public :: bml_scale, bml_scale_cmplx

contains

  !> Scale a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  subroutine scale_one(alpha, a)

    real(C_DOUBLE), intent(in) :: alpha
    type(bml_matrix_t), intent(inout) :: a

    call bml_scale_inplace_C(alpha, a%ptr)

  end subroutine scale_one

  !> Scale a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  !! \param c The matrix
  subroutine scale_two(alpha, a, c)

    real(C_DOUBLE), intent(in) :: alpha
    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: c

    call bml_scale_C(alpha, a%ptr, c%ptr)

  end subroutine scale_two

  !> Scale a bml matrix by a complex parameter.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The complex factor
  !! \param a The matrix
  !! \param c The matrix
  subroutine bml_scale_cmplx(alpha, a)

    complex(C_DOUBLE_COMPLEX), target, intent(in) :: alpha
    type(bml_matrix_t), intent(inout) :: a

    call bml_scale_cmplx_c(c_loc(alpha), a%ptr)

  end subroutine bml_scale_cmplx

end module bml_scale_m
