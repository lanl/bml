!> Matrix scaling for matrices.
module bml_scale_m

  implicit none

  private

  !> Scale a matrix.
  interface bml_scale
     module procedure scale_one
     module procedure scale_two
  end interface bml_scale

  interface

     subroutine bml_scale_C (alpha, a, b) bind(C, name="bml_scale")
       use, intrinsic :: iso_C_binding
       real(C_DOUBLE), value, intent(in) :: alpha
       type(C_PTR), value :: a
       type(C_PTR), value :: b
     end subroutine bml_scale_C

     subroutine bml_scale_inplace_C (alpha, a) bind(C, name="bml_scale_inplace")
       use, intrinsic :: iso_C_binding
       real(C_DOUBLE), value, intent(in) :: alpha
       type(C_PTR), value :: a
     end subroutine bml_scale_inplace_C

  end interface

  public :: bml_scale

contains

  !> Scale a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  subroutine scale_one(alpha, a)

    use bml_types_m

    double precision, intent(in) :: alpha
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

    use bml_types_m

    double precision, intent(in) :: alpha
    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: c

    call bml_scale_C(alpha, a%ptr, c%ptr)

  end subroutine scale_two

end module bml_scale_m
