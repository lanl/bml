!> Adjungate_triangle functions.
module bml_adjungate_triangle_m

  implicit none
  private

  interface

    subroutine bml_adjungate_triangle_C(a, triangle) &
        & bind(C, name="bml_adjungate_triangle")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value :: a
       character(C_CHAR), value, intent(in) :: triangle
     end subroutine bml_adjungate_triangle_C

  end interface

  public :: bml_adjungate_triangle

contains

  !> Adjungates the triangle of a matrix.
  !!
  !! \param a  The matrix.
  !! \param triangle  Which triangle to adjungate ('u': upper, 'l': lower)
  subroutine bml_adjungate_triangle(a, triangle)

    use bml_types_m

    type(bml_matrix_t), intent(inout) :: a
    character(C_CHAR), intent(in) :: triangle

    call bml_adjungate_triangle_C(a%ptr, triangle)

  end subroutine bml_adjungate_triangle

end module bml_adjungate_triangle_m
