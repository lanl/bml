!> Adjungate_triangle functions.
module bml_adjungate_triangle_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_adjungate_triangle

contains

  !> Adjungates the triangle of a matrix.
  !!
  !! \param a  The matrix.
  !! \param triangle  Which triangle to adjungate ('u': upper, 'l': lower)
  subroutine bml_adjungate_triangle(a, triangle)

    type(bml_matrix_t), intent(inout) :: a
    character(C_CHAR), intent(in) :: triangle
    
    call bml_adjungate_triangle_C(a%ptr, triangle)

  end subroutine bml_adjungate_triangle

end module bml_adjungate_triangle_m
