module bml_threshold_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_threshold

contains

  subroutine bml_threshold(a, threshold)

    type(bml_matrix_t), intent(inout) :: a
    real(C_DOUBLE), intent(in) :: threshold

    call bml_threshold_C(a%ptr, threshold)

  end subroutine bml_threshold

end module bml_threshold_m
