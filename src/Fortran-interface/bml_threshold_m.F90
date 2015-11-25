module bml_threshold_m
  use, intrinsic :: iso_c_binding
  use bml_types_m
  implicit none
  private

  interface

     subroutine bml_threshold_C(a, threshold) bind(C, name="bml_threshold")
       import :: C_PTR, C_DOUBLE
       type(C_PTR), value :: a
       real(C_DOUBLE), value, intent(in) :: threshold
     end subroutine bml_threshold_C

  end interface

  public :: bml_threshold

contains

  subroutine bml_threshold(a, threshold)

    type(bml_matrix_t), intent(inout) :: a
    real(C_DOUBLE), intent(in) :: threshold

    call bml_threshold_C(a%ptr, threshold)

  end subroutine bml_threshold

end module bml_threshold_m
