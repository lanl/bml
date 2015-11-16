module bml_threshold_m

  implicit none
  private

  interface

     subroutine bml_threshold_C(a, threshold) bind(C, name="bml_threshold")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value :: a
       real(C_DOUBLE), value, intent(in) :: threshold
     end subroutine bml_threshold_C

  end interface

  public :: bml_threshold

contains

  subroutine bml_threshold(a, threshold)

    use bml_types_m

    type(bml_matrix_t), intent(inout) :: a
    double precision, intent(in) :: threshold

    call bml_threshold_C(a%ptr, threshold)

  end subroutine bml_threshold

end module bml_threshold_m
