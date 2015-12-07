!> Calculate gershoring bounds for a matrix
module bml_gershgorin_m
  use bml_c_interface_m
  use bml_types_m
  implicit none
  private


  public :: bml_gershgorin

contains

  !> Calculate gershgorin bounds fro a matrix
  !!
  !! \ingroup gershgorin_group_F
  !!
  !! \param a Matrix
  !! \param a_gbnd Calculated max and max-min
  subroutine bml_gershgorin(a, a_gbnd)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a

    double precision, allocatable, intent(inout) :: a_gbnd(:)

    type(C_PTR) :: ag_ptr
    double precision, pointer :: a_gbnd_ptr(:)

    ag_ptr = bml_gershgorin_C(a%ptr)
    call c_f_pointer(ag_ptr, a_gbnd_ptr, [2])
    a_gbnd = a_gbnd_ptr

    deallocate(a_gbnd_ptr)

  end subroutine bml_gershgorin

end module bml_gershgorin_m
