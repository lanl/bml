!> Normalize and calculate gershgorin bounds for a matrix
module bml_normalize_m

  implicit none

  private

  interface bml_normalize
       module procedure bml_normalize
  end interface

  interface bml_gershgorin
       module procedure bml_gershgorin
  end interface

  public :: bml_normalize
  public :: bml_gershgorin

contains

  !> Normalize matrix given gershgorin bounds.
  !!
  !! \ingroup normalize_group_F
  !!
  !! \param a Matrix
  !! \param maxeval Calculated max
  !! \param maxminusmin Calculated max-min
  subroutine bml_normalize(a, maxeval, maxminusmin)

    use bml_types_m
    use bml_c_interface_m

    type(bml_matrix_t), intent(inout) :: a
    double precision, intent(in) :: maxeval, maxminusmin

    call bml_normalize_C(a%ptr, maxeval, maxminusmin)

  end subroutine bml_normalize

  !> Calculate gershgorin bounds fro a matrix
  !!
  !! \ingroup normalize_group_F
  !!
  !! \param a Matrix
  !! \param a_gbnd Calculated max and max-min
  subroutine bml_gershgorin(a, a_gbnd)

    use bml_c_interface_m
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

end module bml_normalize_m
