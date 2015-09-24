module bml_add_m

  implicit none

  private

  !> \addtogroup add_group
  !! @{

  !> Add two matrices.
  interface bml_add
  end interface bml_add

  !> Add identity matrix to a matrix.
  interface bml_add_identity
  end interface bml_add_identity
  !> @}

  public :: bml_add
  public :: bml_add_identity

contains

end module bml_add_m
