module bml_add_m

  implicit none

  private

  !> \addtogroup add_group_Fortran
  !! @{

  !> Add two matrices.
  interface bml_add
     module procedure add_two_single_real
     module procedure add_three_single_real
  end interface bml_add

  !> Add identity matrix to a matrix.
  interface bml_add_identity
     module procedure add_identity_one_single_real
     module procedure add_identity_two_single_real
  end interface bml_add_identity
  !> @}

  public :: bml_add
  public :: bml_add_identity

end module bml_add_m
