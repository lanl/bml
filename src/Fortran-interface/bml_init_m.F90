!> Initialize bml.
module bml_init_m

  use bml_c_interface_m
  use bml_types_m

#ifdef DO_MPI
  use mpi
#endif

  implicit none
  private

#ifdef DO_MPI
  public :: bml_initF
#endif

contains

#ifdef DO_MPI
  !> Initialize from Fortran when using MPI.
  !!
  !! \param Comm from Fortran
  subroutine bml_initF(fcomm)

    integer(C_INT), intent(in) ::fcomm

    call bml_initF_C(fcomm)

  end subroutine bml_initF
#endif

end module bml_init_m
