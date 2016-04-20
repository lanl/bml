!> Shutdown bml.
module bml_shutdown_m

  use bml_c_interface_m
  use bml_types_m

#ifdef DO_MPI
  use mpi
#endif

  implicit none
  private

  public :: bml_shutdownF

contains

  !> Shutdown from Fortran when using MPI.
  !!
  subroutine bml_shutdownF()

    call bml_shutdownF_C()

  end subroutine bml_shutdownF

end module bml_shutdown_m
