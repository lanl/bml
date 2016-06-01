!> MPI parallel communication.
module bml_parallel_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_allGatherVParallel

contains

  !> Multiply two matrices.
  !!
  !!
  !! \param a Matrix to be distributed.
  subroutine bml_allGatherVParallel(a)

    type(bml_matrix_t), intent(inout) :: a

    call bml_allGatherVParallel_C(a%ptr)

  end subroutine bml_allGatherVParallel

end module bml_parallel_m
