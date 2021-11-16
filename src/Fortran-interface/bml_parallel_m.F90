!> MPI parallel communication.
module bml_parallel_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_allGatherVParallel
  public :: bml_getNRanks
  public :: bml_getMyRank

contains

  !> Multiply two matrices.
  !!
  !!
  !! \param a Matrix to be distributed.
  subroutine bml_allGatherVParallel(a)

    type(bml_matrix_t), intent(inout) :: a

    call bml_allGatherVParallel_C(a%ptr)

  end subroutine bml_allGatherVParallel

  function bml_getNRanks()

    integer :: bml_getNRanks

    bml_getNRanks = bml_getNRanks_C()

  end function bml_getNRanks

  function bml_getMyRank()

    integer :: bml_getMyRank

    bml_getMyRank = bml_getMyRank_C()

  end function bml_getMyRank

end module bml_parallel_m
