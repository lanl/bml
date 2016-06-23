!> Submatrix routines used for graph partitioning
module bml_submatrix_m

  use bml_types_m
  use bml_c_interface_m

  implicit none
  private

  public :: bml_matrix2submatrix_index
  public :: bml_matrix2submatrix
  public :: bml_submatrix2matrix
  public :: bml_adjacency

contains

  !> Determine element indeces for submatrix, given a set of nodes.
  !!
  !! \ingroup submatrix_group_F
  !!
  !! \param a Matrix
  !! \param b Submatrix
  !! \param nodelist List of nodes to define submatrix
  !! \param nsize Number of nodes
  !! \param core_halo_index Indeces of core+halo
  !! \param core_pos Positions of core rows in core_halo_index
  !! \param vsize Sizes of core_halo_index and core_pos
  !! \param double_jump_flag Flag 0=no 1=yes
  subroutine bml_matrix2submatrix_index(a, b, nodelist, nsize, &
    core_halo_index, core_pos, vsize, double_jump_flag)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(in) :: b
    integer(C_INT), target, intent(in) :: nodelist(*)
    integer(C_INT), target, intent(inout) :: core_halo_index(*)
    integer(C_INT), target, intent(inout) :: core_pos(*)
    integer(C_INT), target, intent(inout) :: vsize(*)
    integer, intent(in) :: nsize
    logical, intent(in) :: double_jump_flag

    integer(C_INT) :: cflag

    if (double_jump_flag .eqv. .true.) then
      cflag = 1
    else
      cflag = 0;
    endif

    call bml_matrix2submatrix_index_C(a%ptr, b%ptr, c_loc(nodelist), &
      nsize, c_loc(core_halo_index), c_loc(core_pos), c_loc(vsize), cflag)

  end subroutine bml_matrix2submatrix_index

  !> Create contracted submatrix from a set of element indeces.
  !!
  !! \ingroup submatrix_group_F
  !!
  !! \param a Matrix
  !! \param b Submatrix
  !! \param core_halo_index Indeces of core+halo
  !! \param lsize Number of indeces in core_halo_index
  subroutine bml_matrix2submatrix(a, b, core_halo_index, lsize)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: b
    integer(C_INT), target, intent(in) :: core_halo_index(*)
    integer(C_INT), intent(in) :: lsize

    call bml_matrix2submatrix_C(a%ptr, b%ptr, c_loc(core_halo_index), lsize)

  end subroutine bml_matrix2submatrix

  !> Assemble a contracted submatrix into the final matrix.
  !!
  !! \ingroup submatrix_group_F
  !!
  !! \param a Submatrix
  !! \param b Matrix
  !! \param core_halo_index Indeces of core+halo
  !! \param lsize Number of indeces in core_halo_index
  !! \param core_pos Positions of core nodes in core_halo_index
  !! \param llsize Number of positions in core_pos
  subroutine bml_submatrix2matrix(a, b, core_halo_index, lsize, core_pos, &
    llsize, threshold)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: b
    integer(C_INT), target, intent(in) :: core_halo_index(*)
    integer(C_INT), target, intent(in) :: core_pos(*)
    integer(C_INT), intent(in) :: lsize, llsize
    real(C_DOUBLE), optional, intent(in) :: threshold

    real(C_DOUBLE) :: threshold_

    if (present(threshold)) then
       threshold_ = threshold
    else
       threshold_ = 0
    end if

    call bml_submatrix2matrix_C(a%ptr, b%ptr, c_loc(core_halo_index), lsize, &
      c_loc(core_pos), llsize, threshold_)

  end subroutine bml_submatrix2matrix

  !> Assemble adjacency vectors for a matrix.
  !!
  !! \ingroup submatrix_group_F
  !!
  !! \param a Matrix
  !! \param xadj Start indeces for each row
  !! \param adjncy Indices of non-zero values
  subroutine bml_adjacency(a, xadj, adjncy)

    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), target, intent(in) :: xadj(*)
    integer(C_INT), target, intent(in) :: adjncy(*)

    call bml_adjacency_C(a%ptr, c_loc(xadj), c_loc(adjncy))

  end subroutine bml_adjacency

end module bml_submatrix_m
