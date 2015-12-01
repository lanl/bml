!> Matrix diagonalization functions.
module bml_diagonalize_m
  use bml_c_interface_m
  use bml_introspection_m
  use bml_types_m
  implicit none
  private

  public :: bml_diagonalize

contains

  !> Diagonalize a matrix.
  !!
  !! @param a The matrix.
  !! @param eigenvalues The corresponding eigenvalues.
  !! @param eigenvectors The set of eigenvectors.
  subroutine bml_diagonalize(a, eigenvalues, eigenvectors)

    type(bml_matrix_t), intent(in) :: a
    real(C_DOUBLE), target, intent(inout) :: eigenvalues(*)
    type(bml_matrix_t), intent(inout) :: eigenvectors

    call bml_diagonalize_C(a%ptr, c_loc(eigenvalues), eigenvectors%ptr)

  end subroutine bml_diagonalize

end module bml_diagonalize_m
