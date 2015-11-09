!> Matrix diagonalization functions.
module bml_diagonalize_m

  implicit none
  private

  interface

     subroutine bml_diagonalize_C(a, eigenvalues, eigenvectors) &
          bind(C, name="bml_diagonalize")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR), value :: eigenvalues
       type(C_PTR), value :: eigenvectors
     end subroutine bml_diagonalize_C

  end interface

  public :: bml_diagonalize

contains

  !> Diagonalize a matrix.
  !!
  !! @param a The matrix.
  !! @param eigenvalues The corresponding eigenvalues.
  !! @param eigenvectors The set of eigenvectors.
  subroutine bml_diagonalize(a, eigenvalues, eigenvectors)

    use bml_introspection_m
    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    double precision, target, intent(inout) :: eigenvalues(:)
    type(bml_matrix_t), intent(inout) :: eigenvectors

    call bml_diagonalize_C(a%ptr, c_loc(eigenvalues), eigenvectors%ptr)

  end subroutine bml_diagonalize

end module bml_diagonalize_m
