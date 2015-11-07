!> Matrix diagonalization functions.
module bml_diagonalize_m

  implicit none
  private

  interface

     subroutine bml_diagonalize_C(a, eigenvalues, eigenvectors) &
          bind(C, name="bml_diagonalize")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR), intent(in) :: eigenvalues
       type(C_PTR), intent(in) :: eigenvectors
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
    type(bml_matrix_t), intent(in) :: eigenvectors
    double precision, pointer, intent(inout) :: eigenvalues(:)

    type(C_PTR) :: eigenvectors_
    type(C_PTR) :: eigenvalues_

    call bml_diagonalize_C(a%ptr, eigenvalues_, eigenvectors_)
    call c_f_pointer(eigenvalues_, eigenvalues, [bml_get_n(a)])

  end subroutine bml_diagonalize

end module bml_diagonalize_m
