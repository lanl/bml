  !> Matrix diagonalization functions.
module bml_diagonalize_m
  implicit none
contains

  !> Diagonalize a matrix.
  !!
  !! @param a The matrix.
  !! @param eigenvectors The set of eigenvectors.
  !! @param eigenvalues The corresponding eigenvalues.
  subroutine diagonalize(a, eigenvectors, eigenvalues)

    use bml_type_m
    use bml_type_dense_m
    use bml_diagonalize_dense_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: a
    double precision, allocatable, intent(out) :: eigenvectors(:, :)
    double precision, allocatable, intent(out) :: eigenvalues(:)

    select type(a)
    class is(bml_matrix_dense_t)
       call diagonalize_dense(a, eigenvectors, eigenvalues)
    class default
       call error(__FILE__, __LINE__, "unknow matrix type")
    end select

  end subroutine diagonalize

end module bml_diagonalize_m
