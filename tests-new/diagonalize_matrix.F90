module diagonalize_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: diagonalize_matrix_t
   contains
     procedure, nopass :: test_function
  end type diagonalize_matrix_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: a_t
    type(bml_matrix_t) :: eigenvectors
    type(bml_vector_t) :: eigenvalues

    test_result = .false.

    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_transpose(a, a_t)
    call bml_add(0.5, a, 0.5, a_t)
    call bml_diagonalize(a, eigenvectors, eigenvalues)
    call bml_print_matrix("A", a, 1, n, 1, n)
    call bml_print_matrix("eigenvectors", eigenvectors, 1, n, 1, n)
    call bml_print_vector("eigenvalues", eigenvalues, 1, n)

  end function test_function

end module diagonalize_matrix_m
