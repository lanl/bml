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
    double precision :: threshold

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: a_t
    type(bml_matrix_t) :: eigenvectors
    double precision, allocatable :: eigenvalues(:)

    threshold = 0.0
    test_result = .false.

    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_print_matrix("A", a, 1, n, 1, n)
    call bml_transpose(a, a_t)
    call bml_print_matrix("A_t", a_t, 1, n, 1, n)
    call bml_add(0.5d0, a, 0.5d0, a_t, threshold)
    call bml_print_matrix("A", a, 1, n, 1, n)
    allocate(eigenvalues(n))
    call bml_zero_matrix(matrix_type, matrix_precision, n, m, eigenvectors)
    call bml_diagonalize(a, eigenvalues, eigenvectors)
    write(*, *) eigenvalues
    call bml_print_matrix("eigenvectors", eigenvectors, 1, n, 1, n)
    !call bml_print_vector("eigenvalues", eigenvalues, 1, n)

    test_result = .true.

  end function test_function

end module diagonalize_matrix_m
