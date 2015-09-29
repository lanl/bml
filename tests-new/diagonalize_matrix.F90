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

    REAL_TYPE, allocatable :: a_dense(:, :)
    REAL_TYPE, allocatable :: eigenvectors(:, :)
    REAL_TYPE, allocatable :: eigenvalues(:)

    test_result = .false.

    allocate(a_dense(n, n))
    a_dense = 0
    call random_number(a_dense)
    a_dense = (a_dense+transpose(a_dense))/2
    call bml_convert_from_dense(matrix_type, a_dense, a)
    call bml_diagonalize(a, eigenvectors, eigenvalues)
    call bml_print_matrix("A", a_dense, python_format=.true.)
    call bml_print_vector("eval", eigenvalues)
    call bml_print_matrix("evec", eigenvectors)

  end function test_function

end module diagonalize_matrix_m
