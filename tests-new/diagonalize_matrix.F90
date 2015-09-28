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

    double precision, allocatable :: a_dense_double(:, :)
    double precision, allocatable :: eigenvectors_double(:, :)
    double precision, allocatable :: eigenvalues_double(:)

    test_result = .false.

    select case(matrix_precision)
    case(BML_PRECISION_DOUBLE)
       allocate(a_dense_double(n, n))
       call random_number(a_dense_double)
       a_dense_double = (a_dense_double+transpose(a_dense_double))/2
       call bml_convert_from_dense(matrix_type, a_dense_double, a)
       call bml_diagonalize(a, eigenvectors_double, eigenvalues_double)
       call bml_print_matrix("A", a_dense_double, python_format=.true.)
       call bml_print_vector("eval", eigenvalues_double)
       call bml_print_matrix("evec", eigenvectors_double)
    case default
       print *, "unknown matrix type"
    end select

  end function test_function

end module diagonalize_matrix_m
