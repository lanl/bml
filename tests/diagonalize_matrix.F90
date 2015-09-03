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

  function test_function(n, matrix_type, matrix_precision) result(test_result)

    integer, intent(in) :: n
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    logical :: test_result

    class(bml_matrix_t), allocatable :: a
    double precision, allocatable :: a_dense_double(:, :)
    double precision, allocatable :: eigenvectors_double(:, :)
    double precision, allocatable :: eigenvalues_double(:)

    test_result = .false.

    select case(matrix_precision)
    case(BML_PRECISION_DOUBLE)
       allocate(a_dense_double(n, n))
       call random_number(a_dense_double)
       a_dense_double = (a_dense_double+transpose(a_dense_double))/2
       call convert_from_dense(matrix_type, a_dense_double, a)
       call diagonalize(a, eigenvectors_double, eigenvalues_double)
       call print_matrix("A", a_dense_double, .true.)
       call print_vector("eval", eigenvalues_double)
       call print_matrix("evec", eigenvectors_double)
    case default
       print *, "unknown matrix type"
    end select

  end function test_function

end module diagonalize_matrix_m
