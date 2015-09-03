module convert_matrix_m

  use bml
  use bml_type_dense_m
  use bml_print_m
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: convert_matrix_t
   contains
     procedure, nopass :: test_function
  end type convert_matrix_t

contains

  function test_function(n, matrix_type, matrix_precision) result(test_result)

    integer, intent(in) :: n
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    logical :: test_result

    class(bml_matrix_t), allocatable :: a
    class(bml_matrix_t), allocatable :: b
    double precision, allocatable :: a_dense_double(:, :)
    double precision, allocatable :: b_dense_double(:, :)
    real, allocatable :: a_dense_real(:, :)
    real, allocatable :: b_dense_real(:, :)

    test_result = .true.

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       allocate(a_dense_real(n, n))
       call random_number(a_dense_real)
       call convert_from_dense(matrix_type, a_dense_real, a)
       call convert_to_dense(a, b_dense_real)
       if(maxval(a_dense_real-b_dense_real) > 1e-12) then
          print *, "Matrix element mismatch"
          test_result = .false.
       end if
    case(BML_PRECISION_DOUBLE)
       allocate(a_dense_double(n, n))
       call random_number(a_dense_double)
       call convert_from_dense(matrix_type, a_dense_double, a)
       call convert_to_dense(a, b_dense_double)
       if(maxval(a_dense_double-b_dense_double) > 1e-12) then
          print *, "Matrix element mismatch"
          test_result = .false.
       end if
    end select
    print *, "Test passed"

  end function test_function

end module convert_matrix_m
