module convert_matrix_m

  use bml
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

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b
    double precision, allocatable :: a_double(:, :)
    double precision, allocatable :: b_double(:, :)
    real, allocatable :: a_real(:, :)
    real, allocatable :: b_real(:, :)

    test_result = .true.

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       allocate(a_real(n, n))
       call random_number(a_real)
       call bml_convert_from_dense(matrix_type, matrix_precision, a_real, a, 0.0d0)
       call bml_convert_to_dense(a, b_real)
       if(maxval(a_real-b_real) > 1e-12) then
          print *, "Matrix element mismatch"
          call bml_print_matrix("A", a_real)
          call bml_print_matrix("B", b_real)
          test_result = .false.
       end if
    case(BML_PRECISION_DOUBLE)
       allocate(a_double(n, n))
       call random_number(a_double)
       call bml_convert_from_dense(matrix_type, matrix_precision, a_double, a, 0.0d0)
       call bml_convert_to_dense(a, b_double)
       if(maxval(a_double-b_double) > 1e-12) then
          print *, "Matrix element mismatch"
          call bml_print_matrix("A", a_double)
          call bml_print_matrix("B", b_double)
          test_result = .false.
       end if
    end select
    if(test_result) then
       print *, "Test passed"
    end if

  end function test_function

end module convert_matrix_m
