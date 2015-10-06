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

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b

    double precision, allocatable :: a_double(:, :)
    double precision, allocatable :: b_double(:, :)
    real, allocatable :: a_real(:, :)
    real, allocatable :: b_real(:, :)

    test_result = .true.

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE_REAL)
       allocate(a_real(n, n))
       call random_number(a_real)
       call bml_convert_from_dense(matrix_type, a_real, a, 0.0d0, m)
       call bml_convert_to_dense(a, b_real)
       if(maxval(a_real-b_real) > 1e-12) then
          print *, "Matrix element mismatch"
          call bml_print_matrix("A", a_real, lbound(a_real, 1), ubound(a_real, 1), &
               lbound(a_real, 2), ubound(a_real, 2))
          call bml_print_matrix("B", b_real, lbound(b_real, 1), ubound(b_real, 1), &
               lbound(b_real, 2), ubound(b_real, 2))
          test_result = .false.
       end if
    case(BML_PRECISION_DOUBLE_REAL)
       allocate(a_double(n, n))
       call random_number(a_double)
       call bml_convert_from_dense(matrix_type, a_double, a, 0.0d0, m)
       call bml_convert_to_dense(a, b_double)
       if(maxval(a_double-b_double) > 1e-12) then
          print *, "Matrix element mismatch"
          call bml_print_matrix("A", a_double, lbound(a_double, 1), ubound(a_double, 1), &
               lbound(a_double, 2), ubound(a_double, 2))
          call bml_print_matrix("B", b_double, lbound(b_double, 1), ubound(b_double, 1), &
               lbound(b_double, 2), ubound(b_double, 2))
          test_result = .false.
       end if
    end select
    if(test_result) then
       print *, "Test passed"
    end if

  end function test_function

end module convert_matrix_m
