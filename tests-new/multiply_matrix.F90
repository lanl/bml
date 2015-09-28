module multiply_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: multiply_matrix_t
   contains
     procedure, nopass :: test_function
  end type multiply_matrix_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b
    type(bml_matrix_t) :: c

    real, allocatable :: a_real(:, :)
    real, allocatable :: b_real(:, :)
    real, allocatable :: c_real(:, :)

    double precision, allocatable :: a_double(:, :)
    double precision, allocatable :: b_double(:, :)
    double precision, allocatable :: c_double(:, :)

    call bml_random_matrix(matrix_type, n, a, matrix_precision, m)
    call bml_identity_matrix(matrix_type, n, b, matrix_precision, m)

    call bml_multiply(a, b, c)

    test_result = .true.

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       call bml_convert_to_dense(a, a_real)
       call bml_convert_to_dense(b, b_real)
       call bml_convert_to_dense(c, c_real)
       if(maxval(matmul(a_real, b_real)-c_real) > 1e-12) then
          test_result = .false.
          print *, "incorrect matrix product"
       endif
    case(BML_PRECISION_DOUBLE)
       call bml_convert_to_dense(a, a_double)
       call bml_convert_to_dense(b, b_double)
       call bml_convert_to_dense(c, c_double)
       if(maxval(matmul(a_double, b_double)-c_double) > 1e-12) then
          test_result = .false.
          print *, "incorrect matrix product"
       endif
    end select

    call bml_deallocate(a)
    call bml_deallocate(b)
    call bml_deallocate(c)

  end function test_function

end module multiply_matrix_m
