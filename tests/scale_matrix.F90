module scale_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: scale_matrix_t
   contains
     procedure, nopass :: test_function
  end type scale_matrix_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    double precision, parameter :: alpha = 1.2

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: c

    REAL_TYPE, allocatable :: a_dense(:, :)
    REAL_TYPE, allocatable :: c_dense(:, :)

    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_scale(alpha, a, c)

    call bml_convert_to_dense(a, a_dense)
    call bml_convert_to_dense(c, c_dense)

    if(maxval(abs(alpha*a_dense-c_dense)) > 1e-12) then
       test_result = .false.
       print *, "matrix element mismatch"
    else
       test_result = .true.
    endif

  end function test_function

end module scale_matrix_m
