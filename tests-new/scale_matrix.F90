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

    type(bml_matrix_t) :: A
    type(bml_matrix_t) :: C

    double precision, allocatable :: A_dense(:, :)
    double precision, allocatable :: C_dense(:, :)

    call bml_random_matrix(BML_MATRIX_DENSE, N, A, M)
    call bml_scale(alpha, A, C)

    call bml_convert_to_dense(A, A_dense)
    call bml_convert_to_dense(C, C_dense)

    if(maxval(alpha*A_dense-C_dense) > 1e-12) then
       test_result = .false.
       print *, "matrix element mismatch"
    else
       test_result = .true.
    endif

  end function test_function

end module scale_matrix_m
