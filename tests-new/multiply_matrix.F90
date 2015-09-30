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

    REAL_TYPE, pointer :: a_dense(:, :)
    REAL_TYPE, pointer :: b_dense(:, :)
    REAL_TYPE, pointer :: c_dense(:, :)

    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_identity_matrix(matrix_type, matrix_precision, n, m, b)

    call bml_multiply(a, b, c)

    test_result = .true.

    call bml_convert_to_dense(a, a_dense)
    call bml_convert_to_dense(b, b_dense)
    call bml_convert_to_dense(c, c_dense)
    !if(maxval(matmul(a_dense, b_dense)-c_dense) > 1e-12) then
    !   test_result = .false.
    !   print *, "incorrect matrix product"
    !endif

    call bml_deallocate(a)
    call bml_deallocate(b)
    call bml_deallocate(c)

  end function test_function

end module multiply_matrix_m
