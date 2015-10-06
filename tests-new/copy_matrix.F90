module copy_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: copy_matrix_t
   contains
     procedure, nopass :: test_function
  end type copy_matrix_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b

    REAL_TYPE, allocatable :: a_dense(:, :)
    REAL_TYPE, allocatable :: b_dense(:, :)

    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_copy(a, b)

    call bml_convert_to_dense(a, a_dense)
    call bml_convert_to_dense(b, b_dense)
    if(maxval(abs(a_dense-b_dense)) > 1e-12) then
       test_result = .false.
       print *, "matrices are not identical"
    else
       test_result = .true.
    end if

  end function test_function

end module copy_matrix_m
