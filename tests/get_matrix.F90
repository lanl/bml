module get_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: get_matrix_t
   contains
     procedure, nopass :: test_function
  end type get_matrix_t

contains

  function test_function(n, matrix_type, matrix_precision) result(test_result)

    integer, intent(in) :: n
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    logical :: test_result

  end function test_function

end module get_matrix_m
