module test_m

  implicit none

  type, abstract :: test_t
   contains
     procedure(test_function_interface), nopass, deferred :: test_function
  end type test_t

  abstract interface
     function test_function_interface(matrix_type, matrix_precision, n, m) result(test_result)
       character(len=*), intent(in) :: matrix_type
       character(len=*), intent(in) :: matrix_precision
       integer, intent(in) :: n, m
       logical :: test_result
     end function test_function_interface
  end interface

end module test_m
