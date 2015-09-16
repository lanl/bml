module test_m

  implicit none

  type, abstract :: test_t
   contains
     procedure(test_function_interface), nopass, deferred :: test_function
  end type test_t

  abstract interface
     function test_function_interface(n, matrix_type, matrix_precision, m) result(test_result)
       integer, intent(in) :: n, m
       character(len=*), intent(in) :: matrix_type
       character(len=*), intent(in) :: matrix_precision
       logical :: test_result
     end function test_function_interface
  end interface

end module test_m
