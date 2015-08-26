module test_m

  interface
     subroutine test_function(n, matrix_type, matrix_precision)
       integer, intent(in) :: n
       character(len=*), intent(in) :: matrix_type
       character(len=*), intent(in) :: matrix_precision
     end subroutine test_function
  end interface

end module test_m
