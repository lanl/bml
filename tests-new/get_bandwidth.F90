module get_bandwidth_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: get_bandwidth_t
   contains
     procedure, nopass :: test_function
  end type get_bandwidth_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a

    integer :: i

    call bml_identity_matrix(matrix_type, n, a, matrix_precision, m)

    test_result = .true.

    do i = 1, n
       if(bml_get_bandwidth(a, i) /= 1) then
          print *, "Wrong bandwidth on row ", i
          print *, "Should be 1, but is ", bml_get_bandwidth(a, i)
          call bml_print_matrix("A", a)
          test_result = .false.
          return
       end if
    end do
    print *, "Test passed"

    call bml_deallocate(a)

  end function test_function

end module get_bandwidth_m
