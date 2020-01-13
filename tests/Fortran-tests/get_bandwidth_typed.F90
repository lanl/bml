module get_bandwidth_typed

  use bml
  use prec
  use bml_add_m

  implicit none

  public :: test_get_bandwidth_typed

contains

  function test_get_bandwidth_typed(matrix_type, element_kind, element_precision, n, m) &
       & result(test_result)

    character(len=*), intent(in) :: matrix_type, element_kind
    integer, intent(in) :: element_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a

    integer :: i

    call bml_identity_matrix(matrix_type, element_kind, element_precision, n, &
         & m, a)

    test_result = .true.

    do i = 1, n
      if(bml_get_row_bandwidth(a, i) /= 1) then
        print *, "Wrong bandwidth on row ", i
        print *, "Should be 1, but is ", bml_get_row_bandwidth(a, i)
        call bml_print_matrix("A", a, 1, n, 1, n)
        test_result = .false.
        return
      end if
    end do
    print *, "Test passed"

    call bml_deallocate(a)

  end function test_get_bandwidth_typed

end module get_bandwidth_typed
