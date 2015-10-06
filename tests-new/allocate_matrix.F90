module allocate_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: allocate_matrix_t
   contains
     procedure, nopass :: test_function
  end type allocate_matrix_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a

    REAL_TYPE, allocatable :: a_dense(:, :)
    integer :: i, j

    test_result = .true.

    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_identity_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_convert_to_dense(a, a_dense)
    call bml_print_matrix("A", a_dense, lbound(a_dense, 1), ubound(a_dense, 1), &
         lbound(a_dense, 2), ubound(a_dense, 2))
    do i = 1, n
       do j = 1, n
          if(i == j) then
             if(abs(a_dense(i, j)-1) > 1e-12) then
                print *, "Incorrect value on diagonal", a_dense(i, j)
                test_result = .false.
                return
             end if
          else
             if(abs(a_dense(i, j)) > 1e-12) then
                print *, "Incorrect value off diagonal", a_dense(i, j)
                test_result = .false.
                return
             end if
          end if
       end do
    end do
    print *, "Identity matrix test passed"

    call bml_zero_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_deallocate(a)

  end function test_function

end module allocate_matrix_m
