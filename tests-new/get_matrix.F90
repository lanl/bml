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

  function test_function(n, matrix_type, matrix_precision, m) result(test_result)

    integer, intent(in) :: n. m
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    logical :: test_result

    class(bml_matrix_t), allocatable :: a

    integer :: i, j

    real, allocatable :: a_real(:, :)
    double precision, allocatable :: a_double(:, :)

    real :: a_ij_real
    double precision :: a_ij_double

    call bml_random_matrix(matrix_type, n, a, matrix_precision, m)

    test_result = .true.

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       call bml_convert_to_dense(a, a_real)
       do i = 1, a%n
          do j = 1, a%n
             a_ij_real = bml_get(a, i, j)
             if(abs(a_ij_real-a_real(i, j)) > 1e-12) then
                test_result = .false.
                print *, "matrix element mismatch"
                return
             end if
          end do
       end do
    case(BML_PRECISION_DOUBLE)
       call bml_convert_to_dense(a, a_double)
       do i = 1, a%n
          do j = 1, a%n
             a_ij_double = bml_get(a, i, j)
             if(abs(a_ij_double-a_double(i, j)) > 1e-12) then
                test_result = .false.
                print *, "matrix element mismatch"
                return
             end if
          end do
       end do
    end select

    call bml_deallocate(a)

  end function test_function

end module get_matrix_m
