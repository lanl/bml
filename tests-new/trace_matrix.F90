module trace_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: trace_matrix_t
   contains
     procedure, nopass :: test_function
  end type trace_matrix_t

contains

  function test_function(n, matrix_type, matrix_precision, m) result(test_result)

    integer, intent(in) :: n, m
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    logical :: test_result

    class(bml_matrix_t), allocatable :: a

    real, allocatable :: a_real(:, :)
    double precision, allocatable :: a_double(:, :)

    double precision :: tr_a
    double precision :: tr_reference

    integer :: i

    call bml_random_matrix(matrix_type, n, a, matrix_precision, m)
    tr_a = bml_trace(a)

    tr_reference = 0
    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       call bml_convert_to_dense(a, a_real)
       do i = 1, n
          tr_reference = tr_reference+a_real(i, i)
       end do
    case(BML_PRECISION_DOUBLE)
       call bml_convert_to_dense(a, a_double)
       do i = 1, n
          tr_reference = tr_reference+a_double(i, i)
       end do
    case default
       print *, "unknown precision"
    end select

    if(abs(tr_a-tr_reference) > 1e-12) then
       test_result = .false.
       print *, "trace incorrect"
    else
       test_result = .true.
    end if

  end function test_function

end module trace_matrix_m
