module add_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: add_matrix_t
   contains
     procedure, nopass :: test_function
  end type add_matrix_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    double precision, parameter :: ALPHA = 1.2
    double precision, parameter :: BETA = 0.8
    double precision, parameter :: THRESHOLD = 0.0

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b
    type(bml_matrix_t) :: c

    double precision, allocatable :: a_dense(:, :)
    double precision, allocatable :: b_dense(:, :)
    double precision, allocatable :: c_dense(:, :)

    integer :: i

    test_result = .true.

    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_identity_matrix(matrix_type, matrix_precision, n, m, b)

    call bml_convert_to_dense(a, a_dense)
    call bml_convert_to_dense(b, b_dense)

    call bml_add(1.0, a, 1.0, b, c)
    call bml_convert_to_dense(c, c_dense)

    if(maxval(ALPHA*a_dense+BETA*b_dense-c_dense) > 1e-12) then
       call bml_error(__FILE__, __LINE__, "incorrect matrix sum")
    endif

    call bml_add_identity(alpha, a, c, beta)
    call bml_convert_to_dense(c, c_dense)

    b_dense = alpha*a_dense
    do i = 1, n
       b_dense(i, i) = b_dense(i, i)+beta
    end do

    if(maxval(b_dense-c_dense) > 1e-12) then
       call bml_error(__FILE__, __LINE__, "incorrect matrix add identity")
    end if

    call bml_add_identity(a, alpha, beta)
    call bml_convert_to_dense(a, c_dense)

    if(maxval(b_dense-c_dense) > 1e-12) then
       call bml_error(__FILE__, __LINE__, "incorrect matrix add identity")
    end if

    call bml_deallocate(a)
    call bml_deallocate(b)
    call bml_deallocate(c)

  end function test_function

end module add_matrix_m
