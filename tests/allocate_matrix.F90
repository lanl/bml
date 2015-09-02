module allocate_matrix_m

  use bml
  use test_m

  implicit none

  type, extends(test_t) :: allocate_matrix_t
   contains
     procedure, nopass :: test_function => allocate_test_function
  end type allocate_matrix_t

contains

  function allocate_test_function(n, matrix_type, matrix_precision) result(test_result)

    integer, intent(in) :: n
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    logical :: test_result

    class(bml_matrix_t), allocatable :: a

    real, allocatable :: a_real(:, :)
    double precision, allocatable :: a_double(:, :)

    integer :: i, j

    test_result = .true.

    if(matrix_type == BML_MATRIX_DENSE) then
       call random_matrix(matrix_type, n, a, matrix_precision)
    else
       print *, "Random matrix not supported for matrix type "//matrix_type
    end if

    call identity_matrix(matrix_type, n, a, matrix_precision)
    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       call convert_to_dense(a, a_real)
       do i = 1, n
          do j = 1, n
             if(i == j) then
                if(abs(a_real(i, j)-1) > 1e-12) then
                   print *, "Incorrect value on diagonal", a_real(i, j)
                   test_result = .false.
                   return
                end if
             else
                if(abs(a_real(i, j)) > 1e-12) then
                   print *, "Incorrect value off diagonal", a_real(i, j)
                   test_result = .false.
                   return
                end if
             end if
          end do
       end do
    case(BML_PRECISION_DOUBLE)
       call convert_to_dense(a, a_double)
       do i = 1, n
          do j = 1, n
             if(i == j) then
                if(abs(a_double(i, j)-1) > 1e-12) then
                   print *, "Incorrect value on diagonal", a_double(i, j)
                   test_result = .false.
                   return
                end if
             else
                if(abs(a_double(i, j)) > 1e-12) then
                   print *, "Incorrect value off diagonal", a_double(i, j)
                   test_result = .false.
                   return
                end if
             end if
          end do
       end do
    end select
    print *, "Test passed"

    call allocate_matrix(matrix_type, n, a, matrix_precision)
    call deallocate_matrix(a)

  end function allocate_test_function

end module allocate_matrix_m
