subroutine test_function(n, matrix_type, matrix_precision)

  use bml

  implicit none

  integer, intent(in) :: n
  character(len=*), intent(in) :: matrix_type
  character(len=*), intent(in) :: matrix_precision

  class(bml_matrix_t), allocatable :: a

  real, allocatable :: a_real(:, :)
  double precision, allocatable :: a_double(:, :)

  integer :: i, j

  call random_matrix(matrix_type, n, a, matrix_precision)

  call identity_matrix(matrix_type, n, a, matrix_precision)
  select case(matrix_precision)
  case(BML_PRECISION_SINGLE)
     call convert_to_dense(a, a_real)
     do i = 1, n
        do j = 1, n
           if(i == j) then
              if(abs(a_real(i, j)-1) > 1e-12) then
                 write(*, *) "incorrect value on diagonal", a_real(i, j)
                 error stop
              end if
           else
              if(abs(a_real(i, j)) > 1e-12) then
                 write(*, *) "incorrect value off diagonal", a_real(i, j)
                 error stop
              end if
           end if
        end do
     end do
  case(BML_PRECISION_DOUBLE)
     call convert_to_dense(a, a_double)
  end select

  call allocate_matrix(matrix_type, n, a, matrix_precision)
  call deallocate_matrix(a)

end subroutine test_function
