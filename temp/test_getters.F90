program test_getters

  use bml

  double precision, parameter :: REL_TOL = 1d-12
  integer, parameter :: N = 4

  type(bml_matrix_t) :: A
  double precision :: A_dense(N, N)
  double precision :: row(N)

  integer :: i, j

  call random_number(A_dense)
  A_dense(1,1) = 1
  A_dense(1,2) = 2
  A_dense(2,1) = 3
  write(*, "(A)") "A_dense ="
  call print_dense_matrix(A_dense)

  call bml_convert_from_dense(BML_MATRIX_DENSE, A_dense, A)
  call bml_print_matrix("A", A, 1, N, 1, N)

  do i = 1, N
    call bml_get_row(A, i, row)
    do j = 1, N
      rel_diff = abs((A_dense(i, j) - row(j)) / A_dense(i, j))
      if (rel_diff > REL_TOL) then
        write(*, "(A,I2,',',I2,A)") "matrices are not identical at A(", i, j, ")"
        write(*, "(A,I2)") "getting row ", i
        call print_dense_vector(row)
        error stop
      end if
    end do
  end do

contains

  subroutine print_dense_vector(x)

    double precision, intent(in) :: x(:)

    integer :: i
    character(len=1000) :: format_string

    format_string = "("
    do i = 1, size(x, 1)
      format_string = trim(format_string)//"f6.2"
      if (i < size(x, 1)) then
        format_string = trim(format_string)//","
      end if
    end do
    format_string = trim(format_string)//")"
    write(*, format_string) (x(i), i = 1, size(x, 1))

  end subroutine print_dense_vector

  subroutine print_dense_matrix(A)

    double precision, intent(in) :: A(:, :)

    integer :: i
    character(len=1000) :: format_string

    format_string = "("
    do i = 1, size(A, 2)
      format_string = trim(format_string)//"f7.3"
      if (i < size(A, 2)) then
        format_string = trim(format_string)//","
      end if
    end do
    format_string = trim(format_string)//")"
    do i = 1, size(A, 1)
      write(*, format_string) (A(i, j), j = 1, size(A, 2))
    end do

  end subroutine print_dense_matrix

end program test_getters
