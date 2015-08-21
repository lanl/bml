!> \copyright Los Alamos National Laboratory 2015

!> Print a dense matrix.
module bml_print_dense_m
  implicit none

  !> Print a matrix.
  interface print_matrix_dense
     module procedure print_matrix_dense_single
     module procedure print_matrix_dense_double
  end interface print_matrix_dense

contains

  !> Print a matrix.
  !!
  !! @param name A tag to be printed before the matrix.
  !! @param A The matrix.
  subroutine print_matrix_dense_single(name, A)

    use bml_type_dense_m
    use bml_error_m

    character(len=*), intent(in) :: name
    type(bml_matrix_dense_single_t), intent(in) :: A

    integer :: i, j
    character(len=10000) :: line_format

    if(A%N > 20) then
       call warning(__FILE__, __LINE__, "matrix is rather large")
    else
       write(*, "(A)") trim(adjustl(name))//" ="
       write(line_format, *) A%N
       line_format = trim(adjustl(line_format))
       write(line_format, "(A)") "("//trim(line_format)//"es12.2)"
       do i = 1, A%N
          write(*, line_format) (A%matrix(i, j), j = 1, A%N)
       end do
    end if

  end subroutine print_matrix_dense_single

  !> Print a matrix.
  !!
  !! @param name A tag to be printed before the matrix.
  !! @param A The matrix.
  subroutine print_matrix_dense_double(name, A)

    use bml_type_dense_m
    use bml_error_m

    character(len=*), intent(in) :: name
    type(bml_matrix_dense_double_t), intent(in) :: A

    integer :: i, j
    character(len=10000) :: line_format

    if(A%N > 20) then
       call warning(__FILE__, __LINE__, "matrix is rather large")
    else
       write(*, "(A)") trim(adjustl(name))//" ="
       write(line_format, *) A%N
       line_format = trim(adjustl(line_format))
       write(line_format, "(A)") "("//trim(line_format)//"es12.2)"
       do i = 1, A%N
          write(*, line_format) (A%matrix(i, j), j = 1, A%N)
       end do
    end if

  end subroutine print_matrix_dense_double

end module bml_print_dense_m
