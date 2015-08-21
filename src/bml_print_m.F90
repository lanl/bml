!> \copyright Los Alamos National Laboratory 2015

!> Print a matrix.
module bml_print_m
  use bml_error_m
  implicit none

  !> Print a matrix.
  interface print_matrix
     module procedure :: print_bml_matrix
     module procedure :: print_dense_matrix
  end interface print_matrix

contains

  !> Print a matrix.
  !!
  !! \param name A tag to be printed before the matrix.
  !! \param A The matrix.
  subroutine print_bml_matrix(name, A)

    use bml_type_dense_m
    use bml_print_dense_m
    use bml_error_m

    character(len=*), intent(in) :: name
    class(bml_matrix_t), allocatable, intent(in) :: A

    if(.not. allocated(A)) then
       write(*, "(A)") trim(name)//" not allocated"
    else
       select type(A)
       type is(bml_matrix_dense_double_t)
          call print_matrix_dense(name, A)
       class default
          call error(__FILE__, __LINE__, "unknown matrix type")
       end select
    end if

  end subroutine print_bml_matrix

  !> Print a matrix.
  !!
  !! This is a utility function, that really does not belong into any
  !! module here. It prints out a dense matrix.
  !!
  !! \param name A tag to be printed before the matrix.
  !! \param A The matrix.
  subroutine print_dense_matrix(name, A)

    character(len=*), intent(in) :: name
    double precision, allocatable, intent(in) :: A(:, :)

    integer :: i, j
    character(len=10000) :: line_format

    if(.not. allocated(A)) then
       write(*, "(A)") trim(name)//" not allocated"
    else
       write(*, "(A)") trim(adjustl(name))//" ="
       write(line_format, *) size(A, 2)
       line_format = trim(adjustl(line_format))
       write(line_format, "(A)") "("//trim(line_format)//"es12.2)"
       do i = 1, size(A, 1)
          write(*, line_format) (A(i, j), j = 1, size(A, 2))
       end do
    end if

  end subroutine print_dense_matrix

end module bml_print_m
