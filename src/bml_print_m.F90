!> \copyright Los Alamos National Laboratory 2015

!> Print a matrix.
module bml_print_m
  use bml_error_m
  implicit none

  !> Print a matrix.
  interface print_matrix
     module procedure :: print_bml_matrix
     module procedure :: print_matrix_dense
  end interface print_matrix

  !> Print a vector.
  interface print_vector
     module procedure :: print_vector_dense
  end interface print_vector

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
  !! \param a The matrix.
  !! \param python_format If true then print matrix in python format
  !! so that the output can be cut and pasted into a python session or
  !! script.
  subroutine print_matrix_dense(name, a, python_format)

    character(len=*), intent(in) :: name
    double precision, allocatable, intent(in) :: a(:, :)
    logical, optional :: python_format

    integer :: i, j
    character(len=10000) :: line_format

    if(.not. allocated(a)) then
       write(*, "(A)") trim(name)//" not allocated"
    else
       if(present(python_format)) then
          if(python_format) then
             write(*, "(A)") trim(adjustl(name))//" = numpy.matrix(["
             do i = 1, size(a, 1)
                write(*, "(A)") "  ["
                do j = 1, size(a, 2)
                   if(j < size(a, 2)) then
                      write(*, "(A,ES20.10,A)") "    ", a(i, j), ","
                   else
                      write(*, "(A,ES20.10)") "    ", a(i, j)
                   end if
                end do
                if(i < size(a, 1)) then
                   write(*, "(A)") "  ],"
                else
                   write(*, "(A)") "  ]"
                end if
             end do
             write(*, "(A)") "])"
          end if
       else
          write(*, "(A)") trim(adjustl(name))//" ="
          write(line_format, *) size(a, 2)
          line_format = trim(adjustl(line_format))
          write(line_format, "(A)") "("//trim(line_format)//"ES12.2)"
          do i = 1, size(a, 1)
             write(*, line_format) (a(i, j), j = 1, size(a, 2))
          end do
       end if
    end if

  end subroutine print_matrix_dense

  !> Print a vector.
  !!
  !! This is a utility function, that really does not belong into any
  !! module here. It prints out a dense vector.
  !!
  !! \param name A tag to be printed before the matrix.
  !! \param a The vector.
  subroutine print_vector_dense(name, a)

    character(len=*), intent(in) :: name
    double precision, allocatable, intent(in) :: a(:)

    integer :: i
    character(len=10000) :: line_format

    if(.not. allocated(a)) then
       write(*, "(A)") trim(name)//" not allocated"
    else
       write(*, "(A)") trim(adjustl(name))//" ="
       write(line_format, *) size(a, 1)
       line_format = trim(adjustl(line_format))
       write(line_format, "(A)") "("//trim(line_format)//"es12.2)"
       write(*, line_format) (a(i), i = 1, size(a, 1))
    end if

  end subroutine print_vector_dense

end module bml_print_m
