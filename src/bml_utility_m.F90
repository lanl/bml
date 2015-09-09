!> \copyright Los Alamos National Laboratory 2015

!> Utility functions.
module bml_utility_m

  implicit none

  private

  !> Utility interfaces for dense matrices.
  interface bml_print_matrix
     module procedure print_dense_single
     module procedure print_dense_double
  end interface bml_print_matrix

  public :: bml_print_matrix

contains

  !> Print a matrix.
  !!
  !! \param name A tag to be printed before the matrix.
  !! \param a The matrix.
  !! \param python_format If true then print matrix in python format
  !! so that the output can be cut and pasted into a python session or
  !! script.
  subroutine print_dense_double(name, a, python_format)

    character(len=*), intent(in) :: name
    double precision, allocatable, intent(in) :: a(:, :)
    logical, optional :: python_format

    integer :: i, j
    character(len=10000) :: line_format

    if(.not. allocated(a)) then
       write(*, "(A)") trim(name)//" not associated"
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

  end subroutine print_dense_double

  !> Print a matrix.
  !!
  !! \param name A tag to be printed before the matrix.
  !! \param a The matrix.
  !! \param python_format If true then print matrix in python format
  !! so that the output can be cut and pasted into a python session or
  !! script.
  subroutine print_dense_single(name, a, python_format)

    character(len=*), intent(in) :: name
    real, allocatable, intent(in) :: a(:, :)
    logical, optional :: python_format

    integer :: i, j
    character(len=10000) :: line_format

    if(.not. allocated(a)) then
       write(*, "(A)") trim(name)//" not associated"
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

  end subroutine print_dense_single

end module bml_utility_m
