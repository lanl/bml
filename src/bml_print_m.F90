!> \copyright Los Alamos National Laboratory 2015

!> Print a matrix.
module bml_print_m

  use bml_error_m

  implicit none

  private

  !> Print a vector.
  interface bml_print_vector
     module procedure :: print_vector_dense
  end interface bml_print_vector

  public :: bml_print_matrix
  public :: bml_print_vector

contains

  !> Print a matrix.
  !!
  !! \param name A tag to be printed before the matrix.
  !! \param a The matrix.
  subroutine bml_print_matrix(name, a)

    use bml_type_m
    use bml_type_dense_m
    use bml_print_dense_m
    use bml_error_m

    character(len=*), intent(in) :: name
    class(bml_matrix_t), intent(in) :: a

    select type(a)
    type is(bml_matrix_dense_double_t)
       call bml_print_matrix_dense(name, a)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine bml_print_matrix

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
       write(*, "(A)") trim(name)//" not associated"
    else
       write(*, "(A)") trim(adjustl(name))//" ="
       write(line_format, *) size(a, 1)
       line_format = trim(adjustl(line_format))
       write(line_format, "(A)") "("//trim(line_format)//"es12.2)"
       write(*, line_format) (a(i), i = 1, size(a, 1))
    end if

  end subroutine print_vector_dense

end module bml_print_m
