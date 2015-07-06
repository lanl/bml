!> \copyright Los Alamos National Laboratory 2015

!> Print a matrix.
module bml_print

  use bml_error

  implicit none

contains

  !> Print a matrix.
  !!
  !! @param name A tag to be printed before the matrix.
  !! @param A The matrix.
  subroutine print_matrix(name, A)

    use bml_print_dense
    use bml_error

    character(len=*), intent(in) :: name
    class(bml_matrix_t), allocatable, intent(in) :: A

    if(.not. allocated(A)) then
       write(*, "(A)") trim(name)//" not allocated"
    else
       select type(A)
       type is(bml_matrix_dense_t)
          call print_matrix_dense(name, A)
       class default
          call error(__FILE__, __LINE__, "unknown matrix type")
       end select
    endif

  end subroutine print_matrix

end module bml_print
