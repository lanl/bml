!> \copyright Los Alamos National Laboratory 2015

!> Print a dense matrix.
module bml_print_dense_m

  implicit none

  private

  public :: bml_print_matrix_dense

contains

  !> Print a matrix.
  !!
  !! @param name A tag to be printed before the matrix.
  !! @param a The matrix.
  subroutine bml_print_matrix_dense(name, a)

    use bml_type_dense_m
    use bml_error_m

    character(len=*), intent(in) :: name
    class(bml_matrix_dense_t), intent(in) :: a

    select type(a)
    type is(bml_matrix_dense_single_t)
       call print_matrix_dense_single(name, a)
    type is(bml_matrix_dense_double_t)
       call print_matrix_dense_double(name, a)
    end select

  end subroutine bml_print_matrix_dense

  !> Print a matrix.
  !!
  !! @param name A tag to be printed before the matrix.
  !! @param a The matrix.
  subroutine print_matrix_dense_single(name, a)

    use bml_type_dense_m
    use bml_error_m
    use bml_utility_m

    character(len=*), intent(in) :: name
    type(bml_matrix_dense_single_t), intent(in) :: a

    integer :: i, j
    character(len=10000) :: line_format

    call bml_print_matrix(name, a%matrix)

  end subroutine print_matrix_dense_single

  !> Print a matrix.
  !!
  !! @param name a tag to be printed before the matrix.
  !! @param A The matrix.
  subroutine print_matrix_dense_double(name, a)

    use bml_type_dense_m
    use bml_error_m
    use bml_utility_m

    character(len=*), intent(in) :: name
    type(bml_matrix_dense_double_t), intent(in) :: a

    call bml_print_matrix(name, a%matrix)

  end subroutine print_matrix_dense_double

end module bml_print_dense_m
