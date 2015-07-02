!> @copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert

  implicit none

contains

  !> Convert a matrix into a dense matrix.
  !!
  !! @param A The bml matrix.
  !! @param A_dense The dense matrix.
  subroutine convert_to_dense(A, A_dense)

    use bml_error
    use bml_type_dense

    class(bml_matrix_t), allocatable, intent(in) :: A
    double precision, allocatable, intent(out) :: A_dense(:, :)

    if(.not. allocated(A)) then
       call warning(__FILE__, __LINE__, "A is not allocated")
    else
       select type(A)
       type is(bml_matrix_dense_t)
          A_dense = A%matrix
       class default
          call error(__FILE__, __LINE__, "unknown matrix type")
       end select
    endif

  end subroutine convert_to_dense

end module bml_convert
