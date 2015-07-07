!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_dense
  implicit none
contains

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param A The bml matrix.
  !! \param A_dense The dense matrix.
  subroutine convert_to_dense_dense(A, A_dense)

    use bml_type_dense

    type(bml_matrix_dense_t), intent(in) :: A
    double precision, allocatable, intent(out) :: A_dense(:, :)

    A_dense = A%matrix

  end subroutine convert_to_dense_dense

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param A_dense The dense matrix.
  !! \param A The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_dense(A_dense, A, threshold)

    use bml_type_dense

    double precision, intent(in) :: A_dense(:, :)
    type(bml_matrix_dense_t), intent(inout) :: A
    double precision, optional, intent(in) :: threshold

    integer :: i, j

    A%matrix = A_dense
    if(present(threshold)) then
       do i = 1, A%N
          do j = 1, A%N
             if(A%matrix(i, j) <= threshold) A%matrix(i, j) = 0
          end do
       end do
    end if

  end subroutine convert_from_dense_dense

end module bml_convert_dense
