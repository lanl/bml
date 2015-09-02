!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_dense_m
  implicit none

  !> Convert bml to dense matrix.
  interface convert_to_dense_dense
     module procedure convert_to_dense_dense_single
     module procedure convert_to_dense_dense_double
  end interface convert_to_dense_dense

  !> Convert bml to dense matrix.
  interface convert_from_dense_dense
     module procedure convert_from_dense_dense_single
     module procedure convert_from_dense_dense_double
  end interface convert_from_dense_dense

contains

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param A The bml matrix.
  !! \param A_dense The dense matrix.
  subroutine convert_to_dense_dense_single(A, A_dense)

    use bml_type_dense_m

    type(bml_matrix_dense_single_t), intent(in) :: A
    real, allocatable, intent(out) :: A_dense(:, :)

    A_dense = A%matrix

  end subroutine convert_to_dense_dense_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param A The bml matrix.
  !! \param A_dense The dense matrix.
  subroutine convert_to_dense_dense_double(A, A_dense)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: A
    double precision, allocatable, intent(out) :: A_dense(:, :)

    A_dense = A%matrix

  end subroutine convert_to_dense_dense_double

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param A_dense The dense matrix.
  !! \param A The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_dense_single(A_dense, A, threshold)

    use bml_type_dense_m

    real, intent(in) :: A_dense(:, :)
    type(bml_matrix_dense_single_t), intent(inout) :: A
    real, optional, intent(in) :: threshold

    integer :: i, j

    A%matrix = A_dense
    if(present(threshold)) then
       do i = 1, A%N
          do j = 1, A%N
             if(A%matrix(i, j) <= threshold) A%matrix(i, j) = 0
          end do
       end do
    end if

  end subroutine convert_from_dense_dense_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param A_dense The dense matrix.
  !! \param A The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_dense_double(A_dense, A, threshold)

    use bml_type_dense_m

    double precision, intent(in) :: A_dense(:, :)
    type(bml_matrix_dense_double_t), intent(inout) :: A
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

  end subroutine convert_from_dense_dense_double

end module bml_convert_dense_m
