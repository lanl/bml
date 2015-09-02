!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_ellpack_m
  implicit none

  !> Convert bml to dense matrix.
  interface convert_to_dense_ellpack
     module procedure convert_to_dense_ellpack_single
     module procedure convert_to_dense_ellpack_double
  end interface convert_to_dense_ellpack

  !> Convert bml to dense matrix.
  interface convert_from_dense_ellpack
     module procedure convert_from_dense_ellpack_single
     module procedure convert_from_dense_ellpack_double
  end interface convert_from_dense_ellpack

contains

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param a The bml matrix.
  !! \param a_dense The dense matrix.
  subroutine convert_to_dense_ellpack_single(a, a_dense)

    use bml_type_ellpack_m

    type(bml_matrix_ellpack_single_t), intent(in) :: a
    real, allocatable, intent(out) :: a_dense(:, :)

    integer :: i, j

    allocate(a_dense(a%n, a%n))
    a_dense = 0

    do i = 1, a%n
       do j = 1, a%number_entries(i)
          a_dense(i, a%column_index(i, j)) = a%matrix(i, j)
       end do
    end do

  end subroutine convert_to_dense_ellpack_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param a The bml matrix.
  !! \param a_dense The dense matrix.
  subroutine convert_to_dense_ellpack_double(a, a_dense)

    use bml_type_ellpack_m

    type(bml_matrix_ellpack_double_t), intent(in) :: a
    double precision, allocatable, intent(out) :: a_dense(:, :)

    integer :: i, j

    allocate(a_dense(a%n, a%n))
    a_dense = 0

    do i = 1, a%n
       do j = 1, a%number_entries(i)
          a_dense(i, a%column_index(i, j)) = a%matrix(i, j)
       end do
    end do

  end subroutine convert_to_dense_ellpack_double

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param A_dense The dense matrix.
  !! \param A The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_ellpack_single(A_dense, A, threshold)

    use bml_type_ellpack_m

    real, intent(in) :: A_dense(:, :)
    type(bml_matrix_ellpack_single_t), intent(inout) :: A
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

  end subroutine convert_from_dense_ellpack_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param A_dense The dense matrix.
  !! \param A The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_ellpack_double(A_dense, A, threshold)

    use bml_type_ellpack_m

    double precision, intent(in) :: A_dense(:, :)
    type(bml_matrix_ellpack_double_t), intent(inout) :: A
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

  end subroutine convert_from_dense_ellpack_double

end module bml_convert_ellpack_m
