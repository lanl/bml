!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_ellpack_m

  implicit none

  private

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

  public :: convert_to_dense_ellpack
  public :: convert_from_dense_ellpack

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
  !! \param a_dense The dense matrix.
  !! \param a The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_ellpack_single(a_dense, a, threshold)

    use bml_type_ellpack_m

    real, intent(in) :: a_dense(:, :)
    type(bml_matrix_ellpack_single_t), intent(inout) :: a
    real, optional, intent(in) :: threshold

    integer :: i, j

  end subroutine convert_from_dense_ellpack_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param a_dense The dense matrix.
  !! \param a The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_ellpack_double(a_dense, a, threshold)

    use bml_type_ellpack_m

    double precision, intent(in) :: a_dense(:, :)
    type(bml_matrix_ellpack_double_t), intent(inout) :: a
    double precision, optional, intent(in) :: threshold

    integer :: i, j

  end subroutine convert_from_dense_ellpack_double

end module bml_convert_ellpack_m
