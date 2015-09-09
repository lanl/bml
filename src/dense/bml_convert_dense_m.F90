!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_dense_m

  implicit none

  private

  !> Convert bml to dense matrix.
  interface bml_convert_to_dense_dense
     module procedure convert_to_dense_dense_single
     module procedure convert_to_dense_dense_double
  end interface bml_convert_to_dense_dense

  !> Convert bml to dense matrix.
  interface bml_convert_from_dense_dense
     module procedure convert_from_dense_dense_single
     module procedure convert_from_dense_dense_double
  end interface bml_convert_from_dense_dense

  public :: bml_convert_to_dense_dense
  public :: bml_convert_from_dense_dense

contains

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param a The bml matrix.
  !! \param a_dense The dense matrix.
  subroutine convert_to_dense_dense_single(a, a_dense)

    use bml_type_dense_m

    type(bml_matrix_dense_single_t), intent(in) :: a
    real, allocatable, intent(out) :: a_dense(:, :)

    allocate(a_dense(a%n, a%n))
    a_dense = a%matrix

  end subroutine convert_to_dense_dense_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param a The bml matrix.
  !! \param a_dense The dense matrix.
  subroutine convert_to_dense_dense_double(a, a_dense)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: a
    double precision, allocatable, intent(out) :: a_dense(:, :)

    allocate(a_dense(a%n, a%n))
    a_dense = a%matrix

  end subroutine convert_to_dense_dense_double

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param a_dense The dense matrix.
  !! \param a The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_dense_single(a_dense, a, threshold)

    use bml_type_dense_m

    real, intent(in) :: a_dense(:, :)
    type(bml_matrix_dense_single_t), intent(inout) :: a
    real, intent(in) :: threshold

    integer :: i, j

    a%matrix = 0
    do i = 1, a%n
       do j = 1, a%n
          if(abs(a_dense(i, j)) > threshold) then
             a%matrix(i, j) = a_dense(i, j)
          end if
       end do
    end do

    !> where(a%matrix > threshold)
    !>    a%matrix = a_dense
    !> end where

  end subroutine convert_from_dense_dense_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param a_dense The dense matrix.
  !! \param a The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_dense_double(a_dense, a, threshold)

    use bml_type_dense_m

    double precision, intent(in) :: a_dense(:, :)
    type(bml_matrix_dense_double_t), intent(inout) :: a
    double precision, intent(in) :: threshold

    integer :: i, j

    a%matrix = 0
    do i = 1, a%n
       do j = 1, a%n
          if(abs(a_dense(i, j)) > threshold) then
             a%matrix(i, j) = a_dense(i, j)
          end if
       end do
    end do

    !> where(a%matrix > threshold)
    !>    a%matrix = a_dense
    !> end where

  end subroutine convert_from_dense_dense_double

end module bml_convert_dense_m
